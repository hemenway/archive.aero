#!/usr/bin/env python3
"""Mirror the openAIP daily data exports to an SD card.

openAIP publishes its complete dataset once a day as flat files in a public S3
bucket. No API key, no paging, no rate limit — the bucket answers anonymous GET,
HEAD and ListObjectsV2:

    https://storage.openaip.net/openaip-system-exports/

Keys look like <country>_<type>[_<variant>].<ext>, e.g. us_apt.geojson (US
airports) or de_asp_v2.txt (German airspaces, Naviter OpenAIR v2). This script
lists the bucket, picks the files you asked for, and copies them to the card.

    Output: <out>/<COUNTRY>/<key>          filenames kept exactly as published
            <out>/manifest.jsonl           append-only log, one line per file

Formats are whatever the bucket serves (geojson, json, mbtiles, cup, cupx, aip,
txt for OpenAIR, ...). The default is geojson: it carries every openAIP field in
the feature properties and opens directly in QGIS, ogr2ogr, Leaflet and most EFB
import paths. `--list` prints what is actually on the server right now.

Usage:
    openaip_download.py                            # every country, geojson, to the card
    openaip_download.py --list                     # what the bucket holds today
    openaip_download.py --country us --country ca  # just those
    openaip_download.py --type apt --type asp      # airports + airspaces
    openaip_download.py --format txt               # OpenAIR v2 airspaces
    openaip_download.py --format mbtiles --out /Volumes/EFB/openaip
    openaip_download.py --dry-run                  # show the plan, fetch nothing

Re-runs only fetch what changed: a file whose size matches the bucket and whose
mtime is not older than the published object is left alone (the mtime is stamped
from the object's Last-Modified after each download), so a daily refresh moves
only the days' worth of edits. Downloads land in a .tmp sibling and are renamed,
so a card pulled mid-copy never leaves a half-written export behind.
"""
import argparse
import calendar
import email.utils
import hashlib
import json
import os
import platform
import queue
import ssl
import sys
import threading
import time
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

BUCKET = os.environ.get(
    'OPENAIP_EXPORTS', 'https://storage.openaip.net/openaip-system-exports/')
UA = 'archive.aero openaip-download/2.0'
CHUNK = 1 << 20

# Friendly names accepted by --type, mapped to the short code used in the keys.
TYPE_ALIASES = {
    'airports': 'apt', 'airspaces': 'asp', 'navaids': 'nav',
    'reporting-points': 'rpp', 'obstacles': 'obs', 'hotspots': 'hot',
    'rc-airfields': 'rcf', 'hang-glidings': 'hgl', 'waypoints': 'wpt',
}


def die(msg):
    print(f'error: {msg}', file=sys.stderr)
    sys.exit(1)


def http(url, tries=4, stream=False):
    last = None
    for i in range(tries):
        try:
            r = urlopen(Request(url, headers={'User-Agent': UA, 'Accept': '*/*'}),
                        timeout=180)
            return r if stream else r.read()
        except HTTPError as e:
            if e.code in (403, 404, 410):
                raise RuntimeError(f'HTTP {e.code} for {url}')
            last = e
        except (URLError, ssl.SSLError, TimeoutError) as e:
            last = e
        time.sleep(2 * (i + 1))
    raise RuntimeError(f'giving up after {tries} tries: {last}')


def tag(el):
    """Local tag name, ignoring whatever S3 namespace the server used."""
    return el.tag.rsplit('}', 1)[-1]


def find(el, name):
    for child in el:
        if tag(child) == name:
            return child
    return None


def list_bucket():
    """Every object in the export bucket, following continuation tokens."""
    objs, token = [], None
    while True:
        q = {'list-type': '2', 'max-keys': '1000'}
        if token:
            q['continuation-token'] = token
        root = ET.fromstring(http(BUCKET + '?' + urlencode(q)))
        for el in root:
            if tag(el) != 'Contents':
                continue
            key = find(el, 'Key')
            if key is None or not (key.text or '').strip():
                continue
            size = find(el, 'Size')
            mod = find(el, 'LastModified')
            etag = find(el, 'ETag')
            objs.append({
                'key': key.text.strip(),
                'size': int(size.text) if size is not None and size.text else None,
                'modified': (mod.text or '').strip() if mod is not None else '',
                'etag': (etag.text or '').strip('"') if etag is not None else '',
            })
        truncated = find(root, 'IsTruncated')
        nxt = find(root, 'NextContinuationToken')
        if (truncated is None or (truncated.text or '').lower() != 'true'
                or nxt is None or not nxt.text):
            break
        token = nxt.text.strip()
    if not objs:
        raise RuntimeError(f'{BUCKET} listed no objects')
    return objs


def parse_key(key):
    """<country>_<type>[_<variant>].<ext> -> dict, or None if it doesn't fit."""
    name = key.rsplit('/', 1)[-1]
    if '.' not in name or '_' not in name:
        return None
    stem, ext = name.rsplit('.', 1)
    parts = stem.split('_')
    if len(parts) < 2 or len(parts[0]) != 2 or not parts[0].isalpha():
        return None
    return {
        'key': key, 'name': name,
        'country': parts[0].upper(),
        'type': parts[1].lower(),
        'variant': parts[2].lower() if len(parts) > 2 else '',
        'ext': ext.lower(),
    }


def parse_modified(s):
    """S3 Last-Modified (ISO 8601 or RFC 2822) -> epoch seconds, or None."""
    if not s:
        return None
    try:
        return calendar.timegm(time.strptime(s.split('.')[0].rstrip('Z'),
                                             '%Y-%m-%dT%H:%M:%S'))
    except ValueError:
        pass
    try:
        return calendar.timegm(email.utils.parsedate(s))
    except (TypeError, ValueError):
        return None


def candidate_volumes():
    """Mounted removable/external volumes, most plausible SD card first."""
    if platform.system() == 'Darwin':
        root = '/Volumes'
    elif platform.system() == 'Linux':
        root = f"/media/{os.environ.get('USER', '')}"
        if not os.path.isdir(root):
            root = '/media'
    else:
        return []
    if not os.path.isdir(root):
        return []
    vols = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p) or os.path.islink(p):
            continue
        if os.path.realpath(p) == '/':      # the boot volume's own /Volumes entry
            continue
        vols.append(p)
    return vols


def default_out():
    if os.environ.get('OPENAIP_OUT'):
        return os.environ['OPENAIP_OUT']
    vols = candidate_volumes()
    if len(vols) == 1:
        return os.path.join(vols[0], 'openaip')
    if not vols:
        die('no removable volume mounted — insert the SD card or pass --out DIR')
    die('several volumes mounted — pick one with --out:\n  ' + '\n  '.join(vols))


def is_current(path, obj):
    """True when the copy on the card already matches the published object."""
    if not os.path.exists(path):
        return False
    st = os.stat(path)
    if obj['size'] is not None and st.st_size != obj['size']:
        return False
    mtime = parse_modified(obj['modified'])
    if mtime is not None and st.st_mtime + 1 < mtime:
        return False
    return True


def download(obj, dest):
    """Stream one object to dest; returns (bytes, sha256)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.tmp'
    h = hashlib.sha256()
    total = 0
    r = http(BUCKET + quote(obj['key']), stream=True)
    try:
        with open(tmp, 'wb') as f:
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                total += len(chunk)
            f.flush()
            os.fsync(f.fileno())
    finally:
        r.close()
    if obj['size'] is not None and total != obj['size']:
        os.unlink(tmp)
        raise RuntimeError(f"short read: got {total} bytes, expected {obj['size']}")
    os.replace(tmp, dest)
    mtime = parse_modified(obj['modified'])
    if mtime is not None:
        os.utime(dest, (mtime, mtime))     # lets the next run see it as current
    return total, h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description='Mirror openAIP daily exports to an SD card.')
    ap.add_argument('--out', help='destination dir (default: the mounted SD card, or $OPENAIP_OUT)')
    ap.add_argument('--country', action='append', default=[],
                    help='ISO 3166-1 alpha-2 code, repeatable (default: every country)')
    ap.add_argument('--type', action='append', default=[], dest='types',
                    help='apt, asp, nav, rpp, obs, ... or a full name like airports '
                         '(repeatable; default: every type)')
    ap.add_argument('--format', default='geojson',
                    help='file extension as published: geojson (default), json, mbtiles, '
                         'cup, cupx, aip, txt (OpenAIR), or all')
    ap.add_argument('--variant', help='OpenAIR flavour: v1 or v2 (default: v2 when --format txt)')
    ap.add_argument('--workers', type=int, default=4, help='parallel downloads (default 4)')
    ap.add_argument('--force', action='store_true', help='re-download files already current')
    ap.add_argument('--list', action='store_true', dest='do_list',
                    help='print what the bucket holds today, then exit')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    a = ap.parse_args()

    print(f'listing {BUCKET}', flush=True)
    try:
        objs = list_bucket()
    except (RuntimeError, ET.ParseError) as e:
        die(f'could not list the bucket: {e}')
    parsed = [{**p, 'size': o['size'], 'modified': o['modified'], 'etag': o['etag']}
              for o in objs for p in [parse_key(o['key'])] if p]
    print(f'{len(objs)} objects, {len(parsed)} recognised country files', flush=True)

    if a.do_list:
        by_ext, by_type, countries = {}, {}, set()
        for p in parsed:
            by_ext[p['ext']] = by_ext.get(p['ext'], 0) + 1
            by_type[p['type']] = by_type.get(p['type'], 0) + 1
            countries.add(p['country'])
        newest = max((p['modified'] for p in parsed), default='')
        print(f'countries: {len(countries)}  ({", ".join(sorted(countries))})')
        print('formats  : ' + ', '.join(f'{k} ({v})' for k, v in sorted(by_ext.items())))
        print('types    : ' + ', '.join(f'{k} ({v})' for k, v in sorted(by_type.items())))
        print(f'newest   : {newest}')
        return

    out = a.out or default_out()
    want_countries = {c.upper() for c in a.country}
    want_types = {TYPE_ALIASES.get(t.lower(), t.lower()) for t in a.types}
    fmt = a.format.lower()
    variant = (a.variant or '').lower()
    if fmt == 'txt' and not variant:
        variant = 'v2'                     # newest OpenAIR flavour unless told otherwise

    todo = []
    for p in parsed:
        if fmt != 'all' and p['ext'] != fmt:
            continue
        if want_countries and p['country'] not in want_countries:
            continue
        if want_types and p['type'] not in want_types:
            continue
        if variant and p['variant'] and p['variant'] != variant:
            continue
        todo.append(p)
    todo.sort(key=lambda p: (p['country'], p['type'], p['name']))

    if not todo:
        # Say which filter emptied the set, not just that the set is empty.
        have_ext = sorted({p['ext'] for p in parsed})
        have_type = sorted({p['type'] for p in parsed})
        have_country = sorted({p['country'] for p in parsed})
        why = []
        if fmt != 'all' and fmt not in have_ext:
            why.append(f'no --format {fmt} on the server (has: {", ".join(have_ext)})')
        for miss, label, have in ((want_types - set(have_type), '--type', have_type),
                                  (want_countries - set(have_country), '--country', have_country)):
            if miss:
                why.append(f'no {label} {", ".join(sorted(miss))} '
                           f'(has: {", ".join(have)})')
        if not why:
            why.append('the filters exclude each other')
        die('nothing matches: ' + '; '.join(why) + '. Try --list')

    total_size = sum(p['size'] or 0 for p in todo)
    print(f'openAIP -> {out}')
    print(f'{len(todo)} files, {total_size/1e9:.2f} GB published '
          f'(format {fmt}{", " + variant if variant else ""})')
    if a.dry_run:
        for p in todo:
            print(f"  {p['country']}/{p['name']}  {(p['size'] or 0)/1e6:.1f} MB  {p['modified']}")
        return

    os.makedirs(out, exist_ok=True)
    mf = open(os.path.join(out, 'manifest.jsonl'), 'a')
    lock = threading.Lock()
    stats = {'ok': 0, 'current': 0, 'failed': 0, 'bytes': 0}
    q = queue.Queue()
    for p in todo:
        q.put(p)

    def worker():
        while True:
            try:
                p = q.get_nowait()
            except queue.Empty:
                return
            dest = os.path.join(out, p['country'], p['name'])
            if not a.force and is_current(dest, p):
                with lock:
                    stats['current'] += 1
                continue
            rec = {k: p[k] for k in ('key', 'country', 'type', 'variant', 'ext', 'modified')}
            rec['file'] = os.path.relpath(dest, out)
            try:
                n, digest = download(p, dest)
                rec.update(status='ok', bytes=n, sha256=digest)
                with lock:
                    stats['ok'] += 1
                    stats['bytes'] += n
                    done = stats['ok'] + stats['failed']
                    print(f"  [{done}/{len(todo)}] {rec['file']} ({n/1e6:.1f} MB)", flush=True)
            except Exception as e:          # one bad file must not sink the mirror
                rec.update(status='error', error=str(e)[:300])
                with lock:
                    stats['failed'] += 1
                    print(f"  FAILED {p['key']}: {e}", file=sys.stderr, flush=True)
            rec['fetched'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            with lock:
                mf.write(json.dumps(rec, ensure_ascii=False) + '\n')
                mf.flush()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(max(1, a.workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    mf.close()
    print(f"done: {stats['ok']} downloaded, {stats['current']} already current, "
          f"{stats['failed']} failed, {stats['bytes']/1e6:.1f} MB")
    sys.exit(1 if stats['failed'] else 0)


if __name__ == '__main__':
    main()
