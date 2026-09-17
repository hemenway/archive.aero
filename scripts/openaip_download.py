#!/usr/bin/env python3
"""Download openAIP datasets (airports, airspaces, navaids, ...) to an SD card.

Uses the openAIP core API (https://api.core.openaip.net/api) with a personal API
key, pages through every record, and writes one GeoJSON file per country and
dataset. GeoJSON is the only output: it is what the API actually serves (every
openAIP field is kept in the feature's properties) and it opens directly in
QGIS, Leaflet, gdal/ogr and most EFB import paths.

    Output: <out>/<COUNTRY>/<country>_<dataset>.geojson
            <out>/manifest.jsonl   one line per dataset (append-only log)

The key never lives in this file. Supply it as, in order of precedence:
    --key <key>
    $OPENAIP_API_KEY
    ~/.openaip_key            (single line; chmod 600)

Usage:
    openaip_download.py                           # auto-detect the card, US only
    openaip_download.py --all-countries           # every region openAIP serves
    openaip_download.py --all-countries --all-types --out /Volumes/EFB/openaip
    openaip_download.py --country US --country CA --type airports
    openaip_download.py --dry-run                 # show the plan, fetch nothing

Re-runs are cheap: a dataset whose file already exists is skipped unless --force,
so an --all-countries run interrupted halfway resumes where it stopped. Files are
written to a .tmp sibling and renamed, so a card yanked mid-download never leaves
a half-written dataset behind.
"""
import argparse
import json
import os
import platform
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = os.environ.get('OPENAIP_API', 'https://api.core.openaip.net/api')  # override for testing
UA = 'archive.aero openaip-download/1.0'
PAGE_LIMIT = 1000          # API maximum
KEY_FILE = os.path.expanduser('~/.openaip_key')

# API path -> short name used in filenames
DATASETS = {
    'airports': 'apt',
    'airspaces': 'asp',
    'navaids': 'nav',
    'reporting-points': 'rpp',
    'obstacles': 'obs',
    'hotspots': 'hot',
    'rc-airfields': 'rcf',
    'hang-glidings': 'hgl',
}
DEFAULT_TYPES = ['airports', 'airspaces', 'navaids', 'reporting-points', 'obstacles']


def die(msg):
    print(f'error: {msg}', file=sys.stderr)
    sys.exit(1)


def load_key(cli_key):
    if cli_key:
        return cli_key.strip()
    env = os.environ.get('OPENAIP_API_KEY', '').strip()
    if env:
        return env
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            k = f.read().strip()
        if k:
            return k
    die(f'no API key. Pass --key, set $OPENAIP_API_KEY, or put it in {KEY_FILE}')


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
    listing = '\n  '.join(vols)
    die('several volumes mounted — pick one with --out:\n  ' + listing)


def fetch_page(session_url, key, tries=4):
    last = None
    for i in range(tries):
        req = Request(session_url, headers={
            'x-openaip-api-key': key,
            'User-Agent': UA,
            'Accept': 'application/json',
        })
        try:
            with urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode('utf-8'))
        except HTTPError as e:
            body = e.read()[:300].decode('utf-8', 'replace')
            if e.code in (401, 403):
                die(f'API rejected the key (HTTP {e.code}): {body}')
            if e.code == 404:
                raise RuntimeError(f'HTTP 404 — no such dataset: {body}')
            if e.code == 429:                       # rate limited: back off hard
                wait = int(e.headers.get('Retry-After') or 0) or 15 * (i + 1)
                print(f'    rate limited, sleeping {wait}s', flush=True)
                time.sleep(wait)
                last = e
                continue
            last = e
        except (URLError, ssl.SSLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
        time.sleep(2 * (i + 1))
    raise RuntimeError(f'giving up after {tries} tries: {last}')


def fetch_countries(key):
    """Every country code openAIP serves, from /countries."""
    codes = []
    page = 1
    while True:
        body = fetch_page(f'{API}/countries?' + urlencode({'limit': 1000, 'page': page}), key)
        items = body.get('items', body if isinstance(body, list) else [])
        for it in items:
            if isinstance(it, str):
                code = it
            else:
                code = (it.get('isoCode') or it.get('code') or it.get('alpha2')
                        or it.get('iso2') or it.get('_id') or '')
            code = str(code).strip().upper()
            if len(code) == 2 and code.isalpha():
                codes.append(code)
        total_pages = body.get('totalPages')
        if (total_pages is not None and page >= total_pages) or len(items) < 1000:
            break
        page += 1
    if not codes:
        raise RuntimeError('/countries returned no usable ISO codes')
    return sorted(set(codes))


def fetch_dataset(dataset, country, key, limit, max_pages=0):
    """All items for one dataset/country, paging until the API runs out."""
    items, page = [], 1
    while True:
        url = f'{API}/{dataset}?' + urlencode({
            'country': country.upper(), 'limit': limit, 'page': page})
        body = fetch_page(url, key)
        batch = body.get('items', body if isinstance(body, list) else [])
        items.extend(batch)
        total_pages = body.get('totalPages')
        total = body.get('totalCount')
        print(f'    page {page}'
              + (f'/{total_pages}' if total_pages else '')
              + f': {len(batch)} items ({len(items)}'
              + (f'/{total}' if total is not None else '') + ')', flush=True)
        if max_pages and page >= max_pages:
            break
        if total_pages is not None:
            if page >= total_pages:
                break
        elif len(batch) < limit:
            break
        page += 1
        time.sleep(0.2)                              # be polite to the API
    return items


def to_geojson(items):
    """FeatureCollection; geometry from the record, everything else a property."""
    feats = []
    for it in items:
        geom = it.get('geometry')
        if not geom:
            lat, lon = None, None
            g = it.get('position') or it.get('coordinates')
            if isinstance(g, dict):
                lat, lon = g.get('lat'), g.get('lon')
            elif isinstance(g, (list, tuple)) and len(g) == 2:
                lon, lat = g
            if lat is None or lon is None:
                continue
            geom = {'type': 'Point', 'coordinates': [lon, lat]}
        props = {k: v for k, v in it.items() if k != 'geometry'}
        feats.append({'type': 'Feature', 'geometry': geom, 'properties': props})
    return {'type': 'FeatureCollection', 'features': feats}


def write_atomic(path, payload):
    """Write via .tmp + rename + fsync — safe on a card that may be pulled."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser(description='Download openAIP data to an SD card.')
    ap.add_argument('--out', help='destination dir (default: the mounted SD card, or $OPENAIP_OUT)')
    ap.add_argument('--country', action='append', default=[],
                    help='ISO 3166-1 alpha-2 code, repeatable (default: US)')
    ap.add_argument('--all-countries', action='store_true',
                    help='every region openAIP serves (enumerated from /countries)')
    ap.add_argument('--type', action='append', default=[], dest='types',
                    choices=sorted(DATASETS), help=f'dataset, repeatable (default: {" ".join(DEFAULT_TYPES)})')
    ap.add_argument('--all-types', action='store_true', help='every dataset openAIP publishes')
    ap.add_argument('--key', help='API key (else $OPENAIP_API_KEY, else ~/.openaip_key)')
    ap.add_argument('--limit', type=int, default=PAGE_LIMIT, help='page size (max 1000)')
    ap.add_argument('--max-pages', type=int, default=0, help='stop after N pages per dataset (testing)')
    ap.add_argument('--force', action='store_true', help='re-download datasets already on the card')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    a = ap.parse_args()

    if a.all_countries and a.country:
        die('--all-countries and --country are mutually exclusive')
    out = a.out or default_out()
    types = sorted(DATASETS) if a.all_types else (a.types or DEFAULT_TYPES)
    key = load_key(a.key) if (a.all_countries or not a.dry_run) else None

    print(f'openAIP -> {out}')
    if a.all_countries:
        countries = fetch_countries(key)
        print(f'countries: all {len(countries)} openAIP serves')
    else:
        countries = [c.upper() for c in (a.country or ['US'])]
        print(f'countries: {", ".join(countries)}')
    print(f'datasets : {", ".join(types)}  ({len(countries) * len(types)} files max)')
    if a.dry_run:
        for c in countries:
            for t in types:
                print(f'  would write {os.path.join(out, c, f"{c.lower()}_{DATASETS[t]}.geojson")}')
        return

    os.makedirs(out, exist_ok=True)
    manifest_path = os.path.join(out, 'manifest.jsonl')
    known_empty = set()
    if os.path.exists(manifest_path) and not a.force:
        with open(manifest_path) as f:
            for line in f:
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if m.get('status') == 'empty':
                    known_empty.add((m.get('country'), m.get('dataset')))
    manifest = open(manifest_path, 'a')
    ok = skipped = empty = failed = total_bytes = 0

    for country in countries:
        for dataset in types:
            short = DATASETS[dataset]
            path = os.path.join(out, country, f'{country.lower()}_{short}.geojson')
            rel = os.path.relpath(path, out)
            if not a.force and os.path.exists(path):
                print(f'  {country} {dataset}: already on the card, skipping (--force to refresh)')
                skipped += 1
                continue
            if (country, dataset) in known_empty:
                # an earlier run already established openAIP has nothing here
                empty += 1
                continue
            print(f'  {country} {dataset}:', flush=True)
            rec = {'country': country, 'dataset': dataset,
                   'fetched': time.strftime('%Y-%m-%dT%H:%M:%S')}
            try:
                items = fetch_dataset(dataset, country, key, a.limit, a.max_pages)
                if not items:
                    # openAIP has nothing here — don't litter the card with empty files
                    rec.update(status='empty', count=0)
                    empty += 1
                    print('    no records, nothing written', flush=True)
                    manifest.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    manifest.flush()
                    continue
                size = write_atomic(path, to_geojson(items))
                rec.update(status='ok', count=len(items), file=rel, bytes=size)
                ok += 1
                total_bytes += size
                print(f'    wrote {len(items)} records -> {rel} ({size/1e6:.1f} MB)', flush=True)
            except Exception as e:                    # one bad dataset must not sink the run
                rec.update(status='error', error=str(e)[:300])
                failed += 1
                print(f'    FAILED: {e}', file=sys.stderr, flush=True)
            manifest.write(json.dumps(rec, ensure_ascii=False) + '\n')
            manifest.flush()

    manifest.close()
    print(f'done: {ok} written, {skipped} already on card, {empty} empty, '
          f'{failed} failed, {total_bytes/1e6:.1f} MB')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
