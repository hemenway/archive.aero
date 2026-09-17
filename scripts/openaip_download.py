#!/usr/bin/env python3
"""Download openAIP country datasets (airports, airspaces, navaids, ...) to an SD card.

Uses the openAIP core API (https://api.core.openaip.net/api) with a personal API
key, pages through every record for each requested country, and writes one file
per (country, dataset) plus a manifest.

    Output: <out>/<COUNTRY>/<country>_<dataset>.geojson   (and/or .json)
            <out>/manifest.jsonl   one line per dataset written (append-only log)

The key never lives in this file. Supply it as, in order of precedence:
    --key <key>
    $OPENAIP_API_KEY
    ~/.openaip_key            (single line; chmod 600)

Usage:
    openaip_download.py                           # auto-detect the card, US only
    openaip_download.py --out /Volumes/EFB/openaip --country US --country CA
    openaip_download.py --type airports --type airspaces --format both
    openaip_download.py --dry-run                 # show the plan, fetch nothing

Re-runs are cheap: a dataset whose file already exists is skipped unless --force.
Files are written to a .tmp sibling and renamed, so a card yanked mid-download
never leaves a half-written dataset behind.
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
    ap.add_argument('--type', action='append', default=[], dest='types',
                    choices=sorted(DATASETS), help=f'dataset, repeatable (default: {" ".join(DEFAULT_TYPES)})')
    ap.add_argument('--all-types', action='store_true', help='every dataset openAIP publishes')
    ap.add_argument('--format', choices=['geojson', 'json', 'both'], default='geojson')
    ap.add_argument('--key', help='API key (else $OPENAIP_API_KEY, else ~/.openaip_key)')
    ap.add_argument('--limit', type=int, default=PAGE_LIMIT, help='page size (max 1000)')
    ap.add_argument('--max-pages', type=int, default=0, help='stop after N pages per dataset (testing)')
    ap.add_argument('--force', action='store_true', help='re-download datasets already on the card')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    a = ap.parse_args()

    out = a.out or default_out()
    countries = [c.upper() for c in (a.country or ['US'])]
    types = sorted(DATASETS) if a.all_types else (a.types or DEFAULT_TYPES)
    exts = ['geojson', 'json'] if a.format == 'both' else [a.format]

    print(f'openAIP -> {out}')
    print(f'countries: {", ".join(countries)}')
    print(f'datasets : {", ".join(types)}  as .{" .".join(exts)}')
    if a.dry_run:
        for c in countries:
            for t in types:
                for e in exts:
                    print(f'  would write {os.path.join(out, c, f"{c.lower()}_{DATASETS[t]}.{e}")}')
        return

    key = load_key(a.key)
    os.makedirs(out, exist_ok=True)
    manifest = open(os.path.join(out, 'manifest.jsonl'), 'a')
    ok = skipped = failed = total_bytes = 0

    for country in countries:
        for dataset in types:
            short = DATASETS[dataset]
            paths = {e: os.path.join(out, country, f'{country.lower()}_{short}.{e}') for e in exts}
            if not a.force and all(os.path.exists(p) for p in paths.values()):
                print(f'  {country} {dataset}: already on the card, skipping (--force to refresh)')
                skipped += 1
                continue
            print(f'  {country} {dataset}:', flush=True)
            rec = {'country': country, 'dataset': dataset,
                   'fetched': time.strftime('%Y-%m-%dT%H:%M:%S')}
            try:
                items = fetch_dataset(dataset, country, key, a.limit, a.max_pages)
                written = {}
                for e, p in paths.items():
                    payload = to_geojson(items) if e == 'geojson' else items
                    written[os.path.relpath(p, out)] = write_atomic(p, payload)
                rec.update(status='ok', count=len(items), files=written)
                ok += 1
                total_bytes += sum(written.values())
                print(f'    wrote {len(items)} records, '
                      + ', '.join(f'{n} ({b/1e6:.1f} MB)' for n, b in written.items()), flush=True)
            except Exception as e:                    # one bad dataset must not sink the run
                rec.update(status='error', error=str(e)[:300])
                failed += 1
                print(f'    FAILED: {e}', file=sys.stderr, flush=True)
            manifest.write(json.dumps(rec, ensure_ascii=False) + '\n')
            manifest.flush()

    manifest.close()
    print(f'done: {ok} written, {skipped} skipped, {failed} failed, {total_bytes/1e6:.1f} MB')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
