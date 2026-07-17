"""Regenerate coverage.json (spatial availability per date) from dates.csv.

For every range in dates.csv this fetches the 127-byte PMTiles header (cached in
bounds_cache.json, so reruns only fetch new/changed ranges) and reads the archive's
lat/lon bounds. It then sweeps the timeline and, for each segment between range
boundaries, records what % of the lower-48 reference bbox is covered by the union
of the bounds of all archives in effect (ranges are end-exclusive, matching
_rangesForDate in index.html).

Chart COUNT is not a usable availability signal: one merged 2026 cycle mosaic
(count=1) covers 100% of the US, while 1961's count of 2 covers ~2%.

Run after every dates.csv update:  ~/venv/bin/python scripts/build_coverage.py
"""
import csv, datetime as dt, json, os, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATES = os.path.join(ROOT, 'dates.csv')
CACHE_PATH = os.path.join(ROOT, 'bounds_cache.json')
OUT = os.path.join(ROOT, 'coverage.json')
BASE = 'https://data.archive.aero/sectionals/pmtiles/'

# Lower-48 reference bbox; AK/HI/territory charts count toward chart count but
# not the area percentage.
LON0, LON1 = -125.0, -66.5
LAT0, LAT1 = 24.5, 49.5
STEP = 0.1
NX = int(round((LON1 - LON0) / STEP))
NY = int(round((LAT1 - LAT0) / STEP))


def load_ranges():
    ranges = []
    with open(DATES) as f:
        for r in csv.DictReader(f):
            k = (r['date_iso'] or '').strip()
            if '_to_' not in k:
                continue
            s, e = k.split('_to_')
            try:
                sd = dt.date.fromisoformat(s)
                ed = dt.date.fromisoformat(e)
            except ValueError:
                continue
            if ed < sd:
                sd, ed = ed, sd
            url = (r.get('url') or '').strip() or f'{BASE}{k}.pmtiles'
            ranges.append((k, sd, ed, url))
    return ranges


def fetch_header_bounds(item):
    k, url = item

    def attempt(u):
        req = urllib.request.Request(u, headers={
            'Range': 'bytes=0-126',
            # Cloudflare 403s urllib's default Python-urllib user agent
            'User-Agent': 'Mozilla/5.0 (Macintosh) archive.aero-bounds-scan',
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()

    try:
        try:
            d = attempt(url)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            # A freshly uploaded object can sit behind a CDN-cached 404 for up
            # to 2h; a unique query string bypasses the cache. A real missing
            # object still 404s here.
            d = attempt(f'{url}?cachebust={k}')
        if len(d) < 118 or d[:7] != b'PMTiles':
            return k, 'badheader'
        return k, [v / 1e7 for v in struct.unpack_from('<iiii', d, 102)]
    except Exception as e:
        code = getattr(e, 'code', type(e).__name__)
        return k, f'error:{code}'


def update_bounds_cache(ranges):
    cache = {}
    if os.path.exists(CACHE_PATH):
        cache = json.load(open(CACHE_PATH))
    todo = [(k, u) for k, _, _, u in ranges if not isinstance(cache.get(k), list)]
    if todo:
        print(f'fetching bounds for {len(todo)} archives...', flush=True)
        with ThreadPoolExecutor(max_workers=24) as ex:
            for k, res in ex.map(fetch_header_bounds, todo):
                cache[k] = res
        json.dump(cache, open(CACHE_PATH, 'w'))
    missing = sorted(k for k, _, _, _ in ranges if isinstance(cache.get(k), str))
    if missing:
        print(f'WARNING: {len(missing)} archives unreachable (in dates.csv but not in the '
              f'bucket?) — they contribute chart count but no area:', file=sys.stderr)
        for k in missing:
            print(f'  {k}: {cache[k]}', file=sys.stderr)
    return cache


def rect_slice(b):
    minlon, minlat, maxlon, maxlat = b
    x0 = max(0, int(np.floor((minlon - LON0) / STEP)))
    x1 = min(NX, int(np.ceil((maxlon - LON0) / STEP)))
    y0 = max(0, int(np.floor((minlat - LAT0) / STEP)))
    y1 = min(NY, int(np.ceil((maxlat - LAT0) / STEP)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (slice(y0, y1), slice(x0, x1))


def main():
    ranges = load_ranges()
    cache = update_bounds_cache(ranges)
    spatial = [(sd, ed, cache[k] if isinstance(cache.get(k), list) else None)
               for k, sd, ed, _ in ranges]

    days = sorted({d for sd, ed, _ in spatial for d in (sd, ed)})
    segments = []
    for a, b in zip(days, days[1:]):
        active = [bb for sd, ed, bb in spatial if sd <= a and b <= ed]
        if not active:
            segments.append([a.isoformat(), b.isoformat(), 0, 0.0])
            continue
        mask = np.zeros((NY, NX), dtype=bool)
        for bb in active:
            sl = rect_slice(bb) if bb else None
            if sl:
                mask[sl] = True
        pct = round(100.0 * mask.sum() / mask.size, 1)
        segments.append([a.isoformat(), b.isoformat(), len(active), pct])

    merged = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if prev and prev[2] == seg[2] and prev[3] == seg[3] and prev[1] == seg[0]:
            prev[1] = seg[1]
        else:
            merged.append(seg)

    out = {
        'generated': dt.date.today().isoformat(),
        'ref': [LON0, LAT0, LON1, LAT1],
        'note': 'pct = % of lower-48 reference bbox covered by union of PMTiles '
                'bounds of ranges in effect (end-exclusive)',
        'segments': merged,
    }
    json.dump(out, open(OUT, 'w'), separators=(',', ':'))
    print(f'{len(segments)} segments -> {len(merged)} merged -> coverage.json '
          f'({os.path.getsize(OUT) // 1024} KB)')


if __name__ == '__main__':
    main()
