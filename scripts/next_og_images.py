#!/usr/bin/env python3
"""Pre-render one 1200x630 social card (og:image) per edition page of the next viewer.

Worklist 09 item B3. Every page at /sectionals/chart/<slug>/<date> gets a JPEG whose
key mirrors the page: `sectionals/chart/<slug>/<date>.jpg`, written under --out and
uploaded next to the chart artifact later (this script never touches R2).

Inputs are the per-location chart index files (one JSON per slug, see
scripts/next_build_overlays.py) and the production tile endpoint
`https://data.archive.aero/t/<pm>/{z}/{x}/{y}` (256 px WebP, 204 = no tile).
A date's page is the first chart row with that date (next/pages/chart-page.mjs
`describe`), and a page exists only when that row carries `pm`; a half-sheet pair
(`pm` is a list) composites both halves into the one image.

Composition: the chart imagery fills the whole canvas, cover-fitted to the artifact
bbox (`pmb`) at the smallest zoom that gives native pixels (no upscaling), capped at
the archive's maxzoom and ~120 tiles; a dark band over the bottom 130 px carries the
page title, the in-effect dates and the site name in Barlow. Missing tiles inside
the bbox show as the page background.

Resumable: a key whose file and manifest line both exist is skipped. Failures go to
failures.jsonl (a later successful run supersedes them by key).

    ~/venv/bin/python scripts/next_og_images.py --index DIR --out /Volumes/projects/archive-next/og
        [--only boston_ma | --only boston_ma/1933-03-01] [--limit N] [--workers 12] [--in-flight 32] [--dry-run]
"""
import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import urllib3
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BAND_H = 130
TILE = 256
MAX_TILES = 120
TARGET_BYTES = 150 * 1024
MIN_QUALITY = 70          # dense modern sheets can exceed TARGET_BYTES rather than go below this
INSET = 0.10              # fraction of the sheet's content box trimmed on each side (collars, legends)
DETAIL = 0.85             # the zoom must give >= 1200 px for this fraction of the bbox width (the inset view stays native)
USER_AGENT = 'archive.aero-og/1.0 (+https://archive.aero/contribute)'
TILE_BASE = 'https://data.archive.aero/t/'
MAX_IN_FLIGHT = 32        # tile requests in flight, process-wide (the Worker fronts R2 and caches at the edge)

# styles.css dark theme tokens: --bg-dark, --text-light, --text-muted (0.6 white).
BG = (0x0a, 0x0e, 0x12)
TEXT = (255, 255, 255)
BAND_ALPHA = 0.92
MUTED = tuple(round(0.6 * 255 + 0.4 * c) for c in BG)

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
TYPES = {'sectional': 'Sectional chart', 'tac': 'Terminal area chart', 'wac': 'World aeronautical chart'}
REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- page text (mirrors chart-page.mjs)
def long_date(iso):
    y, m, d = (int(p) for p in iso.split('-')); return f'{MONTHS[m - 1]} {d}, {y}'


def month_year(iso):
    y, m, _ = (int(p) for p in iso.split('-')); return f'{MONTHS[m - 1]} {y}'


def edition_label(ed):
    ed = ed or ''
    if re.fullmatch(r'\d+', ed) and int(ed) < 150: return f'edition {int(ed)}'
    if ed and not re.fullmatch(r'\d+', ed) and ed != 'Unknown': return ed
    return None


def describe(index, chart):
    name = index['names'][0]
    type_label = TYPES.get(index.get('type') or 'sectional', TYPES['sectional'])
    edition = edition_label(chart.get('ed'))
    title = f"{name} {type_label.lower()}, {month_year(chart['d'])}{f' ({edition})' if edition else ''}"
    dates = f"In effect {long_date(chart['d'])} – {long_date(chart['e'])}" if chart.get('e') else f"Published {long_date(chart['d'])}"
    return title, dates


# ---------------------------------------------------------------- pages
def pages(index_dir, only=None):
    """Yield (index, chart) for every page: first row per date, tiled only (describe's rule)."""
    want_slugs, want_pages = set(), set()
    for o in only or []:
        (want_pages if '/' in o else want_slugs).add(o)
    for path in sorted(Path(index_dir).glob('*.json')):
        index = json.loads(path.read_text())
        slug = index['slug']
        if (want_slugs or want_pages) and slug not in want_slugs and not any(p.startswith(slug + '/') for p in want_pages): continue
        seen = set()
        for chart in index['charts']:
            if chart['d'] in seen: continue
            seen.add(chart['d'])
            if not chart.get('pm'): continue
            if want_pages and f"{slug}/{chart['d']}" not in want_pages and slug not in want_slugs: continue
            yield index, chart


# ---------------------------------------------------------------- mercator
LAT_MAX = 85.0511287798066


def merc(lon, lat, z):
    """Web Mercator pixel coordinates at zoom z (lon may exceed 180 for unwrapped boxes)."""
    n = TILE * (1 << z)
    lat = max(-LAT_MAX, min(LAT_MAX, lat))
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def unwrap(boxes):
    """Union of lon/lat boxes, shifting west-of-dateline halves east of 180 when the set straddles it."""
    boxes = [list(b) for b in boxes]
    if len(boxes) > 1 and any(b[0] < -90 for b in boxes) and any(b[2] > 90 for b in boxes):
        boxes = [[b[0] + 360, b[1], b[2] + 360, b[3]] if b[2] < 0 else b for b in boxes]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def plan(bbox, maxzoom):
    """Zoom, crop window (zoom pixels) and tile range for a cover fit of bbox on W x H."""
    maxzoom = max(0, min(int(maxzoom), 20))
    z = 0
    for z in range(0, maxzoom + 1):
        x0, y1 = merc(bbox[0], bbox[1], z); x1, y0 = merc(bbox[2], bbox[3], z)
        if (x1 - x0) * DETAIL >= W and (y1 - y0) * DETAIL >= H: break
    while True:
        x0, y1 = merc(bbox[0], bbox[1], z); x1, y0 = merc(bbox[2], bbox[3], z)
        bw, bh = x1 - x0, y1 - y0
        scale = max(W / bw, H / bh)                      # cover: the larger scale wins
        cw, ch = W / scale, H / scale
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        win = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        tx0, ty0 = math.floor(win[0] / TILE), math.floor(win[1] / TILE)
        tx1, ty1 = math.ceil(win[2] / TILE), math.ceil(win[3] / TILE)
        count = (tx1 - tx0) * (ty1 - ty0)
        if count <= MAX_TILES or z == 0: break
        z -= 1
    return {'zoom': z, 'win': win, 'tiles': (tx0, ty0, tx1, ty1), 'count': count, 'scale': scale}


# ---------------------------------------------------------------- tiles
class Tiles:
    """Tile fetcher: at most MAX_IN_FLIGHT requests in flight process-wide, keep-alive, retries."""

    class Missing(Exception): pass

    def __init__(self, base, in_flight=MAX_IN_FLIGHT):
        self.base = base
        self.http = urllib3.PoolManager(num_pools=4, maxsize=in_flight, headers={'User-Agent': USER_AGENT, 'Accept': 'image/webp,image/*,*/*'},
                                        timeout=urllib3.Timeout(connect=10, read=60), retries=False)
        self.lock = threading.Lock()
        self.stats = {'requests': 0, 'tiles': 0, 'empty': 0, 'bytes': 0, 'retries': 0}

    def _get(self, url, binary=True):
        delay = 0.5
        for attempt in range(6):
            try:
                r = self.http.request('GET', url, preload_content=True)
                status, body = r.status, r.data
            except Exception as e:                         # connection reset, timeout, TLS hiccup
                status, body = -1, repr(e).encode()
            with self.lock: self.stats['requests'] += 1
            if status == 200: return body
            if status == 204: return None
            if status == 404: raise Tiles.Missing(f'404 {url}')
            if status in (400, 403): raise RuntimeError(f'{status} {url}')
            if attempt == 5: raise RuntimeError(f'{status} after retries {url}: {body[:120]!r}')
            with self.lock: self.stats['retries'] += 1
            time.sleep(delay); delay = min(delay * 2, 8)

    def metadata(self, pm):
        body = self._get(f'{self.base}{pm}/metadata')
        if body is None: raise RuntimeError(f'empty metadata {pm}')
        return json.loads(body)

    def tile(self, pm, z, x, y):
        body = self._get(f'{self.base}{pm}/{z}/{x}/{y}')
        with self.lock:
            if body is None: self.stats['empty'] += 1
            else: self.stats['tiles'] += 1; self.stats['bytes'] += len(body)
        return body

    def fetch_all(self, jobs, concurrency):
        """jobs: iterable of (pm, z, x, y); returns list of (job, bytes|None) in order, raising the first error.
        Each page fetches through its own small pool: the Worker serialises a cold archive's block
        reads, so one shared FIFO queue would pile every in-flight request onto one or two archives."""
        with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix='tile') as pool:
            futures = [(j, pool.submit(self.tile, *j)) for j in jobs]
            return [(j, f.result()) for j, f in futures]


# ---------------------------------------------------------------- fonts
def load_fonts(cache_dir):
    """Barlow 600/500 as TTF (converted once from @fontsource/barlow); Helvetica Neue if that fails."""
    cache_dir = Path(cache_dir).expanduser(); cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for weight in (600, 500):
        ttf = cache_dir / f'Barlow-{weight}.ttf'
        if not ttf.exists():
            src = REPO / 'node_modules' / '@fontsource' / 'barlow' / 'files' / f'barlow-latin-{weight}-normal.woff'
            try:
                from fontTools.ttLib import TTFont
                f = TTFont(src); f.flavor = None; f.save(ttf)
            except Exception as e:
                print(f'warning: Barlow {weight} unavailable ({e}); falling back to Helvetica Neue', file=sys.stderr)
                return {'display': '/System/Library/Fonts/HelveticaNeue.ttc', 'text': '/System/Library/Fonts/HelveticaNeue.ttc', 'fallback': True}
        out['display' if weight == 600 else 'text'] = str(ttf)
    out['fallback'] = False
    return out


def content_window(canvas, win):
    """Cover window over the sheet itself: trim rows/columns that are mostly empty (a tilted
    sheet's dark corners), inset for collars, then re-fit; stays inside the fetched canvas."""
    alpha = np.asarray(canvas.getchannel('A')) > 0
    cols = np.flatnonzero(alpha.mean(axis=0) >= 0.5); rows = np.flatnonzero(alpha.mean(axis=1) >= 0.5)
    if not len(cols) or not len(rows): return win
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    # Only what the window could show counts: rows outside it were never fetched in full.
    y0, y1 = max(y0, win[1]), min(y1, win[3]); x0, x1 = max(x0, win[0]), min(x1, win[2])
    if x1 - x0 < 64 or y1 - y0 < 64: return win
    bw, bh = x1 - x0, y1 - y0
    x0 += bw * INSET; x1 -= bw * INSET; y0 += bh * INSET; y1 -= bh * INSET
    scale = max(W / (x1 - x0), H / (y1 - y0))
    cw, ch = W / scale, H / scale
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw, ch = min(cw, canvas.width), min(ch, canvas.height)
    if cw / ch > W / H: cw = ch * W / H
    else: ch = cw * H / W
    left = min(max(cx - cw / 2, 0), canvas.width - cw); top = min(max(cy - ch / 2, 0), canvas.height - ch)
    return (left, top, left + cw, top + ch)


def fit_font(path, text, size, max_width, min_size=24):
    while True:
        font = ImageFont.truetype(path, size)
        if font.getlength(text) <= max_width or size <= min_size: return font
        size -= 2


# ---------------------------------------------------------------- render
def render(index, chart, tiles, fonts, concurrency=4):
    pms = chart['pm'] if isinstance(chart['pm'], list) else [chart['pm']]
    boxes = [chart['pmb']]
    # A union box wider than a hemisphere is the dateline, not the chart: ask each half for its own bounds.
    if chart['pmb'][2] - chart['pmb'][0] > 180:
        boxes = []
        for pm in pms:
            b = [float(v) for v in tiles.metadata(pm)['bounds'].split(',')]
            boxes.append(b)
        if len(boxes) == 1 and boxes[0][2] - boxes[0][0] > 180: raise RuntimeError('antimeridian artifact with world-wide bounds')
    bbox = unwrap(boxes)
    if bbox[2] - bbox[0] < 0.05 or bbox[3] - bbox[1] < 0.05: raise RuntimeError(f'degenerate bbox {bbox}')
    maxzoom = (chart.get('pmz') or [0, 11])[1]
    p = plan(bbox, maxzoom)
    z = p['zoom']; tx0, ty0, tx1, ty1 = p['tiles']; n = 1 << z
    jobs = [(pm, z, tx % n, ty) for pm in pms for ty in range(max(0, ty0), min(n, ty1)) for tx in range(tx0, tx1)]
    canvas = Image.new('RGBA', ((tx1 - tx0) * TILE, (ty1 - ty0) * TILE), (0, 0, 0, 0))
    for (pm, _, x, y), body in tiles.fetch_all(jobs, concurrency):
        if body is None: continue
        im = Image.open(io.BytesIO(body)); im.load()
        if im.mode != 'RGBA': im = im.convert('RGBA')
        # tx may exceed n for an unwrapped box: place by the unwrapped column, fetch by the wrapped one.
        for col in range(tx0, tx1):
            if col % n == x: canvas.alpha_composite(im, ((col - tx0) * TILE, (y - ty0) * TILE))
    wx0, wy0, wx1, wy1 = p['win']
    box = content_window(canvas, (wx0 - tx0 * TILE, wy0 - ty0 * TILE, wx1 - tx0 * TILE, wy1 - ty0 * TILE))
    flat = Image.new('RGBA', canvas.size, BG + (255,)); flat.alpha_composite(canvas)
    img = flat.convert('RGB').resize((W, H), Image.LANCZOS, box=box, reducing_gap=3.0)
    view_scale = W / (box[2] - box[0])

    # Bottom band: solid page colour at 92 % over the chart, text in Barlow.
    band = Image.new('RGBA', (W, BAND_H), BG + (round(BAND_ALPHA * 255),))
    img.paste(band, (0, H - BAND_H), band)
    title, dates = describe(index, chart)
    draw = ImageDraw.Draw(img)
    margin = 48
    title_font = fit_font(fonts['display'], title, 40, W - 2 * margin)
    small = ImageFont.truetype(fonts['text'], 22)
    top = H - BAND_H
    draw.text((margin, top + 60), title, font=title_font, fill=TEXT, anchor='ls')
    draw.text((margin, top + 98), dates, font=small, fill=MUTED, anchor='ls')
    draw.text((W - margin, top + 98), 'archive.aero', font=small, fill=TEXT, anchor='rs')

    quality = 82
    while True:
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=quality, progressive=True, optimize=True, subsampling=2)
        if buf.tell() <= TARGET_BYTES or quality <= MIN_QUALITY: break
        quality -= 4
    return buf.getvalue(), {'sources': pms, 'zoom': z, 'tiles': len(jobs), 'quality': quality, 'scale': round(view_scale, 3), 'bbox': [round(v, 4) for v in bbox]}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', required=True, help='directory of per-slug chart index JSON files')
    ap.add_argument('--out', required=True, help='output root; images land at <out>/sectionals/chart/<slug>/<date>.jpg')
    ap.add_argument('--tiles', default=TILE_BASE, help='tile endpoint base (default %(default)s)')
    ap.add_argument('--only', action='append', help='slug or slug/date; repeatable')
    ap.add_argument('--limit', type=int, help='render at most N pages')
    ap.add_argument('--workers', type=int, default=12, help='images rendered concurrently')
    ap.add_argument('--in-flight', type=int, default=MAX_IN_FLIGHT, help='tile requests in flight across all workers (default %(default)s)')
    ap.add_argument('--font-cache', default='~/Library/Caches/archive.aero/fonts')
    ap.add_argument('--force', action='store_true', help='re-render keys already in the manifest')
    ap.add_argument('--dry-run', action='store_true', help='plan only: print zoom and tile counts, fetch nothing')
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest_path, failures_path = out / 'manifest.jsonl', out / 'failures.jsonl'
    done = set()
    if manifest_path.exists() and not args.force:
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if (out / row['key']).exists(): done.add(row['key'])

    todo, eligible, skipped = [], 0, 0
    for index, chart in pages(args.index, args.only):
        eligible += 1
        key = f"sectionals/chart/{index['slug']}/{chart['d']}.jpg"
        if key in done: skipped += 1; continue
        todo.append((key, index, chart))
    if args.limit: todo = todo[:args.limit]
    print(f'{eligible} eligible pages, {skipped} already rendered, {len(todo)} to do', flush=True)

    if args.dry_run:
        total = 0
        for key, index, chart in todo:
            b = chart['pmb']
            if b[2] - b[0] > 180: print(f'{key}: antimeridian, needs metadata'); continue
            p = plan(b, (chart.get('pmz') or [0, 11])[1]); total += p['count']
            print(f"{key}: z{p['zoom']} {p['count']} tiles scale {p['scale']:.2f}")
        print(f'{total} tiles across {len(todo)} pages'); return

    fonts = load_fonts(args.font_cache)
    if fonts['fallback']: print('note: rendering with Helvetica Neue, not Barlow', flush=True)
    tiles = Tiles(args.tiles, in_flight=max(1, args.in_flight))
    per_page = max(1, args.in_flight // max(1, args.workers))   # in-flight requests per page; the total stays <= --in-flight
    lock = threading.Lock()
    counts = {'rendered': 0, 'failed': 0, 'bytes': 0}
    started = time.time()
    mf = open(manifest_path, 'a'); ff = open(failures_path, 'a')

    def one(item):
        key, index, chart = item
        try:
            data, info = render(index, chart, tiles, fonts, per_page)
            path = out / key; path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.jpg.part'); tmp.write_bytes(data); os.replace(tmp, path)
            row = {'key': key, 'slug': index['slug'], 'date': chart['d'], 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                   **info, 'rendered_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
            with lock:
                mf.write(json.dumps(row, separators=(',', ':')) + '\n'); mf.flush()
                counts['rendered'] += 1; counts['bytes'] += len(data)
                n = counts['rendered'] + counts['failed']
                if n % 50 == 0 or n == len(todo):
                    el = time.time() - started
                    print(f"{n}/{len(todo)} {el:.0f}s {counts['rendered']} ok {counts['failed']} failed, {tiles.stats['tiles']} tiles {tiles.stats['bytes'] / 1e6:.0f} MB, {tiles.stats['retries']} retries", flush=True)
        except Exception as e:
            with lock:
                ff.write(json.dumps({'key': key, 'slug': index['slug'], 'date': chart['d'], 'pm': chart['pm'], 'error': f'{type(e).__name__}: {e}',
                                     'rendered_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}) + '\n'); ff.flush()
                counts['failed'] += 1
                print(f'FAIL {key}: {type(e).__name__}: {e}', flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers), thread_name_prefix='page') as ex:
        list(ex.map(one, todo))
    el = time.time() - started
    print(f"done: {counts['rendered']} rendered, {counts['failed']} failed, {counts['bytes'] / 1e6:.1f} MB written, {el:.0f}s; "
          f"{tiles.stats['requests']} requests, {tiles.stats['tiles']} tiles ({tiles.stats['bytes'] / 1e6:.0f} MB), {tiles.stats['empty']} empty, {tiles.stats['retries']} retries", flush=True)


if __name__ == '__main__':
    main()
