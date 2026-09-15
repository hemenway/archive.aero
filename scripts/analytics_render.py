#!/usr/bin/env python3
"""Render the decoded view log as a map + time charts.

Reads what analytics_heatmap.py wrote and produces two views of the same data:

  heatmap.png   view density over the sectional grid, for a quick look or a
                slide — Web Mercator, log-scaled, sectional outlines for context
  heatmap.html  an explorable version: Leaflet grid you can hover and filter by
                chart decade, plus the two time axes (chart era vs wall clock)

Density, not tile counts. A z8 tile covers 256x the ground of a z12 tile, so a
raw per-tile count would make one distant glance outweigh a long close read.
Each view is spread evenly over the ground its tile covers and accumulated per
unit area, which is what makes deep-zoom attention show up as a hot spot.

Usage:
  ~/venv/bin/python scripts/analytics_render.py
  ~/venv/bin/python scripts/analytics_render.py --width 1400
"""

import argparse
import collections
import glob
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "worklists/data/analytics"
SECTIONAL_DIR = REPO / "shapefiles/sectional"

# Sequential blue ramp, light -> dark (validated default palette, steps 100-700).
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID_INK = "#d9d8d4"


# ------------------------------------------------------------------ mercator

def merc_y(lat):
    lat = max(min(lat, 85.05112878), -85.05112878)
    return (1 - math.log(math.tan(math.radians(lat))
                         + 1 / math.cos(math.radians(lat))) / math.pi) / 2


def merc_x(lon):
    return (lon + 180.0) / 360.0


def inv_merc_x(x):
    return x * 360.0 - 180.0


def inv_merc_y(y):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))


def tile_span(z, x, y):
    """Normalised Web Mercator square [x0,x1)x[y0,y1) for an XYZ tile."""
    n = 2.0**z
    return x / n, (x + 1) / n, y / n, (y + 1) / n


# ------------------------------------------------------------------ loading

def load_facts(out_dir=None):
    path = (out_dir or OUT_DIR) / "viewed_tiles.jsonl.gz"
    if not path.exists():
        sys.exit(f"{path} missing — run scripts/analytics_heatmap.py first")
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def load_sectional_polygons():
    """[(name, ogr.Geometry)] for point-in-chart lookup."""
    try:
        from osgeo import ogr
        ogr.UseExceptions()
    except ImportError:
        return []
    out = []
    for shp in sorted(SECTIONAL_DIR.glob("*.shp")):
        try:
            ds = ogr.Open(str(shp))
            layer = ds.GetLayer(0)
        except Exception:
            continue
        name = shp.stem.replace("_", " ").title()
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom is not None:
                out.append((name, geom.Clone()))
    return out


def chart_namer():
    """lon, lat -> sectional chart name (or '' when outside every chart)."""
    polys = load_sectional_polygons()
    if not polys:
        return lambda lon, lat: ""
    from osgeo import ogr

    def name_at(lon, lat):
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint_2D(lon, lat)
        for name, geom in polys:
            if geom.Contains(pt):
                return name
        return ""

    return name_at


def load_sectional_outlines():
    """Sectional chart boundaries as lists of (lon, lat) rings."""
    try:
        from osgeo import ogr
        ogr.UseExceptions()
    except ImportError:
        return []
    rings = []
    for shp in sorted(SECTIONAL_DIR.glob("*.shp")):
        try:
            ds = ogr.Open(str(shp))
            layer = ds.GetLayer(0)
        except Exception:
            continue
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            for i in range(geom.GetGeometryCount() or 1):
                part = geom.GetGeometryRef(i) if geom.GetGeometryCount() else geom
                if part is None:
                    continue
                sub = part.GetGeometryRef(0) if part.GetGeometryCount() else part
                if sub is None:
                    continue
                pts = [(sub.GetX(j), sub.GetY(j)) for j in range(sub.GetPointCount())]
                if len(pts) > 2:
                    rings.append(pts)
    return rings


# ------------------------------------------------------------------ density

def density_raster(facts, width, bbox):
    """Accumulate views per unit area into a raster over a mercator bbox."""
    x0, x1, y0, y1 = bbox
    height = max(1, int(round(width * (y1 - y0) / (x1 - x0))))
    acc = np.zeros((height, width), dtype=np.float64)
    for f in facts:
        tx0, tx1, ty0, ty1 = tile_span(f["z"], f["x"], f["y"])
        # to raster pixel coords
        px0 = (tx0 - x0) / (x1 - x0) * width
        px1 = (tx1 - x0) / (x1 - x0) * width
        py0 = (ty0 - y0) / (y1 - y0) * height
        py1 = (ty1 - y0) / (y1 - y0) * height
        ix0, ix1 = int(math.floor(px0)), max(int(math.ceil(px1)), int(px0) + 1)
        iy0, iy1 = int(math.floor(py0)), max(int(math.ceil(py1)), int(py0) + 1)
        ix0, iy0 = max(ix0, 0), max(iy0, 0)
        ix1, iy1 = min(ix1, width), min(iy1, height)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        acc[iy0:iy1, ix0:ix1] += f["views"] / ((ix1 - ix0) * (iy1 - iy0))
    return acc


def data_bbox(facts, pad=0.02, quantile=0.005):
    """Mercator bbox holding the bulk of the views, outliers trimmed."""
    xs, ys, ws = [], [], []
    for f in facts:
        tx0, tx1, ty0, ty1 = tile_span(f["z"], f["x"], f["y"])
        xs.append((tx0 + tx1) / 2)
        ys.append((ty0 + ty1) / 2)
        ws.append(f["views"])
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws, dtype=float)
    order = np.argsort(xs)
    cw = np.cumsum(ws[order]) / ws.sum()
    x0 = xs[order][np.searchsorted(cw, quantile)]
    x1 = xs[order][min(np.searchsorted(cw, 1 - quantile), len(xs) - 1)]
    order = np.argsort(ys)
    cw = np.cumsum(ws[order]) / ws.sum()
    y0 = ys[order][np.searchsorted(cw, quantile)]
    y1 = ys[order][min(np.searchsorted(cw, 1 - quantile), len(ys) - 1)]
    mx, my = (x1 - x0) * pad + 0.004, (y1 - y0) * pad + 0.004
    return (max(x0 - mx, 0), min(x1 + mx, 1), max(y0 - my, 0), min(y1 + my, 1))


# ------------------------------------------------------------------ png

# The lower 48 in mercator. 99% of views land here, and letting Alaska set the
# frame shrinks that 99% into the bottom half of the image.
CONUS = (merc_x(-125.5), merc_x(-66.5), merc_y(49.8), merc_y(24.2))


def choose_bbox(facts):
    """CONUS when it holds nearly everything, otherwise fit the data."""
    inside = sum(
        f["views"] for f in facts
        if CONUS[0] <= merc_x(f["lon"]) <= CONUS[1]
        and CONUS[2] <= merc_y(f["lat"]) <= CONUS[3]
    )
    total = sum(f["views"] for f in facts) or 1
    if inside / total >= 0.95:
        return CONUS, total - inside
    return data_bbox(facts), 0


def draw_panel(ax, facts, bbox, width, title, cmap, fig):
    from matplotlib.colors import LogNorm

    x0, x1, y0, y1 = bbox
    acc = density_raster(facts, width, bbox)
    h, w = acc.shape
    positive = acc[acc > 0]
    if positive.size:
        vmax = float(np.percentile(positive, 99.5))
        vmin = max(float(positive.min()), vmax / 3e3)
    else:
        vmin, vmax = 1.0, 10.0

    ax.set_facecolor(SURFACE)
    im = ax.imshow(np.ma.masked_where(acc <= 0, acc), origin="upper", zorder=2,
                   cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
                   interpolation="nearest")
    # Outlines sit above the density: underneath, the fills bury them.
    for ring in load_sectional_outlines():
        rx = [(merc_x(lon) - x0) / (x1 - x0) * w for lon, lat in ring]
        ry = [(merc_y(lat) - y0) / (y1 - y0) * h for lon, lat in ring]
        ax.plot(rx, ry, lw=0.4, color=INK, alpha=0.22, zorder=3)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    ax.set_title(title, color=INK, fontsize=8, loc="left", pad=6)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04,
                        pad=0.02, aspect=38)
    cbar.ax.tick_params(labelsize=5.5, colors=INK_2, length=2)
    cbar.outline.set_visible(False)
    return h, w


def render_png(facts, width, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    bbox, dropped = choose_bbox(facts)
    cmap = LinearSegmentedColormap.from_list("aa_blue", RAMP)
    cmap.set_bad(SURFACE)

    # Two readings of the same log. A z8 tile covers 256x the ground of a z12,
    # so mixing them lets one distant glance blanket a whole state; splitting
    # separates "had it on screen" from "leaned in and read it".
    broad = [f for f in facts if f["z"] <= 9]
    close = [f for f in facts if f["z"] >= 10]
    panels = [
        (broad, f"Broad browsing — z8-9, {sum(f['views'] for f in broad):,} views"),
        (close, f"Close reading — z10-12, {sum(f['views'] for f in close):,} views"),
    ]

    pw = width // 2
    ratio = (bbox[3] - bbox[2]) / (bbox[1] - bbox[0])
    fig, axes = plt.subplots(
        1, 2, figsize=(width / 100, pw / 100 * ratio + 1.0), dpi=200,
        layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    for ax, (subset, title) in zip(axes, panels):
        draw_panel(ax, subset, bbox, pw, title, cmap, fig)

    total = sum(f["views"] for f in facts)
    note = f"{total:,} sampled tile views"
    if dropped:
        note += f"; {dropped:,} outside the lower 48 not shown"
    fig.suptitle(
        f"archive.aero — where people looked   {note}   "
        f"density is views per unit ground area, log scale; "
        f"sectional outlines for reference",
        color=INK, fontsize=8.5, x=0.006, ha="left")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ html

def svg_bars(pairs, width, height, label_every=1, fmt=str):
    """Thin bars with 4px rounded data-ends, recessive baseline, hover titles."""
    if not pairs:
        return ""
    pad_l, pad_b, pad_t = 34, 22, 8
    iw, ih = width - pad_l - 8, height - pad_b - pad_t
    top = max(v for _, v in pairs) or 1
    n = len(pairs)
    slot = iw / n
    bw = max(2.0, slot - 2.0)  # 2px surface gap between adjacent bars
    out = []
    for i in range(4):
        gy = pad_t + ih * i / 3
        out.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + iw}" '
                   f'y2="{gy:.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" class="tick ar">'
                   f'{fmt(round(top * (3 - i) / 3))}</text>')
    for i, (k, v) in enumerate(pairs):
        bh = ih * v / top
        x = pad_l + i * slot + (slot - bw) / 2
        y = pad_t + ih - bh
        r = min(4, bw / 2, max(bh, 0.1))
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                   f'height="{max(bh, 0.6):.1f}" rx="{r:.1f}" class="bar">'
                   f'<title>{k}: {v:,} views</title></rect>')
        if i % label_every == 0 or i == n - 1:
            out.append(f'<text x="{x + bw / 2:.1f}" y="{height - 7}" '
                       f'class="tick mid">{k}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" class="chart" '
            f'preserveAspectRatio="xMidYMid meet">{"".join(out)}</svg>')


def render_html(facts, summary, geojson_path, out):
    gj = json.loads(Path(geojson_path).read_text())
    # Per-decade grid so the map can be filtered without recomputing in JS.
    Z = gj["zoom"]
    per_decade = collections.defaultdict(collections.Counter)
    for f in facts:
        dec = int(f["era_start"][:4]) // 10 * 10
        z, x, y, w = f["z"], f["x"], f["y"], f["views"]
        if z >= Z:
            per_decade[dec][(x >> (z - Z), y >> (z - Z))] += w
        else:
            span = 1 << (Z - z)
            for dx in range(span):
                for dy in range(span):
                    per_decade[dec][(x * span + dx, y * span + dy)] += w / (span * span)
    cells = collections.defaultdict(dict)
    for dec, counter in per_decade.items():
        for (x, y), v in counter.items():
            cells[f"{x},{y}"][str(dec)] = round(v, 3)

    # Fill the gap years with zero: bars placed by list index instead of by
    # year would silently close the 1980s-2000s hole and misstate the axis.
    years = [int(y) for y in summary["by_era_year"]]
    era_series = [(str(y), summary["by_era_year"].get(str(y), 0))
                  for y in range(min(years), max(years) + 1)]
    era_bars = svg_bars(era_series, 760, 190, label_every=10)
    day_bars = svg_bars(
        [(d[5:], v) for d, v in sorted(summary["by_day"].items())], 760, 170, label_every=3)
    hour_bars = svg_bars(
        [(f"{int(h):02d}", v) for h, v in sorted(summary["by_hour_utc"].items(), key=lambda kv: int(kv[0]))],
        760, 150, label_every=2)

    # Fit the map the same way the PNG frames itself. Fitting to the raw cell
    # extent instead pulls in a handful of Aleutian cells that straddle the
    # dateline and zooms the whole map out to the globe.
    bbox, _ = choose_bbox(facts)
    fit = [[inv_merc_y(bbox[3]), inv_merc_x(bbox[0])],
           [inv_merc_y(bbox[2]), inv_merc_x(bbox[1])]]

    top_cells = sorted(gj["features"], key=lambda f: -f["properties"]["views"])[:15]
    name_at = chart_namer()
    rows = []
    for f in top_cells:
        c = f["geometry"]["coordinates"][0]
        lon = (c[0][0] + c[1][0]) / 2
        lat = (c[0][1] + c[2][1]) / 2
        rows.append(f"<tr><td>{name_at(lon, lat) or '—'}</td>"
                    f"<td>{lat:.2f}, {lon:.2f}</td>"
                    f"<td class='num'>{f['properties']['views']:,.0f}</td></tr>")

    span = summary.get("log_span") or ["?", "?"]
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>archive.aero — view heatmap</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root {{
  color-scheme: light;
  --surface: {SURFACE}; --ink: {INK}; --ink2: {INK_2}; --grid: {GRID_INK};
  --line: #e6e5e1;
  --seq-lo: {RAMP[0]}; --seq-hi: {RAMP[-1]}; --bar: {RAMP[7]};
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme=light])) {{
    color-scheme: dark;
    --surface: #1a1a19; --ink: #fff; --ink2: #c3c2b7; --grid: #333330;
    --line: #2c2c2a; --bar: #3987e5;
  }}
}}
:root[data-theme=dark] {{
  color-scheme: dark;
  --surface: #1a1a19; --ink: #fff; --ink2: #c3c2b7; --grid: #333330;
  --line: #2c2c2a; --bar: #3987e5;
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:28px 24px 60px; background:var(--surface); color:var(--ink);
  font:14px/1.5 "Barlow","Helvetica Neue",system-ui,sans-serif; }}
.wrap {{ max-width: 880px; margin: 0 auto; }}
h1 {{ font-size:20px; font-weight:600; letter-spacing:-0.01em; margin:0 0 4px; }}
h2 {{ font-size:13px; font-weight:600; margin:34px 0 2px; letter-spacing:0.02em;
  text-transform:uppercase; color:var(--ink2); }}
p.sub {{ color:var(--ink2); margin:0 0 18px; font-size:13px; }}
p.note {{ color:var(--ink2); font-size:12px; margin:4px 0 10px; }}
#map {{ height:460px; border:1px solid var(--line); background:var(--surface); }}
.controls {{ display:flex; gap:6px; flex-wrap:wrap; margin:12px 0 8px; }}
.controls button {{ font:inherit; font-size:12px; padding:4px 10px; cursor:pointer;
  border:1px solid var(--line); background:transparent; color:var(--ink2); border-radius:2px; }}
.controls button[aria-pressed=true] {{ border-color:var(--bar); color:var(--ink);
  font-weight:600; }}
.legend {{ display:flex; align-items:center; gap:8px; font-size:11px; color:var(--ink2);
  margin-top:8px; }}
.legend .ramp {{ height:8px; width:180px;
  background:linear-gradient(90deg,var(--seq-lo),var(--seq-hi)); }}
.chart {{ width:100%; height:auto; overflow:visible; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.bar {{ fill:var(--bar); }}
.bar:hover {{ fill:var(--ink); }}
.tick {{ fill:var(--ink2); font-size:9px; }}
.ar {{ text-anchor:end; }} .mid {{ text-anchor:middle; }}
table {{ border-collapse:collapse; font-size:12px; margin-top:6px; }}
th,td {{ text-align:left; padding:3px 14px 3px 0; border-bottom:1px solid var(--line); }}
th {{ color:var(--ink2); font-weight:600; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
details {{ margin-top:10px; }} summary {{ cursor:pointer; color:var(--ink2); font-size:12px; }}
.caveats li {{ color:var(--ink2); font-size:12.5px; margin-bottom:5px; }}
</style>
<div class="wrap">
<h1>Where and when people looked</h1>
<p class="sub">{summary['sampled_views_placed']:,} sampled tile views placed on the
map, {span[0]} to {span[1]}. Two different clocks: <em>chart era</em> is the
historical date being viewed, <em>wall clock</em> is when someone was viewing.</p>

<div class="controls" id="decades"></div>
<div id="map"></div>
<div class="legend"><span>fewer views</span><span class="ramp"></span><span>more</span>
<span style="margin-left:auto">grid: z{Z} cells, density-normalised</span></div>

<h2>Chart era viewed</h2>
<p class="note">Which years of the archive people actually open, by chart edition date.</p>
{era_bars}

<h2>Wall clock — views per day</h2>
{day_bars}

<h2>Wall clock — hour of day (UTC)</h2>
{hour_bars}

<h2>Busiest cells</h2>
<table><thead><tr><th>sectional chart</th><th>centre lat, lon</th>
<th class="num">views</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>

<details><summary>How this is measured, and what it can't tell you</summary>
<ul class="caveats">
<li>Each row is a PMTiles byte-range read decoded back to a map tile through the
archive's own directories. Position is the tile, not a cursor.</li>
<li>The Worker samples {summary['sample_rate']:.0%} of qualifying reads, so the true
tile-read count is roughly {summary['estimated_true_tile_reads']:,}.</li>
<li>Tiles cache in the browser for 24 h, so a revisit inside a day is invisible.
This counts distinct tile fetches, not time spent looking.</li>
<li>Scrubbing the timeline prefetches tiles, so the era histogram counts eras
passed through as well as eras studied.</li>
<li>{summary['sampled_views_undecodable']:,} sampled views could not be placed: they
were logged against archive builds that were later replaced, and byte offsets do
not survive a rebuild.</li>
<li>Reads at or below 1 KB are dropped as metadata, which also drops the smallest
overview tiles — this map is biased toward zoomed-in viewing.</li>
</ul></details>
</div>
<script>
const CELLS = {json.dumps(cells, separators=(',', ':'))};
const FIT = {json.dumps(fit)};
const Z = {Z}, DECADES = {json.dumps(sorted(per_decade))};
const RAMP = {json.dumps(RAMP)};
const map = L.map('map', {{ scrollWheelZoom: false }});
const dark = matchMedia('(prefers-color-scheme: dark)').matches
  && document.documentElement.dataset.theme !== 'light';
L.tileLayer(`https://{{s}}.basemaps.cartocdn.com/${{dark ? 'dark' : 'light'}}_all/{{z}}/{{x}}/{{y}}{{r}}.png`, {{
  attribution: '&copy; OpenStreetMap, &copy; CARTO', maxZoom: 12, opacity: 0.55
}}).addTo(map);
let layer = null, active = 'all';
// Fit after layout: called while the container is still 0x0 (which happens
// when the script runs before the stylesheet settles) fitBounds lands on
// zoom 0 with a nonsense centre and never recovers.
function fitMap() {{ map.invalidateSize(); map.fitBounds(FIT); }}
addEventListener('load', fitMap);
new ResizeObserver(() => map.invalidateSize())
  .observe(document.getElementById('map'));

function tileBounds(x, y) {{
  const n = 2 ** Z;
  const lonW = x / n * 360 - 180, lonE = (x + 1) / n * 360 - 180;
  const latN = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) * 180 / Math.PI;
  const latS = Math.atan(Math.sinh(Math.PI * (1 - 2 * (y + 1) / n))) * 180 / Math.PI;
  return [[latS, lonW], [latN, lonE]];
}}

function draw() {{
  if (layer) map.removeLayer(layer);
  layer = L.layerGroup();
  const vals = [];
  for (const k in CELLS) {{
    const v = active === 'all'
      ? Object.values(CELLS[k]).reduce((a, b) => a + b, 0)
      : (CELLS[k][active] || 0);
    if (v > 0) vals.push([k, v]);
  }}
  if (!vals.length) {{ layer.addTo(map); return; }}
  const max = Math.max(...vals.map(v => v[1]));
  const lmax = Math.log1p(max);
  for (const [k, v] of vals) {{
    const [x, y] = k.split(',').map(Number);
    const t = Math.log1p(v) / lmax;
    const color = RAMP[Math.min(RAMP.length - 1, Math.floor(t * RAMP.length))];
    const b = tileBounds(x, y);
    L.rectangle(b, {{ stroke: false, fillColor: color,
      fillOpacity: 0.25 + 0.6 * t }})
      .bindTooltip(`${{v.toFixed(0)}} views<br>${{b[0][0].toFixed(2)}}, ${{b[0][1].toFixed(2)}}`,
        {{ sticky: true }})
      .addTo(layer);
  }}
  layer.addTo(map);
}}

const bar = document.getElementById('decades');
for (const d of ['all', ...DECADES]) {{
  const b = document.createElement('button');
  b.textContent = d === 'all' ? 'All eras' : d + 's';
  b.setAttribute('aria-pressed', d === 'all');
  b.onclick = () => {{
    active = String(d);
    [...bar.children].forEach(c => c.setAttribute('aria-pressed', c === b));
    draw();
  }};
  bar.appendChild(b);
}}
draw();
fitMap();
</script>
"""
    Path(out).write_text(html)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1100, help="PNG raster width px")
    ap.add_argument("--tag", help="read/write a named subdirectory "
                                  "(matches analytics_heatmap.py --tag/--days)")
    ap.add_argument("--days", type=int,
                    help="shorthand for --tag last<N>d (the subdirectory analytics_heatmap.py --days N writes)")
    args = ap.parse_args()
    if args.days and not args.tag:
        args.tag = f"last{args.days}d"

    out_dir = OUT_DIR / args.tag if args.tag else OUT_DIR
    if not (out_dir / "summary.json").exists():
        sys.exit(f"no summary.json in {out_dir} — run scripts/analytics_heatmap.py"
                 + (f" --days {args.days}" if args.days else "") + " first")
    facts = load_facts(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text())
    if not facts or not summary.get("by_era_year"):
        sys.exit("no placed tile views in this window; nothing to render")
    gj = next(iter(sorted(out_dir.glob("heatmap_z*.geojson"))), None)
    if gj is None:
        sys.exit("no heatmap_z*.geojson — run scripts/analytics_heatmap.py first")

    png = render_png(facts, args.width, out_dir / "heatmap.png")
    html = render_html(facts, summary, gj, out_dir / "heatmap.html")
    print(f"wrote {png.relative_to(REPO)}")
    print(f"wrote {Path(html).relative_to(REPO)}")


if __name__ == "__main__":
    main()
