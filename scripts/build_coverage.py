"""Regenerate coverage.json (spatial availability per date) from timeline_data.json.

For every chart in the collection this rasterizes its location's extent ring
(simplified NAD83 lon/lat rings already emitted by build_timeline_data.py) onto a
0.1-degree grid clipped to the lower-48 reference bbox, sweeps the timeline, and for
each segment between chart validity boundaries records what % of the *charted*
lower-48 area (the union of every extent ever, clipped to the bbox) is covered by
the charts in effect (validity is end-exclusive, matching _rangesForDate in
index.html).

Why per-chart rings and not the PMTiles header bounds of the era archives (the
pre-2026-08 approach): a header bbox is the single union rect of everything in the
mosaic, so one Alaska or Hawaii chart in an archive stretched the rect across ocean
and uncharted CONUS (2020 read 99% when the honest footprint was ~74% of the bbox),
and antimeridian archives wrapped to near-global rects. AK/HI/territory extents fall
outside the reference bbox, so they count toward chart count but never toward area —
regardless of statehood era.

Why the denominator is the union of charted extents and not the raw bbox: the bbox
is ~25% ocean, so even complete lower-48 coverage topped out at 75% of it. Against
the charted-area denominator a full modern cycle reads ~98% ("Full coverage" in the
viewer, threshold 95); 1955 reads ~86% instead of the old misleading 65%.

Caveat: this measures the catalog inventory (what build_timeline_data.py saw), not
bucket reachability — an era mosaic missing from R2 still counts. That is the right
signal for the heat strip, which describes holdings, not uptime.

Run after every build_timeline_data.py run:
  ~/venv/bin/python scripts/build_coverage.py
"""
import datetime as dt, json, os

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMELINE = os.path.join(ROOT, 'timeline_data.json')
OUT = os.path.join(ROOT, 'coverage.json')

# Lower-48 reference bbox; AK/HI/territory extents land outside it and so
# contribute chart count but no area.
LON0, LON1 = -125.0, -66.5
LAT0, LAT1 = 24.5, 49.5
STEP = 0.1
NX = int(round((LON1 - LON0) / STEP))
NY = int(round((LAT1 - LAT0) / STEP))


def rasterize(rings):
    img = Image.new('1', (NX, NY), 0)
    draw = ImageDraw.Draw(img)
    for ring in rings:
        draw.polygon([((lon - LON0) / STEP, (lat - LAT0) / STEP) for lon, lat in ring],
                     fill=1)
    return np.array(img, dtype=bool)


def main():
    td = json.load(open(TIMELINE))
    masks = {ref: rasterize(rings) for ref, rings in td['rings'].items()}
    charted = np.zeros((NY, NX), dtype=bool)
    for m in masks.values():
        charted |= m
    denom = int(charted.sum())

    # (start, end, ref) per chart; ref can be None (Key West local, no extent —
    # counts but adds no area) and AK/HI refs rasterize to empty masks here.
    charts = [(c['d'], c['e'], loc.get('ref'))
              for loc in td['locations'].values() for c in loc['charts']]

    days = sorted({d for c in charts for d in c[:2]})
    segments = []
    for a, b in zip(days, days[1:]):
        active = [ref for d, e, ref in charts if d <= a and b <= e]
        if not active:
            segments.append([a, b, 0, 0.0])
            continue
        mask = np.zeros((NY, NX), dtype=bool)
        for ref in set(active):
            m = masks.get(ref)
            if m is not None:
                mask |= m
        pct = round(100.0 * mask.sum() / denom, 1)
        segments.append([a, b, len(active), pct])

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
        'note': 'pct = % of the charted lower-48 area (union of all extent rings '
                'clipped to ref bbox) covered by the charts in effect '
                '(end-exclusive); AK/HI/territories count toward count, not area',
        'segments': merged,
    }
    json.dump(out, open(OUT, 'w'), separators=(',', ':'))
    print(f'{len(segments)} segments -> {len(merged)} merged -> coverage.json '
          f'({os.path.getsize(OUT) // 1024} KB)')


if __name__ == '__main__':
    main()
