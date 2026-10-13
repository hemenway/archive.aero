#!/usr/bin/env python3
"""Build the data behind the airport pages, /airports/<icao> (worklist 09 B1, URI-POLICY.md).

For each airport in the pilot set: every chart edition in the archive whose sheet covers the airport
point, grouped by decade with first/last dates. An edition covers the airport when the point lies
inside the location's extent ring from timeline_data.json (the charted face) and, for a tiled
edition, inside the artifact's own lon/lat box (`pmb`) as well - a full-sheet artifact box includes
the collar, so on its own it claims airports a neighbour sheet only shows in its margin (Denver on
the Wichita sheet). pmb alone decides when the location has no ring. Untiled editions count toward
the editions held but carry `tiled: false` and have no page to link.

Inputs: the OurAirports CSV (historical-data/), the per-location chart index folder that
scripts/next_build_overlays.py writes (charts.<hash>/) and timeline_data.json. Output: a
content-hashed airports.<hash>/ folder (<ident>.json per airport + index.json), the airports.json
summary the manifest builder reads, and sitemap-airports.xml.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

# The 50 busiest US airports (by enplanements), chosen 2026-10-12 as the B1 pilot set.
PILOT = ('KATL KDFW KDEN KORD KLAX KJFK KLAS KMCO KMIA KCLT KSEA KPHX KEWR KSFO KIAH KBOS KFLL KMSP KLGA KDTW '
         'KPHL KSLC KDCA KSAN KBWI KTPA KAUS KIAD KBNA KMDW PHNL KDAL KPDX KSTL KRDU KHOU KSMF KMSY KSJC KSNA '
         'KOAK KMCI KSAT KRSW KCLE KIND KPIT KCVG KCMH KPBI').split()
MAX_SECTIONALS = 6  # a sectional is ~4x6 degrees; more distinct sectional sheets over one point means bad extents


def compact(value): return json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()
def digest(data): return hashlib.sha256(data).hexdigest()[:12]


def point_in_ring(lon, lat, ring):
    """Even-odd test of one closed polygon ring ([[lon,lat],...]); a ring spanning the antimeridian
    (points both east of 150 and west of -90) is unwrapped westward, the point with it."""
    xs = [float(p[0]) for p in ring]; ys = [float(p[1]) for p in ring]
    if any(x > 150 for x in xs) and any(x < -90 for x in xs):
        xs = [x - 360 if x > 150 else x for x in xs]
        if lon > 150: lon -= 360
    inside = False; j = len(xs) - 1
    for i in range(len(xs)):
        if (ys[i] > lat) != (ys[j] > lat) and lon < (xs[j] - xs[i]) * (lat - ys[i]) / (ys[j] - ys[i]) + xs[i]:
            inside = not inside
        j = i
    return inside


def point_in_rings(lon, lat, rings): return any(point_in_ring(lon, lat, r) for r in rings)


def point_in_box(lon, lat, box):
    w, s, e, n = box
    if not s <= lat <= n: return False
    return (w <= lon <= e) if w <= e else (lon >= w or lon <= e)  # w > e: a box wrapped across the antimeridian


def covers(lon, lat, chart, rings):
    """Inside the location ring and, when tiled, inside the artifact box too (a box miss excludes; a
    box hit outside the ring is the collar, not the chart). pmb alone when there is no ring; None when
    neither is known - the edition cannot be placed."""
    box = point_in_box(lon, lat, chart['pmb']) if chart.get('pmb') else None
    if rings:
        hit = point_in_rings(lon, lat, rings)
        return hit and box if box is not None else hit
    return box


def read_airports(path, idents):
    """The OurAirports rows for the idents, keyed by ident; a missing ident is an error, not a silent gap."""
    wanted = {i.upper() for i in idents}; rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['ident'] in wanted: rows[r['ident']] = r
    missing = sorted(wanted - set(rows))
    if missing: raise ValueError('idents not in the airports CSV: ' + ' '.join(missing))
    out = {}
    for icao, r in rows.items():
        out[icao] = {'ident': icao.lower(), 'icao': icao, 'iata': r.get('iata_code') or None, 'name': r['name'],
                     'city': r.get('municipality') or None, 'region': r.get('iso_region') or None,
                     'lat': round(float(r['latitude_deg']), 4), 'lon': round(float(r['longitude_deg']), 4),
                     'elev_ft': int(float(r['elevation_ft'])) if r.get('elevation_ft') else None}
    return out


def read_idents(path):
    return [w for line in Path(path).read_text().splitlines() for w in [line.split('#')[0].strip()] if w]


def read_chart_index(folder):
    entries = [json.loads(p.read_text()) for p in sorted(Path(folder).glob('*.json'))]
    for e in entries:
        if e['type'] not in ('sectional', 'tac', 'wac'): raise ValueError(f"{e['slug']}: unknown chart type {e['type']!r}")
    return entries


def airport_charts(airport, index, rings, warn=None):
    """Every edition covering the airport point: [{slug,name,type,d,e,ed,tiled,og?}] sorted by date then slug."""
    lon, lat = airport['lon'], airport['lat']; out = []; unplaced = []
    for entry in index:
        ring = rings.get(entry['ref']) if entry.get('ref') else None
        name = entry['names'][0] if entry.get('names') else entry['slug']
        for c in entry['charts']:
            hit = covers(lon, lat, c, ring)
            if hit is None: unplaced.append(entry['slug']); continue
            if not hit: continue
            rec = {'slug': entry['slug'], 'name': name, 'type': entry['type'], 'd': c['d'], 'e': c.get('e'),
                   'ed': c.get('ed'), 'tiled': bool(c.get('pm'))}
            if c.get('og'): rec['og'] = c['og']
            out.append(rec)
    if unplaced and warn: warn(f"{airport['icao']}: {len(unplaced)} unplaceable editions (no pmb, no ring) in {sorted(set(unplaced))}")
    return sorted(out, key=lambda r: (r['d'], r['slug']))


def decades(charts):
    by = {}
    for c in charts:
        dec = int(c['d'][:4]) // 10 * 10
        g = by.setdefault(dec, {'decade': dec, 'editions': 0, 'tiled': 0, 'first': c['d'], 'last': c['d']})
        g['editions'] += 1; g['tiled'] += c['tiled']
        g['first'] = min(g['first'], c['d']); g['last'] = max(g['last'], c['d'])
    return [by[k] for k in sorted(by)]


def airport_record(airport, charts):
    rec = dict(airport)
    rec.update({'first': min((c['d'] for c in charts), default=None), 'last': max((c['d'] for c in charts), default=None),
                'editions': len(charts), 'tiled': sum(c['tiled'] for c in charts), 'decades': decades(charts), 'charts': charts})
    return rec


def index_record(rec): return {k: rec[k] for k in ('ident', 'icao', 'iata', 'name', 'city', 'region', 'lat', 'lon', 'first', 'last', 'editions', 'tiled')}


def sitemap_xml(records, origin):
    """sitemap-airports.xml: the canonical (lowercase) address of every airport page, in ident order."""
    import re
    if not re.fullmatch(r'https://[\w.-]+', origin): raise ValueError('sitemap origin must be an https origin')
    body = ''.join(f' <url><loc>{origin}/airports/{r["ident"]}</loc></url>\n' for r in records)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{body}</urlset>\n', len(records)


def build(airports_csv, charts_dir, timeline_path, out, idents=None, sitemap_origin='https://archive.aero', warn=None):
    warn = warn or (lambda msg: print('warning: ' + msg, file=sys.stderr))
    idents = list(idents or PILOT)
    if len(set(i.upper() for i in idents)) != len(idents): raise ValueError('duplicate idents')
    airports = read_airports(airports_csv, idents); index = read_chart_index(charts_dir)
    rings = json.loads(Path(timeline_path).read_text()).get('rings', {})
    records = []
    for icao in sorted(airports):
        charts = airport_charts(airports[icao], index, rings, warn)
        sectionals = {c['slug'] for c in charts if c['type'] == 'sectional'}
        if len(sectionals) > MAX_SECTIONALS: warn(f'{icao}: covered by {len(sectionals)} sectional locations {sorted(sectionals)} - check the extents')
        if not charts: warn(f'{icao}: no edition covers it')
        records.append(airport_record(airports[icao], charts))
    files = {r['ident'] + '.json': compact(r) for r in records}
    files['index.json'] = compact([index_record(r) for r in records])
    h = digest(b''.join(name.encode() + b'\0' + data for name, data in sorted(files.items())))
    out = Path(out); folder = out / ('airports.' + h); folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items(): (folder / name).write_bytes(data)
    result = {'airports': {'base': 'next/' + folder.name + '/', 'count': len(records)}}
    if sitemap_origin:
        xml, n = sitemap_xml(records, sitemap_origin); (out / 'sitemap-airports.xml').write_text(xml)
        result['sitemap'] = {'file': 'sitemap-airports.xml', 'urls': n}
    (out / 'airports.json').write_bytes(compact(result)); return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--airports', default='historical-data/ourairports-airports-2026-08-22.csv')
    ap.add_argument('--charts', required=True, metavar='DIR', help='per-location chart index folder (charts.<hash>/)')
    ap.add_argument('--timeline', default='timeline_data.json')
    ap.add_argument('--out', default=str(Path.home() / 'archive-next-build/prod/next'))
    ap.add_argument('--idents', metavar='FILE', help='one ICAO ident per line; replaces the pilot set')
    ap.add_argument('--sitemap', metavar='ORIGIN', default='https://archive.aero')
    args = ap.parse_args()
    idents = read_idents(args.idents) if args.idents else None
    print(json.dumps(build(args.airports, args.charts, args.timeline, args.out, idents, args.sitemap)))


if __name__ == '__main__': main()
