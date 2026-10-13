#!/usr/bin/env python3
"""Build C4 airfields, C5 pin shards and the per-location chart index from local, unpublished catalogs.

--og-manifest FILE   scripts/next_og_images.py's manifest.jsonl: every tiled edition whose card image
                     (key sectionals/chart/<slug>/<date>.jpg) is listed there gets `og` in the chart
                     index, which the edition pages serve as og:image (worklist 09 B3).
--sitemap ORIGIN     also write sitemap-sectionals.xml beside the chart index: one <url> per edition
                     page (tiled charts only - the thin-content guard 404s the rest), so the apex
                     sitemap index can list it (09 B4, URI-POLICY checklist item 2).
"""
import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path


def compact(value): return json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()
def digest(data): return hashlib.sha256(data).hexdigest()[:12]
def mercator(lon, lat):
    if not math.isfinite(lon) or not math.isfinite(lat) or not -90 <= lat <= 90:
        raise ValueError('invalid coordinates')
    lat = max(-85.0511287798066, min(85.0511287798066, lat))
    my=(1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2
    return ((lon+180)/360) % 1, max(0,min(1,my))


def airfields(geo):
    arrays = [[] for _ in range(5)]; details = []
    for feature in geo['features']:
        if feature['geometry']['type'] != 'Point': raise ValueError('airfield must be a Point')
        lon, lat = feature['geometry']['coordinates'][:2]; p = dict(feature['properties'])
        mx, my = mercator(float(lon),float(lat))
        status = p.get('status','unknown')
        if status not in ('open','gone','unknown'): raise ValueError('unknown airfield status '+str(status))
        for k in ('start_year','end_year','last_known_year'):
            p[k] = int(p[k]) if p.get(k) else None
            if p[k] is not None and not 1 <= p[k] <= 65535: raise ValueError('year outside uint16')
        p['status'] = status
        for arr, val in zip(arrays,(mx,my,p['start_year'] or 0,p['end_year'] or 0,
                                    ('open','gone','unknown').index(status))): arr.append(val)
        details.append(p)
    n = len(details); out = bytearray(b'AAAF1\0\0\0'+struct.pack('<II',n,0))
    for fmt, values in zip(('f','f','H','H','B'),arrays):
        out.extend(b'\0' * (-len(out)%4)); out.extend(struct.pack('<'+str(n)+fmt,*values))
    return bytes(out), details


def shard_indices(rings, margin=2, zoom=5):
    pts = [p for ring in rings for p in ring]
    if not pts: return []
    xs = [float(p[0]) for p in pts]; ys = [float(p[1]) for p in pts]
    if any(x > 150 for x in xs) and any(x < -90 for x in xs):
        xs = [x-360 if x > 150 else x for x in xs]
    w, e = min(xs)-margin, max(xs)+margin
    s, n = max(-85.0511287798066,min(ys)-margin), min(85.0511287798066,max(ys)+margin)
    size = 2**zoom
    x0, x1 = math.ceil((w+180)/360*size)-1, math.floor((e+180)/360*size)
    y0 = max(0,min(size-1,math.ceil(mercator(0,n)[1]*size)-1))
    y1 = max(0,min(size-1,math.floor(mercator(0,s)[1]*size)))
    return sorted({y*size+(x%size) for y in range(y0,y1+1)
                   for x in range(x0,min(x1,x0+size-1)+1)})


def version_charts(location, versions):
    loc = dict(location); charts = []
    for chart in loc.get('charts',[]):
        c = dict(chart)
        if c.get('pm'):
            keys = c['pm'] if isinstance(c['pm'],list) else [c['pm']]
            rows=[]
            for key in keys:
                r=versions.get(key) or versions.get('sectionals/'+key) or versions.get(key+'.pmtiles') or versions.get('sectionals/'+key+'.pmtiles')
                if r is None or not r.get('z'): raise ValueError('missing version/header for chart '+key)
                rows.append(r)
            paths=[r['path'] for r in rows]; c['pm']=paths if isinstance(c['pm'],list) else paths[0]
            c['pmz']=[min(r['z'][0] for r in rows),max(r['z'][1] for r in rows)]
            bs=[r.get('b') for r in rows]
            if any(b is None for b in bs):
                # A conservative world bbox safely contains wrapped half-sheet pairs.
                c['pmb']=[-180,-85.0512,180,85.0512]
            else: c['pmb']=[min(b[0] for b in bs),min(b[1] for b in bs),max(b[2] for b in bs),max(b[3] for b in bs)]
        charts.append(c)
    loc['charts']=charts; return loc


def pins(timeline, plan, margin=2):
    if margin < 0 or not math.isfinite(margin): raise ValueError('invalid margin')
    versions={r['old']:r for r in plan}; shards={}
    for name, location in sorted(timeline['locations'].items()):
        ref=location.get('ref'); rings=timeline.get('rings',{}).get(ref)
        if not rings: continue # ChartIndex.query also omits locations without a ring.
        loc=version_charts(location,versions)
        for i in shard_indices(rings,margin):
            shard=shards.setdefault(i,{'locations':{},'rings':{}})
            shard['locations'][name]=loc; shard['rings'][ref]=rings
    return shards


# A per-chart artifact key names its edition page: chart/<slug>/<date>[-half].
CHART_KEY = re.compile(r'^(?:sectionals/)?chart/([a-z0-9_]+)/(\d{4}-\d{2}-\d{2})(-[a-z]+)?(?:\.pmtiles)?$')


def chart_slug(pm):
    """The location slug of a chart's artifact key(s); a half-sheet pair must agree."""
    slugs=set()
    for key in (pm if isinstance(pm,list) else [pm]):
        m=CHART_KEY.match(key)
        if not m: raise ValueError('unrecognised chart artifact key '+key)
        slugs.add(m.group(1))
    if len(slugs)!=1: raise ValueError('half-sheet pair spans slugs: '+repr(pm))
    return slugs.pop()


def ring_box(rings):
    pts=[p for ring in rings for p in ring]
    xs=[float(p[0]) for p in pts]; ys=[float(p[1]) for p in pts]
    am=any(x>150 for x in xs) and any(x<-90 for x in xs)
    return [min(xs),min(ys),max(xs),max(ys)], am


def location_type(name):
    return 'tac' if name.endswith(' TAC') else 'wac' if name.endswith(' WAC') else 'sectional'


def og_key(slug, date): return f'sectionals/chart/{slug}/{date}.jpg'


def read_og_manifest(path):
    """The card images rendered so far: {key} lines from scripts/next_og_images.py."""
    keys=set()
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        key=json.loads(line).get('key')
        if not re.fullmatch(r'sectionals/chart/[a-z0-9_]+/\d{4}-\d{2}-\d{2}\.jpg',key or ''): raise ValueError('unexpected og image key '+repr(key))
        keys.add(key)
    return keys


def charts_index(timeline, plan, og_keys=None):
    """One record per location slug: every edition held (tiled or not), versioned artifacts, footprint box.

    The edition pages (/sectionals/chart/<slug>/<date>) and the shell's chart view read these; a
    slug is the artifact key's location segment, so a location without any artifact has no page.
    og_keys (optional): the rendered card images; a tiled edition with one carries it as `og`."""
    versions={r['old']:r for r in plan}; by_slug={}
    for name, location in sorted(timeline['locations'].items()):
        slugs={chart_slug(c['pm']) for c in location.get('charts',[]) if c.get('pm')}
        if len(slugs)>1: raise ValueError(f'{name}: charts under several slugs {sorted(slugs)}')
        if not slugs: continue
        slug=slugs.pop(); loc=version_charts(location,versions)
        entry=by_slug.setdefault(slug,{'slug':slug,'names':[],'ref':location.get('ref'),'type':location_type(name),'bbox':None,'am':False,'charts':[]})
        entry['names'].append(name)
        rings=timeline.get('rings',{}).get(location.get('ref'))
        if rings:
            box,am=ring_box(rings)
            if am: entry['am']=True
            elif entry['bbox'] is None: entry['bbox']=box
            else: entry['bbox']=[min(entry['bbox'][0],box[0]),min(entry['bbox'][1],box[1]),max(entry['bbox'][2],box[2]),max(entry['bbox'][3],box[3])]
        entry['charts'].extend(loc['charts'])
    for entry in by_slug.values():
        if entry['am']: entry['bbox']=None
        seen=set(); charts=[]
        # Alternate scans of one edition share (d, e); the first listed stands for it, as in ChartIndex.
        for c in sorted(entry['charts'],key=lambda c:(c['d'],c.get('e') or '')):
            k=(c['d'],c.get('e') or '')
            if k in seen: continue
            seen.add(k); charts.append(c)
        for c in charts:
            if c.get('pm') and og_keys is not None and og_key(entry['slug'],c['d']) in og_keys: c['og']=og_key(entry['slug'],c['d'])
        entry['charts']=charts
    return by_slug


def sitemap_xml(index, origin):
    """sitemap-sectionals.xml: the canonical address of every tiled edition page, in slug then date order."""
    if not re.fullmatch(r'https://[\w.-]+',origin): raise ValueError('sitemap origin must be an https origin')
    urls=[f'{origin}/sectionals/chart/{slug}/{c["d"]}' for slug,entry in sorted(index.items()) for c in entry['charts'] if c.get('pm')]
    if len(urls)>50000: raise ValueError('a sitemap holds at most 50,000 URLs; split it')
    body=''.join(f' <url><loc>{u}</loc></url>\n' for u in urls)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{body}</urlset>\n', len(urls)


def build(airfield_path, timeline_path, plan_path, out, margin=2, og_manifest=None, sitemap_origin=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); result={'airfields':None,'pins':None,'charts':None}
    if airfield_path:
        data,details=airfields(json.loads(Path(airfield_path).read_text()))
        # Pair identity includes details: changes to a URL or name never overwrite old details.
        details_data=compact(details); h=digest(data+details_data)
        stem='airfields.'+h
        (out/(stem+'.bin')).write_bytes(data); (out/(stem+'.json')).write_bytes(details_data)
        result['airfields']={'bin':'next/'+stem+'.bin','details':'next/'+stem+'.json'}
    if timeline_path:
        plan=json.loads(Path(plan_path).read_text()) if plan_path else []
        shards=pins(json.loads(Path(timeline_path).read_text()),plan,margin)
        contents={i:compact(shard) for i,shard in sorted(shards.items())}
        h=digest(b''.join(struct.pack('<I',i)+data for i,data in contents.items()))
        folder=out/('pins.'+h); folder.mkdir(exist_ok=True)
        for i,data in contents.items(): (folder/(str(i)+'.json')).write_bytes(data)
        result['pins']={'z':5,'margin':margin,'base':'next/'+folder.name+'/', 'shards':list(contents)}
        og_keys=read_og_manifest(og_manifest) if og_manifest else None
        index=charts_index(json.loads(Path(timeline_path).read_text()),plan,og_keys)
        files={slug:compact(entry) for slug,entry in sorted(index.items())}
        h=digest(b''.join(slug.encode()+b'\0'+data for slug,data in files.items()))
        folder=out/('charts.'+h); folder.mkdir(exist_ok=True)
        for slug,data in files.items(): (folder/(slug+'.json')).write_bytes(data)
        result['charts']={'base':'next/'+folder.name+'/'}
        if og_keys is not None:
            carded=sum(1 for e in index.values() for c in e['charts'] if c.get('og')); tiled=sum(1 for e in index.values() for c in e['charts'] if c.get('pm'))
            result['og']={'pages':tiled,'with_card':carded}
        if sitemap_origin:
            xml,n=sitemap_xml(index,sitemap_origin); (out/'sitemap-sectionals.xml').write_text(xml)
            result['sitemap']={'file':'sitemap-sectionals.xml','urls':n}
    (out/'overlays.json').write_bytes(compact(result)); return result


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--airfields'); ap.add_argument('--timeline')
    ap.add_argument('--plan'); ap.add_argument('--out',required=True); ap.add_argument('--margin',type=float,default=2)
    ap.add_argument('--og-manifest'); ap.add_argument('--sitemap',metavar='ORIGIN')
    args=ap.parse_args(); print(json.dumps(build(args.airfields,args.timeline,args.plan,args.out,args.margin,args.og_manifest,args.sitemap)))

if __name__ == '__main__': main()
