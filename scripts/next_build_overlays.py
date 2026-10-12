#!/usr/bin/env python3
"""Build C4 airfields, C5 pin shards and the per-location chart index from local, unpublished catalogs."""
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


def charts_index(timeline, plan):
    """One record per location slug: every edition held (tiled or not), versioned artifacts, footprint box.

    The edition pages (/sectionals/chart/<slug>/<date>) and the shell's chart view read these; a
    slug is the artifact key's location segment, so a location without any artifact has no page."""
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
        entry['charts']=charts
    return by_slug


def build(airfield_path, timeline_path, plan_path, out, margin=2):
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
        index=charts_index(json.loads(Path(timeline_path).read_text()),plan)
        files={slug:compact(entry) for slug,entry in sorted(index.items())}
        h=digest(b''.join(slug.encode()+b'\0'+data for slug,data in files.items()))
        folder=out/('charts.'+h); folder.mkdir(exist_ok=True)
        for slug,data in files.items(): (folder/(slug+'.json')).write_bytes(data)
        result['charts']={'base':'next/'+folder.name+'/'}
    (out/'overlays.json').write_bytes(compact(result)); return result


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--airfields'); ap.add_argument('--timeline')
    ap.add_argument('--plan'); ap.add_argument('--out',required=True); ap.add_argument('--margin',type=float,default=2)
    args=ap.parse_args(); print(json.dumps(build(args.airfields,args.timeline,args.plan,args.out,args.margin)))

if __name__ == '__main__': main()
