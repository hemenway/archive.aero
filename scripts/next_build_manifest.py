#!/usr/bin/env python3
"""Build frozen C2 manifest from versioned local archives; never reads tile payloads.

Archive bytes have already been hashed by next_version_archives.py. --dir mirrors
bucket keys, e.g. DIR/sectionals/<era>.<hash>.pmtiles and DIR/basemap/... .
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from build_metadata_bundle import parse_key_dates, header_bounds
from next_pmtiles import Archive

ROOT=Path(__file__).resolve().parent.parent
PATTERN=re.compile(r'^(.*)\.([a-f0-9]{12})\.pmtiles$')


def validate(path,allow_over_budget=False):
    command=['node',str(ROOT/'next/contract/validate.mjs'),'--manifest',str(path),'--schema-only']
    if allow_over_budget: command.append('--allow-over-budget')
    subprocess.run(command,check=True)


def build(directory,coverage,overlays=None,tile_base='https://data.archive.aero/t/',
          file_base='https://data.archive.aero/', generated=None):
    result={'version':1,'generated':generated or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'tileBase':tile_base,'fileBase':file_base,'eras':[], 'basemap':None,'airspace':None,
            'airfields':None,'pins':None,'coverage':coverage}
    if overlays:
        for k in ('airfields','pins'): result[k]=overlays.get(k)
    for path in sorted(Path(directory).rglob('*.pmtiles')):
        rel=path.relative_to(directory).as_posix(); match=PATTERN.fullmatch(rel)
        if not match: raise ValueError('not content-versioned: '+rel)
        stem,h=match.groups(); a=Archive(path); z=[a.h['min_zoom'],a.h['max_zoom']]
        if stem.startswith('sectionals/chart/'): continue
        if stem.startswith('sectionals/'):
            k=stem[len('sectionals/'):]; dates=parse_key_dates(k)
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}',k):
                print('skipping start-only era '+k); continue
            if not dates: raise ValueError('invalid or start-only era '+k)
            if dates[0]>=dates[1]: raise ValueError('empty/inverted era interval')
            # Helper validity checks are shared with today's bundle; frozen C2 uses
            # nearest 4 decimals (rather than the old bundle's outward 2 decimals).
            b=header_bounds(a.h,k,[])
            if b is not None:
                b=[round(a.h[n]/1e7,4) for n in ('min_lon_e7','min_lat_e7','max_lon_e7','max_lat_e7')]
                if b[0]>=b[2] or b[1]>=b[3]: b=None
            result['eras'].append({'k':k,'h':h,'b':b,'z':z,'c':a.coverage(6)})
        elif stem.startswith(('basemap/','airspace/')):
            kind=stem.split('/')[0]
            if result[kind] is not None: raise ValueError('multiple '+kind+' archives; stage only the selected build')
            result[kind]={'p':stem+'.'+h,'z':z}
            if kind=='basemap': result[kind]['tileSize']=512
        else: raise ValueError('unknown archive namespace '+stem)
    result['eras'].sort(key=lambda e:(*parse_key_dates(e['k']),e['k']))
    if not result['eras']: raise ValueError('no era archives')
    return result


def emit(manifest,out,allow_over_budget=False):
    data=json.dumps(manifest,separators=(',',':'),ensure_ascii=False).encode()
    compressed=len(gzip.compress(data,mtime=0)); print(f'manifest: {len(data)} bytes; gzip: {compressed} bytes')
    if compressed>80000 and not allow_over_budget: raise ValueError('80 KB gzip budget exceeded; request a contract split or use --allow-over-budget')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    path=out/('manifest.'+hashlib.sha256(data).hexdigest()[:12]+'.json')
    path.write_bytes(data)
    try: validate(path,allow_over_budget)
    except Exception: path.unlink(); raise
    return path


def estimate(dates_csv):
    # Deterministic proxy, NOT a measurement of unavailable production directories.
    rng=random.Random(37061); eras=[]
    with open(dates_csv,newline='') as f:
        for row in csv.DictReader(f):
            k=row['date_iso']
            if not parse_key_dates(k): continue
            large=rng.random()<.16
            x,y=rng.randrange(7,22),rng.randrange(16,29)
            width,height=(15,10) if large else (rng.randrange(1,5),rng.randrange(1,4))
            c=sorted({yy*64+xx for yy in range(y,min(64,y+height)) for xx in range(x,min(64,x+width))})
            def latitude(row): return math.degrees(math.atan(math.sinh(math.pi*(1-2*row/64))))
            b=None if rng.random()<.08 else [round(x/64*360-180+rng.random()*.8,4),
                round(latitude(y+height)+rng.random()*.8,4),
                round((x+width)/64*360-180-rng.random()*.8,4),round(latitude(y)-rng.random()*.8,4)]
            eras.append({'k':k,'h':rng.randbytes(6).hex(),'b':b,'z':[4,11],'c':c})
    eras.sort(key=lambda e:e['k'])
    # Approximate coverage sweep with independently changing availability.
    coverage={'segments':[[e['k'][:10],e['k'][14:],rng.randrange(1,140),round(rng.random()*100,1)] for e in eras]}
    data=json.dumps({'version':1,'eras':eras,'coverage':coverage},separators=(',',':')).encode()
    size=len(gzip.compress(data,mtime=0))
    return {'eras':len(eras),'gzip_bytes':size,'over_budget':size>80000,
            'model':'16% continental 150-cell eras; remainder 1–12 cells; 8% null bounds; random SHA256 and 4-decimal bounds; one coverage segment per era'}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--dir'); ap.add_argument('--coverage'); ap.add_argument('--overlays')
    ap.add_argument('--out'); ap.add_argument('--allow-over-budget',action='store_true'); ap.add_argument('--estimate-from')
    ap.add_argument('--tile-base',default='https://data.archive.aero/t/'); ap.add_argument('--file-base',default='https://data.archive.aero/')
    args=ap.parse_args()
    if args.estimate_from: print(json.dumps(estimate(args.estimate_from),indent=2)); return
    if not args.dir or not args.coverage or not args.out: ap.error('need --dir, --coverage, --out')
    m=build(args.dir,json.loads(Path(args.coverage).read_text()),json.loads(Path(args.overlays).read_text()) if args.overlays else None,args.tile_base,args.file_base)
    print(emit(m,args.out,args.allow_over_budget))

if __name__=='__main__': main()
