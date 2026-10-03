#!/usr/bin/env python3
"""Render the local Protomaps cutout as 512px WebP PMTiles using Chromium.

No production access. Assets and vector source must be local. --estimate uses
basemap_build.REGION_BOXES without opening a source. --bbox limits a proof run.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from basemap_build import REGION_BOXES
from next_pmtiles import write_archive,zxy_to_id


def tile_boxes(z,bbox=None):
    boxes=[bbox] if bbox else [[-180,-85.0511287798066,180,85.0511287798066]] if z<=6 else list(REGION_BOXES.values())
    size=2**z;result=[]
    for w,s,e,n in boxes:
        if w>e:
            result.extend(tile_boxes(z,[w,s,180,n]));result.extend(tile_boxes(z,[-180,s,e,n]));continue
        def yy(lat):return (1-math.asinh(math.tan(math.radians(max(-85.0511287798066,min(85.0511287798066,lat)))))/math.pi)/2*size
        result.append((max(0,math.floor((w+180)/360*size)),max(0,math.floor(yy(n))),
                       min(size-1,math.ceil((e+180)/360*size)-1),min(size-1,math.ceil(yy(s))-1)))
    return result


def coordinates(z,bbox=None):
    boxes=tile_boxes(z,bbox)
    for y in range(min(b[1] for b in boxes),max(b[3] for b in boxes)+1):
        intervals=sorted((b[0],b[2]) for b in boxes if b[1]<=y<=b[3]);last=-1
        for left,right in intervals:
            for x in range(max(left,last+1),right+1):yield z,x,y
            last=max(last,right)


def estimate(bbox=None,minzoom=0,maxzoom=13,bytes_per_tile=22000,seconds_per_tile=.15):
    counts={z:sum(1 for _ in coordinates(z,bbox)) for z in range(minzoom,maxzoom+1)}
    count=sum(counts.values())
    return {'counts':counts,'tiles':count,'estimated_gb':round(count*bytes_per_tile/1e9,2),
            'serial_hours':round(count*seconds_per_tile/3600,2),
            'assumptions':{'bytes_per_tile':bytes_per_tile,'seconds_per_tile':seconds_per_tile}}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source');ap.add_argument('--assets');ap.add_argument('--modules')
    ap.add_argument('--out');ap.add_argument('--bbox',nargs=4,type=float);ap.add_argument('--minzoom',type=int,default=0);ap.add_argument('--maxzoom',type=int,default=13)
    ap.add_argument('--estimate',action='store_true');ap.add_argument('--bytes-per-tile',type=int,default=22000);ap.add_argument('--seconds-per-tile',type=float,default=.15)
    args=ap.parse_args()
    if not 0<=args.minzoom<=args.maxzoom<=13:ap.error('zoom range must be 0..13')
    if args.bbox and not (-180<=args.bbox[0]<=180 and -180<=args.bbox[2]<=180 and -85.0512<=args.bbox[1]<args.bbox[3]<=85.0512):ap.error('invalid bbox')
    stats=estimate(args.bbox,args.minzoom,args.maxzoom,args.bytes_per_tile,args.seconds_per_tile);print(json.dumps(stats,indent=2),flush=True)
    if args.estimate:return
    if not args.source or not args.assets or not args.out:ap.error('need --source, --assets, --out')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);jobs=out/'render-jobs.ndjson'
    with jobs.open('w') as f:
        for z in range(args.minzoom,args.maxzoom+1):
            for coord in coordinates(z,args.bbox):f.write(json.dumps(coord)+'\n')
    command=['node',str(Path(__file__).with_suffix('.mjs')),'--source',str(Path(args.source).resolve()),
             '--assets',str(Path(args.assets).resolve()),'--out',str(out.resolve()),'--jobs',str(jobs.resolve())]
    if args.modules:command+=['--modules',str(Path(args.modules).resolve())]
    started=time.monotonic();subprocess.run(command,check=True)
    tiles=[]
    for p in (out/'tiles').rglob('*.webp'):
        rel=p.relative_to(out/'tiles');z,x,y=int(rel.parts[0]),int(rel.parts[1]),int(p.stem)
        tiles.append((zxy_to_id(z,x,y),p))
    # Stream payloads; only the filename/Hilbert index is held in memory.
    dst=out/'raster.pmtiles'
    write_archive(dst,((tid,p.read_bytes()) for tid,p in sorted(tiles)),
                  {'tile_type':4,'tile_compression':1},
                  {'name':'archive.aero dark raster','format':'webp','attribution':'© OpenStreetMap contributors · Protomaps','tileSize':512,'flavor':'dark','lang':'en'})
    print(json.dumps({'archive':str(dst),'bytes':dst.stat().st_size,'tiles':len(tiles),'seconds':round(time.monotonic()-started,3)}))

if __name__=='__main__':main()
