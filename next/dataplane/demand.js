import { planCharts, planBasemap, basemapCover, isVector, ancestor, tileKey } from './plan.js';
import { day, upperBound } from './manifest.js';
export const pathOf = key => key.split('/').slice(0,-3).join('/');
const distance = (t,c={x:.5,y:.5}) => { const n=2**t.z; let dx=Math.abs((t.x+.5)/n-c.x); dx=Math.min(dx,Math.abs(1-dx)); return dx*dx+((t.y+.5)/n-c.y)**2; };
export function buildDemand(manifest,state,previousPaths=new Set(),{occ=null,absent=null,delivered=null}={}) {
  const {date,chartTiles=[],basemapTiles=[],airspaceTiles=[],center,scrub={},solo}=state,cullBy=occ&&{occ,absent,occlude:state.occlude!==false};
  const requests=[],current=planCharts(manifest,date,chartTiles,solo,cullBy),paths=new Set();
  const add = (it,priority,kind='raster') => requests.push({key:it.key,url:new URL(it.key,manifest.raw.tileBase).href,priority,distance:distance(it.dst,center),kind,src:it.src});
  const minz = path => solo ? solo.zoom[0] : manifest.minZoom[manifest.pathIndex.get(path)];
  for(const tile of current) for(const it of tile.items) {
    const path=pathOf(it.key); paths.add(path);
    { const src=ancestor(it.src,Math.max(minz(path),it.dst.z-3)>it.src.z ? it.src.z : Math.max(minz(path),it.dst.z-3)); add({...it,src,key:tileKey(path,src)},previousPaths.has(path)?1:0); }
    add(it,1);
  }
  const baseKind=isVector(manifest.raw.basemap)?'basemap':'raster';
  for(const tile of planBasemap(manifest.raw.basemap,basemapTiles,absent,solo?null:basemapCover(manifest,date,basemapTiles,cullBy),delivered)) for(const it of tile.items) add(it,1,baseKind);
  const as=manifest.raw.airspace;
  if(as) for(const dst of airspaceTiles) if(dst.z>=as.z[0]) { const src=ancestor(dst,Math.min(dst.z,as.z[1])); add({key:tileKey(as.p,src),src,dst},1,'airspace'); }
  if(!solo) {
    // Last frame on or before the date; dates may be ISO strings or integer Days.
    const frames=manifest.frames, idx=Math.max(0, upperBound(manifest.frameDays, day(date))-1);
    const future=new Map();
    const want=(i,priority,low=false)=>{if(i>=0&&i<frames.length){const old=future.get(i);if(!old || priority<old.priority || (!low&&old.low&&priority===old.priority)) future.set(i,{priority,low});}};
    if(scrub.playing) for(let n=1;n<=3;n++) want(idx+n,2);
    else if(scrub.direction) for(let n=1;n<=Math.min(8,2+Math.ceil(Math.abs(scrub.velocity??0)/3));n++) want(idx+n*scrub.direction,2);
    else { want(idx-1,3); want(idx+1,3); for(let n=2;n<=3;n++) {want(idx-n,4,true);want(idx+n,4,true);} }
    for(const [i,{priority,low}] of future) for(const tile of planCharts(manifest,frames[i],chartTiles,null,cullBy)) for(const it of tile.items) {
      if(low) {const path=pathOf(it.key),src=ancestor(it.src,Math.min(it.src.z,Math.max(minz(path),it.dst.z-3))); add({...it,src,key:tileKey(path,src)},priority);} else add(it,priority);
    }
  }
  return {requests,paths};
}
