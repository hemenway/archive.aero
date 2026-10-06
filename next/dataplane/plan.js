import { erasAt } from './manifest.js';
export function tileLonLatBounds(z,x,y) {
  const n = 2 ** z, lat = ty => Math.atan(Math.sinh(Math.PI*(1-2*ty/n)))*180/Math.PI;
  return [x/n*360-180,lat(y+1),(x+1)/n*360-180,lat(y)];
}
export const boundsIntersect = (a,b) => a[0]<b[2] && a[2]>b[0] && a[1]<b[3] && a[3]>b[1];
export function ancestor(dst,z) { if(dst.z===z)return dst;const scale = 2 ** (dst.z-z); return {z,x:Math.floor(dst.x/scale),y:Math.floor(dst.y/scale)}; }
export const tileKey = (path, t) => `${path}/${t.z}/${t.x}/${t.y}`;
export function covered(c,t) {
  if (c == null) return true;
  if (t.z >= 6) { const p = ancestor(t,6); return c.has(p.y*64+p.x); }
  const s = 2**(6-t.z);
  for(let y=t.y*s;y<(t.y+1)*s;y++) for(let x=t.x*s;x<(t.x+1)*s;x++) if(c.has(y*64+x)) return true;
  return false;
}
// Occupancy: what a decoded chart tile says about its own area, on an 8x8 grid of 32 px cells. Bytes 0-7 hold one row
// each of "any pixel drawn", bytes 8-15 of "every pixel opaque". A tile up to three levels below reads its answer from
// the cells it covers, so the low-resolution ancestor every archive loads first settles most of its descendants.
export const EMPTY = new Uint8Array(16);
export function occupancyOf(alpha,size=256) {
  const out=new Uint8Array(16),cell=size/8;
  for(let cy=0;cy<8;cy++) for(let cx=0;cx<8;cx++) {
    let any=false,full=true;
    for(let y=cy*cell;y<(cy+1)*cell;y++) { const row=(y*size)<<2; for(let x=cx*cell;x<(cx+1)*cell;x++) { const a=alpha[row+(x<<2)+3]; if(a) any=true; if(a!==255) full=false; } }
    if(any) out[cy]|=1<<cx; if(full) out[8+cy]|=1<<cx;
  }
  return out;
}
// {any, full} for one source tile of an archive, from the nearest tile at or above it that has been seen; null if none.
export function occupancy(occ,path,src) {
  for(let dz=0;dz<=3&&dz<=src.z;dz++) {
    const g=occ.get(`${path}/${src.z-dz}/${src.x>>dz}/${src.y>>dz}`); if(!g) continue;
    const n=8>>dz,x0=(src.x&((1<<dz)-1))*n,y0=(src.y&((1<<dz)-1))*n,mask=((1<<n)-1)<<x0; let any=false,full=true;
    for(let y=y0;y<y0+n;y++) { if(g[y]&mask) any=true; if((g[8+y]&mask)!==mask) full=false; }
    return {any,full};
  }
  return null;
}
// Items are ordered bottom to top. Drop the ones known to draw nothing here (blank, or missing from the archive) and,
// when charts are opaque, everything under an item known to cover the whole tile.
export function cull(items,cullBy) {
  if(!cullBy||!items.length) return items;
  const {occ,occlude,absent}=cullBy,kept=[];
  for(let i=items.length-1;i>=0;i--) {
    const it=items[i];
    // A tile the archive turned out not to have draws nothing and hides nothing, whatever its ancestor suggested.
    if(absent?.has(it.key)) continue;
    const o=occupancy(occ,it.key.slice(0,it.key.length-`/${it.src.z}/${it.src.x}/${it.src.y}`.length),it.src);
    if(o&&!o.any) continue;
    kept.push(it); if(occlude&&o?.full) break;
  }
  return kept.length===items.length?items:kept.reverse();
}
// Chart tiles ("z/x/y") nothing beneath can show through, each with why: "solid" when one of the charts there is
// known to be solid (at a seam the chart on top is cut off and the one under it fills the tile), "unknown" when one
// is not yet known either way, so what is under it waits for the answer.
export function coveredCells(chartPlan,cullBy) {
  const out=new Map(); if(!cullBy?.occlude) return out;
  for(const {dst,items} of chartPlan) {
    let solid=false,unknown=false;
    for(const it of items) {
      if(it.clip) continue;
      const o=occupancy(cullBy.occ,it.key.slice(0,it.key.length-`/${it.src.z}/${it.src.x}/${it.src.y}`.length),it.src);
      if(!o) unknown=true; else if(o.full) { solid=true; break; }
    }
    if(solid) out.set(`${dst.z}/${dst.x}/${dst.y}`,'solid'); else if(unknown) out.set(`${dst.z}/${dst.x}/${dst.y}`,'unknown');
  }
  return out;
}
// Whether a basemap item, or an ancestor the renderer would draw in its place, has already been delivered.
export function residentAt(resident,source,it) {
  if(!resident?.size) return false;
  for(let z=it.src.z;z>=source.z[0];z--) if(resident.has(tileKey(source.p,ancestor(it.src,z)))) return true;
  return false;
}
function item(path,dst,z,era,clip=null) {
  const src = ancestor(dst,z);
  const key=tileKey(path,src);return clip ? {key,src,dst,clip} : {key,src,dst};
}
export function planCharts(manifest,date,tiles,solo=null,cullBy=null) {
  const active = solo ? null : erasAt(manifest,date), plan=[];
  for(const dst of tiles) {
    const b = tileLonLatBounds(dst.z,dst.x,dst.y), items=[],sources=[],suffixes=[];
    if(solo) {
      if(dst.z >= solo.zoom[0]) for(const path of solo.paths) items.push(item(path,dst,Math.min(dst.z,solo.zoom[1]),-1,solo.clip?.id));
    } else for(const i of active) {
      if(dst.z < manifest.minZoom[i] || (manifest.bounds[i] && !boundsIntersect(manifest.bounds[i],b)) || !covered(manifest.coverage[i],dst)) continue;
      const z=Math.min(dst.z,manifest.maxZoom[i]);
      const src=sources[z]??(sources[z]=ancestor(dst,z));
      const suffix=suffixes[z]??(suffixes[z]=`/${src.z}/${src.x}/${src.y}`);
      items.push({key:manifest.paths[i]+suffix,src,dst});
    }
    plan.push({dst,items:solo?items:cull(items,cullBy)});
  }
  return plan;
}
// The basemap cutout holds the whole world only at low zooms; where a tile is known to be missing, its nearest
// existing ancestor stands in so the map never falls back to nothing.
// resident: keys already delivered (the page's or the core's). A tile wholly under charts of unknown occupancy is
// not fetched until they answer, but one already on screen stays in the plan: scrubbing dates made every basemap
// tile under the next date's still-loading charts vanish and return (2026-10-05).
export function planBasemap(source,tiles,absent=null,covered=null,resident=null) {
  if(!source) return tiles.map(dst=>({dst,items:[]}));
  const under=(z,x,y)=>covered.get(`${z}/${x}/${y}`);
  return tiles.map(dst=>{
    if(dst.z < source.z[0]) return {dst,items:[]};
    let it=item(source.p,dst,Math.min(dst.z,source.z[1]),0);
    while(absent?.has(it.key) && it.src.z>source.z[0]) it=item(source.p,dst,it.src.z-1,0);
    if(absent?.has(it.key)) return {dst,items:[]};
    // Wholly under opaque charts: not drawn, so not fetched or painted either. While a chart over it has yet to show
    // whether it is solid, a tile not yet delivered waits: charts load first and most of a view never needs its basemap.
    if(covered?.size) {
      let solid=true,hidden=true;
      for(let i=0;i<4;i++) { const why=under(dst.z+1,dst.x*2+(i&1),dst.y*2+(i>>1)); if(!why) { hidden=solid=false; break; } if(why!=='solid') solid=false; }
      if(solid || (hidden && !residentAt(resident,source,it))) return {dst,items:[]};
    }
    return {dst,items:[it]};
  });
}
export const isVector = source => source?.format==='mvt';
// The chart cells over a set of basemap tiles, planned for themselves: the basemap's buffer ring reaches past the
// chart tiles of a demand, and a tile out there is as hidden as one in view.
export function basemapCover(manifest,date,basemapTiles,cullBy) {
  if(!cullBy?.occlude||!manifest.raw.basemap) return null;
  const cells=[]; for(const t of basemapTiles) for(let i=0;i<4;i++) cells.push({z:t.z+1,x:t.x*2+(i&1),y:t.y*2+(i>>1)});
  return coveredCells(planCharts(manifest,date,cells,null,cullBy),cullBy);
}
export const planKeys = plan => new Set(plan.flatMap(t=>t.items.map(i=>i.key)));
