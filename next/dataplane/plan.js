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
function item(path,dst,z,era,clip=null) {
  const src = ancestor(dst,z);
  const key=tileKey(path,src);return clip ? {key,src,dst,clip} : {key,src,dst};
}
export function planCharts(manifest,date,tiles,solo=null) {
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
    plan.push({dst,items});
  }
  return plan;
}
export function planBasemap(source,tiles) {
  if(!source) return tiles.map(dst=>({dst,items:[]}));
  return tiles.map(dst=>({dst,items:dst.z < source.z[0] ? [] : [item(source.p,dst,Math.min(dst.z,source.z[1]),0)]}));
}
export const planKeys = plan => new Set(plan.flatMap(t=>t.items.map(i=>i.key)));
