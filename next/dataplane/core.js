import {loadManifest,day} from './manifest.js';
import {planCharts,planBasemap,planKeys} from './plan.js';
import {Scheduler} from './scheduler.js';
import {buildDemand} from './demand.js';
import {decodeRaster} from './decode.js';
import {parseAirfields} from './airfields.js';
import {buildPinIndex,queryPins} from './pins.js';
import {AirspaceIndex,decodeAirspaceTile} from './airspace.js';
export function manifestSummary(m) {
  return {frames:m.frames,dateBounds:m.dateBounds,coverage:m.raw.coverage,eraCount:m.paths.length,
    hasAirspace:!!m.raw.airspace,hasAirfields:!!m.raw.airfields,hasPins:!!m.raw.pins};
}
export async function createCore(options,emit) {
  const fetcher=options.fetch??globalThis.fetch.bind(globalThis);
  const m=await loadManifest(options.manifestUrl,fetcher);
  return new DataCore(m,fetcher,options,emit);
}
export class DataCore {
  constructor(manifest,fetcher,options={},emit=()=>{}) {
    this.m=manifest;this.fetch=fetcher;this.emit=emit;this.options=options;this.delivered=new Set();this.decoding=new Map();this.paths=new Set();this.wanted=new Set();this.shards=new Map();this.jsonLoads=new Map();this.airspace=new AirspaceIndex({});this.dead=false;
    this.scheduler=new Scheduler({...options,fetch:fetcher,cacheBytes:options.cacheBytes??(options.mobile?12:32)*1024*1024,
      onStats:s=>this.emit('stats',s),onResult:(r,b)=>{this.accept(r,b);},onError:e=>this.emit('error',e)});
    // Decodes are bounded so a burst of cache hits after eviction cannot start hundreds of createImageBitmap calls at once.
    this.decodeQueue=[];this.decodesActive=0;this.decodeLimit=options.decodeConcurrency??4;this.earlyDeferred=new Map();
    // Airspace metadata loads lazily (first airspace demand or query) with backoff; it never blocks boot.
    this.metadataLoading=null;this.metadataFailures=0;this.metadataRetryAt=0;
  }
  // Early responses: [key, promise] pairs on this thread; bare [key] entries wait for prime() from the facade.
  adopt(entries) {
    for(const [key,promise] of entries) {
      if(promise) {this.scheduler.adopt(key,Promise.resolve(promise));continue;}
      const d={};d.promise=new Promise((resolve,reject)=>{d.resolve=resolve;d.reject=reject;});this.earlyDeferred.set(key,d);this.scheduler.adopt(key,d.promise);
    }
  }
  prime({key,status,contentType,bytes,error}) {
    const d=this.earlyDeferred.get(key);if(!d)return;this.earlyDeferred.delete(key);
    if(error) d.reject(new Error(error.message??'early fetch failed'));
    else d.resolve(new Response(status===204||bytes==null?null:bytes,{status,headers:contentType?{'content-type':contentType}:{}}));
  }
  planCharts(date,tiles,{solo}={}) { return {id:String(date)+(solo?`:${solo.paths.join(',')}:${solo.clip?.id??''}`:''),tiles:planCharts(this.m,date,tiles,solo)}; }
  planBasemap(tiles) { return planBasemap(this.m.raw.basemap,tiles); }
  setDemand(state) {
    const {requests,paths}=buildDemand(this.m,state,this.paths);this.paths=paths;this.wanted=new Set(requests.map(r=>r.key));const airspaceTiles=state.airspaceTiles??[];
    // Decoded vector geometry is bounded by current spatial/temporal demand.
    for(const key of this.airspace.tiles?.keys()??[]) if(!this.wanted.has(key)) {const coord=this.airspace.tiles.get(key).tile;this.airspace.deleteTile(key);this.delivered.delete(key);this.emit('airspace',{tileId:`${coord.z}/${coord.x}/${coord.y}`,batch:null});}
    for(const key of this.decoding.keys())if(!this.wanted.has(key))this.decoding.delete(key);
    this.scheduler.setDemand(requests.filter(r=>!this.delivered.has(r.key)));
    if(this.m.raw.airspace && airspaceTiles.length && !this.airspaceLoaded) this.ensureAirspaceMetadata();
  }
  accept(request,bytes) {
    const {key}=request;
    if(this.dead || !this.wanted.has(key) || this.delivered.has(key) || this.decoding.has(key)) return;
    if(bytes==null) {this.delivered.add(key);this.emit('absent',{key});return;}
    this.decodeQueue.push({request,bytes});this.pumpDecodes();
  }
  pumpDecodes() {
    while(this.decodesActive<this.decodeLimit && this.decodeQueue.length) {
      const {request,bytes}=this.decodeQueue.shift(),{key}=request;
      if(this.dead || !this.wanted.has(key) || this.delivered.has(key) || this.decoding.has(key)) continue;
      this.decodesActive++;void this.decode(request,bytes).finally(()=>{this.decodesActive--;this.pumpDecodes();});
    }
  }
  async decode(request,bytes) {
    const {key}=request,token={};this.decoding.set(key,token);
    try {
      if(request.kind==='airspace') {
        const decoded=decodeAirspaceTile(bytes,request.src);
        if(!this.wanted.has(key)||this.dead) return;
        this.airspace.setTile(key,request.src,decoded);this.delivered.add(key);
        this.emit('airspace',{tileId:`${request.src.z}/${request.src.x}/${request.src.y}`,batch:decoded.batch});
      } else {
        const result=await decodeRaster(bytes,{bitmap:this.options.decodeBitmap!==false,contentType:request.contentType});
        if(!this.wanted.has(key)||this.dead||this.decoding.get(key)!==token) {result.bitmap?.close?.();return;}
        // Encoded fallback becomes ready only after the facade acknowledges its successful decode.
        if(result.bitmap) this.delivered.add(key);else token.encoded=true;
        this.emit(result.bitmap?'tile':'encoded',{key,...result});
      }
    } catch(error) { if(!this.dead&&this.wanted.has(key)) this.emit('error',{key,error}); }
    finally {if(this.decoding.get(key)===token&&!token.encoded)this.decoding.delete(key);}
  }
  markDelivered(key) { if(this.wanted.has(key)) this.delivered.add(key);this.decoding.delete(key); }
  markEvicted(key) {this.delivered.delete(key);this.decoding.delete(key);}
  readiness(date,tiles) {const items=planCharts(this.m,date,tiles).flatMap(t=>t.items);if(!items.length)return 1;let n=0;for(const {key} of items)if(this.delivered.has(key)||this.scheduler.absent.has(key))n++;return n/items.length;}
  async json(url) {
    if(!this.jsonLoads.has(url)) this.jsonLoads.set(url,(async()=>{const r=await this.fetch(url);if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json();})().catch(e=>{this.jsonLoads.delete(url);throw e;}));
    return this.jsonLoads.get(url);
  }
  async loadAirspaceMetadata() {
    const as=this.m.raw.airspace;if(!as||this.airspaceLoaded)return;
    const metadata=await this.json(new URL(`${as.p}/metadata`,this.m.raw.tileBase).href);
    if(!this.dead) {this.airspace.setMetadata(metadata);this.airspaceLoaded=true;this.metadataFailures=0;this.emit('metadata',{metadata});}
  }
  ensureAirspaceMetadata() {
    if(this.metadataLoading || this.airspaceLoaded || this.scheduler.clock.now()<this.metadataRetryAt) return this.metadataLoading;
    this.metadataLoading=this.loadAirspaceMetadata().catch(error=>{
      this.metadataRetryAt=this.scheduler.clock.now()+Math.min(60000,1000*2**this.metadataFailures++);
      if(!this.dead) this.emit('error',{key:'airspace/metadata',error});
    }).finally(()=>{this.metadataLoading=null;});
    return this.metadataLoading;
  }
  async loadAirfields() {
    const af=this.m.raw.airfields;if(!af)return null;
    if(!this.airfieldsPromise)this.airfieldsPromise=(async()=>{const r=await this.fetch(new URL(af.bin,this.m.raw.fileBase));if(!r.ok)throw new Error(`Airfields HTTP ${r.status}`);return parseAirfields(await r.arrayBuffer());})().catch(e=>{this.airfieldsPromise=null;throw e;});
    return this.airfieldsPromise;
  }
  async airfieldDetails(index) { const af=this.m.raw.airfields;if(!af)return null;const data=await this.json(new URL(af.details,this.m.raw.fileBase).href);const arrays=await this.loadAirfields();if(!Array.isArray(data)||data.length!==arrays.mx.length)throw new Error('Airfields details count mismatch');return data[index]??null; }
  airspaceRegionMask(d) {return this.airspace.regionMask(d);}
  airspaceStatus(d,bounds,options) {return this.airspace.status(d,bounds,options);}
  async queryAirspace(lng,lat,d) {await this.loadAirspaceMetadata();return this.airspace.query(lng,lat,d);}
  async queryPin(lng,lat,date) {
    const p=this.m.raw.pins;if(!p)return [];
    const n=2**p.z,lon=((lng+180)%360+360)%360-180,clat=Math.max(-85.05112878,Math.min(85.05112878,lat));
    const x=Math.min(n-1,Math.floor((lon+180)/360*n)),rad=clat*Math.PI/180;
    const y=Math.max(0,Math.min(n-1,Math.floor((1-Math.asinh(Math.tan(rad))/Math.PI)/2*n))),i=y*n+x;
    if(!p.shards.includes(i))return [];
    if(!this.shards.has(i)) {const data=await this.json(new URL(`${p.base}${i}.json`,this.m.raw.fileBase).href);this.shards.set(i,buildPinIndex(data,this.m.keys));}
    return queryPins(this.shards.get(i),lat,lng,date);
  }
  stats() {return this.scheduler.stats();}
  destroy() {this.dead=true;this.wanted.clear();this.scheduler.destroy();this.delivered.clear();this.jsonLoads.clear();this.shards.clear();this.decoding.clear();this.decodeQueue.length=0;this.earlyDeferred.clear();this.airspace.clear();}
}
