import {loadManifest,parseManifest,day} from './manifest.js';
import {planCharts,planBasemap,planKeys,basemapCover,occupancyOf} from './plan.js';
import {Scheduler} from './scheduler.js';
import {buildDemand} from './demand.js';
import {decodeRaster,probeOccupancy} from './decode.js';
import {parseAirfields} from './airfields.js';
import {buildPinIndex,queryPins} from './pins.js';
import {AirspaceIndex,decodeAirspaceTile} from './airspace.js';
export {manifestSummary} from './manifest.js';
export async function createCore(options,emit) {
  const fetcher=options.fetch??globalThis.fetch.bind(globalThis);
  // The page fetches the manifest itself (its preload is already in flight) and hands the parsed JSON over.
  const m=options.manifest?parseManifest(options.manifest):await loadManifest(options.manifestUrl,fetcher);
  return new DataCore(m,fetcher,options,emit);
}
export class DataCore {
  constructor(manifest,fetcher,options={},emit=()=>{}) {
    this.m=manifest;this.fetch=fetcher;this.emit=emit;this.options=options;this.delivered=new Set();this.decoding=new Map();this.paths=new Set();this.wanted=new Set();this.shards=new Map();this.jsonLoads=new Map();this.airspace=new AirspaceIndex({});this.dead=false;
    // What decoded chart tiles say about blank and solid areas (plan.js); bounded, oldest first out.
    this.occ=new Map();this.state=null;this.basePaths=new Set();this.reculling=null;this.painter=null;
    // Tiles are small and one round trip each, so latency, not bandwidth, sets the pace: HTTP/2 and 3 multiplex these
    // over one connection. (An HTTP/1.1 browser still queues at its own six per host.)
    this.scheduler=new Scheduler({concurrency:options.mobile?16:24,...options,fetch:fetcher,cacheBytes:options.cacheBytes??(options.mobile?12:32)*1024*1024,
      onStats:s=>this.emit('stats',s),onResult:(r,b)=>{this.accept(r,b);},onError:e=>this.emit('error',e)});
    // Decodes are bounded so a burst of cache hits after eviction cannot start hundreds of createImageBitmap calls at once.
    this.decodeQueue=[];this.paintQueue=[];this.painting=false;this.decodesActive=0;this.decodeLimit=options.decodeConcurrency??4;this.earlyDeferred=new Map();
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
  cullBy(occlude=this.state?.occlude) { return {occ:this.occ,absent:this.scheduler.absent,occlude:occlude!==false}; }
  planCharts(date,tiles,{solo}={}) { return {id:String(date)+(solo?`:${solo.paths.join(',')}:${solo.clip?.id??''}`:''),tiles:planCharts(this.m,date,tiles,solo,this.cullBy())}; }
  planBasemap(tiles) { const s=this.state,covered=s&&!s.solo?basemapCover(this.m,s.date,tiles,this.cullBy()):null;return planBasemap(this.m.raw.basemap,tiles,this.scheduler.absent,covered,this.delivered); }
  // A recull replays the last demand after new occupancy or a missing basemap tile; it keeps the path baseline of the
  // demand it replays, so archives new to that demand still load their low-resolution tile first.
  setDemand(state,recull=false) {
    if(!recull) this.basePaths=this.paths;
    this.state=state;
    const {requests,paths}=buildDemand(this.m,state,this.basePaths,{occ:this.occ,absent:this.scheduler.absent,delivered:this.delivered});this.paths=paths;this.wanted=new Set(requests.map(r=>r.key));const airspaceTiles=state.airspaceTiles??[];
    // Decoded vector geometry is bounded by current spatial/temporal demand.
    for(const key of this.airspace.tiles?.keys()??[]) if(!this.wanted.has(key)) {const coord=this.airspace.tiles.get(key).tile;this.airspace.deleteTile(key);this.delivered.delete(key);this.emit('airspace',{tileId:`${coord.z}/${coord.x}/${coord.y}`,batch:null});}
    for(const key of this.decoding.keys())if(!this.wanted.has(key))this.decoding.delete(key);
    this.painter?.then(p=>p.retain(this.wanted),()=>{});
    this.scheduler.setDemand(requests.filter(r=>!this.delivered.has(r.key)));
    if(this.m.raw.airspace && airspaceTiles.length && !this.airspaceLoaded) this.ensureAirspaceMetadata();
  }
  accept(request,bytes) {
    const {key}=request;
    if(this.dead || !this.wanted.has(key) || this.delivered.has(key) || this.decoding.has(key)) return;
    if(bytes==null) {this.delivered.add(key);if(request.kind!=='airspace')this.recullSoon();this.emit('absent',{key});return;}
    if(request.kind==='basemap') {this.paintQueue.push({request,bytes});this.pumpPaints();return;}
    this.decodeQueue.push({request,bytes});this.pumpDecodes();
  }
  // Only a decoded image teaches anything. A 204 is not taken as proof that the tiles below are missing too: an
  // archive with a hole in its overviews would then lose real charts.
  learn(key,grid) {
    this.occ.delete(key);this.occ.set(key,grid);
    if(this.occ.size>8192) this.occ.delete(this.occ.keys().next().value);
    this.recullSoon();
  }
  recullSoon() {
    if(this.reculling||this.dead) return;
    this.reculling=this.scheduler.clock.setTimeout(()=>{this.reculling=null;if(!this.dead&&this.state)this.setDemand(this.state,true);},0);
  }
  // The vector basemap painter (protomaps-leaflet's rules on a 2D canvas) loads with the first basemap tile.
  basemapPainter() {
    return this.painter??=import('./basemap.js').then(m=>m.createPainter({flavor:this.m.raw.basemap.flavor,lang:this.m.raw.basemap.lang,
      repaint:(key,bitmap)=>{if(this.dead||!this.wanted.has(key)||!this.delivered.has(key))bitmap.close?.();else this.emit('tile',{key,bitmap,replace:true});}}));
  }
  // Painting a vector basemap tile holds this thread for tens of milliseconds, so tiles are painted one at a time,
  // nearest the view centre first, with a turn of the event loop between them: chart tiles that arrive meanwhile are
  // decoded and delivered instead of waiting behind the basemap.
  pumpPaints() {
    if(this.painting||this.dead) return;
    const queue=this.paintQueue; let best=-1;
    for(let i=queue.length-1;i>=0;i--) {
      const {key}=queue[i].request;
      if(!this.wanted.has(key)||this.delivered.has(key)||this.decoding.has(key)) queue.splice(i,1),best>i&&best--;
      else if(best<0||(queue[i].request.distance??0)<(queue[best].request.distance??0)) best=i;
    }
    if(best<0) return;
    const [{request,bytes}]=queue.splice(best,1);this.painting=true;
    this.scheduler.clock.setTimeout(()=>{void this.decode(request,bytes).finally(()=>{this.painting=false;this.pumpPaints();});},0);
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
      } else if(request.kind==='basemap') {
        const painter=await this.basemapPainter(),bitmap=await painter.paint(key,request.src,bytes);
        if(!this.wanted.has(key)||this.dead||this.decoding.get(key)!==token) {bitmap.close?.();painter.forget(key);return;}
        this.delivered.add(key);this.emit('tile',{key,bitmap});
      } else {
        const result=await decodeRaster(bytes,{bitmap:this.options.decodeBitmap!==false,contentType:request.contentType});
        if(!this.wanted.has(key)||this.dead||this.decoding.get(key)!==token) {result.bitmap?.close?.();return;}
        // Encoded fallback becomes ready only after the facade acknowledges its successful decode.
        if(result.bitmap) this.delivered.add(key);else token.encoded=true;
        const grid=result.bitmap&&request.kind==='raster'&&this.options.probe!==false?probeOccupancy(result.bitmap,occupancyOf):null;
        if(grid) this.learn(key,grid);
        // The grid is copied for the page: a Worker transfers what it posts.
        this.emit(result.bitmap?'tile':'encoded',grid?{key,...result,occ:grid.slice()}:{key,...result});
      }
    } catch(error) { if(!this.dead&&this.wanted.has(key)) this.emit('error',{key,error}); }
    finally {if(this.decoding.get(key)===token&&!token.encoded)this.decoding.delete(key);}
  }
  markDelivered(key) { if(this.wanted.has(key)) this.delivered.add(key);this.decoding.delete(key); }
  markEvicted(key) {this.delivered.delete(key);this.decoding.delete(key);this.painter?.then(p=>p.forget(key),()=>{});}
  readiness(date,tiles) {const items=planCharts(this.m,date,tiles,null,this.cullBy()).flatMap(t=>t.items);if(!items.length)return 1;let n=0;for(const {key} of items)if(this.delivered.has(key)||this.scheduler.absent.has(key))n++;return n/items.length;}
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
  // The whole details file, index-aligned with the binary: the airfield browser lists every field in view by name.
  async airfieldDetailsAll() { const af=this.m.raw.airfields;if(!af)return null;const data=await this.json(new URL(af.details,this.m.raw.fileBase).href);const arrays=await this.loadAirfields();if(!Array.isArray(data)||data.length!==arrays.mx.length)throw new Error('Airfields details count mismatch');return data; }
  async airfieldDetails(index) { const data=await this.airfieldDetailsAll();return data?.[index]??null; }
  airspaceRegionMask(d) {return this.airspace.regionMask(d);}
  airspaceStatus(d,bounds,options) {return this.airspace.status(d,bounds,options);}
  async queryAirspace(lng,lat,d) {await this.loadAirspaceMetadata();return this.airspace.query(lng,lat,d);}
  // The pin card's stack. Metadata loads under the same backoff as a demand; without it there are no regions to report.
  async airspaceStack(lng,lat,d) {if(this.m.raw.airspace&&!this.airspaceLoaded)await this.ensureAirspaceMetadata();return this.airspace.stack(lng,lat,d);}
  // How many charts share a result's era archive (decides whether solo view clips to the chart's ring). A shard only
  // holds the locations near its cell, so its own count can miss an era's far members; the era's extent tells then.
  eraMembers(index,{location,chart}) {
    const n=index.eraMemberCount.get(chart.eraKey)||1;if(n>1)return n;
    const i=this.m.keyIndex.get(chart.eraKey),b=i==null?null:this.m.bounds[i],l=location.bbox;
    return b&&l&&(b[2]-b[0]>l[2]-l[0]+2||b[3]-b[1]>l[3]-l[1]+2)?2:n;
  }
  async queryPin(lng,lat,date) {
    const p=this.m.raw.pins;if(!p)return [];
    const n=2**p.z,lon=((lng+180)%360+360)%360-180,clat=Math.max(-85.05112878,Math.min(85.05112878,lat));
    const x=Math.min(n-1,Math.floor((lon+180)/360*n)),rad=clat*Math.PI/180;
    const y=Math.max(0,Math.min(n-1,Math.floor((1-Math.asinh(Math.tan(rad))/Math.PI)/2*n))),i=y*n+x;
    if(!p.shards.includes(i))return [];
    if(!this.shards.has(i)) {const data=await this.json(new URL(`${p.base}${i}.json`,this.m.raw.fileBase).href);this.shards.set(i,buildPinIndex(data,this.m.keys));}
    const shard=this.shards.get(i);
    return queryPins(shard,lat,lng,date).map(r=>({...r,members:this.eraMembers(shard,r)}));
  }
  stats() {return this.scheduler.stats();}
  destroy() {this.dead=true;this.wanted.clear();this.scheduler.destroy();this.delivered.clear();this.jsonLoads.clear();this.shards.clear();this.decoding.clear();this.decodeQueue.length=0;this.paintQueue.length=0;this.earlyDeferred.clear();this.airspace.clear();this.occ.clear();if(this.reculling!=null)this.scheduler.clock.clearTimeout(this.reculling);}
}
