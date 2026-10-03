import {createCore,manifestSummary} from './core.js';
import {parseManifest,erasAt} from './manifest.js';
import {planCharts,planBasemap,planKeys} from './plan.js';
import {buildDemand} from './demand.js';
import {decodeRaster} from './decode.js';
import {AirspaceIndex} from './airspace.js';
// Pure planning remains on the main thread; only the compact raw manifest is cloned.
export async function createDataPlane(options={}) {
  const nav=globalThis.navigator;
  const mobile=!!nav && (/iP(ad|hone|od)|Android/.test(nav.userAgent??'') || (nav.platform==='MacIntel' && nav.maxTouchPoints>1));
  options={...options,mobile:options.mobile??mobile};
  const listeners=new Map(),ready=new Set(),absent=new Set(),airspace=new AirspaceIndex({});
  let workerStats={inflight:0,queued:0,bytesCached:0};
  let threadFailure=null,metadataLoaded=false,metadataError=null;
  let backend,thread,pending=new Map(),sequence=0,destroyed=false,generation=0,wanted=new Set(),paths=new Set(),decoding=new Set();
  const emit=(event,payload)=>{for(const fn of listeners.get(event)??[])fn(payload);};
  const receive=async(event,payload)=>{
    if(destroyed) {payload.bitmap?.close?.();return;}
    if(event==='stats'){workerStats=payload;return;}
    if(event==='metadata') {metadataLoaded=true;metadataError=null;airspace.setMetadata(payload.metadata);emit('metadata',{});return;}
    if(event==='encoded') {
      if(decoding.has(payload.key))return;decoding.add(payload.key);
      const gen=generation;
      try {
        const result=await decodeRaster(payload.bytes,{contentType:payload.contentType});
        if(destroyed || (gen!==generation&&!wanted.has(payload.key))) {result.bitmap?.close?.();return;}
        if(!result.bitmap)throw new Error('Image decoder unavailable on main thread');
        ready.add(payload.key);void call('markDelivered',payload.key).catch(reportError);emit('tile',{key:payload.key,bitmap:result.bitmap});
      }catch(error){void call('markEvicted',payload.key).catch(reportError);emit('error',{key:payload.key,error});}
      finally{decoding.delete(payload.key);}
      return;
    }
    if(event==='tile')ready.add(payload.key);
    if(event==='absent')absent.add(payload.key);
    if(event==='error' && !(payload.error instanceof Error))payload.error=Object.assign(new Error(payload.error.message),{name:payload.error.name});
    if(event==='error'&&payload.key==='airspace/metadata')metadataError=payload.error.message;
    emit(event,payload);
  };
  const send=(method,args,transfer=[])=>{
    if(destroyed)return Promise.reject(new Error('Data plane destroyed'));
    if(threadFailure&&!backend)return Promise.reject(threadFailure);
    if(backend)return Promise.resolve(backend[method](...args));
    const id=sequence++;return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});thread.postMessage({id,method,args},transfer);});
  };
  const call=(method,...args)=>send(method,args);
  let manifest;
  if(options.worker!==false && !options.fetch && typeof Worker!=='undefined') {
    try {
      thread=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
      thread.onmessage=({data})=>{if(data.event){void receive(data.event,data.payload);return;}const p=pending.get(data.id);if(p){pending.delete(data.id);data.error?p.reject(Object.assign(new Error(data.error.message),{name:data.error.name})):p.resolve(data.result);}};
      thread.onerror=error=>{threadFailure=new Error(error.message??'Worker failed');for(const p of pending.values())p.reject(threadFailure);pending.clear();emit('error',{key:'worker',error:new Error(error.message??'Worker failed')});};
      // Promises cannot cross postMessage; earlyFetches is adopted below instead.
      const {fetch:_,worker:__,earlyFetches:___,...serializable}=options;
      manifest=parseManifest(await call('init',serializable));
    } catch(error) {thread?.terminate();thread=null;threadFailure=null;pending.clear();backend=await createCore(options,(...args)=>void receive(...args));manifest=backend.m;}
  } else {backend=await createCore(options,(...args)=>void receive(...args));manifest=backend.m;}
  const reportError=error=>emit('error',{key:'demand',error});
  // Tile responses the boot script already requested are handed to the
  // scheduler, which consumes them instead of fetching the same URL twice. On
  // the Worker path the bytes are read here and transferred; the Worker holds a
  // placeholder for each key until they arrive.
  const early=options.earlyFetches instanceof Map ? options.earlyFetches : null;
  if(early?.size) {
    const base=manifest.raw.tileBase,entries=[];
    for(const [url,promise] of early) if(typeof url==='string' && url.startsWith(base)) {entries.push([url.slice(base.length),promise]);early.delete(url);}
    if(backend) backend.adopt(entries);
    else if(entries.length) {
      void call('adopt',entries.map(([key])=>[key])).catch(reportError);
      for(const [key,promise] of entries) Promise.resolve(promise).then(async response=>{
        const bytes=response.status===200 ? await response.arrayBuffer() : null;
        return send('prime',[{key,status:response.status,contentType:response.headers.get('content-type'),bytes}],bytes?[bytes]:[]);
      },error=>call('prime',{key,error:{message:String(error?.message??error)}})).catch(reportError);
    }
  }
  return {
    manifest:manifestSummary(manifest),
    planCharts(date,tiles,{solo}={}) {return {id:String(date)+(solo?`:${solo.paths.join(',')}:${solo.clip?.id??''}`:''),tiles:planCharts(manifest,date,tiles,solo)};},
    planBasemap(tiles) {return planBasemap(manifest.raw.basemap,tiles);},
    setDemand(state) {generation++;const demand=buildDemand(manifest,state,paths);paths=demand.paths;wanted=new Set(demand.requests.map(r=>r.key));void call('setDemand',state).catch(reportError);},
    on(event,fn) {if(!listeners.has(event))listeners.set(event,new Set());listeners.get(event).add(fn);},
    off(event,fn) {listeners.get(event)?.delete(fn);},
    markEvicted(key) {ready.delete(key);void call('markEvicted',key).catch(reportError);},
    readiness(date,tiles) {const items=planCharts(manifest,date,tiles).flatMap(t=>t.items);if(!items.length)return 1;let n=0;for(const {key} of items)if(ready.has(key)||absent.has(key))n++;return n/items.length;},
    loadAirfields:()=>call('loadAirfields'),airfieldDetails:i=>call('airfieldDetails',i),airfieldDetailsAll:()=>call('airfieldDetailsAll'),
    // A failed metadata load reports as unavailable even when the caller says the layer is enabled.
    airspaceRegionMask:d=>airspace.regionMask(d),airspaceStatus:(d,b,o={})=>airspace.status(d,b,{configured:!!manifest.raw.airspace,loaded:metadataLoaded,loadError:metadataError,...o,enabled:(o.enabled??true)&&!metadataError}),
    queryPin:(lng,lat,date)=>call('queryPin',lng,lat,date),queryAirspace:(lng,lat,d)=>call('queryAirspace',lng,lat,d),
    airspaceStack:(lng,lat,d)=>call('airspaceStack',lng,lat,d),airspaceCredits:()=>airspace.credits(),
    // Synchronous manifest lookups for the interface: how many era archives are in effect on a date, and where one lives.
    eraCountAt:date=>erasAt(manifest,date).length,
    eraSource(key) {const i=manifest.keyIndex.get(key);return i==null?null:{path:manifest.paths[i],zoom:[manifest.minZoom[i],manifest.maxZoom[i]]};},
    // Worker sends stats snapshots after demand/result. stats() is always synchronous.
    stats:()=>backend?backend.stats():({...workerStats}),
    destroy() {if(destroyed)return;if(backend)backend.destroy();else thread?.terminate();destroyed=true;for(const p of pending.values())p.reject(new Error('Data plane destroyed'));pending.clear();listeners.clear();ready.clear();},
  };
}
