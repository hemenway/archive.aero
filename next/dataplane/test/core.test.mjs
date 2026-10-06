import test from 'node:test';
import assert from 'node:assert/strict';
import {createDataPlane} from '../index.js';
import {DataCore} from '../core.js';
import {parseManifest} from '../manifest.js';
const era={k:'1950-01-01_to_1960-01-01',h:'0123456789ab',z:[4,8],b:null,c:null};
const manifest={version:1,tileBase:'http://fixture/t/',fileBase:'http://fixture/',eras:[era],basemap:null,airspace:null,airfields:null,pins:{z:5,margin:2,base:'next/pins.0123456789ab/',shards:[391]},coverage:{segments:[]}};
const wait=async f=>{for(let i=0;i<30&&!f();i++)await new Promise(r=>setImmediate(r));assert.ok(f());};
const settle=async()=>{for(let i=0;i<8;i++)await new Promise(r=>setImmediate(r));};
const withBitmaps=(t,value=async()=>({close(){}}))=>{const d=Object.getOwnPropertyDescriptor(globalThis,'createImageBitmap');Object.defineProperty(globalThis,'createImageBitmap',{configurable:true,writable:true,value});t.after(()=>{if(d)Object.defineProperty(globalThis,'createImageBitmap',d);else delete globalThis.createImageBitmap;});};
test('main-thread facade, absent readiness and eviction are deterministic',async()=>{
 const calls=[];const dp=await createDataPlane({worker:false,manifestUrl:'http://fixture/manifest',fetch:async url=>{calls.push(String(url));return String(url).endsWith('manifest')?Response.json(manifest):new Response(null,{status:204});}});
 const tiles=[{z:10,x:238,y:410},{z:10,x:239,y:410}],date='1951-01-01';assert.deepEqual(dp.manifest.frames,['1950-01-01']);assert.equal(dp.readiness(date,tiles),0);let count=0;dp.on('absent',()=>count++);
 const key=dp.planCharts(date,tiles).tiles[0].items[0].key;
 dp.setDemand({date,chartTiles:tiles,scrub:{direction:0}});await wait(()=>dp.readiness(date,tiles)===1);assert.equal(count,2); // shared full-res source plus bootstrap
 // A tile the archive does not have leaves the plan.
 assert.deepEqual(dp.planCharts(date,tiles).tiles.map(t=>t.items.length),[0,0]);
 const n=calls.length;dp.markEvicted(key);assert.equal(dp.readiness(date,tiles),1);dp.setDemand({date,chartTiles:tiles});await wait(()=>count>=2);assert.equal(calls.length,n);dp.destroy();
});
test('queryPin loads only selected shard and failed JSON retries later',async()=>{
 let shardCalls=0;const inventory={locations:{Dallas:{ref:'d',charts:[{d:'1950-01-01',e:'1960-01-01'}]}},rings:{d:[[[-98,31],[-95,31],[-95,34],[-98,34]]]}};
 const dp=await createDataPlane({worker:false,manifestUrl:'http://fixture/manifest',fetch:async url=>{if(String(url).endsWith('manifest'))return Response.json(manifest);shardCalls++;return shardCalls===1?new Response('',{status:503}):Response.json(inventory);}});
 assert.deepEqual(await dp.queryPin(0,0,'1951-01-01'),[]);assert.equal(shardCalls,0);await assert.rejects(dp.queryPin(-96.7,32.7,'1951-01-01'));const rows=await dp.queryPin(-96.7,32.7,'1951-01-01');assert.equal(rows[0].location.name,'Dallas');assert.equal(rows[0].chart.published,true);await dp.queryPin(-96.7,32.7,'1951-01-01');assert.equal(shardCalls,2);dp.destroy();
});
test('a terminal Worker failure rejects future RPC instead of wedging promises',async t=>{
 const previous=globalThis.Worker;let instance;
 class MockWorker {constructor(){instance=this;}postMessage({id,method}){if(method==='init')queueMicrotask(()=>this.onmessage({data:{id,result:manifest}}));}terminate(){}}
 // The page fetches the manifest itself and hands it to the Worker.
 const realFetch=globalThis.fetch;globalThis.fetch=async()=>Response.json(manifest);
 globalThis.Worker=MockWorker;t.after(()=>{globalThis.fetch=realFetch;if(previous===undefined)delete globalThis.Worker;else globalThis.Worker=previous;});
 const dp=await createDataPlane({manifestUrl:'http://fixture/manifest'});let error;dp.on('error',e=>{error=e;});instance.onerror({message:'Worker crash'});assert.equal(error.error.message,'Worker crash');await assert.rejects(dp.loadAirfields(),/Worker crash/);await assert.rejects(dp.queryPin(-96.7,32.7,'1951-01-01'),/Worker crash/);dp.destroy();
});
test('superseded airspace tiles release retained rings and tell renderer to remove batches',async()=>{
 const raw={...manifest,airspace:{p:'airspace/mock.0123456789ab',z:[0,11]}};
 const dp=await createDataPlane({worker:false,manifestUrl:'http://fixture/manifest',fetch:async url=>String(url).endsWith('manifest')?Response.json(raw):String(url).endsWith('metadata')?Response.json({archive_aero:{regions:{}}}):new Response(new Uint8Array(0))});
 const events=[];dp.on('airspace',e=>events.push(e));dp.setDemand({date:'1951-01-01',chartTiles:[],airspaceTiles:[{z:10,x:238,y:410}]});await wait(()=>events.length===1);dp.setDemand({date:'1951-01-01',chartTiles:[],airspaceTiles:[]});assert.equal(events[1].tileId,'10/238/410');assert.equal(events[1].batch,null);dp.destroy();
});
test('early responses are adopted, never fetched twice, and relative manifest bases resolve against the manifest URL',async t=>{
 withBitmaps(t);const calls=[],key='sectionals/1950-01-01_to_1960-01-01.0123456789ab/8/59/102',url='http://fixture/t/'+key;
 const early=new Map([[url,Promise.resolve(new Response(new Uint8Array([1,2,3]),{headers:{'content-type':'image/png'}}))],['http://elsewhere/x',Promise.resolve(new Response(''))]]);
 const dp=await createDataPlane({worker:false,earlyFetches:early,manifestUrl:'http://fixture/manifest',fetch:async u=>{calls.push(String(u));return String(u).endsWith('manifest')?Response.json({...manifest,tileBase:'./t/',fileBase:'./'}):new Response(null,{status:204});}});
 assert.equal(early.size,1);const tiles=[];dp.on('tile',e=>tiles.push(e.key));
 dp.setDemand({date:'1951-01-01',chartTiles:[{z:10,x:238,y:410}],scrub:{direction:0}});await wait(()=>tiles.length===1);
 assert.equal(tiles[0],key);assert.ok(!calls.includes(url));assert.ok(calls.some(u=>u.endsWith('/7/29/51')));dp.destroy();
});
test('Worker-side placeholders wait for primed bytes instead of fetching',async()=>{
 const calls=[],events=[],key='sectionals/1950-01-01_to_1960-01-01.0123456789ab/8/59/102';
 const core=new DataCore(parseManifest(manifest),async u=>{calls.push(String(u));return new Response(null,{status:204});},{decodeBitmap:false},(e,p)=>events.push([e,p]));
 core.adopt([[key]]);core.setDemand({date:'1951-01-01',chartTiles:[{z:10,x:238,y:410}],scrub:{direction:0}});await settle();
 assert.ok(!calls.some(u=>u.endsWith('/8/59/102')));assert.ok(calls.some(u=>u.endsWith('/7/29/51')));
 core.prime({key,status:200,contentType:'image/png',bytes:new Uint8Array([1,2,3]).buffer});await wait(()=>events.some(([e,p])=>e==='encoded'&&p.key===key));core.destroy();
});
test('airspace metadata loads lazily with backoff, never blocks boot, and a load error overrides enabled in status',async()=>{
 let metaCalls=0,now=0;const raw={...manifest,airspace:{p:'airspace/mock.0123456789ab',z:[0,11]}};
 const dp=await createDataPlane({worker:false,clock:{now:()=>now,setTimeout,clearTimeout},manifestUrl:'http://fixture/manifest',fetch:async u=>{if(String(u).endsWith('manifest'))return Response.json(raw);if(String(u).endsWith('metadata')){metaCalls++;return new Response('',{status:503});}return new Response(null,{status:204});}});
 assert.equal(metaCalls,0);const errors=[];dp.on('error',e=>errors.push(e.key));
 const state={date:'1951-01-01',chartTiles:[],airspaceTiles:[{z:10,x:238,y:410}],scrub:{direction:0}};
 dp.setDemand({...state,airspaceTiles:[]});await settle();assert.equal(metaCalls,0); // layer off: no metadata request
 dp.setDemand(state);await wait(()=>errors.length===1);dp.setDemand(state);dp.setDemand(state);await settle();assert.equal(metaCalls,1);
 assert.deepEqual(dp.airspaceStatus(0,[-100,30,-90,40],{configured:true,enabled:true}),['Airspace data unavailable: HTTP 503']);
 now=1000;dp.setDemand(state);await wait(()=>metaCalls===2);dp.destroy();
});
test('decodes are bounded by decodeConcurrency and superseded queue entries are skipped',async t=>{
 const decodes=[];let active=0,peak=0,started=0;withBitmaps(t,()=>{started++;active++;peak=Math.max(peak,active);return new Promise(r=>decodes.push(v=>{active--;r(v);}));});
 const m=parseManifest({version:1,tileBase:'http://fixture/t/',fileBase:'http://fixture/',eras:[{k:'1950-01-01_to_1960-01-01',h:'0123456789ab',z:[4,4],b:null,c:null}]});
 const events=[],core=new DataCore(m,async()=>new Response(new Uint8Array([1]),{headers:{'content-type':'image/png'}}),{decodeConcurrency:3},e=>events.push(e));
 // Ten tiles fetched; centre-first order decodes x=7,8,6,9 before the demand shrinks to x=0..3.
 const tiles=Array.from({length:10},(_,i)=>({z:4,x:i,y:5}));core.setDemand({date:'1951-01-01',chartTiles:tiles,scrub:{direction:0}});
 await wait(()=>decodes.length===3);await settle();assert.equal(decodes.length,3);
 decodes.shift()({close(){}});await wait(()=>decodes.length===3);
 core.setDemand({date:'1951-01-01',chartTiles:tiles.slice(0,4),scrub:{direction:0}});
 while(decodes.length){decodes.shift()({close(){}});await settle();}
 assert.equal(peak,3);assert.equal(started,8);assert.equal(events.filter(e=>e==='tile').length,5);core.destroy();
});
test('interface lookups: eras in effect, an era archive by key, era member counts on pin rows, the airspace stack and credits',async()=>{
 const raw={...manifest,eras:[era,{k:'1955-01-01_to_1960-01-01',h:'123456789abc',z:[6,10],b:[-110,25,-80,45],c:null}],airspace:{p:'airspace/mock.0123456789ab',z:[0,11]}};
 const inventory={locations:{Dallas:{ref:'d',charts:[{d:'1950-01-01',e:'1960-01-01'},{d:'1955-01-01',e:'1960-01-01'}]}},rings:{d:[[[-98,31],[-95,31],[-95,34],[-98,34]]]}};
 const meta={archive_aero:{regions:{us:{name:'US',source:'FAA NASR',source_url:'https://faa.example/',cycles:['1951-01-01'],boxes:[[-130,20,-60,50]]}}}};
 const dp=await createDataPlane({worker:false,manifestUrl:'http://fixture/manifest',fetch:async url=>{const u=String(url);return u.endsWith('manifest')?Response.json(raw):u.endsWith('metadata')?Response.json(meta):u.endsWith('.json')?Response.json(inventory):new Response(null,{status:204});}});
 assert.equal(dp.manifest.hasBasemap,false);assert.equal(dp.eraCountAt('1951-01-01'),1);assert.equal(dp.eraCountAt('1956-01-01'),2);assert.equal(dp.eraCountAt('1970-01-01'),0);
 assert.deepEqual(dp.eraSource('1955-01-01_to_1960-01-01'),{path:'sectionals/1955-01-01_to_1960-01-01.123456789abc',zoom:[6,10]});assert.equal(dp.eraSource('1900-01-01_to_1901-01-01'),null);
 // One chart per era in this shard. The second era's extent reaches far past the chart's own ring: it has members the shard cannot see.
 const rows=await dp.queryPin(-96.7,32.7,'1956-01-01');assert.deepEqual(rows.map(r=>[r.chart.eraKey,r.members]),[['1950-01-01_to_1960-01-01',1],['1955-01-01_to_1960-01-01',2]]);
 assert.deepEqual(dp.airspaceCredits(),[]);
 assert.deepEqual(await dp.airspaceStack(-96.7,32.7,Math.floor(Date.UTC(1951,0,5)/86400000)),{here:[{rg:'us',name:'US',source:'FAA NASR',note:null,cycle:'1951-01-01'}],rows:[]});
 await wait(()=>dp.airspaceCredits().length===1);assert.deepEqual(dp.airspaceCredits(),[{source:'FAA NASR',url:'https://faa.example/'}]);dp.destroy();
});
