import test from 'node:test';
import assert from 'node:assert/strict';
import {createDataPlane} from '../index.js';
const era={k:'1950-01-01_to_1960-01-01',h:'0123456789ab',z:[4,8],b:null,c:null};
const manifest={version:1,tileBase:'http://fixture/t/',fileBase:'http://fixture/',eras:[era],basemap:null,airspace:null,airfields:null,pins:{z:5,margin:2,base:'next/pins.0123456789ab/',shards:[391]},coverage:{segments:[]}};
const wait=async f=>{for(let i=0;i<30&&!f();i++)await new Promise(r=>setImmediate(r));assert.ok(f());};
test('main-thread facade, absent readiness and eviction are deterministic',async()=>{
 const calls=[];const dp=await createDataPlane({worker:false,manifestUrl:'http://fixture/manifest',fetch:async url=>{calls.push(String(url));return String(url).endsWith('manifest')?Response.json(manifest):new Response(null,{status:204});}});
 const tiles=[{z:10,x:238,y:410},{z:10,x:239,y:410}],date='1951-01-01';assert.deepEqual(dp.manifest.frames,['1950-01-01']);assert.equal(dp.readiness(date,tiles),0);let count=0;dp.on('absent',()=>count++);
 dp.setDemand({date,chartTiles:tiles,scrub:{direction:0}});await wait(()=>dp.readiness(date,tiles)===1);assert.equal(count,2); // shared full-res source plus bootstrap
 const n=calls.length,key=dp.planCharts(date,tiles).tiles[0].items[0].key;dp.markEvicted(key);assert.equal(dp.readiness(date,tiles),1);dp.setDemand({date,chartTiles:tiles});await wait(()=>count>=2);assert.equal(calls.length,n);dp.destroy();
});
test('queryPin loads only selected shard and failed JSON retries later',async()=>{
 let shardCalls=0;const inventory={locations:{Dallas:{ref:'d',charts:[{d:'1950-01-01',e:'1960-01-01'}]}},rings:{d:[[[-98,31],[-95,31],[-95,34],[-98,34]]]}};
 const dp=await createDataPlane({worker:false,manifestUrl:'http://fixture/manifest',fetch:async url=>{if(String(url).endsWith('manifest'))return Response.json(manifest);shardCalls++;return shardCalls===1?new Response('',{status:503}):Response.json(inventory);}});
 assert.deepEqual(await dp.queryPin(0,0,'1951-01-01'),[]);assert.equal(shardCalls,0);await assert.rejects(dp.queryPin(-96.7,32.7,'1951-01-01'));const rows=await dp.queryPin(-96.7,32.7,'1951-01-01');assert.equal(rows[0].location.name,'Dallas');assert.equal(rows[0].chart.published,true);await dp.queryPin(-96.7,32.7,'1951-01-01');assert.equal(shardCalls,2);dp.destroy();
});
test('a terminal Worker failure rejects future RPC instead of wedging promises',async t=>{
 const previous=globalThis.Worker;let instance;
 class MockWorker {constructor(){instance=this;}postMessage({id,method}){if(method==='init')queueMicrotask(()=>this.onmessage({data:{id,result:manifest}}));}terminate(){}}
 globalThis.Worker=MockWorker;t.after(()=>{if(previous===undefined)delete globalThis.Worker;else globalThis.Worker=previous;});
 const dp=await createDataPlane({manifestUrl:'http://fixture/manifest'});let error;dp.on('error',e=>{error=e;});instance.onerror({message:'Worker crash'});assert.equal(error.error.message,'Worker crash');await assert.rejects(dp.loadAirfields(),/Worker crash/);await assert.rejects(dp.queryPin(-96.7,32.7,'1951-01-01'),/Worker crash/);dp.destroy();
});
test('superseded airspace tiles release retained rings and tell renderer to remove batches',async()=>{
 const raw={...manifest,airspace:{p:'airspace/mock.0123456789ab',z:[0,11]}};
 const dp=await createDataPlane({worker:false,manifestUrl:'http://fixture/manifest',fetch:async url=>String(url).endsWith('manifest')?Response.json(raw):String(url).endsWith('metadata')?Response.json({archive_aero:{regions:{}}}):new Response(new Uint8Array(0))});
 const events=[];dp.on('airspace',e=>events.push(e));dp.setDemand({date:'1951-01-01',chartTiles:[],airspaceTiles:[{z:10,x:238,y:410}]});await wait(()=>events.length===1);dp.setDemand({date:'1951-01-01',chartTiles:[],airspaceTiles:[]});assert.equal(events[1].tileId,'10/238/410');assert.equal(events[1].batch,null);dp.destroy();
});
