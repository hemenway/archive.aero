import {test,expect} from '@playwright/test';
const tile={z:10,x:238,y:410};
for(const mode of [{worker:true},{worker:true,decodeBitmap:false},{worker:false,decodeBitmap:false}]) test(`decode, upload, eviction and re-request ${JSON.stringify(mode)}`,async({page})=>{
 await page.goto('/');await page.waitForFunction(()=>window.booted);
 const result=await page.evaluate(async mode=>{
  const dp=await window.createDataPlane({manifestUrl:location.origin+'/next/manifest.json',...mode});
  window.dp=dp;const tiles=[{z:10,x:238,y:410}],date='1950-01-10',keys=dp.planCharts(date,tiles).tiles.flatMap(t=>t.items.map(i=>i.key));const received=[];
  const complete=new Promise((resolve,reject)=>{
   dp.on('error',e=>reject(new Error(e.error.message)));dp.on('tile',({key,bitmap})=>{
    const c=document.querySelector('canvas').getContext('2d');c.drawImage(bitmap,0,0);bitmap.close?.();received.push(key);if(keys.every(k=>received.includes(k)))resolve();
   });
  });dp.setDemand({date,chartTiles:tiles,basemapTiles:[],scrub:{direction:0,velocity:0}});await complete;
  await new Promise(r=>setTimeout(r,20));const before=dp.readiness(date,tiles),pixels=[...document.querySelector('canvas').getContext('2d').getImageData(0,0,1,1).data];
  dp.markEvicted(keys[0]);const evicted=dp.readiness(date,tiles);const events=received.length;
  const restored=new Promise(resolve=>dp.on('tile',e=>{if(e.key===keys[0])resolve();}));dp.setDemand({date,chartTiles:tiles,scrub:{direction:0,velocity:0}});await restored;
  const stats=dp.stats();dp.destroy();return {before,evicted,pixels,events,after:received.length,stats};
 },mode);
 expect(result.before).toBe(1);expect(result.evicted).toBe(0);expect(result.pixels[3]).toBe(255);expect(result.after).toBeGreaterThan(result.events);
});
test('module Worker transfers ImageBitmap and fallback transfers encoded bytes',async({page})=>{
 await page.goto('/');await page.waitForFunction(()=>window.booted);
 const outputs=await page.evaluate(async()=>{
  const run=async decodeBitmap=>{
   const worker=new Worker('/next/dataplane/worker.js',{type:'module'});const events=[];
   await new Promise((resolve,reject)=>{worker.onerror=reject;worker.onmessage=({data})=>{if(data.id===0){if(data.error)reject(data.error);else resolve();}};worker.postMessage({id:0,method:'init',args:[{manifestUrl:location.origin+'/next/manifest.json',decodeBitmap}]});});
   const out=await new Promise((resolve,reject)=>{worker.onmessage=({data})=>{if(data.event==='tile'||data.event==='encoded'){const p=data.payload;const c=document.querySelector('canvas').getContext('2d');if(p.bitmap)c.drawImage(p.bitmap,0,0);const r={kind:data.event,bitmap:typeof ImageBitmap!=='undefined'&&p.bitmap instanceof ImageBitmap,bytes:p.bytes?.byteLength};p.bitmap?.close();resolve(r);}if(data.event==='error')reject(data.payload.error);};worker.postMessage({id:1,method:'setDemand',args:[{date:'1950-01-10',chartTiles:[{z:10,x:238,y:410}],scrub:{direction:0}}]});});worker.terminate();return out;
  };return [await run(true),await run(false)];
 });
 // Mobile WebKit may not implement Worker createImageBitmap. Feature detection must still yield bytes.
 expect(['tile','encoded']).toContain(outputs[0].kind);if(outputs[0].kind==='tile')expect(outputs[0].bitmap).toBe(true);else expect(outputs[0].bytes).toBeGreaterThan(0);
 expect(outputs[1].kind).toBe('encoded');expect(outputs[1].bytes).toBeGreaterThan(0);
});
test('real fetch AbortController cancels server response and superseded demand delivers no old tile',async({page,request})=>{
 await page.goto('/');await page.waitForFunction(()=>window.booted);await request.get('/reset');
 const old=await page.evaluate(async()=>{
  const dp=await window.createDataPlane({manifestUrl:location.origin+'/next/manifest.json'});window.dp=dp;window.received=[];dp.on('tile',p=>{window.received.push(p.key);p.bitmap.close?.();});
  dp.setDemand({date:'1950-01-10',chartTiles:[{z:10,x:238,y:410}],solo:{paths:['sectionals/slow.0123456789ab'],zoom:[4,11]},scrub:{direction:0}});
 });
 await expect.poll(async()=>(await(await request.get('/metrics')).json()).requests).toBeGreaterThan(0);
 await page.evaluate(()=>window.dp.setDemand({date:'1950-01-10',chartTiles:[],scrub:{direction:0}}));
 await expect.poll(async()=>(await(await request.get('/metrics')).json()).cancelled).toBeGreaterThan(0);
 expect(await page.evaluate(()=>{window.dp.destroy();return window.received;})).toEqual([]);
});
test('airfields arrays retain backing buffer through transfer and details stay lazy',async({page})=>{
 await page.goto('/');await page.waitForFunction(()=>window.booted);
 const result=await page.evaluate(async()=>{const dp=await window.createDataPlane({manifestUrl:location.origin+'/next/manifest.json'});const a=await dp.loadAirfields(),b=await dp.loadAirfields(),d=await dp.airfieldDetails(0);const r={same:a.mx.buffer===a.status.buffer,mx:a.mx[0],again:b.mx[0],name:d.name,mask:dp.airspaceRegionMask(Math.floor(Date.parse('1950-01-10')/86400000))};dp.destroy();return r;});
 expect(result).toEqual({same:true,mx:.25,again:.25,name:'Mock Field',mask:1});
});
