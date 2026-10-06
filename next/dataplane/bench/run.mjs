import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
import {startServer} from './server.mjs';
import {benchmarkPlanning} from './planning.mjs';
const server=await startServer({port:0}),browser=await chromium.launch();
try {
 const page=await browser.newPage();await page.goto(server.origin);await page.waitForFunction(()=>window.booted);
 const session=await page.evaluate(async concurrency=>{
  const errors=[],dp=await createDataPlane({manifestUrl:location.origin+'/next/manifest.json',concurrency});dp.on('tile',({bitmap})=>bitmap.close?.());dp.on('error',e=>errors.push(e.error.message));
  const tiles=Array.from({length:24},(_,i)=>({z:10,x:235+i%8,y:407+Math.floor(i/8)}));
  const state=date=>({date,chartTiles:tiles,basemapTiles:[],center:{x:238.5/1024,y:408.5/1024},scrub:{direction:1,velocity:8,playing:false}});
  const dates=Array.from({length:30},(_,i)=>new Date(Date.UTC(1950,i*4,10)).toISOString().slice(0,10));
  const wait=async(date)=>{const until=performance.now()+30000;while(dp.readiness(date,tiles)<1){if(performance.now()>until)throw new Error(`Plan timeout ${date}`);await new Promise(r=>setTimeout(r,5));}};
  const steps=[],start=performance.now();dp.setDemand(state(dates[0]));await wait(dates[0]);const first=performance.now()-start;
  // Rapid demand changes exercise real cancellation between buffered steps.
  for(let i=1;i<30;i++){const t=performance.now();dp.setDemand(state(dates[i]));await wait(dates[i]);steps.push(performance.now()-t);await new Promise(r=>setTimeout(r,30));}
  const burstStart=performance.now();for(let i=0;i<8;i++){dp.setDemand(state(dates[i]));await new Promise(r=>setTimeout(r,10));}await wait(dates[7]);const burstMs=performance.now()-burstStart;
  const playback=[];for(let i=8;i<12;i++){const t=performance.now();dp.setDemand({...state(dates[i]),scrub:{direction:1,velocity:.5,playing:true}});await wait(dates[i]);playback.push(performance.now()-t);await new Promise(r=>setTimeout(r,2000));}
  const stats=dp.stats();dp.destroy();steps.sort((a,b)=>a-b);return {firstCompleteMs:first,scrubP50Ms:steps[Math.floor(steps.length*.5)],scrubP95Ms:steps[Math.floor(steps.length*.95)],scrubSteps:30,burstMs,playbackReadyMs:playback,stats,errors};
 },Number(process.env.CONCURRENCY??8));
 const result={concurrency:Number(process.env.CONCURRENCY??8),latencyMs:Number(process.env.LATENCY??80),bandwidthMbps:Number(process.env.MBPS??10),tileBytes:server.tileBytes,planning:benchmarkPlanning(),session,network:server.metrics};
 await writeFile(new URL('./results.json',import.meta.url),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
} finally {await browser.close();await server.close();}
