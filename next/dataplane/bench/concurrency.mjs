import {chromium} from 'playwright';
import {startServer} from './server.mjs';
import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch(),results=[];
try {for(const concurrency of [4,8,12]) {
 const server=await startServer({port:0});try {const page=await browser.newPage();await page.goto(server.origin);await page.waitForFunction(()=>window.booted);
 const ms=await page.evaluate(async concurrency=>{const dp=await createDataPlane({manifestUrl:location.origin+'/next/manifest.json',concurrency});dp.on('tile',({bitmap})=>bitmap.close?.());const tiles=Array.from({length:24},(_,i)=>({z:10,x:235+i%8,y:407+Math.floor(i/8)})),date='1950-01-10',start=performance.now();dp.setDemand({date,chartTiles:tiles,scrub:{direction:1,velocity:8}});while(dp.readiness(date,tiles)<1)await new Promise(r=>setTimeout(r,5));const result=performance.now()-start;dp.destroy();return result;},concurrency);
 results.push({concurrency,firstCompleteMs:ms});await page.close();}finally{await server.close();}
}console.log(JSON.stringify(results,null,2));await writeFile(new URL('./concurrency-results.json',import.meta.url),JSON.stringify(results,null,2)+'\n');}finally{await browser.close();}
