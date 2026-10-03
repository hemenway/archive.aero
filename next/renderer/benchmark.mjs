// Run from the repository root: node next/renderer/benchmark.mjs
// Starts only the local fixture server, builds all image data in the browser.
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { writeFile } from 'node:fs/promises';
const server = spawn(process.execPath, [new URL('./server.mjs', import.meta.url).pathname], { stdio: 'ignore' });
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  for (let i = 0; i < 30; i++) {
    try { await page.goto('http://127.0.0.1:4181/demo.html?test'); break; }
    catch { await new Promise(resolve => setTimeout(resolve, 100)); }
  }
  await page.waitForFunction(() => window.ready);
  const measure = mode => page.evaluate(async mode => {
    document.querySelector('header')?.remove(); document.querySelector('footer')?.remove();
    let canvas = document.querySelector('canvas'); canvas.style.height = '900px'; r.resize();
    r.destroy();canvas.replaceWith(canvas=canvas.cloneNode());const {createRenderer}=await import('/index.js');window.r=createRenderer(canvas,{maxTextureBytes:(mode==='native'?128:64)*1048576});r.setCamera({ x: .27, y: .39, zoom: 9 });
    const charts = r.visibleTiles(256), base = r.visibleTiles(512);
    const prepared = [];
    const chartPlans = charts.map(dst => ({ dst, items: [0,1,2].map(a => {const src=mode==='native'?dst:{z:dst.z-1,x:Math.floor(dst.x/2),y:Math.floor(dst.y/2)};return {dst,src,key:`bench-${a}/${src.z}/${src.x}/${src.y}`};}) }));
    const basePlans = base.map(dst => ({ dst, items: [{ dst, src: dst, key: `base/${dst.z}/${dst.x}/${dst.y}` }] }));
    r.setChartPlan({ id: 'bench', tiles: chartPlans }); r.setBasemapPlan(basePlans);
    for (const tile of basePlans) for (const item of tile.items) prepared.push([item.key, await syntheticTile(512, '#172b3e')]);
    const created=new Set();for (const tile of chartPlans) for (const item of tile.items) if(!created.has(item.key)){created.add(item.key);prepared.push([item.key, await syntheticTile(256, '#946641')]);}
    const residentBytes=prepared.reduce((n,[,bitmap])=>n+bitmap.width*bitmap.height*4,0);
    const count = 8000, mx = new Float32Array(count), my = new Float32Array(count), start = new Uint16Array(count), end = new Uint16Array(count), status = new Uint8Array(count);
    for (let i=0;i<count;i++) { mx[i]=.264+(i%100)/100*.012;my[i]=.385+Math.floor(i/100)/80*.01;start[i]=1940+i%60;end[i]=2000+i%20;status[i]=i%3; }
    const fields={mx,my,start,end,status};
    const positions=[], starts=[0], codes=[];
    for(let l=0;l<96;l++) { const x=.264+l%12*.001,y=.385+Math.floor(l/12)*.001;for(const p of [[x,y],[x+.0008,y+.0002],[x+.0007,y+.0008],[x,y+.0005]])positions.push(...p);starts.push(positions.length/2);codes.push(l%8); }
    const lines={positions:new Float32Array(positions),starts:new Uint32Array(starts),style:new Uint8Array(codes),rg:new Uint8Array(codes.length),from:new Int32Array(codes.length).fill(-20000),to:new Int32Array(codes.length).fill(2147483647)};
    let uploadFrames=0;r.on('render',()=>{if(r.queue.length)uploadFrames++;});
    const uploadStart=performance.now();for(const [key,bitmap] of prepared)r.upload(key,bitmap);
    await new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(new Error('Upload queue stalled: '+r.queue.length+' pending; '+JSON.stringify(r.stats()))),45000);const listener=()=>{if(!r.queue.length){clearTimeout(timeout);r.off('render',listener);r.gl.finish();resolve();}};r.on('render',listener);});
    const uploadMs=performance.now()-uploadStart;r.setAirfields(fields);r.setAirspaceTile('0/0/0',lines);
    const cpu=[],synchronized=[],pixel=new Uint8Array(4);
    for(let i=0;i<140;i++) {
      await new Promise(resolve=>{const listener=()=>{r.off('render',listener);const before=performance.now();r.gl.finish();r.gl.readPixels(720,450,1,1,r.gl.RGBA,r.gl.UNSIGNED_BYTE,pixel);if(i>=20){cpu.push(r.stats().frameMs);synchronized.push(r.stats().frameMs+performance.now()-before);}resolve();};r.on('render',listener);r.setAirfieldFilter({year:1950+i%70});});
    }
    const metrics=a=>{a.sort((x,y)=>x-y);return {median:a[Math.floor(a.length*.5)],p95:a[Math.floor(a.length*.95)],max:a[a.length-1]};};
    const stats=r.stats(),debug=r.gl.getExtension('WEBGL_debug_renderer_info');
    return {mode,maxTextureBytes:r.budget,viewport:[canvas.width,canvas.height],dpr:r.dpr,zoom:r.getCamera().zoom,chartTiles:charts.length,basemapTiles:base.length,itemsPerTile:3,airfields:count,polylines:96,samples:cpu.length,
      cpuFrameMs:metrics(cpu),synchronizedFrameMs:metrics(synchronized),textures:stats.textures,allocatedTextureBytes:stats.textureBytes,residentTextureBytes:residentBytes,
      geometryBytes:r.fieldData.byteLength+Array.from(r.airspace.values()).reduce((s,b)=>s+b.groups.reduce((n,g)=>n+g.data.byteLength,0),0),drawCalls:stats.drawCalls,
      upload:{count:prepared.length,ms:uploadMs,frames:uploadFrames+1,tilesPerSecond:prepared.length/uploadMs*1000,MiBPerSecond:residentBytes/1048576/uploadMs*1000},
      gpu:debug?r.gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):'unavailable',userAgent:navigator.userAgent};
  },mode);
  const result={native:await measure('native'),overzoom:await measure('overzoom')};
  result.measuredAt = new Date().toISOString();
  await writeFile(new URL('./measurements.json', import.meta.url), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result,null,2));
} finally { await browser.close(); server.kill(); await once(server, 'exit').catch(()=>{}); }
