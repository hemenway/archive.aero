import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/demo.html?test');
  await page.waitForFunction(() => window.ready);
  await page.evaluate(() => {
    const c = document.querySelector('canvas');
    document.querySelector('header').style.display = 'none'; document.querySelector('footer').style.display = 'none';
    c.style.flex = 'none'; c.style.width = '256px'; c.style.height = '256px'; r.resize();
    r.setCamera({ x: .625, y: .625, zoom: 2 });
    window.dst = { z: 2, x: 2, y: 2 };
    window.plan = (id, items) => r.setChartPlan({ id, tiles: [{ dst, items: items.map(item => ({ dst, src: dst, ...item })) }] });
    window.tick = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    window.pixel = (x = 128, y = 128) => {
      const p = new Uint8Array(4), gl = r.gl;
      gl.readPixels(Math.round(x * r.dpr), Math.round((r.camera.height - y) * r.dpr - 1), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, p); return Array.from(p);
    };
    window.solid = async (key, color, size = 256) => { r.upload(key, await syntheticTile(size, color)); };
  });
});

test('Web Mercator round trip, wrap, bounds, center-first tile resolution', async ({ page }) => {
  const result = await page.evaluate(() => {
    const locations = [{ lng: 45, lat: -40.97989806962013 }, { lng: -179, lat: 60 }, { lng: 181, lat: -80 }, { lng: 0, lat: 0 }];
    const errors = locations.map(p => { const q = r.unproject(r.project(p)); return [Math.abs(((q.lng - p.lng + 540) % 360) - 180), Math.abs(q.lat - p.lat)]; });
    r.setCamera({ x: 1.01, y: .5, zoom: 4 });
    const wrapped = r.project({ lng: -176.4, lat: 0 });
    const chart = r.visibleTiles(256), base = r.visibleTiles(512);
    r.fitBounds([170, -10, -170, 10], { padding: 20, maxZoom: 8 });
    const fit = r.getCamera();
    return { errors, wrapped, chart, base, fit };
  });
  for (const pair of result.errors) for (const error of pair) expect(error).toBeLessThan(1e-6);
  expect(result.wrapped.x).toBeCloseTo(128, 5);
  expect(result.chart[0].z).toBe(4); expect(result.base[0].z).toBe(3);
  expect(result.chart.every(t => t.x >= 0 && t.x < 2 ** t.z)).toBe(true);
  expect(result.fit.x).toBeCloseTo(1, 5); expect(result.fit.zoom).toBeLessThanOrEqual(8);
});

test('wheel settles on integer zoom around cursor; focused keyboard pans', async ({ page, browserName }) => {
  // Playwright mouse.wheel is unavailable for WebKit with mobile emulation.
  await page.evaluate(() => {
    const c = r.canvas; c.dispatchEvent(new WheelEvent('wheel', { deltaY: -84, clientX: 80, clientY: 90, bubbles: true, cancelable: true }));
  });
  await expect.poll(() => page.evaluate(() => r.getCamera().zoom)).toBe(3);
  const before = await page.evaluate(() => r.getCamera().x);
  await page.locator('canvas').focus(); await page.keyboard.press('ArrowRight');
  expect(await page.evaluate(() => r.getCamera().x)).toBeGreaterThan(before);
  await page.keyboard.press('+'); await expect.poll(() => page.evaluate(() => r.getCamera().zoom)).toBe(4);
});

test('atomic multi-item swaps: every sampled animation frame retains charts; stale delivery stays superseded', async ({ page }) => {
  const result = await page.evaluate(async () => {
    r.setBasemapPlan([{ dst: { z: 1, x: 1, y: 1 }, items: [{ key: 'base/1/1/1', src: { z: 1, x: 1, y: 1 }, dst: { z: 1, x: 1, y: 1 } }] }]);
    await solid('base/1/1/1', '#ff00ff', 512); await solid('red/2/2/2', '#dc1414'); plan('red', [{ key: 'red/2/2/2' }]); await tick();
    const initial = pixel(); const samples = [];
    plan('blue', [{ key: 'lower/2/2/2' }, { key: 'blue/2/2/2' }]);
    await solid('lower/2/2/2', '#00ffff');
    await new Promise(resolve => { let frames = 0; const sample = () => { samples.push(pixel()); if (++frames < 12) requestAnimationFrame(sample); else resolve(); }; requestAnimationFrame(sample); });
    await solid('blue/2/2/2', '#1414dc'); await tick(); const blue = pixel();
    plan('obsolete', [{ key: 'obsolete/2/2/2' }]); plan('green', [{ key: 'green/2/2/2' }]);
    await solid('green/2/2/2', '#14c814'); await tick(); await solid('obsolete/2/2/2', '#0000ff'); await tick();
    return { initial, samples, blue, final: pixel() };
  });
  expect(result.initial.slice(0, 3)).toEqual([220, 20, 20]);
  expect(result.samples.length).toBe(12);
  for (const p of result.samples) expect(p.slice(0, 3)).toEqual([220, 20, 20]);
  expect(result.blue.slice(0, 3)).toEqual([20, 20, 220]); expect(result.final.slice(0, 3)).toEqual([20, 200, 20]);
});

test('resident same-path ancestor draws its UV quadrant, then sharpens', async ({ page }) => {
  const p = await page.evaluate(async () => {
    const c = new OffscreenCanvas(256, 256), ctx = c.getContext('2d');
    ctx.fillStyle = '#ff0000'; ctx.fillRect(0, 0, 256, 256); ctx.fillStyle = '#00ff00'; ctx.fillRect(0, 0, 128, 128);
    r.upload('archive/1/1/1', await createImageBitmap(c)); await tick();
    plan('ancestor', [{ key: 'archive/2/2/2' }]); await tick(); const ancestor = pixel();
    await solid('archive/2/2/2', '#0000ff'); await tick(); return { ancestor, real: pixel() };
  });
  expect(p.ancestor.slice(0, 3)).toEqual([0, 255, 0]); expect(p.real.slice(0, 3)).toEqual([0, 0, 255]);
});

test('clip ring stencil masks outside and handles concave rings', async ({ page }) => {
  const p = await page.evaluate(async () => {
    const points = [[30,30],[220,30],[220,100],[100,100],[100,220],[30,220]].map(([x,y]) => { const p=r.unproject({x,y}); return [p.lng,p.lat]; });
    r.setClipRing('concave', points); await solid('chart/2/2/2', '#dc1414'); plan('clip', [{ key:'chart/2/2/2',clip:'concave' }]); await tick();
    return { inside:pixel(60,60), notch:pixel(160,160), outside:pixel(10,10) };
  });
  expect(p.inside.slice(0,3)).toEqual([220,20,20]); expect(p.notch.slice(0,3)).not.toEqual([220,20,20]); expect(p.outside.slice(0,3)).not.toEqual([220,20,20]);
});

test('C4 visibility is identical in shader and CPU pick, with no buffer uploads on date/filter changes', async ({ page }) => {
  const result = await page.evaluate(async () => {
    let uploads=0; const gl=r.gl, original=gl.bufferData.bind(gl), sub=gl.bufferSubData.bind(gl);
    gl.bufferData=(...args)=>{uploads++;return original(...args);}; gl.bufferSubData=(...args)=>{uploads++;return sub(...args);};
    r.setAirfields({ mx:new Float32Array([.625]),my:new Float32Array([.625]), start:new Uint16Array([1960]),end:new Uint16Array([1980]),status:new Uint8Array([1]) });
    const initial=uploads, results=[];
    for(const year of [null,1959,1960,1980,1981]) { r.setAirfieldFilter({year,statusMask:7}); await tick(); results.push({year,pixel:pixel(),picked:r.pick(128,128)}); }
    r.setAirfieldFilter({year:1970,statusMask:1}); await tick(); const masked=r.pick(128,128),afterFilters=uploads;
    // Open fields ignore an end year; undated fields remain visible.
    r.fields.status[0]=0; r.setAirfields(r.fields); r.setAirfieldFilter({year:2026,statusMask:7}); await tick(); const open=r.pick(128,128);
    return { initial,afterFilters, results,masked,open };
  });
  expect(result.initial).toBe(1); expect(result.afterFilters).toBe(1);
  expect(result.results.map(v=>!!v.picked)).toEqual([true,false,true,true,false]);
  for (let i=0;i<result.results.length;i++) expect(result.results[i].pixel.slice(0,3).join(',') === '194,59,42').toBe([true,false,true,true,false][i]);
  expect(result.masked).toBeNull(); expect(result.open).toEqual({kind:'airfield',index:0});
});

test('pick includes dot radius, stroke and fine/coarse tolerance, including world wrap', async ({ page }) => {
  const result=await page.evaluate(async()=>{
    r.setAirfields({mx:new Float32Array([.625]),my:new Float32Array([.625]),start:new Uint16Array([0]),end:new Uint16Array([0]),status:new Uint8Array([2])});
    const rect=r.canvas.getBoundingClientRect(), picked=[];
    for(const coarse of [false,true]) {r.coarse=coarse; const hit=r._radius()+1+(coarse?14:4);picked.push(!!r.pick(rect.left+128+hit-.1,rect.top+128),!!r.pick(rect.left+128+hit+.1,rect.top+128));}
    r.setCamera({x:1.625,y:.625,zoom:2});picked.push(!!r.pick(rect.left+128,rect.top+128)); return picked;
  });
  expect(result).toEqual([true,false,true,false,true]);
});

test('LRU eviction preserves current plan references and old on-screen textures', async ({ page }) => {
  const result=await page.evaluate(async()=>{
    r.destroy(); const {createRenderer}=await import('/index.js'); window.r=createRenderer(document.querySelector('canvas'),{minZoom:0,maxTextureBytes:4*256*256*4,preserveDrawingBuffer:true});
    r.setCamera({x:.625,y:.625,zoom:2}); const evicted=[];r.on('evict',e=>evicted.push(e.key));
    await solid('old/2/2/2','#dc1414'); plan('old',[{key:'old/2/2/2'}]);await tick();
    plan('pending',[{key:'new/2/2/2'},{key:'not-yet/2/2/2'}]); await solid('new/2/2/2','#00ff00'); await tick();
    for(let i=0;i<10;i++){await solid(`unused-${i}/2/2/2`,'#0000ff');await tick();}
    const during=pixel();await solid('not-yet/2/2/2','#0000ff');await tick();
    return {evicted,during,bytes:r.stats().textureBytes,old:r.hasTexture('old/2/2/2'),current:r.hasTexture('new/2/2/2'),final:pixel()};
  });
  expect(result.evicted.length).toBeGreaterThan(0);expect(result.evicted).not.toContain('old/2/2/2');expect(result.evicted).not.toContain('new/2/2/2');
  expect(result.during.slice(0,3)).toEqual([220,20,20]);expect(result.current).toBe(true);expect(result.bytes).toBeLessThanOrEqual(4*256*256*4);expect(result.final.slice(0,3)).toEqual([0,0,255]);
});

test('context loss restores resources, emits events, and accepts re-requested textures', async ({ page }) => {
  const supported=await page.evaluate(()=>!!r.gl.getExtension('WEBGL_lose_context'));test.skip(!supported,'WEBGL_lose_context unavailable');
  await page.evaluate(async()=>{
    window.contextEvents=[];r.on('contextlost',()=>contextEvents.push('lost'));r.on('contextrestored',()=>contextEvents.push('restored'));
    await solid('chart/2/2/2','#dc1414');plan('loss',[{key:'chart/2/2/2'}]);await tick();
    window.lose=r.gl.getExtension('WEBGL_lose_context');lose.loseContext();
  });
  await expect.poll(()=>page.evaluate(()=>window.contextEvents)).toEqual(['lost']);
  await page.evaluate(()=>lose.restoreContext());await expect.poll(()=>page.evaluate(()=>window.contextEvents)).toEqual(['lost','restored']);
  const result=await page.evaluate(async()=>{const empty=!r.hasTexture('chart/2/2/2');await solid('chart/2/2/2','#00ff00');await tick();return {empty,pixel:pixel(),error:r.gl.getError()};});
  expect(result.empty).toBe(true);expect(result.pixel.slice(0,3)).toEqual([0,255,0]);expect(result.error).toBe(0);
});

test('airspace day/class/region filters execute in the GPU, with no re-upload', async ({ page }) => {
  const result=await page.evaluate(async()=>{
    let uploads=0;const gl=r.gl,old=gl.bufferData.bind(gl);gl.bufferData=(...args)=>{uploads++;return old(...args);};
    r.setAirspaceTile('2/2/2',{positions:new Float32Array([.55,.625,.70,.625]),starts:new Uint32Array([0,2]),from:new Int32Array([10]),to:new Int32Array([20]),style:new Uint8Array([1]),rg:new Uint8Array([1])});
    const initial=uploads, colors=[];
    for(const filter of [{day:15,classMask:3,regionMask:7},{day:20},{day:15,classMask:2},{classMask:3,regionMask:1},{regionMask:2}]) {r.setAirspaceFilter(filter);await tick();colors.push(pixel());}
    return {initial,uploads,colors,error:gl.getError()};
  });
  expect(result.initial).toBe(1);expect(result.uploads).toBe(1);expect(result.error).toBe(0);
  expect(result.colors[0].slice(0,3)).toEqual([79,140,245]);expect(result.colors[4].slice(0,3)).toEqual([79,140,245]);
  for(const i of [1,2,3])expect(result.colors[i].slice(0,3)).not.toEqual([79,140,245]);
});

test('offline demo loads only local assets; draws and swaps synthetic eras',async({page})=>{
  const foreign=[];page.on('request',req=>{if(!req.url().startsWith('http://127.0.0.1:4181/'))foreign.push(req.url());});
  await page.goto('/demo.html');await page.waitForFunction(()=>window.r?.stats().textures>5);
  await page.locator('#era').fill('2');await expect(page.locator('#year')).toHaveText('2000');
  await page.waitForFunction(()=>window.r?.planId>1);expect(foreign).toEqual([]);
});

test('randomly delayed overlapping eras never expose the basemap in any sampled frame', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const baseDst={z:1,x:1,y:1};r.setBasemapPlan([{dst:baseDst,items:[{key:'base/1/1/1',src:baseDst,dst:baseDst}]}]);
    await solid('base/1/1/1','#ff00ff',512);await solid('initial/2/2/2','#dc1414');plan('initial',[{key:'initial/2/2/2'}]);await tick();
    let stop=false,samples=0,leaks=0;const sample=()=>{const p=pixel();samples++;if(p[0]===255&&p[1]===0&&p[2]===255)leaks++;if(!stop)requestAnimationFrame(sample);};requestAnimationFrame(sample);
    for(let era=0;era<5;era++) {
      const keys=[0,1,2].map(layer=>`random-${era}-${layer}/2/2/2`);plan(`era-${era}`,keys.map(key=>({key})));
      await Promise.all(keys.map(key=>new Promise(resolve=>setTimeout(async()=>{await solid(key,era%2?'#1414dc':'#14c814');resolve();},20+Math.random()*150))));await tick();
    }
    stop=true;return {samples,leaks,final:pixel()};
  });
  expect(result.samples).toBeGreaterThan(10);expect(result.leaks).toBe(0);expect(result.final.slice(0,3)).toEqual([20,200,20]);
});

test('empty destination plans remove charts atomically and hidden/opacity style updates draw correctly', async ({ page }) => {
  const result=await page.evaluate(async()=>{
    await solid('style/2/2/2','#dc1414');plan('style',[{key:'style/2/2/2'}]);await tick();const chart=pixel();
    r.setChartStyle({hidden:true});await tick();const hidden=pixel(),calls=r.stats().drawCalls;
    r.setChartStyle({hidden:false,opacity:.5});await tick();const half=pixel();
    plan('empty',[]);await tick();return {chart,hidden,calls,half,empty:pixel()};
  });
  expect(result.chart.slice(0,3)).toEqual([220,20,20]);expect(result.calls).toBe(0);expect(result.hidden).toEqual(result.empty);expect(result.half[0]).toBeGreaterThan(result.hidden[0]);expect(result.half[0]).toBeLessThan(result.chart[0]);
});

test('draws only when invalidated; resize updates physical pixels and camera projection', async ({ page }) => {
  const result=await page.evaluate(async()=>{
    let draws=0;r.on('render',()=>draws++);await new Promise(resolve=>setTimeout(resolve,150));const first=draws;
    await new Promise(resolve=>setTimeout(resolve,120));const idle=draws;
    const rendered=new Promise(resolve=>{const listener=()=>{r.off('render',listener);resolve();};r.on('render',listener);});r.canvas.style.width='320px';r.canvas.style.height='180px';await rendered;
    return {first,idle,resized:draws,width:r.canvas.width,height:r.canvas.height,dpr:r.dpr,point:r.project(r.unproject({x:160,y:90}))};
  });
  expect(result.idle).toBe(result.first);expect(result.resized).toBeGreaterThan(result.idle);expect(result.width).toBe(320*result.dpr);expect(result.height).toBe(180*result.dpr);expect(result.point.x).toBeCloseTo(160,5);expect(result.point.y).toBeCloseTo(90,5);
});

test('pointer pan has inertia, double tap zooms, pinch settles without rotation', async ({ page }) => {
  const result=await page.evaluate(async()=>{
    const c=r.canvas, send=(type,id,x,y)=>c.dispatchEvent(new PointerEvent(type,{pointerId:id,pointerType:'touch',clientX:x,clientY:y,button:0,bubbles:true}));
    // Synthetic events do not establish browser pointer capture.
    const capture=c.setPointerCapture;c.setPointerCapture=()=>{};
    send('pointerdown',1,100,100);send('pointermove',1,120,100);send('pointerup',1,120,100);
    const pan=r.getCamera();await new Promise(resolve=>setTimeout(resolve,550));const inertia=r.getCamera();
    r.setCamera({x:.625,y:.625,zoom:2});
    send('pointerdown',1,80,80);send('pointerup',1,80,80);send('pointerdown',1,80,80);send('pointerup',1,80,80);
    await new Promise(resolve=>setTimeout(resolve,260));const double=r.getCamera().zoom;
    send('pointerdown',1,70,120);send('pointerdown',2,170,120);send('pointermove',1,20,120);send('pointerup',1,20,120);send('pointerup',2,170,120);
    await new Promise(resolve=>setTimeout(resolve,260));c.setPointerCapture=capture;return {pan,inertia,double,pinch:r.getCamera().zoom};
  });
  expect(result.pan.x).toBeLessThan(.625);expect(result.inertia.x).toBeLessThan(result.pan.x);expect(result.double).toBe(3);expect(result.pinch).toBe(4);
});

test('small memory pools repurpose unpinned array pages between tile sizes and hold pinned pixels under pressure',async({page})=>{
  const result=await page.evaluate(async()=>{
    r.destroy();const {createRenderer}=await import('/index.js');window.r=createRenderer(document.querySelector('canvas'),{minZoom:0,maxTextureBytes:1048576,preserveDrawingBuffer:true});r.setCamera({x:.625,y:.625,zoom:2});
    const evicted=[];r.on('evict',e=>evicted.push(e.key));for(let i=0;i<4;i++){await solid(`cache-${i}/2/2/2`,'#0000ff');await tick();}
    await solid('base/1/1/1','#ff00ff',512);await tick();const base=r.hasTexture('base/1/1/1');
    // The unpinned basemap page can then be reclaimed for charts.
    await solid('old/2/2/2','#dc1414');plan('old',[{key:'old/2/2/2'}]);await tick();
    plan('too-large',[0,1,2,3].map(i=>({key:`needed-${i}/2/2/2`})));for(let i=0;i<4;i++)await solid(`needed-${i}/2/2/2`,'#00ff00');await tick();
    const pressure=!!r.queue.length,old= r.hasTexture('old/2/2/2'),held=pixel();
    plan('smaller',[{key:'needed-3/2/2/2'}]);await tick();return {base,evicted,pressure,old,held,final:pixel(),bytes:r.stats().textureBytes};
  });
  expect(result.base).toBe(true);expect(result.evicted.length).toBeGreaterThanOrEqual(4);expect(result.pressure).toBe(true);expect(result.old).toBe(true);expect(result.held.slice(0,3)).toEqual([220,20,20]);expect(result.final.slice(0,3)).toEqual([0,255,0]);expect(result.bytes).toBeLessThanOrEqual(1048576);
});

test('unsupported devices throw the exported error',async({page})=>{
  expect(await page.evaluate(async()=>{const {createRenderer,RendererUnsupportedError}=await import('/index.js');const c=document.createElement('canvas');c.getContext=()=>null;try{createRenderer(c);return false;}catch(e){return e instanceof RendererUnsupportedError&&e.name==='RendererUnsupportedError';}})).toBe(true);
});

test('zoom transitions retain the completed ancestor or child mosaic before new plans arrive',async({page})=>{
  const result=await page.evaluate(async()=>{
    await solid('old/2/2/2','#dc1414');plan('old',[{key:'old/2/2/2'}]);await tick();
    r.setCamera({x:.625,y:.625,zoom:3});await tick();const ancestor=pixel();
    // A parent/old cell should remain drawn even when a new era is pending.
    const child={z:3,x:5,y:5};r.setChartPlan({id:'pending-child',tiles:[{dst:child,items:[{dst:child,src:child,key:'new/3/5/5'}]}]});await tick();const pending=pixel();
    r.destroy();const {createRenderer}=await import('/index.js');window.r=createRenderer(document.querySelector('canvas'),{minZoom:0,preserveDrawingBuffer:true});r.setCamera({x:.6875,y:.6875,zoom:3});
    await solid('new/3/5/5','#00ff00');r.setChartPlan({id:'child-only',tiles:[{dst:child,items:[{dst:child,src:child,key:'new/3/5/5'}]}]});await tick();
    r.setCamera({x:.6875,y:.6875,zoom:2});await tick();const childMosaic=pixel();
    return {ancestor,pending,childMosaic};
  });
  expect(result.ancestor.slice(0,3)).toEqual([220,20,20]);expect(result.pending.slice(0,3)).toEqual([220,20,20]);expect(result.childMosaic.slice(0,3)).toEqual([0,255,0]);
});

test('uploads are capped per frame, transfer bitmap ownership, and close duplicate deliveries',async({page})=>{
  const result=await page.evaluate(async()=>{
    const bitmaps=[];for(let i=0;i<12;i++)bitmaps.push(await syntheticTile(256,'#00ff00'));
    const counts=[];let uploads=0;const gl=r.gl,original=gl.texSubImage3D.bind(gl);gl.texSubImage3D=(...args)=>{uploads++;return original(...args);};
    r.on('render',()=>{counts.push(uploads);uploads=0;});
    for(let i=0;i<12;i++)r.upload(`upload-${i}/2/2/2`,bitmaps[i]);
    const duplicate=await syntheticTile(256,'#0000ff');r.upload('upload-0/2/2/2',duplicate);
    await new Promise(resolve=>{const listener=()=>{if(!r.queue.length){r.off('render',listener);resolve();}};r.on('render',listener);});
    return {counts,closed:bitmaps.every(bitmap=>bitmap.width===0),duplicateClosed:duplicate.width===0,resident:r.stats().textures};
  });
  expect(result.counts.reduce((a,b)=>a+b,0)).toBe(12);expect(Math.max(...result.counts)).toBeLessThanOrEqual(4);expect(result.closed).toBe(true);expect(result.duplicateClosed).toBe(true);expect(result.resident).toBe(12);
});

test('context restoration rebuilds clip, airfield, and airspace GPU buffers',async({page})=>{
  const supported=await page.evaluate(()=>!!r.gl.getExtension('WEBGL_lose_context'));test.skip(!supported,'WEBGL_lose_context unavailable');
  await page.evaluate(async()=>{
    r.setClipRing('clip',[[0,-20],[90,-20],[90,-60],[0,-60]]);
    r.setAirfields({mx:new Float32Array([.625]),my:new Float32Array([.625]),start:new Uint16Array([1900]),end:new Uint16Array([0]),status:new Uint8Array([0])});
    r.setAirspaceTile('2/2/2',{positions:new Float32Array([.55,.65,.70,.65]),starts:new Uint32Array([0,2]),from:new Int32Array([10]),to:new Int32Array([20]),style:new Uint8Array([1]),rg:new Uint8Array([1])});
    r.setAirspaceFilter({day:15,classMask:3,regionMask:7});
    await solid('chart/2/2/2','#dc1414');plan('loss',[{key:'chart/2/2/2',clip:'clip'}]);await tick();
    window.restored=false;r.on('contextrestored',()=>window.restored=true);window.lose=r.gl.getExtension('WEBGL_lose_context');lose.loseContext();
  });
  await expect.poll(()=>page.evaluate(()=>r.lost)).toBe(true);await page.evaluate(()=>lose.restoreContext());await page.waitForFunction(()=>window.restored);
  const result=await page.evaluate(async()=>{await solid('chart/2/2/2','#dc1414');await tick();return {field:pixel(),chart:pixel(110,110),calls:r.stats().drawCalls,error:r.gl.getError(),pick:r.pick(128,128)};});
  expect(result.field.slice(0,3)).toEqual([54,163,93]);expect(result.chart.slice(0,3)).toEqual([220,20,20]);expect(result.calls).toBeGreaterThanOrEqual(5);expect(result.error).toBe(0);expect(result.pick).toEqual({kind:'airfield',index:0});
});

test('offline demo finishes repeated era scrubs within its default texture budget',async({page,browserName})=>{
  if(browserName==='chromium')await page.setViewportSize({width:1440,height:900});
  await page.goto('/demo.html');await page.waitForFunction(()=>window.r&&window.demoPending&&!demoPending()&&!r.queue.length);
  for(const era of ['3','1','2'])await page.locator('#era').fill(era);
  await page.waitForFunction(()=>!demoPending()&&!r.queue.length);
  const stats=await page.evaluate(()=>({stats:r.stats(),budget:r.budget,year:document.querySelector('#year').textContent}));
  expect(stats.year).toBe('2000');expect(stats.stats.textureBytes).toBeLessThanOrEqual(stats.budget);
});

test('pick grid includes fields on the south Mercator edge and antimeridian',async({page})=>{
  const result=await page.evaluate(async()=>{r.setCamera({x:1,y:1,zoom:2});r.setAirfields({mx:new Float32Array([1]),my:new Float32Array([1]),start:new Uint16Array([1900]),end:new Uint16Array([0]),status:new Uint8Array([0])});await tick();return {picked:r.pick(128,128),color:pixel()};});
  expect(result.picked).toEqual({kind:'airfield',index:0});expect(result.color.slice(0,3)).toEqual([54,163,93]);
});
