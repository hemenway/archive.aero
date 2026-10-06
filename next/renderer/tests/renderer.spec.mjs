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
  expect(Math.min(result.fit.x, 1 - result.fit.x)).toBeLessThan(1e-5); expect(result.fit.zoom).toBeLessThanOrEqual(8);
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

test('a small wheel gesture still steps one whole level; steps during a step continue, never snap back', async ({ page }) => {
  // Trackpads deliver deltas far below 120 px: a gesture worth a third of a level snapped back to its start (2026-10-05).
  const wheel = (deltaY, n = 1) => page.evaluate(({ deltaY, n }) => {
    for (let i = 0; i < n; i++) r.canvas.dispatchEvent(new WheelEvent('wheel', { deltaY, clientX: 80, clientY: 90, bubbles: true, cancelable: true }));
  }, { deltaY, n });
  await wheel(-8, 5);
  await expect.poll(() => page.evaluate(() => r.getCamera().zoom)).toBe(3);
  // A second gesture while the first still animates: one more level from the first's target.
  await wheel(-8, 5); await page.waitForTimeout(80); await wheel(-8, 5);
  await expect.poll(() => page.evaluate(() => r.getCamera().zoom)).toBe(5);
  // Large deltas step several levels at once, capped at four.
  await wheel(2400);
  await expect.poll(() => page.evaluate(() => r.getCamera().zoom)).toBe(1);
  // Integer zoom is left alone by an empty debounce.
  await page.waitForTimeout(300);
  expect(await page.evaluate(() => r.getCamera().zoom)).toBe(1);
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

test('zooming back in after a date change draws the newer date, never the stale finer snapshot', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const c3 = { z: 3, x: 5, y: 5 }, c2 = { z: 2, x: 2, y: 2 }, cell = (dst, keys) => ({ dst, items: keys.map(key => ({ key, dst, src: { z: +key.split('/').at(-3), x: +key.split('/').at(-2), y: +key.split('/').at(-1) } })) });
    r.setCamera({ x: .6875, y: .6875, zoom: 3 });
    await solid('A/3/5/5', '#ff0000'); r.setChartPlan({ id: 'A', tiles: [cell(c3, ['A/3/5/5'])] }); await tick(); const a = pixel();
    r.setCamera({ x: .6875, y: .6875, zoom: 2 }); await solid('B/2/2/2', '#00ff00'); r.setChartPlan({ id: 'B', tiles: [cell(c2, ['B/2/2/2'])] }); await tick(); const b = pixel();
    r.setCamera({ x: .6875, y: .6875, zoom: 3 });
    // Date B at z3: its own tile is pending and a second archive has no resident ancestor, so the swap waits.
    r.setChartPlan({ id: 'B-z3', tiles: [cell(c3, ['B/3/5/5', 'other/3/5/5'])] }); await tick();
    return { a, b, back: pixel() };
  });
  expect(result.a.slice(0, 3)).toEqual([255, 0, 0]); expect(result.b.slice(0, 3)).toEqual([0, 255, 0]); expect(result.back.slice(0, 3)).toEqual([0, 255, 0]);
});

test('a snapshot evicted while off-screen is never drawn as a partial composite', async ({ page }) => {
  const result = await page.evaluate(async () => {
    r.destroy(); const { createRenderer } = await import('/index.js'); window.r = createRenderer(document.querySelector('canvas'), { minZoom: 0, maxTextureBytes: 5 * 256 * 256 * 4, preserveDrawingBuffer: true });
    const D = { z: 2, x: 2, y: 2 }, E = { z: 2, x: 3, y: 2 }, cell = (dst, keys) => ({ dst, items: keys.map(key => ({ key, dst, src: dst })) });
    r.setCamera({ x: .625, y: .625, zoom: 2 });
    await solid('a/2/2/2', '#ff0000'); await solid('b/2/2/2', '#0000ff'); r.setChartPlan({ id: 'p1', tiles: [cell(D, ['a/2/2/2', 'b/2/2/2']), cell(E, [])] }); await tick(); const before = pixel();
    // Pan one tile east: D sits in the off-screen buffer while E's four textures exhaust the budget.
    r.setCamera({ x: .875, y: .625, zoom: 2 }); r.setChartPlan({ id: 'p2', tiles: [cell(D, ['c/2/2/2']), cell(E, ['e/2/3/2', 'f/2/3/2', 'g/2/3/2', 'h/2/3/2'])] });
    for (const [key, color] of [['e/2/3/2', '#111111'], ['f/2/3/2', '#222222'], ['g/2/3/2', '#333333'], ['h/2/3/2', '#444444']]) await solid(key, color); await tick(); await tick();
    const evicted = { a: r.hasTexture('a/2/2/2'), b: r.hasTexture('b/2/2/2') };
    r.setCamera({ x: .625, y: .625, zoom: 2 }); await tick(); const back = pixel();
    await solid('c/2/2/2', '#00ff00'); await tick();
    return { before, evicted, back, final: pixel() };
  });
  expect(result.before.slice(0, 3)).toEqual([0, 0, 255]); expect(result.evicted).toEqual({ a: false, b: true });
  expect(result.back.slice(0, 3)).not.toEqual([0, 0, 255]); expect(result.final.slice(0, 3)).toEqual([0, 255, 0]);
});

test('resizing draws synchronously, so no frame is painted empty', async ({ page }) => {
  const result = await page.evaluate(async () => {
    await solid('size/2/2/2', '#dc1414'); plan('size', [{ key: 'size/2/2/2' }]); await tick();
    r.canvas.style.width = '300px'; r.resize(); return pixel(150, 128);
  });
  expect(result.slice(0, 3)).toEqual([220, 20, 20]);
});

test('camera longitude stays wrapped, animations take the short way round, tiny canvases fit without throwing', async ({ page }) => {
  const result = await page.evaluate(async () => {
    r.setCamera({ x: 1.625, y: .625, zoom: 2 }); const wrapped = r.getCamera().x;
    r.setCamera({ x: .99, y: .5, zoom: 4 }); r.setCamera({ x: .01, y: .5, zoom: 4 }, { animate: true });
    await new Promise(resolve => setTimeout(resolve, 120)); const mid = r.getCamera().x;
    await new Promise(resolve => setTimeout(resolve, 300)); const end = r.getCamera().x;
    r.canvas.style.width = '40px'; r.canvas.style.height = '40px'; r.resize();
    let threw = false; try { r.fitBounds([-124.8, 24.4, -67.1, 49.4], { padding: 70, maxZoom: 6 }); } catch { threw = true; }
    return { wrapped, mid, end, threw, zoom: r.getCamera().zoom };
  });
  expect(result.wrapped).toBeCloseTo(.625, 6); expect(Math.min(result.mid, 1 - result.mid)).toBeLessThan(.02); expect(result.end).toBeCloseTo(.01, 6);
  expect(result.threw).toBe(false); expect(result.zoom).toBe(0);
});

test('context loss reports every resident and queued texture evicted', async ({ page }) => {
  const supported = await page.evaluate(() => !!r.gl.getExtension('WEBGL_lose_context')); test.skip(!supported, 'WEBGL_lose_context unavailable');
  const result = await page.evaluate(async () => {
    const evicted = []; r.on('evict', e => evicted.push(e.key));
    await solid('one/2/2/2', '#ff0000'); await solid('two/2/2/2', '#00ff00'); await tick();
    r.upload('queued/2/2/2', await syntheticTile(256, '#0000ff'));
    const lost = new Promise(resolve => r.on('contextlost', resolve)); window.lose = r.gl.getExtension('WEBGL_lose_context'); lose.loseContext(); await lost;
    r.upload('late/2/2/2', await syntheticTile(256, '#ffffff'));
    return evicted.sort();
  });
  expect(result).toEqual(['late/2/2/2', 'one/2/2/2', 'queued/2/2/2', 'two/2/2/2']);
  await page.evaluate(() => lose.restoreContext());
});

test('current-plan textures jump the upload queue; superseded deliveries are dropped under pressure', async ({ page }) => {
  const result = await page.evaluate(async () => {
    r.destroy(); const { createRenderer } = await import('/index.js'); window.r = createRenderer(document.querySelector('canvas'), { minZoom: 0, maxTextureBytes: 4 * 256 * 256 * 4, preserveDrawingBuffer: true });
    r.setCamera({ x: .625, y: .625, zoom: 2 }); const evicted = []; r.on('evict', e => evicted.push(e.key));
    for (let i = 0; i < 4; i++) await solid(`idle-${i}/2/2/2`, '#0000ff'); await tick();
    plan('needed', [0, 1, 2, 3].map(i => ({ key: `needed-${i}/2/2/2` })));
    const stale = await syntheticTile(256, '#ff0000'), bitmaps = await Promise.all([0, 1, 2, 3].map(() => syntheticTile(256, '#00ff00')));
    r.upload('stale/2/2/2', stale); for (let i = 0; i < 4; i++) r.upload(`needed-${i}/2/2/2`, bitmaps[i]);
    await tick(); await tick(); await tick();
    return { evicted, needed: [0, 1, 2, 3].every(i => r.hasTexture(`needed-${i}/2/2/2`)), stale: r.hasTexture('stale/2/2/2'), queue: r.queue.length, pixel: pixel() };
  });
  expect(result.needed).toBe(true); expect(result.stale).toBe(false); expect(result.queue).toBe(0);
  expect(result.evicted).toContain('stale/2/2/2'); expect(result.pixel.slice(0, 3)).toEqual([0, 255, 0]);
});

test('a full basemap budget does not stop needed chart uploads, and the queue keeps draining', async ({ page }) => {
  const result = await page.evaluate(async () => {
    // Room for one 512 px basemap page layer plus three 256 px chart layers (pages take what the budget leaves).
    r.destroy(); const { createRenderer } = await import('/index.js'); window.r = createRenderer(document.querySelector('canvas'), { minZoom: 0, maxTextureBytes: (512 * 512 + 3 * 256 * 256) * 4, preserveDrawingBuffer: true });
    r.setCamera({ x: .625, y: .625, zoom: 2 }); const pressure = []; r.on('texturepressure', e => pressure.push(e.key));
    const base = { z: 1, x: 1, y: 1 };
    r.setBasemapPlan([{ dst: base, items: [{ key: 'base-0/1/1/1', src: base, dst: base }, { key: 'base-1/1/1/1', src: base, dst: base }] }]);
    plan('charts', [0, 1, 2].map(i => ({ key: `chart-${i}/2/2/2` })));
    // Both needed basemap tiles are queued first; only one fits, and the charts behind them must still upload.
    r.upload('base-0/1/1/1', await syntheticTile(512, '#0000ff')); r.upload('base-1/1/1/1', await syntheticTile(512, '#0000ff'));
    for (let i = 0; i < 3; i++) r.upload(`chart-${i}/2/2/2`, await syntheticTile(256, '#00ff00'));
    for (let i = 0; i < 4; i++) await tick();
    return { pressure, charts: [0, 1, 2].every(i => r.hasTexture(`chart-${i}/2/2/2`)), base0: r.hasTexture('base-0/1/1/1'), queue: r.queue.length, pixel: pixel() };
  });
  expect(result.charts).toBe(true); expect(result.base0).toBe(true); expect(result.queue).toBe(1);
  expect(result.pressure).toContain('base-1/1/1/1'); expect(result.pixel.slice(0, 3)).toEqual([0, 255, 0]);
});

test('translucent tiles composite with premultiplied alpha', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const c = new OffscreenCanvas(256, 256), ctx = c.getContext('2d'); ctx.fillStyle = 'rgba(255,0,0,0.5)'; ctx.fillRect(0, 0, 256, 256);
    r.upload('half/2/2/2', await createImageBitmap(c, { premultiplyAlpha: 'premultiply' })); plan('half', [{ key: 'half/2/2/2' }]); await tick();
    return pixel();
  });
  // Clear colour (9,12,16) under 50 % red: 0.5*255 + 0.5*9 = 132, 0.5*12 = 6, 0.5*16 = 8.
  expect(Math.abs(result[0] - 132)).toBeLessThanOrEqual(3); expect(result[1]).toBeLessThanOrEqual(8); expect(result[2]).toBeLessThanOrEqual(10);
});

test('airfield dots are ringed like production; the selected dot gains a blue halo; the pin is a ringed translucent disc',async({page})=>{
  const result=await page.evaluate(async()=>{
    r.setAirfields({mx:new Float32Array([.625]),my:new Float32Array([.625]),start:new Uint16Array([1960]),end:new Uint16Array([0]),status:new Uint8Array([0])});
    r.setAirfieldFilter({year:1970,statusMask:7});await tick();
    const radius=r._radius(),at=d=>pixel(128+d,128).slice(0,3);
    const plain={fill:at(0),ring:at(radius),outside:at(radius+3.8)};
    r.setAirfieldSelected(0);await tick();const selected={fill:at(0),halo:at(radius+3.8),picked:r.pick(128,128)};
    r.setAirfieldSelected(null);await tick();const cleared=at(radius+3.8);
    r.setAirfieldSelected(0);r.setAirfields(null);r.setPin({lng:45,lat:-40.97989806962013});await tick();
    const pin={centre:at(0),ring:at(7),outside:at(11)};
    r.setPin(null);await tick();
    return {plain,selected,cleared,pin,gone:at(0),error:r.gl.getError()};
  });
  const background=[9,12,16];
  expect(result.plain.fill).toEqual([54,163,93]);for(const channel of result.plain.ring)expect(channel).toBeGreaterThan(200);
  expect(result.plain.outside).toEqual(background);expect(result.cleared).toEqual(background);
  expect(result.selected.fill).toEqual([54,163,93]);expect(result.selected.picked).toEqual({kind:'airfield',index:0});
  expect(result.selected.halo[2]).toBeGreaterThan(180);expect(result.selected.halo[0]).toBeLessThan(50);
  // #1e90ff: a solid 2px ring around the same blue at 30%.
  for(const [i,channel] of [30,144,255].entries())expect(Math.abs(result.pin.ring[i]-channel)).toBeLessThanOrEqual(8);
  expect(result.pin.centre[2]).toBeGreaterThan(70);expect(result.pin.centre[2]).toBeLessThan(110);expect(result.pin.centre[0]).toBeLessThan(40);
  expect(result.pin.outside).toEqual(background);expect(result.gone).toEqual(background);expect(result.error).toBe(0);
});
