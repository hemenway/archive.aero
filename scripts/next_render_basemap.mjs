#!/usr/bin/env node
import { createServer } from 'node:http';
import { readFile, open, mkdir, writeFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, relative } from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const args=process.argv.slice(2),arg=(k,d)=>args.includes(k)?args[args.indexOf(k)+1]:d;
for(const k of ['--source','--assets','--out','--jobs'])if(!args.includes(k))throw Error('required '+k);
const modules=resolve(arg('--modules','node_modules')),require=createRequire(pathToFileURL(resolve(modules,'../package.json')));
const { chromium }=require('@playwright/test');
const { layers,namedFlavor }=await import(pathToFileURL(resolve(modules,'@protomaps/basemaps/dist/esm/index.js')));
const source=await open(resolve(arg('--source')),'r'),info=await source.stat(),assets=resolve(arg('--assets')),out=resolve(arg('--out'));
const errors=[];let browser,server;
try {
  server=createServer(async(req,res)=>{
    try {
      const url=new URL(req.url,'http://local');
      if(url.pathname==='/vector.pmtiles') {
        const match=(req.headers.range||'').match(/^bytes=(\d+)-(\d+)$/);
        if(!match){res.writeHead(400);res.end();return;}
        const start=Number(match[1]),end=Math.min(Number(match[2]),info.size-1);
        if(start>end){res.writeHead(416);res.end();return;}
        const b=Buffer.alloc(end-start+1);await source.read(b,0,b.length,start);
        res.writeHead(206,{'content-range':`bytes ${start}-${end}/${info.size}`,'content-length':b.length,'accept-ranges':'bytes','etag':'"local-immutable"'});res.end(b);return;
      }
      if(url.pathname.startsWith('/maplibre/')) {
        const name=url.pathname.slice(10);if(!/^[a-z0-9.-]+$/.test(name))throw Error('unsafe module name');
        res.setHeader('content-type',name.endsWith('.css')?'text/css':'text/javascript');
        res.end(await readFile(resolve(modules,'maplibre-gl/dist',name)));return;
      }
      const scripts={'/pmtiles.js':'pmtiles/dist/pmtiles.js'};
      if(scripts[url.pathname]){res.setHeader('content-type','text/javascript');res.end(await readFile(resolve(modules,scripts[url.pathname])));return;}
      if(url.pathname.startsWith('/assets/')) {
        const path=resolve(assets,decodeURIComponent(url.pathname.slice(8)));
        if(relative(assets,path).startsWith('..'))throw Error('invalid asset path');
        res.end(await readFile(path));return;
      }
      res.setHeader('content-type','text/html');res.end('<!doctype html><style>html,body{margin:0}#map{width:768px;height:768px}</style><div id="map"></div><link rel="stylesheet" href="/maplibre/maplibre-gl.css"><script src="/pmtiles.js"></script><script type="module">import * as ml from "/maplibre/maplibre-gl.mjs";window.maplibregl=ml;</script>');
    } catch(e){res.writeHead(404);res.end(String(e));}
  });
  await new Promise(r=>server.listen(0,'127.0.0.1',r));const base=`http://127.0.0.1:${server.address().port}`;
  browser=await chromium.launch({headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  const page=await browser.newPage({viewport:{width:768,height:768},deviceScaleFactor:1});
  page.on('pageerror',e=>errors.push(e.message));
  // Prevent all network access beyond our local source and mirrored style assets.
  await page.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
  await page.goto(base);
  await page.waitForFunction(()=>!!window.maplibregl);
  const style={version:8,glyphs:base+'/assets/fonts/{fontstack}/{range}.pbf',sprite:base+'/assets/sprites/v4/dark',
    sources:{protomaps:{type:'vector',url:'pmtiles://'+base+'/vector.pmtiles',attribution:'© OpenStreetMap contributors · Protomaps'}},
    layers:layers('protomaps',namedFlavor('dark'),{lang:'en'})};
  await page.evaluate(async({style})=>{
    maplibregl.addProtocol('pmtiles',new pmtiles.Protocol().tile);
    window.renderErrors=[];
    window.map=new maplibregl.Map({container:'map',style,center:[0,0],zoom:0,pixelRatio:1,
      interactive:false,attributionControl:false,fadeDuration:0,preserveDrawingBuffer:true,renderWorldCopies:true});
    map.on('error',e=>renderErrors.push(e.error?.message||String(e)));
    await new Promise((r,j)=>{const timer=setTimeout(()=>j(Error('initial render timed out')),30000);map.once('idle',()=>{clearTimeout(timer);r();});});
  },{style});
  const initialErrors=await page.evaluate(()=>renderErrors);
  if(initialErrors.length)throw Error(initialErrors.join('; '));
  let count=0,total=0;const started=performance.now();
  const jobs=createInterface({input:createReadStream(resolve(arg('--jobs'))),crlfDelay:Infinity});
  for await(const line of jobs) {
    if(!line.trim())continue;const [z,x,y]=JSON.parse(line),n=2**z;
    const file=resolve(out,'tiles',String(z),String(x),y+'.webp');
    // Use an input-specific output directory; every requested tile is rendered again.
    const lng=(x+.5)/n*360-180,lat=Math.atan(Math.sinh(Math.PI*(1-2*(y+.5)/n)))*180/Math.PI;
    const encoded=await page.evaluate(async({lng,lat,z})=>{
      renderErrors.length=0;
      await new Promise((r,j)=>{const timer=setTimeout(()=>j(Error('tile render timed out')),30000);map.once('idle',()=>{clearTimeout(timer);r();});map.jumpTo({center:[lng,lat],zoom:z});map.triggerRepaint();});
      if(renderErrors.length)throw Error(renderErrors.join('; '));
      // Render a 128px gutter for neighboring geometry/label candidates, then crop.
      const canvas=document.createElement('canvas');canvas.width=canvas.height=512;
      canvas.getContext('2d').drawImage(map.getCanvas(),128,128,512,512,0,0,512,512);
      const blob=await new Promise(r=>canvas.toBlob(r,'image/webp',.8));
      if(!blob||blob.type!=='image/webp')throw Error('WebP encoding unavailable');
      return Array.from(new Uint8Array(await blob.arrayBuffer()));
    },{lng,lat,z});
    if(errors.length)throw Error(errors.join('; '));
    await mkdir(resolve(file,'..'),{recursive:true});await writeFile(file,Buffer.from(encoded));count++;total+=encoded.length;
    if(count%100===0)console.log(JSON.stringify({tiles:count,bytes:total,seconds:(performance.now()-started)/1000}));
  }
  console.log(JSON.stringify({tiles:count,bytes:total,seconds:(performance.now()-started)/1000,bytesPerTile:total/count}));
} finally {await browser?.close();if(server)await new Promise(r=>server.close(r));await source.close();}
