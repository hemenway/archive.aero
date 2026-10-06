import http from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {png} from '../../contract/fixtures/make_fixtures.mjs';
import {gzipSync} from 'node:zlib';
const root=path.resolve(fileURLToPath(new URL('../../../',import.meta.url)));
export const HASH='0123456789ab';
export function syntheticManifest(origin) {
 const eras=[];
 for(let frame=0;frame<31;frame++) for(let area=0;area<8;area++) {
  const start=new Date(Date.UTC(1950,frame*4,1)).toISOString().slice(0,10),end=new Date(Date.UTC(1950,(frame+1)*4,1)).toISOString().slice(0,10);
  const x=235+area,w=x/1024*360-180,e=(x+1)/1024*360-180;
  // Unique valid era keys; stagger the exclusive end while sharing exactly 31 frame starts.
  const s=start,en=new Date(Date.parse(end)+area*86400000).toISOString().slice(0,10);
  eras.push({k:`${s}_to_${en}`,h:HASH,z:[4,11],b:[w,30,e,37],c:null});
 }
 return {version:1,generated:'2026-10-03T00:00:00Z',tileBase:`${origin}/t/`,fileBase:`${origin}/`,eras,
  basemap:{p:`basemap/mock.${HASH}`,z:[0,13],tileSize:512},airspace:{p:`airspace/mock.${HASH}`,z:[0,11]},airfields:{bin:`next/airfields.${HASH}.bin`,details:`next/airfields.${HASH}.json`},pins:{z:5,margin:2,base:`next/pins.${HASH}/`,shards:[396]},coverage:{segments:[]}};
}
// Deterministic noise keeps this a ~100 KB stress tile that deflate cannot shrink.
let seed=1234;const raster=png([0,80,150],256,(x,y,p)=>{seed=(seed*1664525+1013904223)>>>0;p[0]=seed>>>24;p[1]=80;p[2]=150;p[3]=255;});
const magic=Buffer.alloc(36);magic.write('AAAF1');magic.writeUInt32LE(1,8);magic.writeFloatLE(.25,16);magic.writeFloatLE(.4,20);magic.writeUInt16LE(1940,24);magic.writeUInt16LE(1970,28);magic[32]=1;
export async function startServer({port=4182,latency=Number(process.env.LATENCY??80),mbps=Number(process.env.MBPS??10)}={}) {
 const metrics={requests:0,bytes:0,cancelled:0},active=new Set();
 // One shared bandwidth budget, across all simultaneous responses.
 const timer=setInterval(()=>{let budget=mbps*1e6/8*.01;const jobs=[...active],share=Math.max(1,Math.floor(budget/jobs.length));for(const job of jobs) {if(budget<=0)break;const count=Math.min(job.bytes.length-job.offset,share);const chunk=job.bytes.subarray(job.offset,job.offset+count);job.response.write(chunk);job.offset+=count;budget-=count;metrics.bytes+=count;if(job.offset===job.bytes.length){active.delete(job);job.response.end();}}},10);
 const server=http.createServer(async(req,res)=>{
  const url=new URL(req.url,`http://${req.headers.host}`),origin=url.origin;
  res.setHeader('Access-Control-Allow-Origin','*');
  if(url.pathname==='/metrics') {res.setHeader('Content-Type','application/json');res.end(JSON.stringify(metrics));return;}
  if(url.pathname==='/reset') {metrics.requests=metrics.bytes=metrics.cancelled=0;res.end();return;}
  if(url.pathname==='/next/manifest.json') {res.setHeader('Content-Type','application/json');res.end(JSON.stringify(syntheticManifest(origin)));return;}
  if(url.pathname.startsWith('/t/')) {
   const parts=url.pathname.slice(3).split('/');const meta=parts.at(-1)==='metadata';const archive=meta?parts.slice(0,-1).join('/'):parts.slice(0,-3).join('/');
   const manifest=syntheticManifest(origin);
   const entry=manifest.eras.find(e=>`sectionals/${e.k}.${e.h}`===archive);
   const zoom=entry?.z??(archive===manifest.basemap.p?manifest.basemap.z:archive===manifest.airspace.p?manifest.airspace.z:['sectionals/slow.0123456789ab','sectionals/absent.0123456789ab'].includes(archive)?[4,11]:null);
   if(!zoom) {res.writeHead(404,{'Cache-Control':'public, max-age=60'});res.end();return;}
   if(meta) {res.setHeader('Cache-Control','public, max-age=31536000, immutable');res.setHeader('Content-Type','application/json');res.end(JSON.stringify({archive_aero:{cycle_days:28,regions:{us:{name:'US',source:'FAA NASR',cycles:['1950-01-01'],boxes:[[-180,-85,180,85]]}}}}));return;}
   const [z,x,y]=parts.slice(-3).map(Number),n=2**z;
   if(!parts.slice(-3).every(p=>/^\d+$/.test(p))||![z,x,y].every(Number.isInteger)||z<0||z>24||x<0||y<0||x>=n||y>=n){res.writeHead(400);res.end();return;}
   metrics.requests++;res.setHeader('Cache-Control','public, max-age=31536000, immutable');
   const vector=archive.startsWith('airspace/'),bytes=vector?gzipSync(Buffer.alloc(0)):raster;
   res.setHeader('Content-Type',vector?'application/vnd.mapbox-vector-tile':'image/png');if(vector)res.setHeader('Content-Encoding','gzip');
   let job;const delay=setTimeout(()=>{
    if(req.method==='HEAD'){res.end();return;}
    if(archive.includes('/absent.')||z<zoom[0]||z>zoom[1]){res.writeHead(204);res.end();return;}
    job={response:res,bytes,offset:0};active.add(job);
   },archive.includes('/slow.')?2000:latency);
   res.on('close',()=>{clearTimeout(delay);if(!res.writableFinished){metrics.cancelled++;if(job)active.delete(job);}});return;
  }
  if(url.pathname.endsWith('.bin')){res.end(magic);return;}
  if(url.pathname.endsWith('.json') && url.pathname.includes('/airfields.')){res.setHeader('Content-Type','application/json');res.end(JSON.stringify([{name:'Mock Field',last_known_year:1969}]));return;}
  if(url.pathname.endsWith('.json') && url.pathname.includes('/pins.')){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({locations:{},rings:{}}));return;}
  const target=path.resolve(root,'.'+(url.pathname==='/'?'/next/dataplane/browser/index.html':url.pathname));
  if(!target.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
  try {const bytes=await readFile(target);res.setHeader('Content-Type',target.endsWith('.js')||target.endsWith('.mjs')?'text/javascript':target.endsWith('.html')?'text/html':'application/octet-stream');res.end(bytes);}catch {res.writeHead(404);res.end();}
 });
 await new Promise(resolve=>server.listen(port,'127.0.0.1',resolve));
 return {server,metrics,origin:`http://127.0.0.1:${server.address().port}`,tileBytes:raster.length,close:async()=>{clearInterval(timer);for(const j of active)j.response.destroy();await new Promise(r=>server.close(r));}};
}
if(process.argv[1]===fileURLToPath(import.meta.url)) {const s=await startServer();console.log(`Mock C1/C2 at ${s.origin}`);}
