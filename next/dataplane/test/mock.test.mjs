import test from 'node:test';
import assert from 'node:assert/strict';
import {startServer} from '../bench/server.mjs';
test('mock C1 status, metadata, HEAD, gzip and immutable cache semantics',async()=>{
 const s=await startServer({port:0,latency:0,mbps:1000});try {
 const m=await(await fetch(s.origin+'/next/manifest.json')).json();const e=m.eras[0],base=s.origin+'/t/sectionals/'+e.k+'.'+e.h;
 const r=await fetch(base+'/4/0/0');assert.equal(r.status,200);assert.match(r.headers.get('cache-control'),/immutable/);assert.equal(r.headers.get('content-type'),'image/png');assert.ok((await r.arrayBuffer()).byteLength>0);
 assert.equal((await fetch(base+'/3/0/0')).status,204);assert.equal((await fetch(base+'/4/16/0')).status,400);assert.equal((await fetch(s.origin+'/t/sectionals/unknown.0123456789ab/4/0/0')).status,404);
 const head=await fetch(base+'/4/0/0',{method:'HEAD'});assert.equal(head.status,200);assert.equal((await head.arrayBuffer()).byteLength,0);
 const meta=await fetch(s.origin+'/t/'+m.airspace.p+'/metadata');assert.equal(meta.status,200);assert.equal((await meta.json()).archive_aero.cycle_days,28);
 const vector=await fetch(s.origin+'/t/'+m.airspace.p+'/4/0/0');assert.equal(vector.headers.get('content-encoding'),'gzip');assert.equal((await vector.arrayBuffer()).byteLength,0);
 }finally{await s.close();}
});
