import assert from 'node:assert/strict';
import { test } from 'node:test';
import { gzipSync, gunzipSync } from 'node:zlib';
import { archive, png, airspaceMvt } from '../../next/contract/fixtures/make_fixtures.mjs';
import { memoryHarness } from '../../next/contract/fixtures/serve.mjs';
import { clearDirectoryCache, directoryCacheStats, IMMUTABLE, tileId } from '../src/tiles.js';
const PATH='sectionals/test.0123456789ab';
function harness(t, options={}) {
  const previous=globalThis.caches;clearDirectoryCache();t.after(()=>{globalThis.caches=previous;clearDirectoryCache();});
  const bytes=options.bytes??png([255,0,0]);
  const packed=archive([{z:6,x:15,y:25,bytes}],options);
  const h=memoryHarness(new Map([[PATH+'.pmtiles',packed]]));
  return {...h,bytes,packed,get:(suffix='6/15/25',method='GET')=>h.fetch('https://fixture.test/t/'+PATH+'/'+suffix,{method})};
}
test('tile 200 has every C1 header; legacy archive URL is still a range proxy',async t=>{
  const h=harness(t);h.env.ALLOWED_ORIGIN='https://legacy.test';
  const r=await h.get();assert.equal(r.status,200);assert.equal(r.headers.get('content-type'),'image/png');
  assert.equal(r.headers.get('cache-control'),IMMUTABLE);assert.equal(r.headers.get('access-control-allow-origin'),'*');
  assert.equal(+r.headers.get('content-length'),h.bytes.length);assert.equal(r.headers.get('content-encoding'),null);
  assert.deepEqual(Buffer.from(await r.arrayBuffer()),h.bytes);
  const old=await h.fetch('https://fixture.test/'+PATH+'.pmtiles',{headers:{Range:'bytes=0-126'}});
  assert.equal(old.status,206);assert.equal(old.headers.get('access-control-allow-origin'),'https://legacy.test');
});
test('204 absence/outside zoom, 404 unknown path, 400 malformed coordinates',async t=>{
  const h=harness(t);
  for(const suffix of ['6/14/25','4/0/0','24/0/0']) {const r=await h.get(suffix);assert.equal(r.status,204);assert.equal(r.headers.get('cache-control'),IMMUTABLE);assert.equal((await r.arrayBuffer()).byteLength,0);}
  const missing=await h.fetch('https://fixture.test/t/sectionals/missing.0123456789ab/6/0/0');
  assert.equal(missing.status,404);assert.equal(missing.headers.get('cache-control'),'public, max-age=60');
  for(const suffix of ['25/0/0','-1/0/0','6/64/0','6/0/64','6/1.5/0','06/1/0','6/a/0','6/NaN/0'])assert.equal((await h.get(suffix)).status,400,suffix);
});
test('gzip root and leaf directories resolve tiles, including offset-zero leaves',async t=>{
  const h=harness(t,{leaf:true});const r=await h.get();assert.equal(r.status,200);assert.deepEqual(Buffer.from(await r.arrayBuffer()),h.bytes);
});
test('uncompressed PMTiles directories work',async t=>{const h=harness(t,{compression:1});assert.equal((await h.get()).status,200);});
test('gzip MVT is served as stored, with content encoding',async t=>{
  const h=harness(t,{type:1,gzip:true,bytes:airspaceMvt()});const r=await h.get();
  assert.equal(r.headers.get('content-type'),'application/vnd.mapbox-vector-tile');assert.equal(r.headers.get('content-encoding'),'gzip');
  const data=Buffer.from(await r.arrayBuffer());assert.deepEqual(gunzipSync(data),h.bytes);assert.equal(+r.headers.get('content-length'),data.length);
});
test('metadata is JSON with immutable caching; HEAD carries the same headers and no body',async t=>{
  const h=harness(t,{metadata:{name:'fixture'}});const m=await h.get('metadata');assert.deepEqual(await m.json(),{name:'fixture'});
  assert.equal(m.headers.get('content-type'),'application/json');assert.equal(m.headers.get('cache-control'),IMMUTABLE);
  for(const suffix of ['metadata','6/15/25']) {const head=await h.get(suffix,'HEAD'),get=await h.get(suffix);assert.equal(head.status,200);assert.equal((await head.arrayBuffer()).byteLength,0);
    for(const k of ['content-length','content-type','content-encoding','cache-control','access-control-allow-origin'])assert.equal(head.headers.get(k),get.headers.get(k));}
});
test('parsed directories survive edge eviction; HEAD never reads the tile data block',async t=>{
  const h=harness(t,{leaf:true,padding:1<<20});const r=await h.get();assert.equal(r.status,200);assert.equal(h.reads.length,2);
  const stats=directoryCacheStats();assert.ok(stats.bytes>0&&stats.bytes<=stats.limit);assert.equal(stats.entries,3);
  h.entries.clear();h.reads.length=0;assert.equal((await h.get('6/15/25','HEAD')).status,200);assert.equal(h.reads.length,0);
  assert.equal((await h.get()).status,200);assert.equal(h.reads.length,1);assert.equal(h.reads[0].range.offset,1<<20);
});
test('30 concurrent cold tile lookups coalesce through the existing block flight',async t=>{
  const h=harness(t,{leaf:true});const responses=await Promise.all(Array.from({length:30},()=>h.get()));
  assert.ok(responses.every(r=>r.status===200));assert.equal(h.reads.length,1);assert.equal(h.entries.size,1);
});
test('sampled tile hits retain the Analytics Engine datapoint shape',async t=>{
  const h=harness(t);const points=[];h.env.SAMPLE_RATE='1';h.env.TILES={writeDataPoint:p=>points.push(p)};
  await h.get();await h.get('metadata');assert.equal(points.length,1);assert.equal(points[0].blobs.length,5);assert.equal(points[0].doubles.length,3);assert.equal(points[0].indexes[0],PATH+'.pmtiles');
});
test('storage errors and corrupt archives give retryable responses',async t=>{
  const h=harness(t);t.mock.method(console,'error',()=>{});h.env.BUCKET.get=async()=>{throw Error('offline');};const r=await h.get();assert.equal(r.status,503);assert.equal(r.headers.get('retry-after'),'1');
});
test('Hilbert ids agree with v3 known coordinates',()=>{assert.equal(tileId(0,0,0),0);assert.equal(tileId(1,0,0),1);assert.equal(tileId(1,0,1),2);assert.equal(tileId(1,1,1),3);assert.equal(tileId(1,1,0),4);assert.ok(Number.isSafeInteger(tileId(24,2**24-1,2**24-1)));});
