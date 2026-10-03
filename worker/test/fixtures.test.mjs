import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp,readFile,readdir,rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join,relative } from 'node:path';
import { createHash } from 'node:crypto';
import { makeFixtures } from '../../next/contract/fixtures/make_fixtures.mjs';
import { memoryHarness } from '../../next/contract/fixtures/serve.mjs';
import { validateManifest,smoke } from '../../next/contract/validate.mjs';
import { clearDirectoryCache } from '../src/tiles.js';
test('canonical fixtures generate, match archive hashes, and pass the shared Worker validator',async t=>{
  const dir=await mkdtemp(join(tmpdir(),'archive-next-fixtures-'));
  const previous=globalThis.caches;clearDirectoryCache();t.after(async()=>{globalThis.caches=previous;clearDirectoryCache();await rm(dir,{recursive:true,force:true});});
  const result=await makeFixtures(dir,'https://fixtures.test/'),objects=new Map();
  async function visit(path){for(const entry of await readdir(path,{withFileTypes:true})){const p=join(path,entry.name);if(entry.isDirectory())await visit(p);else objects.set(relative(dir,p),await readFile(p));}}
  await visit(dir);const m=validateManifest(JSON.parse(objects.get(result.manifest)));
  assert.equal(m.eras.length,4);assert.ok(m.eras.some(e=>e.b===null));assert.deepEqual(m.pins.shards,[391,392]);
  for(const [key,bytes] of objects)if(key.endsWith('.pmtiles'))assert.equal(key.split('.').at(-2),createHash('sha256').update(bytes).digest('hex').slice(0,12));
  const h=memoryHarness(objects);await smoke(m,h.fetch);
  const af=objects.get(m.airfields.bin);assert.equal(af.subarray(0,8).toString(),'AAAF1\0\0\0');assert.equal(af.readUInt32LE(8),21);
  assert.equal(JSON.parse(objects.get(m.airfields.details)).length,21);
  const tile=await h.fetch(m.tileBase+m.airspace.p+'/7/32/32');assert.equal(tile.status,200);assert.equal(tile.headers.get('content-encoding'),'gzip');
  h.env.ALLOWED_ORIGIN='https://legacy.test';
  const preflight=await h.fetch(m.tileBase+m.airspace.p+'/7/32/32',{method:'OPTIONS'});assert.equal(preflight.status,204);assert.equal(preflight.headers.get('access-control-allow-origin'),'*');
});
