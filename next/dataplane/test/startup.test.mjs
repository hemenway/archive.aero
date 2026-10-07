import {test} from 'node:test';
import assert from 'node:assert/strict';
import {startupArtifacts} from '../startup.js';
import {parseManifest,erasAt} from '../manifest.js';
import {createDataPlane} from '../index.js';
import {manifest as raw} from '../../app/stubs/manifest.js';

test('year shards preserve active eras and their paint order at year boundaries',()=>{
  const full=parseManifest(raw),{summary,years}=startupArtifacts(raw);
  assert.equal(summary.eraCount,raw.eras.length);
  for(const [year,shard] of years) {
    const partial=parseManifest(shard);
    for(const date of [`${year}-01-01`,`${year}-06-01`,`${year}-12-31`])
      assert.deepEqual(erasAt(partial,date).map(i=>partial.paths[i]),erasAt(full,date).map(i=>full.paths[i]));
  }
});

test('startup defers the full index, then a historical jump upgrades it once',async()=>{
  const {summary,years}=startupArtifacts(raw),latest=summary.dateBounds.max.slice(0,4);
  const index={version:1,summary,years:Object.fromEntries([...years.keys()].map(y=>[y,`./${y}.json`]))};
  const requests=[];
  const fetcher=async url=>{
    const name=new URL(url).pathname.split('/').pop();requests.push(name);
    return Response.json(name==='startup.json'?index:name==='full.json'?raw:years.get(name.slice(0,4)));
  };
  const dp=await createDataPlane({worker:false,fetch:fetcher,manifestUrl:'https://fixture.test/full.json',startupUrl:'https://fixture.test/startup.json'});
  assert.deepEqual(dp.manifest,summary);
  assert.deepEqual(requests,['startup.json',`${latest}.json`]);
  const date=summary.dateBounds.min;
  await Promise.all([dp.ensureDate(date),dp.loadArchive()]);
  assert.equal(requests.filter(n=>n==='full.json').length,1);
  assert.equal(dp.eraCountAt(date),erasAt(parseManifest(raw),date).length);
  dp.destroy();
});

test('an unavailable startup shard falls back to the complete offline index',async()=>{
  const dp=await createDataPlane({worker:false,startupUrl:'https://fixture.test/startup.json',manifestUrl:'https://fixture.test/full.json',
    fetch:async url=>url.endsWith('full.json')?Response.json(raw):new Response(null,{status:503})});
  assert.equal(dp.manifest.eraCount,raw.eras.length);dp.destroy();
});
