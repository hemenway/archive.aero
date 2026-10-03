#!/usr/bin/env node
import { randomBytes } from 'node:crypto';
import { archive } from './make_fixtures.mjs';
import { memoryHarness } from './serve.mjs';
import { clearDirectoryCache, tileId } from '../../../worker/src/tiles.js';
const tiles=[];for(let y=20;y<25;y++)for(let x=12;x<18;x++)tiles.push({z:6,x,y,bytes:randomBytes(90000)});
tiles.sort((a,b)=>tileId(a.z,a.x,a.y)-tileId(b.z,b.x,b.y));
const bytes=archive(tiles,{leaf:true}),path='sectionals/viewport.0123456789ab';
const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),u64=off=>Number(view.getBigUint64(off,true));
const rootOffset=u64(8),rootLength=u64(16),leafOffset=u64(40),leafLength=u64(48),tileOffset=u64(56);
async function measure(mode) {
  clearDirectoryCache();const h=memoryHarness(new Map([[path+'.pmtiles',bytes]]));let requests=0,delivered=0;
  const send=async(url,headers)=>{requests++;const r=await h.fetch(url,{headers});delivered+=(await r.arrayBuffer()).byteLength;};
  if(mode==='legacy') {
    await send('https://bench.test/'+path+'.pmtiles',{Range:'bytes=0-16383'});
    // Root is embedded in the standard 16KB probe. Explicit leaf read follows.
    await send('https://bench.test/'+path+'.pmtiles',{Range:`bytes=${leafOffset}-${leafOffset+leafLength-1}`});
    for(let i=0;i<tiles.length;i++)await send('https://bench.test/'+path+'.pmtiles',{Range:`bytes=${tileOffset+i*90000}-${tileOffset+(i+1)*90000-1}`});
  }else for(const {z,x,y}of tiles)await send(`https://bench.test/t/${path}/${z}/${x}/${y}`);
  return {requests,r2Reads:h.reads.length,edgeEntries:h.entries.size,deliveredBytes:delivered};
}
console.log(JSON.stringify({tiles:30,tileBytes:90000,archiveBytes:bytes.length,rootOffset,rootLength,
  currentClient:await measure('legacy'),tileEndpoint:await measure('tiles')},null,2));
