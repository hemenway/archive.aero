#!/usr/bin/env node
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import worker from '../../../worker/src/index.js';
import { clearDirectoryCache } from '../../../worker/src/tiles.js';
export function memoryHarness(objects, origin='https://fixture.test') {
  const entries=new Map(), reads=[],heads=[];
  const cache={async match(req){return entries.get(req.url)?.clone();},async put(req,res){entries.set(req.url,new Response(await res.arrayBuffer(),res));}};
  const env={SAMPLE_RATE:'0',BUCKET:{
    async head(key){heads.push(key);const b=objects.get(key);return b?metadata(b,key):null;},
    async get(key,{range}={}) {
      reads.push({key,range});const all=objects.get(key);if(!all)return null;
      const offset=range?.offset??0;if(offset>=all.length)throw Error('range not satisfiable');
      const bytes=all.subarray(offset,offset+(range?.length??all.length));
      return {...metadata(all,key),body:new Response(bytes).body,arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};
    }
  }};
  function metadata(bytes,key){return {size:bytes.length,httpEtag:'"fixture-'+bytes.length+'"',writeHttpMetadata(h){h.set('content-type',key.endsWith('.json')?'application/json':'application/octet-stream');}};}
  async function fetcher(input,init={}) {
    const pending=[];globalThis.caches={default:cache};
    const req=input instanceof Request?input:new Request(new URL(String(input),origin),init);
    const response=await worker.fetch(req,env,{waitUntil(p){pending.push(p);}});
    await Promise.all(pending);return response;
  }
  return {env,entries,reads,heads,fetch:fetcher,cache};
}
export async function serve(directory=fileURLToPath(new URL('./out/',import.meta.url)),port=8765) {
  const root=resolve(directory),entries=new Map(),reads=[];
  clearDirectoryCache();globalThis.caches={default:{async match(req){return entries.get(req.url)?.clone();},async put(req,res){entries.set(req.url,new Response(await res.arrayBuffer(),res));}}};
  function filename(key){const p=resolve(root,key);if(relative(root,p).startsWith('..')||key.startsWith('/'))throw Error('unsafe key');return p;}
  async function metadata(key){try{const path=filename(key),info=await stat(path);if(!info.isFile())return null;return {path,size:info.size,httpEtag:'"fixture-'+info.size+'"',writeHttpMetadata(h){h.set('content-type',key.endsWith('.json')?'application/json':key.endsWith('.bin')?'application/octet-stream':'application/vnd.pmtiles');}};}catch{return null;}}
  const env={SAMPLE_RATE:'0',BUCKET:{head:metadata,async get(key,{range}={}){
    reads.push({key,range});const m=await metadata(key);if(!m)return null;
    // This is a tiny fixture server; full file loading is confined to the local R2 double.
    const all=await readFile(m.path),off=range?.offset??0;if(off>=all.length)throw Error('range not satisfiable');
    const bytes=all.subarray(off,off+(range?.length??all.length));return {...m,body:new Response(bytes).body,arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};
  }}};
  const server=createServer(async(req,res)=>{
    const pending=[];
    try{
      const request=new Request(`http://127.0.0.1:${server.address().port}${req.url}`,{method:req.method,headers:req.headers});
      const response=await worker.fetch(request,env,{waitUntil(p){pending.push(p);}});
      res.writeHead(response.status,Object.fromEntries(response.headers));res.end(Buffer.from(await response.arrayBuffer()));
      await Promise.all(pending);
    }catch(e){res.writeHead(500);res.end(String(e));}
  });
  await new Promise(r=>server.listen(port,'127.0.0.1',r));return {server,reads,entries};
}
if(process.argv[1]===fileURLToPath(import.meta.url)) {
  const h=await serve(process.argv[2],Number(process.argv[3]??8765));console.log(`Fixtures: http://127.0.0.1:${h.server.address().port}/`);
}
