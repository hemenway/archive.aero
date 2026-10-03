#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { IMMUTABLE } from '../../worker/src/tiles.js';
const schema = JSON.parse(await readFile(new URL('./manifest.schema.json',import.meta.url),'utf8'));
// The schema deliberately uses this small, dependency-free JSON Schema subset.
// Fail closed on unknown validation keywords so schema changes cannot skip checks.
const supported = new Set(['$schema','$id','title','description','type','const','enum','anyOf','properties','required','additionalProperties','items','minItems','maxItems','uniqueItems','minimum','maximum','pattern','format']);
export function validateSchema(value, s, at='$') {
  for (const k of Object.keys(s)) if (!supported.has(k)) throw Error('unsupported schema keyword '+k);
  const fail = msg => { throw Error(`${at}: ${msg}`); };
  if (s.anyOf) { for (const option of s.anyOf) { try { validateSchema(value,option,at); return; } catch {} } fail('no anyOf match'); }
  const kind = value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value;
  if (s.type && !(s.type==='integer' ? Number.isSafeInteger(value) : kind === s.type)) fail('expected '+s.type);
  if ('const' in s && value !== s.const) fail('incorrect constant');
  if (s.enum && !s.enum.includes(value)) fail('outside enum');
  if (kind==='number') { if (!Number.isFinite(value) || value < (s.minimum ?? -Infinity) || value > (s.maximum ?? Infinity)) fail('outside numeric bounds'); }
  if (kind==='string') {
    if (s.pattern && !new RegExp(s.pattern).test(value)) fail('pattern mismatch');
    if (s.format==='uri') { let u; try { u=new URL(value); } catch { fail('invalid URI'); } if (!['http:','https:'].includes(u.protocol)) fail('URI must be HTTP(S)'); }
    if (s.format==='date-time' && (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/.test(value) || !Number.isFinite(Date.parse(value)))) fail('invalid UTC datetime');
  }
  if (kind==='array') {
    if (value.length < (s.minItems ?? 0) || value.length > (s.maxItems ?? Infinity)) fail('incorrect array length');
    if (s.uniqueItems && new Set(value.map(v=>JSON.stringify(v))).size!==value.length) fail('duplicate items');
    if (s.items) value.forEach((v,i)=>validateSchema(v,s.items,`${at}[${i}]`));
  }
  if (kind==='object') {
    for (const k of s.required || []) if (!(k in value)) fail('missing '+k);
    for (const [k,v] of Object.entries(value)) {
      if (s.properties?.[k]) validateSchema(v,s.properties[k],at+'.'+k);
      else if (s.additionalProperties===false) fail('unknown property '+k);
    }
  }
}
function isoDate(s) { return /^\d{4}-\d{2}-\d{2}$/.test(s) && new Date(s+'T00:00:00Z').toISOString().slice(0,10)===s; }
export function validateManifest(m) {
  validateSchema(m,schema);
  const sorted = (arr,label) => { if (arr.some((x,i)=>i>0 && x<=arr[i-1])) throw Error(label+' must be sorted and unique'); };
  let previous=''; const keys=new Set();
  for (const e of m.eras) {
    const [start,end]=e.k.split('_to_'); if (!isoDate(start)||!isoDate(end)||start>=end) throw Error('invalid era dates '+e.k);
    if (e.k<previous || keys.has(e.k)) throw Error('eras must be sorted and unique'); previous=e.k; keys.add(e.k);
    if (e.z[0]>e.z[1]) throw Error('inverted zooms');
    if (e.b && !(e.b[0]<e.b[2] && e.b[1]<e.b[3] && e.b[1]>=-86 && e.b[3]<=86)) throw Error('invalid era bounds');
    if (e.c) sorted(e.c,'coverage');
  }
  for (const a of [m.basemap,m.airspace]) if (a && a.z[0]>a.z[1]) throw Error('inverted overlay zooms');
  if (m.pins) sorted(m.pins.shards,'pin shards');
  return m;
}
export async function smoke(m, fetcher=fetch, baseOverride=null) {
  const base=baseOverride ? new URL('/t/',baseOverride).href : m.tileBase;
  async function get(suffix,status,method='GET') {
    const r=await fetcher(new URL(suffix,base),{method});
    if (r.status!==status) throw Error(`${method} ${suffix}: ${r.status}, expected ${status}`);
    if (r.headers.get('access-control-allow-origin')!=='*') throw Error('CORS missing');
    if ([200,204].includes(status) && r.headers.get('cache-control')!==IMMUTABLE) throw Error('immutable caching missing');
    if (status===404 && r.headers.get('cache-control')!=='public, max-age=60') throw Error('404 TTL missing');
    if (method==='HEAD' && (await r.arrayBuffer()).byteLength!==0) throw Error('HEAD body');
    return r;
  }
  for (const e of m.eras.slice(0,4)) {
    const path='sectionals/'+e.k+'.'+e.h;
    const meta=await get(path+'/metadata',200);
    if (meta.headers.get('content-type')!=='application/json') throw Error('metadata content type'); await meta.json();
    await get(path+'/metadata',200,'HEAD'); await get(path+'/25/0/0',400);
    await get(path+'/6/64/0',400);
    const absentZoom=e.z[1]<24?24:e.z[0]>0?0:null;
    if (absentZoom!==null) await get(path+`/${absentZoom}/0/0`,204);
    if (e.z[0]<=6 && e.z[1]>=6 && e.c?.length) {
      const i=e.c[0], suffix=path+`/6/${i%64}/${Math.floor(i/64)}`;
      // c may come from finer tiles in a sparse, pre-migration archive.
      const tile=await fetcher(new URL(suffix,base));
      if (![200,204].includes(tile.status)) throw Error('tile lookup failed');
      if (tile.headers.get('cache-control')!==IMMUTABLE || tile.headers.get('access-control-allow-origin')!=='*') throw Error('tile headers');
      if (tile.status===200) {
        if (!['image/png','image/webp','image/jpeg','image/avif','application/vnd.mapbox-vector-tile'].includes(tile.headers.get('content-type'))) throw Error('tile type');
        const head=await get(suffix,200,'HEAD');
        for (const k of ['content-type','content-length','content-encoding']) if (head.headers.get(k)!==tile.headers.get(k)) throw Error('HEAD header '+k);
        if ((await tile.arrayBuffer()).byteLength===0) throw Error('empty tile');
      }
    }
  }
  await get('sectionals/unknown.000000000000/6/1/1',404);
}
async function main() {
  const args=process.argv.slice(2), value=k=>args[args.indexOf(k)+1];
  if (!args.includes('--manifest')) throw Error('usage: validate.mjs --manifest FILE_OR_URL [--base URL] [--schema-only] [--allow-over-budget]');
  const ref=value('--manifest'); const data=/^https?:/.test(ref)?Buffer.from(await (await fetch(ref)).arrayBuffer()):await readFile(ref);
  const m=validateManifest(JSON.parse(data)); const size=gzipSync(data).length;
  if (size>80000 && !args.includes('--allow-over-budget')) throw Error(`gzip budget exceeded: ${size}`);
  if (!args.includes('--schema-only')) await smoke(m,fetch,args.includes('--base')?value('--base'):null);
  console.log(`OK: ${m.eras.length} eras; ${size} bytes gzip${args.includes('--schema-only')?'; schema only':'; C1 smoke passed'}`);
}
if (process.argv[1] && fileURLToPath(import.meta.url)===fileURLToPath(new URL('file://'+process.argv[1]))) main().catch(e=>{console.error(e.message);process.exitCode=1;});
