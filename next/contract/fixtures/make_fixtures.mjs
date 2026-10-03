#!/usr/bin/env node
// Deterministic canonical dataset. Only Node built-ins; no npm dependencies.
import { mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { gzipSync, deflateSync } from 'node:zlib';
import { tileId } from '../../../worker/src/tiles.js';
export const hash = bytes => createHash('sha256').update(bytes).digest('hex').slice(0,12);
export const json = value => Buffer.from(JSON.stringify(value));
export function varint(n) { const bytes=[]; do { let b=n%128; n=Math.floor(n/128); bytes.push(b+(n?128:0)); } while(n); return Buffer.from(bytes); }
export function directory(entries, compressed=true) {
  let last=0; const ids=entries.map(e=>{const b=varint(e.id-last);last=e.id;return b;});
  const raw=Buffer.concat([varint(entries.length),...ids,...entries.map(e=>varint(e.run??1)),
    ...entries.map(e=>varint(e.length)),...entries.map((e,i)=>varint(i&&e.offset===entries[i-1].offset+entries[i-1].length?0:e.offset+1))]);
  return compressed?gzipSync(raw,{mtime:0}):raw;
}
export function archive(tiles,{type=2,gzip=false,leaf=false,compression=2,bounds=[-102,30,-78,42],metadata={},padding=0}={}) {
  const contents=[],entries=[],seen=new Map(); let offset=0;
  for(const t of [...tiles].sort((a,b)=>tileId(a.z,a.x,a.y)-tileId(b.z,b.x,b.y))) {
    const bytes=gzip?gzipSync(t.bytes,{mtime:0}):Buffer.from(t.bytes), h=hash(bytes);
    let off=seen.get(h);
    if(off===undefined) { off=offset;contents.push(bytes);offset+=bytes.length;seen.set(h,off); }
    entries.push({id:tileId(t.z,t.x,t.y),offset:off,length:bytes.length,run:1});
  }
  const d=directory(entries,compression===2), leaves=leaf?d:Buffer.alloc(0);
  const root=leaf?directory([{id:entries[0].id,offset:0,length:d.length,run:0}],compression===2):d;
  const m=compression===2?gzipSync(json(metadata),{mtime:0}):json(metadata), h=Buffer.alloc(127);
  h.write('PMTiles'); h[7]=3;
  const tileOffset=127+root.length+m.length+leaves.length+padding;
  const values=[127,root.length,127+root.length,m.length,127+root.length+m.length,leaves.length,
    tileOffset,offset,entries.length,entries.length,seen.size];
  values.forEach((n,i)=>h.writeBigUInt64LE(BigInt(n),8+i*8));
  h[96]=1;h[97]=compression;h[98]=gzip?2:1;h[99]=type;
  h[100]=Math.min(...tiles.map(t=>t.z));h[101]=Math.max(...tiles.map(t=>t.z));
  bounds.forEach((n,i)=>h.writeInt32LE(Math.round(n*1e7),102+i*4));h[118]=h[100];
  return Buffer.concat([h,root,m,leaves,Buffer.alloc(padding),...contents]);
}
function crc32(bytes) { let crc=0xffffffff; for(const b of bytes) { crc^=b;for(let j=0;j<8;j++)crc=(crc>>>1)^((crc&1)?0xedb88320:0); } return (crc^0xffffffff)>>>0; }
function chunk(type,data) { const t=Buffer.from(type),h=Buffer.alloc(4),c=Buffer.alloc(4);h.writeUInt32BE(data.length);c.writeUInt32BE(crc32(Buffer.concat([t,data])));return Buffer.concat([h,t,data,c]); }
// pixel(x,y,rgba) may overwrite the default fixture pattern per pixel.
export function png(color,size=256,pixel=null) {
  const raw=Buffer.alloc((size*4+1)*size);
  for(let y=0;y<size;y++)for(let x=0;x<size;x++) {
    const off=y*(size*4+1)+1+x*4; raw[off]=color[0];raw[off+1]=color[1];raw[off+2]=color[2];
    raw[off+3]=(x<4||y<4||x>=size-4||y>=size-4)?0:255;
    // A light diagonal gives fixtures a recognizable spatial pattern.
    if(Math.abs(x-y)<3)raw[off]=raw[off+1]=raw[off+2]=255;
    if(pixel)pixel(x,y,raw.subarray(off,off+4));
  }
  const ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(size);ihdr.writeUInt32BE(size,4);ihdr[8]=8;ihdr[9]=6;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',ihdr),chunk('IDAT',deflateSync(raw)),chunk('IEND',Buffer.alloc(0))]);
}
const field=(n,b)=>Buffer.concat([varint(n*8+2),varint(b.length),b]);
const integer=(n,v)=>Buffer.concat([varint(n*8),varint(v)]);
function layer(name,features,geometry={}) {
  const keys=[],values=[];const fs=[];
  for(const [index,p] of features.entries()) {
    const tags=[];
    for(const [k,v] of Object.entries(p)) {
      let ki=keys.indexOf(k);if(ki<0){ki=keys.length;keys.push(k);}
      const vi=values.length;values.push(typeof v==='number'?integer(5,v):field(1,Buffer.from(v)));
      tags.push(varint(ki),varint(vi));
    }
    const x=geometry.x??150+index*480,y=geometry.y??600+(index%2)*1000,w=geometry.width??380,h=geometry.height??750,zig=n=>n<0?-n*2-1:n*2;
    const geom=geometry.type===1?Buffer.concat([varint(9),varint(zig(x)),varint(zig(y))]):Buffer.concat([varint(9),varint(zig(x)),varint(zig(y)),varint(26),
      varint(w*2),varint(0),varint(0),varint(h*2),varint(zig(-w)),varint(0),varint(15)]);
    fs.push(field(2,Buffer.concat([integer(1,index+1),field(2,Buffer.concat(tags)),integer(3,geometry.type??3),field(4,geom)])));
  }
  return field(3,Buffer.concat([field(1,Buffer.from(name)),...fs,...keys.map(k=>field(3,Buffer.from(k))),...values.map(v=>field(4,v)),integer(5,4096),integer(15,2)]));
}
export function airspaceMvt() {
  const common={from:19400101,to:20300101,rg:'us',name:'Fixture airspace'};
  return Buffer.concat([layer('class',['A','B','C','D','E',null].map((cls,i)=>({...common,rg:['us','fr','br'][i%3],...(cls?{cls}:{}),lt:cls==='E'?'E2':'CTR',v:'fixture'+i}))),
    layer('efloor',[{...common,k:'700',lt:'E5',ex:1},{...common,k:'1200',lt:'E6',ex:1}])]);
}
function airfields() {
  const n=21,out=Buffer.alloc(16+8*n+2*n+2+2*n+2+n); out.write('AAAF1');out.writeUInt32LE(n,8);
  let mx=16,my=mx+4*n,start=my+4*n,end=(start+2*n+3)&~3,status=(end+2*n+3)&~3;
  const details=[]; const spans=[[0,0],[1940,0],[0,1970],[1940,1970],[1940,1950],[1980,0],[0,0]];
  for(let i=0;i<n;i++) {
    const [s,e]=spans[i%7],st=Math.floor(i/7),lon=-98+(i%7)*.5,lat=35+Math.floor(i/7)*.5;
    out.writeFloatLE((lon+180)/360,mx+4*i);out.writeFloatLE((1-Math.asinh(Math.tan(lat*Math.PI/180))/Math.PI)/2,my+4*i);
    out.writeUInt16LE(s,start+2*i);out.writeUInt16LE(e,end+2*i);out[status+i]=st;
    details.push({name:`Fixture field ${i}`,status:['open','gone','unknown'][st],start_year:s||null,end_year:e||null,last_known_year:i%7===6?1988:null,
      state:'TX',rel_location:'Near fixture town',url:'https://example.com/field/'+i,oa:'FIX'+i});
  }
  return {bytes:out.subarray(0,status+n),details};
}
export async function makeFixtures(out=new URL('./out/',import.meta.url),base='http://127.0.0.1:8765/') {
  const dir=typeof out==='string'?out:fileURLToPath(out);await mkdir(dir,{recursive:true});
  async function put(path,bytes) { const url=new URL(path,'file://'+dir.replace(/\/$/,'')+'/');await mkdir(fileURLToPath(new URL('.',url)),{recursive:true});await writeFile(url,bytes); }
  async function packed(stem,tiles,options) { const bytes=archive(tiles,options),p=stem+'.'+hash(bytes);await put(p+'.pmtiles',bytes);return p; }
  const keys=['1948-01-01_to_1952-01-01','1950-01-01_to_1954-01-01','1950-07-01_to_1955-01-01','1953-01-01_to_1957-01-01'];
  const colors=[[205,70,64],[48,153,194],[228,170,40],[115,79,180]],eras=[];let representative;
  for(let i=0;i<4;i++) {
    const tiles=[]; const cover=new Set();
    for(let z=4;z<=11;z++)for(let j=0;j<2;j++) {
      const x=i===2?(j?2**z-1:0):Math.floor(2**z*((-95.625+i*4+180)/360))+j;
      const y=Math.floor(2**z*((1-Math.asinh(Math.tan(36.5*Math.PI/180))/Math.PI)/2));
      tiles.push({z,x,y,bytes:png(colors[i])}); if(z===6)cover.add(y*64+x);
    }
    const p=await packed('sectionals/'+keys[i],tiles,{leaf:i===1,bounds:i===2?[170,30,-170,42]:[-102,30,-78,42],metadata:{name:'Fixture era '+i,format:'png'}});
    eras.push({k:keys[i],h:p.split('.').at(-1),b:i===2?null:[-102,30,-78,42],z:[4,11],c:[...cover].sort((a,b)=>a-b)});
    representative={p,tiles:tiles.map(({z,x,y})=>({z,x,y}))};
  }
  const airspace=await packed('airspace/fixture',[...Array.from({length:8},(_,i)=>({z:i+4,x:2**(i+4)>>2,y:2**(i+4)>>2,bytes:airspaceMvt()})),
    {z:0,x:0,y:0,bytes:airspaceMvt()}],{type:1,gzip:true,leaf:true,metadata:{vector_layers:[{id:'class'},{id:'efloor'}],cycles:{us:['1940-01-01'],fr:['1940-01-01'],br:['1940-01-01']}}});
  const basemap=await packed('basemap/fixture',Array.from({length:14},(_,z)=>({z,x:0,y:0,bytes:png([28,33,42],512)})),{metadata:{name:'Fixture dark basemap'}});
  const af=airfields(),afh=hash(Buffer.concat([af.bytes,json(af.details)]));await put(`next/airfields.${afh}.bin`,af.bytes);await put(`next/airfields.${afh}.json`,json(af.details));
  const chartP=await packed('sectionals/chart/fixture/1950-01-01',[{z:8,x:60,y:100,bytes:png(colors[0])}],{metadata:{name:'Fixture solo'}});
  const shards={};
  for(const [i,lon] of [[391,-95.625],[392,-84.375]]) {
    const ref='sectional/fixture-'+i,ring=[[lon-.3,36.2],[lon+.3,36.2],[lon+.3,36.8],[lon-.3,36.8],[lon-.3,36.2]];
    shards[i]={locations:{['Fixture '+i]:{era:'modern',ref,charts:[{d:'1950-01-01',e:'1954-01-01',ed:'1',f:'',pm:chartP,pmz:[8,8],pmb:[-102,30,-78,42]}]}},rings:{[ref]:[ring]}};
  }
  const pinh=hash(json(shards)); for(const [i,shard] of Object.entries(shards))await put(`next/pins.${pinh}/${i}.json`,json(shard));
  const coverage={generated:'2026-10-03',ref:[-125,24.5,-66.5,49.5],segments:[['1948-01-01','1950-01-01',1,25],['1950-01-01','1957-01-01',4,90]]};
  const manifest={version:1,generated:'2026-10-03T00:00:00Z',tileBase:new URL('t/',base).href,fileBase:base,eras,
    basemap:{p:basemap,z:[0,13],tileSize:512},airspace:{p:airspace,z:[0,11]},
    airfields:{bin:`next/airfields.${afh}.bin`,details:`next/airfields.${afh}.json`},
    pins:{z:5,margin:2,base:`next/pins.${pinh}/`,shards:[391,392]},coverage};
  const path=`next/manifest.${hash(json(manifest))}.json`;await put(path,json(manifest));
  await put('fixture-index.json',json({manifest:path,representative,airspaceTile:{p:airspace,z:7,x:32,y:32}}));
  return {dir,manifest:path};
}
if(process.argv[1]===fileURLToPath(import.meta.url))console.log(await makeFixtures(process.argv[2],process.argv[3]));
export { layer as mvtLayer };
