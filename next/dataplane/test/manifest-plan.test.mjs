import test from 'node:test';
import assert from 'node:assert/strict';
import {parseManifest,erasAt,day,loadManifest} from '../manifest.js';
import {planCharts,planBasemap,tileLonLatBounds} from '../plan.js';
const era=(k,z=[4,11],b=null,c=null)=>({k,h:'0123456789ab',z,b,c});
const m=parseManifest({version:1,eras:[era('1950-01-01_to_1960-01-01'),era('1955-01-01_to_1970-01-01'),era('1970-01-01_to_1980-01-01')]});
test('intervals: exclusive end, manifest paint order and bounds',()=>{
 assert.deepEqual(erasAt(m,'1960-01-01'),[1]);assert.deepEqual(erasAt(m,'1959-01-01'),[0,1]);assert.deepEqual(erasAt(m,'1980-01-01'),[]);
 assert.deepEqual(m.frames,['1950-01-01','1955-01-01','1970-01-01']);assert.deepEqual(m.dateBounds,{min:'1950-01-01',max:'1979-12-31'});
});
test('binary index does not reorder manifest and finds long overlaps',()=>{
 const a=parseManifest({version:1,eras:[era('1970-01-01_to_1980-01-01'),era('1900-01-01_to_2000-01-01'),era('1950-01-01_to_1960-01-01')]});assert.deepEqual(erasAt(a,'1975-01-01'),[0,1]);
});
test('sanity checks reject bad date, zoom, bounds, hash and coverage',async()=>{
 for(const patch of [{z:[8,4]},{b:[5,0,1,2]},{h:'abc'},{c:[5000]},{k:'1950-01-01'},{k:'1950-02-30_to_1960-01-01'}])assert.throws(()=>parseManifest({version:1,eras:[{...era('1950-01-01_to_1960-01-01'),...patch}]}));
 await assert.rejects(loadManifest('x',async()=>new Response('',{status:503})));assert.equal(day('1970-01-01'),0);
});
test('planning culls bounds, z6 coverage, minimum zoom and clamps ancestors',()=>{
 const t={z:10,x:238,y:413},b=tileLonLatBounds(t.z,t.x,t.y),c=[Math.floor(t.y/16)*64+Math.floor(t.x/16)];
 const a=parseManifest({version:1,eras:[era('1950-01-01_to_1960-01-01',[4,8],b,c),era('1951-01-01_to_1961-01-01',[4,11],[0,0,1,1]),era('1952-01-01_to_1962-01-01',[4,11],null,[])]});
 const p=planCharts(a,'1959-01-01',[t]);assert.equal(p[0].items.length,1);assert.deepEqual(p[0].items[0].src,{z:8,x:59,y:103});assert.match(p[0].items[0].key,/\/8\/59\/103$/);
 assert.equal(planCharts(a,'1959-01-01',[{z:3,x:0,y:0}])[0].items.length,0);
});
test('wrapped edge tiles and low zoom coverage work without wrap mutation',()=>{
 const a=parseManifest({version:1,eras:[era('1950-01-01_to_1960-01-01',[0,11],null,[0,4095])]});
 assert.equal(planCharts(a,'1959-01-01',[{z:6,x:63,y:63}])[0].items.length,1);
 assert.equal(planCharts(a,'1959-01-01',[{z:0,x:0,y:0}])[0].items.length,1);
});
test('solo and raster basemap conform to C6',()=>{
 const t={z:10,x:238,y:413},solo={paths:['sectionals/chart/a.0123456789ab'],zoom:[8,9],clip:{id:'a',ring:[[0,0],[1,1],[1,0]]}};
 assert.deepEqual(planCharts(m,'1950-01-01',[t],solo)[0].items[0],{key:'sectionals/chart/a.0123456789ab/9/119/206',src:{z:9,x:119,y:206},dst:t,clip:'a'});
 assert.equal(planBasemap({p:'basemap/a.0123456789ab',z:[0,9]},[t])[0].items[0].src.z,9);
});

// --- occupancy culling and the basemap's missing-tile fallback ---
import {occupancyOf,occupancy,cull,planBasemap as planBase,EMPTY as BLANK} from '../plan.js';
const alphaTile=fn=>{const a=new Uint8ClampedArray(256*256*4);for(let y=0;y<256;y++)for(let x=0;x<256;x++)a[(y*256+x)*4+3]=fn(x,y);return a;};
test('occupancy grid: blank and solid areas of a tile answer for its descendants',()=>{
  // Left half solid, right half blank, one faint pixel at (200,40).
  const grid=occupancyOf(alphaTile((x,y)=>x<128?255:(x===200&&y===40?3:0))),occ=new Map([['p/5/10/12',grid]]);
  assert.deepEqual(occupancy(occ,'p',{z:5,x:10,y:12}),{any:true,full:false});
  assert.deepEqual(occupancy(occ,'p',{z:6,x:20,y:24}),{any:true,full:true});   // top-left quarter
  assert.deepEqual(occupancy(occ,'p',{z:6,x:21,y:25}),{any:false,full:false}); // bottom-right quarter
  assert.deepEqual(occupancy(occ,'p',{z:8,x:86,y:97}),{any:true,full:false});  // the cell holding the faint pixel
  assert.deepEqual(occupancy(occ,'p',{z:8,x:87,y:97}),{any:false,full:false});
  assert.equal(occupancy(occ,'p',{z:9,x:160,y:192}),null);                     // four levels down: not answered
  assert.equal(occupancy(occ,'q',{z:6,x:20,y:24}),null);
});
test('cull drops blank items always and hidden items only when charts are opaque',()=>{
  const solid=occupancyOf(alphaTile(()=>255)),occ=new Map([['top/5/1/1',solid],['mid/5/1/1',BLANK]]);
  const items=['bottom','mid','top'].map(p=>({key:`${p}/8/8/8`,src:{z:8,x:8,y:8},dst:{z:8,x:8,y:8}}));
  assert.deepEqual(cull(items,{occ,occlude:true}).map(i=>i.key),['top/8/8/8']);
  assert.deepEqual(cull(items,{occ,occlude:false}).map(i=>i.key),['bottom/8/8/8','top/8/8/8']);
  assert.equal(cull(items,{occ:new Map(),occlude:true}),items);
  // The solid-looking top tile is missing from its archive: it no longer hides what is under it.
  assert.deepEqual(cull(items,{occ,occlude:true,absent:new Set(['top/8/8/8'])}).map(i=>i.key),['bottom/8/8/8']);
  assert.equal(cull(items,null),items);
});
test('basemap plan substitutes the nearest existing ancestor for a missing tile',()=>{
  const source={p:'basemap/x.0123456789ab',z:[0,13],format:'mvt'},dst={z:9,x:100,y:200},k=s=>`${source.p}/${s}`;
  assert.equal(planBase(source,[dst])[0].items[0].key,k('9/100/200'));
  const item=planBase(source,[dst],new Set([k('9/100/200'),k('8/50/100')]))[0].items[0];
  assert.equal(item.key,k('7/25/50'));assert.deepEqual(item.src,{z:7,x:25,y:50});assert.deepEqual(item.dst,dst);
});
import {coveredCells} from '../plan.js';
test('a basemap tile wholly under solid charts is left out; a partly covered one stays',()=>{
  const source={p:'basemap/x.0123456789ab',z:[0,13],format:'mvt'},solid=occupancyOf(alphaTile(()=>255)),occ=new Map([['c/7/20/20',solid]]);
  const cell=(x,y)=>({dst:{z:10,x,y},items:[{key:`c/10/${x}/${y}`,src:{z:10,x,y},dst:{z:10,x,y}}]});
  const four=[cell(160,160),cell(161,160),cell(160,161),cell(161,161)];
  assert.equal(planBase(source,[{z:9,x:80,y:80}],null,coveredCells(four,{occ,occlude:true}))[0].items.length,0);
  assert.equal(planBase(source,[{z:9,x:80,y:80}],null,coveredCells(four.slice(1),{occ,occlude:true}))[0].items.length,1);
  assert.equal(planBase(source,[{z:9,x:80,y:80}],null,coveredCells(four,{occ,occlude:false}))[0].items.length,1);
  // Not yet known whether the chart on top is solid: the basemap under it waits.
  assert.equal(planBase(source,[{z:9,x:80,y:80}],null,coveredCells(four,{occ:new Map(),occlude:true}))[0].items.length,0);
  const blankTop=new Map([['c/7/20/20',occupancyOf(alphaTile(x=>x<16?255:0))]]);
  assert.equal(planBase(source,[{z:9,x:80,y:80}],null,coveredCells(four,{occ:blankTop,occlude:true}))[0].items.length,1);
});
