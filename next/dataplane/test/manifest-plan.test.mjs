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
