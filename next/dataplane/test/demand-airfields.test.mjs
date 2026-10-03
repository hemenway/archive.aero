import test from 'node:test';
import assert from 'node:assert/strict';
import {parseManifest} from '../manifest.js';
import {buildDemand} from '../demand.js';
import {parseAirfields,airfieldVisible} from '../airfields.js';
const eras=Array.from({length:12},(_,i)=>({k:`${1950+i}-01-01_to_${1951+i}-01-01`,h:'0123456789ab',z:[4,11],b:null,c:null}));
const m=parseManifest({version:1,tileBase:'http://example.org/t/',fileBase:'http://example.org/',eras});
const state={date:'1955-01-01',chartTiles:[{z:10,x:238,y:410}],center:{x:.23,y:.4},scrub:{direction:1,velocity:0}};
test('new archives get low zoom then full resolution, existing paths retain unfinished bootstrap at current priority',()=>{
 const d=buildDemand(m,state);assert.equal(d.requests[0].src.z,7);assert.equal(d.requests[0].priority,0);assert.equal(d.requests[1].src.z,10);assert.equal(d.requests[1].priority,1);
 const repeated=buildDemand(m,state,d.paths);assert.equal(repeated.requests[0].priority,1);assert.equal(repeated.requests[0].key,d.requests[0].key);
});
test('velocity extends lookahead, playback uses next three, idle has ±1 full and ±3 low',()=>{
 const slow=buildDemand(m,state),fast=buildDemand(m,{...state,scrub:{direction:1,velocity:30}});assert.ok(fast.requests.length>slow.requests.length);
 const back=buildDemand(m,{...state,scrub:{direction:-1,velocity:0}});assert.ok(back.requests.some(r=>r.key.includes('1954-01-01')));
 const play=buildDemand(m,{...state,scrub:{playing:true}});assert.equal(play.requests.filter(r=>r.priority===2).length,3);assert.ok(play.requests.filter(r=>r.priority===2).every(r=>r.src.z===10));
 const idle=buildDemand(m,{...state,scrub:{direction:0}});assert.equal(idle.requests.filter(r=>r.priority===3).length,2);assert.equal(idle.requests.filter(r=>r.priority===4).length,4);
});
function binary() {const b=new ArrayBuffer(36),u=new Uint8Array(b);u.set([65,65,65,70,49]);const d=new DataView(b);d.setUint32(8,1,true);d.setFloat32(16,.2,true);d.setFloat32(20,.4,true);d.setUint16(24,1950,true);d.setUint16(28,1970,true);u[32]=1;return b;}
test('C4 typed views are zero-copy and padding for odd count is honored',()=>{const b=binary(),a=parseAirfields(b);assert.equal(a.mx.buffer,b);assert.equal(a.status.buffer,b);assert.equal(a.start[0],1950);assert.equal(a.end[0],1970);assert.equal(a.status[0],1);assert.equal(a.end.byteOffset,28);});
test('binary sanity and visibility rule ordering',()=>{assert.throws(()=>parseAirfields(new ArrayBuffer(3)));const b=binary();new Uint8Array(b)[32]=7;assert.throws(()=>parseAirfields(b));const a=parseAirfields(binary());assert.equal(airfieldVisible(a,0,1960),true);assert.equal(airfieldVisible(a,0,1971),false);assert.equal(airfieldVisible(a,0,null,1),false);a.status[0]=0;assert.equal(airfieldVisible(a,0,1971),true);assert.equal(airfieldVisible(a,0,1949),false);a.start[0]=a.end[0]=0;assert.equal(airfieldVisible(a,0,1000),true);});
test('integer Day dates prefetch the same frames as ISO dates',()=>{const iso=buildDemand(m,{...state,scrub:{direction:0}}),days=buildDemand(m,{...state,date:Math.floor(Date.parse('1955-01-01')/86400000),scrub:{direction:0}});assert.deepEqual(days.requests.map(r=>r.key),iso.requests.map(r=>r.key));assert.ok(iso.requests.some(r=>r.key.includes('1954-01-01')));});
