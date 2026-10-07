import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const script=readFileSync(new URL('../../shell/early.js',import.meta.url),'utf8');
const boot={base:'https://fixture/t/',start:'2021-02-25',end:'2021-09-08',map:null,eras:[
 {p:'expired',e:'2021-04-22',z:[0,12],b:null},
 {p:'current',e:'2021-09-09',z:[0,12],b:null}
]};
function early(search='',data=boot) {
 const requested=[];
 runInNewContext(script,{document:{getElementById:()=>({textContent:JSON.stringify(data)})},location:{href:'https://fixture/next/',search},innerWidth:1440,innerHeight:900,window:{},URL,URLSearchParams,fetch:url=>{requested.push(url);return Promise.resolve();}});
 return requested;
}
test('early tiles use the final covered day and never request an expired edition',()=>{
 const requests=early();assert.ok(requests.length);assert.ok(requests.every(u=>u.includes('/current/')));
 assert.deepEqual(early('?date=2022-01-01'),[]);assert.deepEqual(early('?date=1950-01-01'),[]);
});
test('an edition-date share link requests a few overview tiles before full resolution',()=>{
 const requests=early('?date=2021-02-25&lat=38&lng=-95.95&zoom=10.75');
 assert.ok(requests.some(u=>u.includes('/expired/')));assert.ok(requests.some(u=>u.includes('/current/')));
 assert.ok(requests.every(u=>u.includes('/7/')));assert.ok(requests.length<=8);
});
test('overview requests exclude archives outside the viewport even when their coarse tile spans it',()=>{
 const data={...boot,eras:[{...boot.eras[1],b:[-145,60,-131,68]}]};
 assert.deepEqual(early('?lat=38&lng=-95.95&zoom=5',data),[]);
});
