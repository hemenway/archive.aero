import {performance} from 'node:perf_hooks';
import {parseManifest} from '../manifest.js';
import {planCharts,tileLonLatBounds} from '../plan.js';
const tiles=Array.from({length:300},(_,i)=>({z:10,x:220+i%20,y:390+Math.floor(i/20)}));
const era=(i,dense)=>({k:`1950-01-${String(i%28+1).padStart(2,'0')}_to_1960-01-01`,h:i.toString(16).padStart(12,'0'),z:[4,11],b:dense?null:tileLonLatBounds(10,220+i%20,390+Math.floor(i/20)),c:null});
const quantile=(a,p)=>a[Math.min(a.length-1,Math.floor(a.length*p))];
export function benchmarkPlanning() {
 const result={};for(const dense of [false,true]) {
  const m=parseManifest({version:1,eras:Array.from({length:80},(_,i)=>era(i,dense))});let p;
  for(let i=0;i<100;i++)p=planCharts(m,'1955-01-01',tiles);
  const times=[];for(let i=0;i<1000;i++){const t=performance.now();p=planCharts(m,'1955-01-01',tiles);times.push(performance.now()-t);}times.sort((a,b)=>a-b);
  result[dense?'dense':'spatial']={tiles:300,activeEras:80,items:p.reduce((s,t)=>s+t.items.length,0),p50Ms:quantile(times,.5),p95Ms:quantile(times,.95)};
 }return result;
}
if(process.argv[1]?.endsWith('planning.mjs'))console.log(JSON.stringify(benchmarkPlanning(),null,2));
