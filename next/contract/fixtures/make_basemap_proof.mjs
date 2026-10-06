#!/usr/bin/env node
// Local synthetic vector source, suitable for exercising the renderer offline.
import { archive,mvtLayer } from './make_fixtures.mjs';
import { mkdir,writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
const out=resolve(process.argv[2]||'next/contract/fixtures/out/basemap-proof');await mkdir(out,{recursive:true});
const bytes=Buffer.concat([
  mvtLayer('earth',[{kind:'earth'}],{x:0,y:0,width:4096,height:4096}),
  mvtLayer('water',[{kind:'water'}],{x:0,y:0,width:1800,height:4096}),
  mvtLayer('places',[{kind:'locality',kind_detail:'city',name:'Fixture Town','name:en':'Fixture Town',population:1000000,min_zoom:0}],{type:1,x:2600,y:2000})
]);
const tiles=[];for(let z=0;z<=2;z++)for(let x=0;x<2**z;x++)for(let y=0;y<2**z;y++)tiles.push({z,x,y,bytes});
await writeFile(resolve(out,'vector.pmtiles'),archive(tiles,{type:1,gzip:true,metadata:{vector_layers:[{id:'earth'},{id:'water'},{id:'places'}]}}));
console.log(resolve(out,'vector.pmtiles'));
