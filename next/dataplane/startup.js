import {parseManifest,manifestSummary,resolveBases} from './manifest.js';

// Additive startup artifacts keep the complete C2 manifest available to older
// clients. Every overlapping era stays in its original paint order.
export function startupArtifacts(raw) {
  const summary=manifestSummary(parseManifest(raw)),years=new Map();
  for(let year=+summary.dateBounds.min.slice(0,4);year<=+summary.dateBounds.max.slice(0,4);year++) {
    const start=`${year}-01-01`,end=`${year+1}-01-01`;
    const eras=raw.eras.filter(e=>{const [a,b]=e.k.split('_to_');return a<end&&b>start;});
    years.set(String(year),{...raw,coverage:undefined,eras});
  }
  return {summary,years};
}

export async function loadStartup(url,date,manifestUrl,fetcher=fetch) {
  const read=async u=>{const r=await fetcher(u);if(!r.ok)throw new Error(`Startup HTTP ${r.status}`);return r.json();};
  const index=await read(url);
  if(index.version!==1||!index.summary?.dateBounds||!index.years)throw new Error('Unsupported startup index');
  const {min,max}=index.summary.dateBounds;
  const selected=/^\d{4}-\d{2}-\d{2}$/.test(date??'')?date:max;
  const year=(selected<min?min:selected>max?max:selected).slice(0,4);
  const shard=index.years[year];if(!shard)throw new Error('Missing startup year');
  const raw=resolveBases(await read(new URL(shard,new URL(url,globalThis.location?.href??manifestUrl)).href),manifestUrl);
  return {raw,summary:index.summary,year};
}
