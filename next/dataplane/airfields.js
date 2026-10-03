export function parseAirfields(buffer) {
  const bytes = new Uint8Array(buffer), magic = [65,65,65,70,49,0,0,0];
  if(bytes.length<16 || magic.some((v,i)=>bytes[i]!==v)) throw new Error('Invalid airfields magic');
  const view = new DataView(buffer), n=view.getUint32(8,true);
  if(view.getUint32(12,true)!==0) throw new Error('Invalid airfields reserved word');
  let offset=16;
  const array = Type => { offset=Math.ceil(offset/4)*4; const size=n*Type.BYTES_PER_ELEMENT; if(size>buffer.byteLength-offset) throw new Error('Truncated airfields'); const a=new Type(buffer,offset,n); offset+=size; return a; };
  const result={mx:array(Float32Array),my:array(Float32Array),start:array(Uint16Array),end:array(Uint16Array),status:array(Uint8Array)};
  for(let i=0;i<n;i++) if(!Number.isFinite(result.mx[i]) || !Number.isFinite(result.my[i]) || result.mx[i]<0 || result.mx[i]>1 || result.my[i]<0 || result.my[i]>1 || result.status[i]>2) throw new Error('Invalid airfield values');
  return result;
}
export function airfieldVisible(arrays,i,year,statusMask=7) {
  if(!(statusMask & (1<<arrays.status[i]))) return false;
  if(year==null || (!arrays.start[i] && !arrays.end[i])) return true;
  if(arrays.start[i] && year<arrays.start[i]) return false;
  if(arrays.status[i]===0) return true;
  return !(arrays.end[i] && year>arrays.end[i]);
}
