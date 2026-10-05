import {createCore} from './core.js';
let core;
function transfers(value,out=new Set()) {
  if(!value || typeof value!=='object')return [...out];
  if(typeof ImageBitmap!=='undefined' && value instanceof ImageBitmap)out.add(value);
  else if(ArrayBuffer.isView(value))out.add(value.buffer);
  else if(value instanceof ArrayBuffer)out.add(value);
  else for(const v of Object.values(value))transfers(v,out);
  return [...out];
}
self.onmessage=async({data:{id,method,args=[]}})=>{
  try {
    let result;
    if(method==='init') {core=await createCore(args[0],(event,payload)=>{const safe=event==='error'?{...payload,error:{message:payload.error.message,name:payload.error.name}}:payload;self.postMessage({event,payload:safe},transfers(safe));if(core)self.postMessage({event:'stats',payload:core.stats()});});result=args[0].manifest?null:core.m.raw;}
    else {
      result=await core[method](...args);
      if(method==='loadAirfields' && result) {const buffer=result.mx.buffer.slice(0);result=Object.fromEntries(Object.entries(result).map(([k,a])=>[k,new a.constructor(buffer,a.byteOffset,a.length)]));}
    }
    self.postMessage({id,result},transfers(result));if(core)self.postMessage({event:'stats',payload:core.stats()});
  } catch(error) {self.postMessage({id,error:{message:error.message,name:error.name}});}
};
