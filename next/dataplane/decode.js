export async function decodeRaster(bytes,{bitmap=true,contentType}={}) {
  const blob=new Blob([bytes],{type:contentType??'application/octet-stream'});
  if(bitmap && typeof createImageBitmap==='function') {
    try { return {bitmap:await createImageBitmap(blob)}; } catch(error) {
      // Some engines expose createImageBitmap but cannot decode a particular codec in a Worker.
      if(typeof document==='undefined') return {bytes:bytes.slice(),contentType};
    }
  }
  if(typeof document==='undefined') return {bytes:bytes.slice(),contentType};
  const url=URL.createObjectURL(blob), image=new Image();
  try {
    image.src=url; await image.decode();
    if(typeof OffscreenCanvas==='function') {const canvas=new OffscreenCanvas(image.naturalWidth,image.naturalHeight);const ctx=canvas.getContext('2d');if(ctx && typeof canvas.transferToImageBitmap==='function') {ctx.drawImage(image,0,0);return {bitmap:canvas.transferToImageBitmap()};}}
    // Last-resort TexImageSource for browsers without either bitmap API.
    image.close=()=>{image.src='';};return {bitmap:image};
  }
  finally { URL.revokeObjectURL(url); }
}

// Reads a decoded 256 px chart tile back once to learn where it is blank and where it is solid (plan.js occupancyOf).
// Returns null where no 2D canvas exists on this thread; planning then simply culls nothing.
let probeContext;
export function probeOccupancy(bitmap,occupancyOf) {
  if(bitmap.width!==256||bitmap.height!==256) return null;
  if(probeContext===undefined) {
    try {
      const canvas=typeof OffscreenCanvas==='function'?new OffscreenCanvas(256,256):typeof document!=='undefined'?Object.assign(document.createElement('canvas'),{width:256,height:256}):null;
      probeContext=canvas?.getContext('2d',{willReadFrequently:true})??null;
    } catch { probeContext=null; }
  }
  if(!probeContext) return null;
  try {
    probeContext.globalCompositeOperation='copy';probeContext.drawImage(bitmap,0,0);
    return occupancyOf(probeContext.getImageData(0,0,256,256).data);
  } catch { return null; }
}
