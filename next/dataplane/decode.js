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
