self.onmessage = ({ data }) => {
  const canvas = new OffscreenCanvas(256, 256), ctx = canvas.getContext('2d');
  ctx.fillStyle = data.key.startsWith('basemap/') ? '#182b35' : '#c9b78b'; ctx.fillRect(0, 0, 256, 256);
  ctx.strokeStyle = '#786d53';
  for (let i = 0; i < 256; i += 32) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 256); ctx.moveTo(0, i); ctx.lineTo(256, i); ctx.stroke(); }
  ctx.fillStyle = '#263c49'; ctx.font = '14px system-ui'; ctx.fillText(data.key.split('/')[1].slice(0, 22), 15, 36);
  const bitmap = canvas.transferToImageBitmap();
  self.postMessage({ key: data.key, bitmap }, [bitmap]);
};
