import { createRenderer } from './index.js';
import { mercator } from './camera.js';
export const colors = ['#bc6038', '#427ea4', '#8d7651', '#745ca7'];
export function syntheticTile(size, color, label = '') {
  const canvas = typeof OffscreenCanvas === 'function' ? new OffscreenCanvas(size, size) : Object.assign(document.createElement('canvas'), { width: size, height: size });
  const ctx = canvas.getContext('2d'); ctx.fillStyle = color; ctx.fillRect(0, 0, size, size);
  if (label) {
    ctx.strokeStyle = 'rgba(255,255,255,.2)'; ctx.strokeRect(.5, .5, size - 1, size - 1);
    ctx.fillStyle = '#fff'; ctx.font = '16px system-ui'; ctx.fillText(label, 12, 28);
    ctx.strokeStyle = 'rgba(255,255,255,.1)';
    for (let i = 32; i < size; i += 32) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, size); ctx.moveTo(0, i); ctx.lineTo(size, i); ctx.stroke(); }
  }
  return createImageBitmap(canvas);
}
const canvas = document.querySelector('canvas');
const test = new URLSearchParams(location.search).has('test');
export const renderer = createRenderer(canvas, { minZoom: 0, preserveDrawingBuffer: test });
window.r = renderer; window.syntheticTile = syntheticTile;
window.ready = true;
if (!test) {
  const p = mercator(-98, 39); renderer.setCamera({ x: p[0], y: p[1], zoom: 6 });
  let era = 0, generation = 0;
  const pending = new Set();window.demoPending=()=>pending.size;
  const areas=[null,[-100,36,-95,41],[-98,37,-94,39]].map(b=>b?{nw:mercator(b[0],b[3]),se:mercator(b[2],b[1])}:null);
  function deliver(key, size, color, label, delayed) {
    if (renderer.hasTexture(key) || pending.has(key)) return;
    pending.add(key);
    setTimeout(async () => { const bitmap = await syntheticTile(size, color, label); pending.delete(key); renderer.upload(key, bitmap); }, delayed ? 30 + Math.random() * 450 : 0);
  }
  function plan() {
    const tiles = renderer.visibleTiles(256), base = renderer.visibleTiles(512);
    renderer.setBasemapPlan(base.map(dst => {
      const key = `basemap/offline/${dst.z}/${dst.x}/${dst.y}`; deliver(key, 512, '#172b3e', `BASE ${dst.z}/${dst.x}/${dst.y}`, false);
      return { dst, items: [{ key, dst, src: dst }] };
    }));
    renderer.setChartPlan({ id: String(++generation), tiles: tiles.map(dst => {
      const items = [];
      // Overlapping archive eras, ordered bottom to top; quarter tiles overzoom.
      for (let a = 0; a <= 2; a++) {
        const area=areas[a],n=2**dst.z;
        if(area && ((dst.x+1)/n<=area.nw[0] || dst.x/n>=area.se[0] || (dst.y+1)/n<=area.nw[1] || dst.y/n>=area.se[1]))continue;
        const diff = a === 1 ? 1 : 0, z = Math.max(0, dst.z - diff), k = 2 ** (dst.z - z), src = { z, x: Math.floor(dst.x / k), y: Math.floor(dst.y / k) };
        const key = `sectionals/era-${era}-${a}/${src.z}/${src.x}/${src.y}`;
        deliver(key, 256, colors[(era + a) % colors.length], `${1950 + era * 25} · archive ${a} · ${src.z}/${src.x}/${src.y}`, true);
        items.push({ key, src, dst, ...(a ? { clip: `area-${a}` } : {}) });
      }
      return { dst, items };
    }) });
    renderer.setAirfieldFilter({ year: 1950 + era * 25 });
  }
  renderer.setClipRing('area-1', [[-100, 36], [-95, 36], [-95, 41], [-100, 41]]);
  renderer.setClipRing('area-2', [[-98, 37], [-94, 37], [-94, 39], [-98, 39]]);
  const locations = [[-98,39],[-97,38],[-99,40],[-96,39.5],[-100,38.3]], xy = locations.map(p => mercator(...p));
  renderer.setAirfields({ mx: new Float32Array(xy.map(p => p[0])), my: new Float32Array(xy.map(p => p[1])), start: new Uint16Array([1940,1960,0,1950,1970]), end: new Uint16Array([0,1985,0,2000,0]), status: new Uint8Array([0,1,2,1,0]) });
  const positions = [], starts = [0], styles = [];
  for (let i = 0; i < 8; i++) {
    const x = -102 + i, y = 36;
    for (const p of [[x,y],[x+.7,y+1],[x+.5,y+3],[x-1,y+2],[x,y]]) positions.push(...mercator(...p));
    starts.push(positions.length / 2); styles.push(i);
  }
  renderer.setAirspaceTile('0/0/0', { positions: new Float32Array(positions), starts: new Uint32Array(starts), from: new Int32Array(8).fill(-20000), to: new Int32Array(8).fill(2147483647), style: new Uint8Array(styles), rg: new Uint8Array(8) });
  renderer.setAirspaceFilter({ day: 0 }); renderer.setPin({ lng: -98.8, lat: 38.7 });
  renderer.on('moveend', plan); renderer.on('contextrestored', () => { pending.clear(); plan(); });
  renderer.on('render', () => { const s = renderer.stats(); document.querySelector('#status').textContent = `${s.textures} textures · ${(s.textureBytes / 1048576).toFixed(1)} MiB · ${s.frameMs.toFixed(1)} ms`; });
  document.querySelector('#era').addEventListener('input', e => { era = +e.target.value; document.querySelector('#year').textContent = 1950 + era * 25; plan(); });
  document.querySelector('#opacity').addEventListener('input', e => renderer.setChartStyle({ opacity: +e.target.value }));
  document.querySelector('#fields').addEventListener('change', e => renderer.setAirfieldFilter({ statusMask: e.target.checked ? 7 : 0 }));
  document.querySelector('#restore').addEventListener('click', () => { const ext = renderer.gl.getExtension('WEBGL_lose_context'); ext?.loseContext(); setTimeout(() => ext?.restoreContext(), 600); });
  plan();
}
