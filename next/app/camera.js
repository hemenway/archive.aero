export const project = (lat, lng, z) => {
  const s = Math.sin(Math.max(-85.0511, Math.min(85.0511, lat)) * Math.PI / 180);
  return [(lng + 180) / 360 * 256 * 2 ** z, (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * 256 * 2 ** z];
};
export const unproject = (x, y, z) => {
  const n = 256 * 2 ** z;
  return { lat: Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) * 180 / Math.PI, lng: x / n * 360 - 180 };
};
export function fitLower48(width, height) {
  const a = project(49.4, -124.8, 0), b = project(24.4, -67.1, 0);
  const zoom = Math.max(0, Math.min(12, Math.floor(Math.log2(Math.min(Math.max(256, width - 48) / (b[0] - a[0]), Math.max(256, height - 180) / (b[1] - a[1]))))));
  return { ...unproject((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0), zoom };
}
export function visibleTiles(camera, width, height) {
  const z = Math.floor(camera.zoom), n = 2 ** z, [cx, cy] = project(camera.lat, camera.lng, z), tiles = [];
  for (let x = Math.floor((cx - width / 2) / 256); x <= Math.floor((cx + width / 2) / 256); x++) {
    for (let y = Math.max(0, Math.floor((cy - height / 2) / 256)); y <= Math.min(n - 1, Math.floor((cy + height / 2) / 256)); y++) {
      tiles.push({ z, x: ((x % n) + n) % n, y, screenX: x * 256 - cx + width / 2, screenY: y * 256 - cy + height / 2 });
    }
  }
  return tiles;
}
