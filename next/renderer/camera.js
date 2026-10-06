export const MAX_LAT = 85.0511287798066;
export const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
export const wrap = n => ((n % 1) + 1) % 1;
export function mercator(lng, lat) {
  const s = Math.sin(clamp(lat, -MAX_LAT, MAX_LAT) * Math.PI / 180);
  return [(lng + 180) / 360, .5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)];
}
export function geographic(x, y) {
  return [x * 360 - 180, Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180 / Math.PI];
}
// Explicit longitude continuity: use lng + 360*n to project another world copy.
export class Camera {
  constructor({ center = [-98, 39], zoom = 6, minZoom = 6, maxZoom = 14 } = {}) {
    this.minZoom = minZoom; this.maxZoom = maxZoom;
    this.zoom = clamp(zoom, minZoom, maxZoom);
    [this.x, this.y] = mercator(center[0], center[1]);
    this.width = 1; this.height = 1;
  }
  get world() { return 256 * 2 ** this.zoom; }
  project(lngLat, nearest = true) {
    const p = mercator(lngLat[0], lngLat[1]);
    if (nearest) p[0] += Math.round(this.x - p[0]);
    return [(p[0] - this.x) * this.world + this.width / 2, (p[1] - this.y) * this.world + this.height / 2];
  }
  unproject(point) { return geographic(this.x + (point[0] - this.width / 2) / this.world, this.y + (point[1] - this.height / 2) / this.world); }
  pan(dx, dy) { this.x -= dx / this.world; this.y = clamp(this.y - dy / this.world, 0, 1); }
  zoomAt(zoom, px, py) {
    const x = this.x + (px - this.width / 2) / this.world;
    const y = this.y + (py - this.height / 2) / this.world;
    this.zoom = clamp(zoom, this.minZoom, this.maxZoom);
    this.x = x - (px - this.width / 2) / this.world;
    this.y = clamp(y - (py - this.height / 2) / this.world, 0, 1);
  }
  visibleTiles(size = 256, buffer) {
    const z = Math.max(0, Math.round(this.zoom) - (size === 512 ? 1 : 0)), n = 2 ** z;
    const b = buffer ?? (Math.min(this.width, this.height) < 500 ? .5 : 1);
    const rx = this.width / this.world / 2, ry = this.height / this.world / 2;
    const left = Math.floor((this.x - rx) * n - b), right = Math.floor((this.x + rx) * n + b);
    const top = Math.max(0, Math.floor((this.y - ry) * n - b)), bottom = Math.min(n - 1, Math.floor((this.y + ry) * n + b));
    const tiles = [];
    for (let y = top; y <= bottom; y++) for (let wx = left; wx <= right; wx++) {
      const x = ((wx % n) + n) % n;
      tiles.push({ z, x, y, worldX: wx, key: `${z}/${x}/${y}`, onScreen: wx + 1 > (this.x - rx) * n && wx < (this.x + rx) * n && y + 1 > (this.y - ry) * n && y < (this.y + ry) * n,
        distance: (wx + .5 - this.x * n) ** 2 + (y + .5 - this.y * n) ** 2 });
    }
    tiles.sort((a, b) => a.distance - b.distance);
    return tiles;
  }
  fitBounds(bounds, padding = 0) {
    const a = mercator(...bounds[0]), b = mercator(...bounds[1]);
    if (b[0] < a[0]) b[0] += 1;
    const zoom = Math.floor(Math.log2(Math.min((this.width - 2 * padding) / Math.max(1e-12, b[0] - a[0]), (this.height - 2 * padding) / Math.max(1e-12, Math.abs(b[1] - a[1]))) / 256));
    return { center: geographic((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), zoom: clamp(zoom, this.minZoom, this.maxZoom) };
  }
}
