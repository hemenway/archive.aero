import { readFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';

const root = new URL('../../', import.meta.url);
const libraries = new Map([
  ['https://unpkg.com/leaflet@1.9.4/dist/leaflet.js', 'leaflet/dist/leaflet.js'],
  ['https://unpkg.com/leaflet@1.9.4/dist/leaflet.css', 'leaflet/dist/leaflet.css'],
  ['https://unpkg.com/papaparse@5.4.1/papaparse.min.js', 'papaparse/papaparse.min.js'],
  ['https://unpkg.com/pmtiles@3.0.6/dist/pmtiles.js', 'pmtiles/dist/pmtiles.js'],
  ['https://unpkg.com/protomaps-leaflet@5.0.0/dist/protomaps-leaflet.js', 'protomaps-leaflet/dist/protomaps-leaflet.js'],
]);
export const dates = ['1950-01-01', '1960-01-01', '1970-01-01', '1980-01-01'];
const keys = dates.slice(0, -1).map((date, i) => `${date}_to_${dates[i + 1]}`);
const csv = 'date_iso,url\n' + keys.map(key => key + ',').join('\n');
const inventory = {
  locations: { 'Test Dallas': { era: 'modern', ref: 'dallas', charts: keys.map((_, i) => ({ d: dates[i], e: dates[i + 1], ed: String(i + 1) })) } },
  rings: { dallas: [[[-98, 31], [-95, 31], [-95, 34], [-98, 34], [-98, 31]]] },
};

function varint(value) {
  const out = [];
  do { const b = value % 128; value = Math.floor(value / 128); out.push(b | (value ? 128 : 0)); } while (value);
  return out;
}

// A real v3 PMTiles file: one PNG repeated over all tile IDs through z11.
// Basemap uses an empty vector archive. Real PMTiles parsing/range reads stay active.
function archive(vector = false) {
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64');
  const directory = vector ? Buffer.from([0]) : Buffer.from([1, 0, ...varint((4 ** 12 - 1) / 3), ...varint(png.length), 1]);
  const metadata = Buffer.from('{}');
  const header = Buffer.alloc(127);
  header.write('PMTiles'); header[7] = 3;
  for (const [offset, value] of [[8, 127], [16, directory.length], [24, 127 + directory.length], [32, metadata.length], [40, 129 + directory.length], [48, 0], [56, 129 + directory.length], [64, vector ? 0 : png.length], [72, vector ? 0 : (4 ** 12 - 1) / 3], [80, vector ? 0 : 1], [88, vector ? 0 : 1]]) header.writeBigUInt64LE(BigInt(value), offset);
  header[96] = 1; header[97] = 1; header[98] = 1; header[99] = vector ? 1 : 2;
  header[100] = 0; header[101] = 11;
  for (const [offset, value] of [[102, -180], [106, -85], [110, 180], [114, 85]]) header.writeInt32LE(value * 1e7, offset);
  return Buffer.concat([header, directory, metadata, ...(vector ? [] : [png])]);
}
const raster = archive();
const vector = archive(true);

// Valid metadata bundle exercises the normal boot path as well as CSV fallback.
const index = gzipSync(Buffer.from(JSON.stringify({
  version: 1, baseUrl: 'https://data.archive.aero/sectionals/',
  groups: [{ name: 'fixture', off: 0, len: raster.length * keys.length, i0: 0, i1: keys.length }],
  eras: keys.map((k, i) => ({ k, off: i * raster.length, len: raster.length })),
})));
const preamble = Buffer.alloc(16);
preamble.write('AAMBv1\n\0'); preamble.writeUInt32LE(index.length, 8);
const bundle = Buffer.concat([preamble, index, ...keys.map(() => raster)]);

async function rangeResponse(route, bytes) {
  const range = /^bytes=(\d+)-(\d+)$/.exec(route.request().headers().range || '');
  const start = range ? +range[1] : 0;
  const end = range ? Math.min(+range[2], bytes.length - 1) : bytes.length - 1;
  return route.fulfill({ status: range ? 206 : 200, body: bytes.subarray(start, end + 1), headers: {
    'content-type': 'application/octet-stream', 'access-control-allow-origin': '*',
    'access-control-expose-headers': 'ETag, Content-Range', etag: '"fixture-v1"',
    ...(range ? { 'content-range': `bytes ${start}-${end}/${bytes.length}` } : {}),
  } });
}

export async function installFixtures(page, { bundleFails = false, csvFails = false } = {}) {
  const errors = [];
  const unexpected = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => {
    // Record clipboard writes across Chromium and WebKit without OS permissions.
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: async text => { window.__copied = text; } } });
  });
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    // WebKit routes local image blobs through this hook; they are not network IO.
    if (url.protocol === 'blob:' && url.origin === 'http://127.0.0.1:4173') return route.continue();
    if (libraries.has(url.href)) {
      return route.fulfill({ body: await readFile(new URL('node_modules/' + libraries.get(url.href), root)),
        contentType: url.pathname.endsWith('.css') ? 'text/css' : 'application/javascript', headers: { 'access-control-allow-origin': '*' } });
    }
    if (url.pathname.endsWith('.bundle')) return bundleFails ? route.fulfill({ status: 503, body: '' }) : rangeResponse(route, bundle);
    if (url.pathname.endsWith('.pmtiles')) return rangeResponse(route, url.pathname.includes('/basemap/') ? vector : raster);
    if (url.pathname.endsWith('/dates.csv')) return route.fulfill({ status: csvFails ? 503 : 200, contentType: 'text/csv', body: csv });
    if (url.pathname.endsWith('/timeline_data.json')) return route.fulfill({ json: inventory });
    if (url.pathname.endsWith('/airfields.json')) return route.fulfill({ json: { type: 'FeatureCollection', features: [] } });
    if (url.pathname.endsWith('/coverage.json')) return route.fulfill({ json: { segments: keys.map((_, i) => [dates[i], dates[i + 1], 1, 100]) } });
    if (url.hostname === 'get.geojs.io') return route.fulfill({ json: { latitude: '32.7767', longitude: '-96.7970' } });
    if (['fonts.googleapis.com', 'fonts.gstatic.com', 'plausible.io'].includes(url.hostname)) return route.fulfill({ body: '', contentType: url.hostname === 'plausible.io' ? 'application/javascript' : 'text/css' });
    if (url.origin === 'http://127.0.0.1:4173' && ['/', '/index.html', '/styles.css', '/tests/flicker-regression-guard.html'].includes(url.pathname)) return route.continue();
    unexpected.push(url.href);
    return route.abort();
  });
  return { errors, unexpected };
}
