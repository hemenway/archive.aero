// Copy the exact installed releases (npm ci) without modifying vendor bytes.
// --check verifies tracked assets and SRI, never writes and never uses a CDN.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const check = process.argv.includes('--check');
const packages = [
  { name: 'leaflet', version: '1.9.4', files: {
    'dist/leaflet.js': 'leaflet.js', 'dist/leaflet.js.map': 'leaflet.js.map', 'dist/leaflet.css': 'leaflet.css',
    'dist/images/layers.png': 'images/layers.png', 'dist/images/layers-2x.png': 'images/layers-2x.png',
    'dist/images/marker-icon.png': 'images/marker-icon.png', 'dist/images/marker-icon-2x.png': 'images/marker-icon-2x.png',
    'dist/images/marker-shadow.png': 'images/marker-shadow.png', LICENSE: 'LICENSE',
  } },
  { name: 'pmtiles', version: '3.0.6', files: { 'dist/pmtiles.js': 'pmtiles.js' } },
  { name: 'papaparse', version: '5.4.1', files: { 'papaparse.min.js': 'papaparse.min.js', LICENSE: 'LICENSE' } },
  { name: 'protomaps-leaflet', version: '5.0.0', files: {
    'dist/protomaps-leaflet.js': 'protomaps-leaflet.js', 'dist/protomaps-leaflet.js.map': 'protomaps-leaflet.js.map', LICENSE: 'LICENSE',
  } },
];
const pins = {
  '/vendor/leaflet/1.9.4/leaflet.js': 'sha384-cxOPjt7s7Iz04uaHJceBmS+qpjv2JkIHNVcuOrM+YHwZOmJGBXI00mdUXEq65HTH',
  '/vendor/leaflet/1.9.4/leaflet.css': 'sha384-sHL9NAb7lN7rfvG5lfHpm643Xkcjzp4jFvuavGOndn6pjVqS6ny56CAt3nsEVT4H',
  '/vendor/pmtiles/3.0.6/pmtiles.js': 'sha384-4sbA4B4Oqxkzs6apu8HaZcGrL3BySmt0tO/LQf6al3hkgq8ijZFxkTkvxxZZ/3PU',
  '/vendor/papaparse/5.4.1/papaparse.min.js': 'sha384-D/t0ZMqQW31H3az8ktEiNb39wyKnS82iFY52QPACM+IjKW3jDUhyIgh2PApRqJZs',
  '/vendor/protomaps-leaflet/5.0.0/protomaps-leaflet.js': 'sha384-KeIb4wv6BGhSkhTNhLPQI2jcmQmZxMVQNxqM7UUEF5Zbmy+j2KSKRwba0gvAxLRE',
};
const manifest = {};
async function emit(relative, data) {
  const dest = path.join(root, relative);
  const current = await readFile(dest).catch(error => { if (error.code !== 'ENOENT') throw error; return null; });
  if (current?.equals(data)) return;
  if (check) throw new Error(`${relative} differs from the pinned package. Run npm run vendor:frontend.`);
  await mkdir(path.dirname(dest), { recursive: true });
  await writeFile(dest, data);
}
for (const pkg of packages) {
  const installed = JSON.parse(await readFile(path.join(root, 'node_modules', pkg.name, 'package.json')));
  if (installed.version !== pkg.version) throw new Error(`Expected ${pkg.name}@${pkg.version}, found ${installed.version}; run npm ci.`);
  for (const [source, target] of Object.entries(pkg.files)) {
    const data = await readFile(path.join(root, 'node_modules', pkg.name, source));
    const url = `/vendor/${pkg.name}/${pkg.version}/${target}`;
    const integrity = 'sha384-' + createHash('sha384').update(data).digest('base64');
    if (pins[url] && pins[url] !== integrity) throw new Error(`SRI mismatch: ${url}`);
    manifest[url] = { package: `${pkg.name}@${pkg.version}`, source, bytes: data.length, integrity };
    await emit(url.slice(1), data);
  }
}
await emit('vendor/manifest.json', Buffer.from(JSON.stringify(manifest, null, 2) + '\n'));
// The prebuilt PMTiles / Protomaps distributions also embed these projects.
// Retain their notices without rebuilding or changing any release bytes.
const notices = [
  ['@mapbox/point-geometry', 'node_modules/@mapbox/point-geometry/LICENSE'],
  ['@mapbox/vector-tile', 'node_modules/@mapbox/vector-tile/LICENSE.txt'],
  ['@protomaps/basemaps', 'vendor/licenses/protomaps-basemaps.txt'],
  ['color2k', 'node_modules/color2k/LICENSE'],
  ['pbf', 'node_modules/pbf/LICENSE'],
  ['PMTiles', 'vendor/pmtiles/3.0.6/LICENSE'],
  ['fflate', 'node_modules/fflate/LICENSE'],
  ['quickselect', 'node_modules/quickselect/LICENSE'],
  ['rbush', 'node_modules/rbush/LICENSE'],
  ['potpack', 'node_modules/potpack/LICENSE'],
];
let noticeText = 'Notices for dependencies embedded in the upstream browser distributions.\n\n';
for (const [name, filename] of notices) {
  noticeText += `${name}\n${'='.repeat(name.length)}\n${await readFile(path.join(root, filename), 'utf8')}\n\n`;
}
await emit('vendor/THIRD_PARTY_LICENSES.txt', Buffer.from(noticeText));
console.log(check ? 'Vendor versions, exact bytes, and SRI verified.' : 'Vendored pinned frontend dependencies.');
