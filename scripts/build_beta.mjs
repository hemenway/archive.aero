import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = `${root}beta/dist`;
await rm(output, { recursive: true, force: true });
await mkdir(`${output}/assets`, { recursive: true });
// Public allowlist: never upload the catalog, worklists, or local credentials.
for (const file of ['dates.csv', 'coverage.json', 'social-card.jpg']) await cp(`${root}${file}`, `${output}/${file}`);
for (const [name, version] of [['pmtiles', '3.0.6'], ['papaparse', '5.4.1']]) {
  await cp(`${root}vendor/${name}/${version}`, `${output}/vendor/${name}/${version}`, { recursive: true });
}
const libraries = [
  { name: 'maplibre-gl', version: '6.11.2', files: ['maplibre-gl.mjs', 'maplibre-gl-shared.mjs', 'maplibre-gl-worker.mjs', 'maplibre-gl.css'], license: 'LICENSE.txt' },
  { name: '@protomaps/basemaps', version: '5.7.2', files: ['basemaps.js'], license: 'LICENSE' }
];
const integrity = new Map();
for (const pkg of libraries) {
  const installed = JSON.parse(await readFile(`${root}node_modules/${pkg.name}/package.json`));
  if (installed.version !== pkg.version) throw new Error(`Expected ${pkg.name}@${pkg.version}; run npm ci.`);
  const dir = `${output}/vendor/${pkg.name}/${pkg.version}`;
  await mkdir(dir, { recursive: true });
  for (const file of pkg.files) {
    const bytes = await readFile(`${root}node_modules/${pkg.name}/dist/${file}`);
    await writeFile(`${dir}/${file}`, bytes);
    integrity.set(`/vendor/${pkg.name}/${pkg.version}/${file}`, 'sha384-' + createHash('sha384').update(bytes).digest('base64'));
  }
  await cp(pkg.name === '@protomaps/basemaps' ? `${root}vendor/licenses/protomaps-basemaps.txt` : `${root}node_modules/${pkg.name}/${pkg.license}`, `${dir}/${pkg.license}`);
}
// Static Protomaps v4 sprites are downloaded once and kept as public assets.
await cp(`${root}beta/public`, output, { recursive: true });
const urls = new Map();
for (const name of ['csv', 'chart-layers', 'viewer', 'boot']) {
  let source = await readFile(`${root}beta/src/${name}.js`, 'utf8');
  for (const [dependency, url] of urls) source = source.replaceAll(`'./${dependency}.js'`, `'./${url.split('/').pop()}'`);
  const hash = createHash('sha256').update(source).digest('hex').slice(0, 16);
  const url = `/assets/${name}.${hash}.js`;
  urls.set(name, url);
  await writeFile(output + url, source);
}
let html = await readFile(`${root}index.html`, 'utf8');
html = html.replace('<head>', '<head>\n  <meta name="robots" content="noindex, nofollow" />');
html = html.replace('<title>', '<title>MapLibre Beta — ');
html = html.replace(/href="\/(atc\/|about|sources|contribute)"/g, 'href="https://archive.aero/$1"');
html = html.replace(/\s*<!-- Privacy-friendly analytics by Plausible -->\s*<script[^>]*plausible\.io[^>]*><\/script>\s*<script>[\s\S]*?<\/script>/, '');
html = html.replace('<div class="header-right">', '<div class="header-right">\n      <a href="https://archive.aero/" class="about-btn beta-badge" aria-label="MapLibre beta. Visit the live site.">MAPLIBRE BETA</a>');
html = html.replace(/\s*<script[^>]*src="\/vendor\/leaflet[^>]*><\/script>/, '');
html = html.replace(/<script defer src="\/vendor\/protomaps-leaflet[\s\S]*?<\/script>/, `<script defer src="/vendor/@protomaps/basemaps/5.7.2/basemaps.js" crossorigin="anonymous" integrity="${integrity.get('/vendor/@protomaps/basemaps/5.7.2/basemaps.js')}"></script>`);
html = html.replace(/<link rel="stylesheet" href="\/vendor\/leaflet[\s\S]*?\/>/, `<link rel="stylesheet" href="/vendor/maplibre-gl/6.11.2/maplibre-gl.css" crossorigin="anonymous" integrity="${integrity.get('/vendor/maplibre-gl/6.11.2/maplibre-gl.css')}" />`);
html = html.replace(/(data-viewer-preload href=")[^"]*/, '$1' + urls.get('viewer'));
html = html.replace(/(data-viewer-entry src=")[^"]*/, '$1' + urls.get('boot'));
html = html.replace('href="styles.css"', 'href="/styles.css"');
html = html.replace('Focus the timeline to step through dates.', 'Focus the timeline to step through dates. Use two fingers or Ctrl + drag to rotate and tilt the map; click the compass to reset.');
html = html.replaceAll('leaflet-control', 'maplibregl-ctrl');
html = html.replace('id="map"', 'id="map" data-renderer="maplibre"');
await writeFile(`${output}/index.html`, html);
let css = await readFile(`${root}styles.css`, 'utf8');
// Beta uses the same interface design, with native MapLibre control selectors.
css = css.replace(/\/\* Map Performance Tweaks \*\/[\s\S]*?(?=button,)/, '');
css = css.replaceAll('.leaflet-control-zoom a', '.maplibregl-ctrl-group button')
  .replaceAll('.leaflet-control-zoom-in', '.maplibregl-ctrl-zoom-in')
  .replaceAll('.leaflet-control-zoom-out', '.maplibregl-ctrl-zoom-out')
  .replaceAll('.leaflet-control-zoom', '.maplibregl-ctrl-group')
  .replaceAll('.leaflet-control-attribution', '.maplibregl-ctrl-attrib')
  .replaceAll('.leaflet-top.leaflet-right', '.maplibregl-ctrl-top-right')
  .replaceAll('.leaflet-top', '.maplibregl-ctrl-top-right')
  .replaceAll('.leaflet-control', '.maplibregl-ctrl')
  .replaceAll('.leaflet-tooltip-top.af-tip::before', '.af-tip .maplibregl-popup-tip')
  .replaceAll('.leaflet-tooltip.af-tip', '.af-tip .maplibregl-popup-content')
  .replaceAll('Leaflet Controls Styling', 'MapLibre Controls Styling');
css += '\n' + await readFile(`${root}beta/maplibre.css`, 'utf8');
await writeFile(`${output}/styles.css`, css);
await writeFile(`${output}/robots.txt`, 'User-agent: *\nDisallow: /\n');
await writeFile(`${output}/_headers`, '/*\n  X-Robots-Tag: noindex, nofollow\n/\n  Cache-Control: no-cache\n/index.html\n  Cache-Control: no-cache\n/styles.css\n  Cache-Control: no-cache\n/dates.csv\n  Cache-Control: no-cache\n/coverage.json\n  Cache-Control: no-cache\n/assets/*\n  Cache-Control: public, max-age=31536000, immutable\n/vendor/*\n  Cache-Control: public, max-age=31536000, immutable\n/sprites/*\n  Cache-Control: public, max-age=86400\n');
await writeFile(`${output}/404.html`, '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="robots" content="noindex"><title>Not found — archive.aero beta</title><h1>Page not found</h1><p><a href="/">Return to the beta map</a></p></html>');
const deployed = await readFile(`${output}/index.html`, 'utf8');
if (/leaflet|protomapsL/.test(deployed)) throw new Error('Legacy map dependency in beta HTML');
console.log(`MapLibre beta built in ${output}`);
