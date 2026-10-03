import { build, transform } from 'esbuild';
import { readFile, writeFile, mkdir, readdir, rm, access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { manifest as fixture } from './app/stubs/manifest.js';
const root = fileURLToPath(new URL('../', import.meta.url)), directory = path.join(root, 'next');
const args = process.argv.slice(2), check = args.includes('--check');
const value = flag => { const i = args.indexOf(flag); if (i < 0) return null; if (!args[i + 1] || args[i + 1].startsWith('--')) throw new Error(`${flag} needs a value`); return args[i + 1]; };
// --outdir lets a real build land in a hosting tree (the beta's dist/next/) while next/dist keeps the stub build the tests use.
const outdir = value('--outdir') ? path.resolve(value('--outdir')) : path.join(directory, 'dist');
const previousBuild = check ? JSON.parse(await readFile(path.join(outdir, 'budgets.json'), 'utf8').catch(() => 'null')) : null;
// The 80 KB C2 budget is a contract target; the owner may ship an over-budget manifest while the shard format is decided (2026-10-02).
const allowOverBudget = args.includes('--allow-over-budget') || !!previousBuild?.allowOverBudget;
// A preview host: no page analytics, and site links (/about, /atc/...) point at the origin that has those pages.
const analyticsOn = !(args.includes('--no-analytics') || previousBuild?.analytics === false);
const linksOrigin = value('--links-origin') || previousBuild?.linksOrigin || null;
if (linksOrigin && !/^https:\/\/[\w.-]+$/.test(linksOrigin)) throw new Error(`--links-origin must be an https origin, got ${linksOrigin}`);
const modulesExist = await Promise.all(['renderer/index.js', 'dataplane/index.js'].map(name => access(path.join(directory, name)).then(() => true, () => false)));
const stubs = args.includes('--stubs') || (!args.includes('--manifest') && (previousBuild?.mode === 'stubs' || !modulesExist.every(Boolean)));
const hash = bytes => createHash('sha256').update(bytes).digest('hex').slice(0, 12);
const outputs = new Map();
let source, manifestUrl = value('--manifest') || previousBuild?.manifestUrl;
// Deployed URL prefix of this shell; header rules are scoped to it so a /next/ preview never governs the root viewer.
const deployBase = value('--base') || previousBuild?.base || '/next/';
if (!/^\/(?:[\w-]+\/)*$/.test(deployBase)) throw new Error(`--base must look like /next/ or /, got ${deployBase}`);
if (stubs) {
  source = fixture; const bytes = JSON.stringify(source); const name = `manifest.${hash(bytes)}.json`; outputs.set(name, bytes); manifestUrl ||= `./${name}`;
} else {
  if (!manifestUrl) throw new Error('Real builds require --manifest <immutable URL>. Use --stubs for the standalone shell.');
  if (value('--manifest-file')) source = JSON.parse(await readFile(path.resolve(value('--manifest-file')), 'utf8'));
  else { const response = await fetch(manifestUrl); if (!response.ok) throw new Error(`Manifest build fetch failed: ${response.status}`); source = await response.json(); }
}
if (source.version !== 1 || !source.eras?.length) throw new Error('Invalid C2 manifest');
const manifestGzip = gzipSync(JSON.stringify(source)).length;
if (manifestGzip > 80 * 1024 && !allowOverBudget) throw new Error(`Manifest exceeds 80 KB gzip (${manifestGzip} bytes); pass --allow-over-budget to ship it anyway`);
const common = { absWorkingDir: root, bundle: true, minify: true, format: 'esm', target: 'es2022', sourcemap: 'external', write: false, outdir, entryNames: '[name].[hash]', chunkNames: 'chunk.[hash]', metafile: true };
const workerEntry = path.join(directory, stubs ? 'app/stubs/worker.js' : 'dataplane/worker.js');
const worker = await build({ ...common, entryPoints: { worker: workerEntry } });
for (const file of worker.outputFiles) outputs.set(path.relative(outdir, file.path), file.contents);
const workerName = path.relative(outdir, worker.outputFiles.find(f => f.path.endsWith('.js')).path);
const result = await build({ ...common, splitting: true, entryPoints: { app: path.join(directory, 'app/main.js') }, define: { __MANIFEST_URL__: JSON.stringify(manifestUrl), __SITE_ORIGIN__: JSON.stringify(linksOrigin || ''), __WORKER_URL__: JSON.stringify(`./${workerName}`), __STUBS__: String(stubs) }, plugins: [{ name: 'hashed-data-worker', setup(builder) { builder.onLoad({ filter: /next\/dataplane\/index\.js$/ }, async ({ path: filename }) => ({ contents: (await readFile(filename, 'utf8')).replace(/new URL\(['"]\.\/worker\.js['"],\s*import\.meta\.url\)/g, `new URL('./${workerName}', import.meta.url)`), loader: 'js' })); } }], alias: { '@next/renderer': path.join(directory, stubs ? 'app/stubs/renderer.js' : 'renderer/index.js'), '@next/dataplane': path.join(directory, stubs ? 'app/stubs/dataplane.js' : 'dataplane/index.js') } });
for (const file of result.outputFiles) outputs.set(path.relative(outdir, file.path), file.contents);
const app = path.relative(outdir, result.outputFiles.find(f => /app\..*\.js$/.test(f.path)).path);
// The shell wears the production interface. Its page and stylesheet are derived from the live viewer's index.html and
// styles.css at the repository root, so an interface change is made once, there (then rebuild and commit next/dist).
// Each edit says how many places it expects to touch: a production change it no longer fits stops the build instead of
// shipping a half-converted page.
const edit = (text, what, pattern, count, replacement) => {
  const found = [...text.matchAll(pattern)].length;
  if (found !== count) throw new Error(`Shell derivation: expected ${count} x ${what}, found ${found}. Bring next/build.mjs in line with the production page.`);
  return text.replace(pattern, replacement);
};
// Barlow is production's display face. The shell's CSP has font-src 'self', so the latin files ship with the build
// (content-hashed, immutable) instead of coming from Google Fonts. The service worker does not precache them.
const fontPackage = '@fontsource/barlow', fontDir = path.join(root, 'node_modules', fontPackage);
const fontPinned = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8')).devDependencies?.[fontPackage];
const fontInstalled = JSON.parse(await readFile(path.join(fontDir, 'package.json'), 'utf8').catch(() => '{}')).version;
if (!fontPinned || fontInstalled !== fontPinned) throw new Error(`Expected ${fontPackage}@${fontPinned} in node_modules, found ${fontInstalled || 'none'}; run npm ci.`);
let fontFaces = '', fontBytes = 0;
for (const weight of [500, 600, 700]) {
  const bytes = await readFile(path.join(fontDir, `files/barlow-latin-${weight}-normal.woff2`));
  // <weight>.css lists every subset; its latin block has the unicode-range (latin-<weight>.css leaves it out).
  const range = (await readFile(path.join(fontDir, `${weight}.css`), 'utf8')).match(new RegExp(`/\\* barlow-latin-${weight}-normal \\*/\\s*@font-face\\s*\\{[^}]*?unicode-range:\\s*([^;}]+)`))?.[1].trim();
  if (!range) throw new Error(`${fontPackage}/${weight}.css has no latin unicode-range`);
  const name = `fonts/barlow-${weight}.${hash(bytes)}.woff2`; outputs.set(name, bytes); fontBytes += bytes.length;
  fontFaces += `@font-face{font-family:"Barlow";font-style:normal;font-weight:${weight};font-display:swap;src:url("./${name}") format("woff2");unicode-range:${range}}\n`;
}
outputs.set('fonts/LICENSE', await readFile(path.join(fontDir, 'LICENSE')));
// Stylesheet: production's styles.css as it is, then what it took for granted from Leaflet plus the canvas (shell/next.css),
// then Barlow. One production selector matches a site link by its href, so it follows --links-origin like the link does.
const productionCss = await readFile(path.join(root, 'styles.css'), 'utf8');
const siteCss = edit(productionCss, 'href selector on a site link', /\[href="\/(atc\/|about|sources|contribute)"\]/g, 1, linksOrigin ? `[href="${linksOrigin}/$1"]` : '$&');
const css = (await transform(`${siteCss}\n${await readFile(path.join(directory, 'shell/next.css'), 'utf8')}\n${fontFaces}`, { loader: 'css', minify: true })).code;
const cssName = `styles.${hash(css)}.css`; outputs.set(cssName, css);
// Inline CSS is only what must hold before, or without, the stylesheet: the page colours and a hidden canvas. The colours
// are read from styles.css so the two cannot disagree.
const pageColor = (property, variable) => {
  const value = productionCss.match(new RegExp(`${variable}:\\s*(#[0-9a-fA-F]{3,8});`))?.[1];
  if (!value || !new RegExp(`html,\\s*body\\s*\\{[^}]*\\b${property}:\\s*var\\(${variable}\\)`).test(productionCss)) throw new Error(`Shell derivation: styles.css no longer sets html, body ${property} from a hex ${variable}`);
  return value;
};
const critical = `html,body{height:100%;margin:0;background:${pageColor('background', '--bg-dark')};color:${pageColor('color', '--text-light')}}#mapCanvas{opacity:0}`;
const start = source.eras.map(e => e.k.split('_to_')[0]).sort().at(-1);
const newest = source.eras.filter(e => { const [a, b] = e.k.split('_to_'); return a <= start && start < b; }).map(e => ({ p: `sectionals/${e.k}.${e.h}`, b: e.b, z: e.z }));
const base = new URL(source.tileBase, new URL(manifestUrl, 'https://archive.aero/next/')).href;
const early = (await transform(await readFile(path.join(directory, 'shell/early.js'), 'utf8'), { minify: true, target: 'es2022' })).code.trim();
if (Buffer.byteLength(early) > 1536) throw new Error(`Boot script exceeds 1.5 KB: ${Buffer.byteLength(early)}`);
const bootData = JSON.stringify({ base: stubs ? './t/' : base, eras: newest, start, map: source.basemap }).replaceAll('<', '\\u003c');
const json = JSON.stringify(manifestUrl).replaceAll('<', '\\u003c');
const escape = text => text.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;');
const attr = value => /^[a-zA-Z0-9_:.\/-]+$/.test(value) ? value : `"${escape(value)}"`;
// Only the entry's static imports are on the first-paint path. A chunk reached through import() (the cards) is emitted,
// cached and precached like the rest, but is neither preloaded nor counted as critical JS. Metafile paths are relative to
// absWorkingDir, and --outdir may lie outside it, so both sides are brought to outdir-relative names.
const emitted = name => path.relative(outdir, path.resolve(root, name));
const graph = new Map(Object.entries(result.metafile.outputs).map(([name, output]) => [emitted(name), output.imports.filter(i => i.kind === 'import-statement' && !i.external).map(i => emitted(i.path))]));
const eager = new Set([app]);
for (const name of eager) { if (!graph.has(name)) throw new Error(`esbuild metafile has no output ${name}`); for (const imported of graph.get(name)) eager.add(imported); }
const scripts = result.outputFiles.filter(f => f.path.endsWith('.js')).map(f => path.relative(outdir, f.path));
const chunks = scripts.filter(name => eager.has(name)), lazyChunks = scripts.filter(name => !eager.has(name));
const head = `<style>${critical}</style><link rel=preload as=fetch href=${attr(manifestUrl)} crossorigin><script id=bootData type=application/json>${bootData}</script><script>${early}</script>${chunks.map(name => `<link rel=modulepreload href=./${name}>`).join('')}<link rel=stylesheet href=./${cssName}>`;

// Page: the production index.html, minus what only the Leaflet viewer needs, with the map controls where Leaflet puts them.
// One tokenizer serves the minifier and its check. Attribute values are matched as quoted strings, so the favicon's data
// URI (it contains < > ' and =) and the SVG paths are opaque to every step.
const TAG = /<(\/)?([a-zA-Z][\w-]*)((?:\s+[^\s"'<>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?)*)\s*(\/)?>/g, ATTR = /([^\s"'<>\/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
const tokenize = markup => {
  const list = []; let at = 0;
  const text = end => { if (end === at) return; const run = markup.slice(at, end), bad = run.indexOf('<'); if (bad >= 0) throw new Error(`Shell derivation: unparsed markup near ${JSON.stringify(run.slice(bad, bad + 80))}`); list.push({ text: run }); };
  for (const m of markup.matchAll(TAG)) { text(m.index); list.push({ name: m[2], tag: m[2].toLowerCase(), close: !!m[1], self: !!m[4], attrs: [...m[3].matchAll(ATTR)].map(a => [a[1], a[2] ?? a[3] ?? a[4] ?? null]) }); at = m.index + m[0].length; }
  text(markup.length); return list;
};
const VOID = new Set('area base br col embed hr img input link meta source track wbr'.split(' '));
// Whitespace beside these tags never renders, so it is dropped. Beside anything else it is inline content: a run becomes
// one space and stays, which leaves every inline gap of the production page (header, layers panel, stats line) as it was.
const BLOCK = new Set('html head body title meta link base div p h1 h2 h3 h4 h5 h6 header footer main nav section article aside ul ol li details summary option br hr'.split(' '));
const quoted = value => value === null ? '' : /^[a-zA-Z0-9_:.\/-]+$/.test(value) ? `=${value}` : value.includes('"') ? `='${value}'` : `="${value}"`;
const minify = markup => {
  const list = tokenize(markup); let out = '', svg = 0;
  list.forEach((token, i) => {
    if ('text' in token) {
      let run = token.text.replace(/\s+/g, ' ');
      if (svg) { if (run !== ' ') throw new Error('Shell derivation: text inside <svg> is not supported'); return; }
      if (BLOCK.has(list[i - 1]?.tag ?? 'html')) run = run.trimStart();
      if (BLOCK.has(list[i + 1]?.tag ?? 'html')) run = run.trimEnd();
      out += run; return;
    }
    if (token.close) { if (token.tag === 'svg') svg--; out += `</${token.name}>`; return; }
    if (token.self && !svg && !VOID.has(token.tag)) throw new Error(`Shell derivation: <${token.name} /> is not a void element`);
    const attrs = token.attrs.map(([name, value]) => ` ${name}${quoted(value)}`).join('');
    // Inside <svg> the closing slash is syntax. After an unquoted value it keeps its space, or it would join the value.
    out += `<${token.name}${attrs}${token.self && svg ? (/["']$/.test(attrs) ? '/>' : ' />') : '>'}`;
    if (token.tag === 'svg' && !token.self) svg++;
  });
  return out;
};
const shape = markup => JSON.stringify(tokenize(markup).filter(t => t.tag || t.text.trim()).map(t => t.tag ? [t.close, t.name, t.attrs, t.self && !VOID.has(t.tag)] : t.text.trim().replace(/\s+/g, ' ')));

let page = edit((await readFile(path.join(root, 'index.html'), 'utf8')).replace(/<!--[\s\S]*?-->/g, ''), 'doctype', /^\s*<!doctype html>/gi, 1, '');
// Leaflet, its stylesheet and the Leaflet viewer's scripts give way to the head and app injected by this build; the
// stylesheet link is the hashed one above, the fonts are self-hosted, and analytics is re-added below unless --no-analytics.
page = edit(page, 'vendor script', /<script defer src="\/vendor\/[^"]+"[^>]*><\/script>/g, 3, '');
page = edit(page, 'viewer modulepreload', /<link rel="modulepreload" data-viewer-preload[^>]*>/g, 1, '');
page = edit(page, 'Leaflet stylesheet link', /<link rel="stylesheet" href="\/vendor\/leaflet\/[^"]+"[^>]*>/g, 1, '');
page = edit(page, 'styles.css link', /<link rel="stylesheet" href="styles\.css"[^>]*>/g, 1, '');
page = edit(page, 'Google Fonts preconnect', /<link rel="preconnect" href="https:\/\/fonts\.(?:googleapis|gstatic)\.com"[^>]*>/g, 2, '');
page = edit(page, 'Google Fonts stylesheet link', /<link rel="stylesheet" href="https:\/\/fonts\.googleapis\.com\/[^"]*"[^>]*>/g, 1, '');
page = edit(page, 'viewer entry script', /<script type="module" data-viewer-entry[^>]*><\/script>/g, 1, '');
let plausibleSrc;
page = edit(page, 'Plausible block', /<script async src="(https:\/\/plausible\.io\/[^"]+)"><\/script>\s*<script>[^<]*<\/script>/g, 1, (_, src) => { plausibleSrc = src; return ''; });
page = edit(page, 'site link', /href="\/(atc\/|about|sources|contribute)"/g, 10, linksOrigin ? `href="${linksOrigin}/$1"` : '$&');
// Lifts one element out of the page by its opening tag, balancing the <div>s nested in it.
const take = (what, opening) => {
  const start = page.indexOf(opening);
  if (start < 0 || page.includes(opening, start + 1)) throw new Error(`Shell derivation: expected exactly one ${what}`);
  const divs = /<div\b|<\/div>/g; divs.lastIndex = start;
  for (let depth = 0, m; (m = divs.exec(page));) if (!(depth += m[0] === '</div>' ? -1 : 1)) { const block = page.slice(start, divs.lastIndex); page = page.slice(0, start) + page.slice(divs.lastIndex); return block; }
  throw new Error(`Shell derivation: ${what} is not closed`);
};
// Production's initApp moves #toolsControl into Leaflet's top-right corner and wraps Leaflet's zoom control and #utilRail
// in .ctl-col (src/viewer.js). With no Leaflet to do that, the same structure is built here; the zoom markup is Leaflet 1.9.4's.
const toolsControl = take('#toolsControl', '<div class="tools-control leaflet-control" id="toolsControl">'), utilRail = take('#utilRail', '<div class="util-rail leaflet-control" id="utilRail">');
page = edit(page, '#map', /<div id="map" role="region"((?: [\w-]+="[^"]*")*)><\/div>/g, 1, (_, rest) => {
  if (/ tabindex=/.test(rest)) throw new Error('Shell derivation: #map already has a tabindex');
  return `<div id="map" role="region" tabindex="0"${rest}><canvas id="mapCanvas" aria-hidden="true"></canvas><div class="leaflet-control-container"><div class="leaflet-top leaflet-right">${toolsControl}<div class="ctl-col"><div class="leaflet-control-zoom leaflet-bar leaflet-control"><a class="leaflet-control-zoom-in" id="zoomInBtn" href="#" title="Zoom in" role="button" aria-label="Zoom in"><span aria-hidden="true">+</span></a><a class="leaflet-control-zoom-out" id="zoomOutBtn" href="#" title="Zoom out" role="button" aria-label="Zoom out"><span aria-hidden="true">&#x2212;</span></a></div>${utilRail}</div></div><div class="leaflet-bottom leaflet-right"><div class="leaflet-control-attribution leaflet-control" id="mapAttribution"></div></div></div></div>`;
});
const markup = minify(page);
if (shape(markup) !== shape(page)) throw new Error('Shell derivation: the minifier changed the markup');
// What is left must be static. The CSP admits only the scripts injected here, and no stylesheet or font but the build's own,
// so a script, inline handler or resource link that production gains has to be dealt with above, not carried along.
const tokens = tokenize(markup), headEnd = tokens.findIndex(t => t.tag === 'head' && t.close);
const stray = tokens.find((t, i) => t.tag ? !t.close && (t.tag === 'script' || t.tag === 'style' || t.attrs.some(([name]) => /^on/i.test(name)) || (t.tag === 'link' && !['canonical', 'icon', 'preconnect'].includes(t.attrs.find(([name]) => name === 'rel')?.[1]))) : i < headEnd && tokens[i - 1].tag !== 'title');
if (stray) throw new Error(`Shell derivation: the production page has content the shell does not account for: ${JSON.stringify(stray)}`);
const ids = ['bootData', ...tokens.flatMap(t => t.tag && !t.close ? t.attrs.filter(([name]) => name === 'id').map(([, value]) => value) : [])];
if (new Set(ids).size !== ids.length) throw new Error(`Shell derivation: duplicate element id ${[...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))].join(', ')}`);
const analytics = !analyticsOn ? '' : `<script async src=${attr(plausibleSrc)}></script><script>window.plausible=window.plausible||function(){(plausible.q=plausible.q||[]).push(arguments)};plausible.init=plausible.init||function(i){plausible.o=i||{}};plausible.init()</script>`;
const html = `<!doctype html>${edit(edit(markup, '</head>', /<\/head>/g, 1, () => `${head}</head>`), '</body>', /<\/body>/g, 1, () => `<script type=module src=./${app}></script>${analytics}</body>`)}\n`;
outputs.set('index.html', html);
const shellAssets = ['./index.html', ...[...outputs.keys()].filter(name => /\.(js|css)$/.test(name)).map(name => `./${name}`), manifestUrl];
const version = hash([...outputs].map(([name, data]) => `${name}:${hash(data)}`).join('|'));
const sw = (await transform((await readFile(path.join(directory, 'sw.js'), 'utf8')).replace('__SW_VERSION__', JSON.stringify(version)).replace('__SW_ASSETS__', JSON.stringify(shellAssets)).replace('__SW_MANIFEST__', json), { minify: true, target: 'es2022' })).code;
outputs.set('sw.js', sw);
const scriptHashes = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => `'sha256-${createHash('sha256').update(m[1]).digest('base64')}'`).join(' ');
// Data origins named by this build's manifest join connect-src (a preview or fixture host); production's stays listed.
const dataOrigins = [...new Set([manifestUrl, source.tileBase, source.fileBase].map(u => new URL(u, 'https://archive.aero/next/').origin))].filter(o => !['https://archive.aero', 'https://data.archive.aero'].includes(o));
const plausible = analyticsOn ? ' https://plausible.io' : '';
const csp = `default-src 'self'; script-src 'self' ${scriptHashes}${plausible}; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self' https://data.archive.aero ${dataOrigins.map(o => o + ' ').join('')}https://get.geojs.io${plausible}; worker-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'`;
const immutableHeaders = [...outputs.keys()].filter(name => name !== 'sw.js' && /\.(js|css)$/.test(name)).map(name => `${deployBase}${name}\n  Cache-Control: public, max-age=31536000, immutable\n`).join('');
// Cloudflare reads _headers only from the asset root, so this file is published there with rules already prefixed by the base.
outputs.set('_headers', `${deployBase}*\n  Content-Security-Policy: ${csp}\n  X-Content-Type-Options: nosniff\n${immutableHeaders}${deployBase}fonts/*\n  Cache-Control: public, max-age=31536000, immutable\n${deployBase}manifest.*.json\n  Cache-Control: public, max-age=31536000, immutable\n${deployBase}\n  Cache-Control: no-cache\n${deployBase}index.html\n  Cache-Control: no-cache\n${deployBase}sw.js\n  Cache-Control: no-cache\n`);
const externalJS = chunks.reduce((sum, name) => sum + gzipSync(outputs.get(name)).length, 0);
const inlineJS = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].reduce((sum, match) => sum + gzipSync(match[1]).length, 0);
const js = externalJS + inlineJS + gzipSync(worker.outputFiles.find(f => f.path.endsWith('.js')).contents).length;
const sizes = { criticalJS: js, css: gzipSync(css).length, cssRaw: Buffer.byteLength(css), html: Buffer.byteLength(html), htmlGzip: gzipSync(html).length, worker: gzipSync(worker.outputFiles.find(f => f.path.endsWith('.js')).contents).length, inlineBoot: Buffer.byteLength(early), lazyJS: lazyChunks.reduce((sum, name) => sum + gzipSync(outputs.get(name)).length, 0), fonts: fontBytes };
// HTML is the production page's markup plus the newest frame's era metadata inline; 24 KiB (uncompressed) leaves room for
// both to grow. The CSS limit is on gzip and covers production's styles.css, shell/next.css and the font faces.
if (sizes.criticalJS > 50 * 1024 || sizes.css > 12 * 1024 || sizes.html > 24 * 1024) throw new Error(`Build budgets exceeded: ${JSON.stringify(sizes)}`);
outputs.set('budgets.json', JSON.stringify({ mode: stubs ? 'stubs' : 'real', manifestUrl, base: deployBase, allowOverBudget, analytics: analyticsOn, linksOrigin, manifestGzip, bytes: sizes, limits: { criticalJS: 51200, css: 12288, html: 24576 }, assets: shellAssets }, null, 2) + '\n');
// Outputs can sit in a subdirectory (fonts/), so stale files are looked for through the whole tree; parents sort first.
const expected = new Set([...outputs.keys()].flatMap(name => name.split('/').map((_, i, parts) => parts.slice(0, i + 1).join('/'))));
const existing = (await readdir(outdir, { recursive: true }).catch(e => { if (e.code !== 'ENOENT') throw e; return []; })).map(name => name.split(path.sep).join('/')).sort();
for (const [name, bytes] of outputs) {
  const filename = path.join(outdir, name), current = await readFile(filename).catch(e => { if (e.code !== 'ENOENT') throw e; return null; });
  if (current?.equals(Buffer.from(bytes))) continue;
  if (check) throw new Error(`next/dist/${name} is stale or missing. Build with the same --stubs / --manifest arguments and commit outputs.`);
  await mkdir(path.dirname(filename), { recursive: true }); await writeFile(filename, bytes);
}
for (const name of existing) if (!expected.has(name)) { if (check) throw new Error(`Unexpected stale output next/dist/${name}`); await rm(path.join(outdir, name), { recursive: true, force: true }); }
console.log(`${check ? 'Verified' : 'Built'} ${stubs ? 'stub' : 'real'} shell: ${JSON.stringify(sizes)}`);
