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
const result = await build({ ...common, splitting: true, entryPoints: { app: path.join(directory, 'app/main.js') }, define: { __MANIFEST_URL__: JSON.stringify(manifestUrl), __WORKER_URL__: JSON.stringify(`./${workerName}`), __STUBS__: String(stubs) }, plugins: [{ name: 'hashed-data-worker', setup(builder) { builder.onLoad({ filter: /next\/dataplane\/index\.js$/ }, async ({ path: filename }) => ({ contents: (await readFile(filename, 'utf8')).replace(/new URL\(['"]\.\/worker\.js['"],\s*import\.meta\.url\)/g, `new URL('./${workerName}', import.meta.url)`), loader: 'js' })); } }], alias: { '@next/renderer': path.join(directory, stubs ? 'app/stubs/renderer.js' : 'renderer/index.js'), '@next/dataplane': path.join(directory, stubs ? 'app/stubs/dataplane.js' : 'dataplane/index.js') } });
for (const file of result.outputFiles) outputs.set(path.relative(outdir, file.path), file.contents);
const app = path.relative(outdir, result.outputFiles.find(f => /app\..*\.js$/.test(f.path)).path);
const css = (await transform(await readFile(path.join(directory, 'shell/styles.css'), 'utf8'), { loader: 'css', minify: true })).code;
const cssName = `styles.${hash(css)}.css`; outputs.set(cssName, css);
const critical = 'html,body{height:100%;margin:0;background:#0a0e12;color:white;font:14px system-ui;overflow:hidden}[hidden]{display:none!important}.header-bar{height:64px;display:flex;justify-content:space-between}#main-content{height:calc(100% - 64px);position:relative}#map{position:absolute;inset:0}#mapCanvas{width:100%;height:100%;opacity:0}.controls-card{position:absolute;bottom:20px;left:20px;right:20px}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}.skip-link{position:fixed;transform:translateY(-150%)}';
const start = source.eras.map(e => e.k.split('_to_')[0]).sort().at(-1);
const newest = source.eras.filter(e => { const [a, b] = e.k.split('_to_'); return a <= start && start < b; }).map(e => ({ p: `sectionals/${e.k}.${e.h}`, b: e.b, z: e.z }));
const base = new URL(source.tileBase, new URL(manifestUrl, 'https://archive.aero/next/')).href;
const early = (await transform(await readFile(path.join(directory, 'shell/early.js'), 'utf8'), { minify: true, target: 'es2022' })).code.trim();
if (Buffer.byteLength(early) > 1536) throw new Error(`Boot script exceeds 1.5 KB: ${Buffer.byteLength(early)}`);
const bootData = JSON.stringify({ base: stubs ? './t/' : base, eras: newest, start, map: source.basemap }).replaceAll('<', '\\u003c');
const json = JSON.stringify(manifestUrl).replaceAll('<', '\\u003c');
const escape = text => text.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;');
const chunks = result.outputFiles.filter(f => f.path.endsWith('.js')).map(f => path.relative(outdir, f.path));
const head = `<style>${critical}</style><link rel="preload" as="fetch" href="${escape(manifestUrl)}" crossorigin><script id=bootData type=application/json>${bootData}</script><script>${early}</script>${chunks.map(name => `<link rel="modulepreload" href="./${name}">`).join('')}<link rel="stylesheet" href="./${cssName}">`;
const analytics = '<script async src="https://plausible.io/js/pa-m9DzDEgB7Ebb2zKdbBaMF.js"></script><script>window.plausible=window.plausible||function(){(plausible.q=plausible.q||[]).push(arguments)};plausible.init=plausible.init||function(i){plausible.o=i||{}};plausible.init()</script>';
const html = (await readFile(path.join(directory, 'shell/template.html'), 'utf8')).replace('<!--HEAD-->', head).replace('<!--APP-->', `<script type="module" src="./${app}"></script>${analytics}`).replace(/\s+/g, ' ').replace(/> </g, '><').replace(/="([a-zA-Z0-9_:.\/-]+)"/g, '=$1').trim() + '\n';
outputs.set('index.html', html);
const shellAssets = ['./index.html', ...[...outputs.keys()].filter(name => /\.(js|css)$/.test(name)).map(name => `./${name}`), manifestUrl];
const version = hash([...outputs].map(([name, data]) => `${name}:${hash(data)}`).join('|'));
const sw = (await transform((await readFile(path.join(directory, 'sw.js'), 'utf8')).replace('__SW_VERSION__', JSON.stringify(version)).replace('__SW_ASSETS__', JSON.stringify(shellAssets)).replace('__SW_MANIFEST__', json), { minify: true, target: 'es2022' })).code;
outputs.set('sw.js', sw);
const scriptHashes = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => `'sha256-${createHash('sha256').update(m[1]).digest('base64')}'`).join(' ');
const csp = `default-src 'self'; script-src 'self' ${scriptHashes} https://plausible.io; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self' https://data.archive.aero https://get.geojs.io https://plausible.io; worker-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'`;
const immutableHeaders = [...outputs.keys()].filter(name => name !== 'sw.js' && /\.(js|css)$/.test(name)).map(name => `${deployBase}${name}\n  Cache-Control: public, max-age=31536000, immutable\n`).join('');
// Cloudflare reads _headers only from the asset root, so this file is published there with rules already prefixed by the base.
outputs.set('_headers', `${deployBase}*\n  Content-Security-Policy: ${csp}\n  X-Content-Type-Options: nosniff\n${immutableHeaders}${deployBase}manifest.*.json\n  Cache-Control: public, max-age=31536000, immutable\n${deployBase}\n  Cache-Control: no-cache\n${deployBase}index.html\n  Cache-Control: no-cache\n${deployBase}sw.js\n  Cache-Control: no-cache\n`);
const externalJS = chunks.reduce((sum, name) => sum + gzipSync(outputs.get(name)).length, 0);
const inlineJS = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].reduce((sum, match) => sum + gzipSync(match[1]).length, 0);
const js = externalJS + inlineJS + gzipSync(worker.outputFiles.find(f => f.path.endsWith('.js')).contents).length;
const sizes = { criticalJS: js, css: gzipSync(css).length, html: Buffer.byteLength(html), htmlGzip: gzipSync(html).length, worker: gzipSync(worker.outputFiles.find(f => f.path.endsWith('.js')).contents).length, inlineBoot: Buffer.byteLength(early) };
// HTML carries the newest frame's era metadata inline; 12 KiB leaves room for several overlapping eras (gzip stays near 4 KB).
if (sizes.criticalJS > 50 * 1024 || sizes.css > 12 * 1024 || sizes.html > 12 * 1024) throw new Error(`Build budgets exceeded: ${JSON.stringify(sizes)}`);
outputs.set('budgets.json', JSON.stringify({ mode: stubs ? 'stubs' : 'real', manifestUrl, base: deployBase, allowOverBudget, manifestGzip, bytes: sizes, limits: { criticalJS: 51200, css: 12288, html: 12288 }, assets: shellAssets }, null, 2) + '\n');
const expected = new Set(outputs.keys());
const existing = await readdir(outdir).catch(e => { if (e.code !== 'ENOENT') throw e; return []; });
for (const [name, bytes] of outputs) {
  const filename = path.join(outdir, name), current = await readFile(filename).catch(e => { if (e.code !== 'ENOENT') throw e; return null; });
  if (current?.equals(Buffer.from(bytes))) continue;
  if (check) throw new Error(`next/dist/${name} is stale or missing. Build with the same --stubs / --manifest arguments and commit outputs.`);
  await mkdir(path.dirname(filename), { recursive: true }); await writeFile(filename, bytes);
}
for (const name of existing) if (!expected.has(name)) { if (check) throw new Error(`Unexpected stale output next/dist/${name}`); await rm(path.join(outdir, name), { recursive: true }); }
console.log(`${check ? 'Verified' : 'Built'} ${stubs ? 'stub' : 'real'} shell: ${JSON.stringify(sizes)}`);
