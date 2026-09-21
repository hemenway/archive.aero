// No bundler/transpiler: fingerprint the small native-module graph and commit
// the outputs with index.html. --check is read-only and runs in CI.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const args = process.argv.slice(2);
const rootArg = args.indexOf('--root');
const root = rootArg >= 0 ? path.resolve(args[rootArg + 1]) : fileURLToPath(new URL('../', import.meta.url));
const check = args.includes('--check');
const outputs = new Map();
const urls = new Map();
for (const name of ['csv', 'viewer', 'boot']) {
  let source = await readFile(path.join(root, 'src', `${name}.js`), 'utf8');
  for (const [dependency, url] of urls) {
    source = source.replaceAll(`'./${dependency}.js'`, `'./${path.basename(url)}'`);
  }
  const hash = createHash('sha256').update(source).digest('hex').slice(0, 16);
  const url = `/assets/${name}.${hash}.js`;
  urls.set(name, url);
  outputs.set(url.slice(1), source);
}
const html = await readFile(path.join(root, 'index.html'), 'utf8');
let generated = html;
for (const [marker, attr, name] of [['data-viewer-preload', 'href', 'viewer'], ['data-viewer-entry', 'src', 'boot']]) {
  const pattern = new RegExp(`(<[^>]*\\b${marker}\\b[^>]*\\b${attr}=")[^"]*(")`, 'g');
  if ([...generated.matchAll(pattern)].length !== 1) throw new Error(`Expected one ${marker} in index.html`);
  generated = generated.replace(pattern, (_, a, b) => a + urls.get(name) + b);
}
outputs.set('index.html', generated);
for (const [relative, contents] of outputs) {
  const filename = path.join(root, relative);
  const current = await readFile(filename, 'utf8').catch(error => {
    if (error.code !== 'ENOENT') throw error;
    return null;
  });
  if (current === contents) continue;
  if (check) throw new Error(`${relative} is stale or missing. Run npm run build:frontend and commit the outputs.`);
  await mkdir(path.dirname(filename), { recursive: true });
  await writeFile(filename, contents);
}
console.log(check ? 'Frontend fingerprints verified.' : 'Frontend modules fingerprinted. Keep prior deployed hashes for cached HTML.');
