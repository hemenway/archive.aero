// Deploy-time only: minify the staged copies of the fingerprinted viewer
// modules in _site/. The committed assets/ stay readable, and a fingerprint
// still names its source, not these bytes. Vendor files carry SRI under
// unversioned URLs, so they are never touched here.
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { transform } from 'esbuild';

const site = path.resolve(process.argv[2] || fileURLToPath(new URL('../_site', import.meta.url)));
const assets = path.join(site, 'assets');
const names = (await readdir(assets)).filter(name => /^(boot|viewer|csv)\.[a-f0-9]{16}\.js$/.test(name)).sort();
if (!names.length) throw new Error(`No fingerprinted modules in ${assets}`);
// Minified modules import each other by name, so every export must survive.
const exported = code => [...code.matchAll(/\bexport\s*(?:async\s+)?(?:function\*?|class|const|let|var)\s+([\w$]+)/g)]
  .map(match => match[1]).concat([...code.matchAll(/\bexport\s*\{([^}]*)\}/g)]
    .flatMap(match => match[1].split(',').map(part => part.trim().split(/\s+as\s+/).pop()).filter(Boolean))).sort();
let before = 0;
let after = 0;
for (const name of names) {
  const filename = path.join(assets, name);
  const source = await readFile(filename, 'utf8');
  const { code } = await transform(source, { minify: true, format: 'esm', legalComments: 'none', sourcefile: name });
  if (exported(code).join() !== exported(source).join()) throw new Error(`${name}: exports changed during minification`);
  await writeFile(filename, code);
  before += Buffer.byteLength(source);
  after += Buffer.byteLength(code);
}
console.log(`Minified ${names.length} modules in ${path.relative(process.cwd(), assets) || assets}: ${before} -> ${after} bytes.`);
