// Deliberately expose only the pages needed by tests, never local catalogs or .env.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const files = new Map([
  ['/', ['index.html', 'text/html']],
  ['/index.html', ['index.html', 'text/html']],
  ['/styles.css', ['styles.css', 'text/css']],
  ['/tests/flicker-regression-guard.html', ['tests/flicker-regression-guard.html', 'text/html']],
]);
// Serve only committed module/vendor assets, never the rest of the workspace.
const vendor = JSON.parse(await readFile(new URL('../vendor/manifest.json', import.meta.url)));
for (const url of Object.keys(vendor)) {
  const type = url.endsWith('.js') ? 'text/javascript' : url.endsWith('.css') ? 'text/css'
    : url.endsWith('.png') ? 'image/png' : 'application/json';
  files.set(url, [url.slice(1), type]);
}
// The guard extracts source classes; runtime tests execute the generated module.
files.set('/src/viewer.js', ['src/viewer.js', 'text/javascript']);
createServer(async (req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  const entry = files.get(pathname) || (/^\/assets\/(boot|viewer|csv)\.[a-f0-9]{16}\.js$/.test(pathname)
    ? [pathname.slice(1), 'text/javascript'] : null);
  if (!entry) { res.writeHead(404).end(); return; }
  try {
    const body = await readFile(new URL('../' + entry[0], import.meta.url));
    const versioned = pathname.startsWith('/assets/') || pathname.startsWith('/vendor/');
    const headers = { 'content-type': entry[1], 'cache-control': versioned ? 'public, max-age=7200' : 'no-store',
      etag: '"' + createHash('sha256').update(body).digest('hex') + '"' };
    if (req.headers['if-none-match'] === headers.etag) res.writeHead(304, headers).end();
    else res.writeHead(200, headers).end(body);
  } catch { res.writeHead(500).end(); }
}).listen(4173, '127.0.0.1');
