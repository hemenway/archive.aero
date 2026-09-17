// Deliberately expose only the pages needed by tests, never local catalogs or .env.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';

const files = new Map([
  ['/', ['index.html', 'text/html']],
  ['/index.html', ['index.html', 'text/html']],
  ['/styles.css', ['styles.css', 'text/css']],
  ['/tests/flicker-regression-guard.html', ['tests/flicker-regression-guard.html', 'text/html']],
]);
createServer(async (req, res) => {
  const entry = files.get(new URL(req.url, 'http://localhost').pathname);
  if (!entry) { res.writeHead(404).end(); return; }
  try {
    const body = await readFile(new URL('../' + entry[0], import.meta.url));
    res.writeHead(200, { 'content-type': entry[1], 'cache-control': 'no-store' }).end(body);
  } catch { res.writeHead(500).end(); }
}).listen(4173, '127.0.0.1');
