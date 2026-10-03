import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, extname, sep } from 'node:path';
const root = fileURLToPath(new URL('.', import.meta.url));
http.createServer(async (req, res) => {
  const path = resolve(root, '.' + decodeURIComponent(new URL(req.url, 'http://localhost').pathname === '/' ? '/demo.html' : new URL(req.url, 'http://localhost').pathname));
  if (!path.startsWith(root.endsWith(sep) ? root : root + sep)) { res.writeHead(403).end(); return; }
  try { const body = await readFile(path); res.writeHead(200, { 'Content-Type': ({ '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript' })[extname(path)] ?? 'application/octet-stream', 'Cache-Control': 'no-store' }).end(body); }
  catch { res.writeHead(404).end(); }
}).listen(4181, '127.0.0.1');
