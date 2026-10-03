import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(fileURLToPath(new URL('./dist/', import.meta.url)));
http.createServer(async (request, response) => {
  const url = new URL(request.url, 'http://127.0.0.1:4183');
  const relative = decodeURIComponent(url.pathname).replace(/^\/next\//, '/').replace(/^\//, '') || 'index.html';
  const filename = path.resolve(root, relative);
  if (url.searchParams.has('__testOffline')) { response.writeHead(503); response.end('Injected navigation failure'); return; }
  if (filename !== root && !filename.startsWith(root + path.sep)) { response.writeHead(403); response.end(); return; }
  if (url.pathname.includes('/t/')) { response.writeHead(200, { 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=31536000, immutable' }); response.end(Buffer.from('stub worker synthesizes pixels')); return; }
  try { const data = await readFile(filename); response.writeHead(200, { 'Content-Type': ({ '.html': 'text/html', '.js': 'application/javascript', '.json': 'application/json', '.css': 'text/css', '.map': 'application/json' })[path.extname(filename)] || 'application/octet-stream', 'Cache-Control': 'no-cache' }); response.end(data); }
  catch { response.writeHead(404); response.end('Not found'); }
}).listen(4183, '127.0.0.1', () => console.log('Stub shell: http://127.0.0.1:4183/next/'));
