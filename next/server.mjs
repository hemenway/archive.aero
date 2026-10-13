import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(fileURLToPath(new URL('./dist/', import.meta.url)));
http.createServer(async (request, response) => {
  const url = new URL(request.url, 'http://127.0.0.1:4183');
  // An edition address is the shell page; deployed, next/pages/worker.mjs renders it for the chart first.
  const pathname = /^\/next\/(?:sectionals\/chart\/[a-z0-9_]+\/\d{4}-\d{2}-\d{2}|airports\/[a-z0-9]{3,4})$/.test(url.pathname) ? '/next/index.html' : url.pathname;
  const relative = decodeURIComponent(pathname).replace(/^\/next\//, '/').replace(/^\//, '') || 'index.html';
  const filename = path.resolve(root, relative);
  if (url.searchParams.has('__testOffline')) { response.writeHead(503); response.end('Injected navigation failure'); return; }
  if (filename !== root && !filename.startsWith(root + path.sep)) { response.writeHead(403); response.end(); return; }
  if (url.pathname.includes('/t/')) { response.writeHead(200, { 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=31536000, immutable' }); response.end(Buffer.from('stub worker synthesizes pixels')); return; }
  try { const data = await readFile(filename); response.writeHead(200, { 'Content-Type': ({ '.html': 'text/html', '.js': 'application/javascript', '.json': 'application/json', '.css': 'text/css', '.map': 'application/json', '.woff2': 'font/woff2' })[path.extname(filename)] || 'application/octet-stream', 'Cache-Control': 'no-cache' }); response.end(data); }
  catch { response.writeHead(404); response.end('Not found'); }
}).listen(4183, '127.0.0.1', () => console.log('Stub shell: http://127.0.0.1:4183/next/'));
