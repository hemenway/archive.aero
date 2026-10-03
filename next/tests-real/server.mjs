// Real-module integration server. Canonical fixtures are served by the actual
// tiles Worker handler (port 4185); a real, non-stub shell is built against that
// fixture manifest and served under /next/ (port 4184) with the build's own
// Content-Security-Policy applied, so CSP regressions fail here too.
import http from 'node:http';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { makeFixtures } from '../contract/fixtures/make_fixtures.mjs';
import { serve } from '../contract/fixtures/serve.mjs';

const DATA = 4185, SHELL = 4184, root = fileURLToPath(new URL('../../', import.meta.url));
const work = await mkdtemp(path.join(tmpdir(), 'archive-next-real-'));
const cleanup = () => rm(work, { recursive: true, force: true });
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => cleanup().finally(() => process.exit(0)));

const fixtures = await makeFixtures(path.join(work, 'data'), `http://127.0.0.1:${DATA}/`);
await serve(fixtures.dir, DATA);
const dist = path.join(work, 'dist');
const built = spawnSync(process.execPath, [path.join(root, 'next/build.mjs'),
  '--manifest', `http://127.0.0.1:${DATA}/${fixtures.manifest}`, '--manifest-file', path.join(fixtures.dir, fixtures.manifest),
  '--base', '/next/', '--outdir', dist], { stdio: 'inherit' });
if (built.status !== 0) { await cleanup(); process.exit(1); }

// The first rule of the generated _headers file is the base-wide policy.
const csp = (await readFile(path.join(dist, '_headers'), 'utf8')).match(/Content-Security-Policy: (.*)/)[1];
const types = { '.html': 'text/html', '.js': 'application/javascript', '.json': 'application/json', '.css': 'text/css', '.map': 'application/json' };
http.createServer(async (request, response) => {
  const url = new URL(request.url, `http://127.0.0.1:${SHELL}`);
  if (!url.pathname.startsWith('/next/')) { response.writeHead(404); response.end(); return; }
  const filename = path.resolve(dist, decodeURIComponent(url.pathname.slice('/next/'.length)) || 'index.html');
  if (filename !== dist && !filename.startsWith(dist + path.sep)) { response.writeHead(403); response.end(); return; }
  try {
    const data = await readFile(filename);
    response.writeHead(200, { 'Content-Type': types[path.extname(filename)] || 'application/octet-stream', 'Cache-Control': 'no-cache', 'Content-Security-Policy': csp, 'X-Content-Type-Options': 'nosniff' });
    response.end(data);
  } catch { response.writeHead(404); response.end('Not found'); }
}).listen(SHELL, '127.0.0.1', () => console.log(`Real shell: http://127.0.0.1:${SHELL}/next/ (fixtures on ${DATA})`));
