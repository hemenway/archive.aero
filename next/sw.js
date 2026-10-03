// Build replaces these constants. Tiles intentionally never enter CacheStorage.
const VERSION = __SW_VERSION__, ASSETS = __SW_ASSETS__, MANIFEST = __SW_MANIFEST__;
// One cache family per registration directory, so a /next/ preview worker and a
// root worker never delete each other's caches on activation.
const DIR = new URL('./', self.location).pathname, PREFIX = `archive-next-${DIR.replace(/[^\w]+/g, '_')}-`, CACHE = PREFIX + VERSION;
const SHELL = new URL('./index.html', self.location).href;
self.addEventListener('install', event => event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(ASSETS))));
self.addEventListener('activate', event => event.waitUntil((async () => {
  for (const name of await caches.keys()) if (name.startsWith(PREFIX) && name !== CACHE) await caches.delete(name);
  await self.clients.claim();
})()));
self.addEventListener('fetch', event => {
  const request = event.request, url = new URL(request.url);
  if (request.method !== 'GET') return;
  const shell = ASSETS.some(asset => new URL(asset, self.location).href === url.href);
  const isManifest = url.href === new URL(MANIFEST, self.location).href;
  // Only the shell's own document is network-first with an offline fallback. Other
  // pages under the scope (/atc/, /about when registered at the root) pass through.
  if (request.mode === 'navigate' && url.origin === self.location.origin && (url.pathname === DIR || url.pathname === DIR + 'index.html')) {
    event.respondWith((async () => {
      const cache = await caches.open(CACHE), controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 2500);
      try {
        const response = await fetch(request, { signal: controller.signal });
        // Redirects and a retired shell (404, 410) pass through: masking them with the
        // cached page would pin returning visitors to it forever. 410 also unregisters.
        if (response.type === 'opaqueredirect' || response.redirected || response.status === 404 || response.status === 410) { if (response.status === 410) await self.registration.unregister(); return response; }
        if (!response.ok) throw new Error('Navigation unavailable');
        await cache.put(SHELL, response.clone()); return response;
      }
      catch { return await cache.match(SHELL) || new Response('Chart viewer unavailable offline. Reconnect and reload.', { status: 503, headers: { 'Content-Type': 'text/plain' } }); }
      finally { clearTimeout(timeout); }
    })());
  } else if (shell || isManifest) {
    event.respondWith((async () => { const cache = await caches.open(CACHE), cached = await cache.match(request); if (cached) return cached; const response = await fetch(request); if (response.ok) await cache.put(request, response.clone()); return response; })());
  }
});
