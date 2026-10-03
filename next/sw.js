// Build replaces these constants. Tiles intentionally never enter CacheStorage.
const VERSION = __SW_VERSION__, ASSETS = __SW_ASSETS__, MANIFEST = __SW_MANIFEST__;
const PREFIX = 'archive-next-', CACHE = PREFIX + VERSION;
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
  if (request.mode === 'navigate' && url.origin === self.location.origin && url.pathname.startsWith(self.registration.scope.replace(url.origin, ''))) {
    event.respondWith((async () => {
      const cache = await caches.open(CACHE), controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 2500);
      try { const response = await fetch(request, { signal: controller.signal }); if (!response.ok) throw new Error('Navigation unavailable'); await cache.put(new URL('./index.html', self.location).href, response.clone()); return response; }
      catch { return await cache.match(new URL('./index.html', self.location).href) || new Response('Chart viewer unavailable offline. Reconnect and reload.', { status: 503, headers: { 'Content-Type': 'text/plain' } }); }
      finally { clearTimeout(timeout); }
    })());
  } else if (shell || isManifest) {
    event.respondWith((async () => { const cache = await caches.open(CACHE), cached = await cache.match(request); if (cached) return cached; const response = await fetch(request); if (response.ok) await cache.put(request, response.clone()); return response; })());
  }
});
