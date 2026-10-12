// Edition pages for the next shell: a Cloudflare Worker in front of the shell's
// static assets that answers /<base>sectionals/chart/<slug>/<date> with the
// shell page rendered for that one chart (next/pages/chart-page.mjs). Every
// other path is left to the asset handler (the beta's wrangler.jsonc routes
// only the edition prefix here with run_worker_first).
//
// The Worker configures itself from <base>chart-pages.json, written by
// next/build.mjs beside the page it describes: where the chart index lives,
// the page's CSP, the site origin for links, and the robots policy of a
// preview host. A base with no such file is not an edition-page host.
import { CHART_PREFIX, canonicalChartPath, describe, renderChartHtml } from './chart-page.mjs';

const CONFIG_TTL = 60_000;
// Per asset binding (one per isolate), so a test harness or a redeploy never sees another binding's config.
const configsByAssets = new WeakMap();

async function loadConfig(env, origin, base) {
  let configs = configsByAssets.get(env.ASSETS);
  if (!configs) configsByAssets.set(env.ASSETS, configs = new Map());
  const key = origin + base, hit = configs.get(key);
  if (hit && Date.now() - hit.at < CONFIG_TTL) return hit.config;
  let config = null;
  try {
    const response = await env.ASSETS.fetch(new Request(`${origin}${base}chart-pages.json`));
    if (response.ok) config = await response.json();
  } catch { config = null; }
  if (config && (config.version !== 1 || config.base !== base || typeof config.charts !== 'string')) config = null;
  configs.set(key, { config, at: Date.now() });
  return config;
}

async function notFound(env, origin) {
  let body = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="robots" content="noindex"><title>Chart not found</title><h1>Chart not found</h1><p>No chart of that location starts on that date.</p>';
  try {
    const page = await env.ASSETS.fetch(new Request(`${origin}/404.html`));
    if (page.ok) body = await page.text();
  } catch { /* the inline page stands in */ }
  return new Response(body, { status: 404, headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'public, max-age=60', 'x-robots-tag': 'noindex' } });
}

export async function handle(request, env, ctx, { fetcher = fetch } = {}) {
  const url = new URL(request.url);
  const at = url.pathname.indexOf(CHART_PREFIX);
  const base = at < 0 ? null : url.pathname.slice(0, at);
  if (base === null || !/^\/(?:[\w-]+\/)*$/.test(base)) return env.ASSETS.fetch(request);
  const config = await loadConfig(env, url.origin, base);
  if (!config) return env.ASSETS.fetch(request);
  if (request.method !== 'GET' && request.method !== 'HEAD') return new Response('Method not allowed', { status: 405, headers: { allow: 'GET, HEAD' } });
  const route = canonicalChartPath(url.pathname.slice(at));
  if (!route) return notFound(env, url.origin);
  // Aliases are permanent 301s to the one canonical form (URI-POLICY rules 1, 5, 6).
  if (route.redirect) return Response.redirect(`${url.origin}${base}${route.redirect}${url.search}`, 301);
  const indexResponse = await fetcher(`${config.charts}${route.slug}.json`, { cf: { cacheEverything: true, cacheTtl: 86400 } });
  if (indexResponse.status === 404) return notFound(env, url.origin);
  if (!indexResponse.ok) return new Response('Chart index unavailable', { status: 502, headers: { 'cache-control': 'no-store' } });
  const model = describe(await indexResponse.json(), route.date);
  // Thin-content guard (09 B5): a date with no tiled chart is not a page.
  if (!model?.tiled) return notFound(env, url.origin);
  const shell = await env.ASSETS.fetch(new Request(`${url.origin}${base}`));
  if (!shell.ok) return new Response('Chart viewer unavailable', { status: 502, headers: { 'cache-control': 'no-store' } });
  const canonical = `${url.origin}${base}${CHART_PREFIX}${route.slug}/${route.date}`;
  const html = renderChartHtml(await shell.text(), model, { base, site: config.site || url.origin, canonical, image: config.image || null, robots: config.robots || null });
  const headers = new Headers(shell.headers);
  for (const name of ['content-length', 'etag', 'last-modified', 'content-encoding']) headers.delete(name);
  headers.set('content-type', 'text/html; charset=utf-8');
  headers.set('cache-control', 'no-cache');
  headers.set('x-content-type-options', 'nosniff');
  if (config.csp) headers.set('content-security-policy', config.csp);
  if (config.robots) headers.set('x-robots-tag', config.robots);
  return new Response(request.method === 'HEAD' ? null : html, { status: 200, headers });
}

export default { fetch: (request, env, ctx) => handle(request, env, ctx) };
