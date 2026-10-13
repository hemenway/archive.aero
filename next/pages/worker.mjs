// Edition pages for the next shell: a Cloudflare Worker in front of the shell's
// static assets that answers /<base>sectionals/chart/<slug>/<date> with the
// shell page rendered for that one chart (next/pages/chart-page.mjs). Every
// other path is left to the asset handler (the beta's wrangler.jsonc routes
// only the edition prefix here with run_worker_first).
//
// At the apex (next/site/wrangler.jsonc, route archive.aero/*) the Worker runs
// first for every request, and a path the assets do not hold - /about, /atc/,
// /classic/, the legacy viewer's files - goes on to the zone origin (GitHub
// Pages) with a plain fetch, so the static pages keep their home. That is
// ORIGIN_FALLTHROUGH="1"; without it a missing asset is the assets' own 404.
//
// The Worker configures itself from <base>chart-pages.json, written by
// next/build.mjs beside the page it describes: where the chart index lives,
// the page's CSP, the site origin for links, and the robots policy of a
// preview host. A base with no such file is not an edition-page host.
import { CARD_ALT, CHART_PREFIX, canonicalChartPath, chartCardAlt, describe, renderChartHtml } from './chart-page.mjs';
import { AIRPORT_PREFIX, airportCardAlt, canonicalAirportPath, describeAirport, renderAirportHtml } from './airport-page.mjs';

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

// The assets, then the zone origin for what they do not hold (a 404 with
// ORIGIN_FALLTHROUGH). An assets 404 is a bare response, so nothing of it is
// kept; an origin that is itself down answers for itself.
async function assetsOrOrigin(request, env, fetcher) {
  const fallthrough = env.ORIGIN_FALLTHROUGH === '1';
  if (fallthrough && request.method !== 'GET' && request.method !== 'HEAD') return fetcher(request);
  const response = await env.ASSETS.fetch(request);
  if (fallthrough && response.status === 404) return fetcher(request);
  return response;
}

// A retired host (RETIRED="1": beta.archive.aero after the apex switch) answers
// everything with 410 Gone and a line pointing at archive.aero. The shell's
// service worker unregisters itself on a 410 (next/sw.js), so returning
// visitors are not pinned to the cached preview. Cached an hour at the edge.
export function retired(request) {
  if (request.method !== 'GET' && request.method !== 'HEAD') return new Response('Gone', { status: 410, headers: { 'cache-control': 'public, max-age=3600' } });
  const body = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex"><title>archive.aero preview retired</title>'
    + '<style>html,body{height:100%;margin:0;background:#0a0e12;color:#e8edf2;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:36rem;margin:0 auto;padding:4rem 1rem}a{color:#8ab4ff}</style>'
    + '<main><h1>This preview has been retired</h1><p>The viewer it previewed is live at <a href="https://archive.aero/">archive.aero</a>. Links into the preview have no equivalent here.</p></main>';
  return new Response(request.method === 'HEAD' ? null : body, { status: 410, headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'public, max-age=3600', 'x-robots-tag': 'noindex' } });
}

// Airport pages (/<base>airports/<ident>), configured by the same file: `airports` is the
// record base (null = no airport pages on this host). Same shape as the edition pages.
async function airportPage(request, env, url, base, config, fetcher) {
  if (request.method !== 'GET' && request.method !== 'HEAD') return new Response('Method not allowed', { status: 405, headers: { allow: 'GET, HEAD' } });
  const route = canonicalAirportPath(url.pathname.slice(base.length));
  if (!route) return notFound(env, url.origin);
  if (route.redirect) return Response.redirect(`${url.origin}${base}${route.redirect}${url.search}`, 301);
  const record = await fetcher(`${config.airports}${route.ident}.json`, { cf: { cacheEverything: true, cacheTtl: 86400 } });
  if (record.status === 404) return notFound(env, url.origin);
  if (!record.ok) return new Response('Airport index unavailable', { status: 502, headers: { 'cache-control': 'no-store' } });
  const model = describeAirport(await record.json());
  if (!model) return notFound(env, url.origin);
  const shell = await env.ASSETS.fetch(new Request(`${url.origin}${base}`));
  if (!shell.ok) return new Response('Chart viewer unavailable', { status: 502, headers: { 'cache-control': 'no-store' } });
  const canonical = `${url.origin}${base}${AIRPORT_PREFIX}${route.ident}`;
  const own = !!(model.card && config.files);
  const image = own ? new URL(model.card.og, config.files).href : (config.image || null);
  const html = renderAirportHtml(await shell.text(), model, { base, site: config.site || url.origin, canonical, image, imageAlt: own ? airportCardAlt(model) : CARD_ALT, robots: config.robots || null });
  return pageResponse(request, shell, html, config);
}

function pageResponse(request, shell, html, config) {
  const headers = new Headers(shell.headers);
  for (const name of ['content-length', 'etag', 'last-modified', 'content-encoding']) headers.delete(name);
  headers.set('content-type', 'text/html; charset=utf-8');
  headers.set('cache-control', 'no-cache');
  headers.set('x-content-type-options', 'nosniff');
  if (config.csp) headers.set('content-security-policy', config.csp);
  if (config.robots) headers.set('x-robots-tag', config.robots);
  return new Response(request.method === 'HEAD' ? null : html, { status: 200, headers });
}

export async function handle(request, env, ctx, { fetcher = fetch } = {}) {
  if (env.RETIRED === '1') return retired(request);
  const url = new URL(request.url);
  let at = url.pathname.indexOf(CHART_PREFIX), airport = false;
  if (at < 0) { at = url.pathname.indexOf(AIRPORT_PREFIX); airport = at >= 0; }
  const base = at < 0 ? null : url.pathname.slice(0, at);
  if (base === null || !/^\/(?:[\w-]+\/)*$/.test(base)) return assetsOrOrigin(request, env, fetcher);
  const config = await loadConfig(env, url.origin, base);
  if (!config) return assetsOrOrigin(request, env, fetcher);
  if (airport) return typeof config.airports === 'string' && config.airports ? airportPage(request, env, url, base, config, fetcher) : assetsOrOrigin(request, env, fetcher);
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
  // The chart's own pre-rendered card when the index names one (09 B3); the site card stands in otherwise.
  const own = !!(model.og && config.files);
  const image = own ? new URL(model.og, config.files).href : (config.image || null);
  const html = renderChartHtml(await shell.text(), model, { base, site: config.site || url.origin, canonical, image, imageAlt: own ? chartCardAlt(model) : CARD_ALT, robots: config.robots || null });
  return pageResponse(request, shell, html, config);
}

export default { fetch: (request, env, ctx) => handle(request, env, ctx) };
