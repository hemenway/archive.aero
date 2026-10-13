// Airport pages: /airports/<ident> (URI-POLICY.md address plan, worklist 09
// B1) — "Boston Logan through time": the viewer centred on the airport with a
// panel listing every chart edition in the archive whose sheet covers it,
// by decade, each a link to its edition page. Shared by the shell
// (next/app/airportview.js) and the page Worker (next/pages/worker.mjs) the
// way chart-page.mjs is: strings in, strings out, no DOM.
//
// Airport record (scripts/next_build_airports.py, one JSON per ident under
// next/airports.<hash>/):
//   { ident, icao, iata, name, city, region, lat, lon, elev_ft, first, last, editions, tiled,
//     decades: [{decade, editions, tiled, first, last}],
//     charts: [{slug, name, type, d, e, ed, tiled, og?}] }   // sorted by d
import { chartPath, editionLabel, escapeHtml, longDate, monthYear, shortMonthYear, renderPage } from './chart-page.mjs';

export const AIRPORT_PREFIX = 'airports/';
const IDENT = /^[a-z0-9]{3,4}$/;
export const AIRPORT_ZOOM = 10;
export const airportPath = (base, ident) => `${base}${AIRPORT_PREFIX}${ident}`;

// The canonical address only: {ident}, or null.
export function parseAirportPath(pathname, base = '/') {
  if (!pathname.startsWith(base + AIRPORT_PREFIX)) return null;
  const ident = pathname.slice(base.length + AIRPORT_PREFIX.length);
  return IDENT.test(ident) ? { ident } : null;
}

// Every spelling the Worker answers: uppercase (KBOS, the way idents are
// written) and a trailing slash are permanent 301s to the lowercase canonical
// (URI-POLICY rules 2, 5 and 6); anything else null.
export function canonicalAirportPath(rest) {
  if (!rest.startsWith(AIRPORT_PREFIX)) return null;
  const ident = rest.slice(AIRPORT_PREFIX.length).replace(/\/+$/, '').toLowerCase();
  if (!IDENT.test(ident)) return null;
  const canonical = AIRPORT_PREFIX + ident;
  return canonical === rest ? { ident } : { redirect: canonical, ident };
}

const TYPE_WORD = { sectional: 'sectional', tac: 'terminal area', wac: 'world aeronautical' };

// Everything the page says about one airport, or null for a record the page cannot stand on.
export function describeAirport(data) {
  if (!data || !IDENT.test(data.ident || '') || !data.name || !Array.isArray(data.charts) || !data.charts.length) return null;
  if (!Number.isFinite(data.lat) || !Number.isFinite(data.lon)) return null;
  const charts = data.charts, tiled = charts.filter(c => c.tiled && c.slug && c.d);
  const first = data.first || charts[0].d, last = data.last || charts[charts.length - 1].d;
  const state = typeof data.region === 'string' && data.region.includes('-') ? data.region.split('-')[1] : (data.region || '');
  const place = [data.city, state].filter(Boolean).join(', ');
  const latest = tiled.length ? tiled[tiled.length - 1] : null;
  const card = [...tiled].reverse().find(c => typeof c.og === 'string' && c.og) || null;
  const codes = [data.icao || data.ident.toUpperCase(), data.iata].filter(Boolean).join(' / ');
  const years = `${first.slice(0, 4)}–${last.slice(0, 4)}`;
  const decades = Array.isArray(data.decades) && data.decades.length ? data.decades : groupDecades(charts);
  return {
    ident: data.ident, icao: data.icao || data.ident.toUpperCase(), iata: data.iata || null, name: data.name, city: data.city || '', state, place, codes,
    lat: data.lat, lon: data.lon, first, last, years, editions: data.editions ?? charts.length, tiledCount: data.tiled ?? tiled.length,
    decades, charts, latest, card,
    title: `${data.name} on aeronautical charts, ${years}`,
    description: `${data.name} (${codes})${place ? `, ${place}` : ''}, as charted on ${data.editions ?? charts.length} FAA chart editions from ${monthYear(first)} to ${monthYear(last)} in the archive.aero collection of historical sectional charts, each viewable on an interactive map.`
  };
}

export function groupDecades(charts) {
  const by = new Map();
  for (const c of charts) {
    const decade = Math.floor(+c.d.slice(0, 4) / 10) * 10, g = by.get(decade) || { decade, editions: 0, tiled: 0, first: c.d, last: c.d };
    g.editions++; if (c.tiled) g.tiled++; if (c.d < g.first) g.first = c.d; if (c.d > g.last) g.last = c.d; by.set(decade, g);
  }
  return [...by.values()].sort((a, b) => a.decade - b.decade);
}

// The panel. Edition links are real addresses (crawlable from the airport
// page); the shell intercepts their clicks and opens the chart view.
export function panelHtml(model, { base = '/', site = '' } = {}) {
  const label = c => `${shortMonthYear(c.d)} · ${c.name}${editionLabel(c.ed, true) ? ` · ${editionLabel(c.ed, true)}` : ''}${c.type && c.type !== 'sectional' ? ` (${TYPE_WORD[c.type] || c.type})` : ''}`;
  const latestDecade = model.decades.length ? model.decades[model.decades.length - 1].decade : null;
  const decades = model.decades.map(g => {
    const items = model.charts.filter(c => Math.floor(+c.d.slice(0, 4) / 10) * 10 === g.decade).map(c => c.tiled && c.slug
      ? `<li><a href="${escapeHtml(chartPath(base, c.slug, c.d))}" data-ap-chart="${escapeHtml(c.slug)}/${c.d}">${escapeHtml(label(c))}</a></li>`
      : `<li><span class="pin-unavailable">${escapeHtml(label(c))} · not yet tiled</span></li>`).join('');
    return `<li><details class="ap-decade"${g.decade === latestDecade ? ' open' : ''}><summary>${g.decade}s <span class="cv-pos">${g.editions} edition${g.editions === 1 ? '' : 's'}</span></summary><ul class="cv-list">${items}</ul></details></li>`;
  }).join('');
  return `<div class="cv-head"><span class="cv-kicker">Airport</span><button class="pin-close" type="button" data-ap="exit" aria-label="Close airport panel">&times;</button></div>`
    + `<h1 class="cv-title">${escapeHtml(model.name)}</h1>`
    + `<p class="cv-meta"><span class="chart-badge sectional">${escapeHtml(model.icao)}</span>${model.iata ? ` <span>${escapeHtml(model.iata)}</span>` : ''}${model.place ? ` <span>${escapeHtml(model.place)}</span>` : ''}</p>`
    + `<p class="cv-dates">${model.editions} chart edition${model.editions === 1 ? '' : 's'}, ${escapeHtml(monthYear(model.first))} – ${escapeHtml(monthYear(model.last))}</p>`
    + `<div class="cv-actions"><button type="button" class="solo-exit" data-ap="exit">Show all charts</button><button type="button" class="pin-view-alone" data-ap="share">Copy link</button></div>`
    + `<nav class="ap-decades" aria-label="Chart editions covering ${escapeHtml(model.name)}, by decade"><ol class="ap-list">${decades}</ol></nav>`
    + `<a href="${escapeHtml(site)}/sources" class="pin-sources-link">Sources &amp; Attribution</a>`;
}

export function jsonLd(model, { canonical, site, image }) {
  const data = {
    '@context': 'https://schema.org',
    '@type': 'WebPage',
    name: model.title,
    description: model.description,
    url: canonical,
    inLanguage: 'en',
    about: { '@type': 'Airport', name: model.name, icaoCode: model.icao, ...(model.iata ? { iataCode: model.iata } : {}),
      geo: { '@type': 'GeoCoordinates', latitude: model.lat, longitude: model.lon },
      ...(model.place ? { address: { '@type': 'PostalAddress', addressLocality: model.city, addressRegion: model.state, addressCountry: 'US' } } : {}) },
    temporalCoverage: `${model.first}/${model.last}`,
    isPartOf: { '@type': 'Collection', name: 'archive.aero historical aeronautical charts', url: site + '/' },
    publisher: { '@type': 'Organization', name: 'archive.aero', url: site + '/' }
  };
  if (image) data.image = image;
  return data;
}

export const airportCardAlt = model => model.card
  ? `${model.name} on the ${model.card.name} ${TYPE_WORD[model.card.type] || 'sectional'} chart of ${monthYear(model.card.d)}, in the archive.aero chart viewer`
  : `${model.name} in the archive.aero chart viewer`;

export function headTags(model, { canonical, site, image, imageAlt, robots = null }) {
  const title = `${model.title} · archive.aero`;
  const m = (attr, name, content) => `<meta ${attr}="${name}" content="${escapeHtml(content)}">`;
  const tags = [
    `<title>${escapeHtml(title)}</title>`,
    m('name', 'description', model.description),
    `<link rel="canonical" href="${escapeHtml(canonical)}">`,
    m('property', 'og:type', 'website'), m('property', 'og:site_name', 'archive.aero'),
    m('property', 'og:title', model.title), m('property', 'og:description', model.description), m('property', 'og:url', canonical),
    m('name', 'twitter:card', 'summary_large_image'), m('name', 'twitter:title', model.title), m('name', 'twitter:description', model.description)
  ];
  if (image) tags.push(m('property', 'og:image', image), m('property', 'og:image:width', '1200'), m('property', 'og:image:height', '630'), m('property', 'og:image:alt', imageAlt || airportCardAlt(model)), m('name', 'twitter:image', image));
  if (robots) tags.push(m('name', 'robots', robots));
  tags.push(`<script type="application/ld+json">${JSON.stringify(jsonLd(model, { canonical, site, image })).replaceAll('<', '\\u003c')}</script>`);
  return tags.join('');
}

// The aside carries the view the shell boots on (centre, zoom, the newest
// edition's date), so the first paint is the airport, not the lower 48.
export function asideAttrs(model) {
  return { 'data-airport': model.ident, 'data-lat': model.lat.toFixed(4), 'data-lng': model.lon.toFixed(4), 'data-zoom': String(AIRPORT_ZOOM), ...(model.latest ? { 'data-date': model.latest.d } : {}) };
}

export function renderAirportHtml(shell, model, { base, site, canonical, image, imageAlt, robots }) {
  return renderPage(shell, { head: headTags(model, { canonical, site, image, imageAlt, robots }), panel: panelHtml(model, { base, site }), attrs: asideAttrs(model) });
}
