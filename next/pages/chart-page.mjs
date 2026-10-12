// Edition pages: /sectionals/chart/<slug>/<date> (URI-POLICY.md address plan,
// worklist 09 B2). The page is the viewer opened on one chart alone, with that
// chart's text in the document: title, description, canonical, Open Graph and
// Twitter tags, schema.org JSON-LD, and the chart-view panel rendered as HTML
// with real links to the location's other editions.
//
// This module is shared by the shell (next/app/chartview.js) and the edition
// page Worker (next/pages/worker.mjs), so what a crawler receives is exactly
// what the viewer renders once it boots. It has no DOM and no Workers API:
// strings in, strings out, tested with node:test.
//
// Chart index (one file per location slug, built by scripts/next_build_overlays.py):
//   { slug, names: [..], ref, type: 'sectional'|'tac'|'wac', bbox: [w,s,e,n]|null, am: bool,
//     charts: [{ d, e, ed, f, pm, pmz, pmb }, ...] }   // sorted by d; pm absent = not yet tiled

export const CHART_PREFIX = 'sectionals/chart/';
const SLUG = /^[a-z0-9_]+$/;
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const TYPES = {
  sectional: { label: 'Sectional chart', long: 'sectional aeronautical chart' },
  tac: { label: 'Terminal area chart', long: 'terminal area chart' },
  wac: { label: 'World aeronautical chart', long: 'world aeronautical chart' }
};
// The social card stands in for a per-page image until B3 (pre-rendered og:image per chart) lands.
export const CARD_ALT = 'A 1950 FAA sectional chart of New York City in the archive.aero timeline viewer';

export const validDate = s => { if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return false; const t = Date.parse(s + 'T00:00:00Z'); return Number.isFinite(t) && new Date(t).toISOString().slice(0, 10) === s; };
export const chartPath = (base, slug, date) => `${base}${CHART_PREFIX}${slug}/${date}`;

// The canonical address only: {slug, date}, or null for anything else.
export function parseChartPath(pathname, base = '/') {
  if (!pathname.startsWith(base + CHART_PREFIX)) return null;
  const parts = pathname.slice(base.length + CHART_PREFIX.length).split('/');
  if (parts.length !== 2 || !SLUG.test(parts[0]) || !validDate(parts[1])) return null;
  return { slug: parts[0], date: parts[1] };
}

// Every spelling the Worker answers. `rest` is the path after the shell's
// base ("sectionals/chart/Boston/1953-12-01/"). Uppercase, a trailing slash,
// the artifact's .pmtiles suffix and one half of a half-sheet pair
// (…-north) all have a canonical form and get {redirect}; the canonical form
// gets {slug, date}; anything else null (URI-POLICY rules 2, 5 and 7).
export function canonicalChartPath(rest) {
  if (!rest.startsWith(CHART_PREFIX)) return null;
  const parts = rest.slice(CHART_PREFIX.length).replace(/\/+$/, '').replace(/\.pmtiles$/i, '').split('/');
  if (parts.length !== 2) return null;
  const slug = parts[0].toLowerCase(), date = parts[1].toLowerCase().replace(/^(\d{4}-\d{2}-\d{2})-[a-z]+$/, '$1');
  if (!SLUG.test(slug) || !validDate(date)) return null;
  const canonical = `${CHART_PREFIX}${slug}/${date}`;
  return canonical === rest ? { slug, date } : { redirect: canonical, slug, date };
}

// The page a per-chart artifact belongs to, from its key or versioned path
// ("chart/atlanta/2004-03-18-north", "sectionals/chart/boston/1953-12-01.0123456789ab").
export function slugFromArtifact(path) {
  const m = /(?:^|\/)chart\/([a-z0-9_]+)\/(\d{4}-\d{2}-\d{2})(?:-[a-z]+)?(?:\.[a-f0-9]{12})?(?:\.pmtiles)?$/.exec(String(path || ''));
  return m && validDate(m[2]) ? { slug: m[1], date: m[2] } : null;
}

export const longDate = iso => { const [y, m, d] = iso.split('-').map(Number); return `${MONTHS[m - 1]} ${d}, ${y}`; };
export const monthYear = iso => { const [y, m] = iso.split('-').map(Number); return `${MONTHS[m - 1]} ${y}`; };
export const shortMonthYear = iso => { const [y, m] = iso.split('-').map(Number); return `${MONTHS[m - 1].slice(0, 3)} ${y}`; };

// Old-era rows carry LOC catalog numbers in the edition column: only a
// plausible edition number is an edition (PinInspector.edLabel's rule).
export function editionLabel(ed, short = false) {
  if (ed && /^\d+$/.test(ed) && +ed < 150) return `${short ? 'ed.' : 'edition'} ${+ed}`;
  if (ed && !/^\d+$/.test(ed) && ed !== 'Unknown') return ed;
  return null;
}

const finiteBox = b => Array.isArray(b) && b.length === 4 && b.every(Number.isFinite) && b[2] > b[0] && b[3] > b[1] && b[2] - b[0] <= 180;

// Everything the page says about one edition, or null when the location
// holds no chart starting on that date.
export function describe(index, date) {
  if (!index || !Array.isArray(index.charts) || !index.names?.length) return null;
  const charts = index.charts;
  const i = charts.findIndex(c => c.d === date);
  if (i < 0) return null;
  const chart = charts[i], name = index.names[0], type = TYPES[index.type] || TYPES.sectional;
  const tiled = c => !!c.pm;
  let prev = null, next = null;
  for (let j = i - 1; j >= 0 && !prev; j--) if (tiled(charts[j])) prev = charts[j];
  for (let j = i + 1; j < charts.length && !next; j++) if (tiled(charts[j])) next = charts[j];
  const edition = editionLabel(chart.ed);
  const halves = Array.isArray(chart.pm) && chart.pm.length > 1;
  // The chart's own extent when its artifact says so, else the location's
  // footprint; an antimeridian crossing has no usable box (stay put).
  const bbox = finiteBox(chart.pmb) ? chart.pmb : (!index.am && finiteBox(index.bbox) ? index.bbox : null);
  const when = chart.e ? `in effect from ${longDate(chart.d)} to ${longDate(chart.e)}` : `published ${longDate(chart.d)}`;
  return {
    slug: index.slug, name, names: index.names, ref: index.ref || null, type: index.type || 'sectional', typeLabel: type.label,
    chart, all: charts, index: i, total: charts.length, prev, next, edition, halves, bbox, tiled: tiled(chart),
    title: `${name} ${type.label.toLowerCase()}, ${monthYear(chart.d)}${edition ? ` (${edition})` : ''}`,
    description: `The ${name} ${type.long}${edition ? `, ${edition},` : ''} ${when}. One of ${charts.length} ${name} editions in the archive.aero collection of historical FAA charts, georeferenced and viewable on an interactive map.`,
    dates: chart.e ? `In effect ${longDate(chart.d)} – ${longDate(chart.e)}` : `Published ${longDate(chart.d)}`
  };
}

export const escapeHtml = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');

// The chart-view panel's content. Links are real addresses so the editions of
// a location are crawlable from each other; the shell intercepts their clicks.
export function panelHtml(model, { base = '/', site = '' } = {}) {
  const href = c => escapeHtml(chartPath(base, model.slug, c.d));
  const label = c => `${shortMonthYear(c.d)}${editionLabel(c.ed, true) ? ` · ${editionLabel(c.ed, true)}` : ''}`;
  const nav = [
    model.prev ? `<a class="cv-prev" rel="prev" href="${href(model.prev)}" data-cv-date="${model.prev.d}">&larr; ${escapeHtml(label(model.prev))}</a>` : '<span class="cv-prev cv-end">Earliest held</span>',
    `<span class="cv-pos">${model.index + 1} of ${model.total}</span>`,
    model.next ? `<a class="cv-next" rel="next" href="${href(model.next)}" data-cv-date="${model.next.d}">${escapeHtml(label(model.next))} &rarr;</a>` : '<span class="cv-next cv-end">Latest held</span>'
  ].join('');
  const list = [];
  for (let j = model.total - 1; j >= 0; j--) {
    const c = model.all[j];
    if (j === model.index) list.push(`<li aria-current="page"><span>${escapeHtml(label(c))}</span></li>`);
    else if (c.pm) list.push(`<li><a href="${href(c)}" data-cv-date="${c.d}">${escapeHtml(label(c))}</a></li>`);
    else list.push(`<li><span class="pin-unavailable">${escapeHtml(label(c))} · not yet tiled</span></li>`);
  }
  return `<div class="cv-head"><span class="cv-kicker">Chart view</span><button class="pin-close" type="button" data-cv="exit" aria-label="Close chart view">&times;</button></div>`
    + `<h1 class="cv-title">${escapeHtml(model.name)}</h1>`
    + `<p class="cv-meta">${model.edition ? `<span class="chart-badge sectional">${escapeHtml(model.edition)}</span> ` : ''}${escapeHtml(model.typeLabel)}</p>`
    + `<p class="cv-dates">${escapeHtml(model.dates)}</p>`
    + (model.halves ? '<p class="cv-note">Two sheets, shown together.</p>' : '')
    + `<div class="cv-actions"><button type="button" class="solo-exit" data-cv="exit">Show all charts</button><button type="button" class="pin-view-alone" data-cv="share">Copy link</button></div>`
    + `<nav class="cv-nav" aria-label="Other editions of this chart">${nav}</nav>`
    + (list.length > 1 ? `<details class="cv-all"><summary>All ${model.total} ${escapeHtml(model.name)} editions</summary><ol class="cv-list" reversed>${list.join('')}</ol></details>` : '')
    + `<a href="${escapeHtml(site)}/sources" class="pin-sources-link">Sources &amp; Attribution</a>`;
}

export function jsonLd(model, { canonical, site, image }) {
  const data = {
    '@context': 'https://schema.org',
    '@type': 'Map',
    name: model.title,
    description: model.description,
    url: canonical,
    datePublished: model.chart.d,
    inLanguage: 'en',
    isPartOf: { '@type': 'Collection', name: 'archive.aero historical aeronautical charts', url: site + '/' },
    publisher: { '@type': 'Organization', name: 'archive.aero', url: site + '/' }
  };
  if (image) data.image = image;
  if (model.chart.e) data.temporalCoverage = `${model.chart.d}/${model.chart.e}`;
  if (model.bbox) data.spatialCoverage = { '@type': 'Place', geo: { '@type': 'GeoShape', box: `${model.bbox[1]} ${model.bbox[0]} ${model.bbox[3]} ${model.bbox[2]}` } };
  return data;
}

// Head tags for one edition. The shell's own title/description/canonical and
// social tags are removed by renderChartHtml; these replace them.
export function headTags(model, { canonical, site, image, robots = null }) {
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
  if (image) tags.push(m('property', 'og:image', image), m('property', 'og:image:width', '1200'), m('property', 'og:image:height', '630'), m('property', 'og:image:alt', CARD_ALT), m('name', 'twitter:image', image));
  if (robots) tags.push(m('name', 'robots', robots));
  // A data block, not a script: CSP's script-src does not apply to JSON-LD.
  tags.push(`<script type="application/ld+json">${JSON.stringify(jsonLd(model, { canonical, site, image })).replaceAll('<', '\\u003c')}</script>`);
  return tags.join('');
}

// Same tag grammar as next/build.mjs's minifier: attribute values are matched
// as quoted strings, so a data: URI or an SVG path never confuses a tag end.
const TAG = /<(\/)?([a-zA-Z][\w-]*)((?:\s+[^\s"'<>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?)*)\s*(\/)?>/g;
const ATTR = /([^\s"'<>\/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
const attrsOf = text => Object.fromEntries([...text.matchAll(ATTR)].map(a => [a[1].toLowerCase(), a[2] ?? a[3] ?? a[4] ?? '']));
const REPLACED = /^(?:description|robots|og:[\w:]+|twitter:[\w:]+)$/;

// The built shell page with one edition's head and panel. Throws when the
// shell has no <head> or no #chartView: a page this cannot place must not ship.
export function renderChartHtml(shell, model, { base, site, canonical, image, robots }) {
  const headEnd = shell.search(/<\/head\s*>/i);
  if (headEnd < 0) throw new Error('Edition page: the shell has no </head>');
  let head = shell.slice(0, headEnd), rest = shell.slice(headEnd);
  // Drop the shell's title and the tags being replaced; keep everything else (icon, viewport, preloads, boot script).
  const drops = [];
  for (const m of head.matchAll(TAG)) {
    const tag = m[2].toLowerCase(), attrs = attrsOf(m[3]);
    if (m[1]) continue;
    if (tag === 'title') { const close = head.indexOf('</title>', m.index); if (close < 0) throw new Error('Edition page: unterminated <title>'); drops.push([m.index, close + 8]); }
    else if (tag === 'meta' && REPLACED.test(attrs.name || attrs.property || '')) drops.push([m.index, m.index + m[0].length]);
    else if (tag === 'link' && (attrs.rel || '').toLowerCase() === 'canonical') drops.push([m.index, m.index + m[0].length]);
  }
  for (const [s, e] of drops.sort((a, b) => b[0] - a[0])) head = head.slice(0, s) + head.slice(e);
  head += headTags(model, { canonical, site, image, robots });
  // The panel: the build emits it empty and hidden; here it carries the edition and shows without JavaScript.
  let open = null;
  for (const m of rest.matchAll(TAG)) if (!m[1] && m[2].toLowerCase() === 'aside' && attrsOf(m[3]).id === 'chartView') { open = m; break; }
  if (!open) throw new Error('Edition page: the shell has no #chartView');
  const close = rest.indexOf('</aside>', open.index);
  if (close < 0) throw new Error('Edition page: #chartView is not closed');
  const opening = open[0].replace(/\s+hidden(?=[\s>])/, '');
  rest = rest.slice(0, open.index) + opening + panelHtml(model, { base, site }) + rest.slice(close);
  return head + rest;
}
