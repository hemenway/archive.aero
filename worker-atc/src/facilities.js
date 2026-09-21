// Restore the WordPress facility browser without a WordPress database or JS.
// Generated metadata only; original photographs and articles stay in R2.
import FACILITIES from "./facilities.json" with { type: "json" };

export const FACILITIES_VERSION = "facilities-v1";
const STATES = Object.keys(FACILITIES);
const encoder = new TextEncoder();
const WIDGET = /<div class="facility_locations">[\s\S]*?<\/div>(?=<\/aside>)/g;
const WIDGET_SCRIPT = /<script\b[^>]*>(?:(?!<\/script>)[\s\S])*?\$\('\.facility_locations > h4'\)(?:(?!<\/script>)[\s\S])*?<\/script>/g;
const CONTENT = /<div class="page-content">[\s\S]*?<\/div><!-- \.page-content -->/;

// ASCII entities let replacements coexist with both UTF-8 and cp1252 pages.
function escape(value) {
  return String(value).replace(/[&<>"'\u007f-\uffff]/g, c => `&#${c.charCodeAt(0)};`);
}

function selection(url) {
  if (!/^\/(?:atc\/)?facility-photos\/?$/.test(url.pathname)) return { state: "", city: "", filtered: false };
  return {
    state: url.searchParams.get("state") || "",
    city: url.searchParams.get("city") || "",
    filtered: url.searchParams.has("state") || url.searchParams.has("city"),
  };
}

export function facilityVariant(url) {
  const { state, city, filtered } = selection(url);
  const cities = Object.hasOwn(FACILITIES, state) ? Object.keys(FACILITIES[state]) : [];
  return filtered ? `${STATES.indexOf(state)}-${city ? cities.indexOf(city) : "all"}` : "index";
}

function locationHref(prefix, state, city) {
  return `${prefix}/facility-photos?${new URLSearchParams({ state, city })}`;
}

const CSS = `<style>
.facility_locations details{border-bottom:1px solid #bbb}
.facility_locations summary{cursor:pointer;padding:5px 10px;font:20px "Droid Serif",serif;color:#2b5f71}
.facility_locations summary:hover,.facility_locations a:hover,.facility_locations a[aria-current=page]{background:#f0f0f0}
.facility_locations .facility-cities{margin:10px 0 10px 10px}
.facility_locations .facility-cities a{display:block;padding:5px 10px;font:18px "Droid Serif",serif;text-decoration:none;color:#2b5f71}
.facility_locations summary:focus-visible,.facility_locations a:focus-visible{outline:2px solid #2b5f71;outline-offset:2px}
.facility-results{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:24px;list-style:none!important;margin:24px 0!important;padding:0!important}
.facility-results li{margin:0;padding:0;min-width:0}
.facility-results a{display:block}
.facility-results img{display:block;width:100%;height:200px;object-fit:contain;background:#f4f4f4;margin-bottom:8px}
</style>`;

function widget(prefix, selected) {
  return CSS + '<div class="facility_locations">' + Object.entries(FACILITIES).map(([state, cities]) =>
    `<details${state === selected.state ? " open" : ""}><summary>${escape(state)}</summary>` +
    '<div class="facility-cities">' + Object.keys(cities).map(city =>
      `<a href="${escape(locationHref(prefix, state, city))}"${state === selected.state && city === selected.city ? ' aria-current="page"' : ""}>${escape(city)}</a>`
    ).join("") + "</div></details>"
  ).join("") + "</div>";
}

function gallery(prefix, { state, city }) {
  const cities = Object.hasOwn(FACILITIES, state) ? FACILITIES[state] : null;
  const entries = cities ? (city ? (Object.hasOwn(cities, city) ? cities[city] : []) : Object.values(cities).flat()) : [];
  // Unknown query values are never echoed into the page.
  const label = entries.length ? [city, state].filter(Boolean).join(", ") : "Location not found";
  const local = href => prefix + href.slice("/atc".length);
  return '<div class="page-content">' +
    `<h2>${escape(label)}</h2><p><a href="${prefix}/facility-photos">All facility locations</a></p>` +
    (entries.length ? `<p>${entries.length} archived ${entries.length === 1 ? "entry" : "entries"}. Select a photo or title to read the full entry.</p>` +
      '<ul class="facility-results">' + entries.map(entry =>
        `<li><a href="${escape(local(entry.href))}">` +
        (entry.image ? `<img src="${escape(local(entry.image))}" alt="" loading="lazy" decoding="async">` : "") +
        `<span>${escape(entry.title)}</span></a></li>`
      ).join("") + "</ul>" : "<p>No archived entries match this location. Choose a state and city from Facility Locations.</p>") +
    "</div><!-- .page-content -->";
}

export function restoreFacilities(bytes, url, prefix = "/atc") {
  // Decode only to locate byte offsets, then splice ASCII replacements. Never
  // re-encode preserved prose (TextDecoder's latin1 is actually windows-1252).
  const text = new TextDecoder("latin1").decode(bytes);
  if (!text.includes('<div class="facility_locations">')) return null;
  const selected = selection(url);
  const edits = [];
  for (const match of text.matchAll(WIDGET))
    edits.push({ start: match.index, end: match.index + match[0].length, html: widget(prefix, selected) });
  if (!edits.length) return null;
  for (const match of text.matchAll(WIDGET_SCRIPT))
    edits.push({ start: match.index, end: match.index + match[0].length, html: "" });
  if (selected.filtered) {
    const match = CONTENT.exec(text);
    if (match) edits.push({ start: match.index, end: match.index + match[0].length, html: gallery(prefix, selected) });
  }
  edits.sort((a, b) => a.start - b.start);
  const parts = [];
  let offset = 0;
  for (const edit of edits) {
    parts.push(bytes.subarray(offset, edit.start), encoder.encode(edit.html));
    offset = edit.end;
  }
  parts.push(bytes.subarray(offset));
  const output = new Uint8Array(parts.reduce((n, part) => n + part.length, 0));
  offset = 0;
  for (const part of parts) { output.set(part, offset); offset += part.length; }
  return output;
}
