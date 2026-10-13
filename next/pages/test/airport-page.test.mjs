import assert from 'node:assert/strict';
import { test } from 'node:test';
import { canonicalAirportPath, describeAirport, groupDecades, headTags, panelHtml, parseAirportPath, renderAirportHtml } from '../airport-page.mjs';
import { airport, shell } from './fixtures.mjs';

test('addresses: canonical lowercase ident, uppercase and trailing slash redirect, junk is null', () => {
  assert.deepEqual(parseAirportPath('/airports/kbos'), { ident: 'kbos' });
  assert.deepEqual(parseAirportPath('/next/airports/kbos', '/next/'), { ident: 'kbos' });
  assert.equal(parseAirportPath('/airports/KBOS'), null); assert.equal(parseAirportPath('/airports/kbos/'), null); assert.equal(parseAirportPath('/airports/'), null);
  assert.deepEqual(canonicalAirportPath('airports/kbos'), { ident: 'kbos' });
  assert.deepEqual(canonicalAirportPath('airports/KBOS/'), { redirect: 'airports/kbos', ident: 'kbos' });
  assert.equal(canonicalAirportPath('airports/kbos/1953'), null); assert.equal(canonicalAirportPath('airports/k-bos'), null); assert.equal(canonicalAirportPath('airports/toolongident'), null);
});

test('describe: codes, place, years, latest tiled edition and the newest card', () => {
  const m = describeAirport(airport);
  assert.equal(m.title, 'Boston Logan International Airport on aeronautical charts, 1933–1970');
  assert.equal(m.codes, 'KBOS / BOS'); assert.equal(m.place, 'Boston, MA'); assert.equal(m.editions, 5); assert.equal(m.tiledCount, 4);
  assert.equal(m.latest.d, '1960-01-01', 'the newest tiled edition, not the untiled 1970 one');
  assert.equal(m.card.d, '1953-12-01', 'the newest edition that has a card');
  assert.match(m.description, /5 FAA chart editions from March 1933 to January 1970/);
  assert.equal(describeAirport({ ...airport, charts: [] }), null); assert.equal(describeAirport({ ...airport, lat: 'x' }), null); assert.equal(describeAirport(null), null);
  assert.deepEqual(groupDecades(airport.charts).map(g => [g.decade, g.editions, g.tiled]), [[1930, 1, 1], [1950, 2, 2], [1960, 1, 1], [1970, 1, 0]]);
  const noDecades = describeAirport({ ...airport, decades: undefined });
  assert.equal(noDecades.decades.length, 4, 'decades derived when the record has none');
});

test('panel: decades with edition links to the pages, untiled listed not linked, latest decade open', () => {
  const html = panelHtml(describeAirport(airport), { base: '/', site: 'https://archive.aero' });
  assert.match(html, /<h1 class="cv-title">Boston Logan International Airport<\/h1>/);
  assert.match(html, /<span class="chart-badge sectional">KBOS<\/span> <span>BOS<\/span> <span>Boston, MA<\/span>/);
  assert.match(html, /<a href="\/sectionals\/chart\/boston\/1953-12-01" data-ap-chart="boston\/1953-12-01">Dec 1953 · Boston · ed\. 32<\/a>/);
  assert.match(html, /Jan 1970 · Boston · ed\. 33 · not yet tiled/);
  assert.match(html, /<details class="ap-decade"><summary>1930s/); assert.match(html, /<details class="ap-decade"><summary>1960s/); assert.match(html, /<details class="ap-decade" open><summary>1970s/);
  assert.match(html, /data-ap="exit"/); assert.match(html, /data-ap="share"/); assert.match(html, /href="https:\/\/archive.aero\/sources"/);
  assert.match(html, /\(terminal area\)/, 'a non-sectional edition names its type');
});

test('head and page: title, canonical, og image and alt, JSON-LD Airport, aside carries the view', () => {
  const m = describeAirport(airport);
  const head = headTags(m, { canonical: 'https://archive.aero/airports/kbos', site: 'https://archive.aero', image: 'https://data.archive.aero/sectionals/chart/boston/1953-12-01.jpg' });
  assert.match(head, /<title>Boston Logan International Airport on aeronautical charts, 1933–1970 · archive.aero<\/title>/);
  assert.match(head, /og:image:alt" content="Boston Logan International Airport on the Boston sectional chart of December 1953, in the archive.aero chart viewer"/);
  const ld = JSON.parse(/<script type="application\/ld\+json">([^]*?)<\/script>/.exec(head)[1]);
  assert.equal(ld.about['@type'], 'Airport'); assert.equal(ld.about.icaoCode, 'KBOS'); assert.equal(ld.about.iataCode, 'BOS'); assert.equal(ld.temporalCoverage, '1933-03-01/1970-01-01');
  const page = renderAirportHtml(shell, m, { base: '/', site: 'https://archive.aero', canonical: 'https://archive.aero/airports/kbos', image: null, robots: null });
  assert.match(page, /<aside class=chart-view id=chartView role=region aria-label="Chart view" tabindex=-1 data-airport="kbos" data-lat="42.3643" data-lng="-71.0052" data-zoom="10" data-date="1960-01-01">/);
  assert.doesNotMatch(page, /hidden/); assert.doesNotMatch(page, /og:image/); assert.match(page, /<title>Boston Logan/); assert.doesNotMatch(page, /Historical Aeronautical Chart Viewer<\/title>/);
});
