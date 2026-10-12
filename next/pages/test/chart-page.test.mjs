import assert from 'node:assert/strict';
import { test } from 'node:test';
import { index, shell } from './fixtures.mjs';
import { parseChartPath, canonicalChartPath, slugFromArtifact, describe, panelHtml, headTags, jsonLd, renderChartHtml, chartPath, editionLabel, longDate } from '../chart-page.mjs';

test('addresses: canonical parse, redirects for other spellings, slug from an artifact', () => {
  assert.deepEqual(parseChartPath('/next/sectionals/chart/boston/1953-12-01', '/next/'), { slug: 'boston', date: '1953-12-01' });
  assert.equal(parseChartPath('/next/sectionals/chart/boston/1953-12-01/', '/next/'), null);
  assert.equal(parseChartPath('/sectionals/chart/boston/1953-12-01', '/next/'), null);
  assert.equal(parseChartPath('/sectionals/chart/Boston/1953-12-01', '/'), null);
  assert.equal(parseChartPath('/sectionals/chart/boston/1953-13-01', '/'), null);
  assert.deepEqual(canonicalChartPath('sectionals/chart/boston/1953-12-01'), { slug: 'boston', date: '1953-12-01' });
  for (const rest of ['sectionals/chart/Boston/1953-12-01', 'sectionals/chart/boston/1953-12-01/', 'sectionals/chart/boston/1953-12-01.pmtiles', 'sectionals/chart/boston/1953-12-01-north'])
    assert.deepEqual(canonicalChartPath(rest), { redirect: 'sectionals/chart/boston/1953-12-01', slug: 'boston', date: '1953-12-01' }, rest);
  assert.equal(canonicalChartPath('sectionals/chart/boston'), null);
  assert.equal(canonicalChartPath('sectionals/chart/bos ton/1953-12-01'), null);
  assert.equal(canonicalChartPath('sectionals/chart/boston/1953-02-30'), null);
  assert.deepEqual(slugFromArtifact('chart/atlanta/2004-03-18-north'), { slug: 'atlanta', date: '2004-03-18' });
  assert.deepEqual(slugFromArtifact('sectionals/chart/boston/1953-12-01.0123456789ab'), { slug: 'boston', date: '1953-12-01' });
  assert.equal(slugFromArtifact('sectionals/1953-12-01_to_1954-06-01.0123456789ab'), null);
  assert.equal(chartPath('/next/', 'boston', '1953-12-01'), '/next/sectionals/chart/boston/1953-12-01');
});

test('describe: neighbours skip untiled editions, half pairs, edition labels, extent fallback', () => {
  const m = describe(index, '1953-12-01');
  assert.equal(m.name, 'Boston'); assert.equal(m.edition, 'edition 32'); assert.equal(m.halves, true); assert.equal(m.tiled, true);
  assert.equal(m.index, 1); assert.equal(m.total, 4); assert.equal(m.prev.d, '1953-06-01'); assert.equal(m.next.d, '1954-12-01', 'the untiled 1954-06 edition is skipped');
  assert.deepEqual(m.bbox, [-73.4, 40.1, -67, 44]);
  assert.equal(m.title, 'Boston sectional chart, December 1953 (edition 32)');
  assert.match(m.description, /^The Boston sectional aeronautical chart, edition 32, in effect from December 1, 1953 to June 1, 1954\. One of 4 Boston editions/);
  assert.equal(m.dates, 'In effect December 1, 1953 – June 1, 1954');
  const last = describe(index, '1954-12-01');
  assert.equal(last.edition, null, 'a LOC catalog number is not an edition'); assert.equal(last.next, null); assert.equal(last.prev.d, '1953-12-01');
  assert.deepEqual(last.bbox, index.bbox, 'a world-spanning artifact box falls back to the location footprint');
  assert.equal(last.title, 'Boston sectional chart, December 1954');
  assert.equal(describe(index, '1954-06-01').tiled, false);
  assert.equal(describe(index, '1999-01-01'), null); assert.equal(describe(null, '1953-12-01'), null);
  assert.equal(describe({ ...index, am: true, charts: [{ d: '1950-01-01', e: '1951-01-01', pm: 'x' }] }, '1950-01-01').bbox, null);
  assert.equal(describe({ ...index, type: 'tac' }, '1953-12-01').title, 'Boston terminal area chart, December 1953 (edition 32)');
  assert.equal(editionLabel('32', true), 'ed. 32'); assert.equal(editionLabel('Unknown'), null); assert.equal(editionLabel('Preliminary'), 'Preliminary');
  assert.equal(longDate('2004-03-18'), 'March 18, 2004');
});

test('panel: heading, dates, links to every other tiled edition, escaped text', () => {
  const html = panelHtml(describe({ ...index, names: ['Bos<ton & Co'] }, '1953-12-01'), { base: '/next/', site: 'https://archive.aero' });
  assert.match(html, /<h1 class="cv-title">Bos&lt;ton &amp; Co<\/h1>/);
  assert.match(html, /<span class="chart-badge sectional">edition 32<\/span> Sectional chart/);
  assert.match(html, /Two sheets, shown together/);
  assert.match(html, /<a class="cv-prev" rel="prev" href="\/next\/sectionals\/chart\/boston\/1953-06-01" data-cv-date="1953-06-01">&larr; Jun 1953 · ed\. 31<\/a>/);
  assert.match(html, /<a class="cv-next" rel="next" href="\/next\/sectionals\/chart\/boston\/1954-12-01" data-cv-date="1954-12-01">Dec 1954 &rarr;<\/a>/);
  assert.match(html, /<span class="cv-pos">2 of 4<\/span>/);
  assert.match(html, /<li aria-current="page"><span>Dec 1953 · ed\. 32<\/span><\/li>/);
  assert.match(html, /<li><span class="pin-unavailable">Jun 1954 · ed\. 33 · not yet tiled<\/span><\/li>/);
  assert.equal((html.match(/data-cv-date=/g) || []).length, 4, 'prev, next and the two other tiled editions in the list');
  assert.match(html, /href="https:\/\/archive.aero\/sources"/);
  const first = panelHtml(describe(index, '1953-06-01'), { base: '/' });
  assert.match(first, /<span class="cv-prev cv-end">Earliest held<\/span>/);
  const only = panelHtml(describe({ ...index, charts: [index.charts[0]] }, '1953-06-01'), { base: '/' });
  assert.doesNotMatch(only, /<details/, 'no edition list for a single edition');
});

test('head tags and JSON-LD carry the edition; the robots tag only on request', () => {
  const m = describe(index, '1953-12-01'), opts = { canonical: 'https://archive.aero/sectionals/chart/boston/1953-12-01', site: 'https://archive.aero', image: 'https://archive.aero/social-card.jpg' };
  const head = headTags(m, opts);
  assert.match(head, /<title>Boston sectional chart, December 1953 \(edition 32\) · archive.aero<\/title>/);
  assert.match(head, /<link rel="canonical" href="https:\/\/archive.aero\/sectionals\/chart\/boston\/1953-12-01">/);
  assert.match(head, /<meta property="og:url" content="https:\/\/archive.aero\/sectionals\/chart\/boston\/1953-12-01">/);
  assert.match(head, /<meta property="og:image" content="https:\/\/archive.aero\/social-card.jpg">/);
  assert.doesNotMatch(head, /name="robots"/);
  assert.match(headTags(m, { ...opts, robots: 'noindex, nofollow' }), /<meta name="robots" content="noindex, nofollow">/);
  const ld = jsonLd(m, opts);
  assert.equal(ld['@type'], 'Map'); assert.equal(ld.datePublished, '1953-12-01'); assert.equal(ld.temporalCoverage, '1953-12-01/1954-06-01');
  assert.equal(ld.spatialCoverage.geo.box, '40.1 -73.4 44 -67'); assert.equal(ld.url, opts.canonical);
  const block = head.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)[1];
  assert.deepEqual(JSON.parse(block), ld);
  assert.doesNotMatch(headTags(describe({ ...index, names: ['</script><script>x'] }, '1953-12-01'), opts).match(/ld\+json">([\s\S]*?)<\/script>/)[1], /<\/script>/);
});

test('renderChartHtml replaces the shell head tags, keeps the rest, fills the panel', () => {
  const html = renderChartHtml(shell, describe(index, '1953-12-01'), { base: '/next/', site: 'https://archive.aero', canonical: 'https://beta.archive.aero/next/sectionals/chart/boston/1953-12-01', image: 'https://archive.aero/social-card.jpg', robots: 'noindex, nofollow' });
  assert.equal((html.match(/<title>/g) || []).length, 1); assert.match(html, /<title>Boston sectional chart, December 1953 \(edition 32\) · archive.aero<\/title>/);
  assert.equal((html.match(/rel="?canonical/g) || []).length, 1); assert.match(html, /href="https:\/\/beta.archive.aero\/next\/sectionals\/chart\/boston\/1953-12-01"/);
  assert.equal((html.match(/og:title/g) || []).length, 1); assert.doesNotMatch(html, /archive.aero — Viewer/); assert.doesNotMatch(html, /Explore historical charts/);
  assert.match(html, /<meta charset=utf-8>/); assert.match(html, /theme-color/); assert.match(html, /M1 2>3/, 'the icon data URI survives'); assert.match(html, /<script>boot\(\)<\/script>/);
  assert.match(html, /<aside class=chart-view id=chartView role=region aria-label="Chart view" tabindex=-1><div class="cv-head">/);
  assert.match(html, /<h1 class="cv-title">Boston<\/h1>/); assert.match(html, /<\/aside><div id=soloBar>/);
  assert.match(html, /<meta name="robots" content="noindex, nofollow">/);
  assert.throws(() => renderChartHtml('<html><body></body></html>', describe(index, '1953-12-01'), { base: '/' }), /no <\/head>/);
  assert.throws(() => renderChartHtml('<html><head></head><body></body></html>', describe(index, '1953-12-01'), { base: '/' }), /no #chartView/);
});
