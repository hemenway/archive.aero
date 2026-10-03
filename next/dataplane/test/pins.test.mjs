import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {buildPinIndex, queryPins} from '../pins.js';

const viewer = readFileSync(new URL('../../../src/viewer.js', import.meta.url), 'utf8');
const utils = viewer.slice(viewer.indexOf('const Utils = {'), viewer.indexOf('/** METADATA BUNDLE **/'));
const legacySource = viewer.slice(viewer.indexOf('const ChartIndex = {'), viewer.indexOf('/** MAP CONTROLLER **/'));
const json = value => JSON.parse(JSON.stringify(value));
const epochDay = iso => Date.parse(iso) / 86400000;

function legacy(data, published) {
  const context = vm.createContext({CONFIG: {ranges: published.map(key => ({key}))}});
  vm.runInContext(`${utils}\n${legacySource}\nglobalThis.index = ChartIndex;`, context);
  context.index._build(data);
  return context.index;
}

function assertParity(data, published, points, dates) {
  const current = buildPinIndex(data, published);
  const previous = legacy(data, published);
  assert.deepEqual(json(current.locations), json(previous.locations));
  assert.deepEqual([...current.eraMemberCount], Array.from(previous.eraMemberCount, entry => [...entry]));
  for (const [lat, lon] of points) for (const date of dates) {
    const expected = previous.query(lat, lon, date).map(({loc, ...hit}) => ({location: loc, ...hit}));
    assert.deepEqual(json(queryPins(current, lat, lon, date)), json(expected));
    if (Number.isInteger(epochDay(date))) {
      assert.deepEqual(json(queryPins(current, lat, lon, epochDay(date))), json(expected));
    }
  }
  return current;
}

// Use precisely the inventory authored for the existing browser suite. It is
// private to that module, so extract only its declaration (no browser/network).
const fixtures = readFileSync(new URL('../../../tests/browser/fixtures.mjs', import.meta.url), 'utf8');
const fixtureSource = fixtures.slice(fixtures.indexOf('export const dates'), fixtures.indexOf('function varint'));
const fixtureContext = vm.createContext({});
vm.runInContext(fixtureSource.replace('export const dates', 'const dates') + '\nglobalThis.data = inventory;', fixtureContext);

test('pin inventory and queries match ChartIndex on browser fixtures', () => {
  const data = json(fixtureContext.data);
  const published = ['1950-01-01_to_1960-01-01', '1970-01-01_to_1980-01-01'];
  assertParity(data, published, [[32, -96], [30, -96], [40, -120], [32, 264]],
    ['1949-12-31', '1950-01-01', '1959-12-31', '1960-01-01', '1970-01-01', '1980-01-01']);
});

const ring = (w, s, e, n) => [[w, s], [e, s], [e, n], [w, n], [w, s]];
const chart = {d: '2000-01-01', e: '2001-01-01', ed: '1'};

test('antimeridian unwrap keeps raw clipping ring and accepts wrapped clicks', () => {
  const raw = ring(170, 40, -170, 60);
  const data = {locations: {Aleutians: {ref: 'aleutians', charts: [chart]}}, rings: {aleutians: [raw]}};
  const index = assertParity(data, [], [[50, 175], [50, -175], [50, 535], [50, -535], [50, 0]], ['2000-05-01']);
  assert.equal(index.locations[0].crossesAM, true);
  assert.equal(index.locations[0].ringClip, raw);
  assert.deepEqual(index.locations[0].ringHit, ring(-190, 40, -170, 60));
  assert.equal(queryPins(index, 50, 175, '2000-05-01')[0].contains, true);
});

test('dedupe uses d|e and preserves the first scan and per-era member count', () => {
  const data = {locations: {
    A: {ref: 'a', charts: [chart, {...chart, ed: 'alternate', pm: 'alternate'}, {...chart, e: '2002-01-01'}]},
    B: {ref: 'a', charts: [chart]},
  }, rings: {a: [ring(-100, 30, -90, 40)]}};
  const index = assertParity(data, ['2000-01-01_to_2001-01-01'], [[35, -95]], ['2000-05-01']);
  assert.equal(index.locations[0].charts.length, 2);
  assert.equal(index.locations[0].charts[0].ed, '1');
  assert.equal(index.locations[0].charts[0].published, true);
  assert.equal(index.locations[0].charts[1].published, false);
  assert.equal(index.eraMemberCount.get('2000-01-01_to_2001-01-01'), 2);
});

test('end dates are exclusive and missing or invalid ends infer exactly 182 days', () => {
  const data = {locations: {A: {ref: 'a', charts: [
    {...chart, e: null}, {...chart, d: '2002-01-01', e: 'invalid'},
    {...chart, d: '2004-01-01', e: '2004-01-01'}, {...chart, d: 'invalid'},
  ]}}, rings: {a: [ring(-100, 30, -90, 40)]}};
  const index = assertParity(data, ['2000-01-01'], [[35, -95]],
    ['2000-01-01', '2000-06-30', '2000-07-01', '2002-07-01', '2002-07-02', '2004-01-01', 'invalid']);
  const first = index.locations[0].charts[0];
  assert.equal(first.t1 - first.t0, 182 * 86400000);
  assert.equal(first.published, true);
  assert.equal(queryPins(index, 35, -95, epochDay('2000-01-01') + 182).length, 0);
});

test('containment precedes edge distance, ties preserve inventory order, default limit is six', () => {
  const data = {locations: {}, rings: {}};
  for (let i = 0; i < 8; i++) {
    const name = `Chart ${i}`;
    data.locations[name] = {ref: name, charts: [chart]};
    data.rings[name] = [ring(i === 7 ? -101 : -95 + i, 30, i === 7 ? -99 : -94 + i, 40)];
  }
  const index = assertParity(data, [], [[35, -100], [40, -100]], ['2000-05-01']);
  const result = queryPins(index, 35, -100, '2000-05-01');
  assert.equal(result.length, 6);
  assert.equal(result[0].location.name, 'Chart 7');
  assert.equal(result[0].contains, true);
  assert.deepEqual(result.slice(1).map(hit => hit.location.name), ['Chart 0', 'Chart 1', 'Chart 2', 'Chart 3', 'Chart 4']);
  assert.equal(queryPins(index, 35, -100, '2000-05-01', 2).length, 2);
  data.rings['Chart 1'] = data.rings['Chart 0'];
  const tied = assertParity(data, [], [[35, -100]], ['2000-05-01']);
  const ties = queryPins(tied, 35, -100, '2000-05-01');
  assert.equal(ties[1].dist, ties[2].dist);
  assert.deepEqual(ties.slice(1, 3).map(hit => hit.location.name), ['Chart 0', 'Chart 1']);
});

test('locations without extent rings and invalid start dates are omitted', () => {
  const data = {locations: {
    Missing: {charts: [chart]}, Empty: {ref: 'empty', charts: [chart]},
    Invalid: {ref: 'a', charts: [{d: 'invalid'}]},
  }, rings: {empty: [], a: [ring(-100, 30, -90, 40)]}};
  const index = assertParity(data, [], [[35, -95]], ['2000-05-01']);
  assert.equal(index.locations.length, 2);
  assert.deepEqual(queryPins(index, 35, -95, '2000-05-01'), []);
});

test('missing query date or index produces no result', () => {
  const index = buildPinIndex(json(fixtureContext.data));
  for (const date of [undefined, null, '', NaN, Infinity, 1.25]) {
    assert.deepEqual(queryPins(index, 32, -96, date), []);
  }
  assert.deepEqual(queryPins(null, 32, -96, '1950-01-01'), []);
  assert.deepEqual(buildPinIndex({locations: {}}).locations, []);
});

test('C5 per-chart versioned paths, zoom and bounds survive the index and query', () => {
  const pm = ['sectionals/chart/half-a/2000-01-01.0123456789ab', 'sectionals/chart/half-b/2000-01-01.0123456789ab'];
  const pmz = [4, 11], pmb = [-100, 30, -90, 40];
  const data = {locations: {A: {ref: 'a', charts: [{...chart, pm, pmz, pmb}]}}, rings: {a: [ring(...pmb)]}};
  const hit = queryPins(buildPinIndex(data), 35, -95, '2000-05-01')[0];
  assert.equal(hit.location.name, 'A');
  assert.equal(hit.chart.pm, pm);
  assert.equal(hit.chart.pmz, pmz);
  assert.equal(hit.chart.pmb, pmb);
});
