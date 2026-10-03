import test from 'node:test';
import assert from 'node:assert/strict';
import { AirspaceIndex, sourceDay, decodeAirspaceTile, containsRings, batchTransfers, CURRENT, altSpan } from '../airspace.js';
import { tile } from './mvt-fixture.mjs';

const metadata = { archive_aero: { cycle_days: 28, regions: {
  us: { name: 'US', source: 'FAA NASR', cycles: ['2020-01-02', '2020-01-30', '2020-03-26'], boxes: [[-180, -85, 180, 85]] },
  fr: { name: 'France', source: 'SIA', cycles: ['2020-01-16'], boxes: [[-20, -85, 20, 85]] },
  br: { name: 'Brazil', source: 'DECEA', cycles: ['2020-01-02'], boxes: [[-80, -85, -20, 85]] }
} } };
const square = [[1024, 1024], [3072, 1024], [3072, 3072], [1024, 3072], [1024, 1024]];
const props = (extra = {}) => ({ rg: 'us', cls: 'B', v: 'v1', from: 20200102, lo: 0, loc: 'SFC', hi: 10000, hic: 'MSL', ...extra });
const feature = (p, paths = [square]) => ({ properties: p, paths });

test('source dates convert YYYYMMDD and ISO into Days; invalid calendars fail', () => {
  assert.equal(sourceDay(19700101), 0);
  assert.equal(sourceDay('1969-12-31'), -1);
  assert.equal(sourceDay(20200229), sourceDay('2020-02-29'));
  assert.throws(() => sourceDay(20200230), /Invalid/);
});

test('held cycles have exclusive 28-day expiry; holes and newest expiry suppress regions', () => {
  const index = new AirspaceIndex(metadata), d = sourceDay('2020-01-02');
  assert.equal(index.regionMask(d - 1), 0);
  assert.equal(index.regionMask(d), 5);
  assert.equal(index.regionMask(d + 14), 7);
  assert.equal(index.regionMask(d + 28), 3); // next US cycle starts, Brazil expires
  assert.equal(index.regionMask(d + 42), 1); // France expires
  assert.equal(index.regionMask(d + 56), 0); // gap in US held series
  assert.equal(index.regionMask(sourceDay('2020-03-26')), 1);
  assert.equal(index.regionMask(sourceDay('2020-04-23')), 0);
});

test('C6 line batches use Mercator, stable style codes, Days, masks and offsets', () => {
  const decoded = decodeAirspaceTile(tile([
    { name: 'class', features: ['A', 'B', 'C', 'D', 'E'].map((cls, i) => feature(props({ cls, rg: ['us', 'fr', 'br'][i % 3], to: 20200130 })))
      .concat([feature(props({ cls: '', lt: 'TMA' })), feature(props({ cls: 'E', lt: 'E5' }))]) },
    { name: 'efloor', features: ['700', '1200'].map(k => ({ type: 2, properties: { rg: 'us', k, from: 20200102 }, paths: [[[-32, 0], [4096, 4096]]] })) }
  ]), { z: 1, x: 1, y: 0 });
  const b = decoded.batch;
  assert.deepEqual([...b.style], [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual([...b.rg], [0, 1, 2, 0, 1, 0, 0, 0]);
  assert.equal(b.from[0], sourceDay('2020-01-02'));
  assert.equal(b.to[0], sourceDay('2020-01-30'));
  assert.equal(b.to.at(-1), CURRENT);
  assert.equal(b.starts.length, b.style.length + 1);
  assert.equal(b.starts.at(-1) * 2, b.positions.length);
  assert.equal(b.positions[0], 0.625);
  assert.equal(b.positions[1], 0.125);
  assert.equal(decoded.polygons.length, 7); // floor polygons retained for queries
  const transfers = batchTransfers(b);
  structuredClone(b, { transfer: transfers });
  assert.equal(b.positions.byteLength, 0);
  assert.equal(decoded.polygons[0].rings[0].length, 10); // query rings survived batch transfer
});

test('point-in-polygon handles holes, multipolygons and boundary points', () => {
  const ring = ps => new Int32Array(ps.flat());
  const outer = ring([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]);
  const hole = ring([[3, 3], [3, 7], [7, 7], [7, 3], [3, 3]]);
  const other = ring([[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]);
  assert.equal(containsRings([outer, hole, other], 1, 1), true);
  assert.equal(containsRings([outer, hole, other], 5, 5), false);
  assert.equal(containsRings([outer, hole, other], 25, 5), true);
  assert.equal(containsRings([outer], 0, 0), true);
  assert.equal(containsRings([outer], 15, 0), false);
});

test('query keeps lowest floor per badge, excludes notches, dedupes tiles and honors dates', () => {
  const index = new AirspaceIndex(metadata);
  const bytes = tile([{ name: 'class', features: [
    feature(props({ v: 'high', lo: 4000, name: 'CITY CLASS B' })),
    feature(props({ v: 'low', lo: 2000, name: 'CITY CLASS B' })),
    feature(props({ v: 'D', cls: 'D', lo: 0 })),
    feature(props({ v: 'notch', cls: 'C', ex: true })),
    feature(props({ v: 'future', cls: 'A', from: 20200326 })),
    feature(props({ v: 'expired', cls: 'C', to: 20200116 })),
    feature(props({ v: 'E', cls: 'E', lt: 'E5', lo: 700 })),
    feature(props({ v: 'fr', rg: 'fr', cls: 'A', lo: 60, loc: 'STD' }))
  ] }]);
  index.setTile('tile1', { z: 0, x: 0, y: 0 }, bytes);
  index.setTile('tile2', { z: 0, x: 0, y: 0 }, bytes);
  const rows = index.query(0, 0, sourceDay('2020-01-16'));
  assert.deepEqual(rows.map(r => r.badge), ['D', 'E', 'B', 'A']);
  assert.equal(rows[2].lo, 2000);
  assert.equal(rows[2].shortName, 'CITY');
  assert.equal(rows[2].cycle, '2020-01-02');
  assert.equal(rows[3].floorFt, 6000);
  assert.equal(rows[3].source, 'SIA');
  assert.deepEqual(index.query(0, 0, sourceDay('2020-01-16'), { classMask: 2 }).map(r => r.badge), ['E']);
  assert.deepEqual(index.query(0, 0, sourceDay('2020-03-01')), []);
  assert.deepEqual(index.query(150, 0, sourceDay('2020-01-16')), []);
  index.clear(); assert.deepEqual(index.query(0, 0, sourceDay('2020-01-16')), []);
});

test('wrapped queries and status use region boxes on either side of antimeridian', () => {
  const meta = { regions: { us: { name: 'US', source: 'FAA', cycles: ['2020-01-02'], boxes: [[170, -20, 190, 20]] } } };
  const index = new AirspaceIndex(meta);
  const border = [[3900, 1900], [4200, 1900], [4200, 2200], [3900, 2200], [3900, 1900]];
  index.setTile('tile', { z: 0, x: 0, y: 0 }, tile([{ name: 'class', features: [feature(props(), [border])] }]));
  assert.equal(index.query(-179, 0, sourceDay('2020-01-02')).length, 1);
  assert.equal(index.query(541, 0, sourceDay('2020-01-02')).length, 1);
  assert.deepEqual(index.regionsInView([530, -5, 550, 5]), ['us']);
  assert.deepEqual(index.regionsInView([170, -5, -170, 5]), ['us']);
});

test('status explains before-first, missing cycles and latest expiry with calendar arithmetic', () => {
  const index = new AirspaceIndex(metadata);
  assert.equal(index.regionStatus('us', sourceDay('2020-01-01')), 'US · no FAA NASR data before Jan 2, 2020');
  assert.equal(index.regionStatus('us', sourceDay('2020-01-10')), 'US · FAA NASR cycle Jan 2, 2020');
  assert.equal(index.regionStatus('us', sourceDay('2020-03-01')), 'US · no FAA NASR data Feb 27, 2020 to Mar 25, 2020');
  assert.equal(index.regionStatus('us', sourceDay('2020-04-23')), 'US · no FAA NASR data after Apr 22, 2020 yet');
  assert.deepEqual(index.status(0, [-10, -5, 10, 5], { loaded: false }), ['Loading airspace…']);
  assert.deepEqual(index.status(0, null, { configured: false }), ['Airspace data not configured']);
  assert.deepEqual(index.status(0, null, { enabled: false, loadError: 'HTTP 404' }), ['Airspace data unavailable: HTTP 404']);
  assert.equal(altSpan({ lo: 0, loc: 'SFC', hi: 100, hic: 'STD' }), 'SFC – FL100');
  assert.equal(altSpan({ lo: 2000, loc: 'MSL', hi: 3000, hic: 'MSL' }), '2,000 – 3,000 MSL');
});

test('stack reports the regions with a cycle in effect under a point plus their governing rows; credits list every region', () => {
  const index = new AirspaceIndex({ archive_aero: { regions: {
    us: { name: 'US', source: 'FAA NASR', source_url: 'https://faa.example/', cycles: ['2020-01-02'], boxes: [[-180, -85, 180, 85]] },
    br: { name: 'Brazil', source: 'DECEA', note: 'current snapshots only', cycles: ['2020-01-02'], boxes: [[-80, -85, -20, 85]] }
  } } });
  index.setTile('tile', { z: 0, x: 0, y: 0 }, tile([{ name: 'class', features: [feature(props({ v: 'b', lo: 2000, loc: 'MSL', name: 'CITY CLASS B' }))] }]));
  const day = sourceDay('2020-01-10'), here = index.stack(0, 0, day);
  assert.deepEqual(here.here, [{ rg: 'us', name: 'US', source: 'FAA NASR', note: null, cycle: '2020-01-02' }]);
  assert.deepEqual(here.rows.map(r => [r.badge, r.shortName, r.altSpan]), [['B', 'CITY', '2,000 – 10,000 MSL']]);
  assert.deepEqual(index.stack(-50, 0, day).here.map(h => [h.rg, h.note]), [['us', null], ['br', 'current snapshots only']]);
  assert.deepEqual(index.stack(0, 0, sourceDay('2021-01-01')), { here: [], rows: [] });
  assert.deepEqual(index.stack(NaN, 0, day), { here: [], rows: [] });
  assert.deepEqual(index.credits(), [{ source: 'FAA NASR', url: 'https://faa.example/' }, { source: 'DECEA', url: null }]);
});
