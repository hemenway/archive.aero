export const manifest = {
  version: 1, generated: '2026-10-03T00:00:00Z', tileBase: './t/', fileBase: './',
  eras: [
    { k: '1950-01-01_to_1960-01-01', h: '0123456789ab', b: [-125, 24, -67, 50], z: [4, 14], c: null },
    { k: '1960-01-01_to_1970-01-01', h: '123456789abc', b: [-125, 24, -67, 50], z: [4, 14], c: null },
    { k: '1960-01-01_to_1971-01-01', h: '23456789abcd', b: [-99, 30, -95, 35], z: [4, 14], c: null },
    { k: '1970-01-01_to_1971-01-01', h: '3456789abcde', b: [-125, 24, -67, 50], z: [4, 14], c: null }
  ],
  basemap: { p: 'basemap/fixture.0123456789ab', z: [0, 13], tileSize: 512 },
  airspace: { p: 'airspace/fixture.0123456789ab', z: [0, 11] },
  airfields: { bin: 'airfields.0123456789ab.bin', details: 'airfields.0123456789ab.json' },
  pins: { z: 5, margin: 2, base: 'pins.0123456789ab/', shards: [395] },
  charts: { base: 'charts.0123456789ab/' },
  airports: { base: 'airports.0123456789ab/', count: 1 },
  coverage: { segments: [['1950-01-01', '1960-01-01', 1, 70], ['1960-01-01', '1970-01-01', 2, 100], ['1970-01-01', '1971-01-01', 2, 100]] }
};
export const fields = [
  { name: 'Test Field', lat: 32.7767, lng: -96.797, status: 'open', start_year: 1930, state: 'TX', oa: 'KDAL' },
  { name: 'Closed Field', lat: 33, lng: -97, status: 'gone', start_year: 1940, end_year: 1965, state: 'TX' }
];
// The chart index of the stub pins' one location (served by the stub build itself).
export const chartIndex = {
  test_dallas: {
    slug: 'test_dallas', names: ['Test Dallas'], ref: 'extents/test_dallas', type: 'sectional', bbox: [-99, 30, -95, 35], am: false,
    charts: [
      { d: '1950-01-01', e: '1960-01-01', ed: '1', f: '', pm: 'sectionals/chart/test_dallas/1950-01-01.0123456789ab', pmz: [4, 14], pmb: [-125, 24, -67, 50] },
      { d: '1960-01-01', e: '1970-01-01', ed: '1', f: '', pm: 'sectionals/chart/test_dallas/1960-01-01.0123456789ab', pmz: [4, 14], pmb: [-99, 30, -95, 35] },
      { d: '1970-01-01', e: '1971-01-01', ed: '1', f: '' }
    ]
  }
};
// The one airport page of the stub build: inside the stub location's sheet.
export const airportIndex = {
  ktst: {
    ident: 'ktst', icao: 'KTST', iata: 'TST', name: 'Test Field International Airport', city: 'Dallas', region: 'US-TX', lat: 32.8471, lon: -96.8518, elev_ft: 607,
    first: '1950-01-01', last: '1970-01-01', editions: 3, tiled: 2,
    decades: [{ decade: 1950, editions: 1, tiled: 1, first: '1950-01-01', last: '1950-01-01' }, { decade: 1960, editions: 1, tiled: 1, first: '1960-01-01', last: '1960-01-01' }, { decade: 1970, editions: 1, tiled: 0, first: '1970-01-01', last: '1970-01-01' }],
    charts: [
      { slug: 'test_dallas', name: 'Test Dallas', type: 'sectional', d: '1950-01-01', e: '1960-01-01', ed: '1', tiled: true },
      { slug: 'test_dallas', name: 'Test Dallas', type: 'sectional', d: '1960-01-01', e: '1970-01-01', ed: '1', tiled: true, og: 'sectionals/chart/test_dallas/1960-01-01.jpg' },
      { slug: 'test_dallas', name: 'Test Dallas', type: 'sectional', d: '1970-01-01', e: '1971-01-01', ed: '1', tiled: false }
    ]
  }
};
