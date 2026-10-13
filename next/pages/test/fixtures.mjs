// Shared by the edition-page tests: a four-edition location and a minimal built shell page.
export const index = {
  slug: 'boston', names: ['Boston'], ref: 'sectional/boston', type: 'sectional', bbox: [-73.5, 40, -66.9, 44.1], am: false,
  charts: [
    { d: '1953-06-01', e: '1953-12-01', ed: '31', f: '', pm: 'sectionals/chart/boston/1953-06-01.aaaaaaaaaaaa', pmz: [0, 11], pmb: [-73.4, 40.1, -67, 44] },
    { d: '1953-12-01', e: '1954-06-01', ed: '32', f: '', pm: ['sectionals/chart/boston/1953-12-01-north.bbbbbbbbbbbb', 'sectionals/chart/boston/1953-12-01-south.cccccccccccc'], pmz: [0, 11], pmb: [-73.4, 40.1, -67, 44] },
    { d: '1954-06-01', e: '1954-12-01', ed: '33', f: '' },
    { d: '1954-12-01', e: '1955-06-01', ed: '1234567', f: 'L', pm: 'sectionals/chart/boston/1954-12-01.dddddddddddd', pmz: [0, 11], pmb: [-180, -85.0512, 180, 85.0512] }
  ]
};

export const shell = '<!doctype html><html lang=en><head><meta charset=utf-8><title>archive.aero - Historical Aeronautical Chart Viewer</title>'
  + '<meta name=description content="Explore historical charts."><link rel=canonical href=https://archive.aero/><meta property=og:type content=website>'
  + '<meta property="og:title" content="archive.aero — Viewer"><meta name=twitter:card content=summary_large_image><meta name=theme-color content=#0a0e12>'
  + '<link rel=icon href="data:image/svg+xml,<svg xmlns=\'http://www.w3.org/2000/svg\'><path d=\'M1 2>3\'/></svg>"><link rel=preload as=fetch href=/next/startup.json crossorigin>'
  + '<script>boot()</script></head><body><div id=map></div><aside class=chart-view id=chartView role=region aria-label="Chart view" tabindex=-1 hidden></aside><div id=soloBar></div></body></html>';

// An airport record (scripts/next_build_airports.py) inside the fixture location's sheets.
export const airport = {
  ident: 'kbos', icao: 'KBOS', iata: 'BOS', name: 'Boston Logan International Airport', city: 'Boston', region: 'US-MA', lat: 42.3643, lon: -71.0052, elev_ft: 19,
  first: '1933-03-01', last: '1970-01-01', editions: 5, tiled: 4,
  decades: [{ decade: 1930, editions: 1, tiled: 1, first: '1933-03-01', last: '1933-03-01' }, { decade: 1950, editions: 2, tiled: 2, first: '1953-06-01', last: '1953-12-01' }, { decade: 1960, editions: 1, tiled: 1, first: '1960-01-01', last: '1960-01-01' }, { decade: 1970, editions: 1, tiled: 0, first: '1970-01-01', last: '1970-01-01' }],
  charts: [
    { slug: 'boston', name: 'Boston', type: 'sectional', d: '1933-03-01', e: '1933-08-01', ed: '1', tiled: true },
    { slug: 'boston', name: 'Boston', type: 'sectional', d: '1953-06-01', e: '1953-12-01', ed: '31', tiled: true },
    { slug: 'boston', name: 'Boston', type: 'sectional', d: '1953-12-01', e: '1954-06-01', ed: '32', tiled: true, og: 'sectionals/chart/boston/1953-12-01.jpg' },
    { slug: 'boston_tac', name: 'Boston TAC', type: 'tac', d: '1960-01-01', e: '1960-07-01', ed: '1', tiled: true },
    { slug: 'boston', name: 'Boston', type: 'sectional', d: '1970-01-01', e: '1970-07-01', ed: '33', tiled: false }
  ]
};
