const MS_PER_DAY = 86400000;

// The legacy inventory deliberately uses this frame for far-Aleutian rings.
const unwrap = lon => lon > 150 ? lon - 360 : lon;

export function pointInRing(lon, lat, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > lat) !== (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function distToRing(lon, lat, ring) {
  const k = Math.cos(lat * Math.PI / 180);
  let best = Infinity;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const ax = (ring[j][0] - lon) * k, ay = ring[j][1] - lat;
    const bx = (ring[i][0] - lon) * k, by = ring[i][1] - lat;
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len2)) : 0;
    const px = ax + t * dx, py = ay + t * dy;
    best = Math.min(best, px * px + py * py);
  }
  return Math.sqrt(best);
}

/** Build the inventory index without network or DOM state. */
export function buildPinIndex(data, publishedKeys = []) {
  const published = new Set(publishedKeys);
  const eraMemberCount = new Map();
  const locations = [];
  for (const [name, loc] of Object.entries(data.locations || {})) {
    const rawRings = (loc.ref && data.rings && data.rings[loc.ref]) || null;
    const ringClip = rawRings && rawRings.length ? rawRings[0] : null;
    let ringHit = null, bbox = null, crossesAM = false;
    if (ringClip) {
      let hasEast = false, hasWest = false;
      let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
      for (const pt of ringClip) {
        if (pt[0] > 150) hasEast = true;
        if (pt[0] < -90) hasWest = true;
        w = Math.min(w, pt[0]); e = Math.max(e, pt[0]);
        s = Math.min(s, pt[1]); n = Math.max(n, pt[1]);
      }
      crossesAM = hasEast && hasWest;
      ringHit = ringClip.map(pt => [unwrap(pt[0]), pt[1]]);
      bbox = [w, s, e, n];
    }
    const seen = new Set();
    const charts = [];
    for (const c of loc.charts || []) {
      const dupeKey = `${c.d}|${c.e || ''}`;
      if (seen.has(dupeKey)) continue;
      seen.add(dupeKey);
      const t0 = new Date(c.d).getTime();
      if (isNaN(t0)) continue;
      const t1 = c.e ? new Date(c.e).getTime() : NaN;
      const eraKey = (c.e && c.e !== c.d) ? `${c.d}_to_${c.e}` : c.d;
      eraMemberCount.set(eraKey, (eraMemberCount.get(eraKey) || 0) + 1);
      charts.push({
        d: c.d, e: c.e || null, t0,
        t1: isNaN(t1) ? t0 + 182 * MS_PER_DAY : t1,
        ed: c.ed, f: c.f || '', eraKey, pm: c.pm || null,
        pmz: c.pmz, pmb: c.pmb,
        published: published.has(eraKey),
      });
    }
    if (charts.length) locations.push({name, era: loc.era, ref: loc.ref, ringHit, ringClip, bbox, crossesAM, charts});
  }
  return {locations, publishedKeys: published, eraMemberCount};
}

/** Day is an integer UTC epoch Day or the inventory's ISO date string. */
export function queryPins(index, lat, lng, day, limit = 6) {
  if (!index?.locations) return [];
  const t = typeof day === 'number'
    ? (Number.isInteger(day) ? day * MS_PER_DAY : NaN)
    : (typeof day === 'string' ? new Date(day).getTime() : NaN);
  if (isNaN(t)) return [];
  let qlng = ((lng + 180) % 360 + 360) % 360 - 180;
  qlng = unwrap(qlng);
  const results = [];
  for (const loc of index.locations) {
    if (!loc.ringHit) continue;
    const inEffect = loc.charts.filter(c => t >= c.t0 && t < c.t1);
    if (!inEffect.length) continue;
    const contains = pointInRing(qlng, lat, loc.ringHit);
    const dist = contains ? 0 : distToRing(qlng, lat, loc.ringHit);
    for (const chart of inEffect) results.push({location: loc, chart, contains, dist});
  }
  results.sort((a, b) => (b.contains - a.contains) || (a.dist - b.dist));
  return results.slice(0, limit);
}
