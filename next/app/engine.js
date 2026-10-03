// The map engine behind the ported interface: production's MapController
// (src/viewer.js) rebuilt on the C7 renderer and data plane. The interface
// classes keep calling the handful of Leaflet-shaped methods they always did
// (mapCtrl.map.getCenter(), .on('moveend'), .latLngToContainerPoint() ...);
// MapFacade answers those from the WebGL camera.
import { Utils } from './utils.js';

const MAX_LAT = 85.0511287798066;
const mercator = (lng, lat) => {
  const s = Math.sin(Math.max(-MAX_LAT, Math.min(MAX_LAT, lat)) * Math.PI / 180);
  return [(lng + 180) / 360, 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)];
};
const geographic = (x, y) => [x * 360 - 180, Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180 / Math.PI];

class MapFacade {
  constructor(renderer, container, canvas) {
    this.r = renderer;
    this.container = container;
    this.canvas = canvas;
    this.handlers = new Map();
    this._zoom = renderer.getCamera().zoom;
    this._settledZoom = this._zoom;
    renderer.on('move', () => {
      const zoom = this.r.getCamera().zoom;
      this._fire('move');
      if (zoom !== this._zoom) { this._zoom = zoom; this._fire('zoom'); }
    });
    renderer.on('moveend', () => {
      const zoom = this.r.getCamera().zoom;
      this._fire('moveend');
      if (zoom !== this._settledZoom) { this._settledZoom = zoom; this._fire('zoomend'); }
    });
    renderer.on('click', (e) => this._fire('click', {
      latlng: { lat: e.lat, lng: e.lng }, picked: e.picked, clientX: e.clientX, clientY: e.clientY
    }));
  }

  on(names, fn) {
    for (const name of names.split(' ')) {
      if (!this.handlers.has(name)) this.handlers.set(name, []);
      this.handlers.get(name).push(fn);
    }
    return this;
  }

  _fire(name, payload) {
    for (const fn of this.handlers.get(name) || []) fn(payload);
  }

  getContainer() { return this.container; }

  getCenter() {
    const c = this.r.getCamera();
    const [lng, lat] = geographic(c.x, c.y);
    return { lat, lng };
  }

  getZoom() { return this.r.getCamera().zoom; }

  // The visible box. Longitudes run past ±180 when the view straddles the
  // antimeridian, exactly as Leaflet's do, so contains() tries a full turn
  // either way.
  getBounds() {
    const nw = this.r.unproject({ x: 0, y: 0 });
    const se = this.r.unproject({ x: this.canvas.clientWidth, y: this.canvas.clientHeight });
    return {
      getSouth: () => se.lat, getNorth: () => nw.lat, getWest: () => nw.lng, getEast: () => se.lng,
      contains: ([lat, lng]) => lat >= se.lat && lat <= nw.lat
        && [0, 360, -360].some((k) => lng + k >= nw.lng && lng + k <= se.lng)
    };
  }

  latLngToContainerPoint(latlng) {
    const p = Array.isArray(latlng) ? { lat: latlng[0], lng: latlng[1] } : latlng;
    return this.r.project({ lng: p.lng, lat: p.lat });
  }

  setView([lat, lng], zoom) {
    const [x, y] = mercator(lng, lat);
    this.r.setCamera({ x, y, zoom });
  }

  flyTo([lat, lng], zoom) { this.r.flyTo({ lng, lat }, zoom); }

  // Leaflet order in, C7 order out: [[south, west], [north, east]].
  fitBounds([[s, w], [n, e]], { padding = 0, maxZoom } = {}) {
    this.r.fitBounds([w, s, e, n], { padding: Array.isArray(padding) ? padding[0] : padding, maxZoom });
  }

  panBy(dx, dy) {
    const c = this.r.getCamera(), world = 256 * 2 ** c.zoom;
    this.r.setCamera({ x: c.x + dx / world, y: c.y + dy / world, zoom: c.zoom });
  }

  zoomBy(delta) {
    const c = this.r.getCamera();
    this.r.setCamera({ x: c.x, y: c.y, zoom: Math.round(c.zoom) + delta }, { animate: true });
  }
}

export class MapController {
  constructor(dp, renderer, { onFatal = () => {} } = {}) {
    this.dp = dp;
    this.r = renderer;
    this.onFatal = onFatal;
    this.map = new MapFacade(renderer, document.getElementById('map'), document.getElementById('mapCanvas'));

    this.overlayOpacity = 1;
    this.chartsHidden = false;
    this.soloState = null;
    this.lastFrameDate = null;
    this.playing = false;
    // Set by the airspace switch: airspace tiles are demanded only while it is on.
    this.airspaceOn = false;

    this.plan = null;
    this.chartTiles = [];
    this.scrubDirection = 0;
    // 204 tiles are absent for good; failed ones leave the plans for 30 s so
    // the rest of their tile still swaps in, then become requestable again.
    this.absentKeys = new Set();
    this.failedKeys = new Set();
    this.painted = false;
    this.failed = false;
    this.spinnerTimeout = null;

    // DOM refs read on the hot scrub path, cached once.
    this.els = {
      loader: document.getElementById('loader'),
      opacitySlider: document.getElementById('toolOpacitySlider'),
      chartInfoEffective: document.getElementById('chartInfoEffective')
    };

    dp.on('tile', ({ key, bitmap }) => {
      try {
        renderer.upload(key, bitmap);
      } catch (error) {
        bitmap.close?.();
        this._degrade(key, error);
        return;
      }
      this._syncLoader();
      this._notePaint();
    });
    dp.on('absent', ({ key }) => {
      this.absentKeys.add(key);
      this._renderPlans();
      this._syncLoader();
      this._notePaint();
    });
    dp.on('airspace', ({ tileId, batch }) => renderer.setAirspaceTile(tileId, batch));
    dp.on('error', ({ key, error }) => {
      if (key === 'worker') { this.failed = true; this._syncLoader(); this.onFatal(error); }
      else if (key !== 'airspace/metadata') this._degrade(key, error); // the airspace switch reports its own load failures
    });
    renderer.on('evict', ({ key }) => dp.markEvicted(key));
    renderer.on('contextlost', () => Utils.toast('Graphics interrupted. Restoring charts…'));
    // The renderer reports every lost texture evicted; re-demanding fetches them again.
    renderer.on('contextrestored', () => this._demand());
    // Tiles follow the camera.
    renderer.on('move', () => this._demandSoon());

    // How long a scrub must settle before we repaint.
    this.scrubDebounceDelay = 25;
    this.showFrameDebounced = Utils.debounce(this.showFrame.bind(this), this.scrubDebounceDelay);

    this.resizeObserver = new ResizeObserver(() => { renderer.resize(); this._demandSoon(); });
    this.resizeObserver.observe(this.map.getContainer());
  }

  destroy() {
    cancelAnimationFrame(this._demandFrame);
    clearInterval(this._watchTimer);
    clearTimeout(this.spinnerTimeout);
    this.resizeObserver.disconnect();
  }

  // The data plane prefetches neighbouring dates itself from the scrub state
  // passed with each demand; the interface still calls this after a frame.
  prefetchDates() {}

  setPlaying(playing) {
    this.playing = playing;
    this._demandSoon();
  }

  setAirspaceOn(on) {
    this.airspaceOn = on;
    this._demandSoon();
  }

  // True once every item of the current frame's plan is delivered: playback
  // steps only then, so a slow connection shows whole frames, not a blur.
  frameReady() {
    return !this.lastFrameDate || this.dp.readiness(this.lastFrameDate, this.chartTiles) >= 1;
  }

  async showFrame(selectedDateStr) {
    const selectedTime = new Date(selectedDateStr).getTime();
    if (isNaN(selectedTime)) return;
    // Any frame paint leaves solo view.
    if (this.soloState) this._clearSoloState();
    this.scrubDirection = this.lastFrameDate
      ? Math.sign(selectedTime - new Date(this.lastFrameDate).getTime()) : 0;
    this.lastFrameDate = selectedDateStr;

    // A direct (non-debounced) showFrame supersedes any pending scrub repaint;
    // cancel it so a stale date can't repaint over this newer frame afterward.
    this.showFrameDebounced?.cancel?.();

    const opacityValue = this.chartsHidden ? 0 : parseFloat(this.els.opacitySlider?.value || 100) / 100;
    this.overlayOpacity = opacityValue;
    this._applyStyle();

    const inEffect = this.dp.eraCountAt(selectedDateStr);
    if (this.els.chartInfoEffective) this.els.chartInfoEffective.textContent = String(inEffect);
    if (inEffect === 0) Utils.toast('No charts available for selected date');
    this._demand();
  }

  _applyStyle() {
    this.r.setChartStyle({ opacity: this.overlayOpacity, hidden: this.chartsHidden });
  }

  _demandSoon() {
    if (this._demandFrame || !this.lastFrameDate) return;
    this._demandFrame = requestAnimationFrame(() => {
      this._demandFrame = 0;
      this._demand();
    });
  }

  // Tell the data plane what the view needs and hand the renderer the plan.
  _demand() {
    if (!this.lastFrameDate || this.failed) return;
    const camera = this.r.getCamera();
    this.chartTiles = this.r.visibleTiles(256);
    const basemapTiles = this.r.visibleTiles(512);
    const solo = this.soloState?.source || null;
    this.dp.setDemand({
      date: this.lastFrameDate,
      chartTiles: this.chartTiles,
      basemapTiles,
      airspaceTiles: this.airspaceOn ? this.chartTiles : [],
      center: { x: camera.x, y: camera.y },
      scrub: { direction: this.scrubDirection, velocity: this.playing ? 0.5 : 0, playing: this.playing },
      solo
    });
    this.plan = this.dp.planCharts(this.lastFrameDate, this.chartTiles, { solo });
    this._renderPlans();
    this._watch();
  }

  _skipped(key) { return this.absentKeys.has(key) || this.failedKeys.has(key); }

  _withoutSkipped(tiles) {
    return tiles.map(tile => ({ ...tile, items: tile.items.filter(item => !this._skipped(item.key)) }));
  }

  _renderPlans() {
    if (!this.plan) return;
    this.r.setChartPlan({ ...this.plan, tiles: this._withoutSkipped(this.plan.tiles) });
    this.r.setBasemapPlan(this._withoutSkipped(this.dp.planBasemap(this.r.visibleTiles(512))));
  }

  // C7 has no loading-change or paint event, so poll the data plane's counters
  // while a demand is in flight and stop once it has settled.
  _watch() {
    clearInterval(this._watchTimer);
    let polls = 0;
    this._watchTimer = setInterval(() => {
      this._syncLoader();
      this._notePaint();
      if (this.failed || (++polls > 10 && !this._chartsLoading() && this.frameReady())) clearInterval(this._watchTimer);
    }, 50);
    this._syncLoader();
    this._notePaint();
  }

  // True while tiles for the current demand are queued or in flight.
  _chartsLoading() {
    if (this.failed) return false;
    const stats = this.dp.stats();
    return stats.inflight + stats.queued > 0;
  }

  // Shows the "Loading charts" pill after a 100ms grace (so cache-hit
  // repaints never flash it) and hides it the moment nothing is pending.
  _syncLoader() {
    const loader = this.els.loader;
    if (!loader) return;
    if (this._chartsLoading()) {
      if (this.spinnerTimeout || loader.classList.contains('active')) return;
      this.spinnerTimeout = setTimeout(() => {
        this.spinnerTimeout = null;
        if (this._chartsLoading()) loader.classList.add('active');
      }, 100);
      return;
    }
    clearTimeout(this.spinnerTimeout);
    this.spinnerTimeout = null;
    loader.classList.remove('active');
  }

  _drawable(key) {
    const parts = key.split('/');
    const y = +parts.pop(), x = +parts.pop(), z = +parts.pop(), path = parts.join('/');
    for (let level = z; level >= 0; level--) {
      const factor = 2 ** (z - level);
      if (this.r.hasTexture(`${path}/${level}/${Math.floor(x / factor)}/${Math.floor(y / factor)}`)) return true;
    }
    return false;
  }

  // Reveal the canvas once a tile is fully drawable — or once nothing is left
  // to wait for (no chart here, or every item absent or failed), so overlays
  // and the basemap never stay hidden behind charts that will not paint.
  _notePaint() {
    if (this.painted || !this.plan) return;
    const waiting = item => !this._skipped(item.key) && !this._drawable(item.key);
    const pending = this.plan.tiles.some(t => t.items.some(waiting));
    const ready = this.plan.tiles.some(t => t.items.some(i => !this._skipped(i.key)) && !t.items.some(waiting));
    if (pending && !ready) return;
    // Observe texture residency, then allow the scheduled renderer frame to finish.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (this.painted) return;
      this.painted = true;
      document.body.classList.add('painted');
      performance.mark('first-chart-paint');
      window.dispatchEvent(new Event('first-chart-paint'));
    }));
  }

  // A failed tile degrades the view instead of ending it. Same message and
  // rate limit as production's failed-archive toast.
  _degrade(key, error) {
    console.warn(`archive.aero: ${key || 'data plane'}: ${error?.message || error}`);
    if (typeof key === 'string' && key.includes('/') && !this.failedKeys.has(key)) {
      this.failedKeys.add(key);
      setTimeout(() => this.failedKeys.delete(key), 30000);
    }
    const now = performance.now();
    if (!this._lastFailToast || now - this._lastFailToast > 4000) {
      this._lastFailToast = now;
      Utils.toast('Some chart data failed to load — showing base map only');
    }
    this._renderPlans();
    this._syncLoader();
    this._notePaint();
  }

  _clearSoloState() {
    const clip = this.soloState?.source?.clip;
    if (clip) this.r.setClipRing(clip.id, null);
    this.soloState = null;
    document.getElementById('soloBar')?.classList.remove('visible');
  }

  // Shows one chart alone, replacing the composite until the next frame
  // paint. Preferred source: the chart's own full-sheet PMTiles artifact
  // (durable URI, collar included; a half-sheet pair paints as two
  // sources). Fallback while artifacts roll out: the chart's era archive,
  // clipped to its footprint ring for multi-chart mosaics.
  enterSolo(loc, chart, members = 1) {
    // "View alone" is an explicit request to see this chart, so flip the layers
    // switch back on rather than flying to a blank map. A deliberately lowered
    // opacity slider is preserved — setChartsHidden(false) restores from the slider.
    if (this.chartsHidden) {
      this.setChartsHidden(false);
      document.getElementById('chartsToggleBtn')?.setAttribute('aria-pressed', 'true');
    }
    let source, bounds = null;
    if (chart.pm) {
      // Full-sheet artifacts carry their own extent — no clip ring needed.
      const paths = Array.isArray(chart.pm) ? chart.pm : [chart.pm];
      source = { paths, zoom: chart.pmz };
      // Fit to the union of the artifacts' own bounds. Skip on any
      // world-spanning span (antimeridian charts) — stay where the user is.
      const b = chart.pmb;
      if (b && b.every(Number.isFinite) && b[2] > b[0] && b[3] > b[1] && (b[2] - b[0]) <= 180) bounds = b;
    } else {
      const era = this.dp.eraSource(chart.eraKey);
      if (!era) {
        Utils.toast('This chart is not yet viewable as an overlay');
        return;
      }
      // The antimeridian-crossing ring (Western Aleutians East) can't clip in
      // one world copy — show that archive unclipped instead.
      const clipRing = (members > 1 && loc.ringClip && !loc.crossesAM) ? loc.ringClip : null;
      source = { paths: [era.path], zoom: era.zoom };
      if (clipRing) source.clip = { id: loc.ref || loc.name, ring: clipRing };
      // Fit only to the chart's own footprint; an antimeridian-crossing ring
      // has a degenerate bbox, so stay where the user is: they clicked inside it.
      if (loc.bbox && !loc.crossesAM) bounds = loc.bbox;
    }
    if (this.soloState) this._clearSoloState();
    this.soloState = { key: 'chart:' + source.paths.join('|'), locName: loc.name, source };
    if (source.clip) this.r.setClipRing(source.clip.id, source.clip.ring);

    const label = document.getElementById('soloLabel');
    if (label) label.textContent = `Viewing ${loc.name} · ${Utils.formatDateId(chart.d)}`;
    document.getElementById('soloBar')?.classList.add('visible');

    this._demand();
    if (bounds) this.map.fitBounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]], { padding: [40, 40], maxZoom: 11 });
  }

  exitSolo() {
    if (!this.soloState) return;
    if (this.lastFrameDate) {
      // showFrame's solo guard clears the state and banner, then the normal
      // composite path repaints the frame.
      this.showFrame(this.lastFrameDate).catch(err => console.warn('solo exit repaint failed', err));
    } else {
      this._clearSoloState();
    }
  }

  setOpacity(opacity) {
    this.overlayOpacity = opacity;
    this._applyStyle();
  }

  // The layers-panel charts switch. A persistent flag (not a one-shot
  // opacity write) because showFrame re-reads the slider on every era swap.
  setChartsHidden(hidden) {
    this.chartsHidden = hidden;
    const sliderValue = parseFloat(this.els.opacitySlider?.value || 100) / 100;
    this.setOpacity(hidden ? 0 : sliderValue);
  }
}
