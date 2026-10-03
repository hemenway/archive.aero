// PinInspector, ported from src/viewer.js.
// Map-click pin + a compact popup showing THE chart at that point (the one
// whose footprint contains the click, else the nearest), anchored beside
// the click and following the pin as the map moves. Chart data comes from
// the data plane's pin shards (fetched on the first click in an area);
// "View alone" hands the chart to MapController.enterSolo. Clicking again
// moves the pin.
import { Utils } from './utils.js';

export class PinInspector {
  constructor(mapCtrl, timelineApp) {
    this.mapCtrl = mapCtrl;
    this.timelineApp = timelineApp;
    this.latlng = null;
    this.loaded = false;
    this._revision = 0;
    this.els = {
      panel: document.getElementById('pinPanel'),
      loc: document.getElementById('pinLoc'),
      badge: document.getElementById('pinBadge'),
      dates: document.getElementById('pinDates'),
      actions: document.getElementById('pinActions'),
      airspace: document.getElementById('pinAirspace')
    };
    this.refreshDebounced = Utils.debounce(() => this.refresh(), 150);

    document.getElementById('pinClose')?.addEventListener('click', () => this.clear());
    document.getElementById('soloExit')?.addEventListener('click', () => this.mapCtrl.exitSolo());
    mapCtrl.map.on('click', (e) => {
      // An airfield-dot click opens its own card; don't also move the pin.
      if (e.picked?.kind === 'airfield') return;
      this.setPin(e.latlng);
    });
    mapCtrl.map.getContainer().addEventListener('keydown', e => {
      if (e.target !== mapCtrl.map.getContainer() || e.key !== 'Enter' || e.altKey || e.ctrlKey || e.metaKey) return;
      e.preventDefault();
      this.setPin(mapCtrl.map.getCenter(), true);
    });
    // Keep the popup glued to the pin while panning/zooming.
    mapCtrl.map.on('move zoom viewreset', () => this._updatePosition());
    window.addEventListener('keydown', (e) => {
      if (e.defaultPrevented || e.key !== 'Escape') return;
      if (e.target !== mapCtrl.map.getContainer() && !this.els.panel?.contains(e.target)) return;
      // Step out of solo first; a second Escape clears the pin. Coexists
      // with the panel/overlay Escape handlers above.
      if (this.mapCtrl.soloState) {
        e.preventDefault();
        this.mapCtrl.exitSolo();
      } else if (this.latlng) {
        e.preventDefault();
        this.clear();
      }
    });
  }

  async setPin(latlng, focus = false) {
    this.latlng = latlng;
    this.mapCtrl.r.setPin({ lng: latlng.lng, lat: latlng.lat });
    this.els.panel?.classList.add('visible');
    if (focus) this.els.panel?.focus({ preventScroll: true });
    if (!this.loaded) {
      this._render(null, null, 'Loading chart details…');
      this._updatePosition();
    }
    await this.refresh();
  }

  clear() {
    if (this.els.panel?.contains(document.activeElement)) this.mapCtrl.map.getContainer().focus();
    this.mapCtrl.r.setPin(null);
    this.latlng = null;
    this._revision++;
    this.els.panel?.classList.remove('visible');
    // Deliberately does NOT exit solo: closing the card is the natural way
    // to see the solo chart unobstructed, and the solo bar keeps its own
    // "Show all charts" exit (as do Escape and any timeline change).
  }

  // Re-resolve the chart at the pin for the selected date. Also runs
  // (debounced) when the timeline moves while pinned.
  async refresh() {
    if (!this.latlng) return;
    const dateStr = this.timelineApp.selectedDate
      || this.timelineApp.frames[this.timelineApp.currentIndex]?.date;
    if (!dateStr) return;
    const latlng = this.latlng, revision = ++this._revision;
    let results;
    try {
      results = await this.mapCtrl.dp.queryPin(latlng.lng, latlng.lat, dateStr);
    } catch (e) {
      if (revision === this._revision) {
        this._render(null, null, 'Chart details unavailable — click again to retry.');
        this._updatePosition();
      }
      return;
    }
    if (revision !== this._revision) return; // the pin moved or cleared meanwhile
    this.loaded = true;
    this._render(results[0] || null, dateStr);
    this._updatePosition();
  }

  // Fill the popup with one chart (or a message when result is null).
  // No per-chart source links: exposing the chart -> source-scan mapping
  // invites scraping the catalog; the footer links sources.html instead.
  _render(result, dateStr, message) {
    const els = this.els;
    if (!els.panel) return;
    els.actions.textContent = '';
    this._renderAirspace();
    if (!result) {
      els.loc.textContent = message || 'No chart here';
      els.badge.style.display = 'none';
      els.dates.textContent = (!message && dateStr)
        ? `No charts in effect on ${Utils.formatDateId(dateStr)}` : '';
      return;
    }
    const { location: loc, chart, contains, members } = result;
    els.loc.textContent = loc.name;
    els.badge.style.display = '';
    els.badge.textContent = PinInspector.edLabel(chart.ed);
    els.dates.textContent = chart.e
      ? `${Utils.formatDateId(chart.d)} – ${Utils.formatDateId(chart.e)}`
      : Utils.formatDateId(chart.d);
    if (!contains) {
      const chip = document.createElement('span');
      chip.className = 'pin-near-chip';
      chip.textContent = 'nearby';
      els.dates.appendChild(chip);
    }
    if (chart.pm || chart.published) {
      const btn = document.createElement('button');
      btn.className = 'pin-view-alone';
      btn.type = 'button';
      btn.textContent = 'View alone';
      btn.addEventListener('click', () => this.mapCtrl.enterSolo(loc, chart, members));
      els.actions.appendChild(btn);
    } else {
      const note = document.createElement('span');
      note.className = 'pin-unavailable';
      note.textContent = 'not yet tiled';
      els.actions.appendChild(note);
    }
  }

  // "Airspace here": the class-airspace stack under the pin on the selected
  // date, read from the airspace overlay's decoded tiles — so only while
  // that layer is on, since it is the source of the polygons.
  _renderAirspace() {
    const el = this.els.airspace;
    if (!el) return;
    const layer = this.timelineApp.airspaceLayer;
    if (layer) layer.renderStack(el, this.latlng).then(() => this._updatePosition());
    else el.hidden = true;
  }

  // Anchor the popup beside the pin, flipping to stay inside the viewport
  // and below the header bar. Runs on every render and map move/zoom.
  _updatePosition() {
    const panel = this.els.panel;
    if (!this.latlng || !panel?.classList.contains('visible')) return;
    const map = this.mapCtrl.map;
    const rect = map.getContainer().getBoundingClientRect();
    const pt = map.latLngToContainerPoint(this.latlng);
    const w = panel.offsetWidth, h = panel.offsetHeight;
    let x = rect.left + pt.x + 18;
    let y = rect.top + pt.y - 12;
    if (x + w > window.innerWidth - 8) x = rect.left + pt.x - w - 18;
    if (x < 8) x = 8;
    if (y + h > window.innerHeight - 8) y = window.innerHeight - h - 8;
    if (y < 78) y = 78; // clear the header bar
    panel.style.left = x + 'px';
    panel.style.top = y + 'px';
  }

  // Old-era rows carry LOC catalog numbers in the edition column — only
  // present plausible edition numbers as an edition (same rule as
  // contribute.html's shelf labels).
  static edLabel(ed) {
    if (ed && /^\d+$/.test(ed) && +ed < 150) return `ed. ${ed}`;
    if (ed && !/^\d+$/.test(ed) && ed !== 'Unknown') return ed;
    return 'scan';
  }
}
