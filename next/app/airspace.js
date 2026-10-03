// AirspaceLayer, ported from src/viewer.js: the switch, chips, status line and
// the pin card's "Airspace here" stack. Class airspace from three national AIS
// sources — FAA NASR (US), SIA (France) and DECEA GeoAISWEB (Brazil) — as
// polygon versions tagged with a region and a from/to validity interval. The
// renderer draws the lines with the sectional legend's symbology and applies
// the date, class and region masks on the GPU, so a date change repaints
// what is resident and fetches nothing. A region draws only while one of its
// held cycles is in effect on the date (each is good for 28 days); the
// status line says why when nothing is drawn. Off by default and remembered
// once chosen; a load that fails turns it back off and the status line says why.
import { Utils, dayOf } from './utils.js';

export class AirspaceLayer {
  static E_LABEL = { E2: 'surface area', E3: 'extension', E4: 'extension', E6: 'en route', E7: 'en route' };

  constructor(mapCtrl, timelineApp, { available }) {
    this.mapCtrl = mapCtrl;
    this.map = mapCtrl.map;
    this.r = mapCtrl.r;
    this.dp = mapCtrl.dp;
    this.timelineApp = timelineApp;
    this.available = available; // the manifest carries an airspace archive
    this.enabled = available && Utils.storageGet('airspaceShown') === '1';
    // Chip filters: Class A–D together (the solid/dashed line classes, plus
    // controlled airspace whose source publishes no class), E on its own.
    this.filter = { bcd: true, e: true };
    try {
      const saved = JSON.parse(Utils.storageGet('airspaceFilter') || '{}');
      for (const k of Object.keys(this.filter)) {
        if (typeof saved[k] === 'boolean') this.filter[k] = saved[k];
      }
    } catch (e) { /* keep defaults */ }
    this.loadError = null;   // why the last load failed, shown until the next attempt
    this.onStateChange = () => {}; // the map credits follow the layer (set by main)
    this.dateStr = null;
    this._stackRevision = 0;
    this.els = { status: document.getElementById('asStatus') };
    // The status line speaks for the region(s) under view, so it follows the map.
    this.map.on('moveend', () => this._renderStatus());
    // Metadata (cycles, extents) arrives with the first airspace demand.
    this.dp.on('metadata', () => {
      this.loadError = null;
      this._applyFilter();
      this._renderStatus();
      this.onStateChange();
      this.timelineApp.pinInspector?.refresh();
    });
    this.dp.on('error', ({ key, error }) => {
      if (key === 'airspace/metadata') this._loadFailed(error);
    });
  }

  init(dateStr) {
    this.dateStr = dateStr || null;
    this.mapCtrl.setAirspaceOn(this.enabled);
    this._applyFilter();
    this._renderStatus();
  }

  setEnabled(on) {
    if (!this.available) return;
    this.enabled = on;
    Utils.storageSet('airspaceShown', on ? '1' : '0');
    if (on) this.loadError = null;
    this.mapCtrl.setAirspaceOn(on);
    this._applyFilter();
    this._renderStatus();
    this.onStateChange();
    this.timelineApp.pinInspector?.refresh();
  }

  setFilter(key, on) {
    if (!(key in this.filter)) return;
    this.filter[key] = on;
    Utils.storageSet('airspaceFilter', JSON.stringify(this.filter));
    this._applyFilter();
    this.timelineApp.pinInspector?.refresh();
  }

  onDateChanged(dateStr) {
    if (!dateStr || dateStr === this.dateStr) return;
    this.dateStr = dateStr;
    this._applyFilter();
    this._renderStatus();
  }

  // Date, class and region masks are uniforms: no geometry is rebuilt.
  _applyFilter() {
    if (!this.dateStr) return;
    const day = dayOf(this.dateStr);
    const classMask = this.enabled ? (this.filter.bcd ? 1 : 0) | (this.filter.e ? 2 : 0) : 0;
    this.r.setAirspaceFilter({ day, classMask, regionMask: this.dp.airspaceRegionMask(day) });
  }

  // The switch goes back off; without the reason beside it, an unpublished
  // archive read as a switch that would not stay on.
  _loadFailed(error) {
    if (!this.enabled) return;
    console.error('airspace archive failed to load', error);
    this.enabled = false;
    this.loadError = AirspaceLayer.failureReason(error);
    Utils.storageSet('airspaceShown', '0');
    document.getElementById('airspaceBtn')?.setAttribute('aria-pressed', 'false');
    this.mapCtrl.setAirspaceOn(false);
    this._applyFilter();
    this._renderStatus();
    this.onStateChange();
    Utils.toast(`Airspace data unavailable: ${this.loadError}`, 6000);
  }

  _typeOn(p) {
    return p.cls === 'E' ? this.filter.e : this.filter.bcd;
  }

  // One line per region under view: "France · SIA cycle Oct 5, 2023", or why
  // the region draws nothing on the selected date. The data plane composes the
  // lines from the archive's metadata.
  _renderStatus() {
    const el = this.els.status;
    if (!el) return;
    el.textContent = '';
    if (!this.dateStr) return;
    const b = this.map.getBounds();
    const lines = this.dp.airspaceStatus(dayOf(this.dateStr),
      [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()],
      { configured: this.available, enabled: this.enabled });
    for (const text of lines) {
      const line = document.createElement('div');
      line.textContent = text;
      el.appendChild(line);
    }
  }

  // The airspace stack under a point on the selected date, into `el`
  // (the pin card's section). Reads the decoded tiles in demand, so it only
  // knows what is on screen: the FAA's Class E floors enter the tiles at z7.
  async renderStack(el, latlng) {
    el.textContent = '';
    el.hidden = true;
    const revision = ++this._stackRevision;
    if (!this.enabled || !latlng || !this.dateStr) return;
    let stack;
    try {
      stack = await this.dp.airspaceStack(latlng.lng, latlng.lat, dayOf(this.dateStr));
    } catch (e) {
      return;
    }
    if (revision !== this._stackRevision || !this.enabled) return;
    // Only regions with a cycle in effect under the pin have anything to
    // say; for the others the panel's status line already explains.
    const { here, rows } = stack;
    if (!here.length) return;
    // Rows arrive as the governing stack — one per class, or per type where
    // the source publishes no class, lowest floor first.
    const governing = rows.filter((p) => this._typeOn(p));
    const rg = governing[0]?.rg || here[0].rg;
    const region = here.find((h) => h.rg === rg) || here[0];
    const title = document.createElement('div');
    title.className = 'pin-as-title';
    title.textContent = `Airspace · ${region.source || rg} ${AirspaceLayer.fmtCycle(region.cycle)}`;
    el.appendChild(title);
    for (const p of governing) {
      const row = document.createElement('div');
      row.className = 'pin-as-row';
      const cls = document.createElement('span');
      cls.className = `pin-as-cls pin-as-cls-${p.cls || 'type'}`;
      cls.textContent = p.badge;
      const alt = document.createElement('span');
      alt.className = 'pin-as-alt';
      alt.textContent = p.altSpan;
      const name = document.createElement('span');
      name.className = 'pin-as-name';
      const bits = [p.shortName];
      if (p.rg === 'us') {
        if (AirspaceLayer.E_LABEL[p.lt]) bits.push(AirspaceLayer.E_LABEL[p.lt]);
      } else if (p.cls && p.lt) {
        bits.push(p.lt); // the badge shows the class; say which kind of area it is
      }
      if (p.hrs === 'NOTAM') bits.push('by NOTAM');
      else if (p.hrs === 'RMK' && p.rmk) bits.push(p.rmk);
      else if (p.hrs && p.hrs !== 'H24') bits.push(p.hrs); // AIP hours codes: HJ, HX, HO, HN, TS (H24 is the default)
      name.textContent = bits.filter(Boolean).join(' · ');
      name.title = [p.name, p.rmk].filter(Boolean).join(' — ');
      row.append(cls, alt, name);
      el.appendChild(row);
    }
    const note = document.createElement('div');
    note.className = 'pin-as-note';
    if (rg === 'us' && this.map.getZoom() < 7) {
      note.textContent = governing.length ? 'Zoom in to include Class E floors' : 'Zoom in for Class E; none of B, C or D here';
    } else if (!governing.length) {
      note.textContent = 'No class airspace here';
    } else if (region.note) {
      note.textContent = region.note;
    }
    if (note.textContent) el.appendChild(note);
    el.hidden = false;
  }

  // One line on why the archive's metadata could not be read: the data
  // plane reports a refused read as "HTTP N" and a network or CORS failure
  // as fetch's TypeError.
  static failureReason(err) {
    const msg = String((err && err.message) || err || '');
    const code = /HTTP (\d+)/.exec(msg);
    if (code) return code[1] === '404' ? 'archive not found (HTTP 404)' : `HTTP ${code[1]}`;
    if (/Failed to fetch|NetworkError|Load failed/i.test(msg)) return 'network or CORS error';
    if (!msg) return 'unknown error';
    return msg.length > 80 ? msg.slice(0, 79) + '…' : msg;
  }

  static fmtCycle(s) {
    const [y, m, d] = String(s).split('-').map(Number);
    if (!y || !m || !d) return String(s);
    return new Date(y, m - 1, d).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  }
}
