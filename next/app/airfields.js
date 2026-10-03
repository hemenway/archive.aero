// AirfieldsLayer, ported from src/viewer.js.
// Historical-airfield dots (index built offline from Paul Freeman's
// airfields-freeman.com: coords, operating years, source links). Dots
// follow the timeline — a field shows while the selected year falls in
// its operating span; undated fields always show, fainter. Hover shows
// the name (mouse only); click opens a card (#afPanel) with dates and
// source links. The dots are drawn and picked by the renderer from the C4
// binary; names and links come from the details file on first use.
import { Utils } from './utils.js';

const STATUS = ['open', 'gone', 'unknown'];

export class AirfieldsLayer {
  constructor(mapCtrl, timelineApp) {
    this.mapCtrl = mapCtrl;
    this.map = mapCtrl.map;
    this.r = mapCtrl.r;
    this.dp = mapCtrl.dp;
    this.timelineApp = timelineApp;
    this.arrays = null;      // { mx, my, start, end, status } typed arrays
    this.lats = null;
    this.lngs = null;
    this.detailsLoading = null; // the details file, fetched on first need (hover, card or browser)
    this.attached = false;    // arrays handed to the renderer
    this.loadPromise = null;
    this.year = null;
    // Off until a visitor turns it on (since 2026-09-25; it used to open
    // on), and the choice sticks per browser like the airspace switch.
    this.enabled = Utils.storageGet('airfieldsShown') === '1';
    // Status buckets are exhaustive: open (still operating) / gone
    // (closure evidence) / unknown (no closure evidence).
    this.statusFilter = { open: true, gone: true, unknown: true };
    try {
      const saved = JSON.parse(Utils.storageGet('airfieldsStatus') || '{}');
      for (const k of Object.keys(this.statusFilter)) {
        if (typeof saved[k] === 'boolean') this.statusFilter[k] = saved[k];
      }
    } catch (e) { /* keep defaults */ }
    this.hoverOk = !!(window.matchMedia && matchMedia('(hover: hover) and (pointer: fine)').matches);
    this.selected = null;     // index of the field whose card is open
    this._browserRevision = 0;
    this.applyDebounced = Utils.debounce(() => this._apply(), 120);
    this.els = {
      panel: document.getElementById('afPanel'),
      name: document.getElementById('afName'),
      dates: document.getElementById('afDates'),
      loc: document.getElementById('afLoc'),
      links: document.getElementById('afLinks')
    };
    document.getElementById('afClose')?.addEventListener('click', () => this.closeCard());
    document.getElementById('afBrowser')?.addEventListener('toggle', () => this._updateBrowser());
    document.getElementById('afOpen')?.addEventListener('click', () => {
      const index = Number(document.getElementById('afSelect').value);
      if (!this.enabled || !this.arrays || !this._visible(index)) return;
      this.timelineApp.closeToolsPanel();
      this.openCard(index, true);
    });
    this.map.on('click', (e) => {
      if (this.enabled && e.picked?.kind === 'airfield') this.openCard(e.picked.index);
    });
    this.map.on('move zoom viewreset', () => { this._updatePosition(); this._hideTip(); });
    this.map.on('moveend', () => this._updateBrowser());
    window.addEventListener('keydown', (e) => {
      if (!e.defaultPrevented && e.key === 'Escape' && this.selected != null &&
          (e.target === this.map.getContainer() || this.els.panel?.contains(e.target))) {
        e.preventDefault();
        this.closeCard();
      }
    });
    if (this.hoverOk) this._initHover();
  }

  init(dateStr) {
    if (dateStr) this.year = parseInt(dateStr.slice(0, 4), 10) || null;
    if (this.enabled) this._ensureLoaded();
  }

  setEnabled(on) {
    this.enabled = on;
    Utils.storageSet('airfieldsShown', on ? '1' : '0');
    if (on) {
      this._ensureLoaded();
    } else {
      this.closeCard();
      this._hideTip();
      // Hidden airfields leave the GPU entirely rather than drawing every
      // instance with an empty status mask.
      if (this.attached) { this.r.setAirfields(null); this.attached = false; }
    }
    this._updateBrowser();
  }

  _ensureLoaded() {
    if (!this.loadPromise) {
      this.loadError = false;
      this.loadPromise = this.dp.loadAirfields().then((arrays) => {
        if (!arrays) throw new Error('no airfield data in this build');
        this._build(arrays);
        this._apply();
      }).catch((err) => {
        console.error('airfield data failed to load', err);
        this.loadPromise = null; // next toggle retries
        Utils.toast('Airfield data unavailable');
        this.loadError = true;
        this._updateBrowser();
      });
    } else {
      this._apply();
    }
  }

  _build(arrays) {
    this.arrays = arrays;
    const n = arrays.mx.length;
    this.lats = new Float64Array(n);
    this.lngs = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      this.lngs[i] = arrays.mx[i] * 360 - 180;
      this.lats[i] = Math.atan(Math.sinh(Math.PI * (1 - 2 * arrays.my[i]))) * 180 / Math.PI;
    }
  }

  onDateChanged(dateStr) {
    const y = parseInt((dateStr || '').slice(0, 4), 10);
    if (!y || y === this.year) return;
    this.year = y;
    if (this.enabled && this.arrays) this.applyDebounced();
  }

  setStatusFilter(key, on) {
    if (!(key in this.statusFilter)) return;
    this.statusFilter[key] = on;
    Utils.storageSet('airfieldsStatus', JSON.stringify(this.statusFilter));
    this._apply();
  }

  // A field shows when its status bucket is enabled AND the selected year
  // falls inside its operating span. Missing bounds fail open: undated
  // fields always show (faint), a field with only a start shows from then
  // on, only an end shows until then. The renderer's shader and picker
  // apply the same rule to the same arrays.
  _visible(i) {
    const a = this.arrays, status = STATUS[a.status[i]];
    if (!(this.statusFilter[status] ?? true)) return false;
    const y = this.year, start = a.start[i], end = a.end[i];
    if (y == null) return true;
    if (!start && !end) return true;
    if (start && y < start) return false;
    if (status === 'open') return true;
    if (end && y > end) return false;
    return true;
  }

  _apply() {
    if (!this.enabled || !this.arrays) return;
    if (!this.attached) { this.r.setAirfields(this.arrays); this.attached = true; }
    const statusMask = STATUS.reduce((mask, k, i) => mask | (this.statusFilter[k] ? 1 << i : 0), 0);
    this.r.setAirfieldFilter({ year: this.year, statusMask });
    if (this.selected != null && !this._visible(this.selected)) this.closeCard();
    this._updateBrowser();
  }

  // Names, dates and links for every field, index-aligned with the arrays.
  // One file, loaded once: the browser lists every field in view by name.
  _details() {
    if (!this.detailsLoading) {
      this.detailsLoading = this.dp.airfieldDetailsAll().then((all) => {
        if (!Array.isArray(all)) throw new Error('no airfield details in this build');
        return all;
      });
      this.detailsLoading.catch(() => { this.detailsLoading = null; }); // next use retries
    }
    return this.detailsLoading;
  }

  async _detail(index) {
    return (await this._details())[index] || null;
  }

  // Native controls are the keyboard/touch-AT equivalent of canvas dots.
  // Only rebuild while expanded, retaining the user's selected airfield.
  async _updateBrowser() {
    if (!document.getElementById('afBrowser')?.open) return;
    const select = document.getElementById('afSelect');
    const previous = select.value;
    const revision = ++this._browserRevision;
    let entries = [], detailsFailed = false;
    if (this.enabled && this.arrays) {
      const bounds = this.map.getBounds();
      const indices = [];
      for (let i = 0; i < this.arrays.mx.length; i++) {
        if (this._visible(i) && bounds.contains([this.lats[i], this.lngs[i]])) indices.push(i);
      }
      let details;
      try {
        details = await this._details();
      } catch (e) {
        details = null;
        detailsFailed = true;
      }
      if (revision !== this._browserRevision) return; // a newer view replaced this one
      if (details) {
        entries = indices.map((i) => ({ i, p: details[i] })).filter(e => e.p)
          .sort((a, b) => a.p.name.localeCompare(b.p.name));
      }
    }
    select.replaceChildren();
    for (const { p, i } of entries) {
      const option = document.createElement('option');
      option.value = String(i);
      option.textContent = `${p.name}${p.state ? ' — ' + p.state : ''}`;
      select.appendChild(option);
    }
    if (entries.some(({ i }) => String(i) === previous)) select.value = previous;
    select.disabled = !entries.length;
    document.getElementById('afOpen').disabled = !entries.length;
    document.getElementById('afCount').textContent = !this.enabled ? 'Turn on Airfields to browse.'
      : !this.arrays ? (this.loadError ? 'Airfield data unavailable. Toggle Airfields to retry.' : 'Loading airfields…')
      : detailsFailed ? 'Airfield data unavailable. Toggle Airfields to retry.'
      : entries.length ? `${entries.length} airfields in view.` : 'No airfields match this view, date, and filters.';
  }

  static fmtSpan(p) {
    const s = p.start_year, e = p.end_year, k = p.last_known_year;
    if (p.status === 'open') return s ? `In operation ${s} – present` : 'Open today';
    if (s && e) return `In operation ~${s} – ${e}`;
    if (s && k) return `In operation ~${s} – at least ${k}`;
    if (s) return `In operation from ~${s}`;
    if (e) return `In operation until ~${e}`;
    if (k) return `Still listed in ${k}; closure date unknown`;
    return 'Operating dates unknown';
  }

  async openCard(index, focus = false) {
    let p;
    try {
      p = await this._detail(index);
    } catch (e) {
      Utils.toast('Airfield details unavailable');
      return;
    }
    if (!p) return;
    this.selected = index;
    this.r.setAirfieldSelected?.(index);
    const els = this.els;
    if (!els.panel) return;
    els.name.textContent = p.name;
    els.dates.textContent = AirfieldsLayer.fmtSpan(p);
    if (p.status === 'open') {
      const chip = document.createElement('span');
      chip.className = 'pin-near-chip af-open-chip';
      chip.textContent = 'open';
      els.dates.appendChild(chip);
    }
    els.loc.textContent = p.rel_location || p.state || '';
    els.links.textContent = '';
    const mkLink = (label, href) => {
      // Details come from a data file: only ever link out over http(s).
      let url;
      try { url = new URL(href, location.href); } catch (e) { return; }
      if (!['http:', 'https:'].includes(url.protocol)) return;
      const a = document.createElement('a');
      a.href = url.href;
      a.target = '_blank';
      a.rel = 'noopener';
      a.textContent = label;
      els.links.appendChild(a);
    };
    if (p.url) mkLink('Airfields-Freeman', p.url);
    if (p.oa) mkLink('OurAirports', `https://ourairports.com/airports/${encodeURIComponent(p.oa)}/`);
    els.panel.classList.add('visible');
    this._updatePosition();
    if (focus) els.panel.focus({ preventScroll: true });
  }

  closeCard() {
    if (this.els.panel?.contains(document.activeElement)) {
      document.getElementById('toolsBtn')?.focus();
    }
    if (this.selected != null) this.r.setAirfieldSelected?.(null);
    this.selected = null;
    this.els.panel?.classList.remove('visible');
  }

  // Same anchoring rules as PinInspector: beside the dot, flipping at
  // viewport edges, clear of the header bar; follows map moves.
  _updatePosition() {
    const panel = this.els.panel;
    if (this.selected == null || !panel?.classList.contains('visible')) return;
    const rect = this.map.getContainer().getBoundingClientRect();
    const pt = this.map.latLngToContainerPoint([this.lats[this.selected], this.lngs[this.selected]]);
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

  // Name tooltip on hover (mouse only), where Leaflet bound one per marker.
  _initHover() {
    const canvas = this.map.canvas;
    const tip = document.createElement('div');
    tip.className = 'leaflet-tooltip af-tip leaflet-tooltip-top';
    tip.setAttribute('role', 'tooltip');
    tip.style.display = 'none';
    this.map.getContainer().appendChild(tip);
    this.tip = tip;
    this.tipIndex = null;
    let pending = false, last = null;
    canvas.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse' || e.buttons) return;
      last = e;
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        const picked = this.enabled && this.attached ? this.r.pick(last.clientX, last.clientY) : null;
        this._showTip(picked?.kind === 'airfield' ? picked.index : null);
      });
    });
    canvas.addEventListener('pointerleave', () => this._hideTip());
  }

  async _showTip(index) {
    const canvas = this.map.canvas;
    canvas.style.cursor = index == null ? '' : 'pointer';
    if (index === this.tipIndex) return;
    this.tipIndex = index;
    if (index == null) { this.tip.style.display = 'none'; return; }
    let p;
    try { p = await this._detail(index); } catch (e) { return; }
    if (this.tipIndex !== index || !p) return;
    const tip = this.tip;
    // TextNode content so names render as text, never as HTML.
    tip.textContent = p.name;
    tip.style.display = '';
    const pt = this.map.latLngToContainerPoint([this.lats[index], this.lngs[index]]);
    // Leaflet's direction 'top' with offset [0, -8]; the stylesheet's
    // .leaflet-tooltip-top margin lifts it clear of its 6px arrow.
    tip.style.left = `${Math.round(pt.x - tip.offsetWidth / 2)}px`;
    tip.style.top = `${Math.round(pt.y - tip.offsetHeight - 8)}px`;
  }

  _hideTip() {
    if (!this.tip) return;
    this.tipIndex = null;
    this.tip.style.display = 'none';
    this.map.canvas.style.cursor = '';
  }
}
