// ChartView: one chart shown alone, with its facts beside the map and an
// address of its own — the edition page /sectionals/chart/<slug>/<date>
// (URI-POLICY.md address plan; worklist 09 B2). It wraps MapController's solo
// view: the engine paints the chart, this class owns the panel, the document
// title, the canonical link and the history entry. An edition page rendered by
// next/pages/worker.mjs arrives with the panel already filled from the same
// module; this re-renders it once the viewer is up.
//
// Leaving solo by any route (the panel, Escape, a timeline change, the solo
// pill, the back button) reaches _onSoloExit through MapController.onSoloExit.
import { Utils } from './utils.js';
import { parseChartPath, chartPath, describe, panelHtml, slugFromArtifact } from '../pages/chart-page.mjs';

export class ChartView {
  constructor(mapCtrl, timelineApp, pinInspector, { base, chartsBase, site }) {
    this.mapCtrl = mapCtrl;
    this.timelineApp = timelineApp;
    this.pinInspector = pinInspector;
    this.base = base;
    this.chartsBase = chartsBase;
    this.site = site || '';
    this.el = document.getElementById('chartView');
    this.active = false;
    this.model = null;
    this._indexes = new Map();
    this._switching = false;
    this._fromHistory = false;
    this._revision = 0;
    this._title = document.title;
    this._canonical = document.querySelector('link[rel="canonical"]');
    this._canonicalHref = this._canonical?.getAttribute('href') || null;

    mapCtrl.onSoloExit = () => this._onSoloExit();
    this.el?.addEventListener('click', (e) => this._onClick(e));
    this.el?.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return;
      e.preventDefault();
      this.exit();
      mapCtrl.map.getContainer().focus();
    });
    window.addEventListener('popstate', () => this._onPopState());
  }

  // Without a chart index there are no edition pages: "View alone" stays the plain solo view.
  get enabled() { return !!(this.el && this.chartsBase); }

  loadIndex(slug) {
    if (!this._indexes.has(slug)) {
      const loading = fetch(`${this.chartsBase}${slug}.json`).then(r => (r.ok ? r.json() : null), () => null);
      this._indexes.set(slug, loading);
      loading.then(index => { if (!index) this._indexes.delete(slug); });
    }
    return this._indexes.get(slug);
  }

  // "View alone" from the pin card. The chart paints at once; the panel and
  // the address follow when its location's index has loaded.
  async openChart(loc, chart, members = 1) {
    const route = this.enabled && chart.pm ? slugFromArtifact(Array.isArray(chart.pm) ? chart.pm[0] : chart.pm) : null;
    if (!route) {
      this.mapCtrl.enterSolo(loc, chart, members);
      return false;
    }
    this.pinInspector?.clear();
    this._enter(() => this.mapCtrl.enterSolo(loc, chart, members));
    const revision = ++this._revision;
    const index = await this.loadIndex(route.slug);
    if (revision !== this._revision || !this.mapCtrl.soloState) return false; // superseded, or solo already left
    const model = index && describe(index, route.date);
    if (!model?.tiled) { this._hide(); return false; } // the plain solo view stays
    this._show(model, { push: true, focus: true });
    return true;
  }

  // An edition address: on boot (push: false), from the back button, or from
  // a link in the panel. fit: false keeps a view the address already carries.
  async openRoute({ slug, date }, { push = true, index = null, focus = false, fit = true } = {}) {
    if (!this.enabled) return false;
    const revision = ++this._revision;
    index = index || await this.loadIndex(slug);
    if (revision !== this._revision) return false;
    const model = index && describe(index, date);
    if (!model?.tiled) return false;
    const { chart } = model;
    const loc = { name: model.name, ref: model.ref, bbox: model.bbox, crossesAM: !model.bbox, ringClip: null };
    this._enter(() => {
      // The chart's date first: a frame paint leaves solo, so the new date's
      // paint happens before this chart is entered, not after.
      const selected = this.timelineApp.selectedDate;
      if (!selected || selected < chart.d || (chart.e && selected >= chart.e)) this.timelineApp.updateByDate(chart.d);
      this.pinInspector?.clear();
      this.mapCtrl.enterSolo(loc, chart, 1, { fit });
    });
    this._show(model, { push, focus });
    return true;
  }

  exit() { this.mapCtrl.exitSolo(); }

  // The chart's own view parameters: the edition page plus where the map is.
  shareUrl() {
    const center = this.mapCtrl.map.getCenter();
    const zoom = parseFloat(this.mapCtrl.map.getZoom().toFixed(2));
    const lng = ((center.lng + 180) % 360 + 360) % 360 - 180;
    return `${location.origin}${chartPath(this.base, this.model.slug, this.model.chart.d)}?lat=${center.lat.toFixed(4)}&lng=${lng.toFixed(4)}&zoom=${zoom}`;
  }

  // Solo re-entry (and a date change on the way in) fires onSoloExit; while
  // switching charts the panel must not close or rewrite the address.
  _enter(run) {
    this._switching = true;
    try { run(); } finally { this._switching = false; }
  }

  _show(model, { push, focus }) {
    this.model = model;
    this.active = true;
    document.getElementById('soloBar')?.classList.remove('visible'); // the panel is the solo banner
    this.el.innerHTML = panelHtml(model, { base: this.base, site: this.site });
    this.el.hidden = false;
    document.title = `${model.title} · archive.aero`;
    const path = chartPath(this.base, model.slug, model.chart.d);
    this._canonical?.setAttribute('href', `${location.origin}${path}`);
    const state = { ...(history.state || {}), chartView: `${model.slug}/${model.chart.d}` };
    try { history[push ? 'pushState' : 'replaceState'](state, '', path); } catch { /* file: origin or a rate limit */ }
    if (focus) this.el.focus({ preventScroll: true });
  }

  _hide() {
    this.active = false;
    this.model = null;
    this.el.hidden = true;
    document.title = this._title;
    if (this._canonicalHref) this._canonical?.setAttribute('href', this._canonicalHref);
  }

  _onSoloExit() {
    if (!this.active || this._switching) return;
    this._hide();
    // Leaving by the panel, Escape or the timeline: the address becomes the
    // viewer's permalink of this view. Leaving by the back button: history
    // has already set the address.
    if (!this._fromHistory) { try { history.replaceState(null, '', this.timelineApp.updateShareUrl()); } catch { /* as above */ } }
    this._fromHistory = false;
  }

  _onPopState() {
    const route = parseChartPath(location.pathname, this.base);
    if (route) {
      if (!this.active || `${this.model.slug}/${this.model.chart.d}` !== `${route.slug}/${route.date}`) {
        this.openRoute(route, { push: false }).then(ok => { if (!ok) Utils.toast('That chart is not in the archive'); });
      }
    } else if (this.active) {
      this._fromHistory = true;
      this.mapCtrl.exitSolo();
      this._fromHistory = false;
    }
  }

  _onClick(e) {
    const link = e.target.closest('a[data-cv-date]');
    if (link && this.model) {
      e.preventDefault();
      this.openRoute({ slug: this.model.slug, date: link.dataset.cvDate }, { push: true, focus: true });
      return;
    }
    const action = e.target.closest('[data-cv]')?.dataset.cv;
    if (action === 'exit') this.exit();
    else if (action === 'share') this.timelineApp.copyShareLink();
  }
}
