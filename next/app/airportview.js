// AirportView: an airport page — /airports/<ident> (URI-POLICY.md address
// plan; worklist 09 B1) — in the running viewer: the map centred on the
// airport, the #chartView aside carrying the airport's chart editions by
// decade (markup from pages/airport-page.mjs, the same the Worker rendered
// into the page). An edition link opens that chart in the ChartView; the back
// button returns here. A timeline step leaves the page the way it leaves the
// chart view: the address becomes the viewer's permalink of the view.
import { Utils } from './utils.js';
import { AIRPORT_ZOOM, airportPath, describeAirport, panelHtml, parseAirportPath } from '../pages/airport-page.mjs';

export class AirportView {
  constructor(mapCtrl, timelineApp, chartView, pinInspector, { base, airportsBase, site }) {
    this.mapCtrl = mapCtrl;
    this.timelineApp = timelineApp;
    this.chartView = chartView;
    this.pinInspector = pinInspector;
    this.base = base;
    this.airportsBase = airportsBase;
    this.site = site || '';
    this.el = document.getElementById('chartView');
    this.active = false;
    this.model = null;
    this._records = new Map();
    this._revision = 0;
    this._title = document.title;
    this._canonical = document.querySelector('link[rel="canonical"]');
    this._canonicalHref = this._canonical?.getAttribute('href') || null;
    this.el?.addEventListener('click', (e) => this._onClick(e));
    this.el?.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape' || e.defaultPrevented || !this.active) return;
      e.preventDefault();
      this.exit();
      mapCtrl.map.getContainer().focus();
    });
    window.addEventListener('popstate', () => this._onPopState());
  }

  get enabled() { return !!(this.el && this.airportsBase); }

  load(ident) {
    if (!this._records.has(ident)) {
      const loading = fetch(`${this.airportsBase}${ident}.json`).then(r => (r.ok ? r.json() : null), () => null);
      this._records.set(ident, loading);
      loading.then(data => { if (!data) this._records.delete(ident); });
    }
    return this._records.get(ident);
  }

  // Boot on an airport address (push: false) or return to one from history.
  // fit: false keeps a view the address (or the server-rendered aside) already set.
  async open({ ident }, { push = false, fit = true } = {}) {
    if (!this.enabled) return false;
    const revision = ++this._revision;
    const data = await this.load(ident);
    if (revision !== this._revision) return false;
    const model = describeAirport(data);
    if (!model) return false;
    this.pinInspector?.clear();
    if (this.chartView?.active) this.chartView.exit();
    if (fit) this.mapCtrl.map.setView([model.lat, model.lon], AIRPORT_ZOOM);
    this._show(model, { push });
    return true;
  }

  exit() {
    if (!this.active) return;
    this._hide();
    try { history.replaceState(null, '', this.timelineApp.updateShareUrl()); } catch { /* file: origin or a rate limit */ }
  }

  // A timeline step while the page is up: leave it like the chart view does.
  onDateChange() { if (this.active) this.exit(); }

  _show(model, { push }) {
    this.model = model;
    this.active = true;
    this.el.innerHTML = panelHtml(model, { base: this.base, site: this.site });
    this.el.hidden = false;
    document.title = `${model.title} · archive.aero`;
    const path = airportPath(this.base, model.ident);
    this._canonical?.setAttribute('href', `${location.origin}${path}`);
    try { history[push ? 'pushState' : 'replaceState']({ ...(history.state || {}), airport: model.ident }, '', path); } catch { /* as above */ }
  }

  _hide() {
    this.active = false;
    this.model = null;
    this.el.hidden = true;
    document.title = this._title;
    if (this._canonicalHref) this._canonical?.setAttribute('href', this._canonicalHref);
  }

  _onPopState() {
    const route = parseAirportPath(location.pathname, this.base);
    if (route) {
      if (!this.active || this.model.ident !== route.ident) this.open(route, { fit: false }).then(ok => { if (!ok) Utils.toast('That airport page is not in the archive'); });
    } else if (this.active) {
      this._hide();
    }
  }

  _onClick(e) {
    if (!this.active) return;
    const link = e.target.closest('a[data-ap-chart]');
    if (link) {
      e.preventDefault();
      const [slug, date] = link.dataset.apChart.split('/');
      // The chart view takes the aside and pushes its own address; this page is one step back.
      this.active = false; this.model = null;
      this.chartView.openRoute({ slug, date }, { push: true, focus: true }).then(ok => { if (!ok) Utils.toast('That chart is not in the archive'); });
      return;
    }
    const action = e.target.closest('[data-ap]')?.dataset.ap;
    if (action === 'exit') this.exit();
    else if (action === 'share') this.timelineApp.copyShareLink();
  }
}
