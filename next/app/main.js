import { createRenderer, RendererUnsupportedError } from '@next/renderer';
import { createDataPlane } from '@next/dataplane';
import { createStore } from './store.js';
import { project, unproject } from './camera.js';
const $ = id => document.getElementById(id);
const storage = { get(key) { try { return localStorage.getItem(key); } catch { return null; } }, set(key, value) { try { localStorage.setItem(key, value); } catch {} } };
const number = value => value !== null && value.trim() !== '' && Number.isFinite(+value) ? +value : null;
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));
const day = date => Math.floor(Date.parse(date) / 86400000);
const absentKeys = new Set(), failedKeys = new Set();
let store, r, dp, frames, fields, fieldEntries = [], pinRevision = 0, browserRevision = 0, lastUrl = -Infinity, lastFailureToast = -Infinity, urlTimer, loadingTimer, loading = false, playingTimer, returnFocus, plan, chartTiles, manifest, statsTimer, fieldsPromise, failed = false, scrubDirection = 0, painted = false;
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toast.timer); toast.timer = setTimeout(() => $('toast').hidden = true, 2500); }
function fatal(error) {
  failed = true; clearTimeout(loadingTimer); $('loader').hidden = true;
  if (store) store.set({ playing: false }); clearInterval(statsTimer);
  $('fatalError').replaceChildren(document.createTextNode(error instanceof RendererUnsupportedError ? 'This browser cannot run the new chart renderer. ' : `${error.message || error}. Reload to retry. `));
  const a = document.createElement('a'); a.href = '/'; a.textContent = 'Open the current viewer'; $('fatalError').append(a); $('fatalError').hidden = false;
}
// A failed tile degrades the view instead of ending it: the item leaves the
// plans for 30 s so the rest of the tile still swaps in, then becomes
// requestable again. Only a dead data Worker is fatal.
function degrade(key, error) {
  console.warn(`archive.aero: ${key || 'data plane'}: ${error?.message || error}`);
  if (key === 'airspace/metadata') { if (store) uniforms(store.get()); return; }
  if (typeof key === 'string' && key.includes('/') && !failedKeys.has(key)) { failedKeys.add(key); setTimeout(() => failedKeys.delete(key), 30000); }
  if (performance.now() - lastFailureToast > 10000) { lastFailureToast = performance.now(); toast('Some chart tiles failed to load'); }
  if (plan) { renderPlans(); setLoading(); notePaint(); }
}
const skipped = key => absentKeys.has(key) || failedKeys.has(key);
function closePanel(id, button) { if (store?.get().panels[id]) store.set({ panels: { ...store.get().panels, [id]: false } }); const focused = $(id).contains(document.activeElement); $(id).hidden = true; if (button) $(button).setAttribute('aria-expanded', 'false'); if (focused && button) $(button).focus(); }
function togglePanel(id, button) {
  const was = $(id).hidden; closePanel('siteMenu', 'menuToggle'); closePanel('toolsPanel', 'toolsBtn');
  $(id).hidden = !was; $(button).setAttribute('aria-expanded', String(was));
  store?.set({ panels: { ...store.get().panels, [id]: was } });
}
function showHelp() {
  returnFocus = document.activeElement; $('shortcutsOverlay').hidden = false;
  $('main-content').inert = true; document.querySelector('header').inert = true; $('closeShortcuts').focus();
  store?.set({ panels: { ...store.get().panels, help: true } });
}
function hideHelp() {
  $('shortcutsOverlay').hidden = true; $('main-content').inert = false; document.querySelector('header').inert = false;
  returnFocus?.focus(); store?.set({ panels: { ...store.get().panels, help: false } });
}
// Static chrome works while the manifest is loading or unavailable.
$('menuToggle').onclick = () => togglePanel('siteMenu', 'menuToggle');
$('toolsBtn').onclick = () => togglePanel('toolsPanel', 'toolsBtn');
$('helpBtn').onclick = showHelp; $('closeShortcuts').onclick = hideHelp;
$('shortcutsOverlay').onclick = e => { if (e.target === e.currentTarget) hideHelp(); };
$('shortcutsOverlay').onkeydown = e => { if (e.key === 'Tab') { e.preventDefault(); $('closeShortcuts').focus(); } if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); hideHelp(); } };
document.addEventListener('click', e => { if (!e.target.closest('.menu-trigger')) closePanel('siteMenu', 'menuToggle'); if (!e.target.closest('#toolsControl')) closePanel('toolsPanel', 'toolsBtn'); });
setTimeout(() => { if (!store?.get().playing) $('warningOverlay').hidden = false; }, 1500);
setTimeout(() => $('warningOverlay').hidden = true, 8000);
function shareURL() {
  const state = store.get(), camera = unproject(state.camera.x * 256, state.camera.y * 256, 0), url = new URL(location.href);
  for (const [key, value] of Object.entries({ date: state.date, lat: camera.lat.toFixed(4), lng: camera.lng.toFixed(4), zoom: +state.camera.zoom.toFixed(2), pin: state.pin ? `${state.pin.lat.toFixed(4)},${state.pin.lng.toFixed(4)}` : null })) {
    if (value === null) url.searchParams.delete(key); else url.searchParams.set(key, value);
  }
  return url.href;
}
function writeUrl() { clearTimeout(urlTimer); const wait = 250 - (performance.now() - lastUrl); if (wait > 0) { urlTimer = setTimeout(writeUrl, wait); return; } const url = shareURL(); if (location.href !== url) { history.replaceState(null, '', url); lastUrl = performance.now(); } }
async function share() { if (!store) return; try { await navigator.clipboard.writeText(shareURL()); toast('Share link copied'); } catch { toast('Copy the share link from your address bar'); writeUrl(); } }
async function fullscreen() {
  const element = document.documentElement, exit = document.exitFullscreen || document.webkitExitFullscreen, enter = element.requestFullscreen || element.webkitRequestFullscreen;
  if (!enter) { toast('Fullscreen is unavailable on this browser'); return; }
  try { if (document.fullscreenElement || document.webkitFullscreenElement) await exit?.call(document); else await enter.call(element); } catch { toast('Fullscreen is unavailable on this browser'); }
}
$('shareBtn').onclick = share; $('fullscreenBtn').onclick = fullscreen;
$('locateBtn').onclick = async () => {
  if (!r) return;
  $('locateBtn').disabled = true;
  try {
    let lat, lng, label;
    try {
      const pos = await new Promise((resolve, reject) => navigator.geolocation ? navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 10000, enableHighAccuracy: true, maximumAge: 0 }) : reject(new Error('GPS unavailable')));
      lat = pos.coords.latitude; lng = pos.coords.longitude; label = 'GPS';
    } catch {
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 4000);
      try { const response = await fetch('https://get.geojs.io/v1/ip/geo.json', { signal: controller.signal }); if (!response.ok) throw new Error('IP lookup failed'); const data = await response.json(); lat = number(data.latitude); lng = number(data.longitude); label = 'IP'; } finally { clearTimeout(timer); }
    }
    if (lat === null || lng === null || !Number.isFinite(lat) || !Number.isFinite(lng) || Math.abs(lat) > 90) throw new Error('Invalid location');
    r.flyTo({ lat, lng }, 10); toast(`Centered on your location (${label})`);
  } catch { toast('Unable to determine location'); } finally { $('locateBtn').disabled = false; }
};
function selectDate(date) { if (!store || !Number.isFinite(Date.parse(date))) return; const previous = store.get().date; const value = date < manifest.dateBounds.min ? manifest.dateBounds.min : date > manifest.dateBounds.max ? manifest.dateBounds.max : date; scrubDirection = Math.sign(day(value) - day(previous)); store.set({ date: value, playing: false, solo: null }); }
function frameIndex(date) { let i = frames.findLastIndex(d => d <= date); return Math.max(0, i); }
function step(amount, wrap = false) { const state = store.get(); let index = frameIndex(state.date) + amount; index = wrap ? (index + frames.length) % frames.length : clamp(index, 0, frames.length - 1); store.set({ date: frames[index], solo: null }); }
function manualStep(amount) { store.set({ playing: false }); step(amount); }
function syncPlayback(state, previous) {
  if (state.playing === previous?.playing) return;
  clearTimeout(playingTimer);
  for (const element of document.querySelectorAll('[aria-live], #timelineStatus, #loader, #afCount')) element.setAttribute('aria-live', state.playing ? 'off' : 'polite');
  if (state.playing) {
    $('warningOverlay').hidden = true;
    // Playback steps through eras, so it leaves single-chart view; the readiness gate below watches the era plan.
    if (state.solo) store.set({ solo: null });
    const tick = () => { if (!store.get().playing || failed) return; if (dp.readiness(store.get().date, chartTiles) >= 1) step(1, true); playingTimer = setTimeout(tick, 2000); };
    playingTimer = setTimeout(tick, 2000);
  }
}
function coverageAt(date) { const segments = manifest.coverage?.segments || []; return segments.find(([a, b]) => date >= a && date < b) || [null, null, 0, 0]; }
function coverageText(date) { const [, , count, pct] = coverageAt(date); return !count ? 'No charts in archive' : pct >= 95 ? 'Full lower-48 coverage' : `${Math.round(pct)}% of the lower 48 covered`; }
function paintHeat() {
  if (!manifest) return;
  const c = $('heatCanvas'), ratio = devicePixelRatio || 1, width = c.clientWidth; c.width = width * ratio; c.height = 10 * ratio;
  const ctx = c.getContext('2d'); ctx.scale(ratio, ratio);
  const min = day(manifest.dateBounds.min), span = day(manifest.dateBounds.max) - min || 1;
  for (const [a, b, count, pct] of manifest.coverage?.segments || []) { ctx.fillStyle = count ? `rgba(0,212,170,${.15 + pct / 120})` : '#39424b'; ctx.fillRect((day(a) - min) / span * width, 0, (day(b) - day(a)) / span * width, 10); }
}
function setLoading() {
  if (!dp || failed) return;
  const busy = dp.stats().inflight + dp.stats().queued > 0;
  if (busy && !loading) { loading = true; loadingTimer = setTimeout(() => { if (loading && !failed) $('loader').hidden = false; }, 100); }
  if (!busy) { loading = false; clearTimeout(loadingTimer); $('loader').hidden = true; }
}
function drawable(key) {
  const parts = key.split('/'), y = +parts.pop(), x = +parts.pop(), z = +parts.pop(), path = parts.join('/');
  for (let level = z; level >= 0; level--) { const factor = 2 ** (z - level); if (r.hasTexture(`${path}/${level}/${Math.floor(x / factor)}/${Math.floor(y / factor)}`)) return true; }
  return false;
}
function withoutAbsent(tiles) { return tiles.map(tile => ({ ...tile, items: tile.items.filter(item => !skipped(item.key)) })); }
function renderPlans() { r.setChartPlan({ ...plan, tiles: withoutAbsent(plan.tiles) }); r.setBasemapPlan(withoutAbsent(dp.planBasemap(r.visibleTiles(512)))); }
function notePaint() {
  if (painted || !plan?.tiles.some(t => t.items.some(i => !skipped(i.key)) && t.items.every(i => skipped(i.key) || drawable(i.key)))) return;
  // C7 has no paint event. Observe texture residency, then allow the scheduled renderer frame to finish.
  requestAnimationFrame(() => requestAnimationFrame(() => { if (painted) return; painted = true; document.body.classList.add('painted'); performance.mark('first-chart-paint'); window.dispatchEvent(new Event('first-chart-paint')); }));
}
function uniforms(state) {
  r.setChartStyle({ opacity: state.layers.charts.opacity, hidden: !state.layers.charts.on });
  r.setAirfieldFilter({ year: +state.date.slice(0, 4), statusMask: state.layers.airfields.on ? ['open', 'gone', 'unknown'].reduce((mask, k, i) => mask | (state.layers.airfields.status[k] ? 1 << i : 0), 0) : 0 });
  const d = day(state.date); r.setAirspaceFilter({ day: d, classMask: state.layers.airspace.on ? (state.layers.airspace.classes.bcd ? 1 : 0) | (state.layers.airspace.classes.e ? 2 : 0) : 0, regionMask: dp.airspaceRegionMask(d) }); r.setPin(state.pin);
  $('asStatus').textContent = !manifest.hasAirspace ? 'Airspace unavailable' : !state.layers.airspace.on ? 'Airspace off' : dp.airspaceRegionMask(d) ? 'Airspace coverage available at this date' : 'No airspace coverage at this date';
  if (dp.airspaceStatus) { const nw = r.unproject({ x: 0, y: 0 }), se = r.unproject({ x: $('mapCanvas').clientWidth, y: $('mapCanvas').clientHeight }); $('asStatus').textContent = dp.airspaceStatus(d, [nw.lng, se.lat, se.lng, nw.lat], { configured: manifest.hasAirspace, enabled: state.layers.airspace.on }).join(' · '); }
}
function demand(state) {
  const current = r.getCamera();
  if (current.x !== state.camera.x || current.y !== state.camera.y || current.zoom !== state.camera.zoom) r.setCamera(state.camera);
  chartTiles = r.visibleTiles(256); const basemapTiles = r.visibleTiles(512);
  dp.setDemand({ date: state.date, chartTiles, basemapTiles, airspaceTiles: state.layers.airspace.on ? chartTiles : [], center: { x: state.camera.x, y: state.camera.y }, scrub: { direction: scrubDirection, velocity: state.playing ? .5 : 0, playing: state.playing }, solo: state.solo });
  plan = dp.planCharts(state.date, chartTiles, { solo: state.solo });
  clearInterval(statsTimer);
  let polls = 0;
  statsTimer = setInterval(() => { setLoading(); notePaint(); if (failed || (++polls > 10 && dp.stats().inflight + dp.stats().queued === 0 && dp.readiness(state.date, chartTiles) >= 1)) clearInterval(statsTimer); }, 50);
  if (state.solo?.clip) r.setClipRing(state.solo.clip.id, state.solo.clip.ring);
  renderPlans(); setLoading(); notePaint();
  $('chartInfoEffective').textContent = new Set(plan.tiles.flatMap(t => t.items.map(i => i.key.split('/').slice(0, -3).join('/')))).size;
}
function layerPatch(section, patch) { const state = store.get(); store.set({ layers: { ...state.layers, [section]: { ...state.layers[section], ...patch } } }); }
async function loadFields() {
  if (!fields) { fieldsPromise ||= dp.loadAirfields().catch(error => { fieldsPromise = null; throw error; }); fields = await fieldsPromise; } r.setAirfields(fields); await updateBrowser();
}
function visibleField(index, state) {
  const mask = state.layers.airfields.status[['open', 'gone', 'unknown'][fields.status[index]]], year = +state.date.slice(0, 4);
  if (!mask) return false; if (fields.start[index] && year < fields.start[index]) return false; if (fields.status[index] !== 0 && fields.end[index] && year > fields.end[index]) return false;
  const p = r.project(unproject(fields.mx[index] * 256, fields.my[index] * 256, 0)); return p.x >= 0 && p.y >= 0 && p.x <= $('mapCanvas').clientWidth && p.y <= $('mapCanvas').clientHeight;
}
async function updateBrowser() {
  if (!store || !$('afBrowser').open) return;
  const revision = ++browserRevision, state = store.get(), previous = $('afSelect').value;
  if (!state.layers.airfields.on || !fields) { $('afSelect').disabled = $('afOpen').disabled = true; $('afCount').textContent = state.layers.airfields.on ? 'Loading airfields…' : 'Turn on Airfields to browse.'; return; }
  const entries = await Promise.all(Array.from(fields.mx, (_, index) => index).filter(i => visibleField(i, state)).map(async index => ({ index, detail: await dp.airfieldDetails(index) })));
  if (revision !== browserRevision) return;
  fieldEntries = entries.sort((a, b) => a.detail.name.localeCompare(b.detail.name)); $('afSelect').replaceChildren();
  for (const { index, detail } of fieldEntries) { const option = document.createElement('option'); option.value = index; option.textContent = `${detail.name}${detail.state ? ' — ' + detail.state : ''}`; $('afSelect').append(option); }
  if (fieldEntries.some(e => String(e.index) === previous)) $('afSelect').value = previous;
  $('afSelect').disabled = $('afOpen').disabled = !entries.length; $('afCount').textContent = entries.length ? `${entries.length} airfields in view.` : 'No airfields match this view, date, and filters.';
}
function fieldSpan(p) { if (p.status === 'open') return p.start_year ? `In operation ${p.start_year} – present` : 'Open today'; if (p.start_year && p.end_year) return `In operation ~${p.start_year} – ${p.end_year}`; if (p.start_year && p.last_known_year) return `In operation ~${p.start_year} – at least ${p.last_known_year}`; if (p.start_year) return `In operation from ~${p.start_year}`; if (p.end_year) return `In operation until ~${p.end_year}`; return p.last_known_year ? `Still listed in ${p.last_known_year}; closure date unknown` : 'Operating dates unknown'; }
async function openField(index) {
  try {
    const p = await dp.airfieldDetails(index); closePanel('toolsPanel', 'toolsBtn'); $('afName').textContent = p.name; $('afDates').textContent = fieldSpan(p); $('afLoc').textContent = p.rel_location || p.state || ''; $('afLinks').replaceChildren();
    for (const [label, href] of [['Airfields-Freeman', p.url], ['OurAirports', p.oa && `https://ourairports.com/airports/${encodeURIComponent(p.oa)}/`]]) if (href) { const url = new URL(href, location.href); if (!['http:', 'https:'].includes(url.protocol)) continue; const a = document.createElement('a'); a.textContent = label; a.href = url; a.target = '_blank'; a.rel = 'noopener'; $('afLinks').append(a, document.createTextNode(' ')); }
    $('afPanel').hidden = false; $('afPanel').focus(); store.set({ panels: { ...store.get().panels, airfield: index } });
  } catch { toast('Airfield details unavailable'); }
}
function closeField() { if ($('afPanel').contains(document.activeElement)) $('toolsBtn').focus(); $('afPanel').hidden = true; store.set({ panels: { ...store.get().panels, airfield: null } }); }
function closePin() { if ($('pinPanel').contains(document.activeElement)) $('map').focus(); store.set({ pin: null }); }
async function inspect(pin, focus = false) {
  store.set({ pin }); $('pinPanel').hidden = false; if (focus) $('pinPanel').focus();
}
async function refreshPin(state) {
  const revision = ++pinRevision;
  if (!state.pin) { $('pinPanel').hidden = true; return; }
  $('pinPanel').hidden = false; $('pinLoc').textContent = 'Loading chart details…'; $('pinActions').replaceChildren();
  try {
    const [results, spaces] = await Promise.all([dp.queryPin(state.pin.lng, state.pin.lat, state.date), state.layers.airspace.on ? dp.queryAirspace(state.pin.lng, state.pin.lat, day(state.date)) : []]);
    if (revision !== pinRevision) return;
    const result = results[0]; $('pinLoc').textContent = result ? result.location.name : 'No chart here'; $('pinBadge').textContent = result ? (/^\d+$/.test(result.chart.ed) && +result.chart.ed < 150 ? `ed. ${result.chart.ed}` : result.chart.ed && !/^\d+$/.test(result.chart.ed) && result.chart.ed !== 'Unknown' ? result.chart.ed : 'scan') : ''; $('pinDates').textContent = result ? `${result.chart.d} – ${result.chart.e || ''}${result.contains ? '' : ' · nearby'}` : `No charts in effect on ${state.date}`;
    $('pinAirspace').hidden = !spaces.length; $('pinAirspace').textContent = spaces.filter(s => (s.cls === 'E' || s.class === 'E') ? state.layers.airspace.classes.e : state.layers.airspace.classes.bcd).map(s => `${s.shortName || s.name || ''} · ${s.badge || s.class || s.cls || ''}${s.altSpan ? ' · ' + s.altSpan : ''}${s.cycle ? ' · ' + s.cycle : ''}`).join('; ');
    if (result && !result.chart.pm) $('pinActions').textContent = 'not yet tiled';
    if (result?.chart.pm) { const button = document.createElement('button'); button.textContent = 'View alone'; button.onclick = () => { const chart = result.chart; store.set({ solo: { paths: Array.isArray(chart.pm) ? chart.pm : [chart.pm], zoom: chart.pmz, ...(result.location.ringClip && { clip: { id: result.location.ref || result.location.name, ring: result.location.ringClip } }) } }); }; $('pinActions').append(button); }
  } catch { if (revision === pinRevision) $('pinLoc').textContent = 'Chart details unavailable. Inspect again to retry.'; }
}
function syncUI(state, previous) {
  const index = frameIndex(state.date), min = day(manifest.dateBounds.min), span = day(manifest.dateBounds.max) - min || 1;
  $('timeSelect').value = state.date; $('trackWrapper').setAttribute('aria-valuenow', index); const dateText = new Date(`${state.date}T12:00:00Z`).toLocaleDateString('en-US', { timeZone: 'UTC', year: 'numeric', month: 'long', day: 'numeric' }); $('trackWrapper').setAttribute('aria-valuetext', `${dateText}. ${coverageText(state.date)}`);
  $('handle').style.transform = `translateX(${(day(state.date) - min) / span * ($('trackWrapper').clientWidth - 10)}px)`;
  $('lblRange').textContent = coverageText(state.date); $('timelineStatus').textContent = state.playing ? '' : `${dateText}. ${coverageText(state.date)}`;
  $('playBtn').textContent = state.playing ? 'Ⅱ' : '▶'; $('playBtn').setAttribute('aria-label', state.playing ? 'Pause timeline' : 'Play timeline'); $('playBtn').setAttribute('aria-pressed', state.playing);
  $('soloBar').hidden = !state.solo; $('soloLabel').textContent = 'Viewing one chart'; $('chartInfoZoom').textContent = +state.camera.zoom.toFixed(2);
  for (const [id, value] of [['chartsToggleBtn', state.layers.charts.on], ['airfieldsBtn', state.layers.airfields.on], ['airspaceBtn', state.layers.airspace.on]]) { $(id).setAttribute('aria-pressed', value); $(id).setAttribute('aria-checked', value); }
  $('toolOpacitySlider').value = state.layers.charts.opacity * 100; $('toolOpacityValue').textContent = Math.round(state.layers.charts.opacity * 100);
  for (const chip of document.querySelectorAll('[data-af-status]')) chip.setAttribute('aria-pressed', state.layers.airfields.status[chip.dataset.afStatus]);
  for (const chip of document.querySelectorAll('[data-as-key]')) chip.setAttribute('aria-pressed', state.layers.airspace.classes[chip.dataset.asKey]);
  syncPlayback(state, previous);
  if (!previous || state.date !== previous.date || state.camera !== previous.camera || state.solo !== previous.solo || state.layers.airspace.on !== previous.layers.airspace.on || state.layers.airfields.on !== previous.layers.airfields.on || state.layers.charts.on !== previous.layers.charts.on) demand(state);
  uniforms(state);
  if (!previous || state.pin !== previous.pin || state.date !== previous.date || state.layers.airspace !== previous.layers.airspace) refreshPin(state);
  if (!previous || state.layers.airfields !== previous.layers.airfields || state.camera !== previous.camera || state.date !== previous.date) {
    if (state.layers.airfields.on && !fields) loadFields().catch(() => toast('Airfield data unavailable')); updateBrowser();
  }
  if (previous?.solo?.clip && !state.solo) r.setClipRing(previous.solo.clip.id, null);
  // Hidden airfields leave the GPU entirely; a status mask of zero would still draw every instance per world copy.
  if (previous && fields && state.layers.airfields.on !== previous.layers.airfields.on) r.setAirfields(state.layers.airfields.on ? fields : null);
  storage.set('airfieldsShown', state.layers.airfields.on ? '1' : '0'); storage.set('airfieldsStatus', JSON.stringify(state.layers.airfields.status)); storage.set('airspaceShown', state.layers.airspace.on ? '1' : '0');
  if (!previous || state.date !== previous.date || state.camera !== previous.camera || state.pin !== previous.pin) writeUrl();
}
function installControls() {
  $('prevBtn').onclick = () => manualStep(-1); $('nextBtn').onclick = () => manualStep(1); $('playBtn').onclick = () => store.set({ playing: !store.get().playing });
  $('timeSelect').onchange = () => selectDate($('timeSelect').value);
  $('chartsToggleBtn').onclick = () => layerPatch('charts', { on: !store.get().layers.charts.on });
  $('toolOpacitySlider').oninput = e => layerPatch('charts', { on: true, opacity: +e.target.value / 100 });
  $('airfieldsBtn').onclick = () => { const on = !store.get().layers.airfields.on; layerPatch('airfields', { on }); if (!on) closeField(); };
  $('airspaceBtn').onclick = () => layerPatch('airspace', { on: !store.get().layers.airspace.on });
  for (const chip of document.querySelectorAll('[data-af-status]')) chip.onclick = () => { const status = store.get().layers.airfields.status, key = chip.dataset.afStatus; layerPatch('airfields', { status: { ...status, [key]: !status[key] } }); };
  for (const chip of document.querySelectorAll('[data-as-key]')) chip.onclick = () => { const classes = store.get().layers.airspace.classes, key = chip.dataset.asKey; layerPatch('airspace', { classes: { ...classes, [key]: !classes[key] } }); };
  $('afBrowser').ontoggle = () => updateBrowser(); $('afOpen').onclick = () => openField(+$('afSelect').value); $('afClose').onclick = closeField; $('pinClose').onclick = closePin; $('soloExit').onclick = () => { store.set({ solo: null }); $('map').focus(); };
  const track = $('trackWrapper'); let dragging = false, pending, lastX;
  const dateAt = x => { const rect = track.getBoundingClientRect(); return new Date((day(manifest.dateBounds.min) + clamp((x - rect.left) / rect.width, 0, 1) * (day(manifest.dateBounds.max) - day(manifest.dateBounds.min))) * 86400000).toISOString().slice(0, 10); };
  const schedule = x => { lastX = x; if (!pending) pending = requestAnimationFrame(() => { pending = null; if (dragging) { const date = dateAt(lastX); selectDate(frames[frameIndex(date)]); } }); };
  track.onpointerdown = e => { if (e.button !== 0) return; dragging = true; track.focus(); track.setPointerCapture(e.pointerId); schedule(e.clientX); };
  track.onpointermove = e => { if (dragging) schedule(e.clientX); const date = dateAt(e.clientX), rect = track.getBoundingClientRect(); $('trackTip').textContent = `${date} · ${coverageText(date)}`; $('trackTip').style.left = `${clamp(e.clientX - rect.left, 0, rect.width)}px`; $('trackTip').hidden = false; };
  const cancel = () => { dragging = false; cancelAnimationFrame(pending); pending = null; };
  track.onpointerup = e => { if (dragging) { const date = dateAt(e.clientX); selectDate(frames[frameIndex(date)]); } cancel(); }; track.onpointercancel = cancel; window.addEventListener('touchcancel', cancel); track.onpointerleave = () => $('trackTip').hidden = true;
  $('trackTip').hidden = true;
  track.onkeydown = e => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    let index = frameIndex(store.get().date), page = Math.max(1, Math.round(frames.length / 100));
    if (['ArrowRight', 'ArrowUp'].includes(e.key)) index++; else if (['ArrowLeft', 'ArrowDown'].includes(e.key)) index--; else if (e.key === 'PageUp') index += page; else if (e.key === 'PageDown') index -= page; else if (e.key === 'Home') index = 0; else if (e.key === 'End') index = frames.length - 1; else return;
    e.preventDefault(); e.stopPropagation(); selectDate(frames[clamp(index, 0, frames.length - 1)]);
  };
  $('map').onkeydown = e => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const camera = store.get().camera, n = 256 * 2 ** camera.zoom;
    const delta = { ArrowLeft: [-80, 0], ArrowRight: [80, 0], ArrowUp: [0, -80], ArrowDown: [0, 80] }[e.key];
    if (delta) { e.preventDefault(); r.setCamera({ ...camera, x: camera.x + delta[0] / n, y: camera.y + delta[1] / n }); }
    else if (['+', '=', '-'].includes(e.key)) { e.preventDefault(); r.setCamera({ ...camera, zoom: camera.zoom + (e.key === '-' ? -1 : 1) }); }
    else if (e.key === 'Enter') { e.preventDefault(); inspect(unproject(camera.x * 256, camera.y * 256, 0), true); }
  };
}
document.addEventListener('keydown', e => {
  if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
  if (e.key === 'Escape') {
    if (!$('shortcutsOverlay').hidden) hideHelp(); else if (!$('siteMenu').hidden) closePanel('siteMenu', 'menuToggle'); else if (!$('toolsPanel').hidden) closePanel('toolsPanel', 'toolsBtn'); else if (!$('afPanel').hidden) closeField(); else if (store?.get().solo) store.set({ solo: null }); else if (store?.get().pin) closePin(); else return;
    e.preventDefault(); return;
  }
  if (!store || !['map', 'trackWrapper', 'main-content'].includes(e.target.id)) return;
  if (e.key === ' ') { e.preventDefault(); store.set({ playing: !store.get().playing }); } else if (e.key.toLowerCase() === 'f') { e.preventDefault(); fullscreen(); } else if (e.key.toLowerCase() === 's') { e.preventDefault(); share(); } else if (e.key === '?') { e.preventDefault(); showHelp(); }
});
async function boot() {
  dp = await createDataPlane({ manifestUrl: __MANIFEST_URL__, earlyFetches: window.__earlyFetches || new Map() }); manifest = dp.manifest; frames = manifest.frames;
  const params = new URLSearchParams(location.search), lat = number(params.get('lat')), lng = number(params.get('lng')), zoom = number(params.get('zoom'));
  $('mapCanvas').tabIndex = -1; $('mapCanvas').addEventListener('focus', () => $('map').focus({ preventScroll: true }));
  // Touch devices keep the renderer's smaller default budget; 96 MiB fits three native items per tile on a 1440×900 desktop.
  r = createRenderer($('mapCanvas'), { minZoom: 4, maxZoom: 14, maxTextureBytes: matchMedia('(pointer: coarse)').matches ? undefined : 96 * 1024 * 1024 });
  if (lat !== null && lng !== null && Math.abs(lat) <= 90) { const [x, y] = project(lat, lng, 0); r.setCamera({ x: x / 256, y: y / 256, zoom: zoom ?? 10 }); } else r.fitBounds([-124.8, 24.4, -67.1, 49.4], { padding: 70, maxZoom: 6 });
  const requested = params.get('date'), date = requested && Number.isFinite(Date.parse(requested)) ? requested < manifest.dateBounds.min ? manifest.dateBounds.min : requested > manifest.dateBounds.max ? manifest.dateBounds.max : new Date(requested).toISOString().slice(0, 10) : manifest.dateBounds.max;
  const pinParts = (params.get('pin') || '').split(','), pinLat = number(pinParts[0] ?? null), pinLng = number(pinParts[1] ?? null), pin = pinLat !== null && pinLng !== null && Math.abs(pinLat) <= 90 ? { lat: pinLat, lng: pinLng } : null;
  let saved = {}; try { saved = JSON.parse(storage.get('airfieldsStatus') || '{}'); } catch {}
  const status = Object.fromEntries(['open', 'gone', 'unknown'].map(k => [k, typeof saved?.[k] === 'boolean' ? saved[k] : true]));
  store = createStore({ date, camera: r.getCamera(), pin, solo: null, layers: { charts: { on: true, opacity: 1 }, airfields: { on: manifest.hasAirfields && storage.get('airfieldsShown') === '1', status }, airspace: { on: manifest.hasAirspace && storage.get('airspaceShown') === '1', classes: { bcd: true, e: true } } }, playing: false, panels: { siteMenu: !$('siteMenu').hidden, toolsPanel: !$('toolsPanel').hidden, help: !$('shortcutsOverlay').hidden, airfield: null } });
  $('timeSelect').min = manifest.dateBounds.min; $('timeSelect').max = manifest.dateBounds.max; $('trackWrapper').setAttribute('aria-valuemax', frames.length - 1); $('chartInfoTotal').textContent = manifest.eraCount;
  $('airfieldsBtn').disabled = !manifest.hasAirfields; $('airspaceBtn').disabled = !manifest.hasAirspace;
  const min = day(manifest.dateBounds.min), span = day(manifest.dateBounds.max) - min || 1;
  for (let year = Math.ceil(+manifest.dateBounds.min.slice(0, 4) / 10) * 10; year <= +manifest.dateBounds.max.slice(0, 4); year += 10) { const tick = document.createElement('span'); tick.className = 'tick'; tick.textContent = year; tick.style.left = `${(day(`${year}-01-01`) - min) / span * 100}%`; $('ticksContainer').append(tick); }
  dp.on('tile', ({ key, bitmap }) => { try { r.upload(key, bitmap); setLoading(); notePaint(); } catch (error) { bitmap.close(); degrade(key, error); } });
  dp.on('absent', ({ key }) => { absentKeys.add(key); renderPlans(); setLoading(); notePaint(); }); dp.on('airspace', ({ tileId, batch }) => r.setAirspaceTile(tileId, batch)); r.on('evict', ({ key }) => dp.markEvicted(key));
  dp.on('error', ({ key, error }) => key === 'worker' ? fatal(error) : degrade(key, error));
  dp.on?.('metadata', () => { if (store) uniforms(store.get()); });
  r.on('move', () => { const next = r.getCamera(), old = store.get().camera; if (next.x !== old.x || next.y !== old.y || next.zoom !== old.zoom) store.set({ camera: next }); });
  r.on('click', e => e.picked?.kind === 'airfield' ? openField(e.picked.index) : inspect({ lat: e.lat, lng: e.lng }));
  r.on('contextlost', () => toast('Graphics interrupted. Restoring charts…'));
  // The renderer reports every lost texture evicted; re-demanding fetches them again.
  r.on('contextrestored', () => demand(store.get()));
  installControls(); store.subscribe(syncUI); syncUI(store.get()); paintHeat();
  const observer = new ResizeObserver(() => { r.resize(); demand(store.get()); paintHeat(); syncUI(store.get(), store.get()); }); observer.observe($('map'));
  // A page parked in the back/forward cache (persisted) comes back alive; only a real unload tears the viewer down.
  let torn = false;
  window.addEventListener('pagehide', e => { if (e.persisted || torn) return; torn = true; dp.destroy(); r.destroy(); clearTimeout(playingTimer); clearInterval(statsTimer); observer.disconnect(); });
  if ('serviceWorker' in navigator && !__STUBS__) navigator.serviceWorker.register(new URL('./sw.js', document.baseURI)).catch(() => toast('Offline shell unavailable'));
  document.body.dataset.ready = 'true';
}
boot().catch(fatal);
