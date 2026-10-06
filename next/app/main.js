// Entry point: production's initApp (src/viewer.js) on the C7 renderer and
// data plane. The page, its controls and their behaviour are the current
// viewer's; only the map engine underneath is new.
import { createRenderer, RendererUnsupportedError } from '@next/renderer';
import { createDataPlane } from '@next/dataplane';
import { Utils } from './utils.js';
import { MapController } from './engine.js';
import { TimelineApp } from './timeline.js';

const MIN_ZOOM = 4, MAX_ZOOM = 14;
// First view without ?lat/lng: the lower 48, fitted to the viewport. The
// inline boot script (shell/early.js) fits the same box with the same padding
// to request the first tiles before this module has loaded: keep them equal.
const INITIAL_BOUNDS = [[24.4, -124.8], [49.4, -67.1]];
const INITIAL_PADDING = 70;
const $ = (id) => document.getElementById(id);

/** MENU **/
const menuToggle = $('menuToggle');
const menuPanel = $('siteMenu');
const menuTrigger = document.querySelector('.menu-trigger');

if (menuToggle && menuPanel && menuTrigger) {
  const closeMenu = () => {
    // The panel goes visibility:hidden when closed, so focus sitting on a link
    // inside would be dropped to <body>. Only pull it back when it is actually
    // in there — outside-click and link-click callers must not steal focus.
    if (menuPanel.contains(document.activeElement)) menuToggle.focus();
    menuTrigger.classList.remove('open');
    menuToggle.setAttribute('aria-expanded', 'false');
  };

  menuToggle.addEventListener('click', (event) => {
    event.stopPropagation();
    const toolsControl = $('toolsControl');
    if (toolsControl?.classList.contains('open')) {
      toolsControl.classList.remove('open');
      $('toolsBtn')?.setAttribute('aria-expanded', 'false');
    }
    const isOpen = menuTrigger.classList.toggle('open');
    menuToggle.setAttribute('aria-expanded', String(isOpen));
  });

  menuPanel.querySelectorAll('a').forEach((item) => {
    item.addEventListener('click', () => {
      closeMenu();
    });
  });

  document.addEventListener('click', (event) => {
    if (!menuTrigger.contains(event.target)) {
      closeMenu();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (!event.defaultPrevented && event.key === 'Escape' && menuTrigger.classList.contains('open')) {
      event.preventDefault();
      closeMenu();
    }
  });
}

/** LOADING SPLASH **/
const loadingSplash = $('loadingSplash');
const loadingProgressBar = $('loadingProgressBar');
const loadingStatus = $('loadingStatus');

const updateLoading = (percent, status) => {
  loadingProgressBar.style.width = `${percent}%`;
  loadingStatus.textContent = status;
};

// Fade the splash out, then take it out of layout. The fade is a CSS
// transition; a stalled one (throttled tab, cold mobile WebKit) would
// otherwise leave an overlay that is hidden by class yet still rendered.
let splashFailed = false;
const hideSplash = () => {
  loadingSplash.classList.add('hidden');
  const finish = () => { if (!splashFailed) loadingSplash.hidden = true; };
  loadingSplash.addEventListener('transitionend', finish, { once: true });
  setTimeout(finish, 700);
};

// A failure the viewer cannot recover from. The splash is the error surface,
// as it is for a failed boot of the current viewer: it comes back (or stays)
// with the reason, and the loading pill goes away.
const fail = (message, link) => {
  splashFailed = true;
  loadingSplash.hidden = false;
  loadingSplash.classList.remove('hidden');
  loadingSplash.classList.add('failed');
  loadingStatus.textContent = message;
  if (link) {
    const a = document.createElement('a');
    a.href = link.href;
    a.textContent = link.text;
    loadingStatus.append(' ', a);
  }
  $('loader')?.classList.remove('active');
};

// Map credits, bottom right: only for what this build actually draws.
const renderAttribution = (el, { basemap, airfields, airspace }) => {
  if (!el) return;
  const link = (text, href) => {
    const a = document.createElement('a');
    a.href = href;
    a.textContent = text;
    a.target = '_blank';
    a.rel = 'noopener';
    return a;
  };
  const nodes = [];
  if (basemap) {
    nodes.push('© ', link('OpenStreetMap', 'https://www.openstreetmap.org/copyright'), ' contributors · ',
      link('Protomaps', 'https://protomaps.com'));
  }
  if (airfields) {
    if (nodes.length) nodes.push(' · ');
    nodes.push('airfields ', link('Freeman', 'https://www.airfields-freeman.com/'));
  }
  if (airspace.length) {
    if (nodes.length) nodes.push(', ');
    nodes.push('airspace ');
    airspace.forEach(({ source, url }, i) => {
      if (i) nodes.push(', ');
      nodes.push(/^https?:\/\//.test(url || '') ? link(source, url) : source);
    });
  }
  el.replaceChildren(...nodes);
  el.hidden = !nodes.length;
};

async function initApp() {
  updateLoading(10, 'Loading timeline data...');
  // The pin, airfield and airspace cards are not needed to paint the first
  // chart: their chunk downloads beside the manifest and the first tiles.
  const cardsLoading = import('./cards.js');
  cardsLoading.catch(() => {}); // reported where it is awaited

  // Show warning banner temporarily
  const warningOverlay = $('warningOverlay');
  setTimeout(() => warningOverlay.classList.add('visible'), 1500);
  setTimeout(() => warningOverlay.classList.remove('visible'), 8000);

  const urlParams = new URLSearchParams(window.location.search);
  const dateParam = urlParams.get('date');
  const latParam = urlParams.get('lat');
  const lngParam = urlParams.get('lng');
  const zoomParam = urlParams.get('zoom');

  // The manifest and the first tiles were requested by the inline boot
  // script; the data plane adopts those responses instead of fetching again.
  let dp;
  try {
    dp = await createDataPlane({ manifestUrl: __MANIFEST_URL__, earlyFetches: window.__earlyFetches || new Map() });
  } catch (error) {
    console.error('Initialization failed', error);
    fail('Failed to load chart data. Check your connection and reload.');
    return;
  }
  const manifest = dp.manifest;
  updateLoading(40, 'Initializing map...');

  if (manifest.frames.length === 0) {
    dp.destroy();
    fail('Failed to load chart data. Check your connection and reload.');
    return;
  }

  // Keyboard focus belongs to the #map region, which carries the label and
  // the help text; a click on the canvas hands its focus up.
  const mapEl = $('map');
  const canvas = $('mapCanvas');
  canvas.tabIndex = -1;
  canvas.addEventListener('focus', () => mapEl.focus({ preventScroll: true }));

  let renderer;
  try {
    // The renderer stops uploading once everything the plans need is pinned
    // past its budget, and it stays stopped until demand shrinks. A scrub
    // through the 1940s-50s, where dozens of partial eras overlap, pins the
    // old date's drawn tiles and the new date's items together: ~85 MiB at
    // 2000×1200 z5, ~100 MiB at z6, ~40 MiB on a phone (2026-10-05, when a
    // 96 MiB desktop budget left half the map on its blurry fallback).
    renderer = createRenderer(canvas, {
      minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM,
      maxTextureBytes: (matchMedia('(pointer: coarse)').matches ? 96 : 256) * 1024 * 1024
    });
  } catch (error) {
    console.error('Renderer failed to start', error);
    dp.destroy();
    if (error instanceof RendererUnsupportedError) {
      fail('This browser cannot run the new chart renderer.', { href: `${__SITE_ORIGIN__}/`, text: 'Open the current viewer' });
    } else {
      fail('Failed to start the chart viewer. Reload to try again.');
    }
    return;
  }

  let timelineApp = null;
  const mapCtrl = new MapController(dp, renderer, {
    onFatal: (error) => {
      console.error('Data worker failed', error);
      if (timelineApp?.isPlaying) timelineApp.togglePlay();
      fail('The chart viewer stopped working. Reload the page to continue.');
    }
  });

  // Resolve the initial view BEFORE the first frame is requested so the
  // first tiles fetched are the ones kept. A share link's lat/lng are honored
  // even when zoom is missing or invalid; otherwise the map opens on the
  // lower 48. Nothing geolocates the visitor on load; only the locate button
  // asks where they are.
  const urlLat = parseFloat(latParam);
  const urlLng = parseFloat(lngParam);
  const urlZoom = parseFloat(zoomParam);
  if (!isNaN(urlLat) && !isNaN(urlLng) && Math.abs(urlLat) <= 90) {
    mapCtrl.map.setView([urlLat, urlLng], Number.isFinite(urlZoom) ? urlZoom : 10);
  } else {
    mapCtrl.map.fitBounds(INITIAL_BOUNDS, { padding: INITIAL_PADDING, maxZoom: 6 });
  }

  // Panels drop their backdrop blur while the map zooms (styles.css keys on
  // body.zooming), as they do over Leaflet's zoom animation.
  let zoomingTimer = null;
  mapCtrl.map.on('zoom', () => {
    clearTimeout(zoomingTimer);
    zoomingTimer = null;
    document.body.classList.add('zooming');
  });
  mapCtrl.map.on('moveend', () => {
    if (zoomingTimer || !document.body.classList.contains('zooming')) return;
    zoomingTimer = setTimeout(() => {
      document.body.classList.remove('zooming');
      zoomingTimer = null;
    }, 120);
  });

  // Zoom control: Leaflet's markup and classes, driven by the renderer.
  const zoomIn = $('zoomInBtn');
  const zoomOut = $('zoomOutBtn');
  const syncZoomButtons = () => {
    const zoom = mapCtrl.map.getZoom();
    for (const [el, off] of [[zoomIn, zoom >= MAX_ZOOM], [zoomOut, zoom <= MIN_ZOOM]]) {
      el?.classList.toggle('leaflet-disabled', off);
      el?.setAttribute('aria-disabled', String(off));
    }
  };
  zoomIn?.addEventListener('click', (e) => { e.preventDefault(); mapCtrl.map.zoomBy(1); });
  zoomOut?.addEventListener('click', (e) => { e.preventDefault(); mapCtrl.map.zoomBy(-1); });
  mapCtrl.map.on('zoomend', syncZoomButtons);
  syncZoomButtons();

  // The focused map owns arrow keys for panning and +/− for zooming.
  // Timeline stepping belongs to its own slider so both work by keyboard.
  mapEl.addEventListener('keydown', (e) => {
    if (e.target !== mapEl || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    const pan = { ArrowLeft: [-80, 0], ArrowRight: [80, 0], ArrowUp: [0, -80], ArrowDown: [0, 80] }[e.key];
    if (pan) {
      e.preventDefault();
      mapCtrl.map.panBy(...pan);
    } else if (e.key === '+' || e.key === '=') {
      e.preventDefault();
      mapCtrl.map.zoomBy(1);
    } else if (e.key === '-' || e.key === '_') {
      e.preventDefault();
      mapCtrl.map.zoomBy(-1);
    }
  });
  updateLoading(70, 'Loading controls...');

  // Clamp an out-of-range ?date= into the covered window so a stale/typo'd
  // date opens to the nearest real frame instead of a blank overlay. Resolved
  // before the timeline exists, so the first frame requested is the one shown.
  let initialDate = null;
  if (dateParam) {
    const t = new Date(dateParam).getTime();
    if (!isNaN(t)) {
      const iso = new Date(t).toISOString().slice(0, 10);
      const { min, max } = manifest.dateBounds;
      initialDate = min && iso < min ? min : max && iso > max ? max : iso;
    }
  }

  timelineApp = new TimelineApp(mapCtrl, {
    frames: manifest.frames.map((date) => ({ date, id: Utils.formatDateId(date) })),
    dateBounds: manifest.dateBounds,
    coverage: manifest.coverage,
    initialDate
  });

  // Layers-panel stats footer: era count is static; zoom and charts-in-view
  // update live (the latter from showFrame via #chartInfoEffective).
  const chartInfoTotal = $('chartInfoTotal');
  const chartInfoZoom = $('chartInfoZoom');
  if (chartInfoTotal) chartInfoTotal.textContent = manifest.eraCount.toLocaleString('en-US');

  // Update zoom level display
  const updateZoomDisplay = () => {
    if (chartInfoZoom) {
      // Trimmed of trailing zeros: "8", "8.25" — not "8.00".
      chartInfoZoom.textContent = parseFloat(mapCtrl.map.getZoom().toFixed(2)).toString();
    }
  };
  mapCtrl.map.on('zoomend', updateZoomDisplay);
  updateZoomDisplay(); // Initial value

  // Location Crosshair Button - Request User's Geolocation
  const locateBtn = $('locateBtn');
  locateBtn?.addEventListener('click', () => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const lat = position.coords.latitude;
          const lng = position.coords.longitude;
          mapCtrl.map.flyTo([lat, lng], 10);
          Utils.toast('Centered on your location (GPS)');
        },
        () => {
          console.log('GPS failed, trying IP geolocation...');
          // Fallback to IP Geolocation, capped so a stalled service doesn't
          // leave the button looking dead with no feedback.
          Promise.race([
            fetch('https://get.geojs.io/v1/ip/geo.json').then(response => response.json()),
            new Promise((_, reject) => setTimeout(() => reject(new Error('ipgeo-timeout')), 4000))
          ])
            .then(data => {
              const lat = parseFloat(data.latitude);
              const lng = parseFloat(data.longitude);
              if (isNaN(lat) || isNaN(lng)) throw new Error('bad-ipgeo');
              mapCtrl.map.flyTo([lat, lng], 10);
              Utils.toast('Centered on your approximate location (IP)');
            })
            .catch(err => {
              console.error('IP Geolocation error:', err);
              Utils.toast('Unable to determine location');
            });
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 0
        }
      );
    } else {
      Utils.toast('Geolocation not supported by your browser');
    }
  });

  // Map credits; the airspace sources join them while that layer is drawn.
  let airspaceLayer = null;
  const updateAttribution = () => renderAttribution($('mapAttribution'), {
    basemap: manifest.hasBasemap,
    airfields: manifest.hasAirfields,
    airspace: airspaceLayer?.enabled ? dp.airspaceCredits() : []
  });
  updateAttribution();

  // A page parked in the back/forward cache (persisted) comes back alive;
  // only a real unload tears the viewer down.
  let torn = false;
  window.addEventListener('pagehide', (e) => {
    if (e.persisted || torn) return;
    torn = true;
    if (timelineApp.isPlaying) timelineApp.togglePlay();
    mapCtrl.destroy();
    dp.destroy();
    renderer.destroy();
  });

  // The map and the timeline are live and the first frame is requested:
  // hide the loading splash. Production holds "Ready!" for 300 ms; nothing is
  // still loading behind it here, so the map is shown at once.
  updateLoading(100, 'Ready!');
  hideSplash();

  let cards;
  try {
    cards = await cardsLoading;
  } catch (error) {
    console.error('Viewer cards failed to load', error);
    Utils.toast('Part of the viewer failed to load. Reload to retry.', 6000);
    document.body.dataset.ready = 'true';
    return;
  }
  if (torn) return;
  const { PinInspector, AirfieldsLayer, AirspaceLayer } = cards;

  const pinInspector = new PinInspector(mapCtrl, timelineApp);
  // Exposed so updateShareUrl can encode the pin.
  timelineApp.pinInspector = pinInspector;

  const airfieldsLayer = new AirfieldsLayer(mapCtrl, timelineApp);
  timelineApp.airfieldsLayer = airfieldsLayer;
  const airfieldsBtn = $('airfieldsBtn');
  const syncAirfieldsBtn = () => {
    airfieldsBtn?.setAttribute('aria-pressed', String(airfieldsLayer.enabled));
  };
  airfieldsBtn?.addEventListener('click', () => {
    airfieldsLayer.setEnabled(!airfieldsLayer.enabled);
    syncAirfieldsBtn();
  });
  syncAirfieldsBtn();
  const afChips = document.querySelectorAll('#afFilterRow .af-chip');
  const syncAfChips = () => {
    afChips.forEach((chip) => {
      const k = chip.dataset.afStatus;
      chip.setAttribute('aria-pressed', String(!!airfieldsLayer.statusFilter[k]));
    });
  };
  afChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const k = chip.dataset.afStatus;
      airfieldsLayer.setStatusFilter(k, !airfieldsLayer.statusFilter[k]);
      syncAfChips();
    });
  });
  syncAfChips();

  airspaceLayer = new AirspaceLayer(mapCtrl, timelineApp, { available: manifest.hasAirspace });
  timelineApp.airspaceLayer = airspaceLayer;
  const airspaceBtn = $('airspaceBtn');
  const syncAirspaceBtn = () => {
    airspaceBtn?.setAttribute('aria-pressed', String(airspaceLayer.enabled));
  };
  airspaceBtn?.addEventListener('click', () => {
    airspaceLayer.setEnabled(!airspaceLayer.enabled);
    syncAirspaceBtn();
  });
  syncAirspaceBtn();
  const asChips = document.querySelectorAll('#asFilterRow .af-chip');
  const syncAsChips = () => {
    asChips.forEach((chip) => {
      chip.setAttribute('aria-pressed', String(!!airspaceLayer.filter[chip.dataset.asKey]));
    });
  };
  asChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const k = chip.dataset.asKey;
      airspaceLayer.setFilter(k, !airspaceLayer.filter[k]);
      syncAsChips();
    });
  });
  syncAsChips();

  airspaceLayer.onStateChange = updateAttribution;
  updateAttribution();

  // Restore a shared pin (the one case where the chart inventory loads at
  // boot rather than on a click).
  const pinParam = urlParams.get('pin');
  if (pinParam) {
    const [pinLat, pinLng] = pinParam.split(',').map(parseFloat);
    if (!isNaN(pinLat) && !isNaN(pinLng) && Math.abs(pinLat) <= 90) {
      pinInspector.setPin({ lat: pinLat, lng: pinLng });
    }
  }

  // Follow frame changes: pin list and airfield dots track the timeline.
  const originalUpdate = timelineApp.update.bind(timelineApp);
  timelineApp.update = function (index, lazy, selectedDateOverride) {
    originalUpdate(index, lazy, selectedDateOverride);
    const displayDate = this.selectedDate || this.frames[index]?.date;
    // A pinned list ranks charts in effect on the selected date, so it
    // follows the timeline (debounced for scrubbing).
    pinInspector.refreshDebounced();
    // Airfield dots follow the same timeline (debounced internally).
    if (displayDate) airfieldsLayer.onDateChanged(displayDate);
    // Airspace repaints its resident tiles for the new date.
    if (displayDate) airspaceLayer.onDateChanged(displayDate);
  };
  const initialDisplayDate = timelineApp.selectedDate || timelineApp.frames[timelineApp.currentIndex]?.date;
  airfieldsLayer.init(initialDisplayDate);
  airspaceLayer.init(initialDisplayDate);

  // The offline shell is a convenience; a browser that refuses it still works.
  if ('serviceWorker' in navigator && !__STUBS__) {
    navigator.serviceWorker.register(new URL('./sw.js', document.baseURI))
      .catch((error) => console.warn('service worker unavailable', error));
  }

  document.body.dataset.ready = 'true';
}

initApp().catch((error) => {
  console.error('Initialization failed', error);
  fail('Failed to load chart data. Check your connection and reload.');
});
