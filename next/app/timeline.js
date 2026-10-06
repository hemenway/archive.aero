// TimelineApp, ported from src/viewer.js. Same controls, keyboard model, share
// link, playback and coverage heat strip; the frames, date bounds and coverage
// segments now come from the data plane's manifest instead of dates.csv and
// coverage.json.
import { Utils } from './utils.js';

export class TimelineApp {
  constructor(mapController, { frames, dateBounds, coverage, initialDate = null }) {
    this.mapCtrl = mapController;
    this.frames = frames;
    this.dateBounds = dateBounds;
    this.coverage = coverage;
    // A share link's date. Production painted the newest frame and then this
    // one; here the first frame requested is the one shown.
    this.initialDate = initialDate;

    const dates = this.frames.map(f => new Date(f.date).getTime());
    const min = Math.min(...dates);
    const max = Math.max(...dates);
    this.frames.forEach((f, i) => { const d = dates[i]; f.pct = max === min ? 0 : ((d - min) / (max - min)) * 100; });
    this.minTime = min;
    this.maxTime = max;
    this.coverageSegs = null;

    this.currentIndex = 0;
    this.selectedDate = null;
    this.frameTimes = dates;
    this.trackWidthPx = null;
    this.isPlaying = false;
    this.playInterval = null;
    this.playbackSpeed = 1;

    this.ui = {
      prevBtn: document.getElementById('prevBtn'),
      nextBtn: document.getElementById('nextBtn'),
      playBtn: document.getElementById('playBtn'),
      heatCanvas: document.getElementById('heatCanvas'),
      trackTip: document.getElementById('trackTip'),
      handle: document.getElementById('handle'),
      track: document.getElementById('trackWrapper'),
      timeSelect: document.getElementById('timeSelect'),
      lblRange: document.getElementById('lblRange'),
      ticks: document.getElementById('ticksContainer')
    };

    this.initUI();
    this.initControls();
    this.initCoverage();
    const startDate = this.selectedDate || this.frames[this.frames.length - 1]?.date;
    if (startDate) {
      this.updateByDate(startDate);
    }

    window.addEventListener('keydown', (e) => {
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
      // Each dismissible surface owns Escape; don't also dismiss a card
      // behind the layers panel or modal.
      if (e.key === 'Escape') {
        if (document.getElementById('toolsControl')?.classList.contains('open')) {
          e.preventDefault();
          this.closeToolsPanel();
        }
        return;
      }
      // Leave native controls and panels in charge of their own keys.
      if (e.target.closest?.('input, select, textarea, [contenteditable="true"], [role="dialog"], #toolsPanel, #pinPanel, #afPanel')) return;
      if ((e.key === ' ' || e.key === 'Enter') && e.target.closest?.('button, a')) return;

      // Character shortcuts are active only on the viewer's focused
      // controls; typing elsewhere never triggers a map action.
      if (!['map', 'trackWrapper', 'main-content'].includes(e.target.id)) return;

      if (e.key === ' ') {
        e.preventDefault();
        this.togglePlay();
      }
      if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        Utils.toggleFullscreen();
      }
      if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        this.copyShareLink();
      }
      if (e.key === '?') {
        e.preventDefault();
        this.showShortcuts();
      }
    });
  }

  initControls() {
    // Help button
    document.getElementById('helpBtn')?.addEventListener('click', () => {
      this.showShortcuts();
    });

    document.getElementById('closeShortcuts')?.addEventListener('click', () => {
      this.hideShortcuts();
    });

    document.getElementById('shortcutsOverlay')?.addEventListener('click', (e) => {
      if (e.target.id === 'shortcutsOverlay') {
        this.hideShortcuts();
      }
    });

    document.getElementById('shortcutsOverlay')?.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        this.hideShortcuts();
        return;
      }
      if (e.key !== 'Tab') return;
      const dialog = e.currentTarget.querySelector('[role="dialog"]');
      const focusable = Array.from(dialog?.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      ) || []).filter(el => !el.hidden);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    });

    // Fullscreen button
    document.getElementById('fullscreenBtn')?.addEventListener('click', () => {
      Utils.toggleFullscreen();
    });

    // Tools Dropdown
    const toolsControl = document.getElementById('toolsControl');
    const toolsBtn = document.getElementById('toolsBtn');
    const toolOpacitySlider = document.getElementById('toolOpacitySlider');
    const toolOpacityValue = document.getElementById('toolOpacityValue');

    toolsBtn?.addEventListener('click', (event) => {
      event.stopPropagation();
      if (toolsControl?.classList.contains('open')) {
        this.closeToolsPanel();
      } else {
        this.openToolsPanel();
      }
    });

    toolsControl?.addEventListener('click', (event) => {
      event.stopPropagation();
    });

    document.addEventListener('click', (event) => {
      if (toolsControl && !toolsControl.contains(event.target)) {
        this.closeToolsPanel();
      }
    });

    document.addEventListener('keydown', (event) => {
      if (!event.defaultPrevented && event.key === 'Escape' && toolsControl?.classList.contains('open')) {
        event.preventDefault();
        this.closeToolsPanel();
      }
    });

    // Charts visibility switch. Dragging the opacity slider while hidden
    // flips the switch back on rather than silently fighting it.
    const chartsToggleBtn = document.getElementById('chartsToggleBtn');
    const syncChartsToggle = () => {
      chartsToggleBtn?.setAttribute('aria-pressed', String(!this.mapCtrl.chartsHidden));
    };
    chartsToggleBtn?.addEventListener('click', () => {
      this.mapCtrl.setChartsHidden(!this.mapCtrl.chartsHidden);
      syncChartsToggle();
    });

    toolOpacitySlider?.addEventListener('input', (e) => {
      const value = e.target.value;
      toolOpacityValue.textContent = value;
      if (this.mapCtrl.chartsHidden) {
        this.mapCtrl.chartsHidden = false;
        syncChartsToggle();
      }
      this.mapCtrl.setOpacity(value / 100);
    });

    document.getElementById('shareBtn')?.addEventListener('click', () => {
      this.copyShareLink();
    });

    // Play button
    this.ui.playBtn?.addEventListener('click', () => {
      this.togglePlay();
    });
  }

  hideAllPanels(except = []) {
    if (!except.includes('toolsPanel')) {
      this.closeToolsPanel();
    }
  }

  openToolsPanel() {
    const toolsControl = document.getElementById('toolsControl');
    const toolsBtn = document.getElementById('toolsBtn');
    if (!toolsControl || !toolsBtn) return;
    const menuTrigger = document.querySelector('.menu-trigger');
    const menuToggle = document.getElementById('menuToggle');
    if (menuTrigger?.classList.contains('open')) {
      menuTrigger.classList.remove('open');
      menuToggle?.setAttribute('aria-expanded', 'false');
    }
    toolsControl.classList.add('open');
    toolsBtn.setAttribute('aria-expanded', 'true');
  }

  closeToolsPanel() {
    const toolsControl = document.getElementById('toolsControl');
    const toolsBtn = document.getElementById('toolsBtn');
    if (document.getElementById('toolsPanel')?.contains(document.activeElement)) toolsBtn?.focus();
    toolsControl?.classList.remove('open');
    toolsBtn?.setAttribute('aria-expanded', 'false');
  }

  showShortcuts() {
    const overlay = document.getElementById('shortcutsOverlay');
    if (!overlay || overlay.classList.contains('visible')) return;
    this._shortcutsOpener = document.activeElement;
    if (this.isPlaying) this.togglePlay();
    this.hideAllPanels();
    overlay?.classList.add('visible');
    overlay?.setAttribute('aria-hidden', 'false');
    // Keep virtual and keyboard focus inside the modal. Save only elements
    // changed here so an existing inert state is never accidentally removed.
    this._shortcutsInerted = [];
    for (const child of document.body.children) {
      if (child === overlay || child.tagName === 'SCRIPT' || child.inert) continue;
      child.inert = true;
      this._shortcutsInerted.push(child);
    }
    // Move focus into the dialog so keyboard/AT users land on it (and can Esc).
    document.getElementById('closeShortcuts')?.focus();
  }

  hideShortcuts() {
    const overlay = document.getElementById('shortcutsOverlay');
    if (!overlay?.classList.contains('visible')) return;
    for (const child of this._shortcutsInerted || []) child.inert = false;
    this._shortcutsInerted = [];
    // Restore focus to whatever opened the dialog.
    if (this._shortcutsOpener && typeof this._shortcutsOpener.focus === 'function') {
      this._shortcutsOpener.focus();
    }
    this._shortcutsOpener = null;
    overlay.classList.remove('visible');
    overlay.setAttribute('aria-hidden', 'true');
  }

  updateShareUrl() {
    const selectedDate = this.selectedDate || this.frames[this.currentIndex]?.date;
    const center = this.mapCtrl.map.getCenter();
    // The renderer zooms continuously; share at most two decimals ("8", "8.25").
    const zoom = parseFloat(this.mapCtrl.map.getZoom().toFixed(2));
    // Longitude back into ±180 for a view panned across the antimeridian.
    const lng = ((center.lng + 180) % 360 + 360) % 360 - 180;
    let url = `${window.location.origin}${window.location.pathname}?date=${selectedDate}&lat=${center.lat.toFixed(4)}&lng=${lng.toFixed(4)}&zoom=${zoom}`;
    const pin = this.pinInspector?.latlng;
    if (pin) url += `&pin=${pin.lat.toFixed(4)},${pin.lng.toFixed(4)}`;
    return url;
  }

  async copyShareLink() {
    const url = this.updateShareUrl();
    try {
      await navigator.clipboard.writeText(url);
      Utils.toast('Link copied');
    } catch (err) {
      // Clipboard API can be unavailable (insecure context, old browsers).
      try {
        const ta = document.createElement('textarea');
        ta.value = url;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        ta.setSelectionRange(0, ta.value.length);
        const ok = document.execCommand('copy');
        ta.remove();
        if (ok) {
          Utils.toast('Link copied');
        } else {
          window.prompt('Copy this link:', url);
        }
      } catch (fallbackErr) {
        window.prompt('Copy this link:', url);
      }
    }
  }

  togglePlay() {
    this.isPlaying = !this.isPlaying;
    const playBtn = this.ui.playBtn;
    for (const id of ['pinDates', 'afCount']) {
      document.getElementById(id)?.setAttribute('aria-live', this.isPlaying ? 'off' : 'polite');
    }
    // The data plane prefetches the next frames while playing.
    this.mapCtrl.setPlaying(this.isPlaying);

    if (this.isPlaying) {
      if (document.activeElement === this.ui.track) playBtn.focus({ preventScroll: true });
      playBtn.textContent = '⏸';
      playBtn.title = 'Pause (Space)';
      playBtn.setAttribute('aria-label', 'Pause animation');
      this.playInterval = setInterval(() => {
        // Hold the frame until its tiles are in, rather than stepping past a
        // half-loaded date on a slow connection.
        if (this.mapCtrl.frameReady()) this.step(1);
      }, 2000 / this.playbackSpeed);
    } else {
      playBtn.textContent = '▶';
      playBtn.title = 'Play (Space)';
      playBtn.setAttribute('aria-label', 'Play animation');
      if (this.playInterval) {
        clearInterval(this.playInterval);
        this.playInterval = null;
      }
    }
    this.updateTimelineAccessibility();
  }

  initUI() {
    if (!this.frames.length) return;

    this.ui.lblRange.textContent = `${this.frames[0].id} — ${this.frames[this.frames.length - 1].id}`;

    // Setup date picker with min/max and default to most recent
    const firstDate = this.dateBounds.min || this.frames[0].date;
    const lastDate = this.dateBounds.max || this.frames[this.frames.length - 1].date;
    this.ui.timeSelect.min = firstDate;
    this.ui.timeSelect.max = lastDate;
    this.ui.timeSelect.value = this.initialDate || lastDate; // Default to most recent
    this.selectedDate = this.ui.timeSelect.value || lastDate;
    this.ui.track.setAttribute('aria-valuemax', String(Math.max(0, this.frames.length - 1)));

    // Start at most recent frame
    this.currentIndex = this.frames.length - 1;

    this.ui.timeSelect.addEventListener('change', () => {
      const selectedDate = this.ui.timeSelect.value;
      if (!selectedDate || !this.ui.timeSelect.validity.valid) return;
      if (this.isPlaying) this.togglePlay();
      this.updateByDate(selectedDate);
    });

    // Decade labels across the full span. Derived from the year range, not
    // the frames list — frames vanish during multi-decade gaps, which is
    // exactly where the heat strip needs labels to show how wide a gap is.
    const timeSpan = this.maxTime - this.minTime;
    if (timeSpan > 0) {
      const frag = document.createDocumentFragment();
      const minYear = new Date(this.minTime).getUTCFullYear();
      const maxYear = new Date(this.maxTime).getUTCFullYear();
      for (let y = Math.ceil(minYear / 10) * 10; y <= maxYear; y += 10) {
        const tick = document.createElement('div');
        tick.className = 'tick';
        tick.style.left = `${((Date.UTC(y, 0, 1) - this.minTime) / timeSpan) * 100}%`;
        const label = document.createElement('div');
        label.className = 'tick-label';
        label.textContent = y;
        tick.appendChild(label);
        frag.appendChild(tick);
      }
      this.ui.ticks.appendChild(frag);
    }

    // Jog Controls
    const manualStep = dir => { if (this.isPlaying) this.togglePlay(); this.step(dir); };
    this.ui.prevBtn.onclick = () => manualStep(-1);
    this.ui.nextBtn.onclick = () => manualStep(1);

    // The visible timeline is also a semantic slider. Arrow keys step one
    // edition; Page Up/Down move roughly one percent of the archive; Home
    // and End jump to the oldest/newest edition.
    this.ui.track.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      let next = this.currentIndex;
      const page = Math.max(1, Math.round(this.frames.length / 100));
      if (e.key === 'ArrowRight' || e.key === 'ArrowUp') next += 1;
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') next -= 1;
      else if (e.key === 'PageUp') next += page;
      else if (e.key === 'PageDown') next -= page;
      else if (e.key === 'Home') next = 0;
      else if (e.key === 'End') next = this.frames.length - 1;
      else return;
      e.preventDefault();
      e.stopPropagation();
      if (this.isPlaying) this.togglePlay();
      this.update(Math.max(0, Math.min(this.frames.length - 1, next)));
    });

    // drag on the track/handle
    let trackRect = null;
    let rafPending = false;
    let lastClientX = 0;

    const updateTrackRect = () => {
      trackRect = this.ui.track.getBoundingClientRect();
      this.trackWidthPx = trackRect.width;
    };

    const handleInput = (clientX) => {
      if (!trackRect) updateTrackRect();
      const x = Math.max(0, Math.min(clientX - trackRect.left, trackRect.width));
      const pct = (x / trackRect.width) * 100;
      const closestIdx = this.findClosestFrameIndexByPct(pct);

      if (closestIdx !== this.currentIndex) this.update(closestIdx, true);
    };

    let isDragging = false;
    const scheduleInput = (clientX) => {
      lastClientX = clientX;
      if (rafPending) return;
      rafPending = true;
      requestAnimationFrame(() => {
        rafPending = false;
        handleInput(lastClientX);
      });
    };

    const startDrag = (e) => {
      if (e.type === 'mousedown' && e.button !== 0) return;
      if (this.isPlaying) this.togglePlay();
      this.ui.track.focus({ preventScroll: true });
      isDragging = true;
      updateTrackRect();
      scheduleInput((e.touches?.[0]?.clientX) ?? e.clientX);
    };
    const moveDrag = (e) => {
      if (!isDragging) return;
      // Non-passive specifically so we can stop the page from scrolling while
      // the timeline handle is being dragged on touch devices.
      if (e.cancelable) e.preventDefault();
      scheduleInput((e.touches?.[0]?.clientX) ?? e.clientX);
    };
    const endDrag = () => { isDragging = false; rafPending = false; };

    this.ui.track.addEventListener('mousedown', startDrag);
    this.ui.track.addEventListener('touchstart', startDrag, { passive: false });
    window.addEventListener('mousemove', moveDrag);
    window.addEventListener('touchmove', moveDrag, { passive: false });
    window.addEventListener('mouseup', endDrag);
    window.addEventListener('touchend', endDrag);
    // iOS fires touchcancel (not touchend) when the system interrupts a drag
    // (notification banner, edge-swipe, second finger); without this, isDragging
    // sticks and the next map pan would scrub the timeline.
    window.addEventListener('touchcancel', endDrag);
    window.addEventListener('resize', () => {
      trackRect = null;
      this.trackWidthPx = null;
      // Reposition the handle for the new track width (it uses an absolute px
      // offset), else it points at the wrong date until the next interaction.
      // Geometry only: routing this through update() would repaint the frame,
      // and showFrame clears solo state, so every phone rotation or iOS
      // URL-bar collapse would silently cancel "View alone".
      this.positionHandle(this.currentIndex);
    });
  }

  /* ---- coverage heat strip ----
     The manifest carries coverage.json verbatim (build_coverage.py): for each
     time segment, the % of the charted lower-48 area covered by the charts in
     effect. Chart count is deliberately not the signal — one merged
     current-cycle mosaic covers 100% while 1961's two charts cover ~5%. */
  initCoverage() {
    const segments = this.coverage?.segments;
    if (!Array.isArray(segments) || !segments.length) {
      // No coverage data: the plain grey track stays as-is
      console.warn('coverage unavailable — heat strip disabled');
      return;
    }
    // segments: [startISO, endISO, count, pct], sorted, end-exclusive
    this.coverageSegs = segments.map(([a, b, c, p]) => [Date.parse(a), Date.parse(b), c, p]);
    // ResizeObserver rather than a resize listener: the canvas can be 0-wide
    // while the splash is still up, and the observer also repaints it when
    // the controls card changes size. Observe the wrapper, not the canvas —
    // resizing the canvas bitmap inside its own observer callback could feed
    // back into itself.
    this.paintHeatStrip();
    if (window.ResizeObserver) {
      new ResizeObserver(() => this.paintHeatStrip()).observe(this.ui.track);
    } else {
      window.addEventListener('resize', () => this.paintHeatStrip());
    }
    this.initCoverageTip();
    this.updateCoverageLabel();
  }

  // [count, pct of US covered] at time t
  coverageAt(t) {
    const segs = this.coverageSegs;
    // segments are end-exclusive: clamp so the newest date doesn't read as a gap
    t = Math.min(t, segs[segs.length - 1][1] - 1);
    let lo = 0, hi = segs.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (segs[mid][0] <= t) {
        if (t < segs[mid][1]) return [segs[mid][2], segs[mid][3]];
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return [0, 0];
  }

  coverageText(count, pct) {
    if (count === 0) return 'No charts in archive';
    if (pct >= 95) return 'Full lower-48 coverage';
    return `${Math.round(pct)}% of the lower 48 covered`;
  }

  updateCoverageLabel() {
    if (!this.coverageSegs || !this.ui.lblRange || !this.selectedDate) return;
    const [count, pct] = this.coverageAt(new Date(this.selectedDate).getTime());
    this.ui.lblRange.textContent = this.coverageText(count, pct);
    this.ui.lblRange.style.color = count === 0 ? '#ff5c5c' : '';
    this.updateTimelineAccessibility();
  }

  updateTimelineAccessibility() {
    const track = this.ui.track;
    if (!track || !this.frames.length || !this.selectedDate) return;
    const [y, m, d] = this.selectedDate.split('-').map(Number);
    const date = new Date(y, (m || 1) - 1, d || 1);
    const dateText = date.toLocaleDateString('en-US', {
      year: 'numeric', month: 'long', day: 'numeric'
    });
    const coverage = this.coverageSegs && this.ui.lblRange?.textContent
      ? `. ${this.ui.lblRange.textContent}` : '';
    track.setAttribute('aria-valuenow', String(this.currentIndex));
    track.setAttribute('aria-valuetext', `${dateText}${coverage}`);
    // Don't speak every animation frame or pointer movement. The focused
    // slider already announces its own value; other controls get one
    // settled date/coverage update after scrubbing or playback stops.
    clearTimeout(this._announcementTimer);
    if (!this.isPlaying && document.activeElement !== track) {
      this._announcementTimer = setTimeout(() => {
        document.getElementById('timelineStatus').textContent = `${dateText}${coverage}`;
      }, 250);
    }
  }

  initCoverageTip() {
    const tip = this.ui.trackTip;
    const wrap = this.ui.track;
    if (!tip || !wrap) return;
    wrap.addEventListener('mousemove', (e) => {
      const r = wrap.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
      const t = this.minTime + pct * (this.maxTime - this.minTime);
      const [count, covPct] = this.coverageAt(t);
      const label = new Date(t).toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
      tip.style.display = 'block';
      tip.style.left = `${pct * 100}%`;
      // Built as nodes rather than innerHTML: the page's CSP forbids nothing
      // here, but text stays text.
      const bold = document.createElement('b');
      bold.textContent = label;
      const cov = document.createElement('span');
      cov.className = 'tip-cov';
      cov.textContent = this.coverageText(count, covPct);
      tip.replaceChildren(bold, document.createElement('br'), cov);
    });
    wrap.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
  }

  paintHeatStrip() {
    const canvas = this.ui.heatCanvas;
    const segs = this.coverageSegs;
    if (!canvas || !segs) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    const min = this.minTime, span = this.maxTime - this.minTime;
    if (span <= 0) return;

    const rounded = () => {
      const r = 4;
      ctx.beginPath();
      ctx.moveTo(r, 0);
      ctx.arcTo(w, 0, w, h, r);
      ctx.arcTo(w, h, 0, h, r);
      ctx.arcTo(0, h, 0, 0, r);
      ctx.arcTo(0, 0, w, 0, r);
      ctx.closePath();
    };

    // base: solid dark = "no data", painted over by covered segments
    ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
    rounded();
    ctx.fill();
    ctx.save();
    rounded();
    ctx.clip();
    // One pixel column at a time, aggregating every segment the column
    // spans (exact sweep, no point sampling), so a days-long gap inside a
    // month-wide pixel still shows as dark.
    let si = 0;
    for (let px = 0; px < w; px++) {
      const t0 = min + (px / w) * span;
      const t1 = min + ((px + 1) / w) * span;
      while (si < segs.length && segs[si][1] <= t0) si++;
      let anyGap = false, weighted = 0, covered = 0;
      for (let j = si; j < segs.length && segs[j][0] < t1; j++) {
        const [s0, e0, c, p] = segs[j];
        const overlap = Math.min(e0, t1) - Math.max(s0, t0);
        if (overlap <= 0) continue;
        if (c === 0) anyGap = true;
        weighted += p * overlap;
        covered += overlap;
      }
      if (covered < (t1 - t0) * 0.999) anyGap = true; // time outside all segments
      if (anyGap || covered === 0) continue; // leave the dark gap showing
      // Clear the dark base first so it never darkens a translucent
      // covered column and reads as a gap that isn't there.
      ctx.clearRect(px, 0, 1, h);
      ctx.fillStyle = 'rgba(255,255,255,0.06)';
      ctx.fillRect(px, 0, 1, h);
      ctx.fillStyle = this.heatColor(weighted / covered);
      ctx.fillRect(px, 0, 1, h);
    }
    ctx.restore();
  }

  // Monochrome ramp: white, varying only in opacity. Eased so the
  // 70-90% coverage typical of the pre-1970 years already sits close to
  // fully opaque, while even a single chart still paints a clearly visible
  // light bar against the dark no-data gaps.
  heatColor(pct) {
    const p = Math.max(0, Math.min(100, pct));
    const lin = p / 100;
    const t = 1 - (1 - lin) * (1 - lin);
    const alpha = 0.45 + 0.55 * t;
    return `rgba(255, 255, 255, ${alpha.toFixed(3)})`;
  }

  findClosestFrameIndexByPct(pct) {
    const n = this.frames.length;
    if (n === 0) return 0;
    if (pct <= this.frames[0].pct) return 0;
    if (pct >= this.frames[n - 1].pct) return n - 1;

    let lo = 0;
    let hi = n - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.frames[mid].pct < pct) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    const right = lo;
    const left = Math.max(0, right - 1);
    return (Math.abs(this.frames[right].pct - pct) < Math.abs(this.frames[left].pct - pct)) ? right : left;
  }

  findFrameIndexByDate(selectedTime) {
    let lo = 0;
    let hi = this.frameTimes.length - 1;
    let bestIdx = 0;

    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const frameTime = this.frameTimes[mid];
      if (frameTime <= selectedTime) {
        bestIdx = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }

    return bestIdx;
  }

  updateByDate(dateStr, lazy = false) {
    const selectedTime = new Date(dateStr).getTime();
    if (isNaN(selectedTime)) return;

    const bestIdx = this.findFrameIndexByDate(selectedTime);
    this.update(bestIdx, lazy, dateStr);
  }

  // Handle offset is an absolute px value, so it has to be recomputed
  // whenever the track width changes. Geometry only — never paints a frame.
  positionHandle(index) {
    const f = this.frames[index];
    if (!f || !this.ui.handle) return;
    const pct = Number.isFinite(f.pct) ? Math.max(0, Math.min(100, f.pct)) : 0;
    const trackWidth = this.trackWidthPx ?? this.ui.track?.getBoundingClientRect().width ?? 0;
    this.ui.handle.style.setProperty('--handle-position-px', `${trackWidth * (pct / 100)}px`);
  }

  update(index, lazy = false, selectedDateOverride = null) {
    this.currentIndex = index;
    const f = this.frames[index];
    const selectedDate = selectedDateOverride || f.date;
    this.selectedDate = selectedDate;

    // Update date picker to show the selected date
    if (this.ui.timeSelect && this.ui.timeSelect.value !== selectedDate) {
      this.ui.timeSelect.value = selectedDate;
    }

    this.positionHandle(index);
    this.updateCoverageLabel();
    this.updateTimelineAccessibility();

    if (lazy) {
      this.mapCtrl.showFrameDebounced(selectedDate);
    } else {
      // Neighbouring dates are prefetched by the data plane from the scrub
      // state the engine passes with each demand.
      this.mapCtrl.showFrame(selectedDate).catch(() => {});
    }
  }

  step(dir) {
    let next = this.currentIndex + dir;
    if (next >= this.frames.length) next = 0;
    if (next < 0) next = this.frames.length - 1;
    this.update(next);
  }
}
