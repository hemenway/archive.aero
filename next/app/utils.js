// Interface helpers ported from src/viewer.js (Utils). The archive/tile helpers
// stayed behind: the data plane owns that work on the new engine.
export const Utils = {
  // localStorage access itself throws SecurityError when storage is blocked
  // (block-all-cookies, third-party iframe). It is a per-viewer convenience
  // only, so treat it as absent rather than letting the viewer fail to boot.
  storageGet: (key) => {
    try { return window.localStorage.getItem(key); } catch (_) { return null; }
  },
  storageSet: (key, value) => {
    try { window.localStorage.setItem(key, value); } catch (_) { /* unavailable */ }
  },
  toast: (msg, duration = 3000) => {
    const el = document.getElementById('toast');
    if (el) {
      el.textContent = msg;
      el.classList.add('visible');
      // Clear any prior toast's timer so overlapping toasts each show for their
      // full duration instead of an older timer hiding the newer message early.
      clearTimeout(Utils._toastTimer);
      Utils._toastTimer = setTimeout(() => el.classList.remove('visible'), duration);
    }
  },
  formatDateId: (dateStr) => {
    try {
      const [y, m, d] = dateStr.split('-').map(Number);
      const date = new Date(y, m - 1, d);
      return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
    } catch (e) {
      return dateStr;
    }
  },
  toggleFullscreen: () => {
    const el = document.documentElement;
    const request = el.requestFullscreen || el.webkitRequestFullscreen;
    const exit = document.exitFullscreen || document.webkitExitFullscreen;
    const active = document.fullscreenElement || document.webkitFullscreenElement;
    // iOS WebKit (every browser on iPhone) has no element-level fullscreen, so
    // requestFullscreen is undefined and calling it throws synchronously rather
    // than rejecting — feature-detect and give feedback instead of a silent no-op.
    if (!request) {
      Utils.toast('Fullscreen is not supported on this device');
      return;
    }
    try {
      if (!active) {
        const result = request.call(el);
        if (result && typeof result.catch === 'function') {
          result.catch(err => Utils.toast(`Error entering fullscreen: ${err.message}`));
        }
      } else if (exit) {
        exit.call(document);
      }
    } catch (err) {
      Utils.toast(`Error entering fullscreen: ${err.message}`);
    }
  },
  debounce: (func, wait) => {
    let timeout;
    const debounced = function executedFunction(...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => { timeout = null; func(...args); }, wait);
    };
    // Expose cancel() so a direct showFrame can drop a stale pending scrub
    // repaint instead of letting it paint an old date over the newer frame.
    debounced.cancel = () => { clearTimeout(timeout); timeout = null; };
    return debounced;
  }
};

// Integer days since 1970-01-01 UTC for an ISO date: the data plane's Day.
export const dayOf = (dateStr) => Math.floor(Date.parse(dateStr) / 86400000);
