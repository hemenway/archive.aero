// Classic vendor scripts are deferred, in dependency order. Wait for them even
// if this module finishes fetching first. The app import is preloaded by HTML.
async function boot() {
  if (document.readyState !== 'complete') {
    await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve, { once: true }));
  }
  const status = document.getElementById('loadingStatus');
  if (!window.L || !window.pmtiles || !window.protomapsL) {
    if (status) status.textContent = 'Failed to load map libraries. Check your connection and reload.';
    return;
  }
  let app;
  try {
    app = await import('./viewer.4ac7c3d4abfc0cb5.js');
  } catch (error) {
    console.error('Viewer module failed to load', error);
    if (status) status.textContent = 'Failed to load the chart viewer. Check your connection and reload.';
    return;
  }
  try {
    await app.initApp();
  } catch (error) {
    console.error('Initialization failed', error);
    if (status) status.textContent = 'Failed to load chart data. Check your connection and reload.';
  }
}
boot();
