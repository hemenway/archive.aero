// PapaParse stays off the normal metadata-bundle path, including preloads.
// Share one in-flight load; a failed attempt is retryable.
let parserPromise;
export function loadPapaParse() {
  if (window.Papa?.parse) return Promise.resolve(window.Papa);
  if (parserPromise) return parserPromise;
  parserPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = '/vendor/papaparse/5.4.1/papaparse.min.js';
    script.integrity = 'sha384-D/t0ZMqQW31H3az8ktEiNb39wyKnS82iFY52QPACM+IjKW3jDUhyIgh2PApRqJZs';
    script.crossOrigin = 'anonymous';
    script.async = true;
    const finish = error => {
      clearTimeout(timeout);
      script.onload = script.onerror = null;
      if (error) { script.remove(); reject(error); }
      else resolve(window.Papa);
    };
    const timeout = setTimeout(() => finish(new Error('CSV fallback unavailable: PapaParse timed out')), 15000);
    script.onload = () => finish(window.Papa?.parse ? null : new Error('CSV fallback unavailable: PapaParse did not initialize'));
    script.onerror = () => finish(new Error('CSV fallback unavailable: PapaParse failed to load'));
    document.head.appendChild(script);
  }).catch(error => { parserPromise = null; throw error; });
  return parserPromise;
}
