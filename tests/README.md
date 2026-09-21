# Regression coverage

Use Node.js 22 (the version used in CI). From the repository root:

```sh
npm ci
npx playwright install chromium webkit
npm test
```

On Linux, install browser system dependencies with
`npx playwright install --with-deps chromium webkit`.

## Suites

- `npm run check:frontend`: verifies vendor versions/SRI and generated module fingerprints.
- `npm run test:frontend`: checks legacy publishing-script compatibility and module regeneration.
- `npm run test:workers`: tile Worker HTTP contracts (exact range bytes, boundaries,
  suffix/open ranges, GET/HEAD, ETags, cache hits, republished objects, CORS and
  failure recovery), plus ATC routing, shell, facility filters, old location
  links, and selection-specific cache validators.
- `npm run test:browser`: desktop Chromium and mobile WebKit exercise the shipped
  viewer: metadata boot, CSV fallback, outage messaging, date/keyboard navigation,
  playback, pin/solo view and share-link restoration. It also runs the existing
  pixel-level flicker/stale-render guard against code extracted from `src/viewer.js`.
  Loading tests cover local/deferred dependencies, on-demand PapaParse, missing or
  slow scripts, the no-DecompressionStream fallback, and conditional asset requests.

Browser tests serve the actual page, fingerprinted modules, styles and vendored
libraries on `127.0.0.1:4173`, preserving the page's integrity checks. The local
server emulates cache headers/ETags; it does not establish production cache policy.
Tiny valid PMTiles archives and metadata replace production
data; unexpected network requests fail the tests. No production archive,
credentials or deployment is needed. Clipboard writes are recorded in-page.

Worker tests use Node Request/Response with deterministic R2 and Cache API doubles,
not a Cloudflare runtime emulator. They do not establish production R2/edge-cache
behavior. Browser fixtures likewise do not validate the published archive catalog,
real chart imagery, CDN availability, or performance. Keep staging smoke checks for
those integration boundaries.

## Debugging and CI

The g2p dependency updater has a Python suite (standard library only), also run in CI:

```sh
python3 -m unittest discover -s tests -p 'test_geotiff2pmtiles_dependency.py' -v
```

It checks commit caching, switching to a new upstream revision while preserving
the binary used by an existing batch, and failure handling without stale fallback.
It does not download Go dependencies or convert production chart data.

```sh
npm run test:browser -- --project=chromium --headed
npx playwright show-report
```

Failures retain screenshots and traces under `test-results/` and an HTML report
under `playwright-report/`. The GitHub Actions regression workflow runs both suites
on pushes and pull requests and uploads those artifacts when it fails. To make
passing tests mandatory before merge, require the `regression` job in the
repository's branch protection settings.
