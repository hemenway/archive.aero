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

- `npm run test:workers`: tile Worker HTTP contracts (exact range bytes, boundaries,
  suffix/open ranges, GET/HEAD, ETags, cache hits, republished objects, CORS and
  failure recovery), plus the existing ATC route and shell tests.
- `npm run test:browser`: desktop Chromium and mobile WebKit exercise the shipped
  viewer: metadata boot, CSV fallback, outage messaging, date/keyboard navigation,
  playback, pin/solo view and share-link restoration. It also runs the existing
  pixel-level flicker/stale-render guard against code extracted from `index.html`.

Browser tests serve the actual page and styles on `127.0.0.1:4173`. Third-party
library requests are fulfilled with pinned npm packages, preserving the page's
integrity checks. Tiny valid PMTiles archives and metadata replace production
data; unexpected network requests fail the tests. No production archive,
credentials or deployment is needed. Clipboard writes are recorded in-page.

Worker tests use Node Request/Response with deterministic R2 and Cache API doubles,
not a Cloudflare runtime emulator. They do not establish production R2/edge-cache
behavior. Browser fixtures likewise do not validate the published archive catalog,
real chart imagery, CDN availability, or performance. Keep staging smoke checks for
those integration boundaries.

## Debugging and CI

```sh
npm run test:browser -- --project=chromium --headed
npx playwright show-report
```

Failures retain screenshots and traces under `test-results/` and an HTML report
under `playwright-report/`. The GitHub Actions regression workflow runs both suites
on pushes and pull requests and uploads those artifacts when it fails. To make
passing tests mandatory before merge, require the `regression` job in the
repository's branch protection settings.
