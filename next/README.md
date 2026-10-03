# archive.aero next shell

`CONTRACT.md` is the frozen contract, copied verbatim from the shared prompt.
The shell uses C7 directly. No production viewer file is replaced.

## Standalone development

From the repository root:

```sh
npm ci
npm run build:next
node next/server.mjs
# http://127.0.0.1:4183/next/
npm run test:next
node next/build.mjs --check
```

`--base /next/` (the default) is the URL prefix the shell is deployed under; every
`_headers` rule is scoped to it. `--outdir DIR` writes a build somewhere other
than `next/dist` (a hosting tree, so the committed stub build stays the one the
tests use). `--allow-over-budget` ships a manifest above the 80 KB C2 budget
while the shard format is decided. `--check` reads all three back from that
directory's `budgets.json`.

The committed build uses the 2D renderer and synthetic OffscreenCanvas Worker
selected by `--stubs`. Its hashed C2 manifest has overlapping eras, coverage and
sample airfields. The development server supplies tile responses locally; the
Worker draws fixture imagery. Playwright runs Chromium, WebKit and mobile WebKit
against its own server on port 4183. It fails on unexpected requests and uncaught
page errors. Plausible is stubbed in tests. `npm test` is unchanged.

`node next/build.mjs` also falls back to stubs while either real module is missing.
`--check` infers the committed mode and manifest URL from `dist/budgets.json`;
explicit arguments take precedence. It checks every generated file and rejects
extra stale outputs without writing. Always commit regenerated `next/dist/`.

## Real modules

After renderer and data plane merge:

```sh
node next/build.mjs --manifest https://data.archive.aero/next/manifest.HASH.json
node next/build.mjs --check --manifest https://data.archive.aero/next/manifest.HASH.json
```

The beta preview is built straight into the beta's hosting tree (the beta
checkout is separate; its `beta/` directory is not in git):

```sh
node next/build.mjs --manifest https://data.archive.aero/next/manifest.HASH.json --manifest-file /path/to/manifest.HASH.json \
  --base /next/ --allow-over-budget --outdir /path/to/archive.aero/beta/dist/next
```

Then append `beta/dist/next/_headers` to the beta's root `_headers` (Cloudflare
reads only the root file), delete the nested copy, and deploy the beta Worker.

Replace HASH with an actual immutable manifest hash. A deterministic local copy
of that same manifest can be supplied with `--manifest-file /absolute/path.json`;
its URL remains the browser fetch URL. The build requires the real
`next/renderer/index.js`, `next/dataplane/index.js`, and `next/dataplane/worker.js`
entries. It bundles the Worker separately and rewrites the data plane's
`new URL('./worker.js', import.meta.url)` reference to the hashed worker output.
`--stubs` always selects both stubs, even when real modules are present.

The service worker registers automatically only in real builds. Stub browser
tests register it explicitly for offline/precaching verification. The server is
for fixtures, not a production data proxy. Use the hosting plan for deployment.

## Merge and integration

Merge order: **data contract → renderer → data plane → shell**. Resolve root
`package.json` and lockfile changes by preserving the other branches' additions
alongside esbuild, `build:next`, and `test:next`.

- [x] Run the real modules against agent 1's canonical fixtures/server:
      `npm run test:next:real` builds a real shell against the fixture manifest,
      serves the fixtures through the actual tiles Worker handler and applies
      the build's CSP (Chromium + mobile WebKit; needs WebGL2, so it is not in
      the default `npm test` chain). The stub suite keeps its own fixture data
      and its strict unexpected-request guard.
- [x] Adopt the additive `earlyFetches: Map<absolute URL, Promise<Response>>`
      option described in `shell/NOTES.md` (the data plane strips it from the
      Worker options and primes its scheduler with the responses).
- [x] Verify the hashed Worker URL rewrite against the merged data plane entry
      (confirmed in a simulated one-era real build during the 2026-10-02 review).
- [ ] Verify C7 camera events, bitmap ownership, absent tiles, full-plan swaps,
      resident ancestor fallback, renderer context restoration, clip rings and
      playback readiness on real imagery.
- [ ] Confirm optional airspace status data, pin stack fields, airfield operating
      dates and source links against production fixtures.
- [x] Re-run browser behaviors with the real modules, using canonical airfield
      binaries, pin shards, metadata and real raster/vector tile responses
      (boot, Worker-side data plane, early-fetch adoption, chart pixels, date
      swap, airspace status, airfield browser, pin, solo — see
      `tests-real/real.spec.mjs`; the full keyboard/a11y matrix stays on stubs).
- [ ] Recheck end-to-end budgets including Worker and inline JS, actual request
      counts before first chart paint, manifest gzip ≤80 KB, and no font requests.
- [ ] Measure WebKit mobile memory during repeated full-archive scrubs, playback,
      zoom and rotation; ensure texture eviction and decoded/encoded data caches
      stay bounded. Stub WebKit tests are behavioral coverage, not GPU-memory
      acceptance tests.
- [ ] Validate navigation and manifest recovery offline, then deployed CSP and
      cache headers. Keep rollback shell hashes reachable throughout rollout.
- [ ] Follow HOSTING.md for `/next/` preview and the later root switch. Update the
      unsupported-renderer fallback before replacing the current viewer at `/`.

## Structure

- `app/main.js`: state projection, C7 orchestration, controls and accessibility.
- `app/store.js`: sub-1 KB store; shallow updates and batched notifications.
- `app/stubs/`: exact C7-shaped fixture implementations; approximate 2D drawing.
- `shell/template.html`, `shell/styles.css`, `shell/early.js`: static chrome and boot.
- `build.mjs`: esbuild, hashes, sourcemaps, HTML/CSP/preload generation and budgets.
- `sw.js`: shell-only offline caching; never duplicates the HTTP tile cache.
