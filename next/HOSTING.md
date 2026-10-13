# Hosting

**Live since 2026-10-12 at the apex.** `archive.aero/` is the shell, served by the
`archive-aero-site` Worker (`next/site/wrangler.jsonc`: Static Assets from a
`node next/build.mjs --base / --outdir next/site/dist` build, the edition-page
Worker as `main`, `run_worker_first: true`) on the zone route `archive.aero/*`.
The apex DNS record still points at GitHub Pages behind the Cloudflare proxy:
every path the assets do not hold (`/about`, `/contribute`, `/sources`, the
root sitemap index and the older `/assets/`) is a 404 in the assets that the
Worker turns into a plain `fetch` to that origin (`ORIGIN_FALLTHROUGH="1"`).
`/atc/*` keeps its own, more specific route. Rollback is removing the route (or
`npx wrangler delete` in `next/site/`): `/` is the Pages page again at once.
There is no legacy viewer page (decision 2026-10-12): a browser without WebGL2
gets a message and a link to /about.

Data: the reslice's hashed archives stay in the bucket they were uploaded to,
`charts-beta`, which the `tiles` Worker now binds as `BUCKET_HASHED` beside
`charts` (`bucketFor()` routes hashed keys and `next/*` there), so the manifest
points at `data.archive.aero` and nothing was copied. Airport pages
(`/airports/<ident>`, `next/pages/airport-page.mjs`) ride the same Worker. The
preview host is retired with `RETIRED="1"` and `run_worker_first: true` (next/BETA.md).
The site Worker uses its `DATA` service binding to `tiles` for server-rendered
chart and airport records; a public same-zone fetch bypasses the tiles route
and reaches the old R2 origin. The runbook is worklists/18_apex_cutover.md.


Serve the built shell with Cloudflare Workers Static Assets. For the preview the
shell lives at `beta.archive.aero/next/`, inside the MapLibre beta's assets
(`beta/dist/next/`, written there directly with `build.mjs --outdir`), except
`_headers`, which Cloudflare reads only from the asset root: its rules are
appended to the beta's root `_headers`. Its rules are already prefixed with the build's `--base` (`/next/*`
for the preview), so they never govern the current viewer or any other page.
Keep the current viewer and its data routes available throughout the preview and
switch.

Cloudflare supports response headers through an assets `_headers` file, including
long-lived immutable caching for content-hashed files. The build emits exact paths
for JS and CSS rather than an overlapping `*.js` rule: Cloudflare joins duplicate
headers, so `sw.js` must never inherit the immutable policy. See the
[Static Assets headers documentation](https://developers.cloudflare.com/workers/static-assets/headers/)
and [Static Assets routing](https://developers.cloudflare.com/workers/static-assets/binding/).

GitHub Pages can serve fingerprinted filenames, but does not expose the equivalent
per-path response-header configuration needed for this caching policy and CSP.
Putting `_headers` in a Pages repository does not activate Cloudflare's parser.
GitHub's [Pages header discussion](https://github.com/orgs/community/discussions/54257)
and [Pages caching discussion](https://github.com/orgs/community/discussions/11884)
record this limitation. A Cloudflare proxy in front of Pages could supply these
headers; serving assets directly avoids adding another origin dependency.

## Edition pages

`/<base>sectionals/chart/<slug>/<date>` is served by `next/pages/worker.mjs`
running ahead of the asset handler (`beta/wrangler.jsonc`: `main`, the
`ASSETS` binding and `run_worker_first: true`; retirement must intercept
static pages as well as edition pages). The Worker reads `<base>chart-pages.json`,
which `build.mjs` writes beside the page: the chart-index URL from the
manifest, the page's CSP, the links origin and the robots policy (a preview
host built with `--links-origin` gets `noindex, nofollow`). It fetches the
location's chart index, renders the shell page for that one edition (title,
description, canonical, Open Graph, Twitter, JSON-LD, the chart-view panel as
HTML with links to the location's other editions) and answers with the shell's
CSP and `Cache-Control: no-cache`. Other spellings (uppercase, trailing slash,
`.pmtiles`, one half of a pair) are 301s to the canonical address; a date with
no tiled chart is the host's 404 page (thin-content guard, worklist 09 B5).
The shell's own URLs are absolute under the base for this reason: a
document-relative `./app.js` would resolve inside the chart's directory.
The service worker passes these navigations through (only `<base>` and
`<base>index.html` are network-first with an offline fallback). At the root
switch the same Worker runs at `/sectionals/chart/*` with `base: "/"`.
Per-page `og:image` (09 B3): `scripts/next_og_images.py` renders one 1200x630
card per edition page from the chart's own tiles, uploaded as
`sectionals/chart/<slug>/<date>.jpg` on the data host; `next_build_overlays.py
--og-manifest` records them in the chart index (`og`), and the Worker resolves
`og` against `chart-pages.json`'s `files` (the manifest's `fileBase`), falling
back to the site card. `sitemap-sectionals.xml` (09 B4): written by
`next_build_overlays.py --sitemap https://archive.aero` (tiled charts only),
copied into the build by `build.mjs --sitemap`, listed in the root
`sitemap.xml` index.

## Cache policy

| Resource | Cache-Control | Owner |
| --- | --- | --- |
| HTML, including `/next/` and `/next/index.html` | `no-cache` | Static Assets |
| Hashed app chunks, worker and CSS | `public, max-age=31536000, immutable` | Static Assets |
| Hashed manifest and tile metadata | `public, max-age=31536000, immutable` | Assets or data Worker |
| `sw.js` | `no-cache` | Static Assets |
| Versioned `/t/*` tiles, including 204 | `public, max-age=31536000, immutable` | Tiles Worker |
| Unknown tile path, 404 | `public, max-age=60` | Tiles Worker |
| Non-versioned data pointer, if introduced | `no-cache` | Data Worker |

Do not add tile requests to CacheStorage. Immutable HTTP caching already retains
them. The service worker precaches the shell and current manifest. Only the
shell's own document (`/next/` and `/next/index.html`, or `/` and `/index.html`
once registered at the root) is served network-first with a 2.5-second timeout
and cached HTML fallback; `/atc/`, `/about` and every other page pass through
untouched. Redirects and a retired shell (404, 410) also pass through — 410
unregisters the worker — so returning visitors are never pinned to a stale
page. Each registration directory owns its own cache family
(`archive-next-_next_-*`, `archive-next-_-*`), and activation sweeps only its
own older versions. It waits for the ordinary service worker lifecycle rather
than forcing a new worker into already-open tabs. Offline mode provides the
shell; charts need prior HTTP cache entries or an available connection.

## Optional same-origin data route

Route `/data/*` to the tiles Worker ahead of the static asset handler, preserving
methods, status codes, content types and immutable headers. Rewrite
`/data/t/{path}/{z}/{x}/{y}` to the existing `/t/{path}/{z}/{x}/{y}` endpoint and
`/data/next/*` to `/next/*` for metadata files. Keep the current range endpoints
unchanged. This can use a service binding rather than an extra public-network hop.

Emit a manifest with `tileBase: "https://archive.aero/data/t/"` and
`fileBase: "https://archive.aero/data/"`. Version the manifest after this change;
its hash must change with its bytes. The shell resolves every URL against those
bases, including its early fetches. Same-origin requests avoid the cross-origin
handshake and permit a narrower CSP once all data uses the route. Cache keys must
include the full versioned path; do not remove its hash.

## Content security policy

`next/dist/_headers` contains the concrete policy and fresh SHA-256 hashes for
both inline scripts. Its equivalent template is:

```
default-src 'self';
script-src 'self' <boot-script-hash> <plausible-init-hash> https://plausible.io https://static.cloudflareinsights.com;
style-src 'self' 'unsafe-inline';
img-src 'self' data: blob:;
font-src 'self';
connect-src 'self' https://data.archive.aero https://get.geojs.io https://plausible.io https://cloudflareinsights.com;
worker-src 'self';
object-src 'none';
base-uri 'self';
frame-ancestors 'none'
```

The two cloudflareinsights origins are Cloudflare Web Analytics, which the zone
injects into every HTML page; without them the beacon is blocked (seen on the
beta's console 2026-10-12) and the RUM visit counts go dark. Inline style
permission covers critical CSS and the dynamic timeline/card geometry. Script hashes cover the final minified bytes; do not edit generated
HTML after calculating them. Fonts are system fonts. `data:` enables the SVG
favicon; `blob:` enables decoder fallbacks. `get.geojs.io` is contacted only after
the visitor requests Locate and GPS fails. Add approved manifest origins to
`connect-src` when using a different tile host. Serve sourcemaps separately or
retain them in build artifacts; they are not precached or preloaded.

## Rollout

1. Merge data contract, renderer, data plane, then shell. Complete README's
   integration checklist and rebuild against the immutable production manifest.
2. Serve `/next/` as an opt-in preview with its service worker scoped to `/next/`.
   Preserve `/`, current manifest/bundle URLs, and all existing range routes.
3. Compare cold and warm boots, timeline swaps, long mobile scrubs, memory, errors,
   and actual production transfer sizes. Verify deployed headers with GET/HEAD,
   including 204 tiles and CSP. Keep generated hashed artifacts for old HTML.
4. For the switch, build with `--base /` and publish the shell at `/`; the
   generated `_headers` rules then target the root. Register the root worker only
   after acceptance. Publish HTML last, after every referenced hash is reachable.
   A browser without WebGL2 receives a message and an About link.
   *Done 2026-10-12 without a retained viewer: `next/site/wrangler.jsonc`;
   data first (manifest, chart index, airports to the hashed bucket; cards to
   `charts`), then the tiles Worker with its second bucket, then the push, then
   the site Worker.*
5. Roll back by restoring the prior HTML and unregistering the new root service
   worker if needed. Maintain both shell asset generations through the rollback
   window. To retire `/next/`, serve its document with 410 for at least the
   rollback window: the preview worker unregisters itself on seeing it.

Deployment verification and current data hashes are recorded in
`worklists/18_apex_cutover.md`. Site and beta deployment commands use the pinned
Wrangler installation under `next/`.
