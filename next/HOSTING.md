# Hosting proposal (no deployment performed)

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
script-src 'self' <boot-script-hash> <plausible-init-hash> https://plausible.io;
style-src 'self' 'unsafe-inline';
img-src 'self' data: blob:;
font-src 'self';
connect-src 'self' https://data.archive.aero https://get.geojs.io https://plausible.io;
worker-src 'self';
object-src 'none';
base-uri 'self';
frame-ancestors 'none'
```

Inline style permission covers critical CSS and the dynamic timeline/card
geometry. Script hashes cover the final minified bytes; do not edit generated
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
   Retain the old viewer at a stable rollback path, and change the
   unsupported-browser fallback from `/` to that path when `/` becomes the new
   shell.
5. Roll back by restoring the prior HTML and unregistering the new root service
   worker if needed. Maintain both shell asset generations through the rollback
   window. To retire `/next/`, serve its document with 410 for at least the
   rollback window: the preview worker unregisters itself on seeing it.

No `.github/workflows`, `site-files.json`, `CNAME`, or deployment configuration is
changed by this branch. Hosting changes require a later deployment task.
