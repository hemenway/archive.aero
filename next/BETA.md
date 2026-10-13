# Running a beta of the viewer

How `beta.archive.aero` was set up for the next viewer (2026-10-04 to
2026-10-12), so the next preview is a morning's work, and how a retired one is
switched off.

## Pieces

| Piece | What | Where |
|---|---|---|
| Page host | A Worker with Static Assets on a custom domain (`beta.archive.aero`); `main` is `next/pages/worker.mjs` so edition pages render there too. | `beta/wrangler.jsonc` (assets `beta/dist`, `run_worker_first: ["/next/sectionals/*"]`) |
| Data host | A second copy of the tiles Worker on its own hostname and bucket, no analytics binding, so preview traffic never lands in `tile_logs`. | `worker/wrangler.beta.toml` → `tiles-beta`, `data-beta.archive.aero`, bucket `charts-beta` |
| Data | Hashed era archives, chart artifacts, `next/manifest.<hash>.json`, pins, chart index, airports, airfields; hashed copies of the basemap and airspace. | the bucket named in the data host's config |
| Shell | `node next/build.mjs` with the preview's manifest URL, `--base /next/` (or `/`), `--no-analytics`, `--links-origin https://archive.aero` (site links and robots `noindex` follow from it). | `beta/dist/next/` |

A hostname other than the apex gets `noindex, nofollow` on every page from
`--links-origin`; never run a preview without it or Google indexes two viewers.

## Standing one up

1. Bucket: `charts-beta` already holds the production hashed set (it *is* the
   production hashed bucket since 2026-10-12, `BUCKET_HASHED` of `tiles`). A
   preview of new data goes under a new prefix or a new bucket; name it in a
   copy of `worker/wrangler.beta.toml` and deploy `npx wrangler deploy -c
   worker/wrangler.beta.toml` (routes carry the `custom_domain`, DNS is made
   by the deploy).
2. Data: stage and build as `worklists/18_apex_cutover.md` section 5 does, with
   `next_build_manifest.py --tile-base https://<data host>/t/ --file-base
   https://<data host>/`; `rclone copy` the `next/` outputs to the bucket.
3. Shell: `node next/build.mjs --manifest https://<data host>/next/manifest.<hash>.json
   --manifest-file <local copy> --base /next/ --outdir beta/dist/next
   --no-analytics --links-origin https://archive.aero`, then replace everything
   from the `/next/*` line of `beta/dist/_headers` with `beta/dist/next/_headers`
   (Cloudflare reads only the root `_headers`), `cd beta && npx wrangler deploy`.
4. Check: the page's CSP names the data host in `connect-src`; "View alone"
   changes the address; the console shows no CSP error and the service worker
   registered at the base.

## Retiring one

Set `"RETIRED": "1"` in the host's wrangler vars and deploy: every path answers
`410 Gone` with a line pointing at archive.aero (`retired()` in
`next/pages/worker.mjs`). The shell's service worker unregisters itself when
its document comes back 410, so nobody stays pinned to the cached preview.
Leave it so for a few weeks (bookmarks, crawlers), then delete the Worker and
its custom domain. `tiles-beta` and `data-beta.archive.aero` can go at the same
time unless the bucket behind them is still production's (it is, for
`charts-beta`: keep the bucket, drop only the preview Worker).
