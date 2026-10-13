# 18 — The next viewer at the apex (cutover runbook)

**Prepared 2026-10-12, revised the same evening** after three decisions: the
reslice's hashed archives stay in the bucket they were uploaded to (`charts-beta`
becomes production's second bucket; nothing is copied), there is no legacy
viewer page, and `beta.archive.aero` is retired behind a 410 once the apex is
live. Code is on `main`. The steps below touch the account (R2 writes, Worker
deploys, the push) and are run by hand; each is idempotent. Order: **data,
then the tiles Worker, then the push, then the site Worker** — the page must
never reference a hash that is not yet reachable (next/HOSTING.md).

What is where:

| What | Where it is | Where it goes |
|---|---|---|
| Hashed era archives + chart artifacts (3,761 + 7,389, 359 GB) | `r2:charts-beta/sectionals/` | stays; served by `tiles` as `BUCKET_HASHED` |
| Hashed basemap, airspace, airfields, pins `pins.e8fbe410908f` | `r2:charts-beta/` | stays |
| Chart cards, 7,066 JPEGs (1.15 GB) | `/Volumes/projects/archive-next/og/sectionals/` | `r2:charts/sectionals/` (plain keys, `charts`) |
| Manifest `manifest.7a3ad0e24662.json`, chart index `charts.958e6035f0fa/` (151 files), airport records `airports.370ad3ae35b6/` (51 files) | `~/archive-next-build/prod/next/` | `r2:charts-beta/next/` |
| Root shell build (`sitemap-sectionals.xml` 7,083 URLs, `sitemap-airports.xml` 50 URLs inside) | `next/site/dist/` | the site Worker's assets |

## 1. Data

```bash
rclone copy /Volumes/projects/archive-next/og/sectionals r2:charts/sectionals --include '*.jpg' --s3-no-head --transfers 16 --stats 30s --stats-one-line
```

```bash
rclone copy ~/archive-next-build/prod/next/charts.958e6035f0fa r2:charts-beta/next/charts.958e6035f0fa --s3-no-head
```

```bash
rclone copy ~/archive-next-build/prod/next/airports.370ad3ae35b6 r2:charts-beta/next/airports.370ad3ae35b6 --s3-no-head
```

```bash
rclone copy ~/archive-next-build/prod/next/manifest.7a3ad0e24662.json r2:charts-beta/next/ --s3-no-head
```

Check: `rclone size r2:charts/sectionals --include '*.jpg' --json` → 7,066 objects;
`rclone lsf r2:charts-beta/next/` lists the three new names.

## 2. The tiles Worker — now two buckets

`worker/wrangler.toml` binds `charts` as `BUCKET` and `charts-beta` as
`BUCKET_HASHED`; `src/index.js` routes hashed keys and `next/*` to the second
(and logs z/x/y for `/t/` reads). Deploy, then smoke both halves:

```bash
cd worker && npm run deploy
```

```bash
curl -sI https://data.archive.aero/t/sectionals/1928-04-01_to_1929-11-01.25a9b218f55b/5/8/12 | head -5; curl -sI https://data.archive.aero/next/manifest.7a3ad0e24662.json | head -3; curl -sI https://data.archive.aero/sectionals/timeline_data.json | head -2
```

Expected: 200/204 with `cache-control: public, max-age=31536000, immutable` for
the first two (hashed bucket), 200 for the third (plain bucket).

## 3. Push main (the Pages deploy carries the new sitemap index and the re-fingerprinted viewer)

```bash
git push origin main
```

## 4. The site Worker — the switch

```bash
cd next/site && npx wrangler deploy
```

Verify from the apex:

```bash
for p in / /about /contribute /sources /sitemap.xml /sitemap-sectionals.xml /sitemap-airports.xml /robots.txt /atc/ /sectionals/chart/boston_ma/1933-03-01 /sectionals/chart/Boston_MA/1933-03-01/ /sectionals/chart/boston_ma/1999-01-01 /airports/kbos /airports/KBOS /airports/kxyz; do printf '%-48s ' "$p"; curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' "https://archive.aero$p"; done
```

Expected: `/` 200 with a `content-security-policy` header and the manifest hash
in the page; `/about` `/contribute` `/sources` `/sitemap.xml` `/robots.txt`
`/atc/` 200 (Pages or the ATC Worker, untouched); both sitemaps 200; the edition
page and `/airports/kbos` 200; the mixed-case spellings 301 to lowercase; the
untiled date and the unknown ident 404.

```bash
curl -s https://archive.aero/sectionals/chart/boston_ma/1933-03-01 | grep -o '<meta property="og:image" content="[^"]*"'; curl -s https://archive.aero/airports/kbos | grep -o 'data-airport="[^"]*" data-lat="[^"]*"'
```

Then in a browser: `https://archive.aero/` paints charts, "View alone" changes
the address, `https://archive.aero/airports/kbos` opens on Logan with the
decades panel, the console shows no CSP error (Plausible and the Cloudflare
beacon both load), and the service worker registered at `/`. Resubmit
`https://archive.aero/sitemap.xml` in Search Console.

## 5. After the switch

Refresh the permanent per-chart keys from the reslice (7,306 published from the
August warps + 3 never published; server-side copies `charts-beta` → `charts`,
same keys, sha256-verified, receipts appended; ~1–2 h):

```bash
~/venv/bin/python scripts/publish_chart_pmtiles_from_reslice.py --refresh --copy
```

then `~/venv/bin/python scripts/build_timeline_data.py`, upload
`timeline_data.json` (CLAUDE.md recipe) and `scripts/build_coverage.py`.
Still held/unpublishable: Boston 1957-06-01 (worklist 04), 10 retired and 9
renamed inset keys (listed by the script's report).

Retire the beta (next/BETA.md): in `beta/wrangler.jsonc` set `"RETIRED": "1"`,
then `cd beta && npx wrangler deploy`; every path answers 410, the preview's
service worker unregisters. After a few weeks delete the `archive-aero-beta`
Worker and `beta.archive.aero`; keep `tiles-beta` only if another preview is
coming (the bucket behind it is production's and stays regardless).

Later, once nothing reads them: the 4,031 pre-reslice era archives in `charts`
(`sectionals/<era>.pmtiles`, ~350 GB) served only the Leaflet viewer and can go;
the metadata bundles stay (the heatmap needs them). The Leaflet viewer's own
files (`index.html`, `src/`, `vendor/`, `assets/`) can leave `site-files.json`
once the apex has been the shell for a while.

Two edition pages have no card because their artifacts are broken (bbox a
few metres wide): `denver/2016-01-07`, `san_francisco/2016-08-18` — re-warp.

## Rollback

```bash
cd next/site && npx wrangler delete
```

`/` is the Pages page again at once (the Leaflet viewer, until its files leave
`site-files.json`). Nothing in R2 needs undoing; the tiles Worker's second
bucket is additive.

## Rebuilding the production data

```bash
P=~/archive-next-build/prod/next; C=~/Documents/Codex/2026-10-12/upd/work/data; ~/venv/bin/python scripts/next_build_overlays.py --airfields airfields.json --timeline $C/work/timeline.json --plan $C/work/plan.json --out $P --og-manifest /Volumes/projects/archive-next/og/manifest.jsonl --sitemap https://archive.aero && ~/venv/bin/python scripts/next_build_airports.py --charts $(ls -d $P/charts.*/) --timeline timeline_data.json --out $P --sitemap https://archive.aero && ~/venv/bin/python scripts/next_build_manifest.py --dir $C/stage --coverage coverage.json --overlays $P/overlays.json --airports $P/airports.json --out $P
```

```bash
P=~/archive-next-build/prod/next; M=$(ls -t $P/manifest.*.json | head -1); node next/build.mjs --manifest https://data.archive.aero/next/$(basename $M) --manifest-file $M --base / --outdir next/site/dist --sitemap $P/sitemap-sectionals.xml --sitemap $P/sitemap-airports.xml
```

`$C/stage` is the sparse-stub stage of the complete reslice (the 2026-10-12
Codex refresh; `scripts/next_stage_beta.py` rebuilds one from
`slicer-runs/2026-10-03_reslice/hashed/`). Cards: `scripts/next_og_images.py
--index <charts dir> --out /Volumes/projects/archive-next/og` (resumable;
`--only slug`). Upload the new `next/` files (step 1), redeploy the site Worker
(step 4). The real build's critical JavaScript is within 110 bytes of its
50 KB budget: anything new in `next/app/main.js` goes in the cards chunk.
