# 18 — The next viewer at the apex (cutover runbook)

**Prepared 2026-10-12.** Code landed on `main` in `21da1d7` (next/site Worker,
/classic/ copy, per-chart cards, sitemap, 96 KB budget, /t/ analytics). The
data copies and the two deploys below are the cutover itself; they need the
account (R2 writes and Worker deploys are denied to the assistant's session).
Each step is idempotent and can be rerun. Order matters: **data, then Pages,
then the site Worker** — the page must never reference a hash that is not yet
reachable (next/HOSTING.md, rollout step 4).

Local build inputs (already produced; regenerate with the recipe at the end):

| What | Where |
|---|---|
| Hashed era archives + chart artifacts (3,761 + 7,389) | `r2:charts-beta/sectionals/` (359 GB) |
| Chart cards (`sectionals/chart/<slug>/<date>.jpg`) | `/Volumes/projects/archive-next/og/sectionals/` + `manifest.jsonl` |
| Production `next/` files: manifest, chart index, pins, sitemap | `~/archive-next-build/prod/next/` (see `HASHES` below) |
| Root shell build | `next/site/dist/` (`node next/build.mjs --base / …`) |

Current build (2026-10-12 evening): manifest `manifest.69846aedfac5.json`, chart
index `charts.214a56eed600` (506 of 7,083 pages carried a card when it was built;
the renderer was still running — rebuild per section 5 once
`/Volumes/projects/archive-next/og/manifest.jsonl` is complete, which changes both
hashes), pins `pins.e8fbe410908f` (unchanged from the beta).

## 1. Data into the production bucket (server-side copies, nothing downloads)

```bash
rclone copy r2:charts-beta/sectionals r2:charts/sectionals --s3-no-head --transfers 16 --checkers 32 --stats 30s --stats-one-line
```

```bash
rclone copy r2:charts-beta/basemap/protomaps-20260826.0938e8f9d55b.pmtiles r2:charts/basemap/ --s3-no-head
```

```bash
rclone copy r2:charts-beta/next/pins.e8fbe410908f r2:charts/next/pins.e8fbe410908f --s3-no-head
```

Then the locally built files (uploads):

```bash
rclone copy /Volumes/projects/archive-next/og/sectionals r2:charts/sectionals --include '*.jpg' --s3-no-head --transfers 16
```

```bash
rclone copy ~/archive-next-build/prod/next/charts.214a56eed600 r2:charts/next/charts.214a56eed600 --s3-no-head
```

```bash
rclone copy ~/archive-next-build/prod/next/manifest.69846aedfac5.json r2:charts/next/ --s3-no-head
```

Check (counts must match: 11,144 hashed objects under sectionals/, plus the cards):

```bash
rclone size r2:charts-beta/sectionals --json; rclone size r2:charts/sectionals --include '*.*.pmtiles' --json; rclone size r2:charts/sectionals --include '*.jpg' --json
```

The airspace archive `airspace/class-20260916b.1cb1e18a19b5.pmtiles` and
`next/airfields.3561417ba096.{bin,json}` are already in `charts`.

## 2. The tiles Worker (adds z/x/y to /t/ logging; otherwise the code the beta already runs)

```bash
cd worker && npm run deploy
```

Smoke: `curl -sI https://data.archive.aero/t/sectionals/1928-04-01_to_1929-11-01.25a9b218f55b/5/8/12` → 200 or 204, `cache-control: public, max-age=31536000, immutable`.

## 3. GitHub Pages — publishes /classic/ (push main; the Deploy workflow runs)

```bash
git push origin main
```

Wait for the Pages deploy, then: `curl -sI https://archive.aero/classic/` → 200,
`curl -s https://archive.aero/classic/ | grep -c noindex` → 1. The fallback link
in the new shell points here, so this must be live before step 4.

## 4. The site Worker — this is the switch

```bash
cd next/site && npx wrangler deploy
```

Verify (all from the apex):

```bash
curl -sI https://archive.aero/ | grep -i -E "content-security-policy|cache-control|cf-ray"; curl -s https://archive.aero/ | grep -o 'manifest\.[0-9a-f]*\.json' | head -1
```

```bash
for p in /about /contribute /sources /sitemap.xml /robots.txt /classic/ /atc/ /sectionals/chart/boston_ma/1933-03-01 /sectionals/chart/Boston_MA/1933-03-01/ /sectionals/chart/boston_ma/1999-01-01 /sitemap-sectionals.xml; do printf '%-50s ' "$p"; curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' "https://archive.aero$p"; done
```

Expected: `/about` `/contribute` `/sources` `/sitemap.xml` `/robots.txt` `/classic/`
`/atc/` 200 (Pages or the ATC Worker, unchanged); the edition page 200 with
`og:image` naming `data.archive.aero/sectionals/chart/boston_ma/1933-03-01.jpg`;
the mixed-case spelling 301 to the lowercase one; the untiled date 404;
`/sitemap-sectionals.xml` 200. Open `https://archive.aero/` in a browser: charts
paint, "View alone" changes the address, Plausible and the Cloudflare beacon
load without CSP errors in the console.

Then submit `https://archive.aero/sitemap.xml` again in Search Console (the
index now lists `sitemap-sectionals.xml`).

## 5. Rebuilding the production data (when the inventory or the cards change)

```bash
P=~/archive-next-build/prod/next; C=~/Documents/Codex/2026-10-12/upd/work/data; ~/venv/bin/python scripts/next_build_overlays.py --airfields airfields.json --timeline $C/work/timeline.json --plan $C/work/plan.json --out $P --og-manifest /Volumes/projects/archive-next/og/manifest.jsonl --sitemap https://archive.aero && ~/venv/bin/python scripts/next_build_manifest.py --dir $C/stage --coverage coverage.json --overlays $P/overlays.json --out $P
```

```bash
P=~/archive-next-build/prod/next; M=$(ls -t $P/manifest.*.json | head -1); node next/build.mjs --manifest https://data.archive.aero/next/$(basename $M) --manifest-file $M --base / --outdir next/site/dist --sitemap $P/sitemap-sectionals.xml
```

(`$C/stage` is the sparse-stub stage of the complete reslice that the 2026-10-12
Codex refresh built; `scripts/next_stage_beta.py` rebuilds one from
`slicer-runs/2026-10-03_reslice/hashed/`.) Upload the new manifest and chart
index (section 1), then deploy the site Worker (section 4). Cards are rendered by
`scripts/next_og_images.py` (resumable; `--only slug`).

## Rollback

```bash
cd next/site && npx wrangler delete
```

(or remove the `archive.aero/*` route in the dashboard). `/` is the Pages viewer
again at once; nothing in R2 or on Pages needs undoing. A visitor who installed
the root service worker gets the live page on the next navigation (the worker
passes 404/410 through and unregisters on 410).

## After the switch (not part of the cutover)

- Retire `beta.archive.aero/next/` by serving its document with 410 for a few
  weeks (next/HOSTING.md); keep `charts-beta` until then.
- The durable per-chart keys `sectionals/chart/<slug>/<date>` (extension-less)
  still hold the August warps: refresh them from the reslice with
  `scripts/publish_chart_pmtiles_from_reslice.py` (worklist 06) and republish
  `timeline_data.json`.
- `analytics_export.py` from 2026-10-02 on carries the /t/ rows; the heatmap
  places them by coordinates.
