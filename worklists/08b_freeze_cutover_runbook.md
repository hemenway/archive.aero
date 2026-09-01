# 08b — freeze + cutover runbook (Aug 29–30)

Companion to `08_atchistory_migration.md`. That document is the plan and the
record; this one is the **ordered command list** for the two live days, so
Saturday and Sunday are mechanical. Authored 2026-08-26 (buffer week) after a
pre-freeze health check. Where the two disagree, 08 governs — fix this file.

`BUILD=/Volumes/projects/atchistory_build`. All scripts carry `~/venv` shebangs
and run from the repo root.

---

## Pre-freeze health check — run 2026-08-26, all green

- All 7 `08a` backlink targets: **200, 0 hops**. `archive.aero/atc/` still
  `X-Robots-Tag: noindex, nofollow` (correct until Sunday 11:00).
- SSH to the box OK (`atchisto@74.220.207.111`, key `~/.ssh/atchistory_hostmonster`).
- **Zero content drift live-vs-crawl**: live wp-sitemaps identical to the frozen
  2026-08-09 snapshots (1,334 posts / 26 pages / 25 categories, byte-same URL
  sets, 0 lastmod changes) and **0 files** under `wp-content/uploads` newer than
  Aug 9. If this holds through Friday, freeze day is a re-verify, not a rebuild.
- `wrangler whoami` OK (missing `challenge-widgets.write` scope is irrelevant —
  R2 + deploy scopes present). Token expires every few days; expect to refresh
  Saturday (`npx wrangler login`).
- `/Volumes/projects` mounted; build tree intact (`crawl/ logs/ oldhost/ site/ static/`).

## T-minus (Thu Aug 27 – Fri Aug 28)

1. **Bing Webmaster Tools — the one open pre-cutover checklist item** (user,
   ~5 min): sign in, use **Import from Google Search Console**, confirm both
   domain properties (atchistory.org, archive.aero) appear. Needed for the
   Sunday 14:00 "Bing Site Move" step.
2. Optional, per `08a` pacing rules: 1–2 Wikipedia link repairs per day (COI
   disclosure on the user page first); dreamsmithphotos email. All of 08a stays
   valid after cutover too — the 301s make it optimization, not a dependency.
3. Freeze discipline: **no WP content edits after Friday night.**

---

## Freeze day — Saturday Aug 29

Order matters: logs → inventory → crawl → dump/uploads → maps → flatten →
**parity gate** → sync → verify. Parity must finish before Sunday's redirect
check (its dead-on-live tolerance reads `parity_live_cache.jsonl`).

### 0. Preliminaries

```bash
ls /Volumes/projects/atchistory_build          # volume mounted?
cd ~/archive.aero/worker-atc && npx wrangler whoami   # refresh: npx wrangler login
curl -s https://ifconfig.me                    # our IP for the log filter
```

- Record the IP. It was `76.251.45.110` on 2026-08-21 (residential — can
  rotate). Every log merge below must filter it, or ~30K of our own probe
  requests import as demand.
- **Snapshot the Aug-9 crawl before anything overwrites it** (`atc_fetch_wp.py`
  writes into `crawl/` in place; the old tree is the only copy of pre-freeze WP
  state):

```bash
cp -a /Volumes/projects/atchistory_build/crawl /Volumes/projects/atchistory_build/crawl_2026-08-09
```

### 1. Fresh pulls

1. **Raw logs** (3 weeks accrued since 08-09): cPanel → Raw Access, or pull the
   archived gz's over SSH into `$BUILD/logs/`. Filter our IP when merging.
2. **GSC accrued pages/queries** (user): export from both properties, drop
   beside the logs — inventory tail input.
3. **Inventory refresh** (reads live sitemaps + the new logs):

```bash
~/archive.aero/scripts/atc_url_inventory.py
```

4. **Re-crawl WP** (fetches all wp_* rows + homepage + `/feed/` + all 6
   `wp-sitemap*.xml` — the frozen-snapshot refresh is automatic), then
   pagination:

```bash
~/archive.aero/scripts/atc_fetch_wp.py
~/archive.aero/scripts/atc_fetch_wp.py --probe-pagination
```

5. **Drift gate** — diff fresh vs Aug-9 snapshots:

```bash
for s in wp-sitemap-posts-post-1.xml wp-sitemap-posts-page-1.xml wp-sitemap-taxonomies-category-1.xml; do
  diff <(grep -o '<loc>[^<]*' /Volumes/projects/atchistory_build/crawl_2026-08-09/$s) \
       <(grep -o '<loc>[^<]*' /Volumes/projects/atchistory_build/crawl/$s) && echo "$s: no URL drift"
done
```

   **If no drift** (expected, per the 08-26 check): p_map/route-map regen will
   be no-ops — still run them, but expect empty diffs and skip the worker
   redeploy. **If drift**: new slugs flow through steps 2.2–2.3 and the worker
   redeploys.
6. **Uploads delta** (08-26 preview said 0 files; verify and rsync if any):

```bash
ssh -i ~/.ssh/atchistory_hostmonster atchisto@74.220.207.111 \
  "find public_html/wp-content/uploads -type f -newermt 2026-08-09 | wc -l"
# if >0:
rsync -av -e "ssh -i ~/.ssh/atchistory_hostmonster" \
  atchisto@74.220.207.111:public_html/wp-content/uploads/ \
  /Volumes/projects/atchistory_build/static/wp-content/uploads/
```

   ⚠ Remember the volume is case-insensitive APFS — if rsync reports a
   same-name-different-case conflict, stop and handle it like §A4 (fetch both
   spellings origin-direct, upload straight to R2, extend the sync filter).
7. **Fresh DB dump** → regen the `?p=` map. Server-side mysqldump (creds are in
   `wp-config.php` on the box); fall back to a fresh UpdraftPlus backup zip if
   mysqldump is unavailable:

```bash
ssh -i ~/.ssh/atchistory_hostmonster atchisto@74.220.207.111 \
  'cd public_html && DB=$(grep DB_NAME wp-config.php | cut -d\x27 -f4) && U=$(grep DB_USER wp-config.php | cut -d\x27 -f4) && P=$(grep DB_PASSWORD wp-config.php | cut -d\x27 -f4) && mysqldump -u"$U" -p"$P" "$DB" wp_posts | gzip' \
  > /Volumes/projects/atchistory_build/logs/freeze-db-wp_posts-$(date +%F).gz
```

### 2. Maps + rebuild

1. `?p=` map from the fresh dump (default reads the stale 07-12 backup dump —
   **pass the path explicitly**), then diff against the worker copy:

```bash
~/archive.aero/scripts/atc_p_map.py /Volumes/projects/atchistory_build/logs/freeze-db-wp_posts-2026-08-29.gz
diff <(python3 -m json.tool ~/archive.aero/worklists/data/atc/p_map.json) \
     <(python3 -m json.tool ~/archive.aero/worker-atc/src/p_map.json) | head
```

   If changed: copy into `worker-atc/src/p_map.json`.
2. Canonical map (only produces changes if new slugs appeared):

```bash
~/archive.aero/scripts/atc_canonical_map.py
~/archive.aero/scripts/atc_canonical_map.py --verify   # expect 0 unresolved / 0 chains
```

3. **If (and only if) p_map or route_map changed**: worker tests + redeploy,
   still `MODE="serve"` + `ATC_NOINDEX="1"`:

```bash
cd ~/archive.aero/worker-atc && npm test && npm run deploy
```

4. **Re-fetch theme/plugin/wp-includes assets — run TWICE** (learned the hard
   way on the 08-31 freeze run: the fresh-crawl reset clears `crawl/`, which
   is where `atc_fetch_assets.py` had recovered these on Aug 9/13; skipping
   this surfaces as ~33 deletions in the sync dry-run. Twice because on a
   cold crawl dir the first pass misses url() refs inside stylesheets fetched
   in that same pass; expect run 2 to fetch the collapsing-category-list
   fonts and run 3 / the tail of run 2 to show `fetched: 0` + the 6 known
   losses):

```bash
~/archive.aero/scripts/atc_fetch_assets.py
~/archive.aero/scripts/atc_fetch_assets.py   # until "fetched: 0; failures: 6"
```

5. Full flatten rebuild (idempotent; merges crawl+static → `site/`, reruns all
   rewrite passes, installs the landing page with fresh Recent-additions from
   the new feed, refreshes `oldhost/`):

```bash
~/archive.aero/scripts/atc_flatten_rewrite.py
```

   ⚠ Known `rm -rf` flakiness on this volume: the script retries and moves
   stubborn dirs aside — read its output; don't pipe through `head`/`tail`
   without `set -o pipefail`.
6. Regenerate repo sitemaps from the fresh snapshots (`--verify` runs AFTER the
   R2 sync, step 4.4):

```bash
~/archive.aero/scripts/atc_gen_sitemaps.py
```

### 3. Parity gate

```bash
~/archive.aero/scripts/atc_parity_check.py --refresh-live --tag freeze
```

Expect **GREEN**: 0 FAIL, 0 recoverable closure breaks (exit 0). Policy SKIPs
and live-broken-too closure entries are normal. Fix anything real before
syncing. This run also refreshes `parity_live_cache.jsonl` for Sunday's
redirect check.

### 4. Publish to R2

1. **Dry-run with the filter — non-negotiable** (protects the 4 case-collision
   objects + `_oldhost/**`):

```bash
cd ~/archive.aero
rclone sync /Volumes/projects/atchistory_build/site r2:atc-site \
  --filter-from scripts/atc_r2_sync_filter.txt --checksum --dry-run 2>&1 | tee /tmp/atc_sync_dryrun.log
grep -ci 'delete' /tmp/atc_sync_dryrun.log   # expect 0 — investigate ANY deletion
```

2. Real sync: same command without `--dry-run`.
3. Refresh the frozen old-host sitemaps in R2 (`sync` deliberately skips
   `_oldhost/**`, so copy explicitly from the flatten's fresh `oldhost/`):

```bash
for f in /Volumes/projects/atchistory_build/oldhost/wp-sitemap*.xml; do
  rclone copyto "$f" "r2:atc-site/_oldhost/$(basename "$f")"
done
```

4. Post-sync verify:

```bash
~/archive.aero/scripts/atc_parity_check.py --base prod --limit 600 --tag freezeprod
~/archive.aero/scripts/atc_gen_sitemaps.py --verify    # all URLs 200, 0 hops
rclone lsl r2:atc-site/History/FacilityPhotos/NE/north_platte/north_platte_fss1973.jpg  # collision survivor present
```

   Note: worker edge cache holds HTML up to 1h — a just-republished page can
   serve stale for that long; that's expected, not a parity failure (spot-check
   with a cache-busting query if in doubt).

### 5. Wrap

- Tick the three freeze-day boxes in 08 (workstreams A, B) + log the session.
- Confirm Sunday prereqs: Bing verified (T-minus item 1), GSC both properties,
  fresh parity cache on disk, wrangler token fresh.
- Old site is now hands-off.

---

## Cutover day — Sunday Aug 30 (CDT; each step independently reversible)

The §4 table in worklist 08 governs; this is the same plan with exact commands.

### 09:00 — atchistory.org onto the new stack (same URLs, MODE=serve)

> **Run 2026-08-31. Two corrections from the live run (08 governs; recorded
> here so a re-run is right first time):**
> 1. **Deploy the routes BEFORE the DNS flip** (steps 2–3 before step 1). The
>    routes are inert while the records are grey, so the flip becomes one
>    transition instead of a window where CF proxies to Incapsula for nothing.
> 2. **The noindex trap (fixed in `dc46fb4f`).** `staging` in `src/index.js`
>    meant "any host that is not archive.aero", so routing atchistory.org here
>    made the live indexed site serve `X-Robots-Tag: noindex, nofollow` — and
>    the 11:00 `ATC_NOINDEX="0"` flip could not clear it (the expression
>    short-circuits on `staging`). Verify explicitly after the flip:
>    `curl -sI https://www.atchistory.org/ | grep -i x-robots` → **no output**,
>    while `archive.aero/atc/` and `atc-staging` must still show noindex.

1. CF dashboard, **atchistory.org zone → DNS**: flip `@` (both A records) and
   `www` to **orange-cloud**. (Universal SSL was issuing since 08-09 — after the
   flip confirm the cert serves: `curl -sI https://atchistory.org/ | head -1`.)
2. Uncomment the two host routes in `worker-atc/wrangler.toml` (lines are
   already there as comments):

```toml
  { pattern = "atchistory.org/*", zone_name = "atchistory.org" },
  { pattern = "www.atchistory.org/*", zone_name = "atchistory.org" },
```

3. Deploy (`MODE` still `"serve"`): `cd ~/archive.aero/worker-atc && npm run deploy`
4. Verify — top-100 inventory URLs 200 on the new stack:

```bash
head -101 ~/archive.aero/worklists/data/atc/url_inventory.csv | tail -100 | cut -d, -f1 | while read -r p; do
  printf '%s %s\n' "$(curl -s -o /dev/null -w '%{http_code}' "https://www.atchistory.org$p")" "$p"
done | sort | uniq -c | sort -rn | head
```

   (inventory is hit-sorted; adjust the column if the CSV order differs) plus a
   browser spot-check of the homepage + checklist page.
   **Rollback**: grey-cloud the three records → HostMonster serves again.

### 11:00 — /atc/ goes public

1. `worker-atc/wrangler.toml`: `ATC_NOINDEX = "0"` → `npm run deploy`.
2. Push the repo commit (GH Pages publishes robots/sitemaps/nav). The staged
   set: `robots.txt`, `sitemap.xml`, `sitemap-core.xml`, `sitemap-atc.xml`,
   `index.html`, `about.html`, `contribute.html`, `sources.html`, `README.md`,
   `URI-POLICY.md`, `.gitignore`, `worker-atc/`, `scripts/atc_*.py` +
   `scripts/cf_bulk_redirects_core.csv`, `worklists/08*.md`.
   ⚠ The working tree also carries **unrelated** dirty files
   (`scripts/slicer.py`, worklists 02/04/07/09/10, other new scripts) — commit
   the migration set explicitly by path, don't `git add -A`.
3. Verify:

```bash
curl -sI https://archive.aero/atc | grep -i '^HTTP\|location'      # 301 → /atc/
curl -sI https://archive.aero/atc/ | grep -i '^HTTP\|x-robots'     # 200, NO noindex
for u in robots.txt sitemap.xml sitemap-core.xml sitemap-atc.xml; do
  curl -s -o /dev/null -w "%{http_code} $u\n" "https://archive.aero/$u"; done   # all 200
```

   **Rollback**: revert commit; `ATC_NOINDEX="1"` redeploy.

### 13:00 — flip the old hostnames to redirect

1. `worker-atc/wrangler.toml`: `MODE = "redirect"` → `npm run deploy`.
2. Full verification (the rehearsed harness, now against production):

```bash
~/archive.aero/scripts/atc_redirect_check.py --tag cutover
```

   Expect 0 FAIL (08-20 rehearsal: 16,218 checks / 0 FAIL; NOTEs on junk/
   wp-internal buckets are normal). Also re-run the 08a step-3 loop plus its
   old-URL forms — one-hop 301s.
   **Rollback**: `MODE = "serve"` redeploy (seconds).

### 14:00 — search engines

- GSC: **Change of Address** atchistory.org → archive.aero (domain properties;
  if the tool balks at the subdirectory target, proceed — 301s carry the move).
- Bing: **Site Move** (needs the T-minus verification).
- GSC archive.aero property: submit `https://archive.aero/sitemap.xml`.
- Confirm the frozen old-URL sitemaps serve on the old host:
  `curl -s -o /dev/null -w '%{http_code}\n' https://www.atchistory.org/wp-sitemap.xml` → 200.

### EOD

- Snapshot CF dashboards (both zones).
- First unmatched-404 sweep from `atc_logs`; append any real misses to the
  redirect map (redeploy is cheap).
- Log the day in worklist 08.

## Week 1 after (from 08 §5)

Daily: GSC coverage on both properties ("Page with redirect" should grow),
`atc_logs` 404 feed → map updates, CF zone analytics. No content/design changes
to `/atc/` until rankings stabilize (~60 days). Registrar untouched — the
transfer window analysis in 08 §5 says on/after ~Oct 29.
