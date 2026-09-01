# 08 — atchistory.org → archive.aero/atc/ migration

**Target cutover: Sunday 2026-08-30.** Plan written 2026-08-09 from recon of
`/Volumes/projects/atchistory_backup` (full cPanel copy, 2026-07-13) and the live site.

Goal: fold atchistory.org into archive.aero under the `/atc/` path prefix, eliminate
WordPress (full static flatten), 301 every old URL one-hop to its new home, keep SEO
intact, near-zero downtime. Old hosting (HostMonster + SiteLock) decommissioned ~Oct 1.

---

## Locked decisions

- Destination: **`https://archive.aero/atc/…`** — a *designed* canonical URI space, with
  every historical path aliased into it by permanent 301. **Superseded the original
  "exact path preservation" decision on 2026-08-10** (see §A3): preserving the old
  paths verbatim would have made archive.aero inherit atchistory.org's URI mistakes
  forever, and this corpus has already lost one generation of URLs that way (the ~2020
  WP migration killed `/History/checklst.htm`, the Wikipedia citation target). Doing it
  at cutover costs nothing; doing it later costs a second redirect event on freshly
  moved rankings.
- WordPress is removed: the site becomes fully static (HTML + assets in R2).
- Serving: new **`atc` worker** + new **R2 bucket `atc-site`** (mirrors `tiles`/`charts`
  conventions: Analytics Engine dataset, observability block, range support).
- Redirects: done **in the `atc` worker** (hostname-based mode), not CF redirect rules —
  unlimited map size, versioned in git.
- atchistory.org DNS moves to the existing Cloudflare account (free zone). Domain stays
  registered **forever** (expiry 2028-07-15 at Bluehost; renew/transfer later).
- Content freeze: **Sat 2026-08-29**. Live site is the source of truth for WP content
  (site is active — backup is fallback only); static trees + uploads come from the disk
  backup with a freeze-day delta pass.

## Approved decisions (all approved 2026-08-09)

| # | Decision | Approved outcome |
|---|----------|------------------|
| D1 | battcave.net | Stays on HostMonster exactly as-is (own zone + NS, untouched by this migration); the hosting account is **never cancelled** and Bluehost access continues unchanged |
| D2 | @atchistory.org mail | **✅ COMPLETE 2026-08-09 (~21:30): Cloudflare Email Routing live and round-trip tested.** Old SpamExperts MX + duplicate SPF TXTs removed; CF MX/SPF/DKIM added; `archive@` + catch-all forwarding. `mail` A record kept — IMAP door to the old mailbox for the history export (do the Mail.app export this week; old messages stay on the box until deleted). Sending-as deferred. battcave.net mail unaffected |
| D3 | ads.txt / AdSense | Drop — rewrite pass strips the AdSense code present on 678 crawled pages; ads.txt does not migrate |
| D4 | Pre-move notice banner on old site | None — redirects speak for themselves |
| D5 | Frozen WP comments | Keep visible as static text |
| D6 | Old `/forum/*` backlink URLs (dir gone from disk for years) | 410 tombstone page linking to `/atc/` |

---

## 1. Recon summary (2026-08-09)

- **Live + growing**: ~17.4K visits / 14K uniques (Jun 2026), 3× YoY. Referrers include
  en.wikipedia.org (Checklist, Air safety, B-17), AOPA, Facebook. Weekly UpdraftPlus
  backups ran through 2026-07-12 (DB dumps in `wp-content/updraft/`).
- **Content generations**: FrontPage static (`History/` 1.1 GB / 2,750 files, `pdf/`
  1.1 GB / 869 files, `classphotos/` 13 MB, `video/` 5.8 MB, `Images/`, `Masters/`) +
  WordPress at root (**1,334 posts, 26 pages, 25 categories**, root-level slugs,
  `wp-content/uploads/` 1.3 GB). Total real content **≈ 3.6 GB**. `/forum/` long gone.
- **Top URLs** (all-time hits): `/History/checklst.htm` 49K,
  `/historical-maps/u-s-historical-airway-maps/historical-maps-2/` 37K,
  `/History/FacilityPhotos/index.htm` 33K, `/facility-photos/` 32K,
  `/flight-service-class-photos/` 28K, `/classphotos/PhotoHome.htm` 18K,
  `/History/lighthouse.htm` 14K. Top downloads: Airway Pioneer PDF, Western/Eastern
  Airway Beacons `.xls` (spaces in filenames), early_communications.pdf, FAA World PDFs.
- **Stack**: cPanel on HostMonster behind SiteLock/Incapsula CDN; PHP 7.4; plain WP
  `.htaccess` (no legacy redirect rules); **not** multisite. Malware-cleanup leftovers
  on disk (`deleteme.wp*.php`, `scrapper.php`) — never copy these forward.
- **DNS/registrar**: NS at hostmonster; apex A → Incapsula (45.60.x), www CNAME →
  sitelockcdn. Registrar Bluehost. MX → SpamExperts (×3) → HostMonster; SPF + 2×
  globalsign TXT. `archive@atchistory.org` mailbox active into 2026.
- **GSC**: legacy HTML-file verification present (`google994516164ab8ab88.html`) — keep
  serving it forever.
- **archive.aero side**: apex = GH Pages origin behind CF (proxied); `data.archive.aero`
  = `tiles` worker + R2 `charts`. **No robots.txt or sitemap exist yet.** Worker route on
  `archive.aero/atc/*` intercepts only that prefix; everything else passes to GH Pages.

## 2. Target architecture

```
archive.aero/*            → CF → GH Pages (unchanged: viewer, about, contribute…)
archive.aero/atc/*        → CF route → atc worker → R2 atc-site      (SERVE)
atchistory.org/*          → CF route → atc worker                    (REDIRECT 301)
www.atchistory.org/*      → CF route → atc worker                    (REDIRECT 301)
atc-staging.archive.aero  → atc worker (SERVE + X-Robots-Tag: noindex)  [pre-cutover]
```

`atc` worker behavior by hostname:
- **archive.aero** (path `/atc/…`): strip prefix → R2 key lookup with URL-decode
  (space/percent filenames), directory index (`index.htm`, then `index.html`,
  `Default.htm`), correct Content-Type (no charset override on `.htm` — pages are
  windows-1252 with meta tags), real 404 (custom page, correct status), HEAD + Range
  support (reuse `tiles` range code), Cache-Control (HTML 1h; binaries 1d edge-long),
  Analytics Engine logging (`atc_logs`: path, status, referer) — unmatched-404 feed.
- **atchistory.org / www**: mode var `REDIRECT`:
  1. Exceptions served 200 from R2: `/google994516164ab8ab88.html`, `/robots.txt`
     (permissive + `Sitemap:` pointing at frozen old-URL sitemaps), frozen
     `/wp-sitemap*.xml` snapshots (lists **old** URLs → crawlers re-fetch → see 301s).
  2. 410 Gone (+ tombstone HTML): `/wp-admin/*`, `/wp-login.php`, `/xmlrpc.php`,
     `/wp-cron.php`, `/wp-json/*`, `/forum/*` (D6), per-post comment feeds.
  3. Query-string maps: `/?p=N`, `/?page_id=N` → 301 `/atc/<slug>/` (map generated from
     the freeze-day DB dump); `/?s=…` → 301 `/atc/`.
  4. `/feed/` → 301 `/atc/feed/` (frozen static copy of final RSS).
  5. Default: **301 `https://archive.aero/atc<path>`** — one hop from any of
     http/https × www/apex; query strings dropped unless mapped.

## 3. Workstreams

### A. URL inventory & mapping  → `worklists/data/atc/url_inventory.csv` (gitignored dir)

- [x] `scripts/atc_url_inventory.py` (2026-08-09): merges live `wp-sitemap-*`,
      awstats `report_data.json` content+downloads, cPanel raw logs, and the
      static + uploads tree walks. Columns:
      `old_path, old_url, new_path, class, hits, sources`. **v1 = 14,400 rows.**
      Rerun after pagination probe + at freeze.
- [x] Raw logs (2026-08-09): archiving enabled in cPanel Raw Access; current +
      2020-era logs downloaded to `atchistory_build/logs/` and merged into the
      inventory — **197 log-only paths** exposed the pagination + autoindex gaps
      below, live ads.txt polling, and the browsable `/backup/` dir (excluded from
      migration; tidy up on the live box someday).
- [x] At freeze: pull **fresh** raw logs (3 weeks archived by then) + GSC accrued
      pages/queries; append tail to inventory. (Backup's `access-logs/` is a dead
      symlink — awstats aggregates were the only history.)
      **DONE 2026-08-31 (freeze pass ran a day late, night of Aug 30→31):**
      Aug-2026 archives + the Aug-30 live tail pulled over SSH, merged, own-IP
      filtered (**31,906 lines removed** — IP re-checked, still 76.251.45.110;
      pristine copies in `logs/originals/`). Inventory rebuilt: **15,524 rows**,
      802 log-only discoveries (bot probes — bucketed, see the freeze log below).
      GSC pages/queries export skipped (user item, optional tail input only).
      `atc_url_inventory.py` gained the origin-pin DNS override — Incapsula now
      challenges the browser-UA sitemap fetch that worked on Aug 9.
      **⚠ Filter our own IP (76.251.45.110) from the log merge** — found
      2026-08-21 during the cutover-date validation: the Aug 9–29 archive
      window contains ~30K requests of our own origin-direct tooling (crawl,
      pagination probe, parity live probes, incl. deliberately-probed dead
      paths); unfiltered, the inventory would import our own probes as
      real demand. (Re-check the IP on freeze day — it's a residential
      assignment and can rotate.)
- [x] ~~12-month GSC baseline export~~ — **not possible**: no GSC account existed
      (2026-08-09); Search Console has no pre-verification history. Baseline instead:
      awstats/monthly_traffic.csv (through 2026-07, already mined) + whatever GSC
      accrues between verification and Aug 30. Verify both domain properties ASAP —
      the hard requirement is verified ownership for Change of Address, not history.
- [x] `scripts/atc_p_map.py` (2026-08-09): `post_id → path` map from the 07-12 DB
      dump → `worklists/data/atc/p_map.json`, **1,360 published ids** (exactly
      matching sitemap counts) incl. 3 hierarchical page chains — nested-page bug
      caught in spot-check (`?page_id=10225` → the full
      `/historical-maps/u-s-historical-airway-maps/historical-maps-2/` path).
      Regenerate from the freeze-day dump.

### B. Static flatten (WP removal)  → build at `/Volumes/projects/atchistory_build/`

The backup stays pristine (rawtiffs discipline). Build dir layout:
`crawl/` (raw wget), `static/` (rsync from backup), `site/` (merged + rewritten = upload set).

- [x] rsync static trees + uploads from backup → `static/` (2026-08-09, **3.5 GB**):
      `History Images Masters Search _borders _fpclass _overlay classphotos pdf
      video wp-content/uploads` + root files (favicon/ntqrfavicon/500.shtml/
      index.htm/google994516164ab8ab88.html); excluded `*.php* *.pl *.cgi cgi-bin
      error_log* .htaccess* _vti_* _private/` (FrontPage form-results = PII) +
      Thumbs.db/.DS_Store. `backup/` and `ads.txt` (D3) not copied.
- [x] Crawl live WP (2026-08-09): **1,394/1,394 pages, all HTTP 200**, 88 MB →
      `crawl/`. Incapsula JS-challenges HTML for plain clients, so
      `scripts/atc_fetch_wp.py` pins DNS to the origin box (74.220.207.111 — valid
      AutoSSL cert, SNI/Host unchanged). Docker-WP fallback never needed. Includes
      `/feed/` + all 5 `wp-sitemap*.xml` (refresh these at freeze).
- [x] Autoindex replacement (log-discovered requirement): the live site serves
      Apache directory listings that get real visitors (`/History/Pubs/faa_world/`)
      → `scripts/atc_gen_indexes.py` generated **545 static index.html listings**
      (breadcrumbs/sizes/dates); dirs with FrontPage `index.htm` untouched — worker
      index order `index.htm` → `index.html` keeps curated pages winning.
- [x] Pagination probe (log-discovered: `/category/.../page/N/` absent from
      sitemaps; facilities runs past page/100): `atc_fetch_wp.py
      --probe-pagination` launched 2026-08-09 evening; **done same night (473
      pages) and folded in — box belatedly ticked 2026-08-21 after
      verification**: 473 pagination `index.html` files in `site/`, 474
      `/page/` inventory rows (exactly §A3's count), inventory total 14,849,
      `facilities/page/100` present. Covered by the parity GREEN runs since.
- [x] Freeze-day delta (Aug 29): re-crawl WP HTML + pagination, fresh uploads
      delta (UpdraftPlus uploads zip / cPanel zip / SSH rsync), fresh feed +
      sitemap snapshots, fresh DB dump → regen `p_map.json`.
      **DONE 2026-08-31:** Aug-9 crawl snapshotted (`crawl_2026-08-09`), full
      fresh re-crawl 1,868/1,868 (two timeout retries) + 473 pagination pages;
      drift gate **clean** (0 URL / 0 lastmod changes vs Aug 9); uploads delta
      **0 files**; DB dump = UpdraftPlus's own weekly backup dated freeze day
      (Aug 30 10:59, pulled over SSH — mysqldump recipe not needed); `p_map`
      regen **byte-identical** (1,360 ids). Full record: §"Freeze pass" below.
- [x] `scripts/atc_flatten_rewrite.py` (2026-08-09, byte-safe): merged
      crawl+static → `site/` (**3.6 GB**, 2,515 text files). Rewrites: 177,319
      host refs + 3,014 percent-encoded (oEmbed/share params) + 19 root-relative;
      6 old-URL sitemaps parked in `oldhost/` unrewritten. Strips per D3: AdSense
      from all 1,861 WP pages. Hygiene: **0 eval-obfuscation hits, only external
      script host = archive.aero**; residuals = 3,799 mailto (kept deliberately) +
      17 prose mentions. Report: `worklists/data/atc/rewrite_report.txt`. Rerun
      after freeze-day delta (idempotent full rebuild).
- [x] `/atc/` landing page **live 2026-08-09**: replaces WP homepage; archive.aero
      design language + de-AI conventions (Barlow, dark solid surfaces, no
      gradients/emoji; fresh web-research pass corroborated the house rules —
      left-aligned asymmetric layout, ruled lists not uniform card grids, real
      numbers in copy). 1954AirlineRoutes banner (the old masthead artifact),
      Start-here (7 real top collections), category rail, collection facts,
      can-you-help (archive@atchistory.org), chart-viewer cross-link, footer
      provenance. Source versioned `worker-atc/static/index.html`;
      `atc_flatten_rewrite.py install_landing()` copies it into `site/` AFTER
      the rewrite pass (root links must not gain /atc) + injects Recent
      additions (8 posts) from the frozen feed — freeze-day rebuild refreshes
      automatically (`--landing-only` for quick refresh). Uploaded to R2.
- [x] 404 + 410 tombstone pages (2026-08-09): shared `pageShell()` in the worker —
      self-contained inline HTML (render even if R2/fonts are down), house tokens,
      plane mark, mailto backstop on the 404. Real 404/410 statuses preserved;
      styled 404 verified live on archive.aero/atc/.

### C. Worker + R2  → new repo dir `worker-atc/`

- [x] Bucket `atc-site` created; **15,223 objects / 3.554 GiB uploaded** via the
      pre-existing rclone `r2:` remote (2026-08-09).
- [x] `worker-atc/` built + **deployed 2026-08-09** (tiles conventions: R2 binding,
      `atc_logs` Analytics Engine, observability, `.npmrc ignore-scripts` deploy
      recipe). Serve mode live; redirect mode implemented but inert
      (`MODE="serve"`), incl. `?p=`/`page_id` map (1,360), legacy bonus map (12),
      410 tombstones, old-host robots.txt + frozen `_oldhost/` sitemaps.
- [x] Staging live: **https://atc-staging.archive.aero** (custom domain,
      `X-Robots-Tag: noindex`). Smoke tests all green: WP pages, pagination,
      spaces-in-name `.xls` (correct MIME), generated listings, `/feed/` as
      rss+xml, real 404s, Range→206, extensionless→trailing-slash 301 staying
      on-host (bug caught + fixed pre-deploy).
- [x] Charset trap #2 caught live: rclone stored `.htm` as `text/html;
      charset=utf-8` (Go MIME table) which would override windows-1252 meta tags
      → worker now strips charset on `.htm`/`.shtml` so the page's own
      declaration wins. Verified on staging.
- [x] **Asset-closure incident (2026-08-09/10 night)**: user reported unstyled
      pages → crawl was HTML-only and the backup rsync covered only `uploads/` —
      theme (escapade child of primer) / plugin / wp-includes css+js were never
      captured. Fix: `scripts/atc_fetch_assets.py` (parses all crawled pages,
      fetches missing same-host assets origin-direct, second pass for url() refs
      inside CSS) → 29 files; 6 known-losses (retired escapade-era genericons
      fonts + HTML-entity false alarms whose real `&` files were already on
      disk). Merge+sync re-run. Related fix: `rm -rf` flakiness on the projects
      volume made one rebuild die mid-delete while `| tail` masked the failure —
      rclone then mirrored a partial tree to R2 (briefly); script now retries +
      moves stubborn dirs aside, and sync chains use `set -o pipefail`.
- [x] **`archive.aero/atc/*` route deployed EARLY (2026-08-09) behind
      `ATC_NOINDEX="1"`** — rewritten pages reference final absolute URLs, which
      404'd on GH Pages pre-route; deploying the production route (noindexed)
      makes staging fully real and pre-tests cutover. Viewer untouched (verified).
      **Cutover 11:00 step is now just: set `ATC_NOINDEX="0"` + redeploy + repo
      commit (robots/sitemap/nav).**
- [x] Rendering verified end-to-end on archive.aero/atc/: header photo
      (1954AirlineRoutes.jpg), theme fonts, jQuery 3.7.1 executing. NOTE: the
      Claude-Code browser pane blocks classic **cross-origin** scripts (no SRI)
      — on atc-staging the archive.aero-hosted JS appears dead *in the pane
      only*; real browsers run it fine. Judge visuals via archive.aero/atc/ in
      the pane, or staging in a normal browser.
- [ ] Cutover-day remaining: add both atchistory.org host routes; flip
      `MODE="redirect"` at 13:00 step; `ATC_NOINDEX="0"` at 11:00 step.

### D. Parity & redirect verification  → reports in `worklists/data/atc/`

- [x] `scripts/atc_parity_check.py` **built + first full run 2026-08-09**: every
      inventory row, live (origin-direct, throttled 8 rps, Incapsula-challenge
      abort guard) vs staging. Text: status + `<title>` equality + length band
      (FAIL >max(6KB,20%)); binaries: **exact size vs live AND vs local site/
      AND md5 vs R2 ETag**; generated listings pass on marker; policy buckets
      (gone/junk/wp_internal/feed_secondary/legacy_ghost/host_meta) keep
      by-design absences out of the failure set. Plus **recursive asset
      closure** (all subresources of every served page + css url()/@import
      chains — 4,930 unique refs) with live-verification of breaks: only
      worse-than-live counts as FAIL. Live responses cached
      (`parity_live_cache.jsonl`) so fix-loop reruns only re-hit the new stack.
      Reports: `parity_report.csv`, `parity_failures.csv`,
      `asset_closure_failures.csv`, `parity_summary.txt` (+ run logs).
      **Run 1 (14,849 rows): 14,125 PASS / 602 policy-SKIP / 121 INFO /
      1 FAIL** (`/readme.html` = WP version-disclosure file → policy-excluded,
      never migrated). Closure: 100 broken refs, **all 100 broken on live too**
      (dead `example-location/` FrontPage-template images, retired escapade
      genericons, one .tif upload) — bug-for-bug parity, 0 recoverable. Fixes
      landed en route: 4 collapsing-category-list fonts recovered (live GETs
      406'd on HEAD — mod_security; `atc_fetch_assets.py` hardened with
      Accept:*/* + scans ALL css not just newly-fetched), worker now sends
      content-length on HEAD, `.DS_Store`/`Thumbs.db` junk purged from R2 +
      stripped in future merges. **Run 2 (post-fix): 14,125 PASS / 603 SKIP /
      121 INFO / 0 FAIL, closure 0 recoverable — GREEN (exit 0).** Prod-route
      sample (`--base prod`, top 600 rows by hits): 583 PASS / 0 FAIL — the
      /atc prefix path verified end-to-end too (reports tagged `_prod`).
      Harness hardening from the loop: live-cache loader keyed by
      method+path (reruns were silently re-crawling live), closure
      live-verification always-on, `--tag` for parallel report sets,
      progress prints flushed.
      Eyeballed in a real browser render (house rule): landing, checklist
      post, facility-photos, classphotos/PhotoHome.htm (1252 charset),
      faa_world generated listing — all fully styled; spaces-in-name `.xls`
      byte-exact with Excel MIME; big-PDF Range → 206.
- [x] `scripts/atc_redirect_check.py` **built + rehearsed 2026-08-20** (cutover-day
      run at the 13:00 step still pending): every row — assert **single-hop 301** →
      exact route_map target, query dropped → target 200 with no further hop.
      Matrix: http/https × www/apex; ALL 1,360 `?p=`/`page_id` shortlinks; 410 set;
      exception URLs (old robots, GSC file, 6 frozen sitemaps) 200; never-frozen
      sitemap variant must 404 not 500. Expected-target model is a Python port of
      routes.js + redirectOldHost — keep in lockstep. Preflight sentinel refuses to
      run against the old stack (Incapsula/cf-ray/robots-body check; caught
      serve-mode in the negative test). Target 404s tolerated only for junk/
      wp-internal buckets or dead-on-live rows (per parity_live_cache — run
      freeze-day parity first). Offline model (`--offline`): 60,807 checks,
      **0 would-chain targets**. Full rehearsal against the real worker in
      `wrangler dev` redirect mode (`.dev.vars` + `--local-upstream`; wrangler's
      dev proxy rewrites Host, so one host per instance): see session log.
      Findings fixed en route: `?p=` handler bypassed the drop map (5 junk ids
      chained twice), `?page_id=156` → `/home/` minted a dead URL (live 301s
      /home/ → /; drop seeded in the generator, route map regenerated: 187
      drops), and a never-frozen `/wp-sitemap-*.xml` variant crashed the worker
      on a null R2 response (500 → now 404). Worker redeployed `84dedb15`,
      route tests 8/8 (new test pins every shortlink to one hop).

### E. DNS, zone, mail

- [x] Enumerate current records from authoritative NS (done 2026-08-09). **Zone-import
      checklist — all grey-cloud, replicate exactly** (incl. the duplicate SPF pair —
      technically invalid but status quo; SPF gets rebuilt at the mail flip):

      | Name | Type | Value |
      |------|------|-------|
      | @ | A | 45.60.22.252 |
      | @ | A | 45.60.25.252 |
      | @ | MX 10/20/30 | mx.spamexperts.com / fallbackmx.spamexperts.eu / lastmx.spamexperts.net |
      | @ | TXT | `v=spf1 a mx include:websitewelcome.com ~all` |
      | @ | TXT | `v=spf1 +a +mx +ip4:67.20.112.112 +ip4:74.220.207.111 +include:hostmonster.com ~all` |
      | @ | TXT ×2 | globalsign-domain-verification=C70CB721… / =2840600C… |
      | www | CNAME | 8heirpl.sitelockcdn.net |
      | mail | A | 74.220.207.111  ← IMAP host for the week-2 export |
      | ftp, webmail, cpanel | CNAME | atchistory.org |
      | smtp, pop, imap | CNAME | mail.atchistory.org |
      | autodiscover, autoconfig, webdisk | A | 74.220.207.111 |
      | default._domainkey | TXT | `k=rsa; p=MHwwDQYJKoZIhvcNAQEBBQADawAwaAJhAMbdzCBh1UQx…` (full value via `dig +short @ns1.hostmonster.com TXT default._domainkey.atchistory.org`) |
      | _dmarc | — | none — leave absent |

      battcave.net confirmed **live** (Incapsula 107.154.x + spamexperts MX) — per
      D1 it stays untouched. wrangler OAuth on this machine verified working (incl.
      email_routing scope).
- [x] Add atchistory.org zone to CF → NS flipped at HostMonster (2026-08-09).
      All records grey-cloud; **propagation confirmed same evening** (1.1.1.1 and
      8.8.8.8 both return blair/gannon.ns.cloudflare.com); site serving unchanged
      through Incapsula; CF Universal SSL issuing well before cutover.
- [x] GSC: **domain properties verified for atchistory.org AND archive.aero
      (user, 2026-08-21)** — the hard prerequisite for the cutover 14:00
      Change of Address; every pre-cutover day now accrues search data.
      Legacy URL-prefix property kept.
- [ ] Bing Webmaster: both domains (needed for the 14:00 "Bing Site Move"
      step). Easiest path now: Bing Webmaster Tools offers one-click **import
      from GSC**, which pulls both freshly-verified domain properties — no
      DNS records needed.
- [x] Mail (per D2 as approved — receiving moves to CF Email Routing).
      **CLOSED 2026-08-21 — workstream complete.** Steps 2–3 (Email Routing
      live + round-trip test) were done 2026-08-09. Step 1, the mailbox
      history export, is **moot**: user checked the old HostMonster
      `archive@atchistory.org` mailbox — **it is empty; there is nothing to
      export.** No maildir backup needed; nothing of value remains on the old
      box. The `mail` A record's only remaining purpose (the IMAP door for
      this export) is now spent — it can go whenever the old box gets its
      optional tidy-up (§5), no urgency. Original procedure, for the record:
      1. Export the mailbox: add `archive@atchistory.org` to Mail.app via IMAP
         (server `mail.atchistory.org`, SSL port 993, cPanel mail password) and
         copy everything to a local On-My-Mac mailbox. Backup maildir = 2nd copy.
      2. CF dashboard → atchistory.org → Email → Email Routing: add + verify the
         destination inbox, create `archive@` → forward, enable catch-all →
         forward, accept the automatic MX/SPF record swap (SpamExperts MX retire;
         legacy DKIM TXT can stay, harmless).
      3. Round-trip test from an outside account → confirm the forward arrives.
      After step 2 the HostMonster mailbox stops receiving (keeps old mail until
      deleted). battcave.net mail is in its own zone — unaffected.
- [x] Import fix — resolved 2026-08-09 via the corrected import file (proxy-scan
      trap avoided: CF's scan had defaulted 13 A/CNAME records to Proxied, which
      would have broken IMAP/cPanel/FTP at the flip and double-proxied the site
      through Incapsula). Zone verified all-DNS-only by direct query against
      gannon.ns.cloudflare.com.
- [x] Service CNAMEs `cpanel`/`ftp`/`webmail` → `mail.atchistory.org` — done in
      the corrected import (they aliased the apex, which becomes the redirect
      worker at cutover); verified live on CF NS.
- [x] CF-scan extras kept, DNS only: `whm` A, `localhost` A (cPanel artifact),
      `_autodiscover` SRV (port corrected to 443), `_cpanel-dcv` TXT,
      `_domainkey` `"o=~"`.
- [x] The two records CF's scan missed — `battcave` + `www.battcave` A
      74.220.207.111 (cPanel addon-domain aliases) — included in the corrected
      import; `battcave.atchistory.org` verified resolving via gannon 2026-08-09.
      Import file archived: `worklists/data/atc/atchistory.org.cloudflare-import.txt`.

### F. archive.aero side (repo)

- [x] `robots.txt` **staged 2026-08-20** (uncommitted, cutover 11:00 commit):
      allow all, `Sitemap: https://archive.aero/sitemap.xml`.
- [x] `sitemap.xml` + children **staged 2026-08-20** (same commit):
      `scripts/atc_gen_sitemaps.py` → `sitemap.xml` (index) → `sitemap-atc.xml`
      (**1,379 canonical URIs** — frozen wp-sitemaps posts/pages/categories
      mapped through route_map, lastmod preserved, landing added; users +
      one-entry post_tag sitemaps excluded by §A3 policy) + `sitemap-core.xml`
      (/, about, contribute, sources — an index can't hold bare URLs, hence the
      second child). `--verify` asserts every URL 200 with 0 hops on prod.
      **Rerun the generator + --verify after the freeze-day sitemap snapshot**,
      before the 11:00 commit.
- [x] Nav link to `/atc/` (2026-08-09): index.html menu-panel item "ATC History
      Collection" (Feather radio stroke icon, matches existing pattern) +
      about.html "The ATC History collection" section. **Edits staged
      uncommitted in the working tree** — pushing to main publishes via GH
      Pages, which stays scheduled for the cutover 11:00 commit.
- [x] **Core pages canonicalized extensionless + URI covenant (2026-08-21,
      staged uncommitted for the same 11:00 commit).** New repo-root
      `URI-POLICY.md` promotes the §A3 cool-URI rules to site law (address
      plan, new-collection checklist, redirect inventory); CLAUDE.md points at
      it. The site's own four pages now obey the "no extensions on documents"
      rule *before* their first-ever sitemap submission: `rel=canonical` +
      `og:url` → `/about` `/contribute` `/sources`, every internal link
      extensionless (viewer header/menu/pin-popup/sources-link, all three
      footers, ATC landing source, README), `CORE_PAGES` in
      `atc_gen_sitemaps.py` updated, sitemaps regenerated + `--verify` green
      (all 1,383 URLs 200 / 0 hops — GH Pages serves extensionless natively).
      Nav restructure rides along: viewer menu panel regrouped
      (**Collections**: Sectional Charts, ATC History / **The Archive**: About,
      Sources, Contribute, GitHub — Sources was missing entirely) and a shared
      slim header (wordmark → `/` + flat site links, marker-commented for
      sync) replaces the hub-and-spoke "← Back to the map" link on
      about/contribute/sources. Landing-page `.html` link fix reaches R2 at
      the freeze-day rebuild (or `--landing-only` sooner).
- [x] **CF dashboard — DONE + VERIFIED LIVE 2026-08-21.** `www` DNS record
      added; `scripts/cf_bulk_redirects_core.csv` uploaded as a **Bulk
      Redirect List** (5 rows: www→apex with subpath + path-suffix + query
      preserved, plus the four exact `.html`→extensionless aliases) and the
      account-level Bulk Redirect Rule enabled. Enumerated exact URLs, no
      wildcards → nothing can intercept `/atc/*` ahead of the worker
      (supersedes the wildcard Single-Redirect plan). Verified end-to-end:
      9 alias probes all single-hop 301 to exact canonicals — both schemes on
      www (http://www → https apex in ONE hop, no Always-Use-HTTPS chain),
      query strings preserved incl. viewer deep-link params; all 4 canonical
      targets 200/0-hop; /atc/ untouched (landing, article, generated
      `…/index.html` listing all 200 through the worker).
- [x] CLAUDE.md section added (2026-08-09): atc-site/build-dir/never-in-git,
      private mail tree, worker deploy recipe, parity harness pointers.
      README note still pending.

## 4. Timeline

### Week 1 — Aug 10–16: foundations
Status 2026-08-09 (started a day early): zone + NS flip **done & propagated**;
decisions D1–D6 **all approved**; inventory v1 **done** (14,400 rows); crawl +
rsync + autoindex listings + `?p=` map **done**; pagination probe running.
Rewrite/merge/scan done + Email Routing live (round-trip tested) same night.
Remaining: GSC/Bing properties verified (user — **last user item of week 1**);
mailbox history export via IMAP (user, casual deadline); worker serve-mode on
`atc-staging` (noindexed); first parity run. rclone `r2:` remote verified for
the bulk upload; SSH to hosting box verified.

### Week 2 — Aug 17–23: parity + mail + rehearsal
Parity loop → green. Mail (D2) complete if not already done in week 1: export →
Email Routing → round-trip verified. Landing/404/410 pages. Redirect + `?p=` maps
finalized. Sitemaps + robots.txt staged in repo.
**Full cutover rehearsal on staging**: flip staging worker to redirect mode against a
test hostname, run `atc_redirect_check.py`, flip back.

### Week 3 — Aug 24–29: buffer + freeze
**Runbook: `08b_freeze_cutover_runbook.md`** — the freeze-day + cutover-day
steps below consolidated into one ordered command list (authored 2026-08-26).
Mon–Fri: spillover fixes only; no new scope. **Sat Aug 29 — freeze**: final WP crawl
delta (re-crawl HTML; it's ~1,400 pages), fresh uploads delta, final DB dump (regen
`?p=` map), frozen feed + wp-sitemap snapshots, fresh access-log pull → inventory
tail, final parity run, `rclone sync` to prod bucket, then hands off the old site.

### Freeze pass — executed night of Aug 30→31 (a day late; cutover day TBD)

The planned Sat-freeze/Sun-cutover weekend slipped; the freeze pass ran
overnight 2026-08-30 ≈23:00 → 01:10 CDT per `08b`, ending **GREEN end-to-end**.
Zero content drift held (live sitemaps byte-identical to Aug 9, 0 lastmod
changes, uploads delta 0), so the pass was the expected re-verify. R2 is now
the frozen freeze-day build; the old site is hands-off. Findings worth keeping:

- **`/feed` bare-spelling regression (real bug, caught by the worker tests):**
  the fresh log tail put a bare `/feed` row into the inventory; the canonical
  generator's guard only excluded `/feed/`, so the site feed became a 410 drop.
  Guard now excludes both spellings (`atc_canonical_map.py`).
- **Site-feed serialization variants** (`/feed/atom/` etc., log-discovered):
  now 301 → `/atc/feed` (same resource, one hop — not a 410). Implemented in
  `routes.js` (+ new test, 9/9) and ported in lockstep to **both** checker
  models (`atc_redirect_check.py`, `atc_parity_check.py::expected_canonical`).
  Worker redeployed `16780d59` (serve mode, noindex intact).
- **802 log-only paths** (3 more weeks of bot traffic) needed policy buckets:
  `.well-known/*`, WP year-archive guesses (`/2017/`–`/2025/`), `/BLOG/` case
  probes → junk; bare `/wp-content/`+`/uploads/` autoindexes → wp_internal;
  `wp-sitemap-index.xsl` → host_meta; `/feed/atom/` → new feed_variant bucket.
  Regexes updated in both checkers (they do not share code — keep in lockstep).
- **The dry-run deletion gate earned its "non-negotiable":** first dry-run
  wanted to delete 33 files — the theme/plugin/wp-includes assets
  `atc_fetch_assets.py` had recovered into `crawl/` on Aug 9/13, which the
  fresh-crawl reset (crawl dir cleared for the re-crawl) had discarded.
  Parity could not see it (closure checks staging = R2, which still held
  them). Re-ran fetch-assets — **twice**: on a cold crawl dir its first pass
  misses url() refs inside stylesheets fetched in that same pass (rerun-hiding
  bug, same family as 2026-08-09) — 29+4 files recovered, 6 known-losses
  unchanged, second dry-run 4 deletions, third **0 deletions**. Step added to
  `08b`. (`atc_fetch_wp.py` has no force flag; clearing `crawl/` is the
  re-crawl mechanism, and fetch-assets must follow every time.)
- Numbers: inventory 15,524 rows; route map 14,648 canonical URIs / 1,959
  aliases / 597 drops / 0 unresolved / 0 chains; flatten 2,541 text files,
  rewrite counts identical to the Aug-10 baseline; **parity freeze run:
  14,119 PASS / 1,325 SKIP / 80 INFO / 0 FAIL, 0 canonical drifts, closure 8
  broken all live-broken-too**; sync 2,401 files, 0 deletions; `_oldhost/`
  sitemaps refreshed; post-sync prod sample 416 PASS / 0 FAIL; sitemaps
  verify 1,383/1,383 at 200/0 hops; collision survivor confirmed in R2.
- Fresh `parity_live_cache.jsonl` on disk for the cutover-day redirect check.
  Wrangler token fresh. **Still pending for cutover: Bing Webmaster
  verification (user), then the 09:00/11:00/13:00/14:00 sequence on the new
  cutover day.**

### Cutover day — Sunday Aug 30 (each step independently reversible)

| T | Step | Verify | Rollback |
|---|------|--------|----------|
| 09:00 | Flip atchistory.org + www to orange-cloud, add worker routes, MODE=**serve** (hosting move, same URLs) | Top-100 inventory URLs 200 on new stack; spot browser check | Grey-cloud DNS back → HostMonster serves again |
| 11:00 | Add `archive.aero/atc/*` **and bare `archive.aero/atc`** routes (both already in wrangler.toml); drop staging noindex; push repo commit (robots, sitemaps, nav link) | `/atc/` + samples 200; **`curl -sI /atc` → 301 to `/atc/`**; robots/sitemap fetch | Remove route; revert commit |
| 13:00 | Flip MODE=**redirect** on atchistory hosts; deploy | `atc_redirect_check.py` full pass (one-hop 301s, 410s, exceptions) | MODE=serve redeploy (seconds) |
| 14:00 | GSC Change of Address atchistory.org → archive.aero (domain props; homepage 301 lands inside the target property, which satisfies the check — if the tool balks, proceed on 301s alone); Bing Site Move; submit `sitemap-atc.xml`; confirm frozen old-URL sitemaps fetchable | GSC accepts; sitemaps "Success" | CoA is revertible in GSC for 180 days |
| EOD | Snapshot dashboards; first unmatched-404 sweep from `atc_logs` | — | — |

## 5. Post-cutover

- **Daily (week 1)** → weekly: GSC coverage + "Page with redirect" both properties;
  `atc_logs` unmatched 404s → append to redirect map (redeploy is cheap); CF zone
  analytics (server-side — no client JS needed; add GA later only if wanted);
  rank-watch top queries: `atchistory`, `pilot checklist`, `faa history`, `atc history`.
  Expect impressions wobble 2–8 weeks; clean 1:1 301s recover.
- **Do not** redesign, retitle, or restructure `/atc/` content until rankings
  stabilize (~60 days). Move first, improve later.
- **Decommission (per D1/D2): nothing gets cancelled.** The HostMonster account
  stays indefinitely — it hosts battcave.net (site + mail) and blog.battcave.net;
  Bluehost access stays exactly as before (D1). @atchistory.org *receiving* moves
  to CF Email Routing (D2); the old HostMonster mailbox keeps its history as an
  archive until exported + deleted. SiteLock stays too (battcave's A records run
  through Incapsula). atchistory.org simply stops pointing at the account.
  Optional tidy-up later: remove atchistory web files + the browsable `/backup/`
  dir from public_html. Keep the Bluehost **registration**; optional calm transfer
  to Cloudflare Registrar post-stabilization. **The domain never lapses** — the
  Wikipedia/AOPA backlinks are the crown jewels.
- Backlog (post-stabilization): Pagefind client search over `/atc/`; design
  integration pass; Wikipedia citation URL updates (careful, COI-aware — 301s make
  this optional); mixed-content sweep of old pages; domain transfer.
- **Domain-transfer timing constraints (verified 2026-08-21 against CF Registrar
  docs + PIR RDAP; the post-stabilization date already satisfies all of them):**
  1. *45-day renewal rule (real, financial):* transferring within 45 days of the
     previous owner's renewal means "the registry does not add the extra year even
     though you paid for the transfer" (CF FAQ) — you pay twice for one year. The
     renewal date is NOT in public records (RDAP shows only expiry 2028-07-15);
     if it processed on the 2026-07-15 anniversary, day 46 = Aug 30 — zero margin
     on an unverifiable assumption; if it processed any later (recon saw the 2028
     expiry by Aug 9), the window runs as late as ~Sep 23. **Pin the exact date
     from the Bluehost renewal invoice before initiating.**
  2. *60-day WHOIS-change lock (ICANN, likely binding):* RDAP "last changed" =
     **2026-08-11** — if that was a registrant/contact update (the handover), no
     transfer completes before **~2026-10-10**; only the NS flip wouldn't trigger
     it, but Bluehost's implementation decides. Also: any future registrant-info
     edit restarts the 60 days — don't touch WHOIS contacts within 60 days of the
     intended transfer.
  3. Mechanics when the time comes: clear `clientTransferProhibited` at Bluehost,
     get the EPP/auth code, confirm registrant email is reachable; CF transfer
     adds +1 yr → expiry 2029-07-15 at wholesale.
  **Net: transfer on/after ~Oct 29 (the existing gate) clears every window with
  margin; cutover day Aug 30 involves NO registrar action — keep it that way.**

## 6. Risk register

| Risk | Mitigation |
|------|------------|
| Incapsula blocks the flatten crawl | Local WP resurrection from DB dump (fallback path, tested in week 1 only if needed) |
| Missed URLs → 404s post-cutover | Inventory from 5 sources; default prefix rule catches unknowns that exist in R2; `atc_logs` 404 feed + weekly map updates |
| Duplicate content pre-cutover | Staging noindexed until 11:00 step; `/atc/` route not public before Aug 30 |
| Charset mojibake on 1252-era pages | Byte-level rewrite; no charset in worker Content-Type for `.htm`; parity eyeball |
| Soft-404s (SEO poison) | Worker returns real 404/410 statuses; parity asserts |
| Redirect chains | Worker 301s straight to final `https://archive.aero/atc/...` from every host/scheme variant; checker asserts one hop |
| Mail disruption | D2 flip is user-controlled and instant (CF Email Routing swaps MX): export mailbox BEFORE enabling; round-trip test immediately after; battcave MX in its own zone, untouched. (Proxy-scan trap already caught + fixed pre-NS-flip 2026-08-09) |
| GSC CoA rejects subdirectory move | 301s carry the move regardless; CoA is accelerant, not requirement |
| Old site changes after freeze | Freeze Sat + cutover Sun (24 h window); WP edits stop at freeze |
| Worker/R2 outage risk to main site | `atc` is a separate worker — `tiles` and GH Pages untouched by any failure |
| wp-content junk republished | Only `uploads/` is copied; PHP excluded wholesale; injected-script scan gate |

### A3. Canonical URI scheme — **APPROVED by user 2026-08-10** (full scope)

Framework: **W3C "Cool URIs don't change"** (<https://www.w3.org/Provider/Style/URI>),
applied at the user's direction. Old URLs must keep working; the new archive.aero ones
must make sense in 20 years. That means two spaces, not one.

**What the doc flagged in the 14,849-row inventory** (all real counts):

| Rule | Hits | Example |
|------|------|---------|
| no software mechanism in the path | 8,822 | `/wp-content/uploads/2016/10/…` |
| no subject classification | 217 | `/category/class-photos-archive/class-photos/unknown_fss-photos/` |
| no author name | 132 | `/author/admin/page/107/` |
| no status/collision markers | 141 | `historical-maps-2` — the site's **#2 page, 37,184 hits** |
| no file extensions on documents | 78 | `.htm`, `.shtml` |
| no technology leaks | 560 | `_borders/`, `_fpclass/`, `_overlay/` (FrontPage) |
| — | 474 | `/page/N/` pagination machinery |

WP had already done the good part: 0 doc slugs carry spaces or uppercase. The damage is
in the **prefixes and wrappers**, so this is prefix surgery + ~145 individual calls, not
a 1,334-way re-slug.

**The two spaces.** Canonical = the designed, permanent addresses; sitemaps and
`rel=canonical` point here and nowhere else. Alias = every historical path from both
generations, permanent 301 into canonical. **Alias entries are never deleted — that is
the whole policy.** R2 keys stay the unchanged old paths: the doc's own prescription is
to decouple the URI from storage, not rename storage to match.

```
/atc/<slug>                     articles — flat, extensionless, no trailing slash
/atc/topics/<name>[/<N>]        was /category/<a>/<b>/<c>/[page/N/]
/atc/archive/<N>                was /page/N/  (and /author/*/page/N/)
/atc/media/YYYY/MM/file.jpg     was /wp-content/uploads/…      (1 rule, 8,822 paths)
/atc/assets/{themes,plugins,lib}/  was /wp-content/…, /wp-includes/
/atc/history/…  /atc/class-photos/…  /atc/library/…  /atc/images/…  /atc/video/…
```

**Stated boundaries** (in the generator docstring so they don't drift):
- **Documents get the full treatment now; asset paths get prefix normalization only.**
  Deeper mixed-case components (`/atc/history/Pubs/`) stay. Principled, not lazy:
  document URIs are what people link and what ranks, so changing them later costs a
  second redirect on ranked URLs; asset paths are referenced only from our own pages,
  so normalizing them later is free. Deferring the free half keeps the cutover diff small.
- Format extensions on real files (`.pdf`, `.xls`, `.jpg`) stay — the doc objects to
  `.html`/`.cgi` because they name the *server's* technology, not the artifact's format.
- FrontPage scaffolding (`_borders`, `_fpclass`, `_overlay`, `Search`) keeps serving at
  its old paths — pages reference it — but never enters canonical space or a sitemap.
- **No date partitioning of articles**, the one thing the doc says to keep IN: WP post
  dates are 2016+ bulk-import timestamps, not authorship dates, so `/2016/` would encode
  noise. Re-check against the freeze-day DB dump before this is final.
- Collision suffixes: stripped only where free (22); 102 kept because a sibling still
  holds the bare slug or the stem is a bare post id (`/atc/11769-2` — `/atc/11769` would
  read as an internal identifier, the same disease). The kept list is **published** in
  `route_map.json` and asserted by the route test, not hidden in an exception.

**Built + verified 2026-08-10:**
- [x] `scripts/atc_canonical_map.py` → `worker-atc/src/route_map.json` (versioned):
      **14,497 canonical URIs, 1,952 alias + 1,952 key entries, 182 drops, 10 prefix
      rules.** `--verify` resolves every canonical URI against the build tree and checks
      for chains: **0 unresolved, 0 would chain.** Report:
      `worklists/data/atc/canonical_report.txt` (publishes every unstrippable suffix,
      every legacy target it re-pointed, every path absent from the build).
- [x] A2's 12 legacy targets **re-canonicalized** — they were written in the old
      `/atc/<old path>` shape and would each have cost a second hop. The Wikipedia
      checklist target is now one hop: `atchistory.org/History/checklst.htm` →
      `archive.aero/atc/how-the-pilots-checklist-came-about`.
- [x] `worker-atc/src/routes.js` — shared pure resolution (`canonicalOf`,
      `keyCandidates`, `routeOldPath`) used by BOTH hostnames, so `atchistory.org/X` and
      `archive.aero/atc/X` agree and a stale link costs exactly one hop from either
      direction. `?p=`/`page_id` shortlinks now resolve through the map too.
- [x] `worker-atc/test/routes.test.mjs` — **7/7 green.** Asserts one-hop resolution,
      tombstones, canonical-serves-itself, no chains, and that canonical space contains
      no `wp-*`, `/category/`, `/author/`, `/page/N`, `.htm`, space, or collision suffix.
      **Caught a real bug**: `/atc/feed` (the canonical frozen RSS) returned 410.
- [x] `wrangler deploy --dry-run` clean: 315 KiB / **60.6 KiB gzip**.
- [x] `atc_flatten_rewrite.py` canonical passes so pages ship permanent URIs rather than
      aliases (otherwise every image on every page costs a 301): **154,080 plain +
      1,357 pct-encoded + 2,604 relative** rewrites. The relative pass exists because
      FrontPage pages link each other relatively — ~54 curated pages, 582 such links.
      Host-pass counts unchanged from the parity-green baseline (177,319 / 3,014 / 19).
      Ordering bug caught + fixed: drops must beat the prefix rules.
- [x] Landing page (`worker-atc/static/index.html`) links canonicalized — all 16 plus
      the `og:image` and CSS banner URL, which still pointed at `wp-content`.
      `install_landing()` now runs the canonical pass as a backstop.

**Completed 2026-08-10 (evening) — canonical scheme is LIVE end-to-end:**
- [x] `atc_parity_check.py` taught the canonical map: `check_canonical` asserts every
      row lands on exactly the URI `route_map.json` promises in ≤1 hop (0 tolerance for
      chains/drifts), plus a new **re-point policy bucket** — rows the map deliberately
      sends to a *different* resource (134 `author_collapse`, drop-map junk pages →
      home) assert landing-exists + exact-landing instead of live-content equality,
      which is definitionally not expected there. Two harness bugs fixed en route:
      report writer crashed on the new `canon` column; first analysis run surfaced the
      139 re-points as title FAILs (all 139 accounted for by the policy, 0 unexplained).
- [x] Worker **deployed** (version `828f4dad`, serve mode, ATC_NOINDEX=1 intact, no
      atchistory routes) BEFORE the content sync — R2 keys are the unchanged old paths,
      so canonical routing resolves against the old bucket content, letting the full
      parity gate run pre-sync as the doc required. Route tests 7/7 + spot-checks of
      every rule family (uploads→media, category→topics, page→archive, .htm extension
      drops, feed, 410s) all one-hop.
- [x] **Pre-sync gate run GREEN** (run 3): 13,986 PASS / 742 policy-SKIP / 121 INFO /
      **0 FAIL, 0 canonical chains/drifts**, closure 0 recoverable.
- [x] Pre-canonical R2 text tree backed up first (last copy anywhere):
      `~/archive.aero-attic/atc-site-pre-canonical-text_2026-08-10/` (2,547 files, 128 MB).
- [x] `rclone sync` site/ → `atc-site`: checksum mode, **2,459 text files uploaded,
      0 deletions** (dry-run verified; `_oldhost/` excluded/protected, junk excluded).
- [x] **Post-sync full re-run GREEN** (run 4): identical counts, 0 FAIL, 0 canonical
      problems, closure 12 broken all live-broken-too (bug-for-bug). Prod-route
      top-600 sample GREEN: 583 PASS / 0 FAIL / 0 canonical problems. Browser eyeball:
      checklist article + landing page fully styled at canonical URIs on
      archive.aero/atc/ (banner via /atc/media/, theme via /atc/assets/).

**Still open:**
- [x] Workstream F sitemap must list **canonical URIs only** — done 2026-08-20
      (`atc_gen_sitemaps.py` maps the frozen wp-sitemaps through route_map.json;
      1,379 canonical URIs, verified 200/0-hop).
- [ ] Consider whether `/atc/history/Pubs/` etc. should be lowercased after all (free
      to do post-stabilization, per the boundary above).

### A4. Backlink sweep + completeness audit — **2026-08-13**

Triggered by the pre-cutover inbound-link repair (see `08a_backlink_repair.md`).
Three independent sources were cross-checked against the new stack: Wikimedia
`list=exturlusage` across 16 projects, the Wayback CDX index (6,598 distinct
atchistory.org URLs), and a full `find` over the live box. Findings, in
descending severity:

**1. Case-collision data loss — FOUND AND REPAIRED. This is the one that mattered.**
`/Volumes/projects` is case-**insensitive** APFS (verified: writing `FOO.txt`
after `foo.txt` overwrites the content and keeps the first name). The live Linux
box has four directories holding two files whose names differ only by case, so
the 2026-07-13 backup copy and every build derived from it collapsed each pair
into a single file. **The "pristine" cPanel backup is on the same volume and is
equally collapsed — it is not a recovery source for this class.**

The live `find` is definitive: exactly **4 groups / 8 files**, all under
`History/FacilityPhotos/NE/`. One is a genuinely different photograph
(`north_platte_fss1973.jpg`, 29,932 B ≠ `North_Platte_FSS1973.jpg`, 30,110 B);
the other three are byte-identical content stored under a second spelling. All
four missing spellings were re-fetched from the origin and uploaded to R2 under
their exact keys, md5-verified against live, twins confirmed unclobbered.

> **Freeze-day hazard.** These four objects cannot exist in `site/`, so a plain
> `rclone sync` deletes them. The sync must now run with
> `--filter-from scripts/atc_r2_sync_filter.txt` (new, versioned, documents the
> why inline). Dry-run and confirm 0 deletions before the real sync.

Independent confirmation: the Wayback CDX diff surfaced the same 4 paths and no
others, from a completely different direction.

**2. Full live-vs-build completeness diff — clean apart from the above.**
15,394 live static-tree files compared case-exactly against the build: 2,602
absent, of which 2,598 are plugin/theme *source* files never intended to
migrate (composer vendor trees, `.po`/`.mo` translations, build scripts,
LICENSE/README) and **4 are the collapse victims above**. No other content gap
exists anywhere in the tree.

**3. Parity-harness false negative — ✅ FIXED 2026-08-20.**
`asset_closure_failures.csv` classified 12 broken refs as "live-broken-too"
(→ 0 recoverable → GREEN). Re-probing live directly shows **4 of the 12 are
live-200**: the collapsing-category-list webfonts
(`font-awesome-subset-collapscatlist.{eot,svg,ttf,woff}`). They were genuinely
missing from build and R2 — a regression of the 2026-08-09 recovery that did not
survive the 08-10 canonical rebuild. Re-fetched, md5-verified, uploaded; all four
now serve 200 at both the canonical URI and the legacy alias with correct font
MIME types. The remaining 8 are genuinely live-broken (bug-for-bug, correct).
**Root cause (found + fixed 2026-08-20):** closure assets are reported at their
NEW-stack URL, which lives in §A3 canonical space (`/atc/assets/plugins/…`), and
the live probe used that canonical path verbatim — a URL live never served
(`/wp-content/plugins/…` is the live spelling). Anything under a renamed prefix
(`/assets/*`, `/media/*`) was therefore guaranteed "live-broken-too"; the three
old-shape paths in the 12 were probed correctly, which is why only the fonts were
misclassified. Fix: `live_probe_paths()` reverse-maps canonical → old through
route_map.json (mirroring `keyCandidates()` in routes.js — keep in lockstep), the
verdict requires every candidate to fail, and the probed live path is now a
report column (`live_path`). Verified by replaying all 12 recorded failures:
4 fonts → live-200/FAIL-recoverable, 8 → live-broken-too at the correct URLs;
checklist-slice rerun green with correct probes. A GREEN closure result is
trustworthy again.

**4. Dead FrontPage wrapper pages — pre-existing, no action.** 111 log-requested
and 267 Wayback-known `/History/**.htm` + `/classphotos/**.htm` per-photo wrapper
pages 404 on the new stack **and 404 on live** (killed by the ~2020 WP
migration). Bug-for-bug parity; the underlying JPGs survive and the directory
listings serve. Only the three with live inbound links were mapped (§A2).

**5. Non-issues, checked and dismissed.** WordPress answers 200 for trailing-
hyphen slug variants (`/how-the-pilots-checklist-came-about-/`, `--/`, …) and
301s for truncated ones — an unbounded family of soft duplicates that WP itself
disowns via `rel=canonical`. The new stack's 404 is the better behavior; nothing
to map. Seven `/category/*/feed/` URLs go 200→410 by existing design.

### A2. Legacy bonus redirects — **APPROVED by user 2026-08-10, deployed**

High-traffic URLs that **404 on the live site today** (killed by the ~2020 WP
migration) now 301 to living successors — the worker beats the live site.
Final map (12 entries, in `worker-atc/src/legacy_map.json` +
`worklists/data/atc/legacy_redirect_candidates.csv`; every target verified 200):
- `/History/checklst.htm` (49K hits, Wikipedia "Checklist" target) →
  `/atc/how-the-pilots-checklist-came-about/`
- `/History/FacilityPhotos/index.htm` → `/atc/facility-photos/`
- `/History/lighthouse.htm` → `/atc/category/early-radio-years/light-houses/` (user pick)
- `/History/Maps/Maps.htm` → **archive.aero homepage** (the chart viewer — user
  pick; old maps page traffic feeds the merged property's centerpiece)
- `/History/nightnav.htm` → `/atc/category/early-radio-years/`
- `/History/FacilityPhotos/FSS_Dir/CurrentFSS_Directory.htm` → `/atc/facility-directories/`
- `/index.htm` (old FrontPage homepage) → `/atc/`
- `/forum/` + `/forum/index.php` → 410 tombstone (D6)
- 3 × FAA World vol9 PDFs → **resolved as renames, not losses**: the ~2020 reorg
  moved flat volume-named files into year dirs with month names; vol9
  no7/9/11 ≙ `1979/faa_world_{jul,sep,nov}_1979.pdf` (all 12 months present on
  disk). 301s map old→new names; Wayback recovery unnecessary.
SSH to the hosting box verified working 2026-08-09 — freeze-day pulls (logs,
uploads delta, server-side DB dump) are fully scriptable. **Corrected 2026-08-13:**
the key is `~/.ssh/atchistory_hostmonster` (not `archive_aero_migration`, which
does not exist), and you must connect to **`atchisto@74.220.207.111`** — the host
keys in `known_hosts` are recorded against the IP, so the hostname form dies with
"Host key verification failed". Verified working:

```bash
ssh -i ~/.ssh/atchistory_hostmonster atchisto@74.220.207.111 "cd public_html && find . -type f | wc -l"
```

**Map grew to 16 entries on 2026-08-13** (see §A4) — `/History/fsshist.htm` plus
three `Summit_RadioBeacon*_WY.htm` photo pages, all dead-on-live, all still
carrying live inbound links. Every one of the 16 re-verified end-to-end after
deploy: 14 × one-hop-301→200, 2 × 410, 0 problems.

## 7. Session log

- 2026-08-09: recon + plan authored. Next: week-1 tasks.
- 2026-08-09 (later): CF zone created + records imported (scan complete — caught 5
  records beyond the dig sweep). Import review: 13 records defaulted to Proxied →
  set DNS-only before NS flip; cpanel/ftp/webmail CNAMEs repointed to
  mail.atchistory.org. D1 + D2 resolved: battcave and ALL mail stay on HostMonster;
  hosting account is permanent; decommission section rewritten.
- 2026-08-09 (evening): corrected zone imported from user's zone file (TTL 300 on
  apex/www, SRV port fix, battcave addon aliases); **NS flipped at HostMonster,
  propagation in progress** — zone verified correct + fully DNS-only by querying
  gannon.ns.cloudflare.com directly. Week-1 build started: inventory v1 = 14,203
  URLs (`worklists/data/atc/url_inventory.csv` via `scripts/atc_url_inventory.py`);
  Incapsula JS-challenges HTML for cookie-less clients → crawler fetches
  **origin-direct** (74.220.207.111, valid AutoSSL cert; `scripts/atc_fetch_wp.py`);
  full crawl (1,394 pages) + static/uploads rsync running. Live WP is v7.0.2;
  canonical host confirmed `https://www.atchistory.org`.
- 2026-08-09 (later): HostMonster zone-file export reviewed → 2 missing records
  found (battcave, www.battcave A). Corrected import file generated with all fixes
  baked in (repointed CNAMEs, SRV port 443, quoted TXTs, TTL 300 on apex/www):
  `worklists/data/atc/atchistory.org.cloudflare-import.txt` (copy in ~/Downloads).
  Procedure: bulk-delete scanned records → import file with "Proxy imported DNS
  records" UNCHECKED → expect 28 records, all DNS-only → then NS flip.
- 2026-08-09 (night): **NS propagation confirmed** (both 1.1.1.1 + 8.8.8.8 →
  blair/gannon). **All decisions D1–D6 approved by user**; D2 revised: receiving
  moves to CF Email Routing (export-first procedure in workstream E). Build: crawl
  finished **1,394/1,394 all-200** (origin-direct); raw logs merged → 197 new
  paths → pagination probe launched + **545 autoindex listing pages generated**;
  `p_map.json` built (1,360 ids, nested-page fix); AdSense found on 678 pages →
  strip per D3; browsable `/backup/` dir noted for live-box tidy-up. Doc synced
  to approved decisions.
- 2026-08-10 (early am): **Parity harness + design pass session.**
  `atc_parity_check.py` written (class-aware live-vs-new + recursive asset
  closure with live verification of breaks); full run over all 14,849 rows →
  **one real failure in the whole inventory** (`/readme.html`, policy-excluded)
  + 4 recoverable plugin fonts found & restored (mod_security 406-on-HEAD had
  hidden them from `atc_fetch_assets.py`; script hardened). 100 closure breaks
  all live-broken-too (bug-for-bug). Junk purge: 24 `.DS_Store`/`Thumbs.db`
  objects out of R2; flatten now strips them. Worker: content-length on HEAD;
  designed 404/410 pages (shared shell, real statuses). **Landing page live**
  (worker-atc/static source, feed-injected recent posts via
  `install_landing()`, 1954AirlineRoutes banner, verified desktop+mobile).
  Nav link staged uncommitted (index menu-panel + about section) for the
  cutover 11:00 commit; CLAUDE.md + README migration notes added. Browser
  eyeball: 5 top pages fully styled. **Parity loop closed GREEN same
  session**: run 2 = 0 FAIL / 0 recoverable closure breaks across all
  14,849 URLs; prod-route top-600 sample green too. Week-2's "parity loop →
  green" plus the landing/404/410 design items are done ~a week early;
  remaining week-2 scope = redirect-map finalization, sitemaps/robots
  staging, cutover rehearsal.
- 2026-08-10: **Canonical URI scheme (§A3)** — user directed applying W3C "Cool URIs
  don't change" to the old→new mapping; approved full scope + "strip collision
  suffixes only where free". Supersedes the exact-path-preservation decision.
  Built `atc_canonical_map.py` (14,497 canonical URIs, verify: 0 unresolved / 0
  chains), `worker-atc/src/routes.js` + 7 route tests (green; caught `/atc/feed`
  → 410), canonical + relative rewrite passes (158K refs), landing page + A2 legacy
  targets re-pointed. Real defects the review loop caught and fixed: 4 sibling posts
  stripping to one slug, `index.htm` → `/x/index` instead of the directory, numeric
  post-id slugs "fixed" into bare-id URIs, legacy targets left in pre-canonical shape,
  drops losing to prefix rules, 473 real archive pages nearly dumped to the home page,
  and a case-insensitive scan flagging the canonical `/atc/history/` as an alias.
  **Nothing synced to R2 or deployed** — parity harness still needs canonical
  awareness, and a green re-run gates the sync.
- 2026-08-10 (evening): **Canonical scheme shipped.** Order honored the doc's gate:
  worker deployed first (canonical routing works against unchanged R2 keys), full
  canonical-aware parity GREEN as the pre-sync gate, attic backup of the pre-canonical
  text tree, then rclone sync (2,459 files, 0 deletions) + post-sync full GREEN +
  prod-route sample GREEN + browser eyeball. Harness learned the §A3 re-point policy
  (author collapse + drop-map junk assert exact-landing, not content equality); its
  `canon` report-column crash fixed. §A3 has two deferrables left (canonical-only
  sitemap = workstream F; Pubs/ lowercasing post-stabilization). Staging and
  archive.aero/atc/ now serve the designed URI space end-to-end, noindex guards
  still on, atchistory.org untouched.
- 2026-08-13: **Backlink repair + completeness audit** (§A4, `08a_backlink_repair.md`).
  Enumerated every Wikimedia link to atchistory.org (13 links / 10 pages across
  16 projects) and resolved each through the worker; wrote the COI-aware edit
  runbook. The survey turned up 4 dead-on-live URLs with live inbound links →
  legacy map 12 → **16 entries**, route map regenerated (186 drops, verify 0
  unresolved / 0 chains), tests 7/7, **worker deployed** (version `9e5a24f7`,
  MODE=serve + ATC_NOINDEX=1 intact), all 16 re-verified end-to-end.
  Then swept for further errors from two independent directions — Wayback CDX
  (6,598 URLs) and a full `find` over the live box — which converged on the
  **case-insensitive-APFS collapse**: 4 files present on live, absent from the
  build *and* from the supposedly-pristine backup. Recovered to R2 by exact key,
  md5-verified. Also caught a **parity-harness false negative** that had been
  reporting 4 live-200 webfonts as "live-broken-too" (recovered those too — a
  regression of the 08-09 fix that did not survive the 08-10 rebuild). New
  `scripts/atc_r2_sync_filter.txt` because a plain freeze-day `rclone sync` would
  delete all four recovered objects plus the 6 `_oldhost/` sitemaps — proven:
  10 deletions without the filter, **0 with it**. Doc fix: the SSH key is
  `atchistory_hostmonster` and must connect by IP.
- 2026-08-09 (late night): pagination probe done (**473 pages**, facilities →
  p.100); flatten rewrite complete (177K host refs + 3K pct-encoded; AdSense
  stripped; hygiene scan clean); **legacy ghost-URL discovery** — 12 top URLs
  404/bounce on the live site, mapped to living successors incl. the Wikipedia
  checklist target (`legacy_redirect_candidates.csv`). **Email Routing live +
  round-trip tested (D2 ✅)**. SSH key authorized + connection verified
  (host2108.hostmonster.com). R2 upload done (15,223 obj / 3.554 GiB); **atc
  worker deployed; staging live at atc-staging.archive.aero** — smoke tests
  green; two charset/redirect bugs caught + fixed. Week-1 build track is
  essentially done four days early. Next: parity harness (D), landing + 404/410
  design pass, staging eyeball, GSC/Bing (user).
- 2026-08-20: **Pre-freeze gate items closed** (first movement since 08-13; freeze
  in 9 days). (1) §A4 harness bug **fixed + proven**: closure live-verification
  now reverse-maps canonical→old via `live_probe_paths()` (lockstep with
  routes.js `keyCandidates()`), verdict requires every candidate to fail, probed
  path published in a new `live_path` column; replay of the 12 recorded run-4
  failures gives exactly §A4's ground truth (4 fonts live-200, 8 live-broken at
  the CORRECT urls), checklist-slice rerun green. (2) **`atc_redirect_check.py`
  built** (workstream D): full matrix model ported from the worker, preflight
  old-stack sentinel, dead-on-live target tolerance from the parity cache.
  Offline: 60,807 checks, 0 would-chain. **Full rehearsal against the real worker
  in redirect mode via `wrangler dev`** (`.dev.vars` MODE=redirect +
  `--local-upstream`, one host per instance — the dev proxy rewrites Host, so
  the week-2 "test hostname on staging" rehearsal happened locally instead,
  production untouched): www all 14,849 rows = **16,218 checks 0 FAIL** with all
  14,660 distinct targets verified 200/no-hop against prod (133 NOTEs = the
  known junk/wp-internal 404s); apex top-2,000 = 3,380/3,380; negative test
  confirmed the preflight refuses a serve-mode worker. (3) Rehearsal-driven
  worker fixes, deployed `84dedb15`, tests **8/8** (new test: every `?p=` slug
  one-hop): `?p=` handler now routes through the full table (5 junk ids chained
  twice before), `/home/` seeded as a drop in `atc_canonical_map.py` → route map
  regenerated (**187 drops**, verify 0/0; live 301s /home/→/, and ?page_id=156
  had minted a dead /atc/home/), and null-R2 guards on the old-host exception
  paths (a never-frozen `/wp-sitemap-*.xml` probe 500'd the worker; now 404).
  (4) Workstream F **staged uncommitted for the 11:00 cutover commit**:
  robots.txt + `scripts/atc_gen_sitemaps.py` → sitemap.xml (index) +
  sitemap-core.xml (4 core pages) + sitemap-atc.xml (**1,379 canonical URIs**
  from the frozen wp-sitemaps through the route map, lastmod preserved, junk
  slugs collapsed; users/post_tag excluded per §A3) — `--verify`: all 1,383
  URLs 200 with 0 hops. Freeze-day reminder: regen sitemaps + `--verify` after
  the fresh sitemap snapshot, and run parity BEFORE the cutover redirect check
  (its dead-on-live tolerance reads the parity cache).
- 2026-08-21: **URI covenant + core-page canonicals + nav (workstream F,
  staged for the 11:00 commit).** `URI-POLICY.md` adopted at the repo root —
  §A3's rules promoted to site law: `/` is the viewer permanently; documents
  extensionless (`/about`); one prefix per collection; cross-collection
  entities top-level; lowercase canonical in new spaces; aliases and keys
  permanent; viewer query params (`?date=&lat=&lng=&zoom=`) declared API.
  Core four pages flipped to extensionless canonicals with every internal
  link updated (viewer, three static pages, ATC landing source, README);
  `atc_gen_sitemaps.py` CORE_PAGES updated; sitemaps regenerated, `--verify`
  all 1,383 URLs 200 / 0 hops. Viewer menu panel regrouped
  (Collections / The Archive) with Sources & Attribution added; shared slim
  header replaces "← Back to the map" on the three static pages. Worklist 09
  amended pre-build: `/airports/kbos` + `/sectionals/chart/<slug>/<date>`
  (the latter mirrors the R2 artifact key — page and artifact share one
  identifier). Same day, user added the `www` record and uploaded+enabled the
  Bulk Redirect List (`scripts/cf_bulk_redirects_core.csv`, 5 exact-URL rows —
  no wildcards, so `/atc/*` can't be intercepted); verified live: 9 probes all
  one-hop 301 to exact canonicals with queries preserved, 4 targets 200/0-hop,
  /atc/ pass-through confirmed. Workstream-F alias layer is fully live —
  nothing about it waits for cutover day.
- 2026-08-21 (later): **Cutover date validated against real data — Aug 30
  confirmed.** Calendar: Aug 29/30 verified Sat/Sun; next Sunday (Sep 6) is
  Labor Day weekend; 32-day rollback margin to the ~Oct 1 decommission.
  Traffic shape from the box's own logs (Aug 8–21 ssl log via SSH, our IP
  excluded): **no weekday structure at all** — human page views flat at
  ~120–260/day all seven days (Sun median 234 vs Mon 222 vs Wed 117), hourly
  curve flat 4–6%/hr through the 09:00–14:00 CDT runbook window, trough 3–6 am.
  So Sunday buys nothing traffic-wise *and costs nothing* — no better day
  exists; the date is carried by the operational constraints (24h freeze
  coupling, buffer week, GSC accrual since 08-21, Labor-Day avoidance,
  decommission margin). En route: the raw Sunday numbers were 14× inflated by
  our own 08-09 recon crawl (30,655 req from 76.251.45.110) — filter added to
  the freeze-day log-pull item above.
- 2026-08-21 (later): **Mailbox export closed as moot — workstream E complete.**
  User checked the old HostMonster `archive@atchistory.org` mailbox: empty,
  nothing to export. With Email Routing already live + round-trip tested
  (2026-08-09), all of workstream E is done; the `mail` A record's IMAP-door
  purpose is spent (optional cleanup with the §5 tidy-up, no urgency).
- 2026-08-26: **Buffer-week health check (all green) + runbook authored.**
  Verified: all 7 `08a` backlink targets 200/0-hop; `/atc/` noindex still on;
  SSH to the box OK; wrangler token OK; build volume intact. **Zero content
  drift** live-vs-crawl: live wp-sitemaps byte-identical to the frozen 08-09
  snapshots (1,334/26/25, no lastmod changes) and 0 uploads newer than Aug 9 —
  freeze day should be a re-verify, not a rebuild. Freeze+cutover steps
  consolidated into `08b_freeze_cutover_runbook.md` (ordered commands, hazards
  inline: crawl-dir snapshot before re-crawl, own-IP log filter, sync filter
  dry-run, explicit `_oldhost/` copyto, explicit-path 11:00 commit). Remaining
  pre-cutover item unchanged: **Bing Webmaster verification (user)** via GSC
  import.
- 2026-08-31: **Freeze ran overnight (one day late); cutover 09:00 step executed
  ~22:45 CDT.** Freeze artifacts stamped 00:43–01:09 (`parity_live_cache.jsonl`,
  `freeze` + `freezeprod` parity runs). Cutover 09:00: host routes uncommented in
  `wrangler.toml`, tests 9/9, deployed (`88f3311b`); user orange-clouded the two
  apex A records + `www`; Universal SSL served immediately (HTTP/2 200 via CF on
  both hostnames).
  **Defect found by the step's own verification — `staging` conflated two
  meanings.** `src/index.js` computed `const staging = host !== "archive.aero"`,
  written when `atc-staging` was the only non-archive.aero host. The DNS flip
  swept both atchistory.org hostnames into that bucket, and `staging` gated
  *both* "serve the bare site root without the `/atc/` prefix" (wanted) and
  "emit `X-Robots-Tag: noindex, nofollow`" (catastrophic): the live indexed
  corpus — 1,334 posts incl. the Wikipedia citation target — served noindex to
  crawlers. **The 11:00 step would not have fixed it**: the expression is
  `staging || env.ATC_NOINDEX === "1"`, so it short-circuits on `staging` and
  `ATC_NOINDEX="0"` never reaches it; the noindex would have stood until MODE
  flipped at 13:00. Fixed by splitting the two meanings — old hostnames never
  noindex, staging always does, archive.aero keeps the flag guard — redeployed
  `dc46fb4f`. Exposure a few minutes. Path handling and `selfOrigin`/`selfPrefix`
  still key off `staging` and are correct for old hosts; noindex was the only
  wrong consumer (all three checked).
  **Lesson for any future host route added to this worker: `staging` means
  "bare-root serving", NOT "keep it out of the index" — re-check every consumer
  of a host predicate when the set of hosts changes.**
  Verification: top-100 inventory URLs → 24 real one-hop 301→200, 1 direct 200
  (homepage), 2 → 410 (`/forum/` tombstones, D6), 73 junk
  (`.well-known/acme-challenge/*`, `.DS_Store`) → 404; **0 real failures**.
  301 targets stay on `www.atchistory.org` (no premature hop to archive.aero —
  that is the 13:00 step). `archive.aero/atc/` + `atc-staging` still noindexed.
  `MODE="serve"` / `ATC_NOINDEX="1"` unchanged. Note the runbook's step order was
  inverted deliberately: routes deployed *before* the DNS flip, so the flip is a
  single transition instead of a window where CF proxies to Incapsula.
- 2026-08-31 (later): **11:00 + 13:00 steps executed; cutover functionally
  complete.** 11:00: `ATC_NOINDEX="0"` deployed (`0925370b`); `archive.aero/atc`
  → 301 → `/atc/`, `/atc/` 200 with no `x-robots-tag`. Repo commit `a5c02cf`
  pushed to main, GH Pages built clean.
  ⚠ **The staged-set line `worker-atc/` was a trap**: with no local
  `.gitignore` it staged **1,706 files** — `node_modules/`, `.wrangler/state/`
  (local miniflare R2 blobs) and `node_modules/.cache/wrangler/wrangler-account.json`
  — into a public repo. Fixed by copying `worker/.gitignore` (`node_modules/`,
  `.wrangler/`, `.dev.vars`) to `worker-atc/`, which is what the tiles worker has
  always used; commit dropped to the correct 12 files. Also added
  `scripts/atc_r2_sync_filter.txt` (the `scripts/atc_*.py` glob missed it and it
  was the only copy of a file CLAUDE.md makes mandatory on every `atc-site` sync).
  ⚠ **Cache hazard, self-inflicted:** the sitemaps were probed ~40 s before the
  Pages build finished, so Cloudflare cached a 404 for all three under
  `max-age=7200`. Origin was fine throughout (cache-busted 200s). Wrangler's
  OAuth token has no `cache_purge` scope → dashboard purge required.
  **Next time: verify GH Pages build status (`gh api .../pages/builds/latest`)
  BEFORE the first bare curl — a premature probe poisons the edge for 2 h.**
  13:00: `MODE="redirect"` deployed (`54e9b414`). `atc_redirect_check.py
  --tag cutover`: **63,499 checks, 62,511 PASS, 988 NOTE, 0 FAIL** (170 s,
  4 scheme/host variants, all preflight sentinels OK; up from the 08-20
  rehearsal's 16,218 because the freeze-day inventory refresh grew the matrix).
  Every NOTE is `junk` (800) or `wp_internal` (188) "target 404 tolerated" —
  **0 NOTEs in `bucket=content`**. The only 404s are the 4 deliberate
  `wp-sitemap-bogus-1.xml` probes (never-frozen variant must 404 — guard works).
  08a loop both ways: 7 targets `200 0hop`, 7 old forms one-hop 301 → 200
  including the bare `http://` scheme-upgrade variants. Old host now serves
  `OLD_ROBOTS` + the frozen `_oldhost` sitemaps (200) — the 14:00 prerequisite.
  Remaining: CF cache purge (user), then 14:00 GSC Change of Address / Bing Site
  Move / sitemap submit (user). Registrar still untouched, per §5.
