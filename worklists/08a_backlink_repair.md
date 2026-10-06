# 08a — inbound backlink maintenance

Companion to `08_atchistory_migration.md`. Covers the *external* half of the move:
correcting useful existing inbound links so they point at canonical
`archive.aero/atc/…` URIs instead of atchistory.org paths.

**Status reviewed 2026-09-21:** the August 31 cutover and repository redirect repairs
are complete. **No Wikimedia edits, disclosure, webmaster email or later backlink
survey is recorded as completed.** The external list below is a dated survey, not
a verified current queue. Redirects keep the old links working; this work is optional
maintenance and does not gate the October 30 stabilization review.

## Remaining work

- [ ] Recheck the survey: confirm each old link still exists, its intended target
      still serves the right content, and the canonical target has no redirect hops.
      Keep a dated record of rechecks and any subsequent edits/reversions.
- [ ] If proceeding with Wikimedia maintenance, confirm the relevant wiki's current
      conflict-of-interest/link-maintenance rules and make the disclosed URL-only
      repairs below. Keep archived references and archived discussions intact.
- [ ] Recheck dreamsmithphotos and TVTropes before deciding on outreach/edits;
      the suggested email below is an **unsent draft**.
- [ ] Repeat the backlink survey and inspect external referrers from the existing
      ATC 404 review when doing follow-up. This worklist does not configure a recurring run.

## Survey and repair reference — 2026-08-13

Compiled from: Wikimedia `list=exturlusage` across en/de/fr/es/it/nl/pl/
ru/pt/sv/ja/simple + commons/wikidata/wikisource/wikibooks; awstats all-time
referrer table (`atchistory_backup/traffic_analysis/report_data.json`); raw-log
referrer tally. Every target below was resolved live through the deployed worker
and confirmed 200 **on that date**. The seven-target smoke loop was repeated at the
August 31 cutover, with direct 200s and old forms redirecting in one hop (08 §7).
No fresh external verification was performed for this cleanup.

---

## Original pre-cutover rationale (historical)

- The destinations were available from August 9 behind a production noindex guard;
  that guard was removed at the actual August 31 cutover. Staging remains noindexed.
- Referral traffic and citation accuracy motivated direct-link maintenance even
  before indexing. The Checklist article had sent 6,034 visits in the historical data.
- The original plan proposed a small number of disclosed repairs at a time. The
  domain migration is now complete; there is no remaining pre-cutover deadline.

## Project guardrails for any external repairs

Ryan operates the destination site, so consult
[WP:COI](https://en.wikipedia.org/wiki/Wikipedia:Conflict_of_interest). These are
proposed repairs to existing citations whose URLs moved
([WP:LINKROT](https://en.wikipedia.org/wiki/Wikipedia:Link_rot)), not completed edits.
Check the applicable wiki's current rules before editing. The project's scope is:

1. **Disclose once** on your Wikipedia user page. One sentence is enough:
   *"I operate archive.aero, which now hosts the former atchistory.org collection.
   I limit my editing to updating existing citations whose URLs moved."*
2. **Only repair existing links.** Do not add a new citation, a new external link,
   or a new "Further reading" entry to any article. Not one.
3. **Change the URL, nothing else.** Leave titles, authors, dates, and prose alone.
4. **Keep every `archive-url` / `{{Webarchive}}` parameter** already present. Set
   `url-status=live` where the old value was `dead`.
5. **Edit summary, every time**, verbatim-ish:
   `Update moved URL: atchistory.org content relocated to archive.aero/atc/ (COI: I operate the destination; URL-only change)`
6. **Never edit talk pages or user talk pages** to fix a link. Two hits below are
   on talk pages — they are archived discussion and stay as-is.
7. One or two articles per day, maximum. If anything is reverted, do not re-revert:
   open a talk-page request instead.

---

## Step 0 — repo changes — ✅ DONE 2026-08-13

Four ghost URLs found by this survey (404 on live **and** on the new stack, but
still carrying live inbound links) were added to `worker-atc/src/legacy_map.json`,
taking it from 12 to 16 entries:

```json
 "/History/fsshist.htm": "/atc/flight-service-history-1920-1998/",
 "/History/FacilityPhotos/WY/RadioBeacons/Summit_RadioBeacon1_WY.htm": "/atc/History/FacilityPhotos/WY/RadioBeacons/",
 "/History/FacilityPhotos/WY/RadioBeacons/Summit_RadioBeacon3_WY.htm": "/atc/History/FacilityPhotos/WY/RadioBeacons/",
 "/History/FacilityPhotos/WY/RadioBeacons/Summit_RadioBeacon4_WY.htm": "/atc/History/FacilityPhotos/WY/RadioBeacons/"
```

- `/History/fsshist.htm` — killed by the ~2020 WP migration; successor is the
  Flight Service history post. Linked from `en:User talk:ATCZero` (do not edit).
- The three `Summit_RadioBeacon*_WY.htm` per-photo wrapper pages are gone
  everywhere; the JPGs survive as `Summit-BeaconHill{_1,2,3,4}.jpg` in the same
  directory, which now serves a generated listing. Mapped to the listing rather
  than guessing 1→`_1`, 3→`3`, 4→`4` — the numbering doesn't line up (four JPGs,
  three linked pages). **Still linked live** from dreamsmithphotos.com.

Route map regenerated (186 drops; `--verify` 0 unresolved / 0 chains), tests 7/7,
worker deployed as version `9e5a24f7` with `MODE=serve` and `ATC_NOINDEX=1`
unchanged. All 16 entries re-verified end-to-end: 14 × one-hop 301→200, 2 × 410,
0 problems.

The same sweep uncovered three further issues — a case-collision data loss, four
missing webfonts, and a parity-harness false negative — all recorded in
worklist 08 §A4. Two carry standing obligations:

- **Every `rclone sync` to `atc-site` must pass
  `--filter-from scripts/atc_r2_sync_filter.txt`** or it deletes the four
  recovered case-collision objects and the 6 `_oldhost/` sitemaps.
- ~~**`atc_parity_check.py`'s closure live-verification is unreliable**~~ —
  ✅ fixed 2026-08-20 (canonical→old reverse mapping before the live probe;
  worklist 08 §A4 item 3 has the full account).

---

## Step 1 — Wikimedia candidates (17 URL occurrences across 12 editable pages)

Every "new URL" below was verified in the August 13 survey: 200, correct
content-type, zero further hops. Counts exclude the archived talk-page entry.
Old URLs are the spellings captured then; recheck before editing.

### 1a. The checklist citation — 5 editable pages + 1 archived discussion

`http://www.atchistory.org/History/checklst.htm` (49,463 all-time hits; **404 on
the old live site since ~2020** at survey time; the migration now redirects this path)
→ **`https://archive.aero/atc/how-the-pilots-checklist-came-about`**

| # | Wiki | Page | Occurrences | Notes |
|---|------|------|-------------|-------|
| 1 | en | [Aviation safety](https://en.wikipedia.org/wiki/Aviation_safety) | 1 | `{{Cite web}}`, has `archive-url` (2012 Wayback); keep it, leave `url-status=live` |
| 2 | en | [2nd Operations Group](https://en.wikipedia.org/wiki/2nd_Operations_Group) | 1 | `{{Cite web}}`, `url-status=dead` → change to `live` |
| 3 | en | [Boeing B-17 Flying Fortress](https://en.wikipedia.org/wiki/Boeing_B-17_Flying_Fortress) | **3** | Refs `Checks` and `Checks2` are near-duplicates, each with a `{{Webarchive}}`. Update all three URLs; do **not** merge the refs (that's a content edit) |
| 4 | pt | [Segurança aérea](https://pt.wikipedia.org/wiki/Seguran%C3%A7a_a%C3%A9rea) | 1 | `{{Citar web}}`; also update `obra=www.atchistory.org` → `obra=archive.aero` |
| 5 | sv | [Checklista](https://sv.wikipedia.org/wiki/Checklista) | 1 | Bare link in `== Externa länkar ==`, not a ref. Closest to the COI line — consider posting on `Diskussion:Checklista` first rather than editing directly |
| — | en | [Talk:BUMMMFITCHH](https://en.wikipedia.org/wiki/Talk:BUMMMFITCHH) | 1 | **Do not edit** — archived 2010 discussion |

Note: `en:Checklist` — the article that historically sent the most traffic — no
longer contains the link at all; it was removed sometime after the URL went dead.
Do **not** re-add it. Adding a link to your own site is the one thing the COI
rules actually prohibit. If you want it back, ask on
[Talk:Checklist](https://en.wikipedia.org/wiki/Talk:Checklist) with an
`{{edit COI}}` request and let an uninvolved editor decide.

### 1b. The rest

| # | Wiki | Page | Old URL | New URL |
|---|------|------|---------|---------|
| 6 | en | [Teleprinter](https://en.wikipedia.org/wiki/Teleprinter) | `http://www.atchistory.org/flight-service-history-1920-1998/` | `https://archive.aero/atc/flight-service-history-1920-1998` |
| 7 | en | [Pioneer Airport](https://en.wikipedia.org/wiki/Pioneer_Airport) | `http://www.atchistory.org/mauston-light-station` | `https://archive.aero/atc/mauston-light-station` |
| 8 | en | [Ormond Robbins](https://en.wikipedia.org/wiki/Ormond_Robbins) | 3 PDFs, see below | see below |
| 9 | de | [Regionalflughafen Watertown](https://de.wikipedia.org/wiki/Regionalflughafen_Watertown) | `https://www.atchistory.org/History/SouthDakota/index.htm` | `https://archive.aero/atc/history/SouthDakota/` |
| 10 | de | [Regionalflughafen Pierre](https://de.wikipedia.org/wiki/Regionalflughafen_Pierre) | same as above **plus** `https://www.atchistory.org/pierre-fss-1990/` | `https://archive.aero/atc/history/SouthDakota/` and `https://archive.aero/atc/pierre-fss-1990` |
| 11 | sv | [São Tomé fyr](https://sv.wikipedia.org/wiki/S%C3%A3o_Tom%C3%A9_fyr) | `https://www.atchistory.org/category/early-radio-years/light-houses/` | `https://archive.aero/atc/topics/light-houses` |
| 12 | commons | [File:Alaska International Air Lockheed L-100 Hercules wrecked…png](https://commons.wikimedia.org/wiki/File:Alaska_International_Air_Lockheed_L-100_Hercules_wrecked_near_the_runway_at_Fletcher%27s_Ice_Island,_1973.png) | `https://www.atchistory.org/pdf/faa_world/1974/faa_world_9-1974.pdf` | `https://archive.aero/atc/library/faa_world/1974/faa_world_9-1974.pdf` |

**Ormond Robbins** — three inline refs, each a bare `[url atchistory.org]` link:

```
https://www.atchistory.org/History/Pubs/mukluk_telegraph_pubs/pdf_files/1947/mukluk_telegraph_jul_1947.pdf
  → https://archive.aero/atc/history/Pubs/mukluk_telegraph_pubs/pdf_files/1947/mukluk_telegraph_jul_1947.pdf

https://www.atchistory.org/History/Pubs/mukluk_telegraph_pubs/pdf_files/1948/mukluk_telegraph_1948_nov.pdf
  → https://archive.aero/atc/history/Pubs/mukluk_telegraph_pubs/pdf_files/1948/mukluk_telegraph_1948_nov.pdf

https://www.atchistory.org/History/Pubs/ak_phone_directory/alaskan_region_telephone_directory_Jan_1968.pdf
  → https://archive.aero/atc/history/Pubs/ak_phone_directory/alaskan_region_telephone_directory_Jan_1968.pdf
```

The visible link label is the literal text `atchistory.org` in all three — update it
to `archive.aero` so the rendered citation isn't lying. (Only case where touching
label text is warranted; it *is* the URL, effectively.)

Note the lowercase `history/` — canonical space lowercases the first path
component only. `Pubs/` and the rest keep their case. Copy exactly.

### Per-edit procedure

1. Open the article → **Edit source** (not visual editor — it mangles ref params).
2. Ctrl-F the old URL. Confirm the occurrence count matches the table.
3. Replace URL only. Leave `archive-url`/`{{Webarchive}}` intact.
4. Paste the standard edit summary from the COI rules above.
5. **Show preview**, confirm the ref renders and the link resolves.
6. Save. Watchlist the page.

---

## Step 2 — non-Wikimedia referrers

Checked 2026-08-13; all-time hit counts from awstats.

| Referrer | Hits | Status on 2026-08-13 | Candidate action |
|----------|------|--------------|--------|
| `dreamsmithphotos.com/arrow/States/wy/wyoming.html` | 631 | **200, 3 live links** to the dead `Summit_RadioBeacon*_WY.htm` pages | Email the webmaster. Step 0's legacy map makes them work regardless; the email is upgrade, not repair |
| `atvutah.com` YaBB forum | 1,549 | 200, links are inside threads (not on index) | Low value, links are old thread posts. Skip |
| `flash.aopa.org/asf/flightservice/learn_more.cfm` | 1,547 | **Host dead** (no DNS/connection) | Nothing to do. AOPA retired the Flash-era ASF site |
| `m.facebook.com` / `l.facebook.com` / `facebook.com` | 2,688 combined | n/a | Cannot edit others' posts. If any are on a page you control, update there |
| `afss.com/service/` | 368 | 200, **no atchistory links remain** | Nothing to do |
| `ourtimeonline.com` | 326 | **Host dead** | Nothing to do |
| `tvtropes.org/…/FailedASpotCheck` | 285 | 403 to CLI clients | Check by hand in a browser; if a link is there, TVTropes allows anonymous edits |
| `viglink.com/sites/atchistory.org` | 313 | Affiliate rewriter, not a real backlink | Ignore |
| `atchistory.com/about/` (`.com`, not `.org`) | 263 | **Host dead** | Ignore — unrelated dead domain |

### Unsent draft for dreamsmithphotos.com (recheck links before use)

> Subject: atchistory.org photo links on your Wyoming page
>
> Hi — your Wyoming beacons page links to three atchistory.org pages
> (`Summit_RadioBeacon1/3/4_WY.htm`) that have been dead since a 2020 site
> rebuild. I've taken over hosting that collection; the Summit / Beacon Hill
> photos now live at
> https://archive.aero/atc/history/FacilityPhotos/WY/RadioBeacons/
> — the old links also redirect there following the August 31 migration.
> No action needed if you'd rather leave them; just wanted you to have the working address.

---

## Step 3 — target and redirect verification reference

For any future repair, recheck the direct targets (this seven-target subset is not
the complete Wikimedia candidate list):

```bash
for u in \
  https://archive.aero/atc/how-the-pilots-checklist-came-about \
  https://archive.aero/atc/flight-service-history-1920-1998 \
  https://archive.aero/atc/mauston-light-station \
  https://archive.aero/atc/history/SouthDakota/ \
  https://archive.aero/atc/pierre-fss-1990 \
  https://archive.aero/atc/topics/light-houses \
  https://archive.aero/atc/library/faa_world/1974/faa_world_9-1974.pdf ; do
  printf '%s %s\n' "$(curl -s -o /dev/null -w '%{http_code} %{num_redirects}hop' -L "$u")" "$u"
done
```

Expect `200 0hop` on every line. Any redirect hop means the URL written into
Wikipedia is an alias, not the canonical — fix the wikitext, not the worker.

Also check the old-URL forms
(`https://www.atchistory.org/History/checklst.htm` etc.) and confirm one-hop 301.
`scripts/atc_redirect_check.py` covers the old-URL side; its last recorded cutover
run passed August 31. Preserve its frozen live cache, since the former origin was
reported unavailable September 20.

## Step 4 — post-cutover watch

- Suggested manual cadence: re-run the LinkSearch sweep monthly through November;
  this document does not configure a scheduled monitor. One query per wiki:
  `https://en.wikipedia.org/w/api.php?action=query&list=exturlusage&euquery=atchistory.org&eulimit=500&euprop=title|url&format=json&formatversion=2`
- Watch `atc_logs` referrers for external hosts hitting 404s — that is the
  discovery channel for backlinks no log or sitemap knew about.
- **The atchistory.org registration never lapses** regardless of any of this
  (worklist 08 §5). Every link left unrepaired anywhere on the web keeps working
  through the 301s. This whole worklist is optimization, not a dependency.
