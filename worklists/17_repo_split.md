# 17 — Split the repo: site vs. pipeline

**Drafted 2026-10-02. Plan only; nothing has moved.** Companion to
[06](06_publish_sync.md) (publication) and the `slicerd/` README (the NUC
publish flow this split cuts through).

**The cut:** code that *produces* data versus code that *serves and shows* it.
Today's deploy already splits there — the NUC uploads to R2, then one commit
to `main` updates the site on Pages. Two repos make that visible. The existing
public repo `hemenway/archive.aero` stays the site; a new repo takes the
pipeline, with its history extracted.

**What the history says.** Of 301 commits, 143 touch only the site, 6 only the
pipeline, 21 both. The 21 are the big features (metadata bundle, spatial cull,
basemap, airspace, chart inspector, half-sheets, every "Publish …" batch).
After the split each of those is two coordinated changes, which is why the
shared formats get written down first (§3).

---

## 1. Findings that shape the plan

| # | Finding | Consequence |
|---|---|---|
| F1 | **Catalog derivatives sat in the public history.** `master_dole_corrected.csv.bak` (3.0 MB, 6,301 rows incl. corner coordinates) was committed 2026-07-05 (`144623f`) and removed 2026-07-16 (`707bc38`); `early_master_dole.csv.bak` (1.1 MB) 2026-02-20 → 02-27; `missing_from_master_dole.csv` (0.4 MB) 2026-01-24 → 01-28. All three blobs are still reachable from `main`'s history, so anyone who clones the public repo has them. | The split is the moment to purge (§5). 0 forks, 0 watchers, 0 stars today — a force-push costs nothing now and more later. |
| F2 | The 427 MB pack is one **unreachable 706 MB blob** (nameless; never pushed — unreachable objects don't travel) plus ~55 MB of real history. | `git gc --prune=now` after the split. filter-repo gc's the extract on its own. |
| F3 | **Five redesign branches in flight**, each 1 commit ahead of `main`, all dated 2026-10-02, checked out in `~/.codex/worktrees/*` and `.claude/worktrees/*` (`redesign/{renderer,shell,data-contract,data-plane,combined}`). Stale: `vk/84db-improve-mobile-u` (242 behind, January), `claude/*`. | Any history rewrite invalidates those SHAs and the worktrees. Sequence the split after the redesign lands, or accept recreating the worktrees. `redesign/data-contract` may already be defining what §3 asks for — align, don't duplicate. |
| F4 | **Three site files are pipeline outputs** that travel through git: `dates.csv`, `coverage.json`, and three CONFIG URLs in `src/viewer.js` (`bundleUrl`, `basemapUrl`, `airspaceUrl`). Everything else crosses via R2 (`timeline_data.json`, `airfields.json`, bundles, PMTiles). | That is the whole git-side handoff surface. Keep it that small. |
| F5 | **`viewer_config.py` is imported by three pipeline scripts** (`build_metadata_bundle --update-html`, `airspace_build --update-html`, `era_gif` reads `bundleUrl` off the viewer source) and by `slicerctl finish_publish`. It edits `src/viewer.js` and runs `build_frontend.mjs` — site-repo work. | Make it the site repo's one CLI; pipeline scripts take `--site-repo` and shell out, or just print the URL (§4B). |
| F6 | **`slicerctl` uses one checkout for two jobs**: `sync` stages `scripts/`, `shapefiles/`, the catalog and `dates.csv` as a NUC release stamped with `git HEAD`; `publish` edits/commits/pushes the site. `~/.local/bin/slicerctl` is a symlink into the repo. | Two paths (`--pipeline-repo`, `--site-repo`); re-point the symlink. Release ids stop changing on viewer commits — a bonus. Latent gap found on the way: the git half never `git add`s `dates.csv`, so a new-era publish leaves the site's CSV fallback stale (§4A). |
| F7 | **`CLAUDE.md` is gitignored** (local only), as are worklists 09, 10, 13, 14, 14-drafts, `helper_prompt.md`, `airfields.json`, `1georef_toolv10.py` (only copy), `metadata-*.bundle` (36 × 20 MB in the checkout — analytics needs every one). | Not a git problem, a hand-move problem; `git status --ignored` is the checklist (§6 step 6). The georef tool must be backed up before anything moves. |
| F8 | Machine hooks pinned to `~/archive.aero`: the `slicerctl` symlink; `.claude/launch.json` (georef/georef-test/dupe-verify/viewer/worker-dev/atc-*); 8 of 56 `settings.local.json` allow-rules carry the absolute path; `~/.cloudflared/config.yml` comments reference `scripts/1georef_toolv10.py` (the tunnel itself targets port 5001, not a path); Claude Code memory is keyed by project path (77 files under `-Users-ryanhemenway-archive-aero`). No launchd/cron jobs. | §6 step 7. |
| F9 | `worker/` (tiles) and `worker-atc/` test in the site's `npm test`; the tiles Worker's 1 MiB block cache was co-designed with the viewer's `BundleSource` fetch pattern; `worker-atc` splices the site shell. | Workers stay with the site (open decision D4). |
| F10 | Shared thin modules: `jev_client.py` is used by `airfields_freeman_dates_jev.py` (pipeline) and `atc_posts_jev.py` (site/ATC). | Duplicate the client (≈ a page of code) rather than cross-import. |

---

## 2. Target layout

**Site repo** — the existing `hemenway/archive.aero`, stays at `~/archive.aero`
(keeps the GitHub URL, Pages, Actions, branches, worktrees, CNAME).

| Keeps | Notes |
|---|---|
| `index.html`, `about.html`, `contribute.html`, `sources.html`, `styles.css`, `src/`, `assets/`, `vendor/`, `next/`, `beta/` | the viewer and pages |
| `CNAME`, `robots.txt`, `sitemap*.xml`, `social-card.jpg`, `site-files.json`, `URI-POLICY.md`, `LICENSE` | URI-POLICY stays with the consumer; pipeline links to it |
| `dates.csv`, `coverage.json` | **pipeline-written handoff files** (F4) |
| `worker/`, `worker-atc/` | F9 |
| `scripts/`: `build_frontend.mjs`, `build_pages.py`, `vendor_frontend.mjs`, `build_beta.mjs`, `viewer_config.py`, `dev_server.py`, `cf_bulk_redirects_core.csv` | site build + the config CLI |
| `scripts/atc_*.py`, `atc_r2_sync_filter.txt`, `atc_posts_jev.py`, a copy of `jev_client.py` | ATC content tooling lives beside `worker-atc` (D3) |
| `tests/`: `browser/`, `server.mjs`, `fixtures/`, `flicker-regression-guard.html`, `test_viewer_config.py`, `test_pages_publication.py`, `README.md` | |
| `package.json`, `package-lock.json`, `playwright.config.mjs`, `.github/workflows/pages.yml`, `regression.yml` (minus the g2p test) | |
| `worklists/08*.md` (ATC) | with `worker-atc` |
| `skills-lock.json`, `.env` (own copy), `README.md` (site half), `CLAUDE.md` (site half, still gitignored) | |

**Pipeline repo** — new, `hemenway/archive.aero-pipeline` at
`~/archive.aero-pipeline` (name is D5).

| Takes | Notes |
|---|---|
| `slicerd/` | `deploy.sh` is path-relative (`$0`); nothing to change |
| `shapefiles/` (1,209 files, 4.7 MB; `extents/` is read by `dole_v2`) | |
| `scripts/`: `slicer.py`, `dole_v2.py`, `1georef_toolv10.py` (gitignored), `georef_*`, `import_*`, `stage_*`, `upgrade_*`, `promote_attic_sectionals.py`, `ifly_*`, `sd_slurp.py`, `fatwalk.py`, `wayback_chart_fetch.py`, `ead_aip_*`, `extract_gpo_microfiche.py`, `split_faa_inset_containers.py`, `wasp_fold_cutline.py`, `verify_dupes_app.py`, `timeline_preview_gui.py`, `analyze_dole.py`, `audit_disk_vs_dole.py`, `nasr_diff.py`, `aisdata_pull.py`, `airspace_build.py`, `basemap_build.py`, `build_timeline_data.py`, `build_coverage.py`, `build_metadata_bundle.py`, `publish_chart_pmtiles.py`, `pmtiles_fix_runlengths.py`, `install_geotiff2pmtiles.py`, `era_gif.py`, `analytics_*.py`, `workers_logs_export.py`, `airfields_freeman_*.py`, `jev_client.py` | everything that reads the catalog or writes R2 |
| `tests/test_slicer_16bit_sources.py`, `tests/test_geotiff2pmtiles_dependency.py` | the g2p test leaves `regression.yml` |
| `worklists/` (all but 08*) incl. `search_archive/`, `superseded_sources.csv`, gitignored `data/` | the operations notebook; 09/16 are cross-cutting but belong with the people doing the work |
| `openaip-mirror/`, `historical-data/` (gitignored) | acquisition |
| Local, gitignored: `master_dole_v2.csv` + `.bak`, `timeline_data.json`, `airfields.json`, `bounds_cache.json`, every `metadata-*.bundle`, `worklists/data/`, `CLAUDE.md` (pipeline half) | F7 |
| `.env` (own copy), `skills-lock.json` + `.agents/`, `LICENSE`, `README.md` (pipeline half), its own `.github/workflows/tests.yml` | |

**Neither repo:** `Donation List.numbers` (untracked at the root; donor
names — move to iCloud or `~/archive.aero-attic/`), `_site/`,
`playwright-report/`, `test-results/`, `node_modules/`.

---

## 3. The contract — write it down before cutting

Nine shared formats. Each gets a producer, a consumer, a version/pin, and a
fixture both repos test against. Lives in the site repo as
`docs/data-contract.md` beside `URI-POLICY.md` (the consumer owns the spec;
check what `redesign/data-contract` already holds first — F3). The pipeline
README links to it and vendors the same fixtures.

| # | Artifact | Producer → consumer | Pin |
|---|---|---|---|
| C1 | `metadata-<hash>.bundle` | `build_metadata_bundle.py` → `MetaBundle` in `src/viewer.js`; also `analytics_heatmap.py` (offsets) | header layout + per-era record; fixture already exists in `tests/browser/fixtures.mjs:34-42` — extract it to a shared file |
| C2 | `dates.csv` | `build_metadata_bundle --emit-dates-csv` → viewer CSV fallback **and** the slicerd release's live-era list (`engine.py:146`) | column set; the pipeline repo becomes its home, the site gets a copy |
| C3 | `charts/sectionals/timeline_data.json` (R2) | `build_timeline_data.py` → `contribute.html`, viewer pin card (`pm` field), `build_coverage.py` | per-chart entry schema incl. `pm`, `half`, extent rings |
| C4 | `coverage.json` | `build_coverage.py` → heat strip | `segments: [startISO, endISO, count, pct]`, end-exclusive |
| C5 | `charts/sectionals/airfields.json` (R2) | `airfields_freeman_*` → viewer airfield layer | GeoJSON properties the layer filters on (dates) |
| C6 | `airspace/class-<stamp>.pmtiles` (R2) | `airspace_build.py` → `AirspaceLayer` | layers `class`/`efloor`; attrs `rg`, `from`, `to`, class; metadata `archive_aero.regions.<rg>.{cycles,boxes}` |
| C7 | `basemap/protomaps-<build>.pmtiles` (R2) | `basemap_build.py` → protomaps-leaflet | extract maxzoom **= `CONFIG.basemapMaxDataZoom`** (13); `REGION_BOXES` ⊇ C3 rings is a pipeline-internal check |
| C8 | R2 key layout | `slicerd/publish_era.py`, `publish_chart_pmtiles.py`, `*_build.py --upload` → `worker/`, viewer | `sectionals/<era>.pmtiles`, `sectionals/chart/<slug>/<date>[-half]` (extension-less, permanent), `sectionals/metadata-*.bundle`, dated immutable basemap/airspace keys — already `URI-POLICY.md` rule 7 + exceptions log |
| C9 | CONFIG URL handoff | pipeline publish → `src/viewer.js` `bundleUrl` / `basemapUrl` / `airspaceUrl` | the site repo's `viewer_config.py` CLI is the only writer (§4B) |

---

## 4. Code changes — do these *in the mono-repo first*

Decouple in place while both halves are still side by side and every test
still runs (`npm test`, `slicerctl publish … --dry-run`). Then the split is a
file move, not a refactor.

**A. `slicerd/mac/slicerctl`**
- `--repo` → `--pipeline-repo` (default `~/archive.aero-pipeline`) for `sync`;
  add `--site-repo` (default `~/archive.aero`) for `publish`.
- `cmd_sync` stages from the pipeline repo; release id = pipeline HEAD.
- `git_preflight` and `finish_publish` run against the site repo;
  `sys.path.insert(site/scripts)` for `viewer_config`.
- `finish_publish`: copy the run's `dates.csv` into the site repo and include
  it in the `git add`/`commit` list (closes F6's latent gap). Also copy the
  bundle into the **pipeline** checkout, not the site's — the "keep every
  bundle" rule is an analytics need.
- Update the README's "Everyday use" and the `archive-slicer` MCP
  instructions text (`server.py:28-32`, "the Mac must then push … index.html").

**B. `viewer_config.py` becomes a CLI** (`python scripts/viewer_config.py
bundleUrl <url>`; keeps the importable function). Then:
- `build_metadata_bundle.py --update-html PATH` → `--site-repo PATH`
  (subprocess to the CLI) or drop it and let `slicerctl` do it. Same for
  `airspace_build.py`.
- `era_gif.py`: `--bundle-url` flag, or read the latest `uploaded.jsonl` under
  `/Volumes/projects/slicer-runs/`; stop regex-parsing the viewer source.
- `tests/test_viewer_config.py` stays in the site repo unchanged.

**C. Output paths.** `build_coverage.py` `--timeline` + `--out` (today it
hard-codes `ROOT/coverage.json`); `build_metadata_bundle --emit-dates-csv`
takes the site path. Both default to `../archive.aero/...` so the common case
stays one command.

**D. CI.** `regression.yml`: remove the `test_geotiff2pmtiles_dependency.py`
step. New `archive.aero-pipeline/.github/workflows/tests.yml`: Python unittest
for the two pipeline tests (`test_slicer_16bit_sources` needs GDAL —
`apt-get install gdal-bin python3-gdal` or mark it local-only).
`pages.yml`, `build_pages.py`, `site-files.json` and
`test_pages_publication.py` are unchanged — the allowlist still guards
against publishing anything stray.

**E. `.gitignore` split.** Site: node/playwright/`_site/`/`.wrangler/`,
`CLAUDE.md`, `.claude`. Pipeline: Python, catalog patterns,
`1georef_toolv10.py`, `timeline_data.json`, `worklists/data/`,
`metadata-*.bundle`, `historical-data/`, `bounds_cache.json`, `airfields.json`,
the gitignored worklists, `CLAUDE.md`, `.claude`.

**F. Docs.** `README.md` → two (site: §§ Site publication, Delivery, Frontend,
ATC; pipeline: §§ Chart processing, Tile conversion, Data sources, slicerd).
`CLAUDE.md` → two by hand (≈ 80 % of today's is pipeline). `worklists/README.md`
index moves with the worklists; the site keeps a stub pointing across.

**G. `.claude/launch.json`** split: `georef`, `georef-test`, `dupe-verify` →
pipeline; `viewer`, `static`, `worker-dev*`, `atc-*` → site (the `viewer`
config's `/airspace` mount keeps pointing at `/Volumes/projects/airspace_pmtiles`).
`settings.local.json`: copy, then fix the 8 absolute-path rules per repo.

---

## 5. History — how to cut

**Site repo: no rewrite needed for the split itself.** One commit removes the
pipeline paths ("Move the chart pipeline to archive.aero-pipeline"). History,
branches, worktrees, Pages and Actions all survive.

**Pipeline repo: `git filter-repo` on a fresh clone**, `--paths-from-file`
listing today's pipeline paths **plus their pre-2026-07-16 names**, or the
extract's history starts in July (`git log --follow` shows e.g.
`scripts/slicer.py ← slicer.py ← newslicer.py`). Include from the
all-time top-level inventory: `slicer.py newslicer.py newslicer_parallel.py
oldslicer.py analyze_dole.py timeline_preview_gui.py dole_dataset.py
verify_dole.py check_chart_editions.py faa_chart_slicer_gui.py align_tif_gui.py
ocr_dates*.py get_wayback_data.py sync_early_rawtiffs.py transformer.py
update_dates_csv.py *.sh slicer-go/ slicer-rs/ pmandupload-rs/ docs/
BUILD_PROCESS.md FSD.md PARALLEL_QUICKSTART.md TRANSFORMER_README.md
IMPROVEMENTS.md errors.md runlogs.txt newslicer_logs.txt`. Leave out the
three catalog blobs (F1) — they are root files, so a path allowlist excludes
them by default; verify with
`git rev-list --objects --all | grep -iE 'dole.*csv|corrected'` → empty.

**Purging F1 from the site repo** (D2) is a separate, optional rewrite:
`git filter-repo --invert-paths --path master_dole_corrected.csv.bak --path
early_master_dole.csv.bak --path missing_from_master_dole.csv`, force-push
all branches, then ask GitHub support to clear the dangling objects (they stay
fetchable by SHA until GitHub gc's). Rewrites every SHA: do it in the same
window as F3's worktree recreation, not later.

---

## 6. Sequence

0. Settle D1–D6 below.
1. Land or merge the five redesign branches; delete `vk/…` and the stale
   `claude/…` branches (`git branch -D`, `git push origin --delete`).
2. §4 A–G in the mono-repo. Prove it: `npm test`; `python3 -m unittest`;
   `slicerctl sync` then `slicerctl publish <run> <key> --dry-run`;
   `build_coverage.py --out` round-trip diff = empty.
3. **Back up** (attic rule): `git clone --mirror` →
   `~/archive.aero-attic/repo-mirror-<date>.git`; tar the working tree
   *including ignored files* → `~/archive.aero-attic/worktree-<date>.tar`
   (that is the only copy of the georef tool and of 36 bundles — F7).
4. Build the pipeline repo: fresh clone → `git filter-repo --paths-from-file`
   (§5) → check commit count, `git log --follow scripts/slicer.py` reaches
   January, no catalog blobs → create `hemenway/archive.aero-pipeline` (D1
   decides visibility) → push → clone to `~/archive.aero-pipeline`.
5. Site repo: `git rm -r` the pipeline paths, apply §4 D–F, commit, push,
   watch `pages.yml` go green. Optional D2 rewrite + force-push here.
   `git gc --prune=now` (F2). Recreate the codex/Claude worktrees.
6. Move the local files (F7): catalog + `.bak`, bundles, `timeline_data.json`,
   `airfields.json`, `bounds_cache.json`, `worklists/data/`, `historical-data/`,
   `1georef_toolv10.py`, gitignored worklists, pipeline `CLAUDE.md`. Finish
   when `git status --ignored` in the site repo shows only node/build output.
7. Re-point the machine (F8): `ln -sf ~/archive.aero-pipeline/slicerd/mac/slicerctl
   ~/.local/bin/slicerctl`; per-repo `launch.json` + `settings.local.json`;
   fix the `~/.cloudflared/config.yml` comment; copy pipeline memories to
   `~/.claude/projects/-Users-ryanhemenway-archive-aero-pipeline/memory/`
   (list below); `slicerctl sync` to cut the first release from the new repo.
8. **Verify one case end-to-end** (CLAUDE.md rule 2): the next real reslice
   publishes through the two-repo `slicerctl publish`; the georef tool starts
   from the pipeline repo and `tools.archive.aero` still serves it; `viewer`
   launch config serves the site; `npm test` green in site, unittest green in
   pipeline; Pages deploy green.
9. Update both READMEs, both `CLAUDE.md`s, `worklists/README.md`, and this
   worklist's status line.

**Rollback:** until step 5's push, `git reset --hard` + the mirror. After a
force-push, `git push --mirror` from the attic clone restores every ref.

**Memory split** (from `MEMORY.md`; site keeps the rest):
- pipeline only: slicer→pmtiles, slicer-go port, slicer PDF sources, FAA
  download, dole schema/loader/attic, coverage audit, AVSIM, rawtiffcandidates,
  HUNT26, dupe-verify, slicer run 07-14, georef inference, half-sheet groups,
  warp temp corruption, LOC sp mapping, iFly 2013, ACASIS ×2, SD-card ×2,
  extraction placement, Sarangan, zip-row cutline, Pillow orientation, gap
  audit, georef tool ×3, aisdata, EAD harvest, PMTiles run-length, WASP fold,
  Jev setup, chart scan stitch, M40 scanner, donation scans, mail search,
  chart contributors, timeline coverage shape, codec test, g2p y-shift,
  NotAQuad/Quad-G+ ports, python env.
- both: publish sync, metadata bundle architecture, chart pmtiles pipeline,
  worker cache staleness, pmtiles rejection poisoning, BYPASS p99, viewer
  analytics / CF analytics gotchas, airspace layer design, proprietary
  catalog rule, SkyVector comparison.
- site only: worker deploy, worker INM/flight wedge, preview quirks, pin
  inspector, map controls mocks, de-AI design, site copy voice, contact +
  social card, contribute page, viewer first-load, protomaps label prune,
  ATC shell, ATC origin dead, airfields-freeman mirror, aerofiles, job search.

---

## 7. Open decisions

| # | Decision | Recommendation |
|---|---|---|
| D1 | Pipeline repo public or private? | **Public.** The slicer, slicerd and georef inference are the strongest engineering in the portfolio (job-search memory). Private would allow tracking `1georef_toolv10.py`, but the catalog stays out either way — a private GitHub repo is still a third party. |
| D2 | Purge F1 from the public history? | **Yes, in the same window as step 5**, while forks are 0 and the worktrees are being recreated anyway. |
| D3 | Where does ATC live? | **Site repo for now** (`worker-atc` + `atc_*` + worklists 08*). Self-contained enough to become a third repo later without touching the pipeline. |
| D4 | Workers? | **Site repo** (F9). |
| D5 | Name | `archive.aero-pipeline` / `~/archive.aero-pipeline`. Alternatives: `archive.aero-data`, `sectionals-pipeline`. The memory dir name follows the path. |
| D6 | When | **After the redesign branches land** (F3). Steps 0–3 can start now; §4 is useful on its own even if the split waits. |

**Lighter alternative, for the record:** one repo with `site/` and `pipeline/`
folders and a `CLAUDE.md` in each gives most of the focus benefit with none of
the two-repo coordination cost, but none of D1/D2 — no privacy line, no clean
history, and `slicerctl` releases keep churning on viewer commits.
