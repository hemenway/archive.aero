# 16 — Jev (TypeSafe) as programmable judgment for archive.aero

**Plan written 2026-09-18.** Companion to [09](09_growth_and_revenue.md) (growth) and
[08](08_atchistory_migration.md) (ATC migration). Jev is a *System One* model: it
returns typed answers (a choice, a yes/no probability, a score) over state you send
it — it does not generate text. Docs: <https://docs.typesafe.ai/llms.txt>.

**What it is for here:** the places where the site's pipelines currently stop at a
regex heuristic or a blank field because the source is free text — Paul Freeman's
airfield narratives, 1,333 atchistory WordPress posts, user-agent strings, a
visitor's typed query. Everything numeric, geometric, or exact (dates arithmetic,
extents, coverage %, dedupe by pixels, georef) stays in code, per the model's own
[jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md): no
counting, no date comparison, no math, no generation.

**Verified 2026-09-18 (one case end-to-end before any batch, per CLAUDE.md):**
[data/jev/pilot_2026-09-18.py](data/jev/pilot_2026-09-18.py) →
[results](data/jev/pilot_2026-09-18_results.md). Jev read one ATC post and
three Freeman entries and got every checkable answer right, including two the
regex engine had left `unknown`/`conflict`
(Drummond Hospital Airport: last intact 1972, closing not stated;
Vanderbilt Seaplane Base: opened 1936, closed 1947 — the heuristic had picked
1950, the museum's opening). Whole pilot: ~7,500 input tokens, < $0.001.

---

## Ground rules (apply to every workstream)

| Rule | Why |
|---|---|
| **Key lives in `.env` as `TYPESAFE_API_KEY`** (gitignored; the Python SDK reads that name) and, for any live endpoint, in a Worker secret via `wrangler secret put`. Never in a script, never in `index.html`/`src/`. | Public repo. The key was pasted into a chat transcript on 09-18 — rotate it in the TypeSafe dashboard when convenient. |
| **Pin `jev-1.13.0`** in batch scripts; log the `model` field from every response. | Thresholds tuned on one version don't carry to the next; `jev-latest` moves. |
| **Code finds candidates, Jev selects.** Years/dates/idents/place names are regex- or gazetteer-extracted first; Jev picks among them (Choice) with an explicit `not_stated` option. | Jev doesn't generate or count; a Choice over real spans gives a verbatim value code can normalize. |
| **One narrow question each; independent questions batched in one request.** | Parallel evaluation is ~10× cheaper/faster than serial calls; questions can't see each other's answers, so state each premise explicitly. |
| **Send only the fields the question needs.** Freeman entry segment, not the page; post `entry-content`, not the nav chrome. | Accuracy falls with irrelevant state ("context rot"). |
| **Provenance in the output row:** `<field>_basis=jev`, `<field>_conf`, `jev_model`, plus the request/response pair appended to `worklists/data/jev/<task>.jsonl`. | Same rule as the catalog `note` column; lets thresholds be re-tuned without re-inference. |
| **Thresholds per task, set on a hand-labeled sample of ≥50 rows**, recorded in the script header with the date. Choice/Score `confidence` = distribution concentration, not correctness. | Cookbook thresholds are examples, not rules. |
| **Never build a Noul and its negation, or a Noul and a yes/no Choice, and expect them to add up.** | Documented non-invariant. |
| Shared helper `scripts/jev_client.py`: env key, `jev-1.13.0`, backoff on 429/529, JSONL audit log, content-hash cache so a re-run re-scores without re-paying. | One place for the plumbing; every workstream is then a state builder + a question set. |

Pricing/limits (2026-09-18): $0.042 per M input tokens, output free; 250k tok/s,
1,200 req/min; 64k context (32k for state + longest question). Every offline job
below costs well under $1 in total.

---

## Open decisions

| # | Decision | Recommendation |
|---|---|---|
| J1 | Is a small live endpoint acceptable for W4 (the viewer "Find" box), given the README's "no application backend" claim? | **Yes, as a stateless proxy in the existing `tiles` Worker** (the same shape as the range proxy — no database, no sessions), rate-limited and cached. If that feels like a backend, W4 waits; W1–W3, W5 need nothing live. |
| J2 | Do W1's Jev dates overwrite the regex dates or sit beside them? | **Beside, then adopt by rule:** Jev wins where the regex was blank/conflict and Jev's confidence ≥ threshold; disagreements above threshold go to a review CSV; regex `closed_stated` with a matching Jev answer stays as-is. |
| J3 | Publish Jev-derived ATC metadata inside `/atc/` before the 2026-10-30 stabilization date? | **No.** Build the index now (offline JSON), consume it first from archive.aero-side surfaces (airport pages, viewer layers), and only add `/atc/` navigation after 08 §5 releases it. |

---

## Workstreams (ordered by payoff ÷ effort)

### Status 2026-09-18 evening — W1 + W2 built, verified, **blocked on API credits**

The account returned `HTTP 402 billing_error` ("no available TypeSafe API credits")
after the ~7,500-token pilot; every request since has failed the same way. Top up at
<https://console.typesafe.ai/settings/billing> — the whole W1 + W2 run is ≈ 7.4 M
input tokens ≈ **$0.31** (W1 5.9 M ≈ $0.25, W2 posts 1.4 M ≈ $0.06, W2 PDFs < $0.01),
so $5 covers it many times over. Everything that does not need the API is done:

- `scripts/jev_client.py` — key from `.env`, pinned `jev-1.13.0`, backoff on 429/529,
  **402 fails fast**, sqlite cache (`worklists/data/jev/cache.sqlite`, resumable),
  JSONL audit log per task, thread-pool `ask_many`.
- `scripts/airfields_freeman_dates_jev.py` (W1) — dry-run: all 2,822 entries have a
  narrative + candidates. `--merge-only` with no answers reproduces the live
  `airfields.json` exactly (identity check passed), so every change in the real run is
  Jev-driven. Merge rule implemented as J2 plus three rules the labeling forced:
  a regex `conflict` row takes a consistent confident Jev pair; a weak regex end
  (`last_seen`) yields to a confident closure or a confident *later* gone-by/last-seen;
  an OurAirports `open` whose narrative states a closure is flagged
  (`open_but_narrative_closed`) but never auto-changed. Candidate fix: two-digit slash
  dates ≤ 26 now offer the 19yy reading too (`5/1/25` was only ever 2025, so 1920s
  Airway-Bulletin evidence could not be chosen).
- **W1b labeled sample: done** — 40 rows, stratified over the population Jev will
  change (20 end-blank, 5 start-blank, 5 both-blank, 5 conflict, 5 weak-end), read
  in full and labeled by Claude on 09-18 → `worklists/data/jev/airfields_dates_labels.csv`
  (alternates in `note`). Findings while labeling: regex `closed_stated` can be wrong
  (Heber Springs 2nd location got the *original* field's 1934 closure; Temco-Garland's
  1984 is "depicted as abandoned"), `last_seen` ends can be decades early (Lone Star:
  chart 1968, flown into the late 1990s), and an OA `open` can be a neighbour
  (Thompson Field VA). `--labels` prints per-question accuracy by confidence band.
- `scripts/atc_posts_jev.py` (W2) — dry-run: 1,333 posts, candidates found on 805
  (idents) / 1,045 (places) / 1,186 (dates). Gazetteer built from the 1988 NASR file
  (17,648 airports; **212 FSS idents with an on-airport home** — an authoritative
  1988 FSS location table — 223 FSS names), OurAirports (39,525 codes), and a 24-row
  ARTCC→city table (the sampler's `ARTCCFAC.DAT` is route segments). Join verified on
  ANB/MOB/SAN/TAL/ABQ/WJF/TPH/UMM/ZAN. PDF catalog **done in code**: 1,405 PDFs, 1,333
  periodical issues (1,286 dated from the filename, 1,267 to the month), 454 with a
  text layer, 46 standalone documents with text → Jev.

**Run book (after credits):**
```
~/venv/bin/python scripts/airfields_freeman_dates_jev.py \
  --tree ~/archives/airfields-freeman-full/snapshot-2026-08-21/www.airfields-freeman.com \
  --dated ~/archives/airfields-freeman-full/airfields_2026-08-21_r0914_dated.csv \
  --out-csv ~/archives/airfields-freeman-full/airfields_2026-08-21_r0914_jev_dated.csv \
  --out-geojson ~/archives/airfields-freeman-full/airfields_2026-08-21_r0914_jev_dated.geojson \
  --report --labels worklists/data/jev/airfields_dates_labels.csv
# read the accuracy-by-band table + review CSV, adjust ADOPT_MIN / ADOPT_OVER_WEAK,
# then `--merge-only` re-merges from the cache for free; cp the geojson to airfields.json
# and publish with wrangler r2 object put charts/sectionals/airfields.json (memory recipe)
~/venv/bin/python scripts/atc_posts_jev.py --report
# -> worklists/data/atc/posts_meta.jsonl, posts_index.json, pdfs.jsonl; hand-check 50
```

### W1. Freeman airfield dates — fill the blanks, fix the conflicts (viewer pins + outreach C5)

`airfields.json` (2,822 entries) drives pins that appear/disappear with the timeline.
Today **304 have no end year, 105 no start year, 252 are `status=unknown`, 8 are
`conflict`** — all from `airfields_freeman_dates.py`'s regex rules. The pilot showed
Jev resolving exactly those cases from the same narrative.

- [x] W1a. `scripts/airfields_freeman_dates_jev.py`: per entry, state =
      `{airfield_name, narrative}` (the entry segment from `extract_page(..., keep_segment=True)`,
      capped ~9k chars), candidates = `segment_years()` deduped + `not_stated`. Five
      questions in one request, wording as piloted, with two fixes learned there:
      `last_evidence_year` must exclude "remains/traces/outline still visible"
      (Lost Hills picked a 2025 remnants photo at 0.33), and `earliest_evidence_year`
      should say "dated" evidence (undated photos → `not_stated` is right).
      Questions: `built_year`, `earliest_evidence_year`, `closed_year`,
      `last_evidence_year` (Choice), `still_open` (Noul).
- [x] W1b. Hand-labeled sample (40 rows, stratified over the rows Jev will change —
      regex-confident rows get the free agreement test instead) → thresholds after the run.
- [ ] W1c. Run all 2,822 (~7 M tokens ≈ $0.30, minutes). Merge per J2 into the dated
      CSV/GeoJSON; new columns `start_basis`/`end_basis` gain a `jev` value and a
      `*_conf`. Regenerate `airfields.json`; the viewer's pin logic is unchanged.
- [ ] W1d. Report: how many blanks filled, how many regex values contradicted, the
      review-queue size. Feeds **09 C5** directly: era-correct deep links for the
      Freeman outreach need exactly these years.

### W2. ATC collection metadata layer (F1–F3 inputs, B1 cross-links; ships nothing in `/atc/` yet)

1,333 WordPress posts (~206k content tokens total) carry only category tags and, for
993 "Facilities" posts, a city/state from the old facility-photos page. No post has a
machine-readable facility ident, facility type, date, or coordinates — which is what
F1 (photos on the map), F3 (opening/closing timeline) and B1 ("ATC material for this
airport") all need.

- [x] W2a. `scripts/atc_posts_jev.py` (built; run pending credits): state = `{title, categories, entry_text}`
      (+ first-page text for the 1,405 PDFs via existing extraction, capped). Candidates
      from regex: 3-letter/4-letter idents in title/text, `Month D, YYYY` and bare years,
      `City, ST` spans. Questions per post: `page_kind` (Choice, piloted set),
      `facility_type` (fss / tower / center / other), `facility_ident` (Choice over
      found idents + none), `city_state` (Choice over found spans + none),
      `opened_date` / `closed_date` (Choice over date candidates + not_stated),
      `photo_year` (Choice over years + not_stated — the year the photo/class depicts),
      `about_one_facility`, `about_a_person` (Nouls). Cost ≈ $0.10 for posts, ≈ $0.15
      for PDFs.
- [x] W2b. Join in code (built, verified on 9 idents): ident → coordinates (OurAirports / `historical-data`
      airport lifespan CSV / faa1988), city+state → the same. Output
      `worklists/data/atc/posts_meta.jsonl` and a public
      `atc/posts_index.json` (ident, kind, type, dates, coords, href, preview image).
- [ ] W2c. Consumers, in order: **B1 airport pages** (list the posts whose ident or
      city matches — the fusion 09 asks for), **F1** viewer layer (facility photo pins,
      era-gated by `photo_year`), **F3** (tower/FSS open–close spans as a timeline
      layer). `/atc/` "related pages" and category landings wait for 2026-10-30 (J3).
- [ ] W2d. Acceptance: hand-check 50 posts across kinds; the facilities.json 993
      already have city/state — agreement with those is the free regression test.

### W3. Airport-page pilot (09 B1) — entity alignment for the cross-links

Choosing the 25–50 pilot airports and computing "first/last chart appearance" is
geometry over `timeline_data.json` — code. Jev's narrow job is the join nobody can
regex: is this Freeman entry / ATC post / historical-data record *this* airport?

- [ ] W3a. Per airport, code builds a candidate set (name/ident/city/state string
      similarity + distance ≤ 15 km). One Score per pair, three levels as the
      [entity-alignment cookbook](https://docs.typesafe.ai/cookbooks/entity_alignment.md)
      recommends: `same_airport` / `different` / `needs_curator` — the levels *are*
      the three actions, so there is no threshold to fit. ~1,500 pairs ≈ $0.04.
- [ ] W3b. `needs_curator` rows → a CSV I review; `same_airport` → the page's
      "Elsewhere on archive.aero" block. Thin-content guard (B5) stays in code.
- [ ] W3c. Optional: a citation-check pass over each generated page's factual
      sentences ("first appeared on the 1938 Los Angeles sectional") against the
      timeline row that produced them — Choice `supported / unsupported / partial`.
      Jev verifies claims; it never writes the page copy.

### W4. Viewer "Find" box — natural-language go-to (after J1; after the B pilot ships)

The viewer has no search. A visitor who wants "Boston in 1958" must pan and scrub.

- [ ] W4a. Code first: a gazetteer (151 chart locations + 58 groups with their old
      names, airports from `historical-data`, Freeman entries, states), a year/decade
      regex, and a `?date=&lat=&lng=&zoom=` deep link. Most queries never need Jev.
- [ ] W4b. Jev for the rest, behind `POST /api/find` on the `tiles` Worker (secret
      via `wrangler secret put TYPESAFE_API_KEY`; the key never reaches the browser):
      `intent` (Choice: go_to_place / place_at_date / oldest_here / newest_here /
      not_a_place), `place` (Choice over the top-10 gazetteer hits + none),
      `era_hint` (Choice over decades + not_stated). ~1.2k tokens ≈ $0.00005 per
      query; cache by normalized query in the Worker cache/KV; per-IP rate limit;
      log to Analytics Engine so it can be measured like tiles.
- [ ] W4c. Ship behind a flag, measure use for two weeks, keep only if used.

### W5. Analytics — a defensible automation label for the A4 denominator

09 A4 needs a human-audience denominator; today the only signal is a
`bot|crawler|spider|slurp` regex (53.3% of visits) and the honest note that the
rest "must not be labeled human".

- [ ] W5a. Per *distinct* UA string in the weekly export (thousands, not
      millions): Choice `browser / self_declared_crawler / headless_or_automation /
      library_or_script / unknown` with confidence; join back to visit counts in
      code. ≈ $0.04 per run.
- [ ] W5b. Report as "identified automation (Jev-labeled, conf ≥ t)" beside the regex
      lower bound — never as "humans". Goes into the E1 sponsor summary with the
      method disclosed.

### W6. Catalog & salvage (when the work arrives, not before)

- [ ] W6a. When the catalog gains a non-sectional schema (roadmap: TACs; attic holds
      868 ACASIS charts across 4 families plus 17 TACs), classify attic filenames /
      collar OCR into chart family + location via Choice over the catalog's location
      list — the "WAI variant-name" problem from the SD-card batches.
- [ ] W6b. Hunt worklist 07 stays code-driven (findings are already verified); revisit
      only if a new unverified source dump appears.

### Not for Jev (so nobody tries)

Georef/GCP checks, cutlines, coverage %, date arithmetic and END-ESTIMATED logic
(05), dedupe of scans (visual), PMTiles/caching, airspace geometry, page copy or
`og:image` generation, anything the `master_dole_v2.csv` loader already answers
exactly.

---

## Timeline (slots into 09 §3)

| Window | Work |
|---|---|
| **Sep 19–22** | `scripts/jev_client.py`; W1a–W1d (Freeman dates); W2a–W2b (ATC metadata, offline). Both are inputs the B1 pilot and the C5 outreach want. |
| **Sep 23–30** | W3a–W3b inside the B1 pilot build; W5a alongside the A4 export. |
| **After the B pilot indexes** | W4 (J1 permitting), flagged, measured. |
| **Oct 30+** | W2c's `/atc/`-side consumers, per 08 §5. |

## Provenance / how to regenerate

- Pilot: [data/jev/pilot_2026-09-18.py](data/jev/pilot_2026-09-18.py) (reads the key
  from `.env`; plain HTTP, no SDK). SDK if wanted: `~/venv/bin/pip install typesafe-sdk`
  (`from typesafe_sdk import TypeSafeClient, Choice, Noul, Score`).
- Sizing: 1,333 posts / 206k content tokens from a category-tag scan of
  `/Volumes/projects/atchistory_build/site/*/index.html`; airfields gaps from
  `airfields.json` property counts; pricing and limits from
  <https://docs.typesafe.ai/models.md> on 2026-09-18.
- Skill: `typesafe@typesafe-ai` plugin v0.5.7 (installed; `/typesafe:typesafe-ai`).
