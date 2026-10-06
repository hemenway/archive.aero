# 16 — Jev (TypeSafe) as programmable judgment for archive.aero

**Updated 2026-09-21; plan written 2026-09-18.** Companion to [09](09_growth_and_revenue.md) (growth) and
[08](08_atchistory_migration.md) (ATC migration). Jev is a *System One* model: it
returns typed answers (a choice, a yes/no probability, a score) over state you send
it — it does not generate text. Docs: <https://docs.typesafe.ai/llms.txt>.

**Current state:** W1 and W2 ran locally on 2026-09-21. Freeman dates are merged
into `airfields.json`; the ATC post/PDF indexes exist under `worklists/data/atc/`.
Publication of the airfield data and all W2 consumers remain open. Next: finish
the planned validation samples (40 Freeman labels and about 40 ATC spot checks
recorded so far, against the ≥50-row target), review flagged disagreements, then
publish the airfield data and build the B1 airport-page pilot. W3–W5 have not
started; W6 is conditional future work. The results below are local evidence,
not a fresh check of production.

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
| **Key lives in `.env` as `TYPESAFE_API_KEY`** (gitignored; the Python SDK reads that name) and, for any live endpoint, in a Worker secret via `wrangler secret put`. Never in a script, never in `index.html`/`src/`. | Public repo. The key pasted into a chat transcript on 09-18 was deactivated by the owner on 2026-10-01; its API request now returns HTTP 403. Replacement in the local `.env` is pending. Never paste credentials into chat. |
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

## Decisions

| # | Decision | Recommendation |
|---|---|---|
| J1 | Is a small live endpoint acceptable for W4 (the viewer "Find" box), given the README's "no application backend" claim? | **Yes, as a stateless proxy in the existing `tiles` Worker** (the same shape as the range proxy — no database, no sessions), rate-limited and cached. If that feels like a backend, W4 waits; W1–W3, W5 need nothing live. |
| J2 | How are Jev and regex dates reconciled? | **Implemented 2026-09-21:** retain the regex columns in the dated CSV, then adopt by rule. Fill blanks/conflicts above threshold; retain disagreements for review, with the documented weak-basis and closure-misfire exceptions below. |
| J3 | Publish Jev-derived ATC metadata inside `/atc/` before the 2026-10-30 stabilization date? | **No.** Build the index now (offline JSON), consume it first from archive.aero-side surfaces (airport pages, viewer layers), and only add `/atc/` navigation after 08 §5 releases it. |

---

## Workstreams (ordered by payoff ÷ effort)

### Results 2026-09-21 — W1 and W2 ran; airfields.json not yet published

**W1 (Freeman dates)** — 2,822 entries, 9.09 M input tokens, **$0.38**, 139 s at 8 workers.
Answers in `worklists/data/jev/airfields_dates_answers.jsonl` (keyed `url|name`; cache
makes every re-merge free). Policy after the labeled evaluation (40 rows, per-question
accuracy by confidence band, `--labels`): `built` 20/20 and `earliest` 14/15 at ≥0.9,
`closed` 18/19 at ≥0.9, `last_seen` 23/24 at ≥0.7 — but `gone_by` 18/40: asked for "the
earliest chart on which it was *no longer* depicted", Jev returns the earliest chart that
didn't show it, often **before the field existed** (Amboy 1941, Jackass 1944, Thompson
1930), or a 2018 "no trace remains" aerial for a field last seen in 1919 (Dominguez).
Ordering years is arithmetic, so it moved into code: per-question floors of **0.7**
(`Q_MIN`), gone-by only if later than every evidence of existence *and* within
`GONE_BY_WINDOW = 15` years of a confident last sighting. Derived accuracy on the labels:
start 33/35, end 23/28 within 2 y (the misses are convention: my label said "gone by
1991", Jev said "last seen 1987").
Two regex traits surfaced by the disagreement sample: (1) "closed between 1976-79" —
the regex keeps the first year, Jev the last: 254 such pairs are now counted as agreement
(`end=agree_range`), regex value kept; (2) **"still depicted as an abandoned airfield on
2002 charts" matches the regex closure verb and steals the chart year** — detected in
code (`regex_closure_misfire`) and yielded to a ≥0.9 Jev closure (28 rows). In a sample
of 12 genuine ≥0.9 disagreements Jev was right 8, regex 3, 1 unknown; the rest stay
regex and sit in `airfields_dates_review.csv` (453 end + 298 start `kept_regex`).
The review CSV has **874 audit rows**: 751 retained date disagreements, 25 status
flags, and 98 adopted overrides/conflict resolutions (including 6 start-year
weak-basis overrides). The adopted rows are an audit trail, not unresolved tasks.
**Result:** end years 2,518 → 2,666, start 2,717 → 2,798, `unknown` 252 → 104; adopted:
142 end fills, 81 start fills, 58 weak-end overrides, 28 misfire overrides, 6 conflict
resolutions; 25 status flags (15 regex-gone/Jev-open, 10 OA-open/narrative-closed).
Outputs: `~/archives/airfields-freeman-full/airfields_2026-08-21_r0914_jev_dated.{csv,geojson}`
(CSV keeps the regex columns plus `jev_*` and `*_conf`; GeoJSON keeps the compact
viewer properties), with the GeoJSON copied to the repo `airfields.json`;
the pre-Jev copy is `~/archive.aero-attic/airfields_pre_jev_2026-09-21.json`.
**Not yet published** — `cd worker && npx wrangler r2 object put
charts/sectionals/airfields.json --file ../airfields.json --content-type application/json --remote`.

**W2 (ATC metadata)** — 1,333 posts (1.92 M tokens, **$0.08**) + 46 standalone PDFs
($0.003). `/Volumes/projects` was a stale SMB mount, so the post HTML came from the
`atc-site` bucket via rclone into `~/archives/atc-site-posts/` (1,361 pages, 84 MB); the
PDF scan cache from 09-18 stood in for the tree. Kinds: 617 facility_photo, 261
class_photo, 246 facility_history, 75 airway_infrastructure, 37 publication, 29 map, 27
roster, 10 personal_story, 31 other; facility type 885 fss / 216 office_or_academy / 6
tower / 2 center. 685 idents chosen (611 at ≥0.5; a random dozen all correct), 1,040
places, 1,065 depicted years, 258 opened + 120 closed dates. **885 posts carry
coordinates**: 439 via the 1988 FSS-on-airport table, 159 via 1988 airport ident, 207/76
via city, 4 via OurAirports ident. facilities.json regression: state 98 %, city+state 92 %
(most "misses" are facilities.json labels that are facility names — "Huron FSS" — where
Jev returned the city). Category→kind is coherent (Class Photos 213/214). Outputs in
`worklists/data/atc/`: `posts_meta.jsonl` (raw answers + join), `posts_index.json`
(1,333 records, 572 KB, public-shaped), `pdfs.jsonl` (1,405 records, including 1,333
periodical entries; 1,286 have an issue year and 1,267 an issue month parsed from
filenames). Consumers (W2c) not started; nothing under `/atc/` changed.

### W1. Freeman airfield dates — fill the blanks, fix the conflicts (viewer pins + outreach C5)

`airfields.json` (2,822 entries) drives pins that appear/disappear with the timeline.
The **pre-run regex baseline** had 304 blank end years, 105 blank start years,
252 `status=unknown` and 8 conflicts. The local 2026-09-21 output has **156 blank
end years, 24 blank start years, 104 unknown and 0 conflicts** (plus 52 open).

- [x] W1a. `scripts/airfields_freeman_dates_jev.py`: per entry, state =
      `{airfield_name, narrative}` (the entry segment from `extract_page(..., keep_segment=True)`,
      capped at 14,000 chars), candidates = `segment_years()` deduped + `not_stated`. Six
      questions in one request, wording as piloted, with two fixes learned there:
      `last_evidence_year` must exclude "remains/traces/outline still visible"
      (Lost Hills picked a 2025 remnants photo at 0.33), and `earliest_evidence_year`
      should say "dated" evidence (undated photos → `not_stated` is right).
      Questions: `built_year`, `earliest_evidence_year`, `closed_year`,
      `gone_by_year`, `last_evidence_year` (Choice), `still_open` (Noul).
- [x] W1b. Hand-labeled sample (40 rows, stratified over the rows Jev will change —
      regex-confident rows get the free agreement test instead) → thresholds after the run.
- [x] W1c. Ran 2026-09-21 (9.1 M tokens, $0.38). Merged per J2 (+ the range / misfire /
      window rules above); `airfields.json` regenerated locally, publish pending.
- [x] W1d. Report (above): 142 end + 81 start fills, 98 adopted overrides/conflict
      resolutions, 751 date disagreements + 25 status flags to review. Feeds **09 C5**:
      era-correct deep links for the
      Freeman outreach need exactly these years.
- [ ] W1e. Extend the 40-row labeled sample to the planned ≥50 and review the
      retained disagreements/status flags; record any threshold changes and
      re-merge from cached answers if needed.
- [ ] W1f. Publish the verified `airfields.json` to R2 using the command above,
      then verify the served data and representative viewer pins.

### W2. ATC collection metadata layer (F1–F3 inputs, B1 cross-links; ships nothing in `/atc/` yet)

1,333 WordPress posts (~206k content tokens total) carry only category tags and, for
993 "Facilities" posts, a city/state from the old facility-photos page. That was
the input baseline; the local indexes now supply facility, date and coordinate
metadata for F1 (photos on the map), F3 (opening/closing timeline) and B1 ("ATC
material for this airport"). Those consumers have not shipped.

- [x] W2a. `scripts/atc_posts_jev.py` (ran 2026-09-21): state = `{title, categories, entry_text}`
      (code calls the content field `text`). The PDF inventory has 1,405 records;
      Jev processed capped first-page text for 46 standalone PDFs. Candidates
      from regex: 3-letter/4-letter idents in title/text, `Month D, YYYY` and bare years,
      `City, ST` spans. Questions per post: `page_kind` (Choice, piloted set),
      `facility_type` (fss / tower / center / office_or_academy / other_or_none),
      `facility_ident` (Choice over found idents + none), `place` (Choice over found spans + none),
      `opened_date` / `closed_date` (Choice over date candidates + not_stated),
      `depicted_year` (Choice over years/decades + not_stated — the year the subject depicts),
      `about_one_facility`, `about_a_person` (Nouls). Actual run cost and PDF scope
      are recorded in the results above.
- [x] W2b. Join in code (885 posts with coordinates): ident → coordinates from
      the OurAirports snapshot / 1988 `AIRPORTS.DAT` (plus ARTCC mapping),
      city+state → the same gazetteer. Output
      `worklists/data/atc/posts_meta.jsonl` and the public-shaped local
      `worklists/data/atc/posts_index.json` (ident, kind, type, dates, coords, href,
      preview image). No public index URL has been published.
- [ ] W2c. Consumers, in order: **B1 airport pages** (list the posts whose ident or
      city matches — the fusion 09 asks for), **F1** viewer layer (facility photo pins,
      era-gated by the index's `year`), **F3** (tower/FSS open–close spans as a timeline
      layer). `/atc/` "related pages" and category landings wait for 2026-10-30 (J3).
- [x] W2d. Initial review: ~40 posts read across kinds during candidate design and result
      review (idents, years, closures all correct); facilities.json regression 98 % / 92 %.
- [ ] W2e. Complete the planned ≥50-post acceptance sample across kinds and
      coordinate sources before publishing a consumer; preserve the results.

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
| **Sep 19–22** | W1a–W1d and W2a–W2b complete locally. Remaining: validation/review (W1e, W2e) and airfields publication (W1f). |
| **Sep 23–30 (proposed)** | W3a–W3b inside the B1 pilot build; W5a alongside the A4 export. |
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
- Repository skill: `typesafe-ai` in `.agents/skills/typesafe-ai/SKILL.md`;
  installation provenance is recorded in `skills-lock.json`.
