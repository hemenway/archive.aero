# Worklist 16 — cached-output acceptance review, 2026-10-13

**Publication remains pending.** The expanded sample found incorrect public-facing
dates and coordinates. Deterministic fixes are implemented and tested; no inference
calls, uploads, deployments, catalog edits, or replacement of `airfields.json` and
the existing ATC indexes were performed. All results below concern the saved
2026-09-21 inputs and responses, not current operating status or a live-site audit.

## Evidence and sampling

The Freeman sample is now **50 entries**: the previous 40 labels plus 10
Codex-reviewed narratives, explicitly marked as such in the label notes. The
extension chooses the five shortest retained start disagreements and five shortest
retained end disagreements outside the original labels. It is a diagnostic sample,
not a random estimate of collection-wide accuracy. Exact-year labels keep ambiguous
opening intervals blank; notes preserve conflicting evidence and interval conventions.

The ATC review covers **69 posts**, selected with seed `20261013`: five per kind,
then up to three additional posts per coordinate-source class, then five posts with
the lowest combined opening/closing confidence. All nine kinds and all seven
coordinate-source classes present in the original index are covered. The reviewed
material is each saved title and complete capped entry text actually given to Jev.
Image pixels, linked documents and material beyond the input cap were not reviewed.

Machine-readable evidence is intentionally gitignored under `worklists/data/jev/`:

- `airfields_dates_labels_2026-10-13.csv` — all 50 labels; original labels unchanged.
- `airfields_acceptance_extension_2026-10-13.json` — ten narratives, requests and raw answers.
- `airfields_dates_evaluation_2026-10-13.txt` — confidence-band evaluation.
- `airfields_replay_2026-10-13.json` — cached replay actions, review rows and remaining inverted dates.
- `airfields_status_review_2026-10-13.json` — all 25 status flags with source narratives;
  preserved for review, **not resolved**.
- `atc_acceptance_sample_2026-10-13.json` — all 69 saved inputs and original metadata.
- `atc_acceptance_review_2026-10-13.csv` — per-post findings, source URL and input-state hash.
- `atc_join_replay_2026-10-13.json` — eight before/after coordinate changes.

## W1 — Freeman dates

The original local output has **six `start_year > end_year` pairs**, despite its
status summary containing zero `conflict` values. Three came from Jev adoption:

| Entry | Invalid local interval | Cause fixed |
|---|---|---|
| [Heber Springs, second location](https://www.airfields-freeman.com/AR/Airfields_AR_N.htm#heber2) | 1939–1934 | A start adoption could cross a retained regex end. |
| [Delamar Lake](https://www.airfields-freeman.com/NV/Airfields_NV_NE.htm#delamar) | 1978–1949 | A start fill could cross a retained regex end. |
| [Ushi Point / Tinian North Field](https://www.airfields-freeman.com/HI/Airfields_W_Pacific.htm#northfield) | 1966–1947 | An end override could cross a retained regex start. |

The script now checks each proposed year against the year that will actually
survive the merge, rejects future-year adoptions, and records rejected proposals
in the review queue. A consistent pair can still replace an explicitly conflicted
regex pair. A start fill or weak-start override is checked after the end decision,
so a valid 1939–1945 pair can replace a weak 1934 end in the same merge; a strong
1934 end that survives still prevents adoption of a 1939 start. The final cached
replay has 878 review rows, including two rejected start adoptions. Em-dash closure intervals, such as
[Rock Port's 2003—2015](https://www.airfields-freeman.com/MO/Airfields_MO_NW.htm#rockport),
now receive the same agreement treatment as hyphens and en dashes.

Replaying all 2,822 cached answers leaves **three pre-existing inverted intervals**:

| Entry | Retained interval | Source of retained end |
|---|---|---|
| [Florence Junction, second location](https://www.airfields-freeman.com/AZ/Airfields_AZ_Phoenix_SE.htm#threepoint222) | 1980–1957 | `lifespan_csv` |
| [Maricopa, second location](https://www.airfields-freeman.com/CA/Airfields_CA_Bakersfield.htm#maricopa2) | 1956–1953 | `lifespan_csv` |
| [Sanderson, second location](https://www.airfields-freeman.com/TX/Airfields_TX_BigBend.htm#sanderson) | 1965–1956 | `lifespan_csv` |

Those original records need source-specific reconciliation. The 10 new source
reviews also show why neither side of every disagreement can simply win:

- [Hanford Auxiliary](https://www.airfields-freeman.com/WA/Airfields_WA_C.htm#hanfordaux):
  1942 dates the nuclear site's establishment; the first located depiction of this
  airfield is 1964. A nearby airfield is a different entity.
- [Gandy](https://www.airfields-freeman.com/CA/Airfields_CA_Fresno_SW.htm#gandy):
  the retained 2024 closure is actually a no-trace aerial view. Closure is described
  only as the mid-to-late 1990s; 1972 is a last-photo bound, not the closure.
- [Oasis](https://www.airfields-freeman.com/NV/Airfields_NV_NE.htm#oasis):
  the narrative says a 2024 photo shows no runway, but a 2026 chart still depicts
  an unverified airfield. This is conflicting evidence, not a reliable closure.

On 50 labels, the derived start is within two years of the chosen label in 43/45
cases where Jev returns a year; end is 31/37. High-confidence raw selections
(`>=0.9`) score built 24/24, earliest 23/24, closed 22/23, last-seen 13/13;
gone-by is only 4/7. These small, selected samples do not establish publication
accuracy. Thresholds are unchanged. Existing gone-by ordering/window guards remain.

**Remaining:** reconcile the three original inverted pairs; finish the remaining
retained-disagreement and all 25 status reviews; apply decisions with provenance;
rebuild the local output; then verify pins before publishing. The larger labeled
sample alone does not complete W1e.

## W2 — ATC metadata

The 69-row review records 24 rows with no contradiction in the inspected fields
and 45 with a finding or limitation. The latter includes benign multi-date pages,
approximate city coordinates and ambiguous kind labels, so it is **not an error
rate**. The principal blockers are:

- **Historical callsign reuse:**
  [St. Louis KCQ, 1929](https://archive.aero/atc/st-louis-radio-station-1929)
  joined to Chignik Lake, Alaska. [Montezuma KCEV](https://archive.aero/atc/montezuma-radio-bldg-abt-1944)
  joined to Mettel Field, Indiana despite the text saying Iowa. The join no longer
  retries an ident after discarding its stated-state constraint. It tries the city
  or leaves coordinates blank.
- **Hospital heliports as city proxies:** six posts selected hospital heliports
  merely because the gazetteer lacked a suitable airport for the exact city spelling.
  These are now left unplaced. The replay changes eight coordinate pairs total:
  Montezuma gains an Iowa city proxy; seven records become unplaced. City/airport
  joins still describe approximate locations and do not prove a historical building site.
- **Missing candidate years:** five sampled class pages give only short numeric
  dates, such as [Class 83001's 02/18/83](https://archive.aero/atc/class-83001).
  Code now supplies normalized years from numeric dates and recognizes curly
  apostrophes in full decades such as `1980’s`. Existing cached answers do not
  acquire the newly available options without new inference or curated corrections.
  Abbreviated decades such as `early 70’s` remain unresolved.
- **Unsupported operating dates:** 13 sample rows export an unsupported opening,
  and six an unsupported closing (some overlap). Examples include
  [Akron's 1985 photograph](https://archive.aero/atc/akron-fss-1985) exported as
  opening at confidence 0.08, and [Montezuma](https://archive.aero/atc/montezuma-radio-bldg-abt-1944)
  exporting 1947 as closure at 0.90 when it ends the employee's service period.
  [Kankakee](https://archive.aero/atc/kankakee-afss-building) describes transfer to
  a contractor, not proven closure. A higher threshold alone cannot settle these.
- **Loss of uncertainty:** the compact index previously omitted date confidence,
  raw year/decade strings, kind confidence and the single-facility judgment. The
  builder now retains those fields plus join provenance and model. These are still
  unverified model selections; this change does not make them safe operating spans.

**Remaining:** apply source-specific corrections or add an evaluated evidence gate
for operating dates; decide how ambiguous kinds and date intervals are represented;
resolve candidate gaps; regenerate and review the index. W2e's sample is complete
and preserved, but **acceptance failed**. No B1/F1/F3 consumer should publish these
unverified spans as established facts. W3 entity alignment has not been attempted.

## Verification

`~/venv/bin/python -m unittest discover -s tests -p test_jev_metadata.py -v`
passes all 11 regression tests. The full cached replay used the original regex CSV,
saved narratives and answers in memory and wrote only the dated evidence files
above. Current [TypeSafe confidence guidance](https://docs.typesafe.ai/confidence)
was consulted: confidence describes distribution concentration, not correctness.
