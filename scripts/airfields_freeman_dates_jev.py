#!/usr/bin/env python3
"""Worklist 16 W1 — fill and check the airfields-freeman operating years with Jev.

`airfields_freeman_dates.py` mines start/end years from each entry's narrative
with regexes (96 % start / 91 % end coverage). What it leaves blank, `conflict`
or `unknown` is what this pass resolves: the same narrative segment goes to Jev
as state, the regex year list becomes the candidate set (plus `not_stated`), and
Jev *selects* — it never generates a year the text does not contain.

Five questions per entry (one request, evaluated in parallel):
  built_year, earliest_evidence_year, closed_year, gone_by_year,
  last_evidence_year (Choice over candidate years) and still_open (Noul).

Merge rule (worklist 16 decision J2 — Jev sits beside the regex, adoption by rule):
  * regex value present and Jev agrees            -> keep regex, note agreement
  * regex blank / conflict, Jev confident          -> adopt Jev (basis jev_*)
  * both present, disagree                         -> review.csv; regex stays unless
    its basis is weak (last_seen / after_absence / faa1988_listed) and Jev's is
    strong (jev_closed / jev_built above ADOPT_OVER_WEAK)
  * regex `conflict` (its own end < start)         -> a consistent, confident Jev
    pair replaces both years (the regex start was the suspect one)
  * end < start, year in the future                -> never adopted (code guard)
Thresholds are named constants below; `--report` prints the agreement matrix
that sets them (tune on the first full run, then re-run --merge-only for free
from the cache).

Usage (regex-dated CSV from airfields_freeman_dates.py is the input):
  ~/venv/bin/python scripts/airfields_freeman_dates_jev.py \
      --tree ~/archives/airfields-freeman-full/snapshot-2026-08-21/www.airfields-freeman.com \
      --dated ~/archives/airfields-freeman-full/airfields_2026-08-21_r0914_dated.csv \
      --out-csv  ~/archives/airfields-freeman-full/airfields_2026-08-21_r0914_jev_dated.csv \
      --out-geojson ~/archives/airfields-freeman-full/airfields_2026-08-21_r0914_jev_dated.geojson \
      [--only blank|all] [--limit N] [--dry-run] [--report] [--merge-only]

Then `cp <out-geojson> airfields.json` and publish per the airfields memory
(wrangler r2 object put charts/sectionals/airfields.json ...).
"""
import argparse
import csv
import datetime
import json
import os
import posixpath
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import airfields_freeman_dates as D  # noqa: E402
import airfields_freeman_extract as X  # noqa: E402
from jev_client import DATA_DIR, JevClient, JevError, choice, noul  # noqa: E402

# ---------------------------------------------------------------- policy
ADOPT_MIN = 0.60         # Jev fills a blank only at/above this confidence
ADOPT_OVER_WEAK = 0.90   # Jev overrides a weak regex basis only at/above this
STILL_OPEN_MIN = 0.80    # noul level that flags a regex "gone" for review
WEAK_START = {"after_absence", "faa1988_listed", "lifespan_csv"}
WEAK_END = {"last_seen", "faa1988_listed", "lifespan_csv"}
NOT_STATED = "not_stated"
SEG_CAP = 14000          # chars of narrative sent (≈3.5k tokens; longest entries)
THIS_YEAR = datetime.date.today().year

ANSWERS_PATH = DATA_DIR / "airfields_dates_answers.jsonl"


# ---------------------------------------------------------------- questions
def questions_for(years):
    opts = {str(y): None for y in years}
    opts[NOT_STATED] = "The narrative never states this"
    return {
        "built_year": choice(
            "In which candidate year does `narrative` say the airfield named in "
            "`airfield_name` was built, constructed, established or opened? Pick that "
            "year. Pick not_stated if the narrative never gives a year for when the "
            "airfield itself was built or opened — a hangar, a runway, a later business "
            "or the estate it sat on does not count, and 'not yet depicted on a <year> "
            "chart' is not a build date.", opts),
        "earliest_evidence_year": choice(
            "Which candidate year is the EARLIEST dated chart, map, aerial photo, "
            "directory listing, advertisement or other dated evidence cited in "
            "`narrative` that shows the airfield named in `airfield_name` as existing? "
            "Pick that year. Pick not_stated if no dated evidence shows it (an undated "
            "photo does not count, and 'not yet depicted on the <year> chart' means it "
            "was absent, not present).", opts),
        "closed_year": choice(
            "In which candidate year does `narrative` say the airfield named in "
            "`airfield_name` itself closed, was abandoned, or ceased operation? Pick "
            "that year; if a range such as 'between 1955 & 1957' is given, pick the "
            "later year. Pick not_stated if the narrative never states a closing year "
            "for the airfield itself — a hangar, business or runway closing does not "
            "count, and 'no longer depicted on a <year> chart' is not a stated closure.",
            opts),
        "gone_by_year": choice(
            "Which candidate year is the EARLIEST chart, map, aerial photo or listing "
            "in `narrative` on which the airfield named in `airfield_name` was no "
            "longer depicted, or on which its site had already been built over or "
            "returned to other use (houses, roads, fields, a mall)? Pick that year, or "
            "not_stated if the narrative never gives one.", opts),
        "last_evidence_year": choice(
            "Which candidate year is the LATEST dated chart, map, aerial photo, "
            "directory listing or other dated evidence in `narrative` that shows the "
            "airfield named in `airfield_name` still existing as an airfield (still "
            "depicted, still listed, runways still intact, or still in operation)? Pick "
            "that year. Do not count photos or views described as showing only "
            "remains, traces, an abandoned site, or the site after the airfield had "
            "closed or been redeveloped. Pick not_stated if no dated evidence shows it "
            "as an airfield.", opts),
        "still_open": noul(
            "Does `narrative` say the airfield named in `airfield_name` is still an "
            "operating airfield at the time of writing?",
            true="Described as still open, active, in use or in operation today, or as "
                 "of the most recent date the narrative mentions",
            false="Described as closed, abandoned, gone, redeveloped, or its current "
                  "status is not stated"),
    }


def load_segments(tree):
    tree = os.path.abspath(tree)
    segments = {}
    for path in X.iter_pages(tree):
        rel = os.path.relpath(path, tree)
        rel_url = posixpath.join(*rel.split(os.sep))
        for e in X.extract_page(path, rel_url, keep_segment=True)[0]:
            segments[(e["page"], e["name"])] = e["_segment"]
    return segments


def prep(row, seg):
    seg = re.sub(r"\s+", " ", seg).strip()
    if len(seg) > SEG_CAP:  # keep the head (identity, earliest) and the tail (latest views)
        seg = seg[: SEG_CAP - 4000] + " […] " + seg[-4000:]
    years = {y for _, y in D.segment_years(seg) if 1900 <= y <= THIS_YEAR}
    # segment_years() pivots two-digit years at the current year, so "5/1/25"
    # (a 1925 Airway Bulletin) parses as 2025 and 1925 is never a candidate.
    # Offer the 19yy reading too: Jev picks from context, code never guesses.
    for m in D.SLASHDATE.finditer(seg):
        yy = int(m.group(1))
        if yy <= D.PIVOT_YY:
            years.add(1900 + yy)
    years = sorted(years)
    return seg, years


def wants(row, only):
    if only == "all":
        return True
    # "blank": the rows the regex pass could not settle
    return (not row["start_year"] or not row["end_year"] or row["end_basis"] == "conflict"
            or row["status"] == "unknown")


# ---------------------------------------------------------------- merge
def pick(ans, qid):
    a = ans[qid]
    c = a["choice"]
    return (None if c == NOT_STATED else int(c)), float(a["confidence"])


def jev_columns(ans):
    """Jev's own start/end derivation, mirroring derive() in the regex pass:
    start = built else earliest; end = closed else gone_by else last_evidence."""
    out = {}
    built, cb = pick(ans, "built_year")
    earliest, ce = pick(ans, "earliest_evidence_year")
    closed, cc = pick(ans, "closed_year")
    gone, cg = pick(ans, "gone_by_year")
    last, cl = pick(ans, "last_evidence_year")
    still = float(ans["still_open"]["noul"])
    for k, v in (("jev_built", built), ("jev_built_conf", cb), ("jev_earliest", earliest),
                 ("jev_earliest_conf", ce), ("jev_closed", closed), ("jev_closed_conf", cc),
                 ("jev_gone_by", gone), ("jev_gone_by_conf", cg), ("jev_last_seen", last),
                 ("jev_last_seen_conf", cl), ("jev_still_open", round(still, 3))):
        out[k] = "" if v is None else (round(v, 3) if isinstance(v, float) else v)
    if built is not None and (earliest is None or built <= earliest):
        s, bs, cs = built, "jev_built", cb
    elif earliest is not None:
        s, bs, cs = earliest, "jev_earliest", ce
    else:
        s, bs, cs = None, "", 0.0
    if closed is not None:
        e, be, cend = closed, "jev_closed", cc
    elif gone is not None:
        e, be, cend = gone, "jev_gone_by", cg
    elif last is not None:
        e, be, cend = last, "jev_last_seen", cl
    else:
        e, be, cend = None, "", 0.0
    if s is not None and e is not None and e < s:
        e, be, cend = None, "jev_conflict", 0.0
    out.update({"jev_start_year": s or "", "jev_start_basis": bs, "jev_start_conf": round(cs, 3),
                "jev_end_year": e or "", "jev_end_basis": be, "jev_end_conf": round(cend, 3)})
    return out


def merge(row, jc, review):
    """Apply the J2 rule to one row (in place). Returns the action taken."""
    r_s = int(row["start_year"]) if row["start_year"] else None
    r_e = int(row["end_year"]) if row["end_year"] else None
    j_s = jc["jev_start_year"] or None
    j_e = jc["jev_end_year"] or None
    actions = []

    # -- a regex `conflict` (its end < its start) means one of ITS two years is
    # wrong, so its start is not a trustworthy tie-breaker: a consistent,
    # confident Jev pair replaces both (logged for review).
    if row["end_basis"] == "conflict" and j_s is not None and j_e is not None and j_e >= j_s \
            and jc["jev_start_conf"] >= ADOPT_MIN and jc["jev_end_conf"] >= ADOPT_MIN:
        review.append(_rv(row, "start", r_s, j_s, jc, "adopted_conflict_resolution"))
        row["start_year"], row["start_basis"] = j_s, jc["jev_start_basis"]
        row["end_year"], row["end_basis"], row["status"] = j_e, jc["jev_end_basis"], "gone"
        row["last_known_year"] = ""
        return ["conflict=jev_resolved"]

    # -- start
    if r_s is not None and j_s is not None and r_s == j_s:
        actions.append("start=agree")
    elif r_s is None and j_s is not None and jc["jev_start_conf"] >= ADOPT_MIN:
        row["start_year"], row["start_basis"] = j_s, jc["jev_start_basis"]
        actions.append("start=jev_fill")
    elif r_s is not None and j_s is not None:
        if row["start_basis"] in WEAK_START and jc["jev_start_basis"] == "jev_built" \
                and jc["jev_start_conf"] >= ADOPT_OVER_WEAK:
            review.append(_rv(row, "start", r_s, j_s, jc, "adopted_over_weak"))
            row["start_year"], row["start_basis"] = j_s, jc["jev_start_basis"]
            actions.append("start=jev_over_weak")
        else:
            review.append(_rv(row, "start", r_s, j_s, jc, "kept_regex"))
            actions.append("start=disagree")
    else:
        actions.append("start=none")

    # -- end (an OA-open field has no end by design: leave it alone)
    if row["status"] == "open":
        actions.append("end=open")
    elif r_e is not None and j_e is not None and r_e == j_e:
        actions.append("end=agree")
    elif (r_e is None or row["end_basis"] == "conflict") and j_e is not None \
            and jc["jev_end_conf"] >= ADOPT_MIN:
        s_now = int(row["start_year"]) if row["start_year"] else None
        if s_now is not None and j_e < s_now:
            actions.append("end=jev_conflict")
        else:
            row["end_year"], row["end_basis"], row["status"] = j_e, jc["jev_end_basis"], "gone"
            row["last_known_year"] = ""
            actions.append("end=jev_fill")
    elif r_e is not None and j_e is not None:
        # A weak regex end ("last seen on the 1968 chart") is a lower bound, so a
        # confident Jev closure, or a confident LATER gone-by / last-seen year,
        # is strictly more information and replaces it (logged for review).
        stronger = jc["jev_end_basis"] == "jev_closed" or (
            jc["jev_end_basis"] in ("jev_gone_by", "jev_last_seen") and j_e > r_e)
        if row["end_basis"] in WEAK_END and stronger and jc["jev_end_conf"] >= ADOPT_OVER_WEAK:
            review.append(_rv(row, "end", r_e, j_e, jc, "adopted_over_weak"))
            row["end_year"], row["end_basis"] = j_e, jc["jev_end_basis"]
            actions.append("end=jev_over_weak")
        else:
            review.append(_rv(row, "end", r_e, j_e, jc, "kept_regex"))
            actions.append("end=disagree")
    else:
        actions.append("end=none")

    if row["status"] == "gone" and jc["jev_still_open"] >= STILL_OPEN_MIN:
        review.append(_rv(row, "still_open", row["end_year"], "open?", jc, "kept_regex"))
        actions.append("still_open_flag")
    # The reverse: an OurAirports "open" match on a field whose narrative states
    # a closure and reads as gone (Thompson Field VA: OA matched the planned
    # strip next door). Never auto-changed — the OA join is the stronger source
    # for "open today" — but it is exactly what the review queue is for.
    if row["status"] == "open" and jc.get("jev_closed") and jc["jev_closed_conf"] >= ADOPT_OVER_WEAK \
            and jc["jev_still_open"] <= 1 - STILL_OPEN_MIN:
        review.append(_rv(row, "still_open", "open", jc["jev_closed"], jc, "open_but_narrative_closed"))
        actions.append("open_conflict_flag")
    return actions


def _rv(row, field, regex_val, jev_val, jc, disposition):
    return {"name": row["name"], "state": row["state"], "url": row["url"], "field": field,
            "regex_value": regex_val, "regex_basis": row.get(f"{field}_basis", row.get("end_basis")),
            "jev_value": jev_val,
            "jev_basis": jc.get(f"jev_{field}_basis", ""),
            "jev_conf": jc.get(f"jev_{field}_conf", jc.get("jev_still_open")),
            "disposition": disposition}


GEO_PROPS = ("name", "state", "url", "page", "anchor", "rel_location", "coord_source",
             "start_year", "end_year", "end_basis", "last_known_year", "status", "oa")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--dated", required=True, help="regex-dated CSV (airfields_freeman_dates.py --out-csv)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-geojson", required=True)
    ap.add_argument("--only", choices=("blank", "all"), default="all",
                    help="which rows to ask about (default all: agreement on the regex-confident "
                         "rows is the free regression test)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="build states, print sizes, ask nothing")
    ap.add_argument("--merge-only", action="store_true", help="re-merge from the answers file without asking")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--labels", help="hand-labeled CSV (url, built, earliest, closed, gone_by, last_seen, "
                                     "still_open; blank = not stated) -> per-question accuracy by confidence band")
    ap.add_argument("--sample", type=int, help="write N stratified blank/conflict rows with narratives "
                                               "to worklists/data/jev/airfields_dates_sample.txt for labeling, then exit")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.dated, encoding="utf-8")))
    segments = load_segments(args.tree)
    if args.sample:
        write_sample(rows, segments, args.sample)
        return
    todo = [r for r in rows if wants(r, args.only)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} rows; {len(todo)} selected ({args.only})", file=sys.stderr)

    answers = {}
    if ANSWERS_PATH.exists():
        for line in open(ANSWERS_PATH, encoding="utf-8"):
            rec = json.loads(line)
            answers[rec["id"]] = rec
    if args.merge_only:
        print(f"merge-only: {len(answers)} stored answers", file=sys.stderr)
    else:
        specs = []
        tok_est = 0
        for r in todo:
            seg = segments.get((r["page"], r["name"]))
            if not seg:
                continue
            seg, years = prep(r, seg)
            if not years:
                continue
            state = {"airfield_name": r["name"], "narrative": seg}
            specs.append((r, state, questions_for(years)))
            tok_est += len(seg) // 4 + 700
        print(f"{len(specs)} entries with a narrative and ≥1 candidate year; "
              f"≈{tok_est:,} input tokens ≈ ${tok_est / 1e6 * 0.042:.2f}", file=sys.stderr)
        if args.dry_run:
            for r, state, qs in specs[:3]:
                print(json.dumps({"airfield_name": state["airfield_name"],
                                  "narrative_chars": len(state["narrative"]),
                                  "candidates": list(qs["closed_year"]["criteria"])}, ensure_ascii=False))
            return
        jev = JevClient(task="airfields_dates", workers=args.workers)
        with open(ANSWERS_PATH, "a", encoding="utf-8") as out:
            for (r, state, qs), ans in jev.ask_many(
                    specs, lambda sp: (sp[0]["url"], sp[1], sp[2]), label="airfields "):
                if isinstance(ans, JevError):
                    print(f"  ERROR {r['name']}: {ans}", file=sys.stderr)
                    continue
                rec = {"id": r["url"], "name": r["name"], "model": jev.model, "answers": ans,
                       "candidates": list(qs["closed_year"]["criteria"])}
                answers[r["url"]] = rec
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(jev.summary(), file=sys.stderr)

    # -- merge (keep the regex values for the report before anything is adopted)
    review = []
    tally = Counter()
    jev_fields = None
    for r in rows:
        for k in ("start_year", "start_basis", "end_year", "end_basis", "status"):
            r[f"_orig_{k}"] = r[k]
        rec = answers.get(r["url"])
        if rec is None:
            jc = {}
            tally["no_answer"] += 1
        else:
            jc = jev_columns(rec["answers"])
            jc["jev_model"] = rec["model"]
            for a in merge(r, jc, review):
                tally[a] += 1
        if jev_fields is None and jc:
            jev_fields = list(jc)
        r.update(jc)
    jev_fields = jev_fields or []
    fields = [k for k in rows[0].keys() if not k.startswith("jev_")] + jev_fields
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    def prop(r, k):  # years are ints in the regex pass's GeoJSON; keep the artifact identical in shape
        v = r.get(k, "")
        return int(v) if k.endswith("_year") and v not in ("", None) else v
    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
        "properties": {k: prop(r, k) for k in GEO_PROPS},
    } for r in rows]
    with open(args.out_geojson, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)
    review_path = DATA_DIR / "airfields_dates_review.csv"
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "state", "url", "field", "regex_value", "regex_basis",
                                          "jev_value", "jev_basis", "jev_conf", "disposition"])
        w.writeheader()
        w.writerows(review)

    n = len(rows)
    st = sum(1 for r in rows if r["start_year"])
    en = sum(1 for r in rows if r["end_year"])
    print(f"merged: start {st}/{n}, end {en}/{n}, status "
          f"{dict(Counter(r['status'] for r in rows))}; review rows {len(review)} -> {review_path}")
    print("actions:", dict(sorted(tally.items())))
    print(f"csv: {args.out_csv}\ngeojson: {args.out_geojson}")

    if args.report:
        report(rows, answers)
    if args.labels:
        evaluate(answers, args.labels)


def write_sample(rows, segments, n):
    """Stratified sample of the rows Jev's answers would actually change."""
    import random
    random.seed(20260918)
    strata = {
        "end_blank": [r for r in rows if r["start_year"] and not r["end_year"] and r["status"] != "open"
                      and r["end_basis"] != "conflict"],
        "start_blank": [r for r in rows if not r["start_year"] and r["end_year"]],
        "both_blank": [r for r in rows if not r["start_year"] and not r["end_year"]],
        "conflict": [r for r in rows if r["end_basis"] == "conflict"],
        "weak_end": [r for r in rows if r["end_basis"] in WEAK_END],
    }
    quota = {"end_blank": n // 2, "start_blank": n // 8, "both_blank": n // 8, "conflict": n // 8,
             "weak_end": n - n // 2 - 3 * (n // 8)}
    out = DATA_DIR / "airfields_dates_sample.txt"
    lab = DATA_DIR / "airfields_dates_labels_template.csv"
    with open(out, "w", encoding="utf-8") as f, open(lab, "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["url", "stratum", "name", "built", "earliest", "closed", "gone_by", "last_seen", "still_open", "note"])
        for k, pool in strata.items():
            for r in random.sample(pool, min(quota[k], len(pool))):
                seg = segments.get((r["page"], r["name"]), "")
                seg, years = prep(r, seg)
                f.write(f"##### [{k}] {r['name']}\n{r['url']}\nregex: start={r['start_year']} ({r['start_basis']}) "
                        f"end={r['end_year']} ({r['end_basis']}) status={r['status']}  candidates={years}\n{seg}\n\n")
                w.writerow([r["url"], k, r["name"], "", "", "", "", "", "", ""])
    print(f"sample -> {out}\nlabel template -> {lab}")


def evaluate(answers, labels_path):
    """Accuracy of each Choice question against hand labels, split by confidence band."""
    labels = list(csv.DictReader(open(labels_path, encoding="utf-8")))
    qmap = {"built": "built_year", "earliest": "earliest_evidence_year", "closed": "closed_year",
            "gone_by": "gone_by_year", "last_seen": "last_evidence_year"}
    bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    print(f"\n== accuracy vs {len(labels)} hand labels ==")
    for col, qid in qmap.items():
        tally = {b: [0, 0] for b in bands}
        for lb in labels:
            rec = answers.get(lb["url"])
            if not rec or qid not in rec["answers"]:
                continue
            a = rec["answers"][qid]
            truth = lb[col].strip() or NOT_STATED
            got = a["choice"]
            for b in bands:
                if b[0] <= a["confidence"] < b[1]:
                    tally[b][1] += 1
                    tally[b][0] += (got == truth)
        line = "  ".join(f"[{lo:.1f}-{min(hi, 1):.1f}) {c}/{t}" for (lo, hi), (c, t) in tally.items())
        tot_c = sum(c for c, _ in tally.values()); tot_t = sum(t for _, t in tally.values())
        print(f"{col:10s} {tot_c}/{tot_t} overall   {line}")
    so = [(float(answers[lb["url"]]["answers"]["still_open"]["noul"]), lb["still_open"].strip().lower() in ("1", "y", "yes", "true"))
          for lb in labels if lb["url"] in answers]
    if so:
        tp = sum(1 for p, t in so if p >= STILL_OPEN_MIN and t); fp = sum(1 for p, t in so if p >= STILL_OPEN_MIN and not t)
        fn = sum(1 for p, t in so if p < STILL_OPEN_MIN and t)
        print(f"still_open @{STILL_OPEN_MIN}: tp={tp} fp={fp} fn={fn} of {len(so)}")


def report(rows, answers):
    """Agreement of Jev with the regex pass, by regex basis — the numbers that
    set ADOPT_MIN / ADOPT_OVER_WEAK. Disagreement rate on `closed_stated` is the
    regression test; the rest says where Jev adds evidence."""
    print("\n== agreement by regex basis (rows with both values) ==")
    for field, basis_key in (("start", "start_basis"), ("end", "end_basis")):
        by = {}
        for r in rows:
            rec = answers.get(r["url"])
            if not rec:
                continue
            jc = jev_columns(rec["answers"])
            b = r[f"_orig_{basis_key}"] or "(blank)"
            jv = jc[f"jev_{field}_year"]
            regex_v = r[f"_orig_{field}_year"]
            d = by.setdefault(b, Counter())
            if not regex_v and not jv:
                d["both_blank"] += 1
            elif not regex_v:
                d["jev_only"] += 1
            elif not jv:
                d["regex_only"] += 1
            elif int(regex_v) == int(jv):
                d["agree"] += 1
            elif abs(int(regex_v) - int(jv)) <= 2:
                d["within_2y"] += 1
            else:
                d["differ"] += 1
        print(f"-- {field}")
        for b, d in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
            print(f"   {b:18s} {dict(d)}")
    confs = [jev_columns(a["answers"])["jev_end_conf"] for a in answers.values()]
    if confs:
        confs.sort()
        q = lambda p: confs[int(p * (len(confs) - 1))]
        print(f"jev_end_conf quartiles: {q(.25):.2f} / {q(.5):.2f} / {q(.75):.2f}")


if __name__ == "__main__":
    main()
