#!/usr/bin/env python3
"""Mine operational start/end years for each airfields-freeman entry so map
pins can appear/disappear with the timeline slider.

Evidence, in priority order (all from the entry's own narrative segment,
whose phrasing is highly formulaic):

  start: "built/established/opened in <year>"; "The earliest depiction/photo
         which has been located ... <year>"; fallback: the year after the
         latest "was not yet depicted on the <year> ..." absence
  end:   "closed/abandoned in <year>" (with "between X & Y" / "(or maybe Y)"
         handled); else gone-by "was no longer depicted on the <year> ...";
         else the last dated sighting ("The last photo ...", "still depicted
         on the <year>"); else listed-in-1988 (the FAA snapshot match from
         the enrich pass) as a floor

Years parse as full years (1900-2029) or M/D/YY dates; single-slash M/YY is
deliberately NOT parsed (runway designators like "Runway 9/27" look the
same). An entry whose enrich match is a non-closed OurAirports record gets
status=open and no end year (the field still operates).

Usage:
  airfields_freeman_dates.py --tree .../www.airfields-freeman.com \
      --enriched airfields_2026-08-21_enriched.csv \
      --ourairports historical-data/ourairports-airports-2026-08-22.csv \
      [--lifespan historical-data/airports_start_stop_latlon.csv] \
      --out-csv ..._dated.csv --out-geojson ..._dated.geojson
"""
import argparse
import csv
import json
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import airfields_freeman_extract as X
import airfields_freeman_enrich as E

FULLYEAR = re.compile(r"\b(19\d\d|20[0-2]\d)\b")
SLASHDATE = re.compile(r"\b\d{1,2}/\d{1,2}/(\d\d)\b")
MYY = re.compile(r"(?<![\d/])\b(\d{1,2})/(\d\d)\b(?!\s*/|\d)")
CHARTWORD = re.compile(
    r"chart|sectional|directory|topo|usgs|aerial|photo|view|map|diagram|advertis", re.I
)
RUNWAYISH = re.compile(r"runway|rwy|r/w", re.I)

FILLER = r"(?:\w+\s+){0,3}?"
EARLIEST_RE = re.compile(
    rf"earliest\s+{FILLER}(?:depiction|photo(?:graph)?|aerial|airphoto|listing|reference|chart|view|image)",
    re.I,
)
BUILT_RE = re.compile(
    r"\b(?:built|constructed|established|opened|founded|dedicated|completed)\s+"
    r"(?:in|circa|around|about|during)\s+", re.I
)
NOTYET_RE = re.compile(r"not\s+yet\s+(?:depicted|shown|listed|built|constructed|present)", re.I)
NOLONGER_RE = re.compile(r"no\s+longer\s+(?:depicted|listed|shown|charted|present)", re.I)
CLOSED_RE = re.compile(
    r"\b(?:closed|abandoned|deactivated|decommissioned|shut\s+down|ceased\s+operations?|bulldozed|razed)\b",
    re.I,
)
LAST_RE = re.compile(
    rf"last\s+{FILLER}(?:depiction|photo(?:graph)?|aerial|view|image|listing)", re.I
)
STILL_RE = re.compile(r"still\s+(?:depicted|listed|open|active|in\s+operation|operating)", re.I)
ALSO_YEAR_RE = re.compile(r"^\s*(?:\(?\s*(?:&|and|or|-|–|to)\s*(?:maybe\s+)?)(19\d\d|20[0-2]\d)")
AIRFIELDY = re.compile(
    r"airport|airfield|air\s?park|\bfield\b|\bstrip\b|\bbase\b|station|\bnas\b|naas|nolf|"
    r"\bolf\b|facility|aerodrome|airbase|heliport|flightpark|gliderport", re.I
)
NOT_THE_FIELD = re.compile(r"hangar|terminal\b|tower|depot|barracks|beacon|factory|plant\b", re.I)
PROPOSED_RE = re.compile(r"proposed|would\s+eventually\s+become|never\s+built", re.I)
ABANDONED_IMG_RE = re.compile(
    r"abandoned|closed|remains?\b|remnant|overgrown|deteriorat|traces|razed|demolished|former|no\s+sign",
    re.I,
)


def segment_years(seg):
    ys = []
    for m in FULLYEAR.finditer(seg):
        ys.append((m.start(), int(m.group(1))))
    for m in SLASHDATE.finditer(seg):
        yy = int(m.group(1))
        ys.append((m.start(), 2000 + yy if yy <= 26 else 1900 + yy))
    for m in MYY.finditer(seg):
        mo, yy = int(m.group(1)), int(m.group(2))
        if not 1 <= mo <= 12:
            continue
        if RUNWAYISH.search(seg[max(0, m.start() - 16) : m.start()]):
            continue
        if not CHARTWORD.search(seg[m.end() : m.end() + 45]):
            continue
        ys.append((m.start(), 2000 + yy if yy <= 26 else 1900 + yy))
    ys.sort()
    return ys


def sentence_years(seg, ys, start, maxlen):
    """Years between start and the end of that sentence (capped at maxlen)."""
    stop = seg.find(". ", start, start + maxlen)
    hi = stop if stop != -1 else start + maxlen
    return [y for p, y in ys if start <= p < hi]


def nearest_year(ys, pos, back, fwd):
    cands = [(abs(p - pos), y) for p, y in ys if pos - back <= p < pos + fwd]
    return min(cands)[1] if cands else None


def subject_ok(seg, vstart, vend, title_toks):
    """Does this closed/built/etc. verb plausibly refer to the airfield itself
    (not the fair, a hangar, or Runway 15/33)?"""
    before = seg[max(0, vstart - 45) : vstart]
    after = seg[vend : vend + 28]
    if RUNWAYISH.search(before[-20:]):
        return False
    if NOT_THE_FIELD.search(before):
        return False
    if AIRFIELDY.search(before) or AIRFIELDY.search(after):
        return True
    words = re.findall(r"[a-z0-9']+", before.lower())
    if any(t in words for t in title_toks):
        return True
    if re.search(r"\bit\s+(?:was\s+)?$", before.lower()):
        return True
    return False


def mine_dates(seg, title_toks):
    seg = re.sub(r"\s+", " ", seg)
    ys = segment_years(seg)
    ev = {}

    earliest = []
    for m in EARLIEST_RE.finditer(seg):
        if PROPOSED_RE.search(seg[m.start() : m.start() + 120]):
            continue
        w = sentence_years(seg, ys, m.end(), 170)
        if w:
            earliest.append(w[0])
    if earliest:
        ev["earliest"] = min(earliest)
    built = []
    for m in BUILT_RE.finditer(seg):
        if not subject_ok(seg, m.start(), m.end(), title_toks):
            continue
        w = sentence_years(seg, ys, m.end(), 25)
        if w:
            built.append(w[0])
    if built:
        ev["built"] = min(built)
    notyet = [y for m in NOTYET_RE.finditer(seg) if (y := nearest_year(ys, m.end(), 40, 120))]
    if notyet:
        ev["notyet"] = max(notyet)
    nolonger = [y for m in NOLONGER_RE.finditer(seg) if (y := nearest_year(ys, m.end(), 50, 120))]
    if nolonger:
        ev["nolonger"] = min(nolonger)
    closed = []
    for m in CLOSED_RE.finditer(seg):
        if not subject_ok(seg, m.start(), m.end(), title_toks):
            continue
        w = sentence_years(seg, ys, m.end(), 50)
        if w:
            y = w[0]
            after = seg[seg.find(str(y), m.end()) + 4 :]
            m2 = ALSO_YEAR_RE.match(after)
            if m2:
                y = max(y, int(m2.group(1)))
            closed.append(y)
    last = []
    for m in LAST_RE.finditer(seg):
        stop = seg.find(". ", m.end(), m.end() + 200)
        sent = seg[m.end() : stop if stop != -1 else m.end() + 200]
        # "the last photo ... was a 2016 aerial view of the abandoned site"
        # is recent imagery, not last operation
        if ABANDONED_IMG_RE.search(sent):
            continue
        w = sentence_years(seg, ys, m.end(), 170)
        if w:
            last.append(w[0])
    still = [y for m in STILL_RE.finditer(seg) if (y := nearest_year(ys, m.end(), 60, 80))]
    if last or still:
        ev["lastseen"] = max(last + still)
    if closed:
        ref = ev.get("nolonger") or ev.get("lastseen")
        ev["closed"] = min(closed, key=lambda y: abs(y - ref)) if ref else max(closed)
    return ev


def derive(ev, in_1988, oa_open):
    start = basis_s = None
    cands = {k: ev[k] for k in ("built", "earliest") if k in ev}
    if cands:
        basis_s, start = min(cands.items(), key=lambda kv: kv[1])
        if basis_s == "earliest" and start > 2015:
            start = basis_s = None  # window leakage, not a real first depiction
    if start is None and "notyet" in ev:
        start, basis_s = ev["notyet"] + 1, "after_absence"
    if start and in_1988 and start > 1988:
        start, basis_s = 1988, "faa1988_listed"

    end = basis_e = None
    if "closed" in ev:
        end, basis_e = ev["closed"], "closed_stated"
    elif "nolonger" in ev:
        end, basis_e = ev["nolonger"], "no_longer_by"
    elif "lastseen" in ev:
        end, basis_e = ev["lastseen"], "last_seen"
    if end is None and in_1988:
        end, basis_e = 1988, "faa1988_listed"
    status = "gone" if end else "unknown"
    if oa_open:
        status, end, basis_e = "open", None, "oa_open"
    if start and end and end < start:
        end, basis_e = None, "conflict"
        status = "unknown"
    return start, basis_s, end, basis_e, status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--ourairports", required=True)
    ap.add_argument("--lifespan")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-geojson", required=True)
    args = ap.parse_args()

    import os
    import posixpath

    segments = {}
    for path in X.iter_pages(os.path.abspath(args.tree)):
        rel = os.path.relpath(path, os.path.abspath(args.tree))
        rel_url = posixpath.join(*rel.split(os.sep))
        for e in X.extract_page(path, rel_url, keep_segment=True)[0]:
            segments[(e["page"], e["name"])] = e["_segment"]

    oa_type = {}
    for r in csv.DictReader(open(args.ourairports, encoding="utf-8")):
        oa_type[r["ident"]] = r["type"]

    lifespan = []
    if args.lifespan:
        for r in csv.DictReader(open(args.lifespan)):
            if r["lat"] and r["lon"]:
                lifespan.append(
                    (
                        set(E.norm_name(r["airport"])),
                        float(r["lat"]),
                        float(r["lon"]),
                        r["start_year"],
                        r["stop_year"],
                    )
                )

    rows = list(csv.DictReader(open(args.enriched)))
    stats = {"start": 0, "end": 0, "both": 0, "open": 0, "conflict": 0, "lifespan_fill": 0, "no_segment": 0}
    for row in rows:
        seg = segments.get((row["page"], row["name"]))
        title_toks = set(E.norm_name(E.split_title(row["name"], row["state"])[0]))
        ev = mine_dates(seg, title_toks) if seg else {}
        if not seg:
            stats["no_segment"] += 1
        in_1988 = row["match_donor"] == "faa1988" and row["match_via"] not in ("", "collision-reverted")
        oa_open = (
            row["match_donor"] == "ourairports"
            and row["match_via"] not in ("", "collision-reverted")
            and oa_type.get(row["match_id"], "closed") != "closed"
        )
        start, bs, end, be, status = derive(ev, in_1988, oa_open)

        if lifespan and (start is None or (end is None and status != "open")):
            flat, flon = float(row["lat"]), float(row["lon"])
            head, _ = E.split_title(row["name"], row["state"])
            aliases = [set(E.norm_name(a)) for a in head.split("/") if E.norm_name(a)]
            for toks, llat, llon, ls, le in lifespan:
                if E.haversine_km(flat, flon, llat, llon) > 3.0:
                    continue
                if any(toks and (toks <= al or al <= toks) for al in aliases):
                    if start is None and ls:
                        start, bs = int(ls), "lifespan_csv"
                    if end is None and status != "open" and le:
                        end, be, status = int(le), "lifespan_csv", "gone"
                    stats["lifespan_fill"] += 1
                    break

        row["start_year"] = start or ""
        row["start_basis"] = bs or ""
        row["end_year"] = end or ""
        row["end_basis"] = be or ""
        row["status"] = status
        # OurAirports ident for the viewer's source link (only when the
        # enrich pass confidently matched an OA record)
        row["oa"] = (
            row["match_id"]
            if row["match_donor"] == "ourairports"
            and row["match_via"] not in ("", "collision-reverted")
            else ""
        )
        if start:
            stats["start"] += 1
        if end:
            stats["end"] += 1
        if start and end:
            stats["both"] += 1
        if status == "open":
            stats["open"] += 1
        if be == "conflict":
            stats["conflict"] += 1

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
            "properties": {
                k: r[k]
                for k in (
                    "name", "state", "url", "page", "anchor", "rel_location",
                    "coord_source", "start_year", "end_year", "status", "oa",
                )
            },
        }
        for r in rows
    ]
    with open(args.out_geojson, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)

    n = len(rows)
    print(
        f"{n} entries: start {stats['start']} ({100*stats['start']//n}%), "
        f"end {stats['end']} ({100*stats['end']//n}%), both {stats['both']} "
        f"({100*stats['both']//n}%), open {stats['open']}, "
        f"conflicts {stats['conflict']}, lifespan-fills {stats['lifespan_fill']}, "
        f"no-segment {stats['no_segment']}"
    )
    print(f"csv: {args.out_csv}\ngeojson: {args.out_geojson}")


if __name__ == "__main__":
    main()
