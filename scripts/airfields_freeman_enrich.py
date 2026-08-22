#!/usr/bin/env python3
"""Refine airfields-freeman coordinate precision against reference airport
databases: the FAA snapshot of 1988-11-29 (historical-data/
noaa-1988-data-sampler) and optionally OurAirports (public-domain, includes
~7.5k closed US airfields).

Paul Freeman's coordinates vary from 3-decimal (~100 m) down to integer
degrees. Where an entry can be matched with confidence, the reference
position is the better pin.

Matching, per entry (donor order: faa1988 ident, faa1988 name, ourairports
ident, ourairports name — official era-appropriate data wins ties):
  1. skipped entirely for "(original)" / "(1st location)"-style titles —
     those deliberately point somewhere OLDER than the surviving airport
  2. FAA ident(s) parsed from the title parentheticals -> donor identifiers
     (for OurAirports: ident/icao/iata/gps/local codes plus ident-looking
     keywords tokens), verified by same state + distance gate scaled to
     Freeman's precision
  3. else name-alias match among same-state donor records inside a tight
     gate (6 km for 0-1 decimal entries, 4 km otherwise), with:
     - full token containment, and 2+ shared tokens (or exact equality for
       single-token names within 4 km)
     - aux/intermediate/outlying variants never match a non-variant donor
       beyond 3 km (they are deliberately distinct from the survivor)
     - city-name-only matches to a donor with extra distinctive tokens are
       rejected (that's the town's surviving big airport, not this field)

OurAirports donors must themselves carry >= 3 decimals on both axes — its
closed-airfield entries are often sourced FROM Freeman's site, so a coarse
donor adds nothing (and is likely his own rounded number copied back).

Coordinate policy: a matched donor position replaces Freeman's only when his
precision is <= 2 decimals on either axis (quantization >= ~500 m);
3-decimal entries keep Freeman's pin (he places those from aerials, often
better than the ARP) but still record the match. Freeman's original coords
are always kept in freeman_lat/freeman_lon; coord_source says which won.

Usage:
  airfields_freeman_enrich.py --airfields airfields_2026-08-21.csv \
      --nasr1988 historical-data/noaa-1988-data-sampler/DATA/AIRPORTS.DAT \
      [--ourairports historical-data/ourairports-airports-YYYY-MM-DD.csv] \
      --out-csv ..._enriched.csv --out-geojson ..._enriched.geojson
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict

# AIRPORTS.DAT fixed-width offsets (from DIC/AIRPORTS.DIC, fields AP-01..08)
F_SITENUM = (0, 11)
F_LOCID = (11, 16)
F_NAME = (16, 81)
F_CITY = (81, 107)
F_STATE = (107, 109)
F_LAT = (115, 123)  # N/S + DDMMSS + tenths
F_LON = (123, 132)  # E/W + DDDMMSS + tenths

OA_NAME_OK = {"closed", "small_airport", "medium_airport", "large_airport", "seaplane_base"}
OA_TYPES = OA_NAME_OK | {"heliport", "balloonport"}

LOC_VARIANT_RE = re.compile(
    r"\(\s*(?:original|orignal|1\s*st|2\s*nd|3\s*rd|\d+\s*th)\s*(?:location)?\s*\)", re.I
)
# aux/intermediate/outlying entries are deliberately DISTINCT from the
# surviving same-named airport; only match a non-variant donor point-blank
VARIANTISH_RE = re.compile(
    r"aux|intermediate|outlying|satellite|nolf|naaf|\bolf\b|#\s*\d", re.I
)
PAREN_RE = re.compile(r"\(([^)]{1,40})\)")
IDENT_RE = re.compile(r"^[A-Z0-9]{2,4}$")
IDENT_STOP = {
    "NAS", "AAF", "AFB", "NAAS", "NOLF", "NALF", "NAAF", "MCAS", "MCAF",
    "CGAS", "NAF", "ANG", "AUX", "USN", "USAF", "AAAF", "AFS", "NRAB",
}
NAME_STOP = {
    "airport", "airfield", "field", "airpark", "airstrip", "strip",
    "municipal", "muni", "county", "city", "memorial", "regional",
    "international", "intl", "landing", "intermediate", "auxiliary", "aux",
    "army", "naval", "navy", "air", "force", "base", "station", "the",
    "original", "location", "no", "new", "old",
}


def parse_dms(s, lon=False):
    s = s.strip()
    if not s or s[0] not in "NSEW":
        return None
    hemi = s[0]
    digits = s[1:].strip()
    if not digits.isdigit():
        return None
    dd = 3 if lon else 2
    if len(digits) < dd + 4:
        return None
    deg = int(digits[:dd])
    mn = int(digits[dd : dd + 2])
    sec = int(digits[dd + 2 : dd + 4])
    tenth = int(digits[dd + 4 : dd + 5]) if len(digits) > dd + 4 else 0
    val = deg + mn / 60 + (sec + tenth / 10) / 3600
    return -val if hemi in "SW" else val


def decimals(s):
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0


def ident_like(t):
    return bool(
        IDENT_RE.match(t)
        and t not in IDENT_STOP
        and (any(ch.isdigit() for ch in t) or (t.isalpha() and len(t) in (3, 4)))
    )


def load_1988(path):
    """-> donor records {id, names, state, lat, lon}, keyed indexes built by caller."""
    recs = []
    data = open(path, "rb").read().decode("ascii", errors="replace")
    for line in data.split("\r\n"):
        if len(line) < 132:
            continue
        lat = parse_dms(line[F_LAT[0] : F_LAT[1]])
        lon = parse_dms(line[F_LON[0] : F_LON[1]], lon=True)
        if lat is None or lon is None:
            continue
        locid = line[F_LOCID[0] : F_LOCID[1]].strip()
        recs.append(
            {
                "id": locid,
                "idents": {locid} if locid else set(),
                "names": [line[F_NAME[0] : F_NAME[1]].strip()],
                "state": line[F_STATE[0] : F_STATE[1]].strip(),
                "lat": lat,
                "lon": lon,
                # heliports/balloonports may match by ident but never by name
                # (the Waynesboro-Hospital-heliport failure mode)
                "name_ok": line[10] not in ("B", "H"),
            }
        )
    return recs


def load_ourairports(path):
    recs = []
    skipped_coarse = 0
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["iso_country"] not in ("US", "PR") or r["type"] not in OA_TYPES:
            continue
        lat_s, lon_s = r["latitude_deg"].strip(), r["longitude_deg"].strip()
        if not lat_s or not lon_s:
            continue
        # a coarse donor adds no precision — and for closed fields is often
        # Freeman's own rounded number, contributed back to OurAirports
        if decimals(lat_s) < 3 or decimals(lon_s) < 3:
            skipped_coarse += 1
            continue
        idents = set()
        names = [r["name"]]
        for k in ("ident", "icao_code", "iata_code", "gps_code", "local_code"):
            v = r[k].strip().upper()
            if v and ident_like(v):
                idents.add(v)
        for kw in r["keywords"].split(","):
            kw = kw.strip()
            if not kw:
                continue
            if ident_like(kw.upper()):
                idents.add(kw.upper())
            else:
                names.append(kw)
        region = r["iso_region"]
        state = r["iso_country"] if r["iso_country"] == "PR" else region.split("-", 1)[-1]
        recs.append(
            {
                "id": r["ident"],
                "idents": idents,
                "names": names,
                "state": state,
                "lat": float(lat_s),
                "lon": float(lon_s),
                "name_ok": r["type"] in OA_NAME_OK,
            }
        )
    print(f"ourairports: {len(recs)} usable US/PR records ({skipped_coarse} skipped as coarse)")
    return recs


def build_indexes(recs):
    by_ident = defaultdict(list)
    by_state = defaultdict(list)
    for r in recs:
        for i in r["idents"]:
            by_ident[i].append(r)
        by_state[r["state"]].append(r)
    return by_ident, by_state


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def title_idents(name):
    out = []
    for m in PAREN_RE.finditer(name):
        for t in re.split(r"[/,]", m.group(1)):
            t = t.strip()
            if ident_like(t):
                out.append(t)
    return out


def norm_name(s):
    s = PAREN_RE.sub(" ", s).lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in NAME_STOP]


def split_title(name, state):
    """'A / B, City, ST' or 'A / B, City ST' -> (alias_head, city_str)."""
    head = name
    if state:
        m = re.search(r"[,\s]+%s\.?\s*$" % re.escape(state), head)
        if m:
            head = head[: m.start()]
    if "," in head:
        head, city = head.rsplit(",", 1)
    else:
        city = ""
    return head, city


def match_ident(row, flat, flon, gate, by_ident):
    for ident in title_idents(row["name"]):
        cands = [
            c
            for c in by_ident.get(ident, [])
            if c["state"] == row["state"]
            and haversine_km(flat, flon, c["lat"], c["lon"]) <= gate
        ]
        if cands:
            best = min(cands, key=lambda c: haversine_km(flat, flon, c["lat"], c["lon"]))
            return best, f"ident:{ident}"
    return None, ""


def match_name(row, flat, flon, prec, by_state):
    head, city = split_title(row["name"], row["state"])
    aliases = [(a.strip(), norm_name(a)) for a in head.split("/") if norm_name(a)]
    name_gate = 6.0 if prec <= 1 else 4.0
    city_toks = set(norm_name(city))
    best = None
    for c in by_state.get(row["state"], []):
        if not c["name_ok"]:
            continue
        d = haversine_km(flat, flon, c["lat"], c["lon"])
        if d > name_gate:
            continue
        for cname in c["names"]:
            ctoks = set(norm_name(cname))
            if not ctoks:
                continue
            for raw, al in aliases:
                als = set(al)
                inter = len(als & ctoks)
                contained = als <= ctoks or ctoks <= als
                if not contained or inter == 0:
                    continue
                if inter == 1 and not (als == ctoks and d <= 4.0):
                    continue
                # city-name tokens alone matching a donor that has extra
                # distinctive tokens = the surviving big airport of that
                # town, not this field
                if (als & ctoks) <= city_toks and (ctoks - als):
                    continue
                if (
                    VARIANTISH_RE.search(raw)
                    and not VARIANTISH_RE.search(cname)
                    and d > 3.0
                ):
                    continue
                score = (-inter, d)
                if best is None or score < best[0]:
                    best = (score, c)
    return (best[1], "name") if best else (None, "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--airfields", required=True)
    ap.add_argument("--nasr1988", required=True)
    ap.add_argument("--ourairports", help="optional OurAirports airports.csv")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-geojson", required=True)
    args = ap.parse_args()

    donors = []
    recs88 = load_1988(args.nasr1988)
    print(f"faa1988: {len(recs88)} usable records")
    donors.append(("faa1988",) + build_indexes(recs88))
    if args.ourairports:
        donors.append(("ourairports",) + build_indexes(load_ourairports(args.ourairports)))

    rows = list(csv.DictReader(open(args.airfields)))
    stats = defaultdict(int)
    out_rows = []
    for row in rows:
        flat, flon = float(row["lat"]), float(row["lon"])
        prec = min(decimals(row["lat"]), decimals(row["lon"]))
        gate = {0: 60.0, 1: 25.0}.get(prec, 12.0)
        skip_variant = bool(LOC_VARIANT_RE.search(row["name"]))

        match, how, donor_label = None, "", ""
        if not skip_variant:
            for label, by_ident, by_state in donors:
                match, how = match_ident(row, flat, flon, gate, by_ident)
                if match is None:
                    match, how = match_name(row, flat, flon, prec, by_state)
                if match is not None:
                    donor_label = label
                    break

        dist = haversine_km(flat, flon, match["lat"], match["lon"]) if match else ""
        use_donor = bool(match) and prec <= 2
        out = dict(row)
        out["freeman_lat"], out["freeman_lon"] = row["lat"], row["lon"]
        if use_donor:
            out["lat"] = f"{match['lat']:.5f}"
            out["lon"] = f"{match['lon']:.5f}"
            out["coord_source"] = donor_label
        else:
            out["coord_source"] = "freeman"
        out["match_donor"] = donor_label
        out["match_via"] = how
        out["match_id"] = match["id"] if match else ""
        out["match_name"] = match["names"][0] if match else ""
        out["match_dist_km"] = f"{dist:.2f}" if match else ""
        out["freeman_precision_decimals"] = str(prec)
        out_rows.append(out)

        if skip_variant:
            stats["skipped_location_variant"] += 1
        elif use_donor:
            stats[f"upgraded_{donor_label}_{how.split(':')[0]}"] += 1
        elif match:
            stats["matched_kept_freeman_3dec"] += 1
        else:
            stats["unmatched_kept_freeman"] += 1

    # donor uniqueness: two entries claiming one donor record = at most one
    # of them is that airport; the closer claim wins, the others revert
    claims = defaultdict(list)
    for i, out in enumerate(out_rows):
        if out["match_via"]:
            key = (out["match_donor"], out["match_id"] or out["match_name"], out["state"])
            claims[key].append(i)
    for key, idxs in claims.items():
        if len(idxs) < 2:
            continue
        idxs.sort(key=lambda i: float(out_rows[i]["match_dist_km"]))
        for i in idxs[1:]:
            out = out_rows[i]
            if out["coord_source"] != "freeman":
                stats[f"upgraded_{out['coord_source']}_{out['match_via'].split(':')[0]}"] -= 1
                out["lat"], out["lon"] = out["freeman_lat"], out["freeman_lon"]
                out["coord_source"] = "freeman"
            else:
                stats["matched_kept_freeman_3dec"] -= 1
            stats["reverted_donor_collision"] += 1
            out["match_via"] = "collision-reverted"

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
            "properties": {
                k: r[k]
                for k in ("name", "state", "url", "page", "anchor", "rel_location", "coord_source")
            },
        }
        for r in out_rows
    ]
    with open(args.out_geojson, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)

    total = len(out_rows)
    print(f"{total} entries:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"csv: {args.out_csv}\ngeojson: {args.out_geojson}")


if __name__ == "__main__":
    main()
