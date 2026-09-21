#!/usr/bin/env python3
"""Worklist 16 W2 — machine-readable metadata for the ATC History collection.

The 1,333 atchistory WordPress posts under /atc/ carry only category tags (and,
for the 993 facility-photo posts, a city/state from the old facility-photos
page). Nothing says which facility a page is about, what kind of facility, when
the photo or event dates from, or where it sits on a map — which is what the
09 F1–F3 fusion layers and the B1 airport pages need.

This pass:
  1. loads every post (title, categories, entry text, preview image, PDF links)
     from the preserved static site;
  2. finds candidates in code — 3/4-letter idents in parentheses, "City, State"
     spans, dates/years/decades — and asks Jev (jev-1.13.0, one request per
     post) to SELECT: page_kind, facility_type, facility_ident, place,
     depicted_year, opened_date, closed_date (+ two Nouls);
  3. joins the chosen ident / place to coordinates in code: the 1988 NASR
     airport file (which carries each airport's FSS ident, name and an "FSS on
     airport" flag — 214 Y records = an authoritative 1988 FSS location table),
     then OurAirports; city+state as the fallback;
  4. catalogs the 1,405 PDFs: series and issue date from the filename (code),
     text-layer flag, linking posts; the ~170 non-periodical PDFs with a text
     layer get the same Jev questions over their first page;
  5. writes worklists/data/atc/posts_meta.jsonl (everything, incl. raw
     answers), pdfs.jsonl, and posts_index.json (compact, public-shaped:
     href, title, kind, facility, place, year, lat/lon, image).

Nothing here touches /atc/ itself (08 §5: no navigation or URL changes before
2026-10-30); the index is consumed first from archive.aero-side surfaces.

Usage:
  ~/venv/bin/python scripts/atc_posts_jev.py [--site DIR] [--limit N] [--dry-run]
      [--merge-only] [--no-pdfs] [--report] [--workers 8]
"""
import argparse
import csv
import html
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jev_client import DATA_DIR, JevClient, JevError, choice, noul  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITE_DEFAULT = Path("/Volumes/projects/atchistory_build/site")
OUT_DIR = REPO / "worklists" / "data" / "atc"
NASR1988 = REPO / "historical-data/noaa-1988-data-sampler/DATA/AIRPORTS.DAT"
OURAIRPORTS = REPO / "historical-data/ourairports-airports-2026-08-22.csv"
FACILITIES = REPO / "worker-atc/src/facilities.json"
ROUTES = REPO / "worker-atc/src/route_map.json"

TEXT_CAP = 8000       # chars of entry text sent (≈2k tokens)
PDF_TEXT_CAP = 6000
MAX_DATE_CANDS = 24
IDENT_MIN_CONF = 0.50  # below this the chosen ident is not used for the join
PLACE_MIN_CONF = 0.50

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico", "VI": "Virgin Islands", "GU": "Guam", "AS": "American Samoa",
}
NAME_TO_ABBR = {v.lower(): k for k, v in STATES.items()}
STATE_ALT = "|".join(sorted([re.escape(v) for v in STATES.values()] + list(STATES), key=len, reverse=True))
PLACE_RE = re.compile(
    r"\b([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,4})"
    r"(?:\s*\([A-Z0-9]{2,4}\))?\s*,\s*(" + STATE_ALT + r")\b(?![A-Za-z])"
)
CITY_STOP = {"fss", "afss", "ifss", "artcc", "atct", "radio", "station", "tower", "center",
             "centre", "airport", "field", "flight", "service", "airway", "airways",
             "communication", "communications", "range", "office", "regional", "region",
             "division", "personnel", "employees", "staff", "the", "at", "in", "of", "and",
             "history", "facility", "control", "air", "traffic", "caa", "faa", "insacs",
             "beacon", "beacons", "site", "light", "lights", "arrow", "arrows", "line", "lines",
             "class", "photo", "photos", "building", "employees", "managers", "list"}
IDENT_PAREN_RE = re.compile(r"\(([A-Z][A-Z0-9]{2,3})\)")
IDENT_NEAR_RE = re.compile(r"\b([A-Z]{3})\s+(?:FSS|AFSS|IFSS|Radio|ARTCC|ATCT|Tower|Center)\b")
IDENT_STOP = {"FAA", "CAA", "FSS", "ATC", "USA", "PDF", "OCR", "AND", "THE", "NAS", "AFB",
              "ARMY", "NAVY", "UHF", "VHF", "ILS", "VOR", "NDB", "DME", "AAF", "USAF", "RCO",
              "DOD", "GS", "EFAS", "AWOS", "ASOS", "WWII", "AAF"}
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|"
          "December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
FULLDATE_RE = re.compile(rf"\b(?:{MONTHS})\.?\s+\d{{1,2}},?\s+(?:18|19|20)\d\d\b")
MONTHYEAR_RE = re.compile(rf"\b(?:{MONTHS})\.?,?\s+(?:18|19|20)\d\d\b")
NUMDATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})\b")
YEAR_RE = re.compile(r"\b(?:18[5-9]\d|19\d\d|20[0-2]\d)\b")
DECADE_RE = re.compile(r"\b((?:18|19|20)\d0)'?s\b")
NONE = "none"
NOT_STATED = "not_stated"

KINDS = {
    "facility_photo": "A photograph of an FAA facility building, its equipment, its vehicles "
                      "or its staff, presented with a caption",
    "facility_history": "A written history, log, timeline, chronology or personal account "
                        "of one named facility",
    "roster_or_list": "A list or table of people, managers, employees, classes, stations or "
                      "facilities (a roster, directory, or managers list)",
    "class_photo": "A training class, academy course or graduating-class group photograph",
    "airway_infrastructure": "Airway beacons, light lines, concrete arrows, lighthouses, radio "
                             "ranges, or mail-plane-era equipment and sites",
    "publication_or_document": "A scanned publication, newsletter, manual, memo, directory, "
                               "brochure or article presented as a document",
    "map": "A historical airway, route, region or facility map",
    "personal_story": "A memoir, biography, obituary or tribute about a person",
    "other": "None of the above (site notices, requests for help, miscellany)",
}
FTYPES = {
    "fss": "Flight Service Station, AFSS, IFSS, Flight Watch, or an airway radio / "
           "communication station (INSACS, CAA radio)",
    "tower": "Airport traffic control tower (ATCT) or approach control",
    "center": "Air Route Traffic Control Center (ARTCC)",
    "office_or_academy": "A regional or division office, the FAA Academy, or a training center",
    "other_or_none": "Some other facility, or no single facility",
}


# ---------------------------------------------------------------- posts
def strip_tags(s):
    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    return html.unescape(re.sub(r"\s+", " ", s)).strip()


def load_posts(site):
    routes = json.loads(ROUTES.read_text())["alias"]
    posts = []
    for d in sorted(os.listdir(site)):
        p = site / d / "index.html"
        if not p.is_file():
            continue
        s = p.read_text(encoding="utf-8", errors="replace")
        t = re.search(r'<h1 class="entry-title"[^>]*>(.*?)</h1>', s, re.S)
        if not t:
            continue
        cats = [html.unescape(c) for c in re.findall(r'rel="category tag">([^<]+)<', s)]
        if not cats:
            continue
        m = re.search(r'<div class="entry-content"[^>]*>(.*?)</div><!-- .entry-content -->', s, re.S)
        body = m[1] if m else ""
        img = re.search(r'<img\b[^>]*\bsrc="([^"]+)"', body)
        pdfs = [html.unescape(u) for u in re.findall(r'href="([^"]+\.pdf)"', body, re.I)]
        posts.append({
            "slug": d,
            "href": routes.get(f"/{d}/", f"/atc/{d}"),
            "title": strip_tags(t[1]),
            "categories": cats,
            "text": strip_tags(body),
            "image": html.unescape(img[1]) if img else "",
            "pdf_links": pdfs,
        })
    return posts


def load_facilities():
    """href -> (state, city) from the restored facility-location index."""
    out = {}
    if FACILITIES.exists():
        for state, cities in json.loads(FACILITIES.read_text()).items():
            for city, entries in cities.items():
                for e in entries:
                    out[e["href"]] = (state, city)
    return out


def clean_city(c):
    toks = c.split()
    while toks and toks[-1].lower().strip(".,") in CITY_STOP:
        toks.pop()
    # leading junk from a sentence start ("The Anniston")
    while toks and toks[0].lower() in {"the", "at", "in", "of", "and", "a"}:
        toks.pop(0)
    return " ".join(toks)


def state_abbr(s):
    s = s.strip()
    return s if s.upper() in STATES else NAME_TO_ABBR.get(s.lower(), "")


def candidates(post, fac):
    title, text = post["title"], post["text"][:TEXT_CAP]
    both = title + " . " + text
    idents = []
    for m in IDENT_PAREN_RE.finditer(both):
        v = m.group(1)
        if v not in IDENT_STOP and v not in idents and not v.isdigit():
            idents.append(v)
    for m in IDENT_NEAR_RE.finditer(both):
        v = m.group(1)
        if v not in IDENT_STOP and v not in idents:
            idents.append(v)
    places = []
    if fac:
        st, city = fac
        places.append(f"{city}, {st}")
    for m in PLACE_RE.finditer(both):
        city = clean_city(m.group(1))
        ab = state_abbr(m.group(2))
        if not city or not ab or len(city) > 40:
            continue
        cand = f"{city}, {STATES[ab]}"
        if cand not in places:
            places.append(cand)
    places = places[:8]
    dates = []
    for rx in (FULLDATE_RE, MONTHYEAR_RE, NUMDATE_RE):
        for m in rx.finditer(both):
            if m.group(0) not in dates:
                dates.append(m.group(0))
    years = []
    for m in YEAR_RE.finditer(both):
        if m.group(0) not in years:
            years.append(m.group(0))
    decades = []
    for m in DECADE_RE.finditer(both):
        v = m.group(1) + "s"
        if v not in decades:
            decades.append(v)
    date_cands = (dates + years + decades)[:MAX_DATE_CANDS]
    year_cands = (years + decades)[:MAX_DATE_CANDS]
    return idents[:8], places, date_cands, year_cands


def questions_for(idents, places, date_cands, year_cands):
    id_opts = {i: None for i in idents}
    id_opts[NONE] = "No candidate is the identifier of the facility this page is about"
    pl_opts = {p: None for p in places}
    pl_opts[NONE] = "No candidate is the place this page is about"
    dt_opts = {d: None for d in date_cands}
    dt_opts[NOT_STATED] = "The page does not state this"
    yr_opts = {y: None for y in year_cands}
    yr_opts[NOT_STATED] = "The page does not state this"
    qs = {
        "page_kind": choice("What kind of page is this, judging by `title`, `categories` and `text`?", KINDS),
        "facility_type": choice("What type of FAA facility is the subject of `title` and `text`?", FTYPES),
        "about_one_facility": noul(
            "Is this page (`title` and `text`) about one specific named FAA facility — an FSS, "
            "tower, center, radio station or office — rather than about a person, a class, a "
            "publication or a general topic?"),
        "about_a_person": noul(
            "Is this page (`title` and `text`) mainly about one named person — a memoir, "
            "biography, obituary, tribute or career story?"),
        "depicted_year": choice(
            "Which candidate is the year or decade that the photograph, document, event or "
            "list described by `title` and `text` dates from — the historical date of the "
            "subject itself, not the date the web page was posted and not a person's birth "
            "year? Pick exactly one candidate, or not_stated.", yr_opts),
    }
    if idents:
        qs["facility_ident"] = choice(
            "Which candidate is the FAA location identifier (the 3- or 4-letter code such as "
            "ANB or ZAN) of the facility that `title` and `text` are about? Pick it exactly as "
            "listed, or none. Course numbers, class numbers, years and abbreviations such as FSS "
            "are not identifiers.", id_opts)
    if places:
        qs["place"] = choice(
            "Which candidate is the city and state where the facility, event or subject of "
            "`title` and `text` is located? Pick exactly one candidate, or none if no candidate "
            "is that place (a person's later home or a visiting official's base does not count).",
            pl_opts)
    if date_cands:
        qs["opened_date"] = choice(
            "Which candidate is the date the facility that `title` and `text` are about was "
            "commissioned, opened or dedicated? Pick exactly the candidate string, or not_stated "
            "if the page does not state when the facility opened.", dt_opts)
        qs["closed_date"] = choice(
            "Which candidate is the date the facility that `title` and `text` are about closed, "
            "was decommissioned or was consolidated away? Pick exactly the candidate string, or "
            "not_stated if the page does not state when the facility closed.", dt_opts)
    return qs


def build_spec(post, fac):
    idents, places, date_cands, year_cands = candidates(post, fac)
    state = {"title": post["title"], "categories": post["categories"],
             "text": post["text"][:TEXT_CAP] or "(no text)"}
    return state, questions_for(idents, places, date_cands, year_cands)


# ---------------------------------------------------------------- join tables
# ARTCCs are not airports: ident -> the city the center sits in (the 1988
# "ARTCCFAC.DAT" sampler file is route segments, not facilities), resolved
# through the same city join as everything else.
ARTCC = {
    "ZAB": ("Albuquerque", "NM"), "ZAN": ("Anchorage", "AK"), "ZAU": ("Aurora", "IL"),
    "ZBW": ("Nashua", "NH"), "ZDC": ("Leesburg", "VA"), "ZDV": ("Longmont", "CO"),
    "ZFW": ("Fort Worth", "TX"), "ZHU": ("Houston", "TX"), "ZID": ("Indianapolis", "IN"),
    "ZJX": ("Hilliard", "FL"), "ZKC": ("Olathe", "KS"), "ZLA": ("Palmdale", "CA"),
    "ZLC": ("Salt Lake City", "UT"), "ZMA": ("Miami", "FL"), "ZME": ("Memphis", "TN"),
    "ZMP": ("Farmington", "MN"), "ZNY": ("Ronkonkoma", "NY"), "ZOA": ("Fremont", "CA"),
    "ZOB": ("Oberlin", "OH"), "ZSE": ("Auburn", "WA"), "ZTL": ("Hampton", "GA"),
    "ZHN": ("Honolulu", "HI"), "ZSU": ("San Juan", "PR"), "ZUA": ("Barrigada", "GU"),
}


def parse_dms(s, lon=False):
    s = s.strip()
    if not s or s[0] not in "NSEW" or not s[1:].strip().isdigit():
        return None
    d = s[1:].strip()
    dd = 3 if lon else 2
    if len(d) < dd + 4:
        return None
    val = int(d[:dd]) + int(d[dd:dd + 2]) / 60 + (int(d[dd + 2:dd + 4]) + (int(d[dd + 4:dd + 5]) if len(d) > dd + 4 else 0) / 10) / 3600
    return -val if s[0] in "SW" else val


class Gazetteer:
    """ident / city+state -> coordinates, 1988 NASR first, OurAirports second."""

    def __init__(self):
        self.apt1988 = {}          # locid -> rec
        self.fss_home = defaultdict(list)   # fssid -> [rec with FSS on airport]
        self.fss_name = {}         # fssid -> name (most common)
        self.city1988 = defaultdict(list)   # (CITY, ST) -> [rec]
        self.oa_ident = {}         # code -> rec
        self.oa_city = defaultdict(list)    # (city lower, ST) -> [rec]
        self._load_1988()
        self._load_oa()

    def _load_1988(self):
        if not NASR1988.exists():
            return
        names = defaultdict(Counter)
        data = NASR1988.read_bytes().decode("ascii", "replace")
        for line in data.split("\r\n"):
            if len(line) < 300:
                continue
            lat, lon = parse_dms(line[115:123]), parse_dms(line[123:132], lon=True)
            if lat is None or lon is None:
                continue
            rec = {"id": line[11:16].strip(), "name": line[16:81].strip(), "city": line[81:107].strip(),
                   "state": line[107:109].strip(), "lat": round(lat, 5), "lon": round(lon, 5),
                   "fssid": line[265:269].strip(), "fssname": line[269:295].strip(),
                   "fss_on_apt": line[299:300] == "Y"}
            if rec["id"]:
                self.apt1988.setdefault(rec["id"], rec)
            if rec["fssid"]:
                names[rec["fssid"]][rec["fssname"]] += 1
                if rec["fss_on_apt"]:
                    self.fss_home[rec["fssid"]].append(rec)
            self.city1988[(rec["city"].upper(), rec["state"])].append(rec)
        self.fss_name = {k: v.most_common(1)[0][0] for k, v in names.items()}

    def _load_oa(self):
        if not OURAIRPORTS.exists():
            return
        rank = {"large_airport": 0, "medium_airport": 1, "small_airport": 2, "seaplane_base": 3,
                "closed": 4, "heliport": 5, "balloonport": 6}
        for r in csv.DictReader(open(OURAIRPORTS, encoding="utf-8")):
            if r["iso_country"] != "US":
                continue
            rec = {"id": r["ident"], "name": r["name"], "city": r["municipality"],
                   "state": r["iso_region"].split("-")[-1], "lat": round(float(r["latitude_deg"]), 5),
                   "lon": round(float(r["longitude_deg"]), 5), "type": r["type"], "rank": rank.get(r["type"], 9)}
            for code in (r["local_code"], r["iata_code"], r["gps_code"], r["ident"]):
                if code and code not in self.oa_ident:
                    self.oa_ident[code] = rec
                elif code and rec["rank"] < self.oa_ident[code]["rank"]:
                    self.oa_ident[code] = rec
            if r["municipality"]:
                self.oa_city[(r["municipality"].lower(), rec["state"])].append(rec)

    def by_ident(self, ident, state=None):
        """-> (rec, source) or (None, '')"""
        def ok(rec):
            return rec is not None and (not state or rec["state"] == state)
        if ident in ARTCC:
            rec, src = self.by_city(*ARTCC[ident])
            return (rec, "artcc_city") if rec else (None, "")
        homes = [r for r in self.fss_home.get(ident, []) if ok(r)]
        if homes:
            return homes[0], "fss1988_on_airport"
        rec = self.apt1988.get(ident)
        if ok(rec):
            return rec, "airport1988_ident"
        for code in (ident, "K" + ident if len(ident) == 3 else ident):
            rec = self.oa_ident.get(code)
            if ok(rec):
                return rec, "ourairports_ident"
        return None, ""

    def by_city(self, city, state):
        """City-level precision by construction: the city's principal airport.
        A large/medium OurAirports airport first, else the 1988 airport whose
        FSS sat on it, else an alphabetic (public) 1988 ident, else anything."""
        oa = sorted(self.oa_city.get((city.lower(), state), []), key=lambda r: r["rank"])
        if oa and oa[0]["rank"] <= 1:
            return oa[0], "ourairports_city"
        recs = self.city1988.get((city.upper(), state), [])
        if recs:
            recs = sorted(recs, key=lambda r: (not r["fss_on_apt"], not r["id"].isalpha()))
            return recs[0], "city1988"
        if oa:
            return oa[0], "ourairports_city"
        return None, ""


# ---------------------------------------------------------------- answers -> meta
def choice_of(ans, qid, sentinel):
    a = ans.get(qid)
    if not a:
        return None, 0.0
    return (None if a["choice"] == sentinel else a["choice"]), float(a["confidence"])


def to_year(s):
    if not s:
        return None
    m = re.search(r"(18|19|20)\d\d", s)
    if m:
        return int(m.group(0))
    m = re.search(r"/(\d\d)$", s)
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy > 26 else 2000 + yy
    return None


def interpret(post, ans, gaz, fac):
    kind, kconf = choice_of(ans, "page_kind", "")
    ftype, fconf = choice_of(ans, "facility_type", "")
    ident, iconf = choice_of(ans, "facility_ident", NONE)
    place, pconf = choice_of(ans, "place", NONE)
    year, yconf = choice_of(ans, "depicted_year", NOT_STATED)
    opened, oconf = choice_of(ans, "opened_date", NOT_STATED)
    closed, cconf = choice_of(ans, "closed_date", NOT_STATED)
    city = st = ""
    if place:
        city, _, stname = place.rpartition(", ")
        st = state_abbr(stname)
    if not city and fac:
        st, city = fac[0] if fac[0] in STATES else state_abbr(fac[0]), fac[1]
        place, pconf = f"{city}, {STATES.get(st, st)}", -1.0  # -1: from facilities.json, not Jev
    meta = {
        "href": post["href"], "title": post["title"], "categories": post["categories"],
        "image": post["image"], "pdf_links": post["pdf_links"],
        "kind": kind, "kind_conf": round(kconf, 3),
        "facility_type": ftype, "facility_type_conf": round(fconf, 3),
        "ident": ident or "", "ident_conf": round(iconf, 3),
        "place": place or "", "place_conf": round(pconf, 3), "city": city, "state": st,
        "year": to_year(year) if year else None, "year_raw": year or "", "year_conf": round(yconf, 3),
        "opened": opened or "", "opened_conf": round(oconf, 3),
        "closed": closed or "", "closed_conf": round(cconf, 3),
        "about_one_facility": round(float(ans["about_one_facility"]["noul"]), 3),
        "about_a_person": round(float(ans["about_a_person"]["noul"]), 3),
        "fss_name_1988": gaz.fss_name.get(ident, "") if ident else "",
        "lat": None, "lon": None, "coord_source": "", "coord_name": "",
    }
    rec = src = None
    if ident and iconf >= IDENT_MIN_CONF:
        rec, src = gaz.by_ident(ident, st or None)
        if rec is None and st:
            rec, src = gaz.by_ident(ident)  # ident is stronger than a possibly-wrong state
            src = src and src + "_state_mismatch"
    if rec is None and city and st and (pconf >= PLACE_MIN_CONF or pconf < 0):
        rec, src = gaz.by_city(city, st)
    if rec:
        meta.update({"lat": rec["lat"], "lon": rec["lon"], "coord_source": src, "coord_name": rec["name"]})
    return meta


# ---------------------------------------------------------------- PDFs
SERIES = [
    ("faa_world", re.compile(r"faa_?world", re.I)),
    ("alaskan_region_intercom", re.compile(r"intercom", re.I)),
    ("mukluk_telegraph", re.compile(r"mukluk", re.I)),
    ("nw_mountain_intercom", re.compile(r"nw_?mtn", re.I)),
    ("alaskan_phone_directory", re.compile(r"phone_?directory|telephone_directory", re.I)),
    ("alaskan_trapline", re.compile(r"trapline", re.I)),
    ("alaskan_division_charts", re.compile(r"div_?charts?", re.I)),
    ("faa_horizons", re.compile(r"horizons", re.I)),
    ("kohler_generators", re.compile(r"kohler", re.I)),
    ("medicine_bow_history", re.compile(r"medicine_?bow", re.I)),
]
MON_ABBR = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                                            "sep", "oct", "nov", "dec"])}


def pdf_issue_date(name):
    """(year, month|None) from a periodical filename, or (None, None)."""
    n = name.lower()
    y = None
    m = re.search(r"(19\d\d|20[0-2]\d)", n)
    if m:
        y = int(m.group(1))
    mon = None
    for k, v in MON_ABBR.items():
        if re.search(rf"(?<![a-z]){k}[a-z]*(?![a-z])", n):
            mon = v
            break
    if mon is None:
        m2 = re.search(r"(?<!\d)(\d{1,2})[-_](19\d\d|20[0-2]\d)", n)  # 10-1974, 04-1989
        if m2 and 1 <= int(m2.group(1)) <= 12:
            mon, y = int(m2.group(1)), int(m2.group(2))
        else:
            m3 = re.search(r"[-_](\d{1,2})[-_](\d{1,2})[-_](\d{4}|\d{2})\b", n)  # 01-25-1980
            if m3 and 1 <= int(m3.group(1)) <= 12:
                mon = int(m3.group(1))
                yy = m3.group(3)
                y = int(yy) if len(yy) == 4 else 1900 + int(yy)
    return y, mon


def pdf_text(path, pages=1):
    try:
        r = subprocess.run(["pdftotext", "-l", str(pages), str(path), "-"], capture_output=True,
                           text=True, timeout=120)
        return re.sub(r"\s+", " ", r.stdout).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def pdf_records(site, posts, cache_path):
    cache = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            rec = json.loads(line)
            cache[rec["path"]] = rec
    linking = defaultdict(list)
    for p in posts:
        for u in p["pdf_links"]:
            key = re.sub(r"^https?://[^/]+", "", u).split("?")[0]
            key = re.sub(r"^/atc/", "/", key)
            linking[key.lstrip("/")].append(p["href"])
    recs = []
    with open(cache_path, "a", encoding="utf-8") as out:
        for path in sorted(site.rglob("*.pdf")):
            rel = str(path.relative_to(site))
            rec = cache.get(rel)
            if rec is None:
                series = next((n for n, rx in SERIES if rx.search(rel)), "")
                y, mon = pdf_issue_date(path.name) if series else (None, None)
                text = pdf_text(path)
                rec = {"path": rel, "href": "/atc/" + rel, "name": path.name, "series": series,
                       "issue_year": y, "issue_month": mon, "bytes": path.stat().st_size,
                       "has_text": len(text) > 200, "text_head": text[:PDF_TEXT_CAP]}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rec["linking_posts"] = sorted(set(linking.get(rel, [])))
            recs.append(rec)
    return recs


def pdf_spec(rec, gaz):
    """Jev questions for a non-periodical PDF with a text layer."""
    fake_post = {"title": rec["name"], "text": rec["text_head"], "categories": ["PDF"]}
    idents, places, date_cands, year_cands = candidates(fake_post, None)
    state = {"title": rec["name"], "categories": ["PDF document"], "text": rec["text_head"]}
    return state, questions_for(idents, places, date_cands, year_cands)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", type=Path, default=SITE_DEFAULT)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merge-only", action="store_true", help="re-interpret stored answers; ask nothing")
    ap.add_argument("--no-pdfs", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    posts = load_posts(args.site)
    fac = load_facilities()
    gaz = Gazetteer()
    print(f"{len(posts)} posts; facilities index covers {sum(1 for p in posts if p['href'] in fac)}; "
          f"gazetteer: {len(gaz.apt1988)} 1988 airports, {len(gaz.fss_home)} FSS with on-airport "
          f"homes, {len(gaz.fss_name)} FSS names, {len(gaz.oa_ident)} OurAirports codes", file=sys.stderr)
    todo = posts[: args.limit] if args.limit else posts

    answers_path = DATA_DIR / "atc_posts_answers.jsonl"
    answers = {}
    if answers_path.exists():
        for line in open(answers_path, encoding="utf-8"):
            rec = json.loads(line)
            answers[rec["id"]] = rec

    specs = [(p, *build_spec(p, fac.get(p["href"]))) for p in todo]
    tok = sum(len(json.dumps(s, ensure_ascii=False)) // 4 + len(json.dumps(q)) // 4 for _, s, q in specs)
    print(f"{len(specs)} post requests ≈ {tok:,} input tokens ≈ ${tok / 1e6 * 0.042:.2f}", file=sys.stderr)
    if args.dry_run:
        with_id = sum(1 for _, _, q in specs if "facility_ident" in q)
        with_pl = sum(1 for _, _, q in specs if "place" in q)
        with_dt = sum(1 for _, _, q in specs if "opened_date" in q)
        print(f"candidates: idents on {with_id}, places on {with_pl}, dates on {with_dt} posts", file=sys.stderr)
        for p, s, q in specs[:4]:
            print(json.dumps({"title": p["title"], "idents": [k for k in q.get("facility_ident", {}).get("criteria", {}) if k != NONE],
                              "places": [k for k in q.get("place", {}).get("criteria", {}) if k != NONE],
                              "years": [k for k in q["depicted_year"]["criteria"] if k != NOT_STATED][:8]}, ensure_ascii=False))
    elif not args.merge_only:
        jev = JevClient(task="atc_posts", workers=args.workers)
        with open(answers_path, "a", encoding="utf-8") as out:
            for (p, s, q), ans in jev.ask_many(specs, lambda sp: (sp[0]["href"], sp[1], sp[2]), label="posts "):
                if isinstance(ans, JevError):
                    print(f"  ERROR {p['href']}: {ans}", file=sys.stderr)
                    continue
                rec = {"id": p["href"], "model": jev.model, "answers": ans}
                answers[p["href"]] = rec
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(jev.summary(), file=sys.stderr)

    # -- PDFs (code part always; Jev part for non-periodicals with text)
    pdfs = []
    if not args.no_pdfs:
        pdfs = pdf_records(args.site, posts, OUT_DIR / "pdfs_scan.jsonl")
        eligible = [r for r in pdfs if not r["series"] and r["has_text"]]
        print(f"{len(pdfs)} PDFs: {sum(1 for r in pdfs if r['series'])} in periodical series, "
              f"{sum(1 for r in pdfs if r['has_text'])} with a text layer, {len(eligible)} non-periodical "
              f"with text -> Jev", file=sys.stderr)
        if not args.dry_run and not args.merge_only and eligible:
            jev2 = JevClient(task="atc_pdfs", workers=args.workers)
            with open(answers_path, "a", encoding="utf-8") as out:
                for r, ans in jev2.ask_many(eligible, lambda r: (r["href"], *pdf_spec(r, gaz)), label="pdfs "):
                    if isinstance(ans, JevError):
                        print(f"  ERROR {r['href']}: {ans}", file=sys.stderr)
                        continue
                    rec = {"id": r["href"], "model": jev2.model, "answers": ans}
                    answers[r["href"]] = rec
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(jev2.summary(), file=sys.stderr)

    if args.dry_run:
        return

    # -- interpret + join
    metas = []
    for p in posts:
        rec = answers.get(p["href"])
        if not rec:
            continue
        m = interpret(p, rec["answers"], gaz, fac.get(p["href"]))
        m["jev_model"] = rec["model"]
        m["answers"] = rec["answers"]
        metas.append(m)
    with open(OUT_DIR / "posts_meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    for r in pdfs:
        rec = answers.get(r["href"])
        if rec:
            fake = {"href": r["href"], "title": r["name"], "categories": ["PDF"], "image": "", "pdf_links": []}
            m = interpret(fake, rec["answers"], gaz, None)
            r.update({k: m[k] for k in ("kind", "facility_type", "ident", "place", "city", "state",
                                        "year", "lat", "lon", "coord_source", "coord_name")})
        r.pop("text_head", None)
    with open(OUT_DIR / "pdfs.jsonl", "w", encoding="utf-8") as f:
        for r in pdfs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    index = [{k: m[k] for k in ("href", "title", "kind", "facility_type", "ident", "fss_name_1988", "city",
                                "state", "year", "opened", "closed", "lat", "lon", "coord_source", "image",
                                "categories")} for m in metas]
    (OUT_DIR / "posts_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=0) + "\n")
    print(f"wrote {len(metas)} posts -> {OUT_DIR}/posts_meta.jsonl, posts_index.json; {len(pdfs)} PDFs -> pdfs.jsonl")
    if args.report:
        report(metas, fac)


def report(metas, fac):
    n = len(metas)
    print(f"\n== {n} posts ==")
    print("kind:", dict(Counter(m["kind"] for m in metas).most_common()))
    print("facility_type:", dict(Counter(m["facility_type"] for m in metas).most_common()))
    print(f"ident chosen: {sum(1 for m in metas if m['ident'])} "
          f"(conf>=0.5: {sum(1 for m in metas if m['ident'] and m['ident_conf'] >= 0.5)}); "
          f"place chosen: {sum(1 for m in metas if m['place'])}; year: {sum(1 for m in metas if m['year'])}; "
          f"opened: {sum(1 for m in metas if m['opened'])}; closed: {sum(1 for m in metas if m['closed'])}")
    print("located:", dict(Counter(m["coord_source"] or "(none)" for m in metas).most_common()))
    # regression against the restored facility index (state + city)
    agree = state_ok = total = 0
    for m in metas:
        f = fac.get(m["href"])
        if not f or m["place_conf"] < 0:
            continue
        total += 1
        st = f[0] if f[0] in STATES else state_abbr(f[0])
        if m["state"] == st:
            state_ok += 1
            if m["city"].lower() == f[1].lower():
                agree += 1
    if total:
        print(f"facilities.json regression ({total} posts with a Jev place): state agrees {state_ok} "
              f"({100 * state_ok // total}%), city+state agrees {agree} ({100 * agree // total}%)")
    lows = sorted(metas, key=lambda m: m["kind_conf"])[:8]
    print("lowest kind confidence:", [(m["title"][:40], m["kind"], m["kind_conf"]) for m in lows])


if __name__ == "__main__":
    main()
