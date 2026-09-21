#!/usr/bin/env python3
"""Build archive.aero's airspace overlay -- class airspace from every held AIS
snapshot of three regions -- and publish it to R2.

What it makes: one vector PMTiles archive (tile layer ``class``) holding every
class-airspace polygon from every held cycle of every region, merged so that a
polygon version that is unchanged across consecutive held cycles is stored
once with a validity interval::

    rg    region code: us | fr | br
    from  effective date of the first cycle carrying this version (YYYYMMDD int)
    to    effective date of the first later held cycle without it (absent while
          the version is still present in the region's newest held cycle)

The viewer (index.html, AirspaceLayer) draws the versions whose interval
contains the timeline date -- but only while a held cycle of that region is in
effect on it: a cycle is good for ``cycle_days`` (28), so a hole in a region's
series, or a date past its newest cycle, draws nothing and the panel says why.
Scrubbing changes which airspace is drawn without refetching a tile.

Regions and sources (all under /Volumes/projects/aisdata, pulled by
aisdata_pull.py):

  us  FAA NASR 28-day subscription, ``Additional_Data/Shape_Files/
      Class_Airspace.shp`` in every ``us_faa/nasr/<effective>/*.zip`` (every
      cycle from 2020-03-26 ships it). Class B/C/D/E with the FAA's local
      types (E2..E7); the Class E floor areas also feed the ``efloor`` edge
      layer (the sectional vignette, see e_floor_edges).
  fr  SIA (DGAC) per-AIRAC database export, the ``XML_SIA_<date>.xml`` member
      of every ``fr_sia/cquest_mirror/export_xml_bd_sia_*.zip`` (Christian
      Quest's Licence Ouverte mirror, 2019-02..2023-10, with holes) and of
      every hand-downloaded ``fr_sia/cycles/<date>/*.zip``. An ``Espace``
      (CTR, TMA, CTA, LTA...) has ``Partie`` parts, each with the polygon
      already densified as a lat,lon list (arcs, circles and border-following
      segments resolved by the SIA), and each part has stacked ``Volume``
      shelves carrying the ICAO class, floor, ceiling and hours. Kept: CTR,
      TMA and CTA volumes of class A-E, plus the LTA parts that have a Class E
      volume (the mountain parts drawn on the OACI chart; the national FL115
      Class D blanket restates the general rule and is skipped, as are FIR/
      UIR/UTA/OCA/FRA, ATC sectors, RMZ/TMZ, cross-border delegations and
      every SUA type). When one effective date has two exports (2020-03-26)
      the newer ``SiaExport Date`` wins.
  br  DECEA GeoAISWEB WFS snapshots, ``br_geoaisweb/snapshots/<date>/
      {TMA,CTR,CTA,ATZ}.geojson``. TMA rows include the shelf parts
      (SBXP, SBXP_01, SBXP_02...); ``setores_tma`` is ATC sectorisation and is
      not airspace structure. The WFS publishes no ICAO class, so these
      features carry a type but no ``cls`` and the viewer draws them as
      controlled airspace of unstated class. The cycle is the AIP amendment
      (``emenda``) the snapshot was taken under; a feature's own
      ``effectived`` is kept as ``eff`` for information but never extends
      validity backwards (nothing is known about what else was in force then).

Identity is content in every region (only the ADDS GeoJSON has a stable id):
a *version* is the hash of the polygon (coordinates rounded to 1e-6 deg) plus
the attributes that decide how it is drawn -- class, local type, floor/ceiling,
hours code, sector, exclusion flag. Name, identifier and free-text remarks are
NOT hashed: the FAA re-spelled 1,329 idents (KSTL -> STL) between 2020 and
2021 without touching a boundary. They are taken from the newest cycle that
carries the version.

Stages (each resumable, each skipped when its output is current):

  parse   every unparsed cycle of every selected region -> a per-region cache,
          worklists/data/airspace/class_versions[_<rg>].sqlite (geometry
          stored once per distinct shape as zlib'd GeoJSON)
  merge   membership runs -> intervals -> newline-delimited GeoJSON carrying a
          per-feature tippecanoe minzoom: everything from z5, the FAA's
          E5-E7 floor areas from z6 (the map starts at z6 and protomaps-leaflet
          reads data one level below the map zoom)
  edges   (us) the vignette lines: for every Class E floor polygon (E5/E6/E7)
          the part of its boundary that is a real floor change -- a 700 ft
          area's edge minus edges shared with other 700 ft areas, a 1,200 ft-
          or-higher area's edge minus edges shared with *any* other Class E
          polygon (the state-wide 1,200 ft blankets are cut around every 700 ft
          area, and sectionals draw the blue vignette only where Class E meets
          Class G). Emitted as tile layer ``efloor``, each line oriented with
          the controlled side on its left, so the viewer can shade one side.
  tile    tippecanoe -Z5 -z11, shared borders, 32/256 buffer (the viewer
          strokes a vignette up to ~20 px inside the edges; the buffer keeps
          the stroke along tile-cut edges outside the visible tile), and no
          feature dropping of any kind -- a missing Class D is a data error
  stamp   pmtiles edit: the archive's JSON metadata gains ``archive_aero``
          (per-region cycle lists, extents and counts) so the viewer can name
          the cycle in effect for the region under view
  verify  pmtiles verify, a manifest beside the output, a line in builds.jsonl
  upload  rclone copyto -> r2:charts/airspace/class-<stamp>.pmtiles (--upload)

The published key is dated and immutable (same rule as basemap_build.py):
replacing a PMTiles under a live key moves every byte offset in it while the
tiles Worker's range cache carries no version. The stamp defaults to the build
date; a second build the same day needs --stamp (e.g. 20260915b) and the
upload refuses to overwrite. --update-html rewrites CONFIG.airspaceUrl in
index.html the way build_metadata_bundle.py rewrites bundleUrl. The first
series (``airspace/nasr-20261001.pmtiles``, US only) stays in R2 untouched.

Usage
-----
    ~/venv/bin/python scripts/airspace_build.py --parse-only        # fill the caches
    ~/venv/bin/python scripts/airspace_build.py                     # build, all regions
    ~/venv/bin/python scripts/airspace_build.py --regions fr --since 2023-01-01 --stamp test1
    ~/venv/bin/python scripts/airspace_build.py --report            # churn per cycle, per region
    ~/venv/bin/python scripts/airspace_build.py --upload --update-html index.html
    ~/venv/bin/python scripts/airspace_build.py --publish-only --stamp 20260916 --update-html index.html
                                                    # upload an archive already built and checked locally
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, NamedTuple

from osgeo import gdal, ogr

gdal.UseExceptions()

REPO = Path(__file__).resolve().parent.parent
AIS_ROOT = Path("/Volumes/projects/aisdata")
CACHE_DIR = REPO / "worklists" / "data" / "airspace"      # gitignored (worklists/data/)
OUT_DIR = Path("/Volumes/projects/airspace_pmtiles")     # local mirror, keeps .pmtiles
R2_REMOTE = "r2:charts"
R2_PREFIX = "airspace"
SERIES = "class"
LAYER = "class"
EDGE_LAYER = "efloor"     # oriented Class E floor edges, the vignette lines (see e_floor_edges)
MINZOOM, MAXZOOM = 5, 11
CYCLE_DAYS = 28           # an AIRAC / NASR cycle is good for 28 days
# Data zoom at which an FAA local type first appears in the tiles. The viewer
# maps display zoom z to data zoom z-1, so 5 = visible from the map's minimum
# zoom (6), 6 = from z7, where a vignette is wide enough to read.
US_FEATURE_MINZOOM = {"E5": 6, "E6": 6, "E7": 6}
NULL_ALT = {"", "-9998", "-9999"}


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], dry: bool = False, **kw) -> subprocess.CompletedProcess | None:
    log("+ " + " ".join(str(c) for c in cmd))
    if dry:
        return None
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


# ---------------------------------------------------------------- normalisation

def norm_str(s) -> str | None:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s or None


def norm_alt(s) -> int | None:
    """Altitude field -> int feet (or flight level number); -9998 means 'none'."""
    s = (str(s) if s is not None else "").strip()
    if s in NULL_ALT:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def count_vertices(g: ogr.Geometry) -> int:
    n = g.GetGeometryCount()
    if n == 0:
        return g.GetPointCount()
    return sum(count_vertices(g.GetGeometryRef(i)) for i in range(n))


# ---------------------------------------------------------------- manifests

def manifest_shas(path: Path) -> dict[str, str]:
    """file (relative to the source dir) -> sha256, from aisdata_pull.py's
    manifest (newest line wins)."""
    shas: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("sha256") and row.get("file"):
                shas[row["file"]] = row["sha256"]
    return shas


def file_identity(p: Path, shas: dict[str, str], rel: str) -> str:
    """What 'this cycle has been parsed' is keyed on: the manifest sha256 when
    the pull recorded one, else size+mtime (a re-download shows up as a change)."""
    if rel in shas:
        return shas[rel]
    st = p.stat()
    return f"size{st.st_size}-mtime{int(st.st_mtime)}"


# ---------------------------------------------------------------- regions

class Cycle(NamedTuple):
    effective: str    # YYYY-MM-DD
    identity: str     # parse key (manifest sha256 or size+mtime)
    locator: str      # where it came from, for the log
    handle: object    # region-specific: paths / members


Feature = tuple  # (ogr.Geometry | None, core: dict, desc: dict)


class Region:
    code = ""
    name = ""
    source = ""          # short label the viewer prints ("FAA NASR cycle ...")
    source_url = ""
    licence = ""
    db_name = ""
    cycle_days = CYCLE_DAYS
    note: str | None = None

    def __init__(self) -> None:
        self.skipped: collections.Counter = collections.Counter()

    @property
    def root(self) -> Path:
        raise NotImplementedError

    @property
    def db_path(self) -> Path:
        return CACHE_DIR / self.db_name

    def held_cycles(self) -> list[Cycle]:
        raise NotImplementedError

    def features(self, cyc: Cycle, tmp: Path) -> Iterator[Feature]:
        raise NotImplementedError

    def minzoom(self, core: dict) -> int:
        return MINZOOM

    def box_group(self, desc: dict, bbox: list) -> str:
        """Key for the extent boxes stamped in the metadata (one box per group,
        so a region spread over the globe gets several tight boxes)."""
        return "all"

    def type_key(self, core: dict) -> str:
        lt = core.get("lt") or "?"
        return f"{lt}/{core['cls']}" if core.get("cls") else lt


# ---- United States, FAA NASR ------------------------------------------------

class UsNasr(Region):
    code = "us"
    name = "United States"
    source = "FAA NASR"
    source_url = "https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/"
    licence = "US Government work, public domain"
    db_name = "class_versions.sqlite"        # the original single-region cache, unchanged
    SHP_MEMBER = "Additional_Data/Shape_Files/Class_Airspace.shp"

    @property
    def root(self) -> Path:
        return AIS_ROOT / "us_faa" / "nasr"

    def zip_path(self, eff: str) -> Path:
        return self.root / eff / f"28DaySubscription_Effective_{eff}.zip"

    def held_cycles(self) -> list[Cycle]:
        shas = manifest_shas(AIS_ROOT / "us_faa" / "manifest.jsonl")
        out = []
        for d in sorted(self.root.iterdir()):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name) and self.zip_path(d.name).exists():
                p = self.zip_path(d.name)
                out.append(Cycle(d.name, file_identity(p, shas, f"nasr/{d.name}/{p.name}"), str(p), p))
        return out

    def features(self, cyc: Cycle, tmp: Path) -> Iterator[Feature]:
        # The shapefile is unpacked to a scratch dir and read from disk: reading
        # it through GDAL's /vsizip/ layer leaked ~2 GB per cycle (62 GB peak
        # over a run) and the OS killed the parse twice before this was found.
        with zipfile.ZipFile(cyc.handle) as zf:
            stem = self.SHP_MEMBER[:-4]
            for member in zf.namelist():
                if member.startswith(stem):
                    Path(tmp, Path(member).name).write_bytes(zf.read(member))
        path = str(tmp / Path(self.SHP_MEMBER).name)
        ds = ogr.Open(path)
        if ds is None:
            raise SystemExit(f"{cyc.effective}: cannot open {path}")
        lyr = ds.GetLayer(0)
        for f in lyr:
            g = f.GetGeometryRef()
            if g is None or g.IsEmpty():
                yield None, {}, {}
                continue
            if g.Is3D() or g.IsMeasured():
                g.FlattenTo2D()   # some rows are PolygonZ with z=0; the third ordinate is noise
            core, desc = self.normalize(f)
            yield g.Clone(), core, desc
        lyr = None
        ds = None

    @staticmethod
    def normalize(f: ogr.Feature) -> tuple[dict, dict]:
        """Split a shapefile row into the hashed core and the unhashed description."""
        g = lambda k: f.GetField(k)  # noqa: E731
        lt = (norm_str(g("LOCAL_TYPE")) or "").replace("CLASS_", "")
        core = {
            "cls": norm_str(g("CLASS")),
            "lt": lt or None,
            "lo": norm_alt(g("LOWER_VAL")),
            "loc": norm_str(g("LOWER_CODE")),
            "lou": norm_str(g("LOWER_UOM")),
            "hi": norm_alt(g("UPPER_VAL")),
            "hic": norm_str(g("UPPER_CODE")),
            "hiu": norm_str(g("UPPER_UOM")),
            "hrs": norm_str(g("WKHR_CODE")),
            "rmk": norm_str(g("WKHR_RMK")),
            "sec": norm_str(g("SECTOR")),
            "ex": 1 if norm_str(g("EXCLUSION")) == "1" else 0,
        }
        # Units are FT unless the FAA says FL; ceilings/floors without a value have
        # no meaningful unit. Dropping the defaults keeps the tile properties short.
        for side in ("lo", "hi"):
            if core[side] is None:
                core[side + "u"] = None
            elif core[side + "u"] == "FT":
                core[side + "u"] = None
        if not core["ex"]:
            core["ex"] = None
        desc = {
            "name": norm_str(g("NAME")),
            "id": norm_str(g("IDENT")),
            "lod": norm_str(g("LOWER_DESC")),
            "hid": norm_str(g("UPPER_DESC")),
        }
        return clean(core), clean(desc)

    def minzoom(self, core: dict) -> int:
        return US_FEATURE_MINZOOM.get(core.get("lt") or "", MINZOOM)

    def box_group(self, desc: dict, bbox: list) -> str:
        # The FAA pre-splits the Aleutian chain at 180 deg and Guam/Saipan sit
        # at 145 E: one box per hemisphere keeps the western box off the Pacific.
        return "e" if bbox[0] >= 0 else "w"

    def type_key(self, core: dict) -> str:
        return core.get("lt") or "?"


# ---- France, SIA -------------------------------------------------------------

class FrSia(Region):
    code = "fr"
    name = "France"
    source = "SIA"
    source_url = "https://www.sia.aviation-civile.gouv.fr/produits-numeriques-en-libre-disposition/les-bases-de-donnees-sia.html"
    licence = "Licence Ouverte / Open Licence 2.0 (Etalab)"
    db_name = "class_versions_fr.sqlite"
    TYPES = {"CTR", "TMA", "CTA", "LTA"}
    CLASSES = set("ABCDE")
    ALT_UNITS = {"SFC": "SFC", "ft ASFC": "SFC", "ft AMSL": "MSL", "FL": "STD", "UNL": "UNLTD"}

    @property
    def root(self) -> Path:
        return AIS_ROOT / "fr_sia"

    def held_cycles(self) -> list[Cycle]:
        shas = manifest_shas(self.root / "manifest.jsonl")
        mirror = self.root / "cquest_mirror"
        files = sorted(set(list(mirror.glob("export_xml_bd_sia_*.zip")) + list(mirror.glob("exports_*.zip"))
                           + list(mirror.glob("export_xml_bd_sia_*.xml.gz"))
                           + list((self.root / "cycles").glob("*/*.zip"))))
        cands: dict[str, list[tuple[str, Path, str | None]]] = {}
        for p in files:
            if p.name.endswith(".xml.gz"):
                member = None
                m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
                with gzip.open(p) as fh:
                    head = fh.read(600)
            else:
                with zipfile.ZipFile(p) as zf:
                    members = [n for n in zf.namelist() if re.search(r"XML_SIA[^/]*\.xml$", n)]
                    if not members:
                        log(f"  fr: {p.name} has no XML_SIA member, skipped")
                        continue
                    member = members[0]
                    m = re.search(r"(\d{4}-\d{2}-\d{2})\.xml$", member)
                    with zf.open(member) as fh:
                        head = fh.read(600)
            if not m:
                log(f"  fr: no effective date in {p.name}, skipped")
                continue
            exported = re.search(rb'SiaExport Date="([^"]+)"', head)
            cands.setdefault(m.group(1), []).append(
                (exported.group(1).decode() if exported else "", p, member))
        out = []
        for eff, lst in sorted(cands.items()):
            lst.sort(key=lambda t: t[0])            # newest SiaExport Date last
            exported, p, member = lst[-1]
            if len(lst) > 1:
                log(f"  fr {eff}: {len(lst)} exports held, using {p.name} (exported {exported})")
            rel = str(p.relative_to(self.root))
            out.append(Cycle(eff, file_identity(p, shas, rel), f"{p}::{member or ''}", (p, member)))
        return out

    def features(self, cyc: Cycle, tmp: Path) -> Iterator[Feature]:
        p, member = cyc.handle
        if member is None:
            fh = gzip.open(p)
        else:
            zf = zipfile.ZipFile(p)
            fh = zf.open(member)
        espaces: dict[str, tuple] = {}
        parties: dict[str, dict] = {}
        volumes: list[dict] = []
        eff_seen = None
        stack: list[str] = []
        with fh:
            for ev, el in ET.iterparse(fh, events=("start", "end")):
                if ev == "start":
                    stack.append(el.tag)
                    if el.tag == "Situation":
                        eff_seen = el.get("effDate")
                    continue
                stack.pop()
                if len(stack) != 3:
                    continue
                if el.tag == "Espace":
                    terr = el.find("Territoire")
                    espaces[el.get("pk")] = (el.findtext("TypeEspace"), el.findtext("Nom"),
                                             (terr.get("lk") if terr is not None else "") or "")
                elif el.tag == "Partie":
                    ref = el.find("Espace")
                    parties[el.get("pk")] = {
                        "esp": ref.get("pk") if ref is not None else None,
                        "nom": norm_str(el.findtext("NomPartie")),
                        "geom": el.findtext("Geometrie") or "",
                    }
                elif el.tag == "Volume":
                    ref = el.find("Partie")
                    volumes.append({
                        "partie": ref.get("pk") if ref is not None else None,
                        "seq": norm_str(el.findtext("Sequence")),
                        "cls": norm_str(el.findtext("Classe")),
                        "lo": el.findtext("Plancher"), "lou": el.findtext("PlancherRefUnite"),
                        "lo2": norm_str(el.findtext("Plancher2")),
                        "hi": el.findtext("Plafond"), "hiu": el.findtext("PlafondRefUnite"),
                        "hi2": norm_str(el.findtext("Plafond2")),
                        "hrs": norm_str(el.findtext("HorCode")),
                        "hrt": norm_str(el.findtext("HorTxt")),
                    })
                el.clear()
        if eff_seen and eff_seen != cyc.effective:
            log(f"  fr {cyc.effective}: file says effDate {eff_seen}!")
        # LTA: only the parts where Class E rises above FL115 (Alps, Pyrenees)
        # are drawn on the OACI chart; the national Class D blanket is the rule.
        lta_e_parts = {v["partie"] for v in volumes if v["cls"] == "E"
                       and espaces.get(parties.get(v["partie"], {}).get("esp"), ("",))[0] == "LTA"}
        polys: dict[str, ogr.Geometry | None] = {}
        for v in volumes:
            part = parties.get(v["partie"])
            esp = espaces.get(part["esp"]) if part else None
            if not part or not esp:
                self.skipped["orphan volume"] += 1
                continue
            etype, ename, terr = esp
            if etype not in self.TYPES:
                continue
            if etype == "LTA" and v["partie"] not in lta_e_parts:
                self.skipped["LTA blanket part"] += 1
                continue
            if v["cls"] not in self.CLASSES:
                self.skipped[f"class {v['cls'] or 'none'}"] += 1
                continue
            if v["partie"] not in polys:
                polys[v["partie"]] = self.polygon(part["geom"])
            g = polys[v["partie"]]
            lo, loc, lou = self.alt(v["lo"], v["lou"])
            hi, hic, hiu = self.alt(v["hi"], v["hiu"])
            core = clean({"cls": v["cls"], "lt": etype, "lo": lo, "loc": loc, "lou": lou,
                          "hi": hi, "hic": hic, "hiu": hiu,
                          "hrs": v["hrs"] if v["hrs"] and v["hrs"] != "H24" else None})
            name = ename or ""
            if part["nom"] and part["nom"] != ".":
                name = f"{name} {part['nom']}".strip()
            desc = clean({
                "name": norm_str(name),
                "id": norm_str(f"{etype} {name}"),
                "rmk": v["hrt"][:200] if v["hrt"] and v["hrs"] != "H24" else None,
                "lod": self.alt_text(v["lo"], v["lou"], v["lo2"]),
                "hid": self.alt_text(v["hi"], v["hiu"], v["hi2"]),
                "terr": re.sub(r"[^A-Z-]", "", terr.upper()) or None,
                "seq": v["seq"],
            })
            yield (g.Clone() if g is not None else None), core, desc

    @staticmethod
    def polygon(text: str) -> ogr.Geometry | None:
        """A Partie's Geometrie: one 'lat,lon' per line, already densified."""
        ring = ogr.Geometry(ogr.wkbLinearRing)
        n = 0
        first = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lat, lon = (float(t) for t in line.split(","))
            except ValueError:
                continue
            ring.AddPoint_2D(lon, lat)
            first = first or (lon, lat)
            n += 1
        if n < 4:
            return None
        x, y = ring.GetPoint_2D(n - 1)
        if (x, y) != first:
            ring.AddPoint_2D(*first)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        return poly

    def alt(self, val, unit) -> tuple[int | None, str | None, str | None]:
        """SIA (value, reference unit) -> NASR-style (value, code, unit)."""
        unit = norm_str(unit) or ""
        code = self.ALT_UNITS.get(unit)
        if code is None:
            self.skipped[f"altitude unit {unit!r}"] += 1
            return None, None, None
        if code == "UNLTD":
            return None, "UNLTD", None
        v = norm_alt(val)
        if unit == "SFC" or (code == "SFC" and not v):
            return 0, "SFC", None
        return v, code, ("FL" if code == "STD" else None)

    @staticmethod
    def alt_text(val, unit, alt2) -> str | None:
        v, u = norm_str(val), norm_str(unit)
        if not v:
            return None
        s = "SFC" if u == "SFC" else f"{v} {u}"
        if alt2:
            s += f" / {alt2} ft ASFC"   # Plancher2/Plafond2: the alternative reference
        return s

    def box_group(self, desc: dict, bbox: list) -> str:
        # One box per territory (metropole, Antilles, Guyane, Reunion...), and
        # the territory code [LF] also covers Saint-Pierre-et-Miquelon: keep the
        # western Atlantic out of the metropolitan box.
        return f"{desc.get('terr') or 'LF'}:{'w' if bbox[2] < -30 else 'e'}"


# ---- Brazil, DECEA GeoAISWEB -----------------------------------------------------

class BrGeoAisweb(Region):
    code = "br"
    name = "Brazil"
    source = "DECEA GeoAISWEB"
    source_url = "https://geoaisweb.decea.mil.br/"
    licence = "DECEA AISWEB terms of use (official public AIS data, attribution)"
    db_name = "class_versions_br.sqlite"
    note = "current snapshots only; the WFS publishes no ICAO class"
    LAYERS = ("TMA", "CTR", "CTA", "ATZ")
    TYPES = {"TMA": "TMA", "CTR": "CTR", "CTA": "CTA", "CTA_P": "CTA", "ATZ": "ATZ"}

    @property
    def root(self) -> Path:
        return AIS_ROOT / "br_geoaisweb" / "snapshots"

    def held_cycles(self) -> list[Cycle]:
        shas = manifest_shas(AIS_ROOT / "br_geoaisweb" / "manifest.jsonl")
        by_eff: dict[str, list[tuple[str, Path]]] = {}
        for d in sorted(self.root.iterdir()):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name):
                continue
            paths = [d / f"{name}.geojson" for name in self.LAYERS]
            if not all(p.exists() for p in paths):
                log(f"  br: snapshot {d.name} lacks a layer, skipped")
                continue
            eff = self.amendment(paths) or d.name
            ident = hashlib.sha1("|".join(
                file_identity(p, shas, f"snapshots/{d.name}/{p.name}") for p in paths).encode()).hexdigest()
            by_eff.setdefault(eff, []).append((d.name, d))
        out = []
        for eff, lst in sorted(by_eff.items()):
            snap, d = lst[-1]                      # newest snapshot of the same amendment wins
            if len(lst) > 1:
                log(f"  br {eff}: {len(lst)} snapshots of this amendment, using {snap}")
            paths = [d / f"{name}.geojson" for name in self.LAYERS]
            ident = hashlib.sha1("|".join(
                file_identity(p, shas, f"snapshots/{snap}/{p.name}") for p in paths).encode()).hexdigest()
            out.append(Cycle(eff, ident, str(d), d))
        return out

    @staticmethod
    def amendment(paths: list[Path]) -> str | None:
        """The AIP amendment (``emenda``) the snapshot was taken under: the most
        common value over the layers (ATZ rows carry none)."""
        votes: collections.Counter = collections.Counter()
        for p in paths:
            for f in json.loads(p.read_text())["features"]:
                e = f["properties"].get("emenda")
                if e:
                    votes[str(e)[:10]] += 1
        return votes.most_common(1)[0][0] if votes else None

    def features(self, cyc: Cycle, tmp: Path) -> Iterator[Feature]:
        for name in self.LAYERS:
            for f in json.loads((cyc.handle / f"{name}.geojson").read_text())["features"]:
                p = f["properties"]
                lt = self.TYPES.get(norm_str(p.get("typ")) or "")
                if lt is None:
                    self.skipped[f"type {p.get('typ')}"] += 1
                    continue
                g = ogr.CreateGeometryFromJson(json.dumps(f["geometry"])) if f.get("geometry") else None
                lo, loc, lou = self.alt(p.get("lowerlimi1"), p.get("codedistv1"), p.get("lowerlimit"))
                hi, hic, hiu = self.alt(p.get("upperlimit"), p.get("codedistve"), p.get("uplimituni"))
                hrs = norm_str(p.get("codewrkhr"))
                core = clean({"lt": lt, "lo": lo, "loc": loc, "lou": lou, "hi": hi, "hic": hic, "hiu": hiu,
                              "hrs": hrs if hrs and hrs != "H24" else None})
                desc = clean({"name": norm_str(p.get("nam")), "id": norm_str(p.get("ident")),
                              "eff": (str(p["effectived"])[:10] if p.get("effectived") else None),
                              "fir": norm_str(p.get("relatedfir"))})
                yield g, core, desc

    def alt(self, val, code, unit) -> tuple[int | None, str | None, str | None]:
        code, unit = norm_str(code), norm_str(unit)
        if unit == "UNL" or code == "UNL":
            return None, "UNLTD", None
        v = norm_alt(val)
        if unit == "GND" or code == "SFC":
            return (v or 0), "SFC", None
        if code == "STD":
            return v, "STD", "FL"
        if code == "MSL":
            return v, "MSL", None
        if v is None and code is None:
            return None, None, None
        self.skipped[f"altitude code {code!r}/{unit!r}"] += 1
        return v, code, None


REGIONS: dict[str, Region] = {r.code: r for r in (UsNasr(), FrSia(), BrGeoAisweb())}


# ---------------------------------------------------------------- cache db

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles(
  effective TEXT PRIMARY KEY, zip_id TEXT, n_features INT, n_versions INT,
  n_invalid INT, n_nogeom INT, parsed_at TEXT);
CREATE TABLE IF NOT EXISTS geoms(
  ghash TEXT PRIMARY KEY, gz BLOB, nvert INT, valid INT, bbox TEXT);
CREATE TABLE IF NOT EXISTS versions(
  vhash TEXT PRIMARY KEY, ghash TEXT, core TEXT, desc TEXT, desc_cycle TEXT);
CREATE TABLE IF NOT EXISTS members(
  effective TEXT, vhash TEXT, n INT, PRIMARY KEY(effective, vhash));
CREATE INDEX IF NOT EXISTS members_v ON members(vhash);
"""


def open_db(region: Region) -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # WAL + a long busy timeout: a --report (or a second build) reading the
    # cache while a parse commits must wait, not fail either side.
    db = sqlite3.connect(region.db_path, timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def parse_cycle(region: Region, db: sqlite3.Connection, cyc: Cycle) -> None:
    t0 = time.time()
    region.skipped.clear()
    with tempfile.TemporaryDirectory(prefix=f"airspace_{region.code}_") as tmp:
        stats = ingest(db, cyc, region.features(cyc, Path(tmp)))
    n_feat, n_members, n_invalid, n_nogeom = stats
    dup = n_feat - n_nogeom - n_members
    extra = ", ".join(f"{v} {k}" for k, v in sorted(region.skipped.items()))
    log(f"  {region.code} {cyc.effective}: {n_feat} rows -> {n_members} versions"
        f"{f', {dup} exact dupes' if dup else ''}"
        f"{f', {n_invalid} new invalid geometries' if n_invalid else ''}"
        f"{f', {n_nogeom} without geometry' if n_nogeom else ''}"
        f"{f'; skipped {extra}' if extra else ''}  ({time.time() - t0:.0f}s, rss {rss_gb():.1f} GB)")


def ingest(db: sqlite3.Connection, cyc: Cycle, feats: Iterator[Feature]) -> tuple[int, int, int, int]:
    members: dict[str, int] = {}
    n_feat = n_invalid = n_nogeom = 0
    cur = db.cursor()
    cur.execute("BEGIN")
    # A version's description follows the newest cycle carrying it, so a
    # re-parse of an old cycle must not clobber a newer spelling.
    for g, core, desc in feats:
        n_feat += 1
        if g is None or g.IsEmpty():
            n_nogeom += 1
            continue
        gjson = g.ExportToJson(["COORDINATE_PRECISION=6"])
        ghash = hashlib.sha1(gjson.encode()).hexdigest()
        core_s = json.dumps(core, sort_keys=True, separators=(",", ":"))
        vhash = hashlib.sha1((ghash + "|" + core_s).encode()).hexdigest()
        if cur.execute("SELECT 1 FROM geoms WHERE ghash=?", (ghash,)).fetchone() is None:
            valid = 1 if g.IsValid() else 0
            n_invalid += 0 if valid else 1
            e = g.GetEnvelope()
            cur.execute("INSERT INTO geoms VALUES(?,?,?,?,?)",
                        (ghash, zlib.compress(gjson.encode(), 6), count_vertices(g), valid,
                         json.dumps([round(e[0], 6), round(e[2], 6), round(e[1], 6), round(e[3], 6)])))
        row = cur.execute("SELECT desc_cycle FROM versions WHERE vhash=?", (vhash,)).fetchone()
        desc_s = json.dumps(desc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if row is None:
            cur.execute("INSERT INTO versions VALUES(?,?,?,?,?)", (vhash, ghash, core_s, desc_s, cyc.effective))
        elif cyc.effective >= row[0]:
            cur.execute("UPDATE versions SET desc=?, desc_cycle=? WHERE vhash=?", (desc_s, cyc.effective, vhash))
        members[vhash] = members.get(vhash, 0) + 1
    cur.execute("DELETE FROM members WHERE effective=?", (cyc.effective,))
    cur.executemany("INSERT INTO members VALUES(?,?,?)",
                    [(cyc.effective, v, n) for v, n in members.items()])
    cur.execute("INSERT OR REPLACE INTO cycles VALUES(?,?,?,?,?,?,?)",
                (cyc.effective, cyc.identity, n_feat, len(members), n_invalid,
                 n_nogeom, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    cur.execute("COMMIT")
    return n_feat, len(members), n_invalid, n_nogeom


def parse_all(region: Region, db: sqlite3.Connection, cycles: list[Cycle], reparse: bool,
              limit: int | None = None) -> int:
    """Parse what is missing; returns how many cycles are still unparsed."""
    done = {r[0]: r[1] for r in db.execute("SELECT effective, zip_id FROM cycles")}
    todo = [c for c in cycles if reparse or done.get(c.effective) != c.identity]
    log(f"parse {region.code}: {len(cycles)} cycles held, {len(todo)} to parse"
        + (f", doing {min(limit, len(todo))} this run" if limit else ""))
    for cyc in todo[:limit] if limit else todo:
        parse_cycle(region, db, cyc)
    return max(0, len(todo) - (limit or len(todo)))


# ---------------------------------------------------------------- floor edges (us)

def e_kind(core: dict) -> str | None:
    """'700' for the magenta-vignette floors, '1200' for the blue ones (1,200 ft
    AGL and every MSL floor), None for Class E types that are lines, not floors."""
    if core.get("lt") not in ("E5", "E6", "E7"):
        return None
    if core.get("loc") == "SFC" and (core.get("lo") or 0) <= 700:
        return "700"
    return "1200"


def bbox_overlaps(a: list, b: list) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def line_parts(g: ogr.Geometry) -> list[ogr.Geometry]:
    """The LineString members of any geometry (collections flattened, points dropped)."""
    name = g.GetGeometryName()
    if name == "LINESTRING":
        return [g] if g.GetPointCount() >= 2 else []
    if name in ("MULTILINESTRING", "GEOMETRYCOLLECTION"):
        out = []
        for i in range(g.GetGeometryCount()):
            out.extend(line_parts(g.GetGeometryRef(i)))
        return out
    return []


def oriented(piece: ogr.Geometry, poly: ogr.Geometry) -> ogr.Geometry:
    """Return the piece walking with the polygon interior on its geographic left
    (reversed when needed). Tested at the middle segment, 30 m to each side."""
    n = piece.GetPointCount()
    i = max(1, n // 2)
    x0, y0 = piece.GetPoint_2D(i - 1)
    x1, y1 = piece.GetPoint_2D(i)
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    eps = 3e-4
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    left = ogr.Geometry(ogr.wkbPoint)
    left.AddPoint_2D(mx - dy / length * eps, my + dx / length * eps)
    right = ogr.Geometry(ogr.wkbPoint)
    right.AddPoint_2D(mx + dy / length * eps, my - dx / length * eps)
    left_in, right_in = poly.Contains(left), poly.Contains(right)
    if right_in and not left_in:
        rev = ogr.Geometry(ogr.wkbLineString)
        for k in range(n - 1, -1, -1):
            rev.AddPoint_2D(*piece.GetPoint_2D(k))
        return rev
    return piece


def e_floor_edges(poly: ogr.Geometry, neighbours: list[ogr.Geometry]) -> list[ogr.Geometry]:
    """Boundary of `poly` minus the parts within ~2 m of any neighbour boundary
    (neighbour boundaries are clipped to poly's envelope first: a state-wide
    blanket is a neighbour of hundreds of areas and buffering all of it each
    time was the whole cost). Returns oriented LineStrings."""
    bnd = poly.Boundary()
    if neighbours:
        e = poly.GetEnvelope()
        pad = 1e-3
        clip = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in ((e[0] - pad, e[2] - pad), (e[1] + pad, e[2] - pad), (e[1] + pad, e[3] + pad),
                     (e[0] - pad, e[3] + pad), (e[0] - pad, e[2] - pad)):
            clip.AddPoint_2D(x, y)
        clip_poly = ogr.Geometry(ogr.wkbPolygon)
        clip_poly.AddGeometry(clip)
        union = ogr.Geometry(ogr.wkbMultiLineString)
        for nb in neighbours:
            for part in line_parts(nb.Boundary().Intersection(clip_poly)):
                union.AddGeometry(part)
        if union.GetGeometryCount():
            bnd = bnd.Difference(union.Buffer(2e-5, 2))
    return [oriented(part, poly) for part in line_parts(bnd)]


# ---------------------------------------------------------------- merge

def intervals(db: sqlite3.Connection, cycles: list[str]) -> dict[str, list[tuple[str, str | None]]]:
    """vhash -> [(from_effective, to_effective|None), ...] over the given cycles."""
    idx = {c: i for i, c in enumerate(cycles)}
    per: dict[str, list[int]] = {}
    q = "SELECT vhash, effective FROM members WHERE effective IN (%s)" % ",".join("?" * len(cycles))
    for vhash, eff in db.execute(q, cycles):
        per.setdefault(vhash, []).append(idx[eff])
    out: dict[str, list[tuple[str, str | None]]] = {}
    for vhash, ids in per.items():
        ids.sort()
        runs: list[tuple[str, str | None]] = []
        start = prev = ids[0]
        for i in ids[1:]:
            if i != prev + 1:
                runs.append((cycles[start], cycles[prev + 1]))
                start = i
            prev = i
        runs.append((cycles[start], cycles[prev + 1] if prev + 1 < len(cycles) else None))
        out[vhash] = runs
    return out


def ymd_int(eff: str) -> int:
    return int(eff.replace("-", ""))


def new_stats() -> dict:
    return {"versions": 0, "features": 0, "by_type": {}, "vertices": 0,
            "invalid_geoms": 0, "multi_interval_versions": 0, "boxes": {},
            "edge_features": 0, "edge_km": {"700": 0.0, "1200": 0.0}, "edge_kept": {}}


def write_region(region: Region, db: sqlite3.Connection, cycles: list[str], fh) -> dict:
    """Append the region's interval features (and, for the US, its floor
    edges) to the GeoJSONSeq; returns the region's stats."""
    ivals = intervals(db, cycles)
    stats = new_stats()
    stats["versions"] = len(ivals)
    boxes: dict[str, list] = {}
    if region.code == "us":
        write_floor_edges(db, cycles, ivals, fh, stats)
    for vhash, runs in ivals.items():
        ghash, core_s, desc_s = db.execute(
            "SELECT ghash, core, desc FROM versions WHERE vhash=?", (vhash,)).fetchone()
        gz, nvert, valid, bbox_s = db.execute(
            "SELECT gz, nvert, valid, bbox FROM geoms WHERE ghash=?", (ghash,)).fetchone()
        geom = zlib.decompress(gz).decode()
        core = json.loads(core_s)
        desc = json.loads(desc_s)
        bbox = json.loads(bbox_s)
        grp = region.box_group(desc, bbox)
        b = boxes.get(grp)
        boxes[grp] = bbox[:] if b is None else [min(b[0], bbox[0]), min(b[1], bbox[1]),
                                                max(b[2], bbox[2]), max(b[3], bbox[3])]
        props = {"rg": region.code}
        props.update(core)
        props.update(desc)
        props["v"] = vhash[:8]
        minzoom = region.minzoom(core)
        if len(runs) > 1:
            stats["multi_interval_versions"] += 1
        stats["vertices"] += nvert
        stats["invalid_geoms"] += 0 if valid else 1
        tkey = region.type_key(core)
        for frm, to in runs:
            p = dict(props)
            p["from"] = ymd_int(frm)
            if to:
                p["to"] = ymd_int(to)
            feat = {"type": "Feature",
                    "tippecanoe": {"layer": LAYER, "minzoom": minzoom},
                    "properties": p}
            # geometry last, verbatim (already rounded at parse time)
            fh.write(json.dumps(feat, separators=(",", ":"), ensure_ascii=False)[:-1]
                     + ',"geometry":' + geom + "}\n")
            stats["features"] += 1
            stats["by_type"][tkey] = stats["by_type"].get(tkey, 0) + 1
    pad = 0.25
    stats["boxes"] = {k: [round(b[0] - pad, 3), round(b[1] - pad, 3), round(b[2] + pad, 3), round(b[3] + pad, 3)]
                      for k, b in sorted(boxes.items())}
    return stats


def write_floor_edges(db: sqlite3.Connection, cycles: list[str], ivals: dict, fh, stats: dict) -> None:
    """The ``efloor`` layer: one oriented (Multi)LineString per Class E floor
    version and interval, computed against the Class E polygons of the cycle
    the version first appeared in (a neighbour that changes later changes this
    polygon too, since the FAA models them as complements)."""
    core_of = {}
    for vhash, core_s in db.execute("SELECT vhash, core FROM versions"):
        core = json.loads(core_s)
        if (core.get("lt") or "").startswith("E"):
            core_of[vhash] = core
    by_first: dict[str, list[str]] = {}
    for vhash, runs in ivals.items():
        if vhash in core_of and e_kind(core_of[vhash]):
            by_first.setdefault(runs[0][0], []).append(vhash)
    t0 = time.time()
    kept_len = {"700": [0.0, 0.0], "1200": [0.0, 0.0]}
    for eff in cycles:
        targets = by_first.get(eff)
        if not targets:
            continue
        rows = db.execute(
            "SELECT v.vhash, v.ghash, g.bbox FROM members m JOIN versions v ON v.vhash = m.vhash "
            "JOIN geoms g ON g.ghash = v.ghash WHERE m.effective = ?", (eff,)).fetchall()
        members = [(v, gh, json.loads(bb), e_kind(core_of[v])) for v, gh, bb in rows if v in core_of]
        geom_cache: dict[str, ogr.Geometry] = {}

        def geom(ghash: str) -> ogr.Geometry:
            g = geom_cache.get(ghash)
            if g is None:
                gz = db.execute("SELECT gz FROM geoms WHERE ghash=?", (ghash,)).fetchone()[0]
                g = geom_cache[ghash] = ogr.CreateGeometryFromJson(zlib.decompress(gz).decode())
            return g

        by_vhash = {v: (gh, bb, k) for v, gh, bb, k in members}
        for vhash in targets:
            ghash, bbox, kind = by_vhash[vhash]
            poly = geom(ghash)
            # 700 ft areas dissolve only against each other; higher floors
            # against every Class E polygon, so a blanket's edge survives only
            # where nothing controlled lies beyond it.
            nbs = [geom(gh2) for v2, gh2, bb2, k2 in members
                   if v2 != vhash and bbox_overlaps(bbox, bb2) and (kind == "1200" or k2 == "700")]
            pieces = e_floor_edges(poly, nbs)
            kept_len[kind][0] += poly.Boundary().Length()
            kept_len[kind][1] += sum(p.Length() for p in pieces)
            if not pieces:
                continue
            multi = ogr.Geometry(ogr.wkbMultiLineString)
            for p in pieces:
                multi.AddGeometry(p)
            gjson = multi.ExportToJson(["COORDINATE_PRECISION=6"])
            core = core_of[vhash]
            props = {"rg": "us", "k": kind, "lt": core.get("lt"), "v": vhash[:8]}
            if core.get("lo") is not None:
                props["lo"] = core["lo"]
                props["loc"] = core.get("loc")
            for frm, to in ivals[vhash]:
                p = dict(props, **{"from": ymd_int(frm)})
                if to:
                    p["to"] = ymd_int(to)
                feat = {"type": "Feature", "tippecanoe": {"layer": EDGE_LAYER, "minzoom": US_FEATURE_MINZOOM["E5"]},
                        "properties": p}
                fh.write(json.dumps(feat, separators=(",", ":"))[:-1] + ',"geometry":' + gjson + "}\n")
                stats["edge_features"] += 1
                stats["edge_km"][kind] += multi.Length() * 111.0
    for kind, (total, kept) in kept_len.items():
        stats["edge_kept"][kind] = round(kept / total, 3) if total else None
    log(f"edges: {stats['edge_features']} efloor features in {time.time() - t0:.0f}s; "
        f"boundary kept 700: {stats['edge_kept'].get('700')}, 1200+: {stats['edge_kept'].get('1200')}")


# ---------------------------------------------------------------- tile / stamp / verify

def tippecanoe(src: Path, dst: Path, regions: list[Region], dry: bool) -> None:
    names = ", ".join(f"{r.name} ({r.source})" for r in regions)
    cmd = ["tippecanoe", "-o", dst, "--force",
           "-Z", MINZOOM, "-z", MAXZOOM,
           "-P",                              # parallel read (needs line-delimited input)
           "--detect-shared-borders",
           "--buffer=32",
           "--no-feature-limit", "--no-tile-size-limit",
           "--no-tiny-polygon-reduction",    # never stand in a placeholder square for an area
           "-n", f"archive.aero airspace, class airspace: {names}",
           "-N", "Class airspace from every held AIS cycle of each region, one feature per "
                 "polygon version with a from/to validity interval (scripts/airspace_build.py)",
           "-A", "; ".join(f"{r.source} ({r.licence})" for r in regions),
           src]
    run(cmd, dry)


def stamp_metadata(dst: Path, info: dict, dry: bool) -> None:
    """Add an ``archive_aero`` object to the archive's JSON metadata in place."""
    if dry:
        log(f"+ pmtiles edit {dst} --metadata=<archive_aero {list(info)}>")
        return
    shown = subprocess.run(["pmtiles", "show", "--metadata", str(dst)],
                           check=True, capture_output=True, text=True).stdout
    meta = json.loads(shown)
    meta["archive_aero"] = info
    tmp = dst.with_suffix(".metadata.json")
    tmp.write_text(json.dumps(meta, ensure_ascii=False))
    run(["pmtiles", "edit", dst, f"--metadata={tmp}"])
    tmp.unlink()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_html(path: Path, url: str) -> None:
    from viewer_config import update_viewer_config
    target = update_viewer_config(path, "airspaceUrl", url)
    log(f"updated airspaceUrl in {target}")


# ---------------------------------------------------------------- report

def report(region: Region, db: sqlite3.Connection, cycles: list[str]) -> None:
    prev: set[str] = set()
    print(f"\n{region.code}  {region.name}, {region.source}")
    print(f"{'cycle':<12}{'rows':>6}{'versions':>10}{'added':>7}{'gone':>6}")
    for eff in cycles:
        row = db.execute("SELECT n_features, n_versions FROM cycles WHERE effective=?", (eff,)).fetchone()
        if not row:
            print(f"{eff:<12}  (not parsed)")
            continue
        cur = {r[0] for r in db.execute("SELECT vhash FROM members WHERE effective=?", (eff,))}
        print(f"{eff:<12}{row[0]:>6}{row[1]:>10}{len(cur - prev):>7}{len(prev - cur):>6}")
        prev = cur
    n_geoms, n_versions = db.execute("SELECT (SELECT COUNT(*) FROM geoms), (SELECT COUNT(*) FROM versions)").fetchone()
    print(f"{n_versions} distinct versions over {n_geoms} distinct shapes")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--regions", default=",".join(REGIONS), metavar="us,fr,br",
                    help="regions to build (default: all)")
    ap.add_argument("--since", metavar="YYYY-MM-DD", help="only cycles effective on/after this date")
    ap.add_argument("--stamp", help="output stamp (default: today as YYYYMMDD); must be new for --upload")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--parse-only", action="store_true", help="fill/refresh the caches, build nothing")
    ap.add_argument("--reparse", action="store_true", help="re-parse cycles already in the cache")
    ap.add_argument("--limit", type=int, metavar="N", help="parse at most N cycles per region this run")
    ap.add_argument("--report", action="store_true", help="print per-cycle churn from the caches and exit")
    ap.add_argument("--upload", action="store_true", help="rclone copyto the result into R2")
    ap.add_argument("--publish-only", action="store_true",
                    help="skip parse/merge/tile: upload the already built <stamp> archive (after a local check)")
    ap.add_argument("--update-html", metavar="PATH", help="rewrite airspaceUrl via index.html or src/viewer.js and rebuild frontend modules")
    ap.add_argument("--keep-geojson", action="store_true", help="keep the merged GeoJSONSeq beside the output")
    ap.add_argument("--dry-run", action="store_true", help="print the tippecanoe/upload commands, run nothing")
    args = ap.parse_args()

    codes = [c.strip() for c in args.regions.split(",") if c.strip()]
    unknown = [c for c in codes if c not in REGIONS]
    if unknown:
        sys.exit(f"unknown region(s) {unknown}; know {list(REGIONS)}")
    regions = [REGIONS[c] for c in codes]
    if not AIS_ROOT.exists():
        sys.exit(f"{AIS_ROOT} not found (is /Volumes/projects mounted?)")
    for r in regions:
        if not r.root.exists():
            sys.exit(f"{r.code}: {r.root} not found (run aisdata_pull.py --only {r.code})")
    for tool in ("tippecanoe", "pmtiles"):
        if not args.parse_only and not args.report and not shutil.which(tool):
            sys.exit(f"{tool} not on PATH (brew install {tool})")

    if args.publish_only:
        return publish(args)

    selected: dict[str, list[Cycle]] = {}
    for r in regions:
        cycles = r.held_cycles()
        if args.since:
            cycles = [c for c in cycles if c.effective >= args.since]
        if not cycles:
            sys.exit(f"{r.code}: no cycles selected")
        selected[r.code] = cycles
    dbs = {r.code: open_db(r) for r in regions}

    if args.report:
        for r in regions:
            report(r, dbs[r.code], [c.effective for c in selected[r.code]])
        return 0

    remaining = 0
    for r in regions:
        remaining += parse_all(r, dbs[r.code], selected[r.code], args.reparse, args.limit)
    if args.parse_only:
        if remaining:
            log(f"{remaining} cycles still unparsed (rerun)")
        return 0
    if remaining:
        sys.exit(f"{remaining} cycles still unparsed; rerun without --limit (or with a larger one) before building")

    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{SERIES}-{stamp}.pmtiles"
    seq = args.out_dir / f"{SERIES}-{stamp}.geojsonseq"

    t0 = time.time()
    per_region: dict[str, dict] = {}
    with seq.open("w") as fh:
        for r in regions:
            cycles = [c.effective for c in selected[r.code]]
            t1 = time.time()
            st = write_region(r, dbs[r.code], cycles, fh)
            per_region[r.code] = st
            log(f"merge {r.code}: {st['versions']} versions -> {st['features']} interval features "
                f"({st['vertices'] / 1e6:.1f} M vertices) in {time.time() - t1:.0f}s")
            log(f"       by type: {json.dumps(st['by_type'], sort_keys=True)}")
            log(f"       boxes: {json.dumps(st['boxes'])}")
            if st["invalid_geoms"]:
                log(f"       {st['invalid_geoms']} shapes fail GEOS IsValid (tippecanoe cleans rings; check the manifest)")
    total_feat = sum(s["features"] for s in per_region.values()) + sum(s["edge_features"] for s in per_region.values())
    total_vers = sum(s["versions"] for s in per_region.values())
    log(f"merge: {total_vers} versions -> {total_feat} features ({seq.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")

    tippecanoe(seq, out, regions, args.dry_run)
    if args.dry_run:
        return 0

    info = {
        "series": SERIES,
        "layer": LAYER,
        "edge_layer": EDGE_LAYER,
        "cycle_days": CYCLE_DAYS,
        "regions": {},
        "features": total_feat,
        "versions": total_vers,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "scripts/airspace_build.py",
    }
    for r in regions:
        cycles = [c.effective for c in selected[r.code]]
        st = per_region[r.code]
        entry = {
            "name": r.name,
            "source": r.source,
            "source_url": r.source_url,
            "licence": r.licence,
            "cycles": cycles,
            "first": cycles[0],
            "last": cycles[-1],
            "boxes": list(st["boxes"].values()),
            "features": st["features"],
            "versions": st["versions"],
        }
        if r.note:
            entry["note"] = r.note
        info["regions"][r.code] = entry
    stamp_metadata(out, info, args.dry_run)
    run(["pmtiles", "verify", out])
    show = subprocess.run(["pmtiles", "show", str(out)], capture_output=True, text=True).stdout
    log(show)

    size = out.stat().st_size
    key = f"{R2_PREFIX}/{out.name}"
    manifest = dict(info, stamp=stamp, r2_key=key, output=str(out), output_size=size,
                    sha256=sha256_file(out),
                    stats={code: {k: v for k, v in st.items() if k != "boxes"} for code, st in per_region.items()},
                    tippecanoe=subprocess.run(["tippecanoe", "--version"], capture_output=True,
                                              text=True).stderr.strip() or None)
    (args.out_dir / f"{SERIES}-{stamp}.manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    with (CACHE_DIR / "builds.jsonl").open("a") as fh:
        fh.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    if not args.keep_geojson:
        seq.unlink(missing_ok=True)
    log(f"\n{out}  {size / 1e6:.1f} MB")

    return publish_archive(out, manifest, args.upload, args.update_html)


def publish_archive(out: Path, manifest: dict, upload: bool, update: str | None) -> int:
    """rclone the built archive to its immutable key (when asked), record the
    upload, and point index.html at it (when asked)."""
    key = manifest["r2_key"]
    cmd = ["rclone", "copyto", str(out), f"{R2_REMOTE}/{key}",
           "--s3-upload-concurrency=8", "--s3-chunk-size=64M", "--stats-one-line", "--stats", "30s", "-v"]
    url = f"https://data.archive.aero/{key}"
    if upload:
        # rclone lsjson prints "[\n]" for a missing object: parse, don't compare.
        probe = subprocess.run(["rclone", "lsjson", f"{R2_REMOTE}/{key}"], capture_output=True, text=True)
        try:
            exists = probe.returncode == 0 and bool(json.loads(probe.stdout or "[]"))
        except ValueError:
            exists = False
        if exists:
            raise SystemExit(f"{R2_REMOTE}/{key} already exists in R2 - the dated airspace key is immutable. "
                             f"Rebuild with a new --stamp instead.")
        if sha256_file(out) != manifest["sha256"]:
            raise SystemExit(f"{out} does not match its manifest sha256 - rebuild before publishing")
        run(cmd)
        with (CACHE_DIR / "uploads.jsonl").open("a") as fh:
            fh.write(json.dumps({"stamp": manifest["stamp"], "r2_key": key, "sha256": manifest["sha256"],
                                 "output_size": manifest["output_size"],
                                 "uploaded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "regions": {c: {"first": e["first"], "last": e["last"], "cycles": len(e["cycles"])}
                                             for c, e in manifest["regions"].items()}}) + "\n")
        log(f"\npublished {url}")
    else:
        log("\nto publish:\n  " + " ".join(cmd))
    if update:
        update_html(Path(update), url)
    else:
        log(f"then point CONFIG.airspaceUrl in index.html at {url} (or rerun with --update-html index.html)")
    return 0


def publish(args) -> int:
    """--publish-only: the archive for --stamp (default: today) was built and
    checked in the viewer already; upload it as it stands."""
    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    out = args.out_dir / f"{SERIES}-{stamp}.pmtiles"
    mpath = args.out_dir / f"{SERIES}-{stamp}.manifest.json"
    if not out.exists() or not mpath.exists():
        sys.exit(f"{out} (and its manifest) not found - build it first")
    manifest = json.loads(mpath.read_text())
    return publish_archive(out, manifest, True, args.update_html)


if __name__ == "__main__":
    sys.exit(main())
