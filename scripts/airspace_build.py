#!/usr/bin/env python3
"""Build archive.aero's airspace overlay from FAA NASR and publish it to R2.

What it makes: one vector PMTiles archive (tile layer ``class``) holding every
Class B/C/D/E polygon from every held NASR 28-day cycle, merged so that a
polygon version that is unchanged across consecutive cycles is stored once with
a validity interval::

    from  effective date of the first cycle carrying this version (YYYYMMDD int)
    to    effective date of the first later cycle without it (absent while the
          version is still present in the newest held cycle)

The viewer (index.html, AirspaceLayer) draws the versions whose interval
contains the timeline date, so scrubbing changes which airspace is drawn
without refetching a tile.

Source: ``Additional_Data/Shape_Files/Class_Airspace.shp`` inside each
``/Volumes/projects/aisdata/us_faa/nasr/<effective>/28DaySubscription_*.zip``
(pulled by aisdata_pull.py; every cycle from 2020-03-26 ships it). The
shapefile carries no stable feature id (GLOBAL_ID exists only in the ADDS
GeoJSON), so identity is content: a *version* is the hash of the polygon
(coordinates rounded to 1e-6 deg, NAD83 treated as WGS84) plus the attributes
that decide how it is drawn or described -- class, local type, floor/ceiling,
hours, sector, exclusion flag. Name and identifier are NOT hashed: the FAA
re-spelled 1,329 idents (KSTL -> STL) between 2020 and 2021 without touching a
boundary, and hashing them would have split every one of those versions in
two. They are taken from the newest cycle that carries the version.

Stages (each resumable, each skipped when its output is current):

  parse   every unparsed cycle -> worklists/data/airspace/class_versions.sqlite
          (geometry stored once per distinct shape as zlib'd GeoJSON)
  merge   membership runs -> intervals -> newline-delimited GeoJSON carrying a
          per-feature tippecanoe minzoom: B/C/D/E2-E4 from z5, E5-E7 from z6
          (the map starts at z6 and protomaps-leaflet reads data one level
          below the map zoom)
  edges   the vignette lines: for every Class E floor polygon (E5/E6/E7) the
          part of its boundary that is a real floor change -- a 700 ft area's
          edge minus edges shared with other 700 ft areas, a 1,200 ft-or-higher
          area's edge minus edges shared with *any* other Class E polygon (the
          state-wide 1,200 ft blankets are cut around every 700 ft area, and
          sectionals draw the blue vignette only where Class E meets Class G).
          Emitted as tile layer ``efloor``, each line oriented with the
          controlled side on its left, so the viewer can shade one side.
  tile    tippecanoe -Z5 -z11, shared borders, 32/256 buffer (the viewer
          strokes a vignette up to ~20 px inside the edges; the buffer keeps
          the stroke along tile-cut edges outside the visible tile), and no
          feature dropping of any kind -- a missing Class D is a data error
  stamp   pmtiles edit: the archive's JSON metadata gains ``archive_aero``
          (cycle list + build info) so the viewer can name the cycle in effect
  verify  pmtiles verify, a manifest beside the output, a line in builds.jsonl
  upload  rclone copyto -> r2:charts/airspace/nasr-<stamp>.pmtiles (--upload)

The published key is dated and immutable (same rule as basemap_build.py):
replacing a PMTiles under a live key moves every byte offset in it while the
tiles Worker's range cache carries no version. A rebuild for the same last
cycle needs a fresh --stamp (e.g. 20261001b); the upload refuses to overwrite.
--update-html rewrites CONFIG.airspaceUrl in index.html the way
build_metadata_bundle.py rewrites bundleUrl.

Usage
-----
    ~/venv/bin/python scripts/airspace_build.py --parse-only        # fill the cache
    ~/venv/bin/python scripts/airspace_build.py                     # build, all cycles
    ~/venv/bin/python scripts/airspace_build.py --since 2026-01-01 --stamp test1
    ~/venv/bin/python scripts/airspace_build.py --report            # churn per cycle
    ~/venv/bin/python scripts/airspace_build.py --upload --update-html index.html
"""

from __future__ import annotations

import argparse
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
from datetime import datetime, timezone
from pathlib import Path

from osgeo import gdal, ogr

gdal.UseExceptions()

REPO = Path(__file__).resolve().parent.parent
NASR_ROOT = Path("/Volumes/projects/aisdata/us_faa/nasr")
NASR_MANIFEST = Path("/Volumes/projects/aisdata/us_faa/manifest.jsonl")
CACHE_DIR = REPO / "worklists" / "data" / "airspace"      # gitignored (worklists/data/)
DB_PATH = CACHE_DIR / "class_versions.sqlite"
OUT_DIR = Path("/Volumes/projects/airspace_pmtiles")     # local mirror, keeps .pmtiles
R2_REMOTE = "r2:charts"
R2_PREFIX = "airspace"
SHP_MEMBER = "Additional_Data/Shape_Files/Class_Airspace.shp"
LAYER = "class"
EDGE_LAYER = "efloor"     # oriented Class E floor edges, the vignette lines (see e_floor_edges)
MINZOOM, MAXZOOM = 5, 11
# Data zoom at which a local type first appears in the tiles. The viewer maps
# display zoom z to data zoom z-1, so 5 = visible from the map's minimum zoom
# (6), 6 = from z7, where a vignette is wide enough to read.
FEATURE_MINZOOM = {"B": 5, "C": 5, "D": 5, "E2": 5, "E3": 5, "E4": 5,
                   "E5": 6, "E6": 6, "E7": 6}
NULL_ALT = {"", "-9998", "-9999"}


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], dry: bool = False, **kw) -> subprocess.CompletedProcess | None:
    log("+ " + " ".join(str(c) for c in cmd))
    if dry:
        return None
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


# ---------------------------------------------------------------- cycles

def held_cycles() -> list[str]:
    """Effective dates (YYYY-MM-DD) of every cycle whose zip is on disk, ascending."""
    out = []
    for d in sorted(NASR_ROOT.iterdir()):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name) and cycle_zip(d.name).exists():
            out.append(d.name)
    return out


def cycle_zip(eff: str) -> Path:
    return NASR_ROOT / eff / f"28DaySubscription_Effective_{eff}.zip"


def manifest_shas() -> dict[str, str]:
    """zip relative path -> sha256, from aisdata_pull.py's manifest (newest line wins)."""
    shas: dict[str, str] = {}
    if NASR_MANIFEST.exists():
        for line in NASR_MANIFEST.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("sha256") and row.get("file"):
                shas[row["file"]] = row["sha256"]
    return shas


def zip_identity(eff: str, shas: dict[str, str]) -> str:
    """What 'this cycle has been parsed' is keyed on: the manifest sha256 when
    the pull recorded one, else size+mtime (a re-download shows up as a change)."""
    rel = f"nasr/{eff}/{cycle_zip(eff).name}"
    if rel in shas:
        return shas[rel]
    st = cycle_zip(eff).stat()
    return f"size{st.st_size}-mtime{int(st.st_mtime)}"


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
    return {k: v for k, v in core.items() if v is not None}, {k: v for k, v in desc.items() if v is not None}


def count_vertices(g: ogr.Geometry) -> int:
    n = g.GetGeometryCount()
    if n == 0:
        return g.GetPointCount()
    return sum(count_vertices(g.GetGeometryRef(i)) for i in range(n))


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


def open_db() -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # WAL + a long busy timeout: a --report (or a second build) reading the
    # cache while a parse commits must wait, not fail either side.
    db = sqlite3.connect(DB_PATH, timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def parse_cycle(db: sqlite3.Connection, eff: str, zip_id: str) -> None:
    # The shapefile is unpacked to a scratch dir and read from disk: reading it
    # through GDAL's /vsizip/ layer leaked ~2 GB per cycle (62 GB peak over a
    # run) and the OS killed the parse twice before this was found.
    with tempfile.TemporaryDirectory(prefix="airspace_") as tmp:
        with zipfile.ZipFile(cycle_zip(eff)) as zf:
            stem = SHP_MEMBER[:-4]
            for member in zf.namelist():
                if member.startswith(stem):
                    Path(tmp, Path(member).name).write_bytes(zf.read(member))
        _parse_shapefile(db, eff, zip_id, str(Path(tmp, Path(SHP_MEMBER).name)))


def _parse_shapefile(db: sqlite3.Connection, eff: str, zip_id: str, path: str) -> None:
    t0 = time.time()
    ds = ogr.Open(path)
    if ds is None:
        raise SystemExit(f"{eff}: cannot open {path}")
    lyr = ds.GetLayer(0)
    members: dict[str, int] = {}
    n_feat = n_invalid = n_nogeom = 0
    cur = db.cursor()
    cur.execute("BEGIN")
    # A version's description follows the newest cycle carrying it, so a
    # re-parse of an old cycle must not clobber a newer spelling.
    for f in lyr:
        n_feat += 1
        g = f.GetGeometryRef()
        if g is None or g.IsEmpty():
            n_nogeom += 1
            continue
        if g.Is3D() or g.IsMeasured():
            g.FlattenTo2D()   # some rows are PolygonZ with z=0; the third ordinate is noise
        gjson = g.ExportToJson(["COORDINATE_PRECISION=6"])
        ghash = hashlib.sha1(gjson.encode()).hexdigest()
        core, desc = normalize(f)
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
        desc_s = json.dumps(desc, sort_keys=True, separators=(",", ":"))
        if row is None:
            cur.execute("INSERT INTO versions VALUES(?,?,?,?,?)", (vhash, ghash, core_s, desc_s, eff))
        elif eff >= row[0]:
            cur.execute("UPDATE versions SET desc=?, desc_cycle=? WHERE vhash=?", (desc_s, eff, vhash))
        members[vhash] = members.get(vhash, 0) + 1
    cur.execute("DELETE FROM members WHERE effective=?", (eff,))
    cur.executemany("INSERT INTO members VALUES(?,?,?)",
                    [(eff, v, n) for v, n in members.items()])
    cur.execute("INSERT OR REPLACE INTO cycles VALUES(?,?,?,?,?,?,?)",
                (eff, zip_id, n_feat, len(members), n_invalid,
                 n_nogeom, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    cur.execute("COMMIT")
    lyr = None
    ds = None
    dup = n_feat - n_nogeom - len(members)
    log(f"  {eff}: {n_feat} rows -> {len(members)} versions"
        f"{f', {dup} exact dupes' if dup else ''}"
        f"{f', {n_invalid} new invalid geometries' if n_invalid else ''}"
        f"{f', {n_nogeom} without geometry' if n_nogeom else ''}  ({time.time() - t0:.0f}s, "
        f"rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB)")


def parse_all(db: sqlite3.Connection, cycles: list[str], reparse: bool, limit: int | None = None) -> int:
    """Parse what is missing; returns how many cycles are still unparsed."""
    shas = manifest_shas()
    done = {r[0]: r[1] for r in db.execute("SELECT effective, zip_id FROM cycles")}
    todo = [c for c in cycles if reparse or done.get(c) != zip_identity(c, shas)]
    log(f"parse: {len(cycles)} cycles held, {len(todo)} to parse"
        + (f", doing {min(limit, len(todo))} this run" if limit else ""))
    for eff in todo[:limit] if limit else todo:
        parse_cycle(db, eff, zip_identity(eff, shas))
    return max(0, len(todo) - (limit or len(todo)))


# ---------------------------------------------------------------- floor edges

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


def write_geojsonseq(db: sqlite3.Connection, cycles: list[str], out: Path) -> dict:
    ivals = intervals(db, cycles)
    stats = {"versions": len(ivals), "features": 0, "by_type": {}, "vertices": 0,
             "invalid_geoms": 0, "multi_interval_versions": 0,
             "edge_features": 0, "edge_km": {"700": 0.0, "1200": 0.0}, "edge_kept": {}}
    with out.open("w") as fh:
        write_floor_edges(db, cycles, ivals, fh, stats)
        for vhash, runs in ivals.items():
            ghash, core_s, desc_s = db.execute(
                "SELECT ghash, core, desc FROM versions WHERE vhash=?", (vhash,)).fetchone()
            gz, nvert, valid = db.execute(
                "SELECT gz, nvert, valid FROM geoms WHERE ghash=?", (ghash,)).fetchone()
            geom = zlib.decompress(gz).decode()
            core = json.loads(core_s)
            props = dict(core)
            props.update(json.loads(desc_s))
            props["v"] = vhash[:8]
            lt = core.get("lt") or ""
            minzoom = FEATURE_MINZOOM.get(lt, MINZOOM)
            if len(runs) > 1:
                stats["multi_interval_versions"] += 1
            stats["vertices"] += nvert
            stats["invalid_geoms"] += 0 if valid else 1
            for frm, to in runs:
                p = dict(props)
                p["from"] = ymd_int(frm)
                if to:
                    p["to"] = ymd_int(to)
                feat = {"type": "Feature",
                        "tippecanoe": {"layer": LAYER, "minzoom": minzoom},
                        "properties": p}
                # geometry last, verbatim (already rounded at parse time)
                fh.write(json.dumps(feat, separators=(",", ":"))[:-1] + ',"geometry":' + geom + "}\n")
                stats["features"] += 1
                stats["by_type"][lt] = stats["by_type"].get(lt, 0) + 1
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
            props = {"k": kind, "lt": core.get("lt"), "v": vhash[:8]}
            if core.get("lo") is not None:
                props["lo"] = core["lo"]
                props["loc"] = core.get("loc")
            for frm, to in ivals[vhash]:
                p = dict(props, **{"from": ymd_int(frm)})
                if to:
                    p["to"] = ymd_int(to)
                feat = {"type": "Feature", "tippecanoe": {"layer": EDGE_LAYER, "minzoom": FEATURE_MINZOOM["E5"]},
                        "properties": p}
                fh.write(json.dumps(feat, separators=(",", ":"))[:-1] + ',"geometry":' + gjson + "}\n")
                stats["edge_features"] += 1
                stats["edge_km"][kind] += multi.Length() * 111.0
    for kind, (total, kept) in kept_len.items():
        stats["edge_kept"][kind] = round(kept / total, 3) if total else None
    log(f"edges: {stats['edge_features']} efloor features in {time.time() - t0:.0f}s; "
        f"boundary kept 700: {stats['edge_kept'].get('700')}, 1200+: {stats['edge_kept'].get('1200')}")


# ---------------------------------------------------------------- tile / stamp / verify

def tippecanoe(src: Path, dst: Path, cycles: list[str], dry: bool) -> None:
    cmd = ["tippecanoe", "-o", dst, "--force",
           "-Z", MINZOOM, "-z", MAXZOOM,
           "-P",                              # parallel read (needs line-delimited input)
           "--detect-shared-borders",
           "--buffer=32",
           "--no-feature-limit", "--no-tile-size-limit",
           "--no-tiny-polygon-reduction",    # never stand in a placeholder square for an area
           "-n", f"archive.aero airspace, FAA NASR class airspace {cycles[0]}..{cycles[-1]}",
           "-N", "Class B/C/D/E airspace from every FAA NASR 28-day cycle held, "
                 "one feature per polygon version with a from/to validity interval "
                 "(scripts/airspace_build.py)",
           "-A", "FAA NASR (public domain)",
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
    tmp.write_text(json.dumps(meta))
    run(["pmtiles", "edit", dst, f"--metadata={tmp}"])
    tmp.unlink()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_html(path: Path, url: str) -> None:
    html = path.read_text()
    pattern = r"(airspaceUrl:\s*)(null|'[^']*')"
    if not re.search(pattern, html):
        sys.exit(f"{path}: no airspaceUrl config entry found - wire the client first")
    path.write_text(re.sub(pattern, rf"\g<1>'{url}'", html, count=1))
    log(f"updated airspaceUrl in {path}")


# ---------------------------------------------------------------- report

def report(db: sqlite3.Connection, cycles: list[str]) -> None:
    prev: set[str] = set()
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
    print(f"\n{n_versions} distinct versions over {n_geoms} distinct shapes")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", metavar="YYYY-MM-DD", help="only cycles effective on/after this date")
    ap.add_argument("--stamp", help="output stamp (default: last cycle as YYYYMMDD); must be new for --upload")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--parse-only", action="store_true", help="fill/refresh the cache, build nothing")
    ap.add_argument("--reparse", action="store_true", help="re-parse cycles already in the cache")
    ap.add_argument("--limit", type=int, metavar="N", help="parse at most N cycles this run (chunked cache fills)")
    ap.add_argument("--report", action="store_true", help="print per-cycle churn from the cache and exit")
    ap.add_argument("--upload", action="store_true", help="rclone copyto the result into R2")
    ap.add_argument("--update-html", metavar="PATH", help="rewrite airspaceUrl in the given index.html")
    ap.add_argument("--keep-geojson", action="store_true", help="keep the merged GeoJSONSeq beside the output")
    ap.add_argument("--dry-run", action="store_true", help="print the tippecanoe/upload commands, run nothing")
    args = ap.parse_args()

    if not NASR_ROOT.exists():
        sys.exit(f"{NASR_ROOT} not found (is /Volumes/projects mounted?)")
    for tool in ("tippecanoe", "pmtiles"):
        if not args.parse_only and not args.report and not shutil.which(tool):
            sys.exit(f"{tool} not on PATH (brew install {tool})")

    cycles = held_cycles()
    if args.since:
        cycles = [c for c in cycles if c >= args.since]
    if not cycles:
        sys.exit("no cycles selected")
    db = open_db()

    if args.report:
        report(db, cycles)
        return 0

    remaining = parse_all(db, cycles, args.reparse, args.limit)
    if args.parse_only:
        if remaining:
            log(f"{remaining} cycles still unparsed (rerun)")
        return 0
    if remaining:
        sys.exit(f"{remaining} cycles still unparsed; rerun without --limit (or with a larger one) before building")

    stamp = args.stamp or cycles[-1].replace("-", "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"nasr-{stamp}.pmtiles"
    seq = args.out_dir / f"nasr-{stamp}.geojsonseq"

    t0 = time.time()
    stats = write_geojsonseq(db, cycles, seq)
    log(f"merge: {stats['versions']} versions -> {stats['features']} interval features "
        f"({stats['vertices'] / 1e6:.1f} M vertices, {seq.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")
    log(f"       by type: {json.dumps(stats['by_type'], sort_keys=True)}")
    if stats["invalid_geoms"]:
        log(f"       {stats['invalid_geoms']} shapes fail GEOS IsValid (tippecanoe cleans rings; check the manifest)")

    tippecanoe(seq, out, cycles, args.dry_run)
    if args.dry_run:
        return 0

    info = {
        "series": "nasr-class",
        "layer": LAYER,
        "edge_layer": EDGE_LAYER,
        "cycles": cycles,
        "first": cycles[0],
        "last": cycles[-1],
        "features": stats["features"],
        "versions": stats["versions"],
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "scripts/airspace_build.py",
    }
    stamp_metadata(out, info, args.dry_run)
    run(["pmtiles", "verify", out])
    show = subprocess.run(["pmtiles", "show", str(out)], capture_output=True, text=True).stdout
    log(show)

    size = out.stat().st_size
    key = f"{R2_PREFIX}/{out.name}"
    manifest = dict(info, stamp=stamp, r2_key=key, output=str(out), output_size=size,
                    sha256=sha256_file(out), by_type=stats["by_type"],
                    multi_interval_versions=stats["multi_interval_versions"],
                    edge_features=stats["edge_features"], edge_km=stats["edge_km"], edge_kept=stats["edge_kept"],
                    invalid_geoms=stats["invalid_geoms"],
                    tippecanoe=subprocess.run(["tippecanoe", "--version"], capture_output=True,
                                              text=True).stderr.strip() or None)
    (args.out_dir / f"nasr-{stamp}.manifest.json").write_text(json.dumps(manifest, indent=2))
    with (CACHE_DIR / "builds.jsonl").open("a") as fh:
        fh.write(json.dumps(manifest) + "\n")
    if not args.keep_geojson:
        seq.unlink(missing_ok=True)
    log(f"\n{out}  {size / 1e6:.1f} MB")

    upload = ["rclone", "copyto", str(out), f"{R2_REMOTE}/{key}",
              "--s3-upload-concurrency=8", "--s3-chunk-size=64M", "--stats-one-line", "--stats", "30s", "-v"]
    url = f"https://data.archive.aero/{key}"
    if args.upload:
        probe = subprocess.run(["rclone", "lsjson", f"{R2_REMOTE}/{key}"], capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip() not in ("", "[]"):
            raise SystemExit(f"{R2_REMOTE}/{key} already exists in R2 - the dated airspace key is immutable. "
                             f"Rebuild with a new --stamp instead.")
        run(upload)
        log(f"\npublished {url}")
    else:
        log("\nto publish:\n  " + " ".join(upload))
    if args.update_html:
        update_html(Path(args.update_html), url)
    else:
        log(f"then point CONFIG.airspaceUrl in index.html at {url} (or rerun with --update-html index.html)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
