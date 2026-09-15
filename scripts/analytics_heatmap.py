#!/usr/bin/env python3
"""Turn the tiles Worker's byte-range log into a space/time heatmap.

Every logged read is a PMTiles range request, so its start offset identifies
exactly one tile once you have that archive's directories — and those already
live in the metadata bundles this repo builds. Decoding offset -> tile id ->
z/x/y recovers *where* on the map each view landed; the R2 key recovers *which
chart era* was on screen; the timestamp recovers *when* someone looked.

Two different "times" come out of this and must not be conflated:
  era   -- the historical chart date being viewed (archive time)
  clock -- when the visitor was viewing (wall time)

Offsets are only meaningful against the exact archive build that served them:
the 2026-07-28/29 republishes moved every tile, so a row logged before then
only decodes against a bundle generated before then. All local bundles are
tried (newest first, deduped by the era's file size), then a direct R2 range
read of the live archive's leaf directories for the ~35 multi-GB modern eras
whose directories are too big to inline. Resolutions are cached, so re-runs
after a fresh export only decode the new rows.

Inputs:  worklists/data/analytics/tile_logs/*.jsonl.gz  (analytics_export.py)
         metadata-*.bundle
Outputs (all under worklists/data/analytics/, gitignored):
         viewed_tiles.jsonl.gz   resolved fact table
         summary.json            era / clock / country / zoom aggregates
         heatmap_z<N>.geojson    density-normalised grid

Then run scripts/analytics_render.py to draw the map and the time charts.

Usage:
  ~/venv/bin/python scripts/analytics_heatmap.py
  ~/venv/bin/python scripts/analytics_heatmap.py --no-filter   # keep bots/self
"""

import argparse
import collections
import gzip
import json
import math
import re
import struct
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from pmtiles.tile import (
        deserialize_directory,
        deserialize_header,
        tileid_to_zxy,
    )
except ImportError as e:
    sys.exit(f"needs the pmtiles package (run with ~/venv/bin/python): {e}")

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "worklists/data/analytics/tile_logs"
OUT_DIR = REPO / "worklists/data/analytics"
CACHE_DIR = OUT_DIR / "cache"
BASE_URL = "https://data.archive.aero/sectionals/"
# data.archive.aero's WAF 403s the default Python-urllib agent.
UA = "archive.aero-analytics/1.0"


def worker_sample_rate():
    """The tiles Worker's SAMPLE_RATE from worker/wrangler.toml (the gate
    that decides which reads are logged at all; Analytics Engine's own
    _sample_interval is a separate, second sampling). Hard-coding 0.05 here
    would drift silently if the Worker's var changed."""
    toml = REPO / "worker" / "wrangler.toml"
    try:
        m = re.search(r'^\s*SAMPLE_RATE\s*=\s*"?([0-9.]+)"?', toml.read_text(), re.M)
        return float(m.group(1)) if m else 0.05
    except OSError:
        return 0.05


SAMPLE_RATE = worker_sample_rate()

# Traffic that is the site's own machinery rather than a visitor.
SELF_UA = {"aamb-builder"}
SELF_REF = {"http://localhost:3000/"}

# Only date-range keys are tile archives; the bucket also serves the metadata
# bundles, which are ordinary reads with no tile behind them.
ERA_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------- bundles

class Bundle:
    """A metadata-*.bundle: era index plus each era's PMTiles metadata prefix."""

    def __init__(self, path):
        self.path = path
        self.buf = path.read_bytes()
        if self.buf[:8] != b"AAMBv1\n\0":
            raise ValueError(f"{path.name}: not a metadata bundle")
        glen = struct.unpack_from("<I", self.buf, 8)[0]
        idx = json.loads(gzip.decompress(self.buf[16 : 16 + glen]))
        self.blobs = 16 + glen
        self.generated = idx.get("generated", "")
        self.eras = {e["k"]: e for e in idx["eras"]}

    def size_of(self, era):
        e = self.eras.get(era)
        return e["size"] if e else None

    def tile_data_offset(self, era):
        """Where tile bodies begin in this build of the era, or None."""
        e = self.eras.get(era)
        if not e:
            return None
        try:
            return deserialize_header(self.buf[self.blobs + e["off"] : self.blobs + e["off"] + 127])["tile_data_offset"]
        except Exception:
            return None

    def offset_table(self, era):
        """{absolute file offset: (tile_id, length)} or None if leaves absent."""
        e = self.eras.get(era)
        if not e:
            return None
        blob = self.buf[self.blobs + e["off"] : self.blobs + e["off"] + e["len"]]
        return build_table(blob, blob)


def build_table(header_bytes, dir_bytes, leaf_bytes=None):
    """Decode a PMTiles directory tree into {file offset: (tile_id, length)}.

    dir_bytes holds the root directory at the header's root_offset; leaf_bytes
    (when given) holds the leaf directory region on its own, as fetched by
    range read. Returns None when leaves are needed but unavailable.
    """
    head = deserialize_header(header_bytes[:127])
    ro, rl = head["root_offset"], head["root_length"]
    if len(dir_bytes) < ro + rl:
        return None
    root = deserialize_directory(dir_bytes[ro : ro + rl])
    lo, ll = head["leaf_directory_offset"], head["leaf_directory_length"]
    if leaf_bytes is None:
        if ll and len(dir_bytes) < lo + ll:
            return None
        leaf_bytes = dir_bytes[lo : lo + ll] if ll else b""
        leaf_base = 0
    else:
        leaf_base = 0

    table = {}
    for entry in root:
        if entry.run_length == 0:
            start = leaf_base + entry.offset
            chunk = leaf_bytes[start : start + entry.length]
            if len(chunk) < entry.length:
                return None
            for leaf in deserialize_directory(chunk):
                table.setdefault(
                    leaf.offset + head["tile_data_offset"], (leaf.tile_id, leaf.length)
                )
        else:
            table.setdefault(
                entry.offset + head["tile_data_offset"], (entry.tile_id, entry.length)
            )
    return table


def http_range(url, start, length):
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{start + length - 1}", "User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def http_size(url):
    """Content-Length of the live object (HEAD). Identifies the build: a
    republish under the same key changes the archive size."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def live_offset_table(era):
    """(offset table, archive size) for the archive currently in R2.

    Directories are cached per (era, size): an offset only decodes against
    the exact build that served it, and a republish moves every tile. The
    cache used to be keyed by era alone, so after the 2026-08/09 republishes
    nine eras were decoded against stale directories and every view on them
    was memoised as unresolvable.
    """
    url = f"{BASE_URL}{era}.pmtiles"
    try:
        size = http_size(url)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        print(f"  ! {era}: live HEAD failed ({e})", file=sys.stderr)
        return None, None
    cache = CACHE_DIR / "livedirs" / f"{era}.{size}.bin"
    if cache.exists():
        blob = cache.read_bytes()
    else:
        try:
            head_chunk = http_range(url, 0, 16384)
            head = deserialize_header(head_chunk[:127])
            lo, ll = head["leaf_directory_offset"], head["leaf_directory_length"]
            # Header + root + (usually) metadata sit in the first 16 KB; leaves
            # are a separate multi-MB region in the big modern archives.
            need = max(16384, head["root_offset"] + head["root_length"])
            prefix = head_chunk if need <= len(head_chunk) else http_range(url, 0, need)
            leaves = http_range(url, lo, ll) if ll and lo + ll > len(prefix) else b""
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print(f"  ! {era}: live fetch failed ({e})", file=sys.stderr)
            return None, size
        blob = struct.pack("<II", len(prefix), len(leaves)) + prefix + leaves
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(blob)

    plen, llen = struct.unpack_from("<II", blob, 0)
    prefix = blob[8 : 8 + plen]
    leaves = blob[8 + plen : 8 + plen + llen]
    return build_table(prefix, prefix, leaves if llen else None), size


# ---------------------------------------------------------------- resolving

def resolve_offsets(wanted, bundles, use_live=True):
    """{era: {(off, len)}} -> {(era, off, len): tile_id}, cached on disk.

    Each era is tried against every distinct archive build available: bundles
    newest-first (skipping any whose recorded file size was already tried, since
    equal size means the same layout), then the live R2 archive.
    """
    cache_path = CACHE_DIR / "offset_index.json.gz"
    resolved = {}
    # A miss is remembered together with the archive sizes (builds) it was
    # tried against, so a rerun retries it against any build it has not seen
    # (a fresh bundle, or a republished live archive). A miss memoised as a
    # bare None used to be permanent.
    misses = {}
    if cache_path.exists():
        with gzip.open(cache_path, "rt") as fh:
            for k, v in json.load(fh).items():
                era, off, ln = k.rsplit("\t", 2)
                key = (era, int(off), int(ln))
                if isinstance(v, dict):
                    misses[key] = set(v.get("tried", []))
                elif v is None:
                    misses[key] = set()  # legacy permanent miss: retry everywhere
                else:
                    resolved[key] = v

    todo = {
        era: {p for p in pairs if (era, p[0], p[1]) not in resolved}
        for era, pairs in wanted.items()
    }
    todo = {e: p for e, p in todo.items() if p}
    if not todo:
        print(f"all {len(wanted)} eras already resolved from cache")
        return resolved

    print(f"resolving {sum(len(p) for p in todo.values())} new offsets "
          f"across {len(todo)} eras")
    # Sizes already tried per PAIR (from the miss records) and per era (this run).
    tried_sizes = collections.defaultdict(set)
    for bi, bundle in enumerate(bundles):
        remaining = [e for e in todo if todo[e]]
        if not remaining:
            break
        hits = 0
        for era in remaining:
            size = bundle.size_of(era)
            if size is None or size in tried_sizes[era]:
                continue
            if all(size in misses.get((era, o, l), ()) for o, l in todo[era]):
                tried_sizes[era].add(size)
                continue  # every pending pair already failed against this build
            table = bundle.offset_table(era)
            if table is None:
                continue  # leaves not inlined; the live fetch below covers it
            tried_sizes[era].add(size)
            for off, ln in list(todo[era]):
                got = table.get(off)
                if got and got[1] == ln:
                    resolved[(era, off, ln)] = got[0]
                    todo[era].discard((off, ln))
                    hits += 1
        print(f"  {bundle.path.name} (gen {bundle.generated[:10]}): +{hits} resolved")

    # Only go to the network for eras whose *current* layout was never tabled —
    # the ~35 multi-GB archives whose leaf directories are too big to inline,
    # plus any era missing from the bundles. An offset that failed against a
    # complete table is from a retired build and the live archive cannot help.
    leftover = {}
    for era, pairs in todo.items():
        if not pairs:
            continue
        current = next(
            (b.size_of(era) for b in bundles if b.size_of(era) is not None), None
        )
        if current is None or current not in tried_sizes[era]:
            leftover[era] = pairs
    stale = sum(
        len(p) for e, p in todo.items() if p and e not in leftover
    )
    if stale:
        print(f"  {stale} offsets predate the archive rebuild that served them "
              f"(no directory exists to decode them)")
    if leftover and use_live:
        n = sum(len(p) for p in leftover.values())
        print(f"  live R2 directories for {len(leftover)} eras ({n} offsets)")
        hits = 0
        for i, era in enumerate(sorted(leftover, key=lambda e: -len(leftover[e]))):
            pending = [p for p in leftover[era]]
            table, live_size = live_offset_table(era)
            if live_size is not None:
                tried_sizes[era].add(live_size)
            if table is None:
                continue
            for off, ln in pending:
                got = table.get(off)
                if got and got[1] == ln:
                    resolved[(era, off, ln)] = got[0]
                    leftover[era].discard((off, ln))
                    todo[era].discard((off, ln))
                    hits += 1
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(leftover)} eras, +{hits}")
        print(f"  live: +{hits} resolved")

    # Record misses with the builds they were tried against; a rerun retries
    # them only against builds not in that list.
    for era, pairs in todo.items():
        for off, ln in pairs:
            key = (era, off, ln)
            if key not in resolved:
                misses[key] = misses.get(key, set()) | tried_sizes[era]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {f"{e}\t{o}\t{l}": v for (e, o, l), v in resolved.items()}
    out.update({f"{e}\t{o}\t{l}": {"tried": sorted(t)} for (e, o, l), t in misses.items()
                if (e, o, l) not in resolved})
    with gzip.open(cache_path, "wt") as fh:
        json.dump(out, fh)
    return {k: v for k, v in resolved.items()} | {k: None for k in misses if k not in resolved}


# ---------------------------------------------------------------- geometry

def tile_bounds(z, x, y):
    """(west, south, east, north) in degrees for an XYZ tile."""
    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def era_start(era):
    return era.split("_to_")[0]


# ---------------------------------------------------------------- pipeline

def load_rows(keep_self, since=None, until=None):
    rows = []
    for path in sorted(LOG_DIR.glob("*.jsonl.gz")):
        day = path.name[:10]
        if (since and day < since) or (until and day > until):
            continue
        with gzip.open(path, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if not keep_self:
                    ua = r["ua"]
                    if ua in SELF_UA or r["ref"] in SELF_REF:
                        continue
                    low = ua.lower()
                    if "bot" in low or "crawl" in low or "spider" in low:
                        continue
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=9, help="grid zoom for the GeoJSON")
    ap.add_argument("--raster", type=int, default=900, help="density raster width px")
    ap.add_argument("--no-filter", action="store_true", help="keep bot/self traffic")
    ap.add_argument("--offline", action="store_true", help="skip live R2 fetches")
    ap.add_argument("--days", type=int, help="only the last N days of log")
    ap.add_argument("--since", help="first log day to include, YYYY-MM-DD")
    ap.add_argument("--until", help="last log day to include, YYYY-MM-DD")
    ap.add_argument("--tag", help="write outputs to a named subdirectory, so a "
                                  "windowed run does not overwrite the all-time one")
    args = ap.parse_args()

    since, until = args.since, args.until
    if args.days:
        days = sorted(p.name[:10] for p in LOG_DIR.glob("*.jsonl.gz"))
        if not days:
            sys.exit(f"no logs at {LOG_DIR} — run scripts/analytics_export.py first")
        # Window from the newest day present, not from today: a run made before
        # the day's export would otherwise silently drop a day off the end.
        end = datetime.strptime(days[-1], "%Y-%m-%d")
        since = max(since or "", (end - timedelta(days=args.days - 1))
                    .strftime("%Y-%m-%d"))
        until = until or days[-1]

    tag = args.tag or (f"last{args.days}d" if args.days else None)
    out_dir = OUT_DIR / tag if tag else OUT_DIR

    if not LOG_DIR.exists():
        sys.exit(f"no logs at {LOG_DIR} — run scripts/analytics_export.py first")
    out_dir.mkdir(parents=True, exist_ok=True)
    if since or until:
        print(f"window: {since or 'start'} .. {until or 'end'}"
              + (f"  -> {out_dir.name}/" if tag else ""))

    all_rows = load_rows(args.no_filter, since, until)
    era_rows = [r for r in all_rows if ERA_RE.match(r["era"])]
    # Keys that are not era archives: solo per-chart artifacts (chart/<slug>/
    # <date>), the basemap, metadata bundles, timeline/airfields JSON. Each
    # class is reported on its own instead of lumped as "metadata bundles".
    other = {"chart_solo": collections.Counter(), "basemap": collections.Counter(),
             "bundle": collections.Counter(), "other": collections.Counter()}
    for r in all_rows:
        if ERA_RE.match(r["era"]):
            continue
        k = r["era"]
        cls = ("chart_solo" if k.startswith("chart/") else "basemap" if "protomaps" in k or k.startswith("basemap")
               else "bundle" if k.startswith("metadata-") else "other")
        other[cls][k] += r["views"]
    # The tiles Worker logs every sampled range read above METADATA_BYTES and
    # every whole-file GET as if it were a tile read. Header probes (offset 0,
    # 16 KiB), whole-file GETs (len 0) and reads inside the directory region
    # are directory traffic, not tile views: separate them before counting.
    rows, dir_traffic = [], collections.Counter()
    for r in era_rows:
        if r["len"] == 0:
            dir_traffic["whole-file GET"] += r["views"]
        elif r["off"] == 0:
            dir_traffic["header probe"] += r["views"]
        else:
            rows.append(r)
    total_views = sum(r["views"] for r in rows)
    print(f"{len(rows)} tile-archive log rows, {total_views} sampled views"
          + (f" (+{sum(dir_traffic.values())} header/whole-file reads set aside)" if dir_traffic else "")
          + "".join(f" (+{sum(c.values())} {name} reads)" for name, c in other.items() if c))

    bundles = []
    for p in sorted(REPO.glob("metadata-*.bundle")):
        try:
            bundles.append(Bundle(p))
        except (ValueError, struct.error) as e:
            print(f"  skipping {p.name}: {e}", file=sys.stderr)
    bundles.sort(key=lambda b: b.generated, reverse=True)
    print(f"{len(bundles)} metadata bundles: "
          + ", ".join(f"{b.path.name[9:17]}@{b.generated[:10]}" for b in bundles))

    # Reads that start inside the directory region of the era's current
    # build are leaf-directory traffic, not tiles.
    tdo = {}
    for r in rows:
        if r["era"] not in tdo:
            tdo[r["era"]] = next((b.tile_data_offset(r["era"]) for b in bundles
                                  if b.tile_data_offset(r["era"]) is not None), None)
    kept = []
    for r in rows:
        t = tdo.get(r["era"])
        if t is not None and r["off"] < t:
            dir_traffic["leaf-directory read"] += r["views"]
        else:
            kept.append(r)
    rows = kept
    total_views = sum(r["views"] for r in rows)
    if dir_traffic:
        print("directory traffic excluded from tile views: "
              + ", ".join(f"{k} {v}" for k, v in dir_traffic.most_common()))

    wanted = collections.defaultdict(set)
    for r in rows:
        wanted[r["era"]].add((r["off"], r["len"]))
    resolved = resolve_offsets(wanted, bundles, use_live=not args.offline)

    # ---- fact table
    facts = []
    unresolved = 0
    for r in rows:
        tid = resolved.get((r["era"], r["off"], r["len"]))
        if tid is None:
            unresolved += r["views"]
            continue
        z, x, y = tileid_to_zxy(tid)
        w, s, e, n = tile_bounds(z, x, y)
        facts.append(
            {
                "hour": r["hour"],
                "era": r["era"],
                "era_start": era_start(r["era"]),
                "z": z, "x": x, "y": y,
                "lat": round((s + n) / 2, 4), "lon": round((w + e) / 2, 4),
                "country": r["country"], "cache": r["cache"],
                "views": r["views"],
            }
        )
    placed = sum(f["views"] for f in facts)
    denom = placed + unresolved
    print(f"\nplaced {placed} of {denom} views "
          f"({(placed / denom * 100) if denom else 0:.1f}%) — "
          f"{unresolved} undecodable (served by a build no bundle records)")
    if not facts:
        print("no tile views in this window; nothing to render", file=sys.stderr)

    with gzip.open(out_dir / "viewed_tiles.jsonl.gz", "wt") as fh:
        for f in facts:
            fh.write(json.dumps(f, separators=(",", ":")) + "\n")

    # ---- aggregates
    by_era_year = collections.Counter()
    by_era_decade = collections.Counter()
    by_day = collections.Counter()
    by_hour_utc = collections.Counter()
    by_country = collections.Counter()
    by_zoom = collections.Counter()
    for f in facts:
        yr = int(f["era_start"][:4])
        by_era_year[yr] += f["views"]
        by_era_decade[yr // 10 * 10] += f["views"]
        by_day[f["hour"][:10]] += f["views"]
        by_hour_utc[int(f["hour"][11:13])] += f["views"]
        by_country[f["country"]] += f["views"]
        by_zoom[f["z"]] += f["views"]

    # ---- z-normalised grid: a coarse tile spreads its weight over the cells
    # it covers, so the map reads as views per unit area rather than per tile.
    grid = collections.Counter()
    Z = args.zoom
    for f in facts:
        z, x, y, w = f["z"], f["x"], f["y"], f["views"]
        if z >= Z:
            grid[(x >> (z - Z), y >> (z - Z))] += w
        else:
            span = 1 << (Z - z)
            share = w / (span * span)
            for dx in range(span):
                for dy in range(span):
                    grid[(x * span + dx, y * span + dy)] += share

    features = []
    for (gx, gy), v in sorted(grid.items(), key=lambda kv: -kv[1]):
        west, south, east, north = tile_bounds(Z, gx, gy)
        features.append(
            {
                "type": "Feature",
                "properties": {"views": round(v, 3), "x": gx, "y": gy},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[west, south], [east, south], [east, north],
                                     [west, north], [west, south]]],
                },
            }
        )
    for old in out_dir.glob("heatmap_z*.geojson"):
        old.unlink()
    geojson = {"type": "FeatureCollection", "zoom": Z, "features": features}
    (out_dir / f"heatmap_z{Z}.geojson").write_text(json.dumps(geojson))

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "log_span": [min(by_day), max(by_day)] if by_day else None,
        "sampled_views_total": total_views,
        "sampled_views_placed": placed,
        "sampled_views_undecodable": unresolved,
        "sample_rate": SAMPLE_RATE,
        "estimated_true_tile_reads": round(total_views / SAMPLE_RATE) if SAMPLE_RATE else None,
        "directory_traffic_excluded": dict(dir_traffic.most_common()),
        "distinct_eras_viewed": len(set(f["era"] for f in facts)),
        "distinct_tiles": len(set((f["z"], f["x"], f["y"]) for f in facts)),
        "grid_zoom": Z,
        "grid_cells": len(grid),
        "by_era_year": dict(sorted(by_era_year.items())),
        "by_era_decade": dict(sorted(by_era_decade.items())),
        "by_day": dict(sorted(by_day.items())),
        "by_hour_utc": dict(sorted(by_hour_utc.items())),
        "by_country": dict(by_country.most_common()),
        "by_zoom": dict(sorted(by_zoom.items())),
        "non_tile_reads": {name: dict(c.most_common()) for name, c in other.items() if c},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\nera decades viewed: "
          + "  ".join(f"{d}s:{v}" for d, v in sorted(by_era_decade.items())))
    print(f"zoom mix: " + "  ".join(f"z{z}:{v}" for z, v in sorted(by_zoom.items())))
    print(f"\nwrote {out_dir.relative_to(REPO)}/"
          f"{{viewed_tiles.jsonl.gz, summary.json, heatmap_z{Z}.geojson}}")
    return facts, summary, grid, Z, args


if __name__ == "__main__":
    main()
