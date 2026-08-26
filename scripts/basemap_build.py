#!/usr/bin/env python3
"""Build archive.aero's self-hosted Protomaps vector basemap and publish it to R2.

Why self-hosted: Protomaps publishes a daily planet build at build.protomaps.com,
but the docs are explicit that "URLs may change and hotlinking to these downloads
are discouraged. Instead, you should copy the tileset to your own Cloud Storage."
Their only *served* endpoint (api.protomaps.com) is a keyed, metered API. So the
viewer reads a cutout we own, on data.archive.aero, through the tiles Worker.

Shape of the output (one PMTiles archive, two extracts merged):

  z0-6   whole world   ~45 MB   so zooming out never shows a torn planet
  z7-13  archive area  ~5.4 GB  every sectional footprint, plus panning slack

z13 is the ceiling because protomaps-leaflet's default ``levelDiff: 1`` reads the
data tile one level below the map zoom: the viewer's max zoom of 14 asks for z13,
so z13 is native detail, not overzoom. Extracting z14 as well would double the
archive for tiles nothing requests.

Usage
-----
    ~/venv/bin/python scripts/basemap_build.py              # build latest
    ~/venv/bin/python scripts/basemap_build.py --upload     # build + publish
    ~/venv/bin/python scripts/basemap_build.py --build 20260826 --dry-run

The published key is dated and immutable — ``basemap/protomaps-<build>.pmtiles``.
That is deliberate, and the one place archive.aero does *not* overwrite in place
(URI-POLICY rule 7): replacing a PMTiles file under a live key rewrites every
byte offset in it, while the Worker's per-range edge-cache entries carry no
version, so viewers would splice new directories onto old tile bodies for up to
24 h. Dated builds sidestep that entirely — same reasoning as metadata-*.bundle.
Retire an old build only once no deployed index.html references it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BUILDS_JSON = "https://build-metadata.protomaps.dev/builds.json"
BUILD_BASE = "https://build.protomaps.com/"
OUT_DIR = Path("/Volumes/projects/protomaps_basemap")
R2_REMOTE = "r2:charts"
R2_PREFIX = "basemap"

WORLD_MAXZOOM = 6
REGION_MAXZOOM = 13

# The extract footprint: rectangles covering every sectional the archive holds,
# with panning slack. Kept as an explicit constant rather than derived at build
# time so the basemap doesn't churn when the catalog gains a chart. Every build
# checks the constant still covers timeline_data.json and refuses to run if not.
# Rectangles must stay pairwise non-overlapping (a MultiPolygon of overlapping
# rings is not valid GeoJSON) and must not cross the antimeridian — the western
# Aleutians are split into their own east-of-180 box for exactly that reason.
REGION_BOXES = {
    #                west,    south,   east,   north
    "conus":      (-128.0,  21.0,  -58.5,  52.0),
    "alaska":     (-180.0,  47.0, -128.0,  74.0),
    "aleut_east": ( 166.0,  48.0,  180.0,  57.0),
    "hawaii":     (-163.0,  16.0, -152.0,  25.0),
    "caribbean":  ( -86.0,  14.0,  -58.5,  21.0),
    "mariana":    ( 140.0,   9.0,  150.0,  23.0),
    "samoa":      (-176.0, -18.0, -165.0, -10.0),
}


def run(cmd: list[str], dry: bool = False) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return
    subprocess.run([str(c) for c in cmd], check=True)


def fetch_builds() -> list[dict]:
    # The builds host 403s a bare Python-urllib user agent, same as
    # data.archive.aero's WAF does; send anything else.
    req = urllib.request.Request(BUILDS_JSON, headers={"User-Agent": "archive.aero-basemap/1.0"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)


def latest_build() -> dict:
    return max(fetch_builds(), key=lambda b: b["key"])


def find_build(stamp: str) -> dict:
    builds = fetch_builds()
    for b in builds:
        if b["key"] == f"{stamp}.pmtiles":
            return b
    raise SystemExit(
        f"build {stamp} not in the daily-build channel (it keeps ~1 week plus the "
        f"latest of each patch version); available: {', '.join(b['key'] for b in builds[-8:])}"
    )


def region_geojson() -> dict:
    boxes = list(REGION_BOXES.items())
    for i, (n1, b1) in enumerate(boxes):
        for n2, b2 in boxes[i + 1:]:
            overlap_x = min(b1[2], b2[2]) - max(b1[0], b2[0])
            overlap_y = min(b1[3], b2[3]) - max(b1[1], b2[1])
            if overlap_x > 0 and overlap_y > 0:
                raise SystemExit(f"REGION_BOXES {n1} and {n2} overlap; MultiPolygon would be invalid")

    def ring(b):
        w, s, e, n = b
        return [[[w, s], [e, s], [e, n], [w, n], [w, s]]]

    return {
        "type": "Feature",
        "properties": {"name": "archive.aero basemap coverage"},
        "geometry": {"type": "MultiPolygon", "coordinates": [ring(b) for b in REGION_BOXES.values()]},
    }


def verify_region(timeline: Path) -> None:
    """Assert REGION_BOXES still contains every chart footprint in the catalog."""
    if not timeline.exists():
        print(f"! {timeline} not present — skipping region coverage check", file=sys.stderr)
        return
    rings = json.loads(timeline.read_text())["rings"]
    outside = [
        (name, pt)
        for name, poly in rings.items()
        for r in poly
        for pt in r
        if not any(w <= pt[0] <= e and s <= pt[1] <= n for w, s, e, n in REGION_BOXES.values())
    ]
    if outside:
        names = sorted({n for n, _ in outside})
        raise SystemExit(
            "REGION_BOXES no longer covers the catalog — charts outside the extract "
            f"footprint would sit on a blank basemap: {', '.join(names)}\n"
            f"  first stray point: {outside[0][1]}\n"
            "Widen the box that should hold them and rebuild."
        )
    print(f"region covers all {len(rings)} chart footprints in {timeline.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", help="daily build stamp, e.g. 20260826 (default: newest)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--maxzoom", type=int, default=REGION_MAXZOOM)
    ap.add_argument("--world-maxzoom", type=int, default=WORLD_MAXZOOM)
    ap.add_argument("--threads", type=int, default=8, help="pmtiles extract download threads")
    ap.add_argument("--upload", action="store_true", help="rclone copyto the result into R2")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--keep-parts", action="store_true", help="keep the world/region extracts after merge")
    args = ap.parse_args()

    if not shutil.which("pmtiles"):
        raise SystemExit("pmtiles CLI not on PATH (brew install pmtiles)")
    if not args.out_dir.parent.exists():
        raise SystemExit(f"{args.out_dir.parent} not mounted")

    build = find_build(args.build) if args.build else latest_build()
    stamp = build["key"].removesuffix(".pmtiles")
    src = BUILD_BASE + build["key"]
    print(f"source build {stamp}  v{build.get('version')}  {build['size'] / 1e9:.1f} GB  uploaded {build['uploaded']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    verify_region(Path(__file__).resolve().parent.parent / "timeline_data.json")

    region_path = args.out_dir / "archive_region.geojson"
    region_path.write_text(json.dumps(region_geojson()))
    print(f"wrote {region_path}")

    world = args.out_dir / f"world-z0-{args.world_maxzoom}-{stamp}.pmtiles"
    region = args.out_dir / f"region-z{args.world_maxzoom + 1}-{args.maxzoom}-{stamp}.pmtiles"
    out = args.out_dir / f"protomaps-{stamp}.pmtiles"
    for stale in (world, region, out):
        if stale.exists() and not args.dry_run:
            stale.unlink()

    run(["pmtiles", "extract", src, world,
         "--bbox=-180,-85,180,85",
         "--minzoom=0", f"--maxzoom={args.world_maxzoom}",
         f"--download-threads={args.threads}"], args.dry_run)
    run(["pmtiles", "extract", src, region,
         f"--region={region_path}",
         f"--minzoom={args.world_maxzoom + 1}", f"--maxzoom={args.maxzoom}",
         f"--download-threads={args.threads}"], args.dry_run)
    # Zoom-disjoint inputs: merge concatenates their tile sets and rebuilds one
    # directory, so the result is a single archive spanning z0-maxzoom.
    run(["pmtiles", "merge", world, region, out], args.dry_run)
    run(["pmtiles", "verify", out], args.dry_run)

    if args.dry_run:
        return 0

    size = out.stat().st_size
    show = subprocess.run(["pmtiles", "show", str(out)], capture_output=True, text=True).stdout
    print(show)

    manifest = {
        "source_build": build["key"],
        "source_url": src,
        "source_size": build["size"],
        "source_b3sum": build.get("b3sum"),
        "basemap_version": build.get("version"),
        "world_maxzoom": args.world_maxzoom,
        "region_maxzoom": args.maxzoom,
        "region_boxes": REGION_BOXES,
        "output": out.name,
        "output_size": size,
        "r2_key": f"{R2_PREFIX}/{out.name}",
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out_dir / f"protomaps-{stamp}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{out}  {size / 1e9:.2f} GB")

    if not args.keep_parts:
        world.unlink(missing_ok=True)
        region.unlink(missing_ok=True)

    key = f"{R2_PREFIX}/{out.name}"
    upload = ["rclone", "copyto", str(out), f"{R2_REMOTE}/{key}",
              "--s3-upload-concurrency=8", "--s3-chunk-size=64M",
              "--stats-one-line", "--stats", "30s", "-v"]
    if args.upload:
        run(upload)
        print(f"\npublished https://data.archive.aero/{key}")
    else:
        print("\nto publish:\n  " + " ".join(upload))
    print("then point CONFIG.basemapUrl in index.html at "
          f"https://data.archive.aero/{key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
