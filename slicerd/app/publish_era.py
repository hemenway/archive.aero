#!/usr/bin/env python3
"""Server half of an era publish: eligibility, metadata bundle, R2 uploads.

Ported from the 2026-09-29 run's bin/publish_loop.py (steps 1-4 of its
per-era order); the git half (viewer config, commit, push, Pages wait, CDN
check) stays on the Mac in `slicerctl publish`, which calls this through a
`publish` job and finishes the moment it returns.

  publish_era.py --release DIR --key KEY --pm FILE [--mosaic FILE]
                 --align-log FILE --stamp FILE --bundle-out DIR
                 [--allow-new] [--allow-unverified] [--max-tile-delta 0.02]
                 [--dry-run]

Order matters: the bundle is built (this era from the local file, every other
era range-read no-cache from R2) and uploaded BEFORE the era, so the window in
which the live viewer pairs new bytes with an old bundle is upload + push.
The last stdout line is `RESULT <json>`.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from osgeo import gdal
from pmtiles.reader import MmapSource, Reader
from pmtiles.tile import deserialize_header

gdal.UseExceptions()
ORIG = 20037508.342789244
UA = {"User-Agent": "archive-slicer-publish", "Cache-Control": "no-cache", "Pragma": "no-cache"}


def say(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def sh(*args):
    p = subprocess.run(list(args), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])}… rc={p.returncode}: {(p.stderr or p.stdout).strip()[-400:]}")
    return p.stdout.strip()


def fetch(url, start=None, end=None, timeout=120):
    h = dict(UA)
    if start is not None:
        h["Range"] = f"bytes={start}-{end}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, b"", e.headers
            if attempt == 4:
                raise
        except Exception:
            if attempt == 4:
                raise
        time.sleep(2 ** attempt)


def local_header(p):
    with open(p, "rb") as f:
        return Reader(MmapSource(f)).header()


def grid_zoom(mosaic):
    """Zoom whose 256-px tile pixel the mosaic sits on, or None."""
    gt = gdal.Open(str(mosaic)).GetGeoTransform()
    z = math.log2(2 * ORIG / 256 / gt[1])
    zr = round(z)
    res = 2 * ORIG / 256 / 2 ** zr
    if abs(gt[1] / res - 1) < 1e-9 and abs(-gt[5] / res - 1) < 1e-9:
        return zr
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--key", required=True)
    ap.add_argument("--pm", required=True, type=Path)
    ap.add_argument("--mosaic", type=Path)
    ap.add_argument("--align-log", type=Path)
    ap.add_argument("--stamp", type=Path)
    ap.add_argument("--bundle-out", required=True, type=Path)
    ap.add_argument("--remote-prefix", default="r2:charts/sectionals/")
    ap.add_argument("--allow-new", action="store_true")
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--max-tile-delta", type=float, default=0.02)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(a.release / "scripts"))
    import build_metadata_bundle as bmb  # noqa: E402  (from the synced code release)
    base = bmb.DEFAULT_BASE_URL
    key, pm = a.key, a.pm

    # ---- 1. eligibility -------------------------------------------------
    if not pm.exists():
        raise SystemExit(f"no local pmtiles: {pm}")
    converter = a.stamp.read_text().strip() if a.stamp and a.stamp.exists() else None
    if not converter:
        raise SystemExit("no converter stamp beside the pmtiles: convert it with this server first")
    worst = probes = None
    if a.align_log and a.align_log.exists():
        m = re.search(r"worst \|shift\| ([\d.]+) px over (\d+) probes \(tol ([\d.]+)\)", a.align_log.read_text())
        if m:
            worst, probes, tol = float(m.group(1)), int(m.group(2)), float(m.group(3))
            if probes and worst > tol:
                raise SystemExit(f"alignment failed: worst {worst} px > tol {tol}")
    if not probes and not a.allow_unverified:
        raise SystemExit("alignment unverified (no probe had chart coverage, or no align log); "
                         "pass allow_unverified after eyeballing the tiles")
    zoom = grid_zoom(a.mosaic) if a.mosaic and a.mosaic.exists() else None
    if a.mosaic and a.mosaic.exists() and zoom is None:
        raise SystemExit("mosaic is not on a tile-pixel grid: old-slicer build?")
    h = local_header(pm)
    if h["addressed_tiles_count"] <= 0:
        raise SystemExit("local pmtiles header has no tiles")
    st, data, _ = fetch(f"{base}{key}.pmtiles", 0, 126)
    live = None
    if st == 404:
        if not a.allow_new:
            raise SystemExit(f"{key} is not live yet: pass allow_new to publish a new era")
        say(f"{key}: new era (not live)")
    else:
        live = deserialize_header(data[:127])
        if h["max_zoom"] != live["max_zoom"]:
            raise SystemExit(f"max zoom z{h['max_zoom']} != live z{live['max_zoom']}")
        for f in ("min_lon_e7", "min_lat_e7", "max_lon_e7", "max_lat_e7"):
            if abs(h[f] - live[f]) > 0.05e7:
                raise SystemExit(f"{f} {h[f] / 1e7:.4f} vs live {live[f] / 1e7:.4f}")
        ratio = h["addressed_tiles_count"] / max(1, live["addressed_tiles_count"])
        if abs(ratio - 1) > a.max_tile_delta:
            raise SystemExit(f"tile count {h['addressed_tiles_count']} vs live {live['addressed_tiles_count']} "
                             f"({ratio - 1:+.1%}; limit ±{a.max_tile_delta:.0%})")
    say(f"eligible: converter {converter!r}, align worst {worst} px over {probes} probes, grid z{zoom}, "
        f"tiles {h['addressed_tiles_count']} vs live {live['addressed_tiles_count'] if live else '—'}")

    # ---- 2. bundle: this era local, every other era from R2 --------------
    class MixedSource:
        def __init__(self):
            self.local, self.remote = bmb.LocalSource(pm.parent), bmb.RemoteSource(base)

        @staticmethod
        def _retry(fn, *args):
            for attempt in range(3):
                try:
                    return fn(*args)
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(3 * 2 ** attempt)

        # The tiles Worker can wedge one key (2026-09-30); a remote read that
        # keeps failing is taken straight from R2 with rclone instead.
        @staticmethod
        def _r2(k, length=None):
            path = f"{a.remote_prefix}{k}.pmtiles"
            size = int(sh("rclone", "lsl", path).split()[0])
            if length is None:
                return size
            out = subprocess.run(["rclone", "cat", "--count", str(min(length, size)), path],
                                 capture_output=True, check=True).stdout
            return out, size

        def _remote(self, k, fn, *args):
            try:
                return self._retry(fn, *args)
            except Exception as e:
                say(f"  ! {k}: CDN read failed ({type(e).__name__}); reading from R2 directly")
                return None

        def size(self, k):
            if k == key:
                return pm.stat().st_size
            r = self._remote(k, self.remote.size, k)
            return r if r is not None else self._r2(k)

        def read(self, k, length):
            if k == key:
                with open(pm, "rb") as f:
                    return f.read(length)
            r = self._remote(k, self.remote.read, k, length)
            return r if r is not None else self._r2(k, length)[0]

        def read_prefix(self, k):
            if k == key:
                size = pm.stat().st_size
                with open(pm, "rb") as f:
                    return f.read(min(bmb.HEADER_READ, size)), size
            r = self._remote(k, self.remote.read_prefix, k)
            return r if r is not None else self._r2(k, bmb.HEADER_READ)

    with open(a.release / "dates.csv") as f:
        next(f)
        keys = {line.split(",", 1)[0].strip() for line in f if line.strip()}
    if key not in keys:
        say(f"{key} not in the release's dates.csv: adding it to the bundle")
        keys.add(key)
    keys = sorted(keys)
    t0 = time.time()
    src = MixedSource()
    warnings, records, failed = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(bmb.extract_one, src, k, warnings): k for k in keys}
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                rec = fut.result()
                if rec:
                    records.append(rec)
            except Exception as e:
                failed.append(f"{futs[fut]}: {type(e).__name__}: {e}")
            if n % 500 == 0:
                say(f"  bundle: {n}/{len(keys)} era prefixes read")
    if failed:
        raise SystemExit(f"bundle: {len(failed)} archives unreadable, e.g. {failed[:3]}")
    data, stats = bmb.build_bundle(records, base)
    name = f"metadata-{hashlib.sha256(data).hexdigest()[:8]}.bundle"
    a.bundle_out.mkdir(parents=True, exist_ok=True)
    bpath = a.bundle_out / name
    bpath.write_bytes(data)
    _, index, blobs_start = bmb.parse_bundle(bpath)
    era = next(e for e in index["eras"] if e["k"] == key)
    with open(pm, "rb") as f:
        assert data[blobs_start + era["off"]: blobs_start + era["off"] + era["len"]] == f.read(era["len"])
    assert era["size"] == pm.stat().st_size
    say(f"bundle {name}: {len(data) / 1e6:.1f} MB, {stats['eras']} eras, {len(warnings)} warnings, "
        f"{time.time() - t0:.0f}s")
    for w in warnings[:10]:
        say(f"  warning: {w}")

    with open(pm, "rb") as f:
        prefix_sha = hashlib.sha256(f.read(1 << 20)).hexdigest()
    result = {"key": key, "bundle": name, "prefix_sha256": prefix_sha, "bundle_path": str(bpath), "bundle_url": base + name,
              "bundle_size": len(data), "era_size": pm.stat().st_size, "eras": stats["eras"],
              "worst_px": worst, "probes": probes, "converter": converter, "new_era": live is None,
              "dry_run": a.dry_run}
    if a.dry_run:
        say("dry run: nothing uploaded")
        print("RESULT " + json.dumps(result), flush=True)
        return

    # ---- 3/4. bundle, then era, to R2 (copyto, never sync) ----------------
    def remote_size(remote):
        p = subprocess.run(["rclone", "lsl", remote], capture_output=True, text=True)
        parts = p.stdout.split()
        return int(parts[0]) if p.returncode == 0 and parts else None

    def upload(local, remote):
        size = local.stat().st_size
        before = remote_size(remote)
        t = time.time()
        sh("rclone", "copyto", "--s3-no-check-bucket", str(local), remote)
        after = remote_size(remote)
        if after != size:
            raise SystemExit(f"size mismatch {remote}: local {size} remote {after}")
        say(f"  ↑ {remote}: {size} B (was {before}) in {time.time() - t:.0f}s")
        return before

    upload(bpath, f"{a.remote_prefix}{name}")
    result["era_size_before"] = upload(pm, f"{a.remote_prefix}{key}.pmtiles")
    result["uploaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    say(f"uploaded {key} + {name}; the viewer still names the old bundle until index.html is pushed")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
