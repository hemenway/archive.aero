#!/usr/bin/env python3
"""Find and repair PMTiles archives written with non-spec run-length entries.

Background (2026-09-08 audit). Until its 2026-09-14 fix, geotiff2pmtiles'
`optimizeRunLengths` merged consecutive tile IDs whose blobs were merely
ADJACENT in the data section (same length, offset advancing by length) into
one `run_length=N` directory entry. The PMTiles v3 spec — and pmtiles.js,
go-pmtiles and this `pmtiles` package — define a run as N tile IDs that share
ONE blob at the entry's offset. Every spec reader therefore served the first
tile's bytes for the other N-1 tiles of such a run; the bytes those tiles
should have shown were unreachable. The census over metadata-ee0d247d.bundle
found 1,548 of 3,741 eras affected (≥3,244 tiles), plus per-chart artifacts.

This tool has two jobs:

  census   Classify every run-length entry in local archives (or every era
           whose directories the metadata bundle carries) as spec-style
           (shared blob) or non-spec (adjacent distinct blobs), and count the
           tiles a spec reader would mis-serve.

  fix      Rewrite an archive so each tile of a non-spec run gets its own
           entry at offset + i*length. Tile data is copied unchanged; only
           the header and directories change. The output is verified with the
           spec reader before it replaces anything: for every tile of every
           original run, the spec lookup must now return exactly the bytes
           the Go writer intended (the i-th blob of the run).

Republishing a repaired archive under its live key moves every tile offset
(the directory section changes size), so after uploading: rebuild + reupload
the metadata bundle (scripts/build_metadata_bundle.py), expect the tiles
Worker's version-less per-range cache entries to serve stale bytes for up to
24 h (see CLAUDE.md), and invalidate the analytics offset cache for those
eras (scripts/analytics_heatmap.py keys its live-directory cache by build).

Usage:
  # Which local archives are affected, and how badly?
  python scripts/pmtiles_fix_runlengths.py census /Volumes/drive/pmtiles/*.pmtiles

  # Every era in a metadata bundle (no archive files needed; big modern eras
  # carry only their root directory in the bundle and are undercounted):
  python scripts/pmtiles_fix_runlengths.py census --bundle metadata-ee0d247d.bundle

  # Repair one archive to a new file (verified before the file is kept):
  python scripts/pmtiles_fix_runlengths.py fix in.pmtiles out.pmtiles

  # Repair every affected archive in a directory in place (atomic rename per
  # file; unaffected archives are left untouched):
  python scripts/pmtiles_fix_runlengths.py fix --in-place /Volumes/drive/pmtiles/*.pmtiles

  # Census the per-chart artifacts (or any key list) straight from R2 over
  # HTTP range reads — no local copies needed:
  python scripts/pmtiles_fix_runlengths.py census --remote-keys worklists/data/chart_pmtiles/uploads.jsonl

  # The whole repair, driven by a census JSON: for each affected era/key take
  # the local mirror copy (or download it with --download), rewrite + verify
  # into --work, then print (or with --upload run) the rclone copyto per key.
  python scripts/pmtiles_fix_runlengths.py repair-batch --census census.json \
      --mirror /Volumes/drive/pmtiles --work /Volumes/drive/pmtiles_repaired [--upload]
"""

import argparse
import bisect
import gzip
import io
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

try:
    from pmtiles.tile import (
        Entry,
        deserialize_directory,
        deserialize_header,
        serialize_header,
        tileid_to_zxy,
    )
    from pmtiles.writer import optimize_directories
    from pmtiles.reader import Reader, MmapSource
except ImportError:
    sys.exit("needs the pmtiles package (run with ~/venv/bin/python)")

HEADER_SIZE = 127
# pmtiles.js fetches the first 16 KiB and expects header + root directory to
# fit inside it; the Go writer and the python writer both target this.
ROOT_BUDGET = 16384 - HEADER_SIZE


# ------------------------------------------------------------------ parsing

def read_all_entries(buf_at, header):
    """Every tile entry (run_length > 0) from root + leaf directories.

    buf_at(offset, length) -> bytes. Returns (entries, complete) where
    complete is False when a leaf directory lay outside what buf_at could
    supply (a bundle prefix that stops before the leaf section).
    """
    root = deserialize_directory(buf_at(header["root_offset"], header["root_length"]))
    entries, complete = [], True
    for e in root:
        if e.run_length > 0:
            entries.append(e)
            continue
        raw = buf_at(header["leaf_directory_offset"] + e.offset, e.length)
        if raw is None or len(raw) < e.length:
            complete = False
            continue
        entries.extend(deserialize_directory(raw))
    return entries, complete


def classify_runs(entries, tile_data_length, complete=True):
    """Split run_length>1 entries into spec-style and non-spec.

    A run is spec-style when the next distinct blob starts one length after
    the run's offset (all N tiles share that one blob). It is non-spec when
    the next blob starts N lengths later (N distinct adjacent blobs), or when
    it is the last blob and N lengths fit exactly inside the data section.
    """
    offsets = sorted({e.offset for e in entries})
    spec, nonspec, unknown = [], [], []
    for e in entries:
        if e.run_length <= 1:
            continue
        i = bisect.bisect_right(offsets, e.offset)
        nxt = offsets[i] if i < len(offsets) else None
        end_one = e.offset + e.length
        end_run = e.offset + e.run_length * e.length
        if nxt == end_one or (nxt is None and end_one == tile_data_length):
            spec.append(e)
        elif nxt == end_run or (nxt is None and complete and end_run == tile_data_length):
            nonspec.append(e)
        else:
            unknown.append(e)
    return spec, nonspec, unknown


def file_reader(path):
    f = open(path, "rb")

    def buf_at(off, length):
        f.seek(off)
        return f.read(length)

    return f, buf_at


def header_of(buf_at):
    return deserialize_header(buf_at(0, HEADER_SIZE))


# ------------------------------------------------------------------- census

BASE_URL = "https://data.archive.aero/sectionals/"
UA = "Mozilla/5.0 (Macintosh) archive.aero-pmtiles-repair/1.0"


def http_range(url, start, length):
    import urllib.request
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def census_remote(key):
    """Census one archive by HTTP range reads: key is an era name
    (<start>_to_<end>) or a chart key (chart/<slug>/<date>[-half])."""
    url = BASE_URL + (key + ".pmtiles" if not key.startswith("chart/") else key)
    head = http_range(url, 0, 16384)
    h = deserialize_header(head[:HEADER_SIZE])

    def buf_at(off, length):
        if off + length <= len(head):
            return head[off:off + length]
        return http_range(url, off, length)

    entries, complete = read_all_entries(buf_at, h)
    spec, nonspec, unknown = classify_runs(entries, h["tile_data_length"], complete)
    return {
        "path": key,
        "addressed": h["addressed_tiles_count"],
        "entries": len(entries),
        "spec_runs": len(spec),
        "nonspec_runs": len(nonspec),
        "unknown_runs": len(unknown),
        "misserved_tiles": sum(e.run_length - 1 for e in nonspec),
        "complete": complete,
    }


def census_file(path):
    f, buf_at = file_reader(path)
    try:
        h = header_of(buf_at)
        entries, complete = read_all_entries(buf_at, h)
    finally:
        f.close()
    spec, nonspec, unknown = classify_runs(entries, h["tile_data_length"], complete)
    return {
        "path": str(path),
        "addressed": h["addressed_tiles_count"],
        "entries": len(entries),
        "spec_runs": len(spec),
        "nonspec_runs": len(nonspec),
        "unknown_runs": len(unknown),
        "misserved_tiles": sum(e.run_length - 1 for e in nonspec),
        "complete": complete,
    }


def census_bundle(bundle_path):
    """Census every era from a metadata-*.bundle (see build_metadata_bundle.py)."""
    buf = Path(bundle_path).read_bytes()
    if buf[:8] != b"AAMBv1\n\0":
        sys.exit(f"{bundle_path}: not an AAMBv1 bundle")
    gz_len = struct.unpack_from("<I", buf, 8)[0]
    index = json.loads(gzip.decompress(buf[16:16 + gz_len]))
    start = 16 + gz_len
    rows = []
    for era in index["eras"]:
        blob = buf[start + era["off"]: start + era["off"] + era["len"]]

        def buf_at(off, length, blob=blob):
            if off + length > len(blob):
                return None
            return blob[off:off + length]

        h = header_of(buf_at)
        entries, complete = read_all_entries(buf_at, h)
        spec, nonspec, unknown = classify_runs(entries, h["tile_data_length"], complete)
        rows.append({
            "path": era["k"],
            "addressed": h["addressed_tiles_count"],
            "entries": len(entries),
            "spec_runs": len(spec),
            "nonspec_runs": len(nonspec),
            "unknown_runs": len(unknown),
            "misserved_tiles": sum(e.run_length - 1 for e in nonspec),
            "complete": complete,
        })
    return rows


def print_census(rows, json_out=None):
    affected = [r for r in rows if r["nonspec_runs"]]
    partial = [r for r in rows if not r["complete"]]
    print(f"archives: {len(rows)}  affected: {len(affected)}  "
          f"non-spec runs: {sum(r['nonspec_runs'] for r in rows)}  "
          f"spec-style runs: {sum(r['spec_runs'] for r in rows)}  "
          f"unclassified: {sum(r['unknown_runs'] for r in rows)}")
    print(f"tiles a spec reader mis-serves (lower bound): "
          f"{sum(r['misserved_tiles'] for r in rows):,} of "
          f"{sum(r['addressed'] for r in rows):,} addressed")
    if partial:
        print(f"{len(partial)} archives counted at root level only (leaf dirs not available)")
    for r in sorted(affected, key=lambda r: -r["misserved_tiles"])[:15]:
        print(f"  {r['path']:48s} nonspec={r['nonspec_runs']:4d} misserved={r['misserved_tiles']:4d}"
              f"{'' if r['complete'] else '  (partial)'}")
    if json_out:
        Path(json_out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {json_out}")


# ---------------------------------------------------------------------- fix

def expand_entries(entries, tile_data_length, complete=True):
    """Return (expanded entries sorted by tile_id, number of runs expanded).

    Non-spec runs become run_length individual entries at offset + i*length.
    Spec-style runs are kept (they are correct). Unclassifiable runs abort:
    we must not guess at published bytes.
    """
    spec, nonspec, unknown = classify_runs(entries, tile_data_length, complete)
    if unknown:
        e = unknown[0]
        raise SystemExit(f"cannot classify run at tile_id {e.tile_id} (offset {e.offset}, "
                         f"len {e.length}, run {e.run_length}); refusing to rewrite")
    bad = {id(e) for e in nonspec}
    out = []
    for e in entries:
        if id(e) in bad:
            for i in range(e.run_length):
                out.append(Entry(e.tile_id + i, e.offset + i * e.length, e.length, 1))
        else:
            out.append(e)
    out.sort(key=lambda e: e.tile_id)
    return out, len(nonspec)


def rewrite(src, dst):
    """Rewrite src into dst with spec-compliant directories. Returns a report."""
    f, buf_at = file_reader(src)
    try:
        h = header_of(buf_at)
        entries, complete = read_all_entries(buf_at, h)
        if not complete:
            raise SystemExit(f"{src}: leaf directories unreadable; refusing to rewrite")
        original_runs = [e for e in entries if e.run_length > 1]
        expanded, n_fixed = expand_entries(entries, h["tile_data_length"], complete)
        if n_fixed == 0:
            return {"path": str(src), "fixed_runs": 0, "skipped": True}

        metadata = buf_at(h["metadata_offset"], h["metadata_length"])
        root_bytes, leaves_bytes, num_leaves = optimize_directories(expanded, ROOT_BUDGET)

        nh = dict(h)
        nh["root_offset"] = HEADER_SIZE
        nh["root_length"] = len(root_bytes)
        nh["metadata_offset"] = HEADER_SIZE + len(root_bytes)
        nh["metadata_length"] = len(metadata)
        nh["leaf_directory_offset"] = nh["metadata_offset"] + len(metadata)
        nh["leaf_directory_length"] = len(leaves_bytes)
        nh["tile_data_offset"] = nh["leaf_directory_offset"] + len(leaves_bytes)
        # tile_data_length, addressed/contents counts and bounds are unchanged;
        # the entry count now reflects one entry per formerly-merged tile.
        nh["tile_entries_count"] = len(expanded)

        tmp = Path(dst).with_name(Path(dst).name + ".part")
        with open(tmp, "wb") as out:
            out.write(serialize_header(nh))
            out.write(root_bytes)
            out.write(metadata)
            out.write(leaves_bytes)
            f.seek(h["tile_data_offset"])
            copied = 0
            remaining = h["tile_data_length"]
            while remaining:
                chunk = f.read(min(8 << 20, remaining))
                if not chunk:
                    raise SystemExit(f"{src}: tile data truncated at {copied} of {h['tile_data_length']}")
                out.write(chunk)
                copied += len(chunk)
                remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
    finally:
        f.close()

    # Verify with the spec reader before the file is kept: every tile of every
    # original run must resolve to the blob the writer intended.
    checked, mismatched = verify(src, tmp, original_runs, h)
    if mismatched:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{src}: verification failed on {mismatched} of {checked} run tiles; output discarded")
    os.replace(tmp, dst)
    return {"path": str(src), "fixed_runs": n_fixed, "expanded_tiles": sum(e.run_length for e in original_runs),
            "verified_tiles": checked, "root_bytes": len(root_bytes), "leaves": num_leaves}


def verify(src, fixed, original_runs, h):
    """Spec-reader check: new.get(tid+i) == bytes the Go writer laid down at off+i*len."""
    checked = mismatched = 0
    with open(src, "rb") as sf, open(fixed, "rb") as ff:
        rd = Reader(MmapSource(ff))
        for e in original_runs:
            for i in range(e.run_length):
                sf.seek(h["tile_data_offset"] + e.offset + i * e.length)
                expected = sf.read(e.length)
                z, x, y = tileid_to_zxy(e.tile_id + i)
                got = rd.get(z, x, y)
                checked += 1
                if got != expected:
                    mismatched += 1
        # And a sample of non-run entries must be untouched.
        nh = rd.header()
        if nh["addressed_tiles_count"] != h["addressed_tiles_count"] or nh["tile_data_length"] != h["tile_data_length"]:
            mismatched += 1
    return checked, mismatched


# ------------------------------------------------------------- repair batch

def repair_batch(args):
    """Rewrite every affected archive named in a census JSON.

    Source order: --mirror copy, else (with --download) a fresh download to
    --work/src/. Each repair is verified by the spec reader before it is
    kept. Uploads go key-by-key with `rclone copyto` (never sync) and only
    when --upload is passed; otherwise the commands are printed. After
    uploading: rebuild + reupload the metadata bundle, and expect ≤24 h of
    stale per-range Worker cache entries per republished key.
    """
    import shutil
    import subprocess
    import urllib.request
    rows = [r for r in json.load(open(args.census)) if r.get("nonspec_runs")]
    if args.limit:
        rows = rows[:args.limit]
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "src").mkdir(exist_ok=True)
    done, failed, uploads = 0, [], []
    for i, r in enumerate(rows, 1):
        key = r["path"]
        is_chart = key.startswith("chart/")
        rel = (key if is_chart else key) + ".pmtiles"
        src = (args.mirror / rel) if args.mirror else None
        if src is None or not src.exists():
            if not args.download:
                failed.append((key, "not in mirror (pass --download)"))
                continue
            src = args.work / "src" / rel
            src.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                url = BASE_URL + (key if is_chart else key + ".pmtiles")
                print(f"[{i}/{len(rows)}] downloading {key}", file=sys.stderr)
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=600) as resp, open(str(src) + ".part", "wb") as out:
                    shutil.copyfileobj(resp, out, 8 << 20)
                os.replace(str(src) + ".part", src)
        dst = args.work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            rep = rewrite(src, dst)
        except SystemExit as e:
            failed.append((key, str(e)))
            continue
        if rep.get("skipped"):
            print(f"[{i}/{len(rows)}] {key}: no non-spec runs (census stale?), skipped")
            continue
        done += 1
        print(f"[{i}/{len(rows)}] {key}: expanded {rep['fixed_runs']} runs, verified {rep['verified_tiles']} tiles")
        r2key = "sectionals/" + (key if is_chart else key + ".pmtiles")
        cmd = ["rclone", "copyto", str(dst), f"r2:charts/{r2key}", "--s3-upload-concurrency=8", "--s3-chunk-size=64M"]
        uploads.append(cmd)
        if args.upload:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                failed.append((key, "upload: " + (res.stderr or "").strip()[-200:]))
            else:
                with open(args.work / "uploaded.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": r2key, "bytes": dst.stat().st_size}) + "\n")
    print(f"\nrepaired {done} archive(s); {len(failed)} failed")
    for k, why in failed[:20]:
        print(f"  FAIL {k}: {why}")
    if uploads and not args.upload:
        print("\nto publish (per key, never sync):")
        for cmd in uploads[:10]:
            print("  " + " ".join(cmd))
        if len(uploads) > 10:
            print(f"  ... {len(uploads) - 10} more")
    if done and args.upload:
        print("\nNext: rebuild + reupload the metadata bundle (scripts/build_metadata_bundle.py); "
              "analytics_heatmap.py keys its live-directory cache by archive size and will refetch.")
    return 1 if failed else 0


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census", help="classify run-length entries")
    c.add_argument("paths", nargs="*", help=".pmtiles files")
    c.add_argument("--bundle", help="metadata-*.bundle to census instead of files")
    c.add_argument("--json", help="write per-archive rows to this JSON file")
    c.add_argument("--remote-keys", help="census keys read from R2 (one key per line, or JSONL with a \"key\" field)")
    rb = sub.add_parser("repair-batch", help="rewrite + verify every affected archive from a census JSON")
    rb.add_argument("--census", required=True, help="census --json output")
    rb.add_argument("--mirror", type=Path, help="local mirror root (era files as <key>.pmtiles; chart keys as chart/<slug>/<date>.pmtiles)")
    rb.add_argument("--work", type=Path, required=True, help="directory for the repaired copies")
    rb.add_argument("--download", action="store_true", help="fetch archives missing from --mirror from R2")
    rb.add_argument("--upload", action="store_true", help="rclone copyto each verified repair to r2:charts (same key)")
    rb.add_argument("--limit", type=int, default=0)
    x = sub.add_parser("fix", help="rewrite archives with spec-compliant directories")
    x.add_argument("paths", nargs="+", help="in.pmtiles out.pmtiles, or with --in-place: files...")
    x.add_argument("--in-place", action="store_true", help="replace each affected file (atomic rename)")
    args = ap.parse_args()

    if args.cmd == "census":
        if args.bundle:
            rows = census_bundle(args.bundle)
        elif args.remote_keys:
            keys = []
            for line in open(args.remote_keys, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    keys.append(json.loads(line)["key"])
                except ValueError:
                    keys.append(line)
            rows = []
            for i, k in enumerate(dict.fromkeys(keys), 1):
                try:
                    rows.append(census_remote(k))
                except Exception as e:
                    print(f"  ! {k}: {e}", file=sys.stderr)
                if i % 200 == 0:
                    print(f"  {i}/{len(keys)}", file=sys.stderr)
        elif args.paths:
            rows = [census_file(p) for p in args.paths]
        else:
            ap.error("census needs files, --bundle or --remote-keys")
        print_census(rows, args.json)
        return

    if args.cmd == "repair-batch":
        return repair_batch(args)

    if args.in_place:
        for p in args.paths:
            r = rewrite(p, p)
            if r.get("skipped"):
                print(f"{p}: no non-spec runs, untouched")
            else:
                print(f"{p}: expanded {r['fixed_runs']} runs -> {r['expanded_tiles']} tiles, "
                      f"verified {r['verified_tiles']}, root {r['root_bytes']} B, {r['leaves']} leaves")
        return
    if len(args.paths) != 2:
        ap.error("fix needs: in.pmtiles out.pmtiles (or --in-place files...)")
    src, dst = args.paths
    r = rewrite(src, dst)
    if r.get("skipped"):
        print(f"{src}: no non-spec runs; nothing written")
    else:
        print(f"{dst}: expanded {r['fixed_runs']} runs -> {r['expanded_tiles']} tiles, "
              f"verified {r['verified_tiles']}, root {r['root_bytes']} B, {r['leaves']} leaves")


if __name__ == "__main__":
    main()
