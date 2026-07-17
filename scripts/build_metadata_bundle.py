#!/usr/bin/env python3
"""Build the PMTiles metadata bundle for archive.aero.

One artifact replaces dates.csv AND eliminates per-archive header/directory
reads during timeline scrubbing: it packs every era's PMTiles metadata prefix
(header + root directory + JSON metadata + leaf directories, when small)
behind a gzipped JSON index, grouped by decade so the client can pull a whole
decade's metadata in a single range read.

Bundle layout (little-endian):
    [0:8)     magic b"AAMBv1\\n\\0"
    [8:12)    uint32 gzipped-index length G
    [12:16)   uint32 flags (0)
    [16:16+G) gzip(JSON index)
    [16+G:)   metadata blobs, contiguous, decade-grouped, chronological

Index JSON (offsets are RELATIVE to blobs_start = 16 + G):
    {
      "version": 1,
      "generated": "...Z",
      "baseUrl": "https://data.archive.aero/sectionals/",
      "eras":   [{"k": key, "size": fileSize, "off": rel, "len": blobLen}, ...],
      "groups": [{"name": "1950s", "off": rel, "len": n, "i0": a, "i1": b}, ...]
    }

An era's blob is byte-for-byte the first `len` bytes of its .pmtiles file.
If `len` covers only header+root+metadata (big modern archives), leaf
directory reads fall through to normal network range requests.

Usage:
  # After a regeneration finishes uploading (local files are fastest):
  python scripts/build_metadata_bundle.py --dir /Volumes/drive/pmtiles \\
      --out . --emit-dates-csv dates.csv --update-html index.html

  # From production (no local copies needed):
  python scripts/build_metadata_bundle.py --remote --keys-from dates.csv --out .

  # Round-trip check of an existing bundle against its sources:
  python scripts/build_metadata_bundle.py --verify metadata-abc12345.bundle \\
      --dir /Volumes/drive/pmtiles
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
import random
import re
import struct
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from pmtiles.tile import deserialize_header
except ImportError:
    sys.exit("needs the pmtiles package (run with ~/venv/bin/python)")

MAGIC = b"AAMBv1\n\0"
PREAMBLE_LEN = 16
DEFAULT_BASE_URL = "https://data.archive.aero/sectionals/"
# Include leaf directories in the blob when the full metadata prefix is at
# most this many bytes; beyond it, bundle only header+root+JSON metadata and
# let leaf reads hit the network (only the ~34 multi-GB modern archives).
FULL_CAP = 256 * 1024
HEADER_READ = 16384

DATE_KEY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})$")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- sources

class LocalSource:
    """Reads era files from a directory of .pmtiles."""

    def __init__(self, directory):
        self.dir = Path(directory)

    def list_keys(self):
        return sorted(p.stem for p in self.dir.glob("*.pmtiles"))

    def size(self, key):
        return (self.dir / f"{key}.pmtiles").stat().st_size

    def read(self, key, length):
        with open(self.dir / f"{key}.pmtiles", "rb") as f:
            return f.read(length)


class RemoteSource:
    """Range-reads era files from the CDN."""

    def __init__(self, base_url):
        self.base_url = base_url

    def _url(self, key):
        return f"{self.base_url}{key}.pmtiles"

    def size(self, key):
        req = urllib.request.Request(self._url(key), method="HEAD",
                                     headers={"User-Agent": "aamb-builder"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Length"])

    def read(self, key, length):
        req = urllib.request.Request(
            self._url(key),
            headers={"Range": f"bytes=0-{length - 1}",
                     "User-Agent": "aamb-builder"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) < length:
            raise IOError(f"{key}: short read {len(data)} < {length}")
        return data


# ---------------------------------------------------------------- extraction

def parse_key_dates(key):
    m = DATE_KEY_RE.match(key)
    if not m:
        return None
    return m.group(1), m.group(2)


def metadata_prefix_len(header, file_size, key, warnings):
    """How many leading bytes of the file hold metadata worth bundling."""
    root_end = header["root_offset"] + header["root_length"]
    meta_end = header["metadata_offset"] + header["metadata_length"]
    leaf_len = header["leaf_directory_length"]
    leaf_off = header["leaf_directory_offset"]
    tile_off = header["tile_data_offset"]

    # Expected spec v3 layout: header, root dir, JSON metadata, leaf dirs,
    # tile data. Anything else: bundle nothing, era falls back to network.
    ordered = (127 <= header["root_offset"]
               and root_end <= header["metadata_offset"]
               and meta_end <= (leaf_off if leaf_len else tile_off)
               and (not leaf_len or leaf_off + leaf_len <= tile_off)
               and tile_off <= file_size)
    if not ordered:
        warnings.append(f"{key}: unexpected section layout, not bundling")
        return None

    full = tile_off  # header + root + metadata (+ leaves) all precede tiles
    if full <= FULL_CAP:
        return full
    trimmed = leaf_off if leaf_len else tile_off
    if trimmed > FULL_CAP:
        warnings.append(
            f"{key}: header+root+meta alone is {trimmed:,}B (> {FULL_CAP:,}B cap), bundling anyway")
    return trimmed


def extract_one(source, key, warnings):
    """Returns (key, start, end, file_size, blob bytes) or None to skip."""
    dates = parse_key_dates(key)
    if not dates:
        warnings.append(f"{key}: filename is not <start>_to_<end>, skipped")
        return None
    size = source.size(key)
    head = source.read(key, min(HEADER_READ, size))
    try:
        header = deserialize_header(head[:127])
    except Exception as e:
        warnings.append(f"{key}: header parse failed ({e}), skipped")
        return None
    prefix = metadata_prefix_len(header, size, key, warnings)
    if prefix is None:
        return None
    blob = head[:prefix] if prefix <= len(head) else source.read(key, prefix)
    return key, dates[0], dates[1], size, blob


# ---------------------------------------------------------------- assembly

def build_bundle(records, base_url):
    """records: list of (key, start, end, size, blob). Returns (bytes, stats)."""
    # Chronological within decade groups; groups chronological.
    records.sort(key=lambda r: (r[1][:3] + "0", r[1], r[2], r[0]))

    eras, groups, blobs = [], [], []
    rel = 0
    for rec in records:
        key, start, _end, size, blob = rec
        decade = start[:3] + "0s"
        if not groups or groups[-1]["name"] != decade:
            groups.append({"name": decade, "off": rel, "len": 0,
                           "i0": len(eras), "i1": len(eras)})
        eras.append({"k": key, "size": size, "off": rel, "len": len(blob)})
        groups[-1]["len"] += len(blob)
        groups[-1]["i1"] = len(eras)
        blobs.append(blob)
        rel += len(blob)

    index = {
        "version": 1,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseUrl": base_url,
        "eras": eras,
        "groups": groups,
    }
    index_gz = gzip.compress(
        json.dumps(index, separators=(",", ":")).encode(), mtime=0)

    out = io.BytesIO()
    out.write(MAGIC)
    out.write(struct.pack("<II", len(index_gz), 0))
    out.write(index_gz)
    for blob in blobs:
        out.write(blob)
    data = out.getvalue()

    stats = {
        "eras": len(eras),
        "groups": [(g["name"], g["i1"] - g["i0"], g["len"]) for g in groups],
        "index_gz": len(index_gz),
        "blobs": rel,
        "total": len(data),
    }
    return data, stats


def emit_dates_csv(records, base_url, path):
    rows = sorted(records, key=lambda r: (r[1], r[2]), reverse=True)
    with open(path, "w") as f:
        f.write("date_iso,url\n")
        for key, _s, _e, _size, _blob in rows:
            f.write(f"{key},{base_url}{key}.pmtiles\n")


def update_html(path, bundle_url):
    html = Path(path).read_text()
    pattern = r"(bundleUrl:\s*)(null|'[^']*')"
    if not re.search(pattern, html):
        sys.exit(f"{path}: no bundleUrl config entry found — wire the client first")
    html = re.sub(pattern, rf"\g<1>'{bundle_url}'", html, count=1)
    Path(path).write_text(html)


# ---------------------------------------------------------------- verify

def parse_bundle(path):
    data = Path(path).read_bytes()
    assert data[:8] == MAGIC, "bad magic"
    (index_gz_len, _flags) = struct.unpack("<II", data[8:16])
    index = json.loads(gzip.decompress(data[16:16 + index_gz_len]))
    blobs_start = PREAMBLE_LEN + index_gz_len
    return data, index, blobs_start


def verify_bundle(path, source, sample=25):
    data, index, blobs_start = parse_bundle(path)
    eras = index["eras"]
    log(f"verify: {len(eras)} eras, {len(index['groups'])} groups, "
        f"blobs_start={blobs_start}")
    for g in index["groups"]:
        i0, i1 = g["i0"], g["i1"]
        assert i1 > i0, f"group {g['name']} empty"
        assert eras[i0]["off"] == g["off"], f"group {g['name']} offset mismatch"
        span = eras[i1 - 1]["off"] + eras[i1 - 1]["len"] - g["off"]
        assert span == g["len"], f"group {g['name']} length mismatch"
    picks = random.sample(eras, min(sample, len(eras)))
    for era in picks:
        blob = data[blobs_start + era["off"]: blobs_start + era["off"] + era["len"]]
        fresh = source.read(era["k"], era["len"])
        assert blob == fresh, f"{era['k']}: bundle bytes differ from source"
        header = deserialize_header(bytes(blob[:127]))
        assert header["tile_data_offset"] >= era["len"], f"{era['k']}: prefix overruns tile data"
        assert era["size"] == source.size(era["k"]), f"{era['k']}: size changed"
    log(f"verify: OK ({len(picks)} eras byte-compared against source)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="directory of local .pmtiles files")
    ap.add_argument("--remote", action="store_true",
                    help="range-read files from --base-url instead of --dir")
    ap.add_argument("--keys-from", help="dates.csv to take the era list from "
                    "(default: --dir listing)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--emit-dates-csv", metavar="PATH",
                    help="also regenerate dates.csv at PATH")
    ap.add_argument("--update-html", metavar="PATH",
                    help="rewrite bundleUrl in the given index.html")
    ap.add_argument("--limit", type=int, help="only first N eras (testing)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--verify", metavar="BUNDLE",
                    help="verify an existing bundle against the source instead of building")
    args = ap.parse_args()

    if args.remote:
        source = RemoteSource(args.base_url)
    elif args.dir:
        source = LocalSource(args.dir)
    else:
        ap.error("need --dir or --remote")

    if args.verify:
        verify_bundle(args.verify, source)
        return

    if args.keys_from:
        keys = []
        with open(args.keys_from) as f:
            next(f)
            for line in f:
                key = line.split(",", 1)[0].strip()
                if key:
                    keys.append(key)
        keys = sorted(set(keys))
    elif isinstance(source, LocalSource):
        keys = source.list_keys()
    else:
        ap.error("--remote needs --keys-from (no way to list the CDN)")

    if args.limit:
        keys = keys[:args.limit]
    log(f"extracting metadata from {len(keys)} archives "
        f"({'remote' if args.remote else args.dir})")

    warnings, records, failed = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(extract_one, source, k, warnings): k for k in keys}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            try:
                rec = fut.result()
                if rec:
                    records.append(rec)
            except Exception as e:
                failed.append(f"{key}: {type(e).__name__}: {e}")
            done += 1
            if done % 500 == 0:
                log(f"  {done}/{len(keys)}")

    for w in warnings:
        log(f"  warn: {w}")
    for f_ in failed:
        log(f"  FAIL: {f_}")
    if not records:
        sys.exit("no usable archives")
    if failed:
        sys.exit(f"aborting: {len(failed)} archives unreadable — bundle would "
                 f"silently miss them (rerun when uploads finish)")

    data, stats = build_bundle(records, args.base_url)
    digest = hashlib.sha256(data).hexdigest()[:8]
    out_path = Path(args.out) / f"metadata-{digest}.bundle"
    out_path.write_bytes(data)

    log(f"\nwrote {out_path} ({stats['total'] / 1e6:.1f}MB: "
        f"index {stats['index_gz'] / 1e3:.0f}KB gz + blobs {stats['blobs'] / 1e6:.1f}MB)")
    log(f"eras: {stats['eras']}")
    for name, count, size in stats["groups"]:
        log(f"  {name}: {count:4} eras, {size / 1e6:6.2f}MB")

    bundle_url = f"{args.base_url}{out_path.name}"
    if args.emit_dates_csv:
        emit_dates_csv(records, args.base_url, args.emit_dates_csv)
        log(f"wrote {args.emit_dates_csv} ({stats['eras']} rows)")
    if args.update_html:
        update_html(args.update_html, bundle_url)
        log(f"updated bundleUrl in {args.update_html}")

    log(f"\nupload:  cd worker && npx wrangler r2 object put "
        f"charts/sectionals/{out_path.name} --file {out_path.resolve()} --remote")
    log(f"serves:  {bundle_url}")


if __name__ == "__main__":
    main()
