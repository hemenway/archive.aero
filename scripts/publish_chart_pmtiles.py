#!/usr/bin/env python3
"""
Upload per-chart full-sheet PMTiles (emitted by slicer.py --chart-pmtiles) to R2.

Keys are the durable chart URIs — extension-less, per
https://www.w3.org/Provider/Style/URI:

    r2:charts/sectionals/chart/<slug>/<date>[-half]
    -> https://data.archive.aero/sectionals/chart/<slug>/<date>[-half]

The local mirror keeps the .pmtiles extension; `rclone copyto` strips it per
file (never `sync` — see the atc-site case-collision incident). Upload events
are appended to uploads.jsonl beside the manifest; build_timeline_data.py
stamps `pm` on chart entries only for uploaded keys.

Usage:
    ~/venv/bin/python scripts/publish_chart_pmtiles.py --dry-run
    ~/venv/bin/python scripts/publish_chart_pmtiles.py
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "worklists" / "data" / "chart_pmtiles" / "manifest.jsonl"
DEFAULT_MIRROR = Path("/Volumes/drive/pmtiles_charts")
DEFAULT_REMOTE = "r2:charts"
DEFAULT_PREFIX = "sectionals/chart"


def load_jsonl(path: Path):
    entries = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def manifest_index(path: Path):
    """key -> latest manifest entry (append-only log, last line wins)."""
    index = {}
    for entry in load_jsonl(path):
        if "key" in entry:
            index[entry["key"]] = entry
    return index


def uploaded_keys(uploads_path: Path):
    return {e["key"] for e in load_jsonl(uploads_path) if "key" in e}


def remote_listing(remote: str, prefix: str):
    """{relative path under prefix: size} via rclone lsjson (server-side)."""
    cmd = ["rclone", "lsjson", "-R", "--files-only", f"{remote}/{prefix}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        # An empty/nonexistent prefix is fine on first publish.
        if "directory not found" in stderr.lower() or "doesn't exist" in stderr.lower():
            return {}
        print(f"ERROR: rclone lsjson failed: {stderr}", file=sys.stderr)
        sys.exit(1)
    try:
        items = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        items = []
    return {item["Path"]: item.get("Size", -1) for item in items if not item.get("IsDir")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR,
                    help=f"Local mirror root written by slicer --chart-pmtiles (default: {DEFAULT_MIRROR})")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help=f"Manifest JSONL from the slicer (default: {DEFAULT_MANIFEST})")
    ap.add_argument("--remote", default=DEFAULT_REMOTE, help=f"rclone remote:bucket (default: {DEFAULT_REMOTE})")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"Key prefix in the bucket (default: {DEFAULT_PREFIX})")
    ap.add_argument("--dry-run", action="store_true", help="Print what would upload; change nothing")
    ap.add_argument("--limit", type=int, default=0, help="Upload at most N files this run (0 = no limit)")
    ap.add_argument("--batch", action="store_true",
                    help="Stage pending files as extension-less hardlinks and upload with one parallel "
                         "`rclone copy` (fast path for thousands of files), then verify sizes via lsjson")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)
    if not args.mirror.exists():
        print(f"ERROR: mirror dir not found (volume mounted?): {args.mirror}", file=sys.stderr)
        sys.exit(1)

    uploads_path = args.manifest.parent / "uploads.jsonl"
    manifest = manifest_index(args.manifest)
    already = uploaded_keys(uploads_path)
    remote = remote_listing(args.remote, args.prefix)
    print(f"{len(manifest)} manifest keys; {len(already)} recorded uploads; {len(remote)} objects at {args.remote}/{args.prefix}")

    todo, missing_local, in_sync = [], [], 0
    for key, entry in sorted(manifest.items()):
        # key = chart/<slug>/<name>; local mirror path adds .pmtiles
        rel = key.split("/", 1)[1]  # <slug>/<name>
        local = args.mirror / (rel + ".pmtiles")
        if not local.exists():
            missing_local.append(key)
            continue
        size = local.stat().st_size
        if remote.get(rel) == size:
            in_sync += 1
            if key not in already and not args.dry_run:
                with open(uploads_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "size": size, "uploaded_at": "preexisting"}) + "\n")
            continue
        todo.append((key, rel, local, size))

    print(f"{in_sync} already in sync; {len(todo)} to upload; {len(missing_local)} manifest keys missing locally")
    for key in missing_local[:10]:
        print(f"  missing local: {key}")

    if args.limit and len(todo) > args.limit:
        todo = todo[: args.limit]
        print(f"Limiting to {args.limit} uploads this run")

    if args.batch and todo and not args.dry_run:
        # Hardlink the pending files into an extension-less staging tree, push
        # it with one parallel rclone copy, then verify by re-listing sizes.
        stage = args.mirror.parent / (args.mirror.name + "_publish_stage")
        if stage.exists():
            shutil.rmtree(stage)
        for key, rel, local, size in todo:
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(local, dest)
        print(f"Batch upload: {len(todo)} files via rclone copy ...")
        proc = subprocess.run(
            ["rclone", "copy", str(stage), f"{args.remote}/{args.prefix}",
             "--transfers", "8", "--checkers", "16", "--stats-one-line", "--stats", "30s", "-v"],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"WARNING: rclone copy exited {proc.returncode}: "
                  f"{(proc.stderr or '').strip()[-400:]}", file=sys.stderr)
        shutil.rmtree(stage, ignore_errors=True)
        after = remote_listing(args.remote, args.prefix)
        ok = bad = 0
        now = datetime.datetime.now().isoformat(timespec="seconds")
        with open(uploads_path, "a", encoding="utf-8") as f:
            for key, rel, local, size in todo:
                if after.get(rel) == size:
                    ok += 1
                    f.write(json.dumps({"key": key, "size": size, "uploaded_at": now}) + "\n")
                else:
                    bad += 1
                    print(f"  ✗ not verified in R2: {rel} (local {size}, remote {after.get(rel)})",
                          file=sys.stderr)
        print(f"Batch done: {ok} verified, {bad} failed verification.")
        if ok:
            print("Next: rerun build_timeline_data.py and republish timeline_data.json "
                  "so the viewer picks up the new charts (see CLAUDE.md recipe).")
        sys.exit(1 if bad else 0)

    failed = 0
    for i, (key, rel, local, size) in enumerate(todo, 1):
        dest = f"{args.remote}/{args.prefix}/{rel}"
        if args.dry_run:
            print(f"  [dry-run {i}/{len(todo)}] {local} -> {dest} ({size / 1e6:.1f} MB)")
            continue
        print(f"  [{i}/{len(todo)}] {rel} ({size / 1e6:.1f} MB)")
        proc = subprocess.run(["rclone", "copyto", str(local), dest], capture_output=True, text=True)
        if proc.returncode != 0:
            failed += 1
            print(f"    ✗ upload failed: {(proc.stderr or '').strip()[-200:]}", file=sys.stderr)
            continue
        with open(uploads_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "key": key, "size": size,
                "uploaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }) + "\n")

    if args.dry_run:
        print("Dry run complete — nothing uploaded.")
    else:
        print(f"Done: {len(todo) - failed} uploaded, {failed} failed.")
        if len(todo) - failed:
            print("Next: rerun build_timeline_data.py and republish timeline_data.json "
                  "so the viewer picks up the new charts (see CLAUDE.md recipe).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
