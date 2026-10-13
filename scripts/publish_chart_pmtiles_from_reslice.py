#!/usr/bin/env python3
"""
Publish chart artifacts that are missing from R2 by copying them, server-side,
from a slicerd run's hashed beta bucket to their permanent keys.

Background (worklist 06, 2026-10-04): the per-chart artifacts of the August 29
run were never uploaded and no local copy survives. The 2026-10-03 reslice
regenerates every chart artifact and uploads it content-hashed to
`r2:charts-beta/sectionals/chart/<slug>/<date>.<sha256[:12]>.pmtiles`, deleting
the local file afterwards, so the beta bucket holds the only copy. The
permanent URI for the same chart is

    r2:charts/sectionals/chart/<slug>/<date>[-half]        (extension-less)

An rclone `copyto` between two buckets of the same remote is a server-side
CopyObject: nothing is downloaded. The copy keeps the object's metadata
(`content-type: application/vnd.pmtiles`, `sha256`); the tiles Worker passes
the content type through and the viewer does not look at it.

What is "missing": a key present in the canonical chart manifest
(`worklists/data/chart_pmtiles/manifest.jsonl`) with no record in
`uploads.jsonl` beside it. `--keys FILE` narrows that to an explicit list.
Keys with no record in the run's `hashed/plan.jsonl` yet are reported as
pending; rerun as the reslice advances.

Modes
    (default)   report: ready / pending / held, print the copy commands
    --copy      run the copies, verify size + sha256 metadata on the target,
                append receipts to uploads.jsonl
    --verify    no copies: check the target keys that already exist and
                write receipts for the ones that match the plan record
                (use after running the printed commands by hand)
    --refresh   not only the missing keys: every manifest key whose current
                receipt does not name the run's artifact (sha256 differs) is
                overwritten in place with the reslice's artifact - the
                permanent URI keeps its address, the bytes improve (URI-POLICY
                rule 7). Combine with --copy; the receipt it appends supersedes
                the older one (readers take the last receipt per key).

Receipts are appended in publish_chart_pmtiles.py's record shape
(`key`, `size`, `uploaded_at`) plus `sha256`, `source` and `run`, so
build_timeline_data.py stamps `pm` on them like any other upload.

Usage
    ~/venv/bin/python scripts/publish_chart_pmtiles_from_reslice.py
    ~/venv/bin/python scripts/publish_chart_pmtiles_from_reslice.py --copy
    ~/venv/bin/python scripts/publish_chart_pmtiles_from_reslice.py --verify
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHART_PM_DIR = REPO / "worklists" / "data" / "chart_pmtiles"
DEFAULT_RUN = Path("/Volumes/projects/slicer-runs/2026-10-03_reslice")
DEFAULT_REMOTE = "r2:charts"
DEFAULT_PREFIX = "sectionals"
# Keys deliberately not published even though the manifest has them.
HELD = {
    "chart/boston_ma/1957-06-01": "worklist 04: GCPs suspect, hand redo pending",
}


def read_jsonl(path: Path):
    out = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return out


def rclone_json(args):
    proc = subprocess.run(["rclone", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        return None, (proc.stderr or "").strip()
    try:
        return json.loads(proc.stdout or "[]"), ""
    except json.JSONDecodeError:
        return None, "unparseable rclone output"


def stat_object(remote_path: str):
    """{size, sha256 metadata} of one object, or None when absent."""
    items, err = rclone_json(["lsjson", "--metadata", "--files-only", remote_path])
    if items is None:
        if "directory not found" in err.lower() or "not found" in err.lower():
            return None
        print(f"ERROR: rclone lsjson {remote_path}: {err}", file=sys.stderr)
        sys.exit(1)
    for it in items:
        if it.get("IsDir"):
            continue
        meta = it.get("Metadata") or {}
        return {"size": it.get("Size", -1), "sha256": meta.get("sha256"),
                "content_type": meta.get("content-type")}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help=f"slicerd run dir (default: {DEFAULT_RUN})")
    ap.add_argument("--manifest", type=Path, default=CHART_PM_DIR / "manifest.jsonl")
    ap.add_argument("--uploads", type=Path, default=CHART_PM_DIR / "uploads.jsonl")
    ap.add_argument("--keys", type=Path, help="Only these keys (one chart/<slug>/<date> per line)")
    ap.add_argument("--catalog", type=Path, default=REPO / "master_dole_v2.csv",
                    help="dole CSV; a manifest key whose source file is no longer catalogued is reported as retired, not pending")
    ap.add_argument("--remote", default=DEFAULT_REMOTE, help=f"target rclone remote:bucket (default: {DEFAULT_REMOTE})")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"target key prefix (default: {DEFAULT_PREFIX})")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true", help="run the server-side copies and record receipts")
    mode.add_argument("--verify", action="store_true", help="record receipts for targets already copied by hand")
    ap.add_argument("--refresh", action="store_true",
                    help="also overwrite keys already published from an older warp (receipt sha256 != the run's artifact)")
    ap.add_argument("--limit", type=int, help="stop after this many keys (a dry batch)")
    args = ap.parse_args()

    plan_path = args.run / "hashed" / "plan.jsonl"
    if not plan_path.exists():
        sys.exit(f"ERROR: {plan_path} not found (share mounted? run name right?)")

    manifest = {}
    for e in read_jsonl(args.manifest):
        if "key" in e:
            manifest[e["key"]] = e
    receipts_by_key = {}
    for e in read_jsonl(args.uploads):
        if "key" in e:
            receipts_by_key[e["key"]] = e  # the last receipt per key is current
    uploaded = set(receipts_by_key)
    missing = {k: e for k, e in manifest.items() if k not in uploaded}
    if args.keys:
        wanted = {ln.strip() for ln in open(args.keys) if ln.strip()}
        unknown = wanted - set(manifest)
        if unknown:
            print(f"WARNING: {len(unknown)} requested keys not in the manifest: {sorted(unknown)[:5]}", file=sys.stderr)
        missing = {k: e for k, e in missing.items() if k in wanted}

    # plan.jsonl: one record per hashed object; `old` is the permanent path.
    plan = {}
    for r in read_jsonl(plan_path):
        if r.get("old", "").startswith("sectionals/chart/") and r.get("uploaded_at"):
            plan[r["old"][len("sectionals/"):]] = r  # chart/<slug>/<date>; a re-converted artifact supersedes

    # --refresh: published keys whose bytes are an older warp than the run's.
    stale = {}
    if args.refresh:
        for k, rec in receipts_by_key.items():
            if k in manifest and k in plan and k not in HELD and rec.get("sha256") != plan[k]["sha256"]:
                stale[k] = manifest[k]
        if args.keys:
            stale = {k: e for k, e in stale.items() if k in wanted}

    # A manifest entry outlives its catalog row: the 2016-20 Honolulu/Samoan
    # inset entries of the August 29 run name ACASIS source files that the
    # September 19 duplicate cleanup retired (their NARA twins publish under
    # the catalog's current location names, chart/honolulu/..., live since
    # August). Such keys are retired, never pending.
    catalogued = {}
    if args.catalog and args.catalog.exists():
        sys.path.insert(0, str(REPO / "scripts"))
        import dole_v2
        catalogued = {r["filename"]: r["location"] for r in dole_v2.load_rows(args.catalog)}

    ready, pending, held, retired, renamed = [], [], [], [], []
    for key in sorted(missing):
        entry = manifest[key]
        if key in HELD:
            held.append(key)
        elif key in plan:
            ready.append(key)
        elif catalogued and entry.get("filename") not in catalogued:
            retired.append(key)
        elif catalogued and catalogued[entry["filename"]] != entry.get("location"):
            # The row lives on under another location name, so its artifact
            # publishes under that slug; this key is the old spelling.
            renamed.append(key)
        else:
            pending.append(key)

    print(f"{len(manifest)} manifest keys, {len(uploaded)} with upload receipts, "
          f"{len(missing)} missing from R2" + (f", {len(stale)} published from an older warp (--refresh)" if args.refresh else ""))
    if args.refresh:
        ready = sorted(set(ready) | set(stale))
    if args.limit:
        ready = ready[:args.limit]
    print(f"  ready in {plan_path.parent.parent.name}: {len(ready)}   pending (reslice has not reached them): "
          f"{len(pending)}   held: {len(held)}   retired: {len(retired)}   renamed: {len(renamed)}")
    for k in held:
        print(f"  held   {k}  ({HELD[k]})")
    for k in retired:
        print(f"  retired {k}  (source {manifest[k].get('filename', '?')[-50:]} no longer in the catalog)")
    for k in renamed:
        print(f"  renamed {k}  (row is now location '{catalogued[manifest[k]['filename']]}', "
              f"was '{manifest[k].get('location')}')")
    if pending:
        eras = sorted({manifest[k].get("date_key", "?") for k in pending})
        print(f"  pending eras ({len(eras)}): {', '.join(eras)}")

    if not ready:
        print("Nothing ready to publish.")
        return

    now = datetime.datetime.now().isoformat(timespec="seconds")
    ok = bad = 0
    receipts = []
    for key in ready:
        r = plan[key]
        src = f"{r['remote']}/{r['path']}.pmtiles"
        dst = f"{args.remote}/{args.prefix}/{key}"
        cmd = ["rclone", "copyto", src, dst]
        if not (args.copy or args.verify):
            print(" ".join(cmd))
            continue
        if args.copy:
            print(f"  {'refresh' if key in stale else 'copy'} {key}  ({r['size'] / 1e6:.1f} MB)")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                bad += 1
                print(f"    ✗ rclone copyto failed: {(proc.stderr or '').strip()[-300:]}", file=sys.stderr)
                continue
        got = stat_object(dst)
        if got is None:
            bad += 1
            print(f"    ✗ {dst}: not present", file=sys.stderr)
            continue
        if got["size"] != r["size"]:
            bad += 1
            print(f"    ✗ {dst}: size {got['size']} != plan {r['size']}", file=sys.stderr)
            continue
        if got["sha256"] and got["sha256"] != r["sha256"]:
            bad += 1
            print(f"    ✗ {dst}: sha256 metadata {got['sha256'][:12]} != plan {r['sha256'][:12]}", file=sys.stderr)
            continue
        ok += 1
        receipts.append({"key": key, "size": r["size"], "uploaded_at": now,
                         "sha256": r["sha256"], "source": f"{r['remote']}/{r['path']}.pmtiles",
                         "run": args.run.name})
        print(f"    ✓ {key}  {got['size']} B  sha256 {('match' if got['sha256'] else 'not on object')}")

    if not (args.copy or args.verify):
        print(f"\n{len(ready)} command(s) above. Run them, then rerun with --verify to record receipts;"
              f" or rerun with --copy to do both.")
        return

    if receipts:
        with open(args.uploads, "a", encoding="utf-8") as f:
            for rec in receipts:
                f.write(json.dumps(rec) + "\n")
        print(f"\n{ok} receipt(s) appended to {args.uploads}")
        print("Next: ~/venv/bin/python scripts/build_timeline_data.py, upload timeline_data.json "
              "(CLAUDE.md recipe), then scripts/build_coverage.py.")
    if bad:
        print(f"{bad} key(s) failed verification; no receipt written for them.", file=sys.stderr)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
