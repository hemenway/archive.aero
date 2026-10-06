#!/usr/bin/env python3
"""Restrict hashed plans to what the live viewer uses, and stage the stubs for the manifest.

R2 keeps era archives the timeline no longer lists (start-only keys from before the
`{start}_to_{end}` rename, ranges superseded by a republish). The manifest builder
includes every era archive it is given, so those must be filtered out first: an
old range overlapping its replacement would draw the wrong chart.

  next_select_live.py --plan PLAN.json [--plan ...] --dates dates.csv \\
      --out-plan final.json [--stubs STUBS --out-stubs LIVE] [--shards N --shard-dir DIR]

Kept: era archives whose key is in dates.csv, every per-chart artifact, and every
basemap/airspace record. --out-stubs receives APFS clones of the kept era, basemap
and airspace stubs (chart stubs are not needed by the manifest). --shard-dir
receives the kept records dealt largest-first into shard-N.json files for
next_execute_plan.sh.
"""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


def live_eras(dates_csv):
    keys = set()
    with open(dates_csv, newline='') as f:
        for row in csv.DictReader(f):
            key = row['url'].rsplit('/', 1)[-1].removesuffix('.pmtiles')
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}', key): raise ValueError('dates.csv era key is not {start}_to_{end}: '+key)
            keys.add(key)
    return keys


def select(records, live):
    kept, dropped, found = [], [], set()
    for r in records:
        old = r['old']
        if old.startswith('sectionals/') and '/chart/' not in old:
            key = old[len('sectionals/'):].removesuffix('.pmtiles')
            if key not in live: dropped.append(old); continue
            found.add(key)
        kept.append(r)
    missing = sorted(live-found)
    if missing: raise ValueError(f'{len(missing)} live eras have no plan record, e.g. {missing[:3]}')
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', action='append', required=True); ap.add_argument('--dates', required=True); ap.add_argument('--out-plan', required=True)
    ap.add_argument('--stubs'); ap.add_argument('--out-stubs'); ap.add_argument('--shards', type=int, default=12); ap.add_argument('--shard-dir')
    args = ap.parse_args()
    records = [r for p in args.plan for r in json.loads(Path(p).read_text())]
    kept, dropped = select(records, live_eras(args.dates))
    Path(args.out_plan).write_text(json.dumps(kept, indent=1)+'\n')
    eras = sum(1 for r in kept if r['old'].startswith('sectionals/') and '/chart/' not in r['old'])
    print(f'{len(kept)} records kept ({eras} live eras, {sum("/chart/" in r["old"] for r in kept)} chart artifacts); {len(dropped)} stale era archives dropped')
    if args.out_stubs:
        if not args.stubs: ap.error('--out-stubs needs --stubs')
        for r in kept:
            if '/chart/' in r['old']: continue
            source = Path(args.stubs)/r['new']; target = Path(args.out_stubs)/r['new']
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): target.unlink()
            subprocess.run(['cp', '-c', str(source), str(target)], check=True)  # APFS clone: holes stay holes
    if args.shard_dir:
        out = Path(args.shard_dir); out.mkdir(parents=True, exist_ok=True)
        ordered = sorted(kept, key=lambda r: -(r.get('size') or 0))
        for i in range(args.shards): (out/f'shard-{i}.json').write_text(json.dumps(ordered[i::args.shards]))
        print(f'{args.shards} execution shards in {out}')


if __name__ == '__main__': main()
