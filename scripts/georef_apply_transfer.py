#!/usr/bin/env python3
"""Write inferred corner GCPs + LCC params from georef_transfer.json into
master_dole_v2.csv (only entries whose correlation validated)."""
import json
import csv
import sys
from pathlib import Path

sys.path.insert(0, "/Users/ryanhemenway/archive.aero/scripts")
import dole_v2

SCRATCH = Path("/private/tmp/claude-501/-Users-ryanhemenway-archive-aero/979396e4-d26f-44e2-a132-01bc48223900/scratchpad")
CSV = "/Users/ryanhemenway/archive.aero/master_dole_v2.csv"

with open(SCRATCH / "georef_transfer2.json") as f:
    transfer = json.load(f)

rows = dole_v2.load_rows(CSV)
by_fn = {}
for r in rows:
    by_fn.setdefault(r["filename"], r)

applied, skipped = [], []
for fn, e in transfer.items():
    if e["status"] != "ok":
        if e["status"] not in ("zip-covered", "already-georeferenced"):
            skipped.append((fn, e["status"]))
        continue
    row = by_fn.get(fn)
    if row is None:
        skipped.append((fn, "row-not-found"))
        continue
    for i, g in enumerate(e["gcps"], start=1):
        row[f"gcp{i}_px"] = str(g["px"])
        row[f"gcp{i}_py"] = str(g["py"])
        row[f"gcp{i}_lat"] = str(g["lat"])
        row[f"gcp{i}_lon"] = str(g["lon"])
    lat1, lat2, lat0, lon0 = e["lcc"]
    row["lcc_lat1"] = str(lat1)
    row["lcc_lat2"] = str(lat2)
    row["lcc_lat0"] = str(lat0)
    row["lcc_lon0"] = str(lon0)
    sib = Path(e["sibling"]).parent.name
    row["note"] = (row["note"] + "; " if row["note"] else "") + (
        f"GEOREF-INFERRED 2026-07-22 from {sib} via NCC similarity fit "
        f"(scale={e['scale']}, T=({e['T'][0]},{e['T'][1]})px, {e['inliers']}/{e['windows']} windows, "
        f"max resid {e['resid_max']}px): source carries no usable georeferencing; corner GCPs assume "
        f"the standing 300-dpi conversion grid ({e['dims'][0]}x{e['dims'][1]})")
    applied.append(fn)

with open(CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=dole_v2.V2_FIELDS)
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in dole_v2.V2_FIELDS})

print(f"applied GCPs to {len(applied)} rows")
for fn, why in skipped:
    print(f"  SKIPPED {why}: {fn[:90]}")
