"""Stage a next/ manifest build for the eras uploaded to charts-beta so far."""
import json, os, sys, shutil
from pathlib import Path
H = Path("/Volumes/projects/slicer-runs/2026-10-03_reslice/hashed")
OUT = Path.home() / "archive-next-build" / "beta"
stage, work = OUT / "stage", OUT / "work"
if stage.exists(): shutil.rmtree(stage)
(stage / "sectionals").mkdir(parents=True); (stage / "airspace").mkdir(); work.mkdir(exist_ok=True)
recs = [json.loads(l) for l in open(H / "plan.jsonl") if l.strip()]
latest = {}
for r in recs: latest[r["old"]] = r          # a re-converted era supersedes its earlier upload
eras = [r for r in latest.values() if "/chart/" not in r["old"]]
for r in eras:
    with open(H / "stubs" / r["new"], "rb") as f: head = f.read(r["stub_bytes"])
    assert len(head) == r["stub_bytes"], r["new"]
    with open(stage / r["new"], "wb") as f: f.write(head); f.truncate(r["size"])
air = next((Path.home() / "archive-next-build/live/airspace").glob("*.pmtiles"))
os.symlink(air, stage / "airspace" / air.name)
# Production's vector basemap, under the content-hashed key it has in charts-beta (a server-side copy of
# charts/basemap/protomaps-20260826.pmtiles; the 12 hex are the first of the local mirror's sha256).
BASEMAP = (Path("/Volumes/projects/protomaps_basemap/protomaps-20260826.pmtiles"), "protomaps-20260826.0938e8f9d55b.pmtiles")
(stage / "basemap").mkdir(); os.symlink(BASEMAP[0], stage / "basemap" / BASEMAP[1])
(work / "plan.json").write_text(json.dumps(list(latest.values())))
old = json.loads(Path("/Volumes/projects/archive-next/next/manifest.815c12fba3f7.json").read_text())
(work / "coverage.json").write_text(json.dumps(old["coverage"]))
# timeline: keep "View alone" only for charts whose artifact is in the beta bucket
tl = json.loads(Path("/Users/ryanhemenway/archive.aero/timeline_data.json").read_text())
have = set(latest); kept = dropped = 0
for loc in tl["locations"].values():
    for c in loc.get("charts", []):
        if c.get("pm"):
            keys = c["pm"] if isinstance(c["pm"], list) else [c["pm"]]
            if all(("sectionals/" + k) in have for k in keys): kept += 1
            else: del c["pm"]; dropped += 1
(work / "timeline.json").write_text(json.dumps(tl))
ks = sorted(r["old"][11:-8] for r in eras)
print(f"{len(eras)} eras staged ({ks[0]} .. {ks[-1]}); chart artifacts kept {kept}, not yet uploaded {dropped}; airspace {air.name}")
