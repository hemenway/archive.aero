#!/usr/bin/env python
"""Side-by-side visual verification of candidate-vs-holdings duplicate calls.

Joins _dedupe_vs_holdings.csv (dup verdicts) with _analysis.csv (candidate
paths), resolves each row's `how` string to an actual file under
/Volumes/projects/rawtiffs, and serves a two-pane review UI.

Decisions land in /Volumes/projects/rawtiffcandidates/_visual_verify.csv.

Run:  ~/venv/bin/python scripts/verify_dupes_app.py [--stats] [--port 8765]

Keys in the UI:
  y / d  confirm duplicate      n  not a duplicate      u  unsure
  <- ->  prev / next            space  next undecided    m  cycle alt matches
  s      toggle pan/zoom sync   f / 0  fit               dbl-click  full-res crop
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_file, abort

from osgeo import gdal

gdal.UseExceptions()

CAND_DIR = Path("/Volumes/projects/rawtiffcandidates")
RAWTIFFS = Path("/Volumes/projects/rawtiffs")
DOLE_CSV = Path.home() / "archive.aero" / "master_dole_v2.csv"
DEDUPE_CSV = CAND_DIR / "_dedupe_vs_holdings.csv"
ANALYSIS_CSV = CAND_DIR / "_analysis.csv"
DECISIONS_CSV = CAND_DIR / "_visual_verify.csv"
SIM_CSV = CAND_DIR / "_visual_similarity.csv"
CACHE_DIR = CAND_DIR / "_verify_cache"

THUMB_WIDTH = 2600
CROP_W, CROP_H = 1500, 1050

IMG_EXTS = {".tif", ".tiff", ".pdf", ".png", ".jpg", ".jpeg"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ---------------------------------------------------------------- resolution

class RawtiffsIndex:
    """Lookup tables over /Volumes/projects/rawtiffs down to depth 3."""

    def __init__(self):
        self.files = {}      # basename lower -> [paths]
        self.files_norm = {} # norm(basename) -> [paths]
        self.dirs = {}       # dirname lower -> [paths]
        self.zips = {}       # zipname lower -> [paths]
        root_depth = len(RAWTIFFS.parts)
        for cur, dirnames, filenames in os.walk(RAWTIFFS):
            depth = len(Path(cur).parts) - root_depth
            if depth >= 3:
                dirnames[:] = []
                continue
            for d in dirnames:
                self.dirs.setdefault(d.lower(), []).append(Path(cur) / d)
            for f in filenames:
                p = Path(cur) / f
                ext = p.suffix.lower()
                if ext in IMG_EXTS:
                    self.files.setdefault(f.lower(), []).append(p)
                    self.files_norm.setdefault(norm(f), []).append(p)
                elif ext == ".zip":
                    self.zips.setdefault(f.lower(), []).append(p)

    def lookup_file(self, name):
        hits = self.files.get(name.lower())
        if hits:
            return list(hits)
        return list(self.files_norm.get(norm(name), []))


def dir_images(d):
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    imgs.sort(key=lambda p: (p.suffix.lower() not in (".tif", ".tiff"), p.name))
    return imgs


def load_dole_rows():
    with open(DOLE_CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dole_filename_to_paths(filename, index):
    """Map a dole catalog filename to candidate on-disk paths in rawtiffs."""
    if not filename:
        return []
    if filename.lower().endswith(".zip"):
        stem = filename[:-4]
        hits = []
        for d in index.dirs.get(stem.lower(), []):
            hits.extend(dir_images(d))
        if hits:
            return hits
        # not extracted: fall back to the zip itself (rendered via inner tif)
        return list(index.zips.get(filename.lower(), []))
    return index.lookup_file(filename)


def resolve_how(how, index, dole_rows):
    """Return list of Path matches for a dedupe `how` string (best first)."""
    m = re.match(r"^rawtiffs cycle (\S+) has (.+)$", how)
    if m:
        cyc_dir = RAWTIFFS / m.group(1)
        want = norm(m.group(2))
        if cyc_dir.is_dir():
            hits = [p for p in sorted(cyc_dir.iterdir())
                    if p.suffix.lower() in IMG_EXTS and want in norm(p.name)]
            return hits
        return []

    m = re.match(r"^rawtiffs (.+)$", how)
    if m:
        return index.lookup_file(m.group(1))

    m = re.match(r"^basename (\S+) in rawtiffs", how)
    if m:
        return index.lookup_file(m.group(1))

    m = re.match(r"^dole sectional_files (.+) ed (\S+)$", how)
    if m:
        loc, ed = m.group(1), m.group(2)
        hits = []
        for r in dole_rows:
            if ("sectional_files" in (r.get("filename") or "")
                    and norm(r.get("location") or "") == norm(loc)
                    and (r.get("edition") or "").strip() == ed):
                hits.extend(dole_filename_to_paths(r["filename"], index))
        return hits

    m = re.match(r"^dole location/date (.+) (\S+)$", how)
    if m:
        loc, date = m.group(1), m.group(2)
        hits = []
        for r in dole_rows:
            if (norm(r.get("location") or "") == norm(loc)
                    and (r.get("date") or "").strip() == date):
                hits.extend(dole_filename_to_paths(r["filename"], index))
        if not hits:  # e.g. "western aleutian islands" vs "... Islands East/West"
            for r in dole_rows:
                if (norm(r.get("location") or "").startswith(norm(loc))
                        and (r.get("date") or "").strip() == date):
                    hits.extend(dole_filename_to_paths(r["filename"], index))
        return hits

    return []


def match_confidence(how):
    """How strong was the dedupe call? 1 = weakest (review first)."""
    if how.startswith("dole location/date"):
        return 1, "matched on loc+date only"
    if how.startswith("rawtiffs cycle"):
        return 2, "name found in cycle dir"
    if how.startswith("dole sectional_files"):
        return 2, "matched on loc+edition"
    return 3, "exact filename match"


def build_items():
    with open(ANALYSIS_CSV, newline="") as f:
        ana = {r["rid"]: r for r in csv.DictReader(f)}
    with open(DEDUPE_CSV, newline="") as f:
        ded = list(csv.DictReader(f))

    index = RawtiffsIndex()
    dole_rows = load_dole_rows()

    items = []
    for r in ded:
        if r["dup"] == "new":
            continue
        a = ana.get(r["rid"], {})
        cand_path = a.get("path") or ""
        matches = resolve_how(r["how"], index, dole_rows)
        # de-dup match list, keep order
        seen, uniq = set(), []
        for p in matches:
            if str(p) not in seen:
                seen.add(str(p))
                uniq.append(p)
        conf, conf_label = match_confidence(r["how"])
        items.append({
            "rid": r["rid"],
            "name": r["name"],
            "source": r["source"],
            "dup": r["dup"],
            "how": r["how"],
            "conf": conf,
            "conf_label": conf_label,
            "cand": cand_path,
            "cand_w": a.get("w") or "",
            "cand_h": a.get("h") or "",
            "cand_bytes": a.get("bytes") or "",
            "matches": [str(p) for p in uniq],
        })
    items.sort(key=lambda i: i["rid"])
    return items


# ---------------------------------------------------------------- decisions

_dec_lock = threading.Lock()


def load_decisions():
    if not DECISIONS_CSV.exists():
        return {}
    with open(DECISIONS_CSV, newline="") as f:
        return {r["rid"]: r for r in csv.DictReader(f)}


def save_decisions(decisions):
    fields = ["rid", "name", "dup", "how", "decision", "match_path", "note",
              "decided_at"]
    fd, tmp = tempfile.mkstemp(dir=str(CAND_DIR), suffix=".csv")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rid in sorted(decisions):
            w.writerow(decisions[rid])
    os.replace(tmp, DECISIONS_CSV)


# ---------------------------------------------------------------- rendering

_render_locks = {}
_render_locks_guard = threading.Lock()


def _lock_for(key):
    with _render_locks_guard:
        return _render_locks.setdefault(key, threading.Lock())


def cache_key(path, tag):
    h = hashlib.sha1(str(path).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{h}_{tag}.jpg"


def _zip_inner_tif(zip_path):
    """Extract the first tif inside a candidate zip into the cache dir."""
    out_dir = CACHE_DIR / ("zip_" + hashlib.sha1(str(zip_path).encode()).hexdigest()[:12])
    existing = list(out_dir.glob("*")) if out_dir.is_dir() else []
    if existing:
        return existing[0]
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist()
                 if Path(n).suffix.lower() in (".tif", ".tiff")]
        if not names:
            names = [n for n in z.namelist()
                     if Path(n).suffix.lower() in IMG_EXTS]
        if not names:
            raise RuntimeError(f"no image inside {zip_path}")
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / Path(names[0]).name
        tmp = target.with_name(f".{os.getpid()}.{target.name}")
        with z.open(names[0]) as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        os.replace(tmp, target)
        return target


def _effective_source(path):
    p = Path(path)
    if p.suffix.lower() == ".zip":
        return _zip_inner_tif(p)
    return p


def _tool(name):
    for p in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if Path(p).exists():
            return p
    return name


PDFINFO = _tool("pdfinfo")
PDFTOPPM = _tool("pdftoppm")

# clean env: host-app DYLD/venv vars can crash poppler's dylib loading
_CLEAN_ENV = {k: v for k, v in os.environ.items()
              if not k.startswith(("DYLD_", "PYTHON", "VIRTUAL_ENV"))}
_CLEAN_ENV["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"


def _pdf_page_pts(pdf_path):
    out = subprocess.run([PDFINFO, str(pdf_path)], capture_output=True,
                         text=True, check=True, env=_CLEAN_ENV).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", out)
    if not m:
        raise RuntimeError(f"pdfinfo gave no page size for {pdf_path}")
    return float(m.group(1)), float(m.group(2))


def _pdftoppm(pdf_path, args, out_prefix):
    subprocess.run([PDFTOPPM, "-jpeg", "-f", "1", "-l", "1", *args,
                    str(pdf_path), out_prefix],
                   check=True, capture_output=True, env=_CLEAN_ENV)
    hits = sorted(Path(out_prefix).parent.glob(Path(out_prefix).name + "-*.jpg"))
    if not hits:
        raise RuntimeError("pdftoppm produced no output")
    return hits[0]


def _gdal_translate_jpeg(src_path, out_path, width=None, src_win=None):
    ds = gdal.Open(str(src_path))
    band1 = ds.GetRasterBand(1)
    opts = {
        "format": "JPEG",
        "creationOptions": ["QUALITY=87"],
    }
    if band1.GetColorTable() is not None:
        opts["rgbExpand"] = "rgb"
    elif ds.RasterCount >= 3:
        opts["bandList"] = [1, 2, 3]
    else:
        opts["bandList"] = [1]
    if band1.DataType != gdal.GDT_Byte:
        opts["outputType"] = gdal.GDT_Byte
        opts["scaleParams"] = [[]]
    if width:
        opts["width"] = min(width, ds.RasterXSize)
    if src_win:
        opts["srcWin"] = src_win
    gdal.Translate(str(out_path), ds, **opts)
    return ds.RasterXSize, ds.RasterYSize


def render_thumb(path):
    """Render a fit-to-screen JPEG preview; returns (jpg_path, meta_dict)."""
    out = cache_key(path, f"w{THUMB_WIDTH}")
    meta_path = out.with_suffix(".json")
    if out.exists() and meta_path.exists():
        return out, json.loads(meta_path.read_text())
    with _lock_for(str(out)):
        if out.exists() and meta_path.exists():
            return out, json.loads(meta_path.read_text())
        src = _effective_source(path)
        tmp_out = out.with_suffix(f".{os.getpid()}.tmp.jpg")
        if src.suffix.lower() == ".pdf":
            prefix = str(out.with_suffix("")) + f"_pp{os.getpid()}"
            jpg = _pdftoppm(src, ["-scale-to", str(THUMB_WIDTH)], prefix)
            os.replace(jpg, tmp_out)
            wpts, hpts = _pdf_page_pts(src)
            meta = {"kind": "pdf", "w_pts": wpts, "h_pts": hpts,
                    "src_w": round(wpts / 72 * 300), "src_h": round(hpts / 72 * 300)}
        else:
            w, h = _gdal_translate_jpeg(src, tmp_out, width=THUMB_WIDTH)
            meta = {"kind": "raster", "src_w": w, "src_h": h}
        os.replace(tmp_out, out)
        for junk in out.parent.glob(out.stem + "*.aux.xml"):
            junk.unlink(missing_ok=True)
        meta_path.write_text(json.dumps(meta))
        return out, meta


def render_crop(path, fx, fy):
    """Full-resolution crop around fractional point (fx, fy)."""
    tag = f"crop{round(fx * 1000):04d}x{round(fy * 1000):04d}"
    out = cache_key(path, tag)
    if out.exists():
        return out
    with _lock_for(str(out)):
        if out.exists():
            return out
        src = _effective_source(path)
        tmp_out = out.with_suffix(f".{os.getpid()}.tmp.jpg")
        if src.suffix.lower() == ".pdf":
            wpts, hpts = _pdf_page_pts(src)
            full_w, full_h = wpts / 72 * 300, hpts / 72 * 300
            x = max(0, min(round(fx * full_w - CROP_W / 2), round(full_w - CROP_W)))
            y = max(0, min(round(fy * full_h - CROP_H / 2), round(full_h - CROP_H)))
            prefix = str(out.with_suffix("")) + f"_pp{os.getpid()}"
            jpg = _pdftoppm(src, ["-r", "300", "-x", str(x), "-y", str(y),
                                  "-W", str(CROP_W), "-H", str(CROP_H)], prefix)
            os.replace(jpg, tmp_out)
        else:
            ds = gdal.Open(str(src))
            full_w, full_h = ds.RasterXSize, ds.RasterYSize
            ds = None
            w, h = min(CROP_W, full_w), min(CROP_H, full_h)
            x = max(0, min(round(fx * full_w - w / 2), full_w - w))
            y = max(0, min(round(fy * full_h - h / 2), full_h - h))
            _gdal_translate_jpeg(src, tmp_out, src_win=[x, y, w, h])
        os.replace(tmp_out, out)
        for junk in out.parent.glob(out.stem + "*.aux.xml"):
            junk.unlink(missing_ok=True)
        return out


# ---------------------------------------------------------------- similarity

def load_sim():
    if not SIM_CSV.exists():
        return {}
    with open(SIM_CSV, newline="") as f:
        return {r["rid"]: r for r in csv.DictReader(f)}


def _thumb_gray(path, np):
    """256x256 grayscale float array of a file's preview (cached in-process)."""
    key = "gray:" + str(path)
    with _lock_for(key):
        if key in _GRAY_CACHE:
            return _GRAY_CACHE[key]
        jpg, _meta = render_thumb(path)
        mem = f"/vsimem/gray_{threading.get_ident()}.tif"
        ds = gdal.Open(str(jpg))
        small = gdal.Translate(mem, ds, width=256, height=256,
                               resampleAlg="average")
        arr = small.ReadAsArray().astype("float32")
        small = None
        gdal.Unlink(mem)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        _GRAY_CACHE[key] = arr
        return arr


_GRAY_CACHE = {}


def _ncc(a, b, np):
    a = a - a.mean()
    b = b - b.mean()
    d = float((a * a).sum()) ** 0.5 * float((b * b).sum()) ** 0.5
    return float((a * b).sum() / d) if d else 0.0


def pair_similarity(item, np):
    """Best NCC between candidate and any alternate match, any 90° rotation."""
    ca = _thumb_gray(item["cand"], np)
    best, best_alt = -1.0, 0
    for ai, mp in enumerate(item["matches"]):
        try:
            mb = _thumb_gray(mp, np)
        except Exception:
            continue
        s = max(_ncc(np.ascontiguousarray(np.rot90(ca, k)), mb, np)
                for k in range(4))
        if s > best:
            best, best_alt = s, ai
    return best, best_alt


def run_scan(items, workers):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    done = load_sim()
    todo = [it for it in items if it["rid"] not in done and it["matches"]]
    print(f"scan: {len(todo)} pairs to score ({len(done)} already done)", flush=True)

    write_lock = threading.Lock()
    new_file = not SIM_CSV.exists()
    out = open(SIM_CSV, "a", newline="")
    w = csv.DictWriter(out, fieldnames=["rid", "sim", "best_alt"])
    if new_file:
        w.writeheader()
        out.flush()
    counter = [0]

    def one(it):
        try:
            sim, best_alt = pair_similarity(it, np)
        except Exception as e:
            print(f"  {it['rid']} FAILED: {e}", flush=True)
            sim, best_alt = -1.0, 0
        with write_lock:
            w.writerow({"rid": it["rid"], "sim": f"{sim:.4f}",
                        "best_alt": best_alt})
            out.flush()
            counter[0] += 1
            n = counter[0]
        if n % 25 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] scored", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    out.close()

    sims = sorted((float(r["sim"]), rid) for rid, r in load_sim().items())
    print("\nlowest-similarity pairs (most likely NOT duplicates):")
    for s, rid in sims[:30]:
        it = next((i for i in items if i["rid"] == rid), None)
        if it:
            print(f"  {s:6.3f}  {rid}  {it['name']}  <-  {it['how']}")


# ---------------------------------------------------------------- app

app = Flask(__name__)
ITEMS = []
ITEM_BY_RID = {}
DECISIONS = {}


def _side_path(item, side, alt=0):
    if side == "cand":
        return item["cand"]
    alts = item["matches"]
    if not alts:
        return None
    return alts[min(alt, len(alts) - 1)]


@app.get("/api/items")
def api_items():
    sim = load_sim()  # re-read each page load so a running scan shows up
    out = []
    for it in ITEMS:
        d = DECISIONS.get(it["rid"], {})
        row = dict(it)
        row["decision"] = d.get("decision", "")
        row["note"] = d.get("note", "")
        s = sim.get(it["rid"])
        row["sim"] = float(s["sim"]) if s else None
        row["best_alt"] = int(s["best_alt"]) if s else 0
        out.append(row)
    return jsonify(out)


@app.get("/api/img/<rid>/<side>")
def api_img(rid, side):
    it = ITEM_BY_RID.get(rid)
    if not it or side not in ("cand", "match"):
        abort(404)
    path = _side_path(it, side, alt=int(request.args.get("alt", 0)))
    if not path or not Path(path).exists():
        abort(404)
    try:
        jpg, meta = render_thumb(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    resp = send_file(jpg, mimetype="image/jpeg", max_age=86400)
    resp.headers["X-Src-Width"] = str(meta.get("src_w", ""))
    resp.headers["X-Src-Height"] = str(meta.get("src_h", ""))
    return resp


@app.get("/api/crop/<rid>/<side>")
def api_crop(rid, side):
    it = ITEM_BY_RID.get(rid)
    if not it or side not in ("cand", "match"):
        abort(404)
    path = _side_path(it, side, alt=int(request.args.get("alt", 0)))
    if not path or not Path(path).exists():
        abort(404)
    fx = min(1.0, max(0.0, float(request.args.get("fx", 0.5))))
    fy = min(1.0, max(0.0, float(request.args.get("fy", 0.5))))
    try:
        jpg = render_crop(path, fx, fy)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return send_file(jpg, mimetype="image/jpeg", max_age=86400)


@app.post("/api/decide")
def api_decide():
    body = request.get_json(force=True)
    rid = body.get("rid")
    decision = body.get("decision", "")
    it = ITEM_BY_RID.get(rid)
    if not it:
        abort(404)
    if decision not in ("dup", "not_dup", "unsure", ""):
        abort(400)
    with _dec_lock:
        if decision == "":
            DECISIONS.pop(rid, None)
        else:
            DECISIONS[rid] = {
                "rid": rid,
                "name": it["name"],
                "dup": it["dup"],
                "how": it["how"],
                "decision": decision,
                "match_path": _side_path(it, "match", alt=int(body.get("alt", 0))) or "",
                "note": body.get("note", ""),
                "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        save_decisions(DECISIONS)
    return jsonify({"ok": True, "decided": len(DECISIONS), "total": len(ITEMS)})


@app.get("/")
def index_page():
    return PAGE


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>dupe verify</title>
<style>
:root { --bg:#14161a; --panel:#1d2026; --fg:#d6d8de; --dim:#8b8f99;
        --dup:#e05555; --not:#4cae6e; --unsure:#d9a441; --accent:#5b8dd9; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--fg);
       font:13px/1.45 -apple-system, "SF Mono", Menlo, monospace;
       height:100vh; display:flex; flex-direction:column; overflow:hidden; }
#top { display:flex; gap:14px; align-items:center; padding:7px 12px;
       background:var(--panel); border-bottom:1px solid #000; flex:none; }
#top select { background:#2a2e36; color:var(--fg); border:1px solid #3a3f4a;
              border-radius:4px; padding:2px 4px; font:inherit; }
#prog { flex:1; text-align:center; color:var(--dim); }
#prog b { color:var(--fg); }
#bar { height:4px; background:#2a2e36; border-radius:2px; overflow:hidden; margin-top:3px; }
#bar div { height:100%; background:var(--accent); width:0%; }
#panes { flex:1; display:flex; min-height:0; }
.pane { flex:1; display:flex; flex-direction:column; min-width:0; }
.pane + .pane { border-left:1px solid #000; }
.phead { padding:5px 10px; background:var(--panel); flex:none; overflow:hidden;
         white-space:nowrap; text-overflow:ellipsis; border-bottom:1px solid #000; }
.phead .tag { font-weight:700; letter-spacing:.08em; font-size:11px; color:var(--accent); }
.phead .fn { color:var(--fg); }
.phead .sub { color:var(--dim); font-size:11px; overflow:hidden;
              white-space:nowrap; text-overflow:ellipsis; }
.imgbox { flex:1; position:relative; overflow:hidden; background:#0b0c0e; cursor:grab; }
.imgbox img.main { position:absolute; left:0; top:0; transform-origin:0 0;
                   image-rendering:auto; user-select:none; -webkit-user-drag:none; }
.imgbox .msg { position:absolute; inset:0; display:flex; align-items:center;
               justify-content:center; color:var(--dim); padding:20px;
               text-align:center; white-space:pre-wrap; }
#bottom { display:flex; gap:8px; align-items:center; padding:8px 12px;
          background:var(--panel); border-top:1px solid #000; flex:none; }
button { background:#2a2e36; border:1px solid #3a3f4a; color:var(--fg);
         border-radius:5px; padding:5px 14px; font:inherit; cursor:pointer; }
button:hover { border-color:var(--accent); }
button.dup    { border-color:var(--dup);    color:var(--dup); }
button.not    { border-color:var(--not);    color:var(--not); }
button.unsure { border-color:var(--unsure); color:var(--unsure); }
button.active { color:#fff; }
button.dup.active    { background:var(--dup); }
button.not.active    { background:var(--not); }
button.unsure.active { background:var(--unsure); }
#note { flex:1; background:#2a2e36; border:1px solid #3a3f4a; color:var(--fg);
        border-radius:5px; padding:5px 8px; font:inherit; }
#badge { position:fixed; top:52px; right:16px; padding:3px 12px; border-radius:4px;
         font-weight:700; z-index:5; display:none; color:#fff; }
#croplay { position:fixed; inset:0; background:rgba(0,0,0,.82); z-index:10;
           display:none; align-items:center; justify-content:center; cursor:zoom-out; }
#croplay img { max-width:96vw; max-height:94vh; border:1px solid #444; }
kbd { background:#2a2e36; border-radius:3px; padding:0 4px; color:var(--dim); }
#keys { color:var(--dim); font-size:11px; }
.chip { display:inline-block; padding:0 7px; border-radius:8px; font-size:10px;
        font-weight:700; vertical-align:1px; margin-right:4px; color:#14161a; }
.chip.c1 { background:var(--dup); }
.chip.c2 { background:var(--unsure); }
.chip.c3 { background:var(--not); }
</style></head><body>
<div id="top">
  <span style="font-weight:700">dupe&nbsp;verify</span>
  <select id="ftype"><option value="">all types</option>
    <option value="dup_disk">dup_disk</option><option value="dup_dole">dup_dole</option></select>
  <select id="fstat"><option value="">all</option><option value="undecided" selected>undecided</option>
    <option value="dup">dup</option><option value="not_dup">not dup</option>
    <option value="unsure">unsure</option></select>
  <select id="fsort"><option value="weak">weakest match first</option>
    <option value="sim">least similar first</option>
    <option value="rid">rid order</option></select>
  <div id="prog"><span id="ptext"></span><div id="bar"><div id="barfill"></div></div></div>
  <span id="keys"><kbd>y</kbd>dup <kbd>n</kbd>not <kbd>u</kbd>unsure <kbd>&larr;&rarr;</kbd>nav
    <kbd>space</kbd>next-undecided <kbd>m</kbd>alt <kbd>s</kbd>sync <kbd>f</kbd>fit <kbd>dblclk</kbd>HD</span>
</div>
<div id="panes">
  <div class="pane">
    <div class="phead"><span class="tag">CANDIDATE</span> <span class="fn" id="cfn"></span>
      <div class="sub" id="csub"></div></div>
    <div class="imgbox" id="boxC"><img class="main" id="imgC"><div class="msg" id="msgC"></div></div>
  </div>
  <div class="pane">
    <div class="phead"><span class="tag" style="color:var(--unsure)">MATCH</span> <span class="fn" id="mfn"></span>
      <div class="sub" id="msub"></div></div>
    <div class="imgbox" id="boxM"><img class="main" id="imgM"><div class="msg" id="msgM"></div></div>
  </div>
</div>
<div id="bottom">
  <button class="dup" id="bDup" onclick="decide('dup')">Duplicate (y)</button>
  <button class="not" id="bNot" onclick="decide('not_dup')">Not a dup (n)</button>
  <button class="unsure" id="bUns" onclick="decide('unsure')">Unsure (u)</button>
  <input id="note" placeholder="note (saved with decision)">
  <button onclick="nav(-1)">&larr; Prev</button>
  <button onclick="nav(1)">Next &rarr;</button>
</div>
<div id="badge"></div>
<div id="croplay" onclick="this.style.display='none'"><img id="cropimg"></div>
<script>
let items = [], view = [], pos = 0, alt = 0, sync = true;
const V = { z:1, cx:0.5, cy:0.5 };            // shared view state
const P = { C:{el:imgC, box:boxC, iw:0, ih:0, v:null},
            M:{el:imgM, box:boxM, iw:0, ih:0, v:null} };

function cur() { return view[pos]; }
function simval(it) { return typeof it.sim === 'number' ? it.sim : 2; } // unscored last

function applyFilter(keepRid) {
  const t = ftype.value, s = fstat.value;
  view = items.filter(it => (!t || it.dup === t) &&
    (!s || (s === 'undecided' ? !it.decision : it.decision === s)));
  if (!view.length) view = items.slice();
  if (fsort.value === 'weak')
    view.sort((a, b) => (a.conf - b.conf) ||
      ((b.matches.length > 1) - (a.matches.length > 1)) ||
      a.rid.localeCompare(b.rid));
  else if (fsort.value === 'sim')
    view.sort((a, b) => (simval(a) - simval(b)) || a.rid.localeCompare(b.rid));
  let idx = keepRid ? view.findIndex(i => i.rid === keepRid) : -1;
  pos = idx >= 0 ? idx : 0;
}

function progress() {
  const done = items.filter(i => i.decision).length;
  ptext.innerHTML = `<b>${done}</b> / ${items.length} decided &nbsp;&middot;&nbsp; item ${pos+1} of ${view.length} in view`;
  barfill.style.width = (100 * done / items.length) + '%';
}

function fmtBytes(b) { b = +b; if (!b) return '';
  return b > 1e9 ? (b/1e9).toFixed(2)+' GB' : b > 1e6 ? (b/1e6).toFixed(1)+' MB' : Math.round(b/1e3)+' KB'; }

function show() {
  const it = cur(); if (!it) return;
  alt = Math.min(it.best_alt || 0, Math.max(0, it.matches.length - 1));
  cfn.textContent = it.name;
  csub.textContent = `${it.source}  ·  ${it.cand_w}×${it.cand_h}px  ·  ${fmtBytes(it.cand_bytes)}  ·  ${it.cand}`;
  loadSide('C', `/api/img/${it.rid}/cand`, it.cand);
  showMatch(it);
  V.z = 1; V.cx = .5; V.cy = .5; P.C.v = null; P.M.v = null;
  note.value = it.note || '';
  paintDecision();
  progress();
  prefetch();
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;'); }

function showMatch(it) {
  const n = it.matches.length;
  let chip = `<span class="chip c${it.conf}">${esc(it.conf_label)}</span>`;
  if (typeof it.sim === 'number')
    chip += `<span class="chip ${it.sim < 0.5 ? 'c1' : it.sim < 0.8 ? 'c2' : 'c3'}">sim ${it.sim.toFixed(2)}</span>`;
  if (!n) {
    imgM.style.display = 'none'; msgM.textContent = 'match not resolved on disk\\n\\nhow: ' + it.how;
    mfn.textContent = '(unresolved)'; msub.innerHTML = chip + esc(it.how); return;
  }
  const p = it.matches[Math.min(alt, n-1)];
  mfn.textContent = p.split('/').pop();
  msub.innerHTML = chip + esc(`${it.how}  ·  match ${Math.min(alt,n-1)+1}/${n}  ·  ${p}`);
  loadSide('M', `/api/img/${it.rid}/match?alt=${alt}`, p);
}

function loadSide(k, url, path) {
  const s = P[k];
  s.el.style.display = 'none';
  s.box.querySelector('.msg').textContent = 'rendering…';
  const probe = new Image();
  probe.onload = () => {
    if ((k==='C'?cur().cand:path) !== path && k==='C') return;
    s.el.src = probe.src; s.iw = probe.naturalWidth; s.ih = probe.naturalHeight;
    s.el.style.display = ''; s.box.querySelector('.msg').textContent = '';
    layout(k);
  };
  probe.onerror = () => { s.box.querySelector('.msg').textContent = 'failed to render\\n' + path; };
  probe.src = url;
}

function layout(k) {
  const s = P[k]; if (!s.iw) return;
  const st = (!sync && s.v) ? s.v : V;
  const cw = s.box.clientWidth, ch = s.box.clientHeight;
  const fit = Math.min(cw / s.iw, ch / s.ih);
  const sc = fit * st.z;
  const tx = cw/2 - st.cx * s.iw * sc, ty = ch/2 - st.cy * s.ih * sc;
  s.el.style.transform = `translate(${tx}px,${ty}px) scale(${sc})`;
}
function layoutAll() { layout('C'); layout('M'); }

function stateFor(k) { if (sync) return V; const s = P[k];
  if (!s.v) s.v = {...V}; return s.v; }

function bindBox(k) {
  const s = P[k];
  s.box.addEventListener('wheel', e => {
    e.preventDefault(); if (!s.iw) return;
    const st = stateFor(k);
    const cw = s.box.clientWidth, ch = s.box.clientHeight;
    const fit = Math.min(cw/s.iw, ch/s.ih);
    const r = s.box.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const sc = fit * st.z;
    const tx = cw/2 - st.cx*s.iw*sc, ty = ch/2 - st.cy*s.ih*sc;
    const ix = (mx-tx)/sc, iy = (my-ty)/sc;
    st.z = Math.max(1, Math.min(40, st.z * Math.exp(-e.deltaY*0.0015)));
    const sc2 = fit * st.z;
    st.cx = (cw/2 - (mx - ix*sc2)) / (s.iw*sc2);
    st.cy = (ch/2 - (my - iy*sc2)) / (s.ih*sc2);
    layoutAll();
  }, {passive:false});
  let drag = null;
  s.box.addEventListener('mousedown', e => { drag = {x:e.clientX, y:e.clientY}; s.box.style.cursor='grabbing'; });
  window.addEventListener('mousemove', e => {
    if (!drag || !s.iw) return;
    const st = stateFor(k);
    const cw = s.box.clientWidth, ch = s.box.clientHeight;
    const sc = Math.min(cw/s.iw, ch/s.ih) * st.z;
    st.cx -= (e.clientX - drag.x)/(s.iw*sc);
    st.cy -= (e.clientY - drag.y)/(s.ih*sc);
    drag = {x:e.clientX, y:e.clientY};
    layoutAll();
  });
  window.addEventListener('mouseup', () => { if (drag) { drag = null; s.box.style.cursor='grab'; } });
  s.box.addEventListener('dblclick', e => {
    if (!s.iw) return;
    const st = (!sync && s.v) ? s.v : V;
    const cw = s.box.clientWidth, ch = s.box.clientHeight;
    const sc = Math.min(cw/s.iw, ch/s.ih) * st.z;
    const r = s.box.getBoundingClientRect();
    const tx = cw/2 - st.cx*s.iw*sc, ty = ch/2 - st.cy*s.ih*sc;
    const fx = ((e.clientX-r.left)-tx)/sc/s.iw, fy = ((e.clientY-r.top)-ty)/sc/s.ih;
    const it = cur();
    const side = k === 'C' ? 'cand' : 'match';
    cropimg.src = '';
    croplay.style.display = 'flex';
    cropimg.src = `/api/crop/${it.rid}/${side}?fx=${fx.toFixed(3)}&fy=${fy.toFixed(3)}&alt=${alt}`;
  });
}

function paintDecision() {
  const d = cur().decision || '';
  bDup.classList.toggle('active', d==='dup');
  bNot.classList.toggle('active', d==='not_dup');
  bUns.classList.toggle('active', d==='unsure');
  const names = {dup:['DUPLICATE','var(--dup)'], not_dup:['NOT A DUP','var(--not)'], unsure:['UNSURE','var(--unsure)']};
  if (d) { badge.style.display='block'; badge.textContent = names[d][0]; badge.style.background = names[d][1]; }
  else badge.style.display = 'none';
}

async function decide(d) {
  const it = cur();
  const next = it.decision === d ? '' : d;   // click again to clear
  it.decision = next; it.note = note.value;
  paintDecision(); progress();
  await fetch('/api/decide', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({rid: it.rid, decision: next, note: note.value, alt})});
  if (next) setTimeout(() => nextUndecided(), 120);
}

function nav(d) {
  pos = Math.min(view.length-1, Math.max(0, pos + d));
  show();
}
function nextUndecided() {
  const start = pos;
  for (let i = 1; i <= view.length; i++) {
    const j = (start + i) % view.length;
    if (!view[j].decision) { pos = j; show(); return; }
  }
  ptext.innerHTML = '<b>all items in view decided 🎉</b>';
}

function prefetch() {
  for (let i = 1; i <= 3; i++) {
    const it = view[pos + i]; if (!it) break;
    (new Image()).src = `/api/img/${it.rid}/cand`;
    if (it.matches.length) (new Image()).src = `/api/img/${it.rid}/match?alt=0`;
  }
}

document.addEventListener('keydown', e => {
  if (e.target === note) { if (e.key === 'Escape') note.blur(); return; }
  if (croplay.style.display === 'flex' && (e.key === 'Escape' || e.key === ' ')) {
    croplay.style.display = 'none'; e.preventDefault(); return; }
  switch (e.key) {
    case 'y': case 'd': decide('dup'); break;
    case 'n': decide('not_dup'); break;
    case 'u': decide('unsure'); break;
    case 'ArrowLeft': nav(-1); break;
    case 'ArrowRight': nav(1); break;
    case ' ': e.preventDefault(); nextUndecided(); break;
    case 'm': { const it = cur(); if (it.matches.length > 1) {
        alt = (alt + 1) % it.matches.length; showMatch(it); } break; }
    case 's': sync = !sync; P.C.v = sync ? null : {...V}; P.M.v = sync ? null : {...V};
      layoutAll(); break;
    case 'f': case '0': V.z = 1; V.cx = .5; V.cy = .5; P.C.v = null; P.M.v = null; layoutAll(); break;
  }
});

ftype.onchange = fstat.onchange = fsort.onchange = () => { applyFilter(cur() && cur().rid); show(); };
window.onresize = layoutAll;
bindBox('C'); bindBox('M');

fetch('/api/items').then(r => r.json()).then(d => {
  items = d;
  if (items.some(it => typeof it.sim === 'number')) fsort.value = 'sim';
  applyFilter();
  // start at first undecided
  const i = view.findIndex(it => !it.decision);
  if (i >= 0) pos = i;
  show();
});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--stats", action="store_true",
                    help="print match-resolution stats and exit")
    ap.add_argument("--scan", action="store_true",
                    help="batch-score visual similarity of all pairs and exit")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    global ITEMS, ITEM_BY_RID, DECISIONS
    print("building rawtiffs index + resolving matches…", flush=True)
    ITEMS = build_items()
    ITEM_BY_RID = {i["rid"]: i for i in ITEMS}
    DECISIONS = load_decisions()

    unresolved = [i for i in ITEMS if not i["matches"]]
    multi = [i for i in ITEMS if len(i["matches"]) > 1]
    print(f"{len(ITEMS)} dup pairs | {len(unresolved)} unresolved matches | "
          f"{len(multi)} with multiple match files | {len(DECISIONS)} already decided")
    if unresolved:
        from collections import Counter
        pat = Counter(re.sub(r"\d", "#", i['how'].split()[0]) for i in unresolved)
        print("unresolved by how-prefix:", dict(pat))
    if args.stats:
        for i in unresolved[:40]:
            print("  UNRESOLVED", i["rid"], i["how"])
        return

    CACHE_DIR.mkdir(exist_ok=True)
    if args.scan:
        run_scan(ITEMS, args.workers)
        return
    print(f"\n  http://127.0.0.1:{args.port}\n", flush=True)
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
