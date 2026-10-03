"""archive-slicer engine: job store, scheduler, job handlers, host stats.

Jobs run one at a time per lane - `cpu` (slice, convert, script) and `io`
(archive, publish) - so one era's archive copy or upload overlaps the next
era's slice. Each job's commands run as subprocesses in their own process
group (cancel kills the tree); everything they print lands in
/state/logs/<job id>.log, which is also copied into the run's logs/ dir.

Storage (container paths; see compose.yml for the host side):
  /data/rawtiffs   rawtiffs, plain read-only bind (never written, by anyone here)
  /data/sources    overlay: rawtiffs as the untouched lower layer + an SSD upper
                   layer; the slicer reads its sources here, and anything it
                   writes into the source tree (unzips, 16-bit/PDF conversions,
                   downloads) lands in the upper layer, never in rawtiffs
  /data/src-upper  read-only view of that upper layer (what the slicer wrote)
  /work            NVMe scratch: warp temps, fresh mosaics + pmtiles
  /runs            HDD archive (projects/slicer-runs): finished outputs + logs
  /code            code releases synced from the Mac (current -> releases/<id>)
  /state           job DB, job logs
"""
import functools
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import sqlite3

E = os.environ.get
WORK = Path(E("SLICERD_WORK", "/work"))
RUNS = Path(E("SLICERD_RUNS", "/runs"))
STATE = Path(E("SLICERD_STATE", "/state"))
CODE = Path(E("SLICERD_CODE", "/code"))
SOURCES = Path(E("SLICERD_SOURCES", "/data/sources"))
RAW = Path(E("SLICERD_RAW", "/data/rawtiffs"))
UPPER = Path(E("SLICERD_UPPER", "/data/src-upper"))
G2P = E("SLICERD_G2P", "/usr/local/bin/geotiff2pmtiles")
PY = E("SLICERD_PYTHON", "/opt/venv/bin/python")
APP = Path(__file__).resolve().parent
G2P_CONCURRENCY = int(E("SLICERD_G2P_CONCURRENCY", "12"))
G2P_MEM_MB = int(E("SLICERD_G2P_MEM_MB", "20000"))
MIN_FREE_SLICE_GB = float(E("SLICERD_MIN_FREE_SLICE_GB", "90"))
MIN_FREE_CONVERT_GB = float(E("SLICERD_MIN_FREE_CONVERT_GB", "12"))
ABORT_FREE_GB = float(E("SLICERD_ABORT_FREE_GB", "6"))
REMOTE_PREFIX = E("SLICERD_REMOTE_PREFIX", "r2:charts/sectionals/")
ALIGN_TOL = float(E("SLICERD_ALIGN_TOL", "0.3"))
LANES = {"cpu": int(E("SLICERD_CPU_LANE", "1")), "io": int(E("SLICERD_IO_LANE", "1"))}
ORIG = 20037508.342789244

KEY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(_to_\d{4}-\d{2}-\d{2})?$")
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
JOB_TYPES = {"slice": "cpu", "convert": "cpu", "script": "cpu", "archive": "io", "publish": "io"}
SCRIPTS = {"slicer.py", "build_metadata_bundle.py", "publish_chart_pmtiles.py"}
DONE = ("succeeded", "failed", "cancelled", "interrupted")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def gb(n):
    return round(n / 1e9, 2)


class Cancelled(Exception):
    pass


@functools.lru_cache(maxsize=1)
def converter_id():
    """Identity of the bundled geotiff2pmtiles: base commit + patch + binary hash."""
    import hashlib
    h = hashlib.sha256(Path(G2P).read_bytes()).hexdigest()[:8]
    return f"b300c9e+halfpx linux-amd64 sha256:{h}"


# ---------------------------------------------------------------- paths

def check_run(run):
    if not run or not RUN_RE.match(run):
        raise ValueError(f"bad run name {run!r}: letters, digits, . _ - (max 64), no spaces")
    return run


def check_key(key):
    if not key or not KEY_RE.match(key):
        raise ValueError(f"bad era key {key!r}: expected YYYY-MM-DD_to_YYYY-MM-DD")
    return key


def work_run(run):
    return WORK / "runs" / check_run(run)


def hdd_run(run):
    return RUNS / check_run(run)


def locate(run, sub, name):
    """SSD copy first, then the HDD archive; None when neither exists."""
    for base in (work_run(run), hdd_run(run)):
        p = base / sub / name
        if p.exists():
            return p
    return None


def releases():
    d = CODE / "releases"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def current_release():
    cur = CODE / "current"
    if not cur.exists():
        return None, {}
    rid = os.path.basename(os.readlink(cur)) if cur.is_symlink() else cur.resolve().name
    meta = {}
    try:
        meta = json.loads((CODE / "releases" / rid / "RELEASE.json").read_text())
    except Exception:
        pass
    return rid, meta


def release_dir(rid):
    p = CODE / "releases" / rid
    if not (p / "scripts" / "slicer.py").exists():
        raise RuntimeError(f"code release {rid} has no scripts/slicer.py")
    return p


def era_keys(rid=None):
    rid = rid or current_release()[0]
    if not rid:
        return []
    with open(CODE / "releases" / rid / "dates.csv") as f:
        next(f)
        return sorted({line.split(",", 1)[0].strip() for line in f if line.strip()})


def grid_zoom(path):
    """Zoom whose 256-px tile pixel a mosaic sits on (None if off-grid)."""
    from osgeo import gdal
    gdal.UseExceptions()
    gt = gdal.Open(str(path)).GetGeoTransform()
    zr = round(math.log2(2 * ORIG / 256 / gt[1]))
    res = 2 * ORIG / 256 / 2 ** zr
    ok = abs(gt[1] / res - 1) < 1e-9 and abs(-gt[5] / res - 1) < 1e-9
    return zr if ok else None


def pm_header(path):
    from pmtiles.reader import MmapSource, Reader
    with open(path, "rb") as f:
        h = Reader(MmapSource(f)).header()
    return {"min_zoom": h["min_zoom"], "max_zoom": h["max_zoom"], "tiles": h["addressed_tiles_count"],
            "bounds": [h["min_lon_e7"] / 1e7, h["min_lat_e7"] / 1e7, h["max_lon_e7"] / 1e7, h["max_lat_e7"] / 1e7]}


# ---------------------------------------------------------------- overlay

def _is_whiteout(entry):
    try:
        st = entry.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISCHR(st.st_mode) and st.st_rdev == 0


def overlay_diff(limit=20):
    """Entries where the overlay's view disagrees with rawtiffs itself.

    The overlay caches directory listings; files added to (or removed from)
    rawtiffs over SMB while it is mounted may not show through until it is
    remounted. Whiteouts (sources the slicer renamed or replaced) and the
    slicer's own additions in the upper layer are expected and excluded.
    """
    diffs = []
    for root, dirs, files in os.walk(RAW):
        rel = os.path.relpath(root, RAW)
        ov = SOURCES if rel == "." else SOURCES / rel
        up = UPPER if rel == "." else UPPER / rel
        try:
            up_entries = {e.name: e for e in os.scandir(up)}
        except (FileNotFoundError, NotADirectoryError):
            up_entries = {}
        whiteouts = {n for n, e in up_entries.items() if _is_whiteout(e)}
        try:
            ov_names = set(os.listdir(ov))
        except (FileNotFoundError, NotADirectoryError):
            ov_names = set()
        raw_names = set(dirs) | set(files)
        for n in sorted(raw_names - ov_names - whiteouts):
            diffs.append(f"missing in overlay: {os.path.normpath(os.path.join(rel, n))}")
        # Listed but not resolvable: a cached "not found" from before the file arrived.
        for n in sorted((raw_names & ov_names) - set(up_entries)):
            try:
                os.lstat(ov / n)
            except FileNotFoundError:
                diffs.append(f"unresolvable in overlay: {os.path.normpath(os.path.join(rel, n))}")
        for n in sorted(ov_names - raw_names - set(up_entries)):
            diffs.append(f"stale in overlay: {os.path.normpath(os.path.join(rel, n))}")
        if len(diffs) >= limit:
            break
        dirs[:] = [d for d in dirs if d not in whiteouts]
    return diffs[:limit]


def overlay_writes(limit=500):
    """What the slicer has written into the source tree (the overlay's upper layer)."""
    out, total, n = [], 0, 0
    for root, dirs, files in os.walk(UPPER):
        for name in dirs + files:
            p = Path(root) / name
            try:
                st = p.lstat()
            except OSError:
                continue
            rel = str(p.relative_to(UPPER))
            if stat.S_ISCHR(st.st_mode) and st.st_rdev == 0:
                kind = "whiteout (hides the rawtiffs original in the slicer's view)"
            elif stat.S_ISDIR(st.st_mode):
                continue
            else:
                kind = "copied-up (modified)" if (RAW / rel).exists() else "new"
                total += st.st_size
            n += 1
            if len(out) < limit:
                out.append({"path": rel, "kind": kind, "size_gb": gb(st.st_size),
                            "mtime": time.strftime("%F %T", time.localtime(st.st_mtime))})
    return {"entries": n, "total_gb": gb(total), "items": out,
            "note": "These live only in the SSD upper layer (/srv/archive-slicer/src-upper on the host); "
                    "rawtiffs is unchanged. Promoting any of them into rawtiffs is a manual decision."}


# ---------------------------------------------------------------- host stats

def _read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def host_stats():
    mem = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        mem[k] = int(v.split()[0]) * 1024 if v.split() else 0
    cg = {}
    for k in ("memory.current", "memory.max", "memory.peak"):
        v = _read(f"/sys/fs/cgroup/{k}")
        cg[k] = v if v == "max" or v is None else gb(int(v))
    cgstat = {}
    for line in (_read("/sys/fs/cgroup/memory.stat") or "").splitlines():
        k, _, v = line.partition(" ")
        if k in ("anon", "file"):
            cgstat[k] = gb(int(v))
    temps = {}
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        if _read(hw / "name") == "coretemp":
            for t in sorted(hw.glob("temp*_input")):
                label = _read(str(t).replace("_input", "_label"), t.name)
                temps[label] = int(_read(t, "0")) / 1000
    mhz = [float(line.split(":")[1]) for line in (_read("/proc/cpuinfo") or "").splitlines()
           if line.startswith("cpu MHz")]
    disks = {}
    for name, p in (("ssd_work", WORK), ("hdd_runs", RUNS), ("rawtiffs", RAW)):
        try:
            s = os.statvfs(p)
            disks[name] = {"free_gb": gb(s.f_bavail * s.f_frsize), "size_gb": gb(s.f_blocks * s.f_frsize)}
        except OSError as e:
            disks[name] = {"error": str(e)}
    up = float((_read("/proc/uptime") or "0").split()[0])
    return {
        "load_avg": (_read("/proc/loadavg") or "").split()[:3],
        "cpus": os.cpu_count(),
        "cpu_mhz_avg": round(sum(mhz) / len(mhz)) if mhz else None,
        "cpu_temps_c": temps,
        "host_mem_gb": {"total": gb(mem.get("MemTotal", 0)), "available": gb(mem.get("MemAvailable", 0)),
                        "cached": gb(mem.get("Cached", 0))},
        "container_mem_gb": {"used": cg["memory.current"], "limit": cg["memory.max"],
                             "peak": cg["memory.peak"], **cgstat},
        "disks": disks,
        "host_uptime_h": round(up / 3600, 1),
    }


# ---------------------------------------------------------------- store

class Store:
    COLS = ("id", "type", "lane", "status", "run", "key", "params", "result", "error", "depends_on",
            "release", "created", "started", "finished", "pid")

    def __init__(self, path):
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, type TEXT, lane TEXT, status TEXT, run TEXT, key TEXT,
            params TEXT, result TEXT, error TEXT, depends_on TEXT, release TEXT,
            created TEXT, started TEXT, finished TEXT, pid INTEGER)""")

    def q(self, sql, *args):
        with self.lock:
            rows = [dict(r) for r in self.db.execute(sql, args).fetchall()]
        for r in rows:
            for k in ("params", "result"):
                if r.get(k):
                    r[k] = json.loads(r[k])
        return rows

    def x(self, sql, *args):
        with self.lock:
            self.db.execute(sql, args)

    def get(self, jid):
        r = self.q("SELECT * FROM jobs WHERE id=?", jid)
        return r[0] if r else None

    def update(self, jid, **kw):
        for k in ("params", "result"):
            if k in kw and kw[k] is not None:
                kw[k] = json.dumps(kw[k])
        sets = ", ".join(f"{k}=?" for k in kw)
        self.x(f"UPDATE jobs SET {sets} WHERE id=?", *kw.values(), jid)


# ---------------------------------------------------------------- job context

class Ctx:
    def __init__(self, engine, job):
        self.engine, self.job, self.id = engine, job, job["id"]
        self.params = job["params"] or {}
        self.log_path = STATE / "logs" / f"{self.id}.log"
        self.fh = open(self.log_path, "ab", buffering=0)
        self.cancel = threading.Event()
        self.abort_reason = None
        self.rel = release_dir(job["release"]) if job["release"] else None

    def log(self, msg):
        self.fh.write(f"[{time.strftime('%F %T')}] {msg}\n".encode())

    def check(self):
        if self.cancel.is_set():
            raise Cancelled(self.abort_reason or "cancelled")

    def run(self, args, cwd=None, env=None):
        """Run a command, output appended to the job log. Returns (rc, its output)."""
        self.check()
        args = [str(a) for a in args]
        self.log("$ " + shlex.join(args))
        start = self.log_path.stat().st_size
        full_env = dict(os.environ)
        full_env.update(env or {})
        p = subprocess.Popen(args, cwd=cwd, env=full_env, stdout=self.fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        self.engine.store.update(self.id, pid=p.pid)
        try:
            while p.poll() is None:
                if self.cancel.wait(2):
                    self._kill(p)
        finally:
            self.engine.store.update(self.id, pid=None)
        with open(self.log_path, "rb") as f:
            f.seek(start)
            out = f.read().decode(errors="replace")
        self.log(f"exit {p.returncode}")
        self.check()
        return p.returncode, out

    def _kill(self, p):
        self.log(f"cancelling: {self.abort_reason or 'requested'} — SIGTERM to process group {p.pid}")
        try:
            os.killpg(p.pid, 15)
            p.wait(20)
        except subprocess.TimeoutExpired:
            self.log("still running after 20 s — SIGKILL")
            os.killpg(p.pid, 9)
            p.wait()
        except ProcessLookupError:
            pass

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------- handlers

def copy_file(ctx, src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    size, done, t0, last = src.stat().st_size, 0, time.time(), time.time()
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while True:
            ctx.check()
            buf = fi.read(64 << 20)
            if not buf:
                break
            fo.write(buf)
            done += len(buf)
            if time.time() - last > 30:
                ctx.log(f"  {dst.name}: {done / max(size, 1):.0%} of {gb(size)} GB "
                        f"({done / 1e6 / (time.time() - t0):.0f} MB/s)")
                last = time.time()
        fo.flush()
        os.fsync(fo.fileno())
    if tmp.stat().st_size != size:
        raise RuntimeError(f"copy size mismatch for {dst}: {tmp.stat().st_size} != {size}")
    os.replace(tmp, dst)
    ctx.log(f"  ✓ {src} → {dst} ({gb(size)} GB, {time.time() - t0:.0f}s)")
    return size


def h_slice(ctx):
    p = ctx.params
    run, key = check_run(p["run"]), check_key(p["key"])
    start = key.split("_to_")[0]
    w = work_run(run)
    for sub in ("mosaics", "pmtiles", "logs"):
        (w / sub).mkdir(parents=True, exist_ok=True)
    mosaic = w / "mosaics" / f"{key}.tif"
    temp, chart_temp = WORK / "temp" / run, WORK / "chartfull" / run
    archived = hdd_run(run) / "mosaics" / f"{key}.tif"
    if not p.get("force"):
        for m in (mosaic, archived):
            if m.exists() and m.stat().st_size > 0:
                ctx.log(f"mosaic already exists at {m}; not re-slicing (pass force to redo)")
                return {"mosaic": str(m), "skipped": True, "size_gb": gb(m.stat().st_size)}
    for m in (mosaic, w / "mosaics" / f"{key}.partial.tif"):
        m.unlink(missing_ok=True)

    args = [PY, "-u", ctx.rel / "scripts" / "slicer.py",
            "-s", SOURCES, "-o", w / "mosaics", "-t", temp,
            "-c", ctx.rel / "master_dole_v2.csv", "-b", ctx.rel / "shapefiles",
            "-y", "--start-date", start, "--end-date", start,
            "--geotiff2pmtiles-bin", G2P, "--chart-temp", chart_temp]
    if p.get("charts"):
        manifest = w / "charts" / "manifest.jsonl"
        if not manifest.exists():
            # Seed with the catalog-wide manifest so the slicer's Cool-URIs
            # collision check still sees every chart key published before.
            seed = ctx.rel / "chart_pmtiles" / "manifest.jsonl"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            if seed.exists():
                shutil.copyfile(seed, manifest)
                (w / "charts" / "manifest.seed_lines").write_text(
                    str(sum(1 for _ in open(seed))) + "\n")
        args += ["--chart-pmtiles", w / "charts", "--chart-manifest", manifest]
    args += [str(a) for a in p.get("slicer_args", [])]
    env = {"TMPDIR": str(WORK / "tmp"), "CPL_TMPDIR": str(WORK / "tmp")}
    (WORK / "tmp").mkdir(exist_ok=True)

    t0 = time.time()
    before = {f.name for f in (w / "mosaics").glob("*.tif")}
    attempts = 2 if p.get("retry", True) else 1
    for attempt in range(1, attempts + 1):
        extra = []
        if attempt == 2:
            ctx.log("attempt 1 failed: wiping this era's temps and retrying with --parallel-warp 1")
            shutil.rmtree(temp / key, ignore_errors=True)
            (w / "mosaics" / f"{key}.partial.tif").unlink(missing_ok=True)
            extra = ["--parallel-warp", "1"]
        rc, out = ctx.run(args + extra, cwd=w / "logs", env=env)
        # --start-date D slices every era starting on D (the 1950s Boston
        # sheets share starts), so look for this key's own completion line.
        complete = f"{key.upper()} GEOTIFF COMPLETE" in out
        grid = re.findall(r"Mosaic grid: z(\d+)", out)
        if rc == 0 and complete and grid and mosaic.exists() and mosaic.stat().st_size > 0:
            break
        ctx.log(f"attempt {attempt}: rc={rc} complete={complete} grid={grid} mosaic={mosaic.exists()}")
    else:
        raise RuntimeError(f"slicer did not complete {key} (see log)")
    fails = [ln.strip() for ln in out.splitlines() if "✗" in ln]
    also = sorted(f.name[:-4] for f in (w / "mosaics").glob("*.tif")
                  if f.name not in before and f.name != mosaic.name and ".partial" not in f.name)
    if not p.get("keep_temps"):
        for k in [key, *also]:  # every era this run completed
            shutil.rmtree(temp / k, ignore_errors=True)
            shutil.rmtree(chart_temp / k, ignore_errors=True)
    if also:
        ctx.log(f"also produced (same start date): {also} — archive them with their own archive jobs")
    return {"mosaic": str(mosaic), "also_produced": also, "size_gb": gb(mosaic.stat().st_size), "grid_zoom": grid_zoom(mosaic),
            "attempt": attempt, "seconds": round(time.time() - t0), "failure_lines": len(fails),
            "failure_samples": fails[:8]}


def h_convert(ctx):
    from osgeo import gdal
    gdal.UseExceptions()
    p = ctx.params
    run, key = check_run(p["run"]), check_key(p["key"])
    src = locate(run, "mosaics", f"{key}.tif")
    if not src:
        raise RuntimeError(f"no mosaic for {key} in run {run} (SSD or HDD)")
    w = work_run(run)
    (w / "pmtiles").mkdir(parents=True, exist_ok=True)
    (w / "logs").mkdir(parents=True, exist_ok=True)
    out, part = w / "pmtiles" / f"{key}.pmtiles", w / "pmtiles" / f"{key}.part.pmtiles"
    stamp, align_log = w / "logs" / f"converter_{key}.txt", w / "logs" / f"align_{key}.log"
    cid = converter_id()
    if out.exists() and stamp.exists() and stamp.read_text().strip() == cid and not p.get("force"):
        ctx.log(f"{out} already converted by {cid}; pass force to redo")
        return {"pmtiles": str(out), "skipped": True, **pm_header(out)}
    ds = gdal.Open(str(src))
    dt, nb = ds.GetRasterBand(1).DataType, ds.RasterCount
    ds = None
    conc = int(p.get("concurrency") or G2P_CONCURRENCY)
    args = [G2P, "-format", p.get("format", "webp"), "-quality", str(p.get("quality", 80)),
            "-concurrency", str(conc), "-mem-limit", str(int(p.get("mem_limit_mb") or G2P_MEM_MB))]
    if dt != gdal.GDT_Byte:
        args += ["-rescale", "linear", "-rescale-range", "0,65535", "-alpha-band", str(nb)]
    args += [str(a) for a in p.get("g2p_args", [])]
    part.unlink(missing_ok=True)
    stamp.unlink(missing_ok=True)
    t0 = time.time()
    ctx.log(f"converting {src} ({gb(src.stat().st_size)} GB {gdal.GetDataTypeName(dt)}x{nb}) with {cid}")
    rc, _ = ctx.run(args + [src, part])
    if rc != 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"geotiff2pmtiles rc={rc}")
    os.replace(part, out)
    stamp.write_text(cid + "\n")
    hdr = pm_header(out)
    secs = round(time.time() - t0)
    ctx.log(f"✓ {out.name}: {gb(out.stat().st_size)} GB z{hdr['min_zoom']}-{hdr['max_zoom']} "
            f"tiles={hdr['tiles']} {secs}s")
    tol = float(p.get("align_tol", ALIGN_TOL))
    rc, text = ctx.run([PY, APP / "verify_align.py", src, out, "--tol", tol,
                        "--zoom", hdr["max_zoom"], "--auto-probes"])
    align_log.write_text(text)
    m = re.search(r"worst \|shift\| ([\d.]+) px over (\d+) probes", text)
    if not m:
        raise RuntimeError(f"alignment check crashed (rc={rc}); see {align_log} — pmtiles kept at {out}")
    worst, probes = float(m.group(1)), int(m.group(2))
    res = {"pmtiles": str(out), "size_gb": gb(out.stat().st_size), **hdr, "seconds": secs,
           "converter": cid, "align_worst_px": worst, "align_probes": probes, "align_tol": tol,
           "aligned": (worst <= tol) if probes else None}
    if probes and worst > tol:
        raise RuntimeError(f"misaligned: worst {worst} px > tol {tol} (result kept at {out})")
    if not probes:
        ctx.log("alignment not checkable: no probe had chart coverage (publish needs allow_unverified)")
    return res


def h_archive(ctx):
    p = ctx.params
    run = check_run(p["run"])
    key = check_key(p["key"]) if p.get("key") else None
    w, h = work_run(run), hdd_run(run)
    mosaic_mode = p.get("mosaic", "move")  # move | delete | keep
    moved, total = [], 0
    items = []
    if key:
        items += [("pmtiles", f"{key}.pmtiles", "move")]
        items += [("mosaics", f"{key}.tif", mosaic_mode)]
        for f in sorted((w / "logs").glob(f"*{key}*")) if (w / "logs").exists() else []:
            items.append(("logs", f.name, "move"))
    for f in sorted((w / "mosaics").glob("processing_review_*.csv")) if (w / "mosaics").exists() else []:
        items.append(("mosaics", f.name, "move"))
    if p.get("charts", True) and (w / "charts").exists():
        for f in sorted((w / "charts").rglob("*.pmtiles")):
            items.append(("charts", str(f.relative_to(w / "charts")), "move"))
        for f in ("manifest.jsonl", "manifest.seed_lines"):
            if (w / "charts" / f).exists():
                items.append(("charts", f, "copy"))
    for sub, name, mode in items:
        src = w / sub / name
        if not src.exists() or mode == "keep":
            continue
        if mode == "delete":
            ctx.log(f"  deleting {src} (mosaic=delete)")
            src.unlink()
            continue
        total += copy_file(ctx, src, h / sub / name)
        moved.append(f"{sub}/{name}")
        if mode == "move":
            src.unlink()
    if not moved:
        ctx.log("nothing on the SSD to archive")
    return {"archived": moved, "total_gb": gb(total), "dest": str(h)}


def h_publish(ctx):
    p = ctx.params
    run, key = check_run(p["run"]), check_key(p["key"])
    dry = bool(p.get("dry_run"))
    if not dry and p.get("confirm") is not True:
        raise RuntimeError("publish uploads to the live R2 bucket: resubmit with confirm=true (or dry_run=true)")
    pm = locate(run, "pmtiles", f"{key}.pmtiles")
    if not pm:
        raise RuntimeError(f"no pmtiles for {key} in run {run}")
    args = [PY, APP / "publish_era.py", "--release", ctx.rel, "--key", key, "--pm", pm,
            "--bundle-out", hdd_run(run) / "bundles", "--remote-prefix", REMOTE_PREFIX,
            "--max-tile-delta", p.get("max_tile_delta", 0.02)]
    for flag, name in (("--mosaic", "mosaics/%s.tif"), ("--align-log", "logs/align_%s.log"),
                       ("--stamp", "logs/converter_%s.txt")):
        sub, fname = (name % key).split("/")
        f = locate(run, sub, fname)
        if f:
            args += [flag, f]
    for flag in ("allow_new", "allow_unverified", "dry_run"):
        if p.get(flag):
            args.append("--" + flag.replace("_", "-"))
    hdd_run(run).mkdir(parents=True, exist_ok=True)
    rc, out = ctx.run(args)
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
    if rc != 0 or not lines:
        tail = [ln for ln in out.splitlines() if ln.strip()][-1:] or ["(no output)"]
        raise RuntimeError(f"publish_era rc={rc}: {tail[0]}")
    res = json.loads(lines[-1][7:])
    res["bundle_on_mac"] = f"/Volumes/projects/slicer-runs/{run}/bundles/{res['bundle']}"
    if not dry:
        (hdd_run(run) / "logs").mkdir(parents=True, exist_ok=True)
        with open(hdd_run(run) / "logs" / "uploaded.jsonl", "a") as f:
            f.write(json.dumps({**res, "job": ctx.id}) + "\n")
        res["next"] = ("era + bundle are on R2 but the live viewer still names the old bundle: "
                       f"finish on the Mac now (slicerctl publish does this) — bundleUrl → {res['bundle_url']}")
    return res


def h_script(ctx):
    p = ctx.params
    script = p.get("script")
    if script not in SCRIPTS:
        raise RuntimeError(f"script must be one of {sorted(SCRIPTS)}")
    args = [str(a) for a in p.get("args", [])]
    if script == "publish_chart_pmtiles.py" and "--dry-run" not in args and p.get("confirm") is not True:
        raise RuntimeError("publish_chart_pmtiles.py uploads to R2: pass confirm=true or --dry-run")
    cwd = WORK / "scratch" / ctx.id
    cwd.mkdir(parents=True, exist_ok=True)
    rc, out = ctx.run([PY, "-u", ctx.rel / "scripts" / script, *args], cwd=cwd,
                      env={"TMPDIR": str(WORK / "tmp"), "CPL_TMPDIR": str(WORK / "tmp")})
    tail = out.splitlines()[-15:]
    if rc != 0:
        raise RuntimeError(f"{script} rc={rc}: {tail[-1] if tail else ''}")
    return {"rc": rc, "cwd": str(cwd), "tail": tail}


HANDLERS = {"slice": h_slice, "convert": h_convert, "archive": h_archive,
            "publish": h_publish, "script": h_script}


# ---------------------------------------------------------------- engine

class Engine:
    def __init__(self):
        for d in (STATE / "logs", STATE / "home", WORK / "runs", WORK / "tmp"):
            d.mkdir(parents=True, exist_ok=True)
        self.store = Store(STATE / "slicerd.db")
        self.ctxs = {}
        self.blocked = {}
        self.lock = threading.Lock()
        self.started = now()
        self.paused = (STATE / "PAUSED").exists()
        self._diff_cache = (0.0, [])
        self.remount_requested = None
        (STATE / "remount.request").unlink(missing_ok=True)
        self.store.x("UPDATE jobs SET status='interrupted', error='controller restarted while running', "
                     "finished=?, pid=NULL WHERE status='running'", now())
        try:
            self.restart_note = json.loads((STATE / "restart.json").read_text())
        except Exception:
            self.restart_note = {}
        threading.Thread(target=self._loop, name="scheduler", daemon=True).start()

    # -- public API ------------------------------------------------------
    def submit(self, jtype, params, depends_on=None):
        if jtype not in JOB_TYPES:
            raise ValueError(f"type must be one of {sorted(JOB_TYPES)}")
        params = dict(params or {})
        run = params.get("run")
        if jtype != "script" or run:
            check_run(run)
        key = params.get("key")
        if key:
            check_key(key)
        elif jtype in ("slice", "convert", "publish"):
            raise ValueError(f"{jtype} needs params.key")
        if jtype == "script" and params.get("script") not in SCRIPTS:
            raise ValueError(f"script must be one of {sorted(SCRIPTS)}")
        if depends_on and not self.store.get(depends_on):
            raise ValueError(f"no such job {depends_on}")
        jid = f"{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        self.store.x("INSERT INTO jobs (id, type, lane, status, run, key, params, depends_on, created) "
                     "VALUES (?,?,?,?,?,?,?,?,?)", jid, jtype, JOB_TYPES[jtype], "queued", run, key,
                     json.dumps(params), depends_on, now())
        return self.job(jid)

    def pipeline(self, keys, run, steps=("slice", "convert", "archive"), **opts):
        """Chain the steps per key; each key's chain runs in order, keys in the order given."""
        check_run(run)
        keys = [check_key(k) for k in keys]
        bad = [s for s in steps if s not in ("slice", "convert", "archive", "publish")]
        if bad or not steps:
            raise ValueError(f"steps must be from slice, convert, archive, publish (got {list(steps)})")
        if "publish" in steps and not (opts.get("confirm") is True or opts.get("dry_run")):
            raise ValueError("a pipeline with publish needs confirm=true (or dry_run=true)")
        common = {k: v for k, v in opts.items() if v is not None}
        created = []
        for key in keys:
            prev = None
            for step in steps:
                j = self.submit(step, {**common, "run": run, "key": key}, depends_on=prev)
                prev = j["id"]
                created.append(j["id"])
        return created

    def cancel(self, jid, reason=None):
        j = self.store.get(jid)
        if not j:
            raise KeyError(jid)
        if j["status"] == "queued":
            self._finish(jid, "cancelled", error=reason or "cancelled before start")
        elif j["status"] == "running":
            ctx = self.ctxs.get(jid)
            if ctx:
                ctx.abort_reason = reason or "cancelled"
                ctx.cancel.set()
        return self.job(jid)

    def retry(self, jid):
        j = self.store.get(jid)
        if not j:
            raise KeyError(jid)
        if j["status"] not in ("failed", "cancelled", "interrupted"):
            raise ValueError(f"job is {j['status']}; only failed/cancelled/interrupted jobs can be retried")
        requeued = []
        stack = [jid]
        while stack:
            cur = stack.pop()
            self.store.update(cur, status="queued", error=None, result=None, started=None, finished=None)
            requeued.append(cur)
            for d in self.store.q("SELECT id FROM jobs WHERE depends_on=? AND status='cancelled'", cur):
                stack.append(d["id"])
        return requeued

    def set_paused(self, paused):
        self.paused = paused
        if paused:
            (STATE / "PAUSED").write_text(now())
        else:
            (STATE / "PAUSED").unlink(missing_ok=True)

    def job(self, jid, tail=0):
        j = self.store.get(jid)
        if not j:
            return None
        if j["status"] == "queued" and jid in self.blocked:
            j["blocked"] = self.blocked[jid]
        lp = STATE / "logs" / f"{jid}.log"
        j["log_bytes"] = lp.stat().st_size if lp.exists() else 0
        if j["status"] == "running" and lp.exists():
            j["last_line"] = (self.log_tail(jid, 1) or [""])[-1]
        if tail:
            j["log_tail"] = self.log_tail(jid, tail)
        return j

    def jobs(self, status=None, run=None, limit=50):
        sql, args = "SELECT * FROM jobs", []
        conds = []
        if status:
            conds.append("status IN (%s)" % ",".join("?" * len(status.split(","))))
            args += status.split(",")
        if run:
            conds.append("run=?")
            args.append(run)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY rowid DESC LIMIT ?"
        args.append(int(limit))
        out = []
        for j in self.store.q(sql, *args):
            row = {k: j[k] for k in ("id", "type", "status", "run", "key", "depends_on", "created",
                                     "started", "finished", "error")}
            if j["status"] == "queued" and j["id"] in self.blocked:
                row["blocked"] = self.blocked[j["id"]]
            if j["status"] == "running":
                row["last_line"] = (self.log_tail(j["id"], 1) or [""])[-1]
            out.append(row)
        return out

    def log_tail(self, jid, n=100, grep=None):
        lp = STATE / "logs" / f"{jid}.log"
        if not lp.exists():
            return []
        size = lp.stat().st_size
        want = max(65536, n * 400)
        with open(lp, "rb") as f:
            f.seek(max(0, size - (want * 8 if grep else want)))
            lines = f.read().decode(errors="replace").splitlines()
        if grep:
            rx = re.compile(grep)
            lines = [ln for ln in lines if rx.search(ln)]
        return lines[-n:]

    def status(self):
        rid, meta = current_release()
        counts = Counter(r["status"] for r in self.store.q("SELECT status FROM jobs"))
        running = [self.job(r["id"]) for r in self.store.q("SELECT id FROM jobs WHERE status='running'")]
        return {
            "controller_started": self.started,
            "paused": self.paused,
            "lanes": LANES,
            "jobs": dict(counts),
            "running": [{k: j.get(k) for k in ("id", "type", "run", "key", "started", "last_line")}
                        for j in running],
            "queued_blocked": dict(list(self.blocked.items())[:10]),
            "code_release": {"id": rid, **{k: meta.get(k) for k in ("git_head", "dirty", "synced_at", "csv_sha256")}},
            "converter": converter_id(),
            "host": host_stats(),
            "settings": {"g2p_concurrency": G2P_CONCURRENCY, "g2p_mem_limit_mb": G2P_MEM_MB,
                         "min_free_slice_gb": MIN_FREE_SLICE_GB, "abort_free_gb": ABORT_FREE_GB,
                         "align_tol": ALIGN_TOL, "remote_prefix": REMOTE_PREFIX},
        }

    # -- scheduler -------------------------------------------------------
    def _finish(self, jid, status, result=None, error=None):
        self.store.update(jid, status=status, result=result, error=error, finished=now(), pid=None)
        self.blocked.pop(jid, None)

    def _free_gb(self):
        s = os.statvfs(WORK)
        return s.f_bavail * s.f_frsize / 1e9

    def _loop(self):
        while True:
            try:
                self._tick()
            except Exception as e:  # keep scheduling whatever happens
                print(f"scheduler error: {type(e).__name__}: {e}", flush=True)
            time.sleep(2)

    def _tick(self):
        free = self._free_gb()
        if free < ABORT_FREE_GB:
            for jid, ctx in list(self.ctxs.items()):
                if ctx.job["lane"] == "cpu" and not ctx.cancel.is_set():
                    ctx.abort_reason = f"SSD free space {free:.1f} GB < {ABORT_FREE_GB} GB guard"
                    ctx.cancel.set()
        if self.remount_requested:
            if time.time() - self.remount_requested > 180:
                (STATE / "remount.request").unlink(missing_ok=True)
                self.remount_requested = None
                jid = json.loads((STATE / "restart.json").read_text()).get("job")
                if jid:
                    self._finish(jid, "failed", error="host remount helper did not respond in 180 s "
                                 "(is archive-slicer-remount.path enabled on the NUC?)")
            return
        if self.paused:
            return
        running = self.store.q("SELECT id, lane FROM jobs WHERE status='running'")
        busy = Counter(r["lane"] for r in running)
        for job in self.store.q("SELECT * FROM jobs WHERE status='queued' ORDER BY rowid"):
            jid, lane = job["id"], job["lane"]
            dep = job["depends_on"]
            if dep:
                d = self.store.get(dep)
                if d is None or d["status"] in ("failed", "cancelled", "interrupted"):
                    self._finish(jid, "cancelled", error=f"dependency {dep} {d['status'] if d else 'missing'}")
                    continue
                if d["status"] != "succeeded":
                    self.blocked[jid] = f"waiting for {dep} ({d['type']} {d['status']})"
                    continue
            if busy[lane] >= LANES[lane]:
                self.blocked[jid] = f"{lane} lane busy"
                continue
            reason = self._precheck(job, free, running)
            if reason == "failed":
                continue
            if reason:
                self.blocked[jid] = reason
                continue
            self.blocked.pop(jid, None)
            self._start(job)
            busy[lane] += 1
            running.append({"id": jid, "lane": lane})

    def _precheck(self, job, free, running):
        if not current_release()[0] and job["type"] in ("slice", "convert", "publish", "script"):
            return "no code release synced yet (run `slicerctl sync` on the Mac)"
        if job["type"] == "slice":
            if free < MIN_FREE_SLICE_GB:
                return f"waiting for SSD space ({free:.0f} GB free < {MIN_FREE_SLICE_GB:.0f} GB)"
            return self._overlay_gate(job, running)
        if job["type"] == "convert" and free < MIN_FREE_CONVERT_GB:
            return f"waiting for SSD space ({free:.0f} GB free < {MIN_FREE_CONVERT_GB:.0f} GB)"
        return None

    def _overlay_gate(self, job, running):
        """A slice must see rawtiffs as it is now: remount the overlay if it lags."""
        t, diffs = self._diff_cache
        if time.time() - t > 30:
            diffs = overlay_diff()
            self._diff_cache = (time.time(), diffs)
        if not diffs:
            return None
        note = self.restart_note
        if note.get("job") == job["id"] and time.time() - note.get("at", 0) < 900:
            self._finish(job["id"], "failed",
                         error="overlay still differs from rawtiffs after a remount: " + "; ".join(diffs[:5]))
            return "failed"
        if running:
            return "rawtiffs changed since the overlay was mounted: waiting for idle to remount"
        self.request_remount(job["id"], diffs)
        return f"rawtiffs changed since the overlay was mounted ({diffs[0]}): remount requested"

    def request_remount(self, job_id=None, diffs=None):
        """Ask the host helper (archive-slicer-remount.path) to stop + start this container."""
        if self.store.q("SELECT id FROM jobs WHERE status='running'"):
            raise RuntimeError("jobs are running; a remount would kill them")
        (STATE / "restart.json").write_text(json.dumps({"job": job_id, "at": time.time(),
                                                        "diffs": (diffs or [])[:5]}))
        (STATE / "remount.request").write_text(now() + "\n")
        self.remount_requested = time.time()
        print(f"remount requested ({(diffs or ['manual'])[0]})", flush=True)
        return {"requested": True, "note": "the container will be stopped and started within seconds"}

    def _start(self, job):
        rid = current_release()[0]
        self.store.update(job["id"], status="running", started=now(), release=rid)
        job = self.store.get(job["id"])
        try:
            ctx = Ctx(self, job)
        except Exception as e:
            self._finish(job["id"], "failed", error=f"{type(e).__name__}: {e}")
            return
        self.ctxs[job["id"]] = ctx
        threading.Thread(target=self._run, args=(ctx,), name=f"job-{job['id']}", daemon=True).start()

    def _run(self, ctx):
        jid, job = ctx.id, ctx.job
        ctx.log(f"job {jid}: {job['type']} {json.dumps(job['params'])} (release {job['release']})")
        try:
            res = HANDLERS[job["type"]](ctx)
            ctx.log(f"✓ succeeded: {json.dumps(res)[:2000]}")
            self._finish(jid, "succeeded", result=res)
        except Cancelled as e:
            ctx.log(f"✗ cancelled: {e}")
            self._finish(jid, "cancelled", error=str(e))
        except Exception as e:
            ctx.log(f"✗ failed: {type(e).__name__}: {e}")
            self._finish(jid, "failed", error=f"{type(e).__name__}: {e}")
        finally:
            ctx.close()
            self.ctxs.pop(jid, None)
            run = job.get("run")
            if run and RUN_RE.match(run):
                try:
                    d = (hdd_run(run) if job["lane"] == "io" else work_run(run)) / "logs"
                    d.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ctx.log_path, d / f"job_{job['type']}_{job.get('key') or 'run'}_{jid}.log")
                except OSError:
                    pass

    # -- inventory -------------------------------------------------------
    def runs(self):
        out = {}
        for where, base in (("ssd", WORK / "runs"), ("hdd", RUNS)):
            if not base.exists():
                continue
            for d in sorted(base.iterdir()):
                if d.is_dir() and RUN_RE.match(d.name):
                    r = out.setdefault(d.name, {"ssd": {}, "hdd": {}})
                    for sub, ext in (("mosaics", ".tif"), ("pmtiles", ".pmtiles")):
                        files = [f for f in (d / sub).glob(f"*{ext}") if ".part" not in f.name] \
                            if (d / sub).exists() else []
                        r[where][sub] = {"count": len(files), "gb": gb(sum(f.stat().st_size for f in files))}
        return out

    def run_detail(self, run):
        out = {"run": run, "keys": {}}
        for where, base in (("ssd", work_run(run)), ("hdd", hdd_run(run))):
            for sub, ext in (("mosaics", ".tif"), ("pmtiles", ".pmtiles")):
                for f in sorted((base / sub).glob(f"*{ext}")) if (base / sub).exists() else []:
                    k = f.name[: -len(ext)]
                    out["keys"].setdefault(k, {})[f"{sub[:-1]}_{where}_gb"] = gb(f.stat().st_size)
            for f in (base / "logs").glob("align_*.log") if (base / "logs").exists() else []:
                m = re.search(r"worst \|shift\| ([\d.]+) px over (\d+) probes", f.read_text())
                if m:
                    out["keys"].setdefault(f.stem[6:], {})["align"] = f"{m.group(1)} px / {m.group(2)} probes"
            up = base / "logs" / "uploaded.jsonl"
            if up.exists():
                for line in up.read_text().splitlines():
                    try:
                        r = json.loads(line)
                        out["keys"].setdefault(r["key"], {})["uploaded"] = r.get("uploaded_at")
                    except Exception:
                        pass
            charts = base / "charts"
            if charts.exists():
                out[f"charts_{where}"] = sum(1 for _ in charts.rglob("*.pmtiles"))
        return out

    def eras(self, start=None, end=None, contains=None, limit=200):
        keys = era_keys()
        if start:
            keys = [k for k in keys if k[:10] >= start]
        if end:
            keys = [k for k in keys if k[:10] <= end]
        if contains:
            keys = [k for k in keys if contains in k]
        return {"count": len(keys), "keys": keys[:limit], "source": "dates.csv of the current code release (live eras)"}

    def cleanup(self, run, key=None, what=("temp",)):
        """Delete SSD (never HDD, never sources) artifacts for a run/key."""
        check_run(run)
        if key:
            check_key(key)
        busy = self.store.q("SELECT id FROM jobs WHERE status='running' AND run=?", run)
        if busy:
            raise RuntimeError(f"run {run} has a running job ({busy[0]['id']})")
        removed, freed = [], 0
        targets = []
        w = work_run(run)
        for item in what:
            if item == "temp":
                targets += [WORK / "temp" / run / key] if key else [WORK / "temp" / run, WORK / "chartfull" / run]
            elif item in ("mosaic", "pmtiles"):
                sub, ext = ("mosaics", ".tif") if item == "mosaic" else ("pmtiles", ".pmtiles")
                if key:
                    targets += [w / sub / f"{key}{ext}", w / sub / f"{key}.partial{ext}", w / sub / f"{key}.part{ext}"]
                else:
                    targets += [w / sub]
            elif item == "run":
                targets += [w, WORK / "temp" / run, WORK / "chartfull" / run]
            else:
                raise ValueError("what: temp | mosaic | pmtiles | run")
        for t in targets:
            t = Path(t)
            if not t.exists() or WORK.resolve() not in t.resolve().parents:
                continue
            size = sum(f.stat().st_size for f in t.rglob("*") if f.is_file()) if t.is_dir() else t.stat().st_size
            shutil.rmtree(t) if t.is_dir() else t.unlink()
            removed.append(str(t))
            freed += size
        return {"removed": removed, "freed_gb": gb(freed)}
