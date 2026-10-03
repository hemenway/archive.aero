"""archive-slicer controller: MCP (/mcp) + REST (/api/*) on one port.

Every route except /health needs `Authorization: Bearer $SLICERD_TOKEN`.
LLM clients (Claude Code etc.) use the MCP endpoint; `slicerctl` on the Mac
and curl use REST. Both drive the same engine (engine.py).
"""
import hmac
import os
from pathlib import Path
from typing import Optional

import anyio
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

import engine as eng

PORT = int(os.environ.get("SLICERD_PORT", "8765"))
ENGINE: Optional[eng.Engine] = None

INSTRUCTIONS = """\
archive-slicer runs the archive.aero chart pipeline on the OMV NUC (i7-10710U 6C/12T, 62 GB RAM).
Pipeline per era key (YYYY-MM-DD_to_YYYY-MM-DD): slice (slicer.py → mosaic GeoTIFF on the NVMe
scratch SSD) → convert (geotiff2pmtiles b300c9e+halfpx → .pmtiles, then a 12-probe alignment check,
tol 0.3 px) → archive (move outputs to the HDD: /Volumes/projects/slicer-runs/<run>/ on the Mac)
→ publish (metadata bundle + era upload to R2; ONLY with confirm=true, and the Mac must then push
the new bundleUrl in index.html — `slicerctl publish` on the Mac does the whole thing).
rawtiffs is read-only here: the slicer reads it through an overlay whose writes land on the SSD
(see source_overlay). Jobs queue per lane (cpu: slice/convert/script, io: archive/publish) and run one
at a time per lane. A modern era takes roughly 1-2 h to slice and 2-3 h to convert on this box.
Start with status(); find keys with list_eras(); queue work with submit_pipeline(); watch with
get_job()/job_log(). Never publish without the user's explicit go-ahead.
"""

mcp = FastMCP(
    "archive-slicer",
    instructions=INSTRUCTIONS,
    host="0.0.0.0",
    port=PORT,
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


async def call(fn, *args, **kw):
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kw))


# ---------------------------------------------------------------- MCP tools

@mcp.tool()
async def status() -> dict:
    """Controller + host status: running/queued jobs, CPU load and temperature, RAM (host and
    container), free space on the SSD scratch and HDD, code release synced from the Mac, converter id."""
    return await call(ENGINE.status)


@mcp.tool()
async def list_jobs(status: Optional[str] = None, run: Optional[str] = None, limit: int = 30) -> list:
    """Recent jobs, newest first. status filter is comma-separated
    (queued,running,succeeded,failed,cancelled,interrupted). Queued jobs show why they are blocked."""
    return await call(ENGINE.jobs, status, run, limit)


@mcp.tool()
async def get_job(job_id: str, log_lines: int = 40) -> dict:
    """One job: params, result (sizes, zoom, alignment, timings), error, and the last log lines."""
    j = await call(ENGINE.job, job_id, log_lines)
    if not j:
        raise ValueError(f"no job {job_id}")
    return j


@mcp.tool()
async def job_log(job_id: str, lines: int = 200, grep: Optional[str] = None) -> list:
    """Tail of a job's log (slicer / geotiff2pmtiles / alignment output). grep is a regex filter
    applied over the recent part of the log, e.g. '✗|ERROR|Traceback'."""
    return await call(ENGINE.log_tail, job_id, lines, grep)


@mcp.tool()
async def list_eras(start: Optional[str] = None, end: Optional[str] = None,
                    contains: Optional[str] = None, limit: int = 200) -> dict:
    """Era keys known to the live site (dates.csv of the synced code release), filtered by start date
    range (YYYY-MM-DD) or substring. Keys not live yet can still be sliced by name."""
    return await call(ENGINE.eras, start, end, contains, limit)


@mcp.tool()
async def submit_pipeline(keys: list[str], run: str, steps: list[str] = ["slice", "convert", "archive"],
                          charts: bool = False, slicer_args: Optional[list[str]] = None,
                          concurrency: Optional[int] = None, mosaic: str = "move",
                          force: bool = False) -> dict:
    """Queue slice → convert → archive (default steps) for each era key, chained per key, keys in order.
    run: output folder name (letters/digits/._-), e.g. '2026-10-02_reslice'; reuse a run to add eras.
    charts: also emit per-chart full-sheet PMTiles (slicer --chart-pmtiles).
    slicer_args: extra slicer.py flags (e.g. ['--parallel-warp','6']). concurrency: geotiff2pmtiles workers.
    mosaic: what archive does with the 35 GB mosaic — move (to HDD, default) | delete | keep (on SSD).
    force: redo even when a mosaic/pmtiles already exists. 'publish' is not allowed here; use publish_era."""
    if "publish" in steps:
        raise ValueError("use publish_era for uploads (it needs explicit confirmation)")
    ids = await call(ENGINE.pipeline, keys, run, steps, charts=charts, slicer_args=slicer_args,
                     concurrency=concurrency, mosaic=mosaic, force=force)
    return {"queued": ids}


@mcp.tool()
async def submit_job(type: str, params: dict, depends_on: Optional[str] = None) -> dict:
    """Queue one job. type: slice | convert | archive | publish | script.
    params always include run (and key for slice/convert/publish). slice: charts, slicer_args, force,
    retry, keep_temps. convert: concurrency, quality, format, mem_limit_mb, align_tol, g2p_args, force.
    archive: mosaic (move|delete|keep), charts. script: script (slicer.py | build_metadata_bundle.py |
    publish_chart_pmtiles.py), args list. Publishing still needs confirm=true."""
    return await call(ENGINE.submit, type, params, depends_on)


@mcp.tool()
async def publish_era(run: str, key: str, confirm: bool = False, dry_run: bool = True,
                      allow_new: bool = False, allow_unverified: bool = False,
                      max_tile_delta: float = 0.02, depends_on: Optional[str] = None) -> dict:
    """Upload an era to the LIVE R2 bucket: checks (alignment, tile grid, zoom/bounds/tile count vs the
    live object), builds the metadata bundle, uploads bundle then era. Default is a dry run (bundle
    built, nothing uploaded). A real upload needs dry_run=false AND confirm=true, only with the user's
    explicit go-ahead — and then the Mac must immediately commit+push the new bundleUrl (prefer running
    `slicerctl publish <run> <key>` on the Mac, which does both halves)."""
    params = {"run": run, "key": key, "dry_run": dry_run, "confirm": confirm, "allow_new": allow_new,
              "allow_unverified": allow_unverified, "max_tile_delta": max_tile_delta}
    return await call(ENGINE.submit, "publish", params, depends_on)


@mcp.tool()
async def cancel_job(job_id: str) -> dict:
    """Cancel a queued job, or stop a running one (SIGTERM, then SIGKILL after 20 s)."""
    return await call(ENGINE.cancel, job_id)


@mcp.tool()
async def retry_job(job_id: str) -> dict:
    """Re-queue a failed/cancelled/interrupted job and the dependents that were cancelled because of it."""
    return {"requeued": await call(ENGINE.retry, job_id)}


@mcp.tool()
async def pause_queue(paused: bool) -> dict:
    """Pause (true) or resume (false) starting new jobs. Running jobs are not affected."""
    await call(ENGINE.set_paused, paused)
    return {"paused": paused}


@mcp.tool()
async def list_runs() -> dict:
    """All runs with mosaic/pmtiles counts and sizes on the SSD and the HDD."""
    return await call(ENGINE.runs)


@mcp.tool()
async def run_detail(run: str) -> dict:
    """Per-key state of one run: where its mosaic/pmtiles live, alignment result, upload record."""
    return await call(ENGINE.run_detail, run)


@mcp.tool()
async def source_overlay(check_stale: bool = False) -> dict:
    """Files the slicer has written into its view of rawtiffs (unzips, 16-bit/PDF conversions,
    downloads). They live only in the SSD upper layer; rawtiffs itself is never modified.
    check_stale also compares the overlay with rawtiffs (a remount happens automatically before a slice)."""
    out = await call(eng.overlay_writes)
    if check_stale:
        out["stale"] = await call(eng.overlay_diff)
    return out


@mcp.tool()
async def remount_sources() -> dict:
    """Remount the rawtiffs overlay now (the container is stopped and started by a host helper; only
    allowed when no job is running). Slices do this on their own when rawtiffs changed since mount."""
    return await call(ENGINE.request_remount, None, ["manual"])


@mcp.tool()
async def cleanup_ssd(run: str, key: Optional[str] = None, what: list[str] = ["temp"]) -> dict:
    """Free NVMe scratch for a run (optionally one key). what: temp | mosaic | pmtiles | run.
    Only ever deletes under the SSD scratch — never HDD outputs or sources."""
    return await call(ENGINE.cleanup, run, key, what)


# ---------------------------------------------------------------- REST

def route(path, methods=("GET",)):
    def deco(fn):
        async def handler(request: Request):
            try:
                return JSONResponse(await fn(request))
            except KeyError as e:
                return JSONResponse({"error": f"not found: {e}"}, status_code=404)
            except (ValueError, RuntimeError) as e:
                return JSONResponse({"error": str(e)}, status_code=400)
        mcp.custom_route(path, methods=list(methods))(handler)
        return fn
    return deco


async def body(request):
    try:
        return await request.json()
    except Exception:
        return {}


@route("/health")
async def r_health(request):
    return {"ok": True}


@route("/api/status")
async def r_status(request):
    return await call(ENGINE.status)


@route("/api/jobs")
async def r_jobs(request):
    qp = request.query_params
    return await call(ENGINE.jobs, qp.get("status"), qp.get("run"), int(qp.get("limit", 50)))


@route("/api/jobs", methods=("POST",))
async def r_submit(request):
    b = await body(request)
    return await call(ENGINE.submit, b.get("type"), b.get("params"), b.get("depends_on"))


@route("/api/jobs/{jid}")
async def r_job(request):
    j = await call(ENGINE.job, request.path_params["jid"], int(request.query_params.get("tail", 0)))
    if not j:
        raise KeyError(request.path_params["jid"])
    return j


@route("/api/jobs/{jid}/log")
async def r_log(request):
    qp = request.query_params
    return await call(ENGINE.log_tail, request.path_params["jid"], int(qp.get("lines", 200)), qp.get("grep"))


@route("/api/jobs/{jid}/cancel", methods=("POST",))
async def r_cancel(request):
    return await call(ENGINE.cancel, request.path_params["jid"])


@route("/api/jobs/{jid}/retry", methods=("POST",))
async def r_retry(request):
    return {"requeued": await call(ENGINE.retry, request.path_params["jid"])}


@route("/api/pipeline", methods=("POST",))
async def r_pipeline(request):
    b = await body(request)
    keys, run = b.pop("keys", []), b.pop("run", None)
    steps = b.pop("steps", ["slice", "convert", "archive"])
    return {"queued": await call(ENGINE.pipeline, keys, run, steps, **b)}


@route("/api/eras")
async def r_eras(request):
    qp = request.query_params
    return await call(ENGINE.eras, qp.get("start"), qp.get("end"), qp.get("contains"), int(qp.get("limit", 200)))


@route("/api/runs")
async def r_runs(request):
    return await call(ENGINE.runs)


@route("/api/runs/{run}")
async def r_run(request):
    return await call(ENGINE.run_detail, request.path_params["run"])


@route("/api/overlay")
async def r_overlay(request):
    out = await call(eng.overlay_writes)
    if request.query_params.get("stale"):
        out["stale"] = await call(eng.overlay_diff)
    return out


@route("/api/cleanup", methods=("POST",))
async def r_cleanup(request):
    b = await body(request)
    return await call(ENGINE.cleanup, b.get("run"), b.get("key"), b.get("what", ["temp"]))


@route("/api/remount", methods=("POST",))
async def r_remount(request):
    return await call(ENGINE.request_remount, None, ["manual"])


@route("/api/pause", methods=("POST",))
async def r_pause(request):
    b = await body(request)
    await call(ENGINE.set_paused, bool(b.get("paused", True)))
    return {"paused": ENGINE.paused}


# ---------------------------------------------------------------- auth + main

class BearerAuth:
    def __init__(self, app, token):
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] != "/health":
            got = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(got, self.expected):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def main():
    global ENGINE
    os.umask(0o002)
    token = os.environ.get("SLICERD_TOKEN") or Path("/secrets/token").read_text().strip()
    if len(token) < 24:
        raise SystemExit("SLICERD_TOKEN missing or too short")
    ENGINE = eng.Engine()
    app = BearerAuth(mcp.streamable_http_app(), token)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
