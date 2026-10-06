# slicerd — archive.aero slicing on the OMV NUC

A Docker job server on the NUC (`ssh omv`, 192.168.1.203) that runs the era
pipeline — `slicer.py` → mosaic → patched `geotiff2pmtiles` → alignment check
→ archive to the HDD → (on command) R2 upload — driven from the Mac by
`slicerctl` or by any LLM client through MCP.

## Layout

| Container | Host | Notes |
|---|---|---|
| `/data/rawtiffs` | `projects/rawtiffs` | read-only bind; nothing in the container can write it |
| `/data/sources` | overlay volume | what `slicer.py -s` reads: rawtiffs as the untouched lower layer + SSD upper |
| `/data/src-upper` | `/srv/archive-slicer/src-upper` | read-only view of what the slicer wrote into its source tree |
| `/work` | `/srv/archive-slicer/work` (NVMe) | warp temps, fresh mosaics + pmtiles |
| `/runs` | `projects/slicer-runs` (HDD) | finished outputs; `/Volumes/projects/slicer-runs` on the Mac |
| `/code` | `/srv/archive-slicer/code` | releases pushed by `slicerctl sync`; `current` → newest |
| `/state` | `/srv/archive-slicer/state` | job DB + logs, remount requests |
| `/secrets` | `/srv/archive-slicer/secrets` | `token`, `rclone.conf` (`[r2]` only) |

**Why an overlay.** `slicer.py` writes into its source tree in places
(unzips, 16-bit → 8-bit, PDF → 300 dpi TIFF, Wayback downloads). Through the
overlay those writes succeed but land in the SSD upper layer, so rawtiffs stays
*as found*. `slicerctl overlay` lists them. Promoting one into rawtiffs is a
deliberate manual step (see CLAUDE.md, "rawtiffs holds sources exactly as
found").

**Remounts.** A mounted overlay does not see files added to rawtiffs afterwards
(tested 2026-10-02: new entries missing from listings, cached "not found"
lookups). Before every slice the controller compares the overlay with rawtiffs.
If they differ, it waits until idle and asks the host helper
(`archive-slicer-remount.path` → `.service`) to `docker stop` + `docker start`
the container. Only stop + start remounts; a restart-policy restart does not.

**Resources.** `mem_limit 54g`: the slicer's GDAL cache, geotiff2pmtiles' tile
store (`-mem-limit 20000`), and page cache for the 35 GB mosaic the converter
reads straight after the slicer writes it. `cpu_shares 256` and
`oom_score_adj 500`: Jellyfin and Samba win any contention.

## Lanes and slots

Jobs run in two lanes: `cpu` (slice, convert, script) and `io` (archive,
upload, publish). A lane has slots (`slicerctl set cpu_slots=6 io_slots=2`,
live, no restart). A slice takes one slot per catalog row of its date's largest
era, because that is how many charts the slicer warps at once: single-chart
eras run side by side and a modern era has the lane to itself. A convert takes
the lane for a mosaic of 2 GB or more and two slots (four converter threads)
below that. The lane is strictly first in, first out once a job is waiting for
slots, so a full-lane job is never starved. Two slices of the same start date
never overlap (`--start-date` slices every era of the date), and each start
date has its own warp temp root.

The NUC is heat-limited: three single-chart slices already hold the package at
about 97 °C and 2.7 GHz (2026-10-04). More slots still add throughput, at a
lower clock; lower `cpu_slots` if the box should run cooler.

## Hashed uploads (beta bucket)

`upload` is a job type in the io lane. It sends an era's archive, and every
chart artifact the run has archived so far, to the remote named by the
`hashed_remote` setting (`r2:charts-beta` since 2026-10-04) under the next/
viewer's content-versioned keys:

    sectionals/<era>.<sha256[:12]>.pmtiles
    sectionals/chart/<slug>/<date>.<sha256[:12]>.pmtiles

The object carries `sha256` metadata and `application/vnd.pmtiles`. After the
upload is listed at the right size **the local file is deleted**: the bucket
holds the only copy. The run keeps `hashed/plan.jsonl` (one record per object,
the shape `scripts/next_version_archives.py` writes) and `hashed/stubs/`
(sparse header + directory stubs for `next_build_manifest.py --dir`; copy them
off the share with `rsync --sparse`). Mosaics stay on the HDD, so a convert can
be redone without reslicing. With no remote set, upload jobs wait.

`slicerctl pipeline RUN --keys-from FILE --charts --steps slice,convert,archive,upload`
adds the upload step to eras that already hold the first three. Every 50th
era's upload also sweeps the chart artifacts archived so far, and one last job
takes the rest; `slicerctl submit upload --run RUN` sweeps on demand.

## The Mac as a second machine

`slicerctl worker` takes start-date groups off the server's queue (every queued
slice, convert and archive job of the eras starting that day), runs the same
code release and the same patched converter
(`~/slicerd-work/bin/geotiff2pmtiles`, built from `g2p/` with
`DEVELOPER_DIR=/Library/Developer/CommandLineTools`), writes the outputs into
`/Volumes/projects/slicer-runs/RUN/` where the server's archive step would put
them, and reports each job back. Upload jobs stay on the server.

    nohup caffeinate -i slicerd/mac/slicerctl worker --jobs 4 >> ~/slicerd-work/worker.log 2>&1 &
    pkill -TERM -f "slicerctl worker"     # stop: running steps are killed, claims go back to the queue

- It reads rawtiffs only through a **read-only** mount of the share
  (`~/mnt/projects-ro`, mounted on demand), so nothing on the Mac can write
  into rawtiffs. An era whose source must be unpacked or converted in place
  fails there and is handed back marked `local_only` for the server's overlay.
- `--max-charts` (default 4) is the largest era it takes; the 16 GB Mac leaves
  the big eras to the NUC.
- A group the server has already begun is never claimed. A worker that sends no
  heartbeat for 15 minutes has its claims requeued.
- Its chart manifest lines go to `charts/manifest.worker-<name>.jsonl` beside
  the server's `manifest.jsonl`; merge both when the run is done.
- The Mac's GDAL is 3.12.2, the container's 3.12.1; converter stamps say which
  machine built an archive (`darwin-arm64` vs `linux-amd64`).

## Everyday use (Mac)

```bash
slicerctl status
slicerctl sync                      # after changing scripts/ or the catalog
slicerctl eras --start 2024-01-01
slicerctl pipeline 2026-10-03_reslice 2024-12-26_to_2025-02-20 2025-02-20_to_2025-04-17
slicerctl jobs ; slicerctl log <id> -f
slicerctl errors 2026-10-03_reslice # failed jobs + every slicer ✗/⚠ line, by cause
slicerctl publish 2026-10-03_reslice 2024-12-26_to_2025-02-20 --dry-run
slicerctl publish 2026-10-03_reslice 2024-12-26_to_2025-02-20   # live: R2 + index.html push
```

`pipeline` converts with `-min-zoom 0` (`--min-zoom -1` restores the converter's
auto floor, the zoom where the era fits one tile), and the slicer's per-chart
artifacts go down to z0 as well. The Leaflet viewer never asks below z8; the
`next/` renderer draws nothing below an archive's minimum zoom. The MCP
`submit_pipeline` tool has no such default: queue the convert step with
`submit_job` and `g2p_args: ["-min-zoom", "0"]`.

`pipeline` queues one era at a time and retries a refused job, because the
deployed server's job ids collide within about a hundred jobs queued in one
call (fixed in `app/engine.py`, live after the next `deploy.sh`). A step the run
already holds for a key is not queued again, so repeating the command resumes
an interrupted submission; `--force` queues everything regardless.

`errors RUN` reads the server's job records and logs and writes
`worklists/data/slicer_runs/RUN/errors.md` (a checklist grouped by cause) and
`errors.jsonl` (one record per problem, with job ids). It can be run at any
point of a run and says when the list is partial.

`publish` is split across machines. The NUC checks eligibility, builds the
metadata bundle, and uploads the bundle and then the era. Straight away the Mac
copies the bundle into the repo, rewrites `bundleUrl`, commits, pushes, waits
for Pages, and verifies through the CDN. The MCP `publish_era` tool only does
the NUC half; finish with `slicerctl publish` or the equivalent git steps.

Claude Code: the `archive-slicer` MCP server is registered at user scope.
Other clients: streamable HTTP at `http://192.168.1.203:8765/mcp` with header
`Authorization: Bearer <token from ~/.config/slicerd/config.json>`.

## Deploy / change

`slicerd/deploy.sh` syncs this directory to the NUC and builds
`archive-slicer:latest` there. It installs the remount helper, pushes
`compose.yml` into the OMV compose plugin (stack `archive-slicer`), and runs
`up -d`. Any running job dies with the old container: `slicerctl pause`, wait
for `slicerctl jobs --status running` to empty, deploy, `slicerctl resume`.
Queued jobs, settings and a worker's claims survive the restart.
