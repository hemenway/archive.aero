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

## Everyday use (Mac)

```bash
slicerctl status
slicerctl sync                      # after changing scripts/ or the catalog
slicerctl eras --start 2024-01-01
slicerctl pipeline 2026-10-03_reslice 2024-12-26_to_2025-02-20 2025-02-20_to_2025-04-17
slicerctl jobs ; slicerctl log <id> -f
slicerctl publish 2026-10-03_reslice 2024-12-26_to_2025-02-20 --dry-run
slicerctl publish 2026-10-03_reslice 2024-12-26_to_2025-02-20   # live: R2 + index.html push
```

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
`up -d`. Any running job dies with the old container.
