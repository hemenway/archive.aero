# 11 — ACASIS `.imgcv2` recovery: 891 charts not in the catalog

Status: **survey complete, 9 sectionals extracted, bulk extraction pending**
Opened 2026-08-24.

## The source

`/Volumes/ACASIS/.image.imgcv2.rawcopy-baee01e5e96f42c49531fd4993c9444f.partial`
— 1,597,363,000,048 bytes. A **paused** HDD Raw Copy Tool v2.6 acquisition of a
3 TB WDC WD30EZRX (`WD-WCC1T0799404`, volumes `D:` + `E:`), taken 2026-08-24.
The disk is an **iFly GPS / iFly EFB build machine**: `/Data/FAADownloads/`,
`PlatesEngine*`, `_iFlyGPS_Data*`, `SeamlessEFB`. See [[ifly-card-2013-editions]].

Not a raw image — a proprietary block container. Decoder lives in
`~/imgcv2-toolkit/` (outside git; it is a recovery tool, not site code):

| file | purpose |
|---|---|
| `imgcv2.py` | container reader → virtual raw disk (`read_at`) |
| `ntfs.py` | GPT + NTFS + MFT + runlists over any byte reader |
| `build_index.py` | walks the record chain → `block_index.bin` |
| `scan_mft.py` | every MFT record → `files.jsonl` |
| `analyze_charts.py`, `report.py` | catalog diff |
| `validate.py` | content validation (see the trap below) |
| `extract.py` | pull files out by MFT record number |

Format, all little-endian: `IMGCV2\0\0`, metadata table at `0xd0` whose entry
offsets are **relative to the table base**, then a chain of `REC2` records from
byte 2042. Type 1 = data block (`+32 method, +40 blockIndex, +48 sourceOffset,
+56 uncompressedSize, +60 compressedSize, +64 payload`), 2 = checkpoint,
3 = session, 4 = log text. Advance by `recordSize`; there is no master index.
Method 0 = stored raw, 1 = all-zero block elided, 3 = **LZ4 block format**
(not frame — the uncompressed size must be passed in explicitly).

Chain walk: 108,361 records (96,316 data / 12,040 checkpoint / 4 session /
1 log), block indices 0–96,315 contiguous, terminating exactly at EOF.

## What was captured — and what was not

**53.9 %.** Captured bytes run to **1,615,914,336,256** (block 96,315) of
3,000,592,982,016. Consequences:

- Partition 2 (`D:`, 2,738 GB, NTFS, cluster 8192, `$MFT` at cluster 393,216,
  31 fragments) — MFT fully captured, 3,948,023 name-bearing records.
- Partition 3 (`E:`, 262 GB) starts at byte 2,738,448,498,688 — **never reached**.
  Its VBR is not in the image; nothing on E: is recoverable from this file.

## The trap: captured ≠ intact

A deleted file's clusters can have been reallocated. Two "100 % of clusters
captured" Aleutian sectionals extracted as random bytes and zeroes. Cluster
presence is **not** proof — `validate.py` reads each file's TIFF magic and IFD
dimensions before a chart is counted. 84 candidates failed that check.
Same spirit as the null-island rule in CLAUDE.md: numeric success is not proof.

## Result: 891 charts, 17.0 GB, none in `master_dole_v2.csv`

1,545 distinct chart editions on disk; 569 already catalogued; 976 not; 891 of
those content-validated and fully recoverable. Full list:
`~/imgcv2-toolkit/recoverable_charts.csv`.

Filenames follow `<Place> <TYPE> <edition>.tif` — FAA **edition numbers**, which
map onto the catalog's `edition` column. Files are georeferenced GeoTIFF
(LZW, Lambert Conformal Conic / NAD83, ~16800×12350, 8-bit palette), so they
need **no GCP work** — unlike the archive2004 jpgs in [[avsim-pack12-staged]].

| type | count | note |
|---|---|---|
| TAC (Terminal Area) | 411 | **category absent from the archive** — 33 charts, ~12 consecutive editions each |
| FLY (Flyway Planning) | 229 | **category absent** — 21 charts |
| HEL (Helicopter Route) | 111 | **category absent** — 15 charts |
| INSET | 47 | |
| WAC (World Aeronautical) | 21 | **category absent** — incl. CC-8 ed 24, CJ-27 ed 20 |
| GRAPHIC | 13 | Anchorage-style graphics |
| SEC (Sectional) | **9** | see the correction below — 50 of the original 59 were duplicates |

The catalog is sectionals-only (151 locations), so TAC/FLY/HEL/WAC are four new
collections, not just new rows — each needs a URI-space decision per `URI-POLICY.md`.

### The 9 gap-filling sectionals — extracted and GDAL-verified

Already in `~/imgcv2-toolkit/out/`, decoded, centres checked against expected
geography (no null-island):

| chart | ed | px | catalog neighbours |
|---|---|---|---|
| Atlanta | 94 | 17951×12354 | 92, **·**, 95, 96 |
| Brownsville | 94 | 17260×12323 | 92, 93, **·**, 95, 96 |
| Charlotte | 97 | 16867×12358 | 95, **·**, 98, 99 |
| Dutch Harbor | 50 | 16600×12331 | 48, 49, **·**, 51, 52 |
| Halifax | 92 | 16598×12349 | 90, **·**, 93, 94 |
| Jacksonville | 95 | 16617×12412 | 93, **·**, 96, 97 |
| Miami | 96 | 17198×12313 | 94, **·**, 97, 98 |
| Washington | 97 | 16708×12341 | 95, **·**, 98, 99 |
| Wichita | 94 | 16680×12378 | 92, **·**, 95, 96 |

Every one lands in a hole the catalog already shows. Not yet catalogued —
these still need dole rows + `note` provenance before slicing.

## Correction (same day): the sectional count is 9, not 59

A second pass widened the search to every naming convention and format. It found a
**complete `Sectional.zip`** — all 57 sheets plus `.tfw` world files and FAA FGDC
`.htm` metadata — deleted to the recycle bin on 2021-04-01 as
`$RECYCLE.BIN/S-9279~1/$RMGT70B.zip`, 3.25 GB, **100 % captured**. Its metadata
gives `Publication_Date: 20210225` for all 57.

Every one of those 57 is already in `master_dole_v2.csv`, and **all 50 of the
edition-less "new" sectionals byte-match entries in that zip** — they are loose
copies of the 2021-02-25 cycle, not new charts. Only the 9 editioned sheets are
genuine gaps. Per-sheet verdicts: `~/imgcv2-toolkit/sectionals_verdict.csv`.

The lesson generalises: a chart file with no edition in its name cannot be diffed
against the catalog by name alone. Match it by **byte size against a dated
container**, or read the FAA `.htm` sidecar, before calling it new.

## Sectionals in other formats

- **`Sec_<Place>.dat`** — iFly device packs, format now decoded
  (`~/imgcv2-toolkit/rebuild_vfrsec.py`):
  - `.gtj` plaintext: name, height, width, m/px, N/W/S/E, effective/expiry
    dates, 8 zoom scales. Projection is **ellipsoidal Mercator (EPSG:3395)** —
    the height matches the ellipsoidal formula (spherical is 0.5 % off) and
    m/px = a·Δλ/width.
  - `.fil2`: u32 count + (u16 tileId = col·256+row, u16 level 1–8, u32 offset)
    entries, offsets ascending, 0xFFFF sentinel = dat size. Tile payload runs
    to the next entry's offset.
  - `.dat`: concatenated JPEG tiles, 257 px with a 1-px shared edge (stride 256).

  **Rebuilt, catalogued and moved to rawtiffs 2026-08-24: 8 of the 10 sheets
  missing at the 2026-01-22 cycle** — `rawtiffs/ifly_vfrsec_20260122/` holds
  the decoded EPSG:3395 GeoTIFFs *and* the byte-exact encoded packs
  (`ifly_packs/`), README + verification inside; same deliberate rawtiffs
  exception as `ifly_efb_card_2013/`. 8 dole rows appended (date 2026-01-22,
  end 2026-03-19, cutline `sectional/*`, provenance in `note`; CSV backed up
  to `pre_vfrsec20260122_2026-08-24.csv`) → the cycle now has 51 of 53 sheets.
  Cutline rings run ~0.1° past the sheets' east edges — that's the FAA
  printed-chart overlap iFly trims off; identical signature to the published
  2013 card rows, so slicing-safe. **Still to do: slice the 8 into the
  2026-01-22 era, then timeline_data rebuild + R2 upload + coverage** (upload
  deferred until slicing so the viewer never lists charts it can't render).

  Original rebuild notes:
  from `/iFly/Backup/SDMMCDisk_ProdV9LowRes/VfrSec` (build 2026-01-15, every
  sheet effective 1/22/2026) → `~/imgcv2-toolkit/out/vfrsec_20260122/`, 1.15 GB
  of EPSG:3395 GeoTIFFs, zero missing tiles. Verified per sheet: gtj corner
  round-trip 0.0 m **plus** an independent landmark test (a known airport's
  WGS84 coordinate projected into each image lands on its charted symbol —
  `contact_sheet.png`). Lossy JPEG at ~78 m/px, collar stripped: gap-filler;
  retire if FAA originals for 2026-01-22 surface. **Cincinnati and Los Angeles
  blocked**: their 1/22 `.dat` payloads are 0 % captured in both ProdV9 builds
  (LowRes ~37 MB + HighRes ~88 MB each) — another resume-the-copy item.
- **`Sec_<Place>-BIG.png`** — full-res renders with `.pgw` + `.prj`, so
  georeferenced. 40 of them, but only 1 is captured; the rest are past the edge.
- **Small `Sec_*.png`** (~1200×950) — UI previews, no archival value.

## Resume the copy to get 7 more sectionals

Seven editioned sectionals sit **just past** the capture boundary, between
1.67 and 1.78 TB — the copy stopped at 1.616 TB:

| chart | ed | data at |
|---|---|---|
| Cheyenne | 91 | 1.669 TB |
| Denver | 92 | 1.669 TB |
| El Paso | 94 | 1.669 TB |
| San Francisco | 94 | 1.676 TB |
| Montreal | 92 | 1.723 TB |
| Las Vegas | 93 | 1.773 TB |

plus `CF-16 WAC 44` / `CG-19 WAC 45` (1.677 TB), `CG-20 WAC 45` (1.749 TB) and
`Chicago HEL` (2.211 TB). **Resuming to ~1.8 TB (≈185 GB more, ~11 %) recovers
all but Chicago HEL.**

A resume also reaches, in rough order of archival value:

- **`SDMMCDisk - Copy/Sectionals_HighRes`** — 105 half-sheet packs modified
  **2014-10 to 2015-02**, plus `VfrSec`/`VfrSec_HighRes` from 2015-03. The catalog
  holds only 56 sheets for all of 2014 and 129 for 2015, so this is the thinnest
  era on the site and the most interesting thing still behind the boundary.
- **Eight more `Sectional.zip` cycles** in the recycle bin (deleted 2021-06-01,
  2021-08-23, 2026-01-07, 2026-02-03, 2026-04-28, 2026-06-22, 2026-08-17, plus the
  live current one) — complete 57-sheet FAA sets, all 0 % captured.
- **39 of the 40 `-BIG.png`** renders. The source was copied unlocked (`D:` could not be locked,
see the session log), so it is not a point-in-time image.

## Next

1. Resume the acquisition to ≥1.8 TB before the drive is disturbed.
2. Bulk-extract the 891 into a staging dir — **not** into `rawtiffs`, which
   holds sources exactly as downloaded; these are recovered, not downloaded.
3. Decide URI space for TAC / FLY / HEL / WAC per `URI-POLICY.md`.
4. Catalogue the 9 sectionals (back up the CSV first), then slice.

---

# Resume procedure — exact restart point

Derived from the container, not guessed. The tool's durable resume anchor is
header `+0x68` = **file offset of the last checkpoint record**, and `+0x80` =
the block count at that checkpoint. Both were read back and confirmed against a
tail-walk of the chain.

| | block | source byte | LBA (512 B) | % of disk |
|---|---|---|---|---|
| **Tool's last checkpoint** (`+0x68`/`+0x80`) | 96,313 | **1,615,864,004,608** | **3,155,984,384** | 53.8515 |
| Chain truth (last record written) | 96,316 | 1,615,914,336,256 | 3,156,082,688 | 53.8532 |

The header lags the chain by **3 blocks (48 MiB, 98,304 sectors)** — blocks
96,313–96,315 were written after the last checkpoint flush. The tool will
re-copy them. That is expected and harmless: `build_index.py` now reports the
overlap and takes the **later** record for a duplicated block index.

**If asked for a manual start offset, give the checkpoint value —
byte `1,615,864,004,608` / LBA `3,155,984,384`.** Overlap is safe; a gap is not.

## Stop points

| goal | stop at byte | extra to copy | free after (of 2,403 GB) |
|---|---|---|---|
| 7 stranded sectionals + 3 WACs | 1,800,000,000,000 | 184 GB | ~2,219 GB |
| …plus Chicago HEL | 2,250,000,000,000 | 634 GB | ~1,769 GB |
| whole disk | 3,000,592,982,016 | 1,385 GB | ~1,018 GB |

## Before touching it

1. **Extract the 891 charts first.** The catastrophic failure mode is the tool
   restarting at block 0 and overwriting 1.6 TB. 17 GB of extraction makes the
   image expendable — cheapest insurance available.
2. Baseline recorded: `fingerprint_preflight.json` (187 sampled blocks, hashed
   header + payload edges). Verify afterwards with `fingerprint.py check`.
3. `sudo mdutil -i off /Volumes/ACASIS` — Spotlight is indexing a 1.6 TB file
   for no benefit and writing to the volume.
4. **Eject cleanly.** ACASIS is exFAT; a yanked exFAT volume corrupts.

## At the Windows box

5. **Confirm the disk by serial, not by number.** The image records
   `WDC WD30EZRX-00DC0B0` / `WD-WCC1T0799404` / 3,000,592,982,016 bytes.
   Enumeration order is not stable — if another drive comes up as
   `PhysicalDrive0`, the tool would append a different disk into this image.
   `wmic diskdrive get Index,Model,SerialNumber,Size`
6. **Take the source offline first.** `D:` could not be locked last time, so the
   image is not point-in-time. `diskpart` → `select disk N` → `offline disk`
   leaves raw `\\.\PhysicalDriveN` reads working while guaranteeing nothing
   writes to it. This is strictly better than retrying an unlocked copy.
7. Confirm ACASIS remounts with enough free space (it was `G:`).

## The canary

`.partial` must stay **≥ 1,597,363,000,048 bytes and grow**. If the size resets
to something small, that is a restart, not a resume — kill it immediately.

## Afterwards

```
python fingerprint.py check      # proves blocks 0..96,315 were not rewritten
python build_index.py            # contiguity + overlap report
python validate.py               # re-check the previously stranded charts
```

## Cheaper alternative — skip Windows entirely

The MFT is fully captured, so the exact clusters holding every stranded chart
are already known. The 38 SEC/WAC/HEL charts that failed validation need
**92 blocks = 1.5 GB across 20 contiguous runs** (lowest 1,617,491,394,560;
highest 2,621,154,787,328) — **119× less I/O than resuming to 1.8 TB**, and it
reaches charts as deep as 2.62 TB that a 1.8 TB resume would still miss.

Requires the WD30EZRX on a dock reading raw offsets (`/dev/rdiskN` + `dd`, or
the same on the Windows box). Output is a sidecar the toolkit splices over the
image. Worth doing instead of, not after, the sequential resume.

---

# Resume plan (2026-08-25)

## The exact spot

The chain ends cleanly — `chain_end == file_size`, no torn record — so the
resume point is unambiguous:

| | |
|---|---|
| last captured block | **96,315** |
| first block needed | **96,316** |
| byte offset | **1,615,914,336,256** (`0x1783C000000`) |
| LBA, 512-byte sectors | **3,156,082,688** |
| alignment | sector- and 16 MiB-block-aligned |
| bytes remaining | 1,384,678,645,760 (1.385 TB), 82,534 blocks |
| final block 178,849 | partial — only 4,677,632 bytes |

## Identify the disk by GPT GUID, not by drive number

`\\.\PhysicalDrive0` is an enumeration artefact and can move. The disk's own
identity is in `source_identity.json`:

- GPT DiskGUID **`57963A23-36D8-4031-9D0B-1F671E5C125C`**
- size exactly 3,000,592,982,016 · WDC WD30EZRX-00DC0B0 · SN WD-WCC1T0799404

`win_fetch.py` searches PhysicalDrive0–15 for that GUID and refuses to run
without a match. It then re-hashes eight blocks the image already holds
(**tripwires**, in the same file): if any differ, Windows wrote to `D:` since
2026-08-24 and the two halves no longer describe one point in time. The first
session already had to proceed unlocked, so this check is the only evidence
available.

## Don't reopen the `.partial`

`win_fetch.py` writes a **sidecar** (`resume_sidecar.bin` + `.json`) and never
opens the 1.6 TB file. Because every block record is self-locating, a
continuation does not have to be appended — `Imgcv2Image.attach_sidecar()`
layers it in at read time. Verified byte-exact, including reads spanning the
image/sidecar seam. Whether HDD Raw Copy Tool v2.6 can itself resume after an
app restart is **untested**; the `.partial` + acquisitionGuid naming suggests
it is designed to, but a wrong guess truncates 1.6 TB, so prefer the sidecar.

## Fetch 3.66 GB, not 1,005 GB

The MFT is fully addressable, so nothing on `D:` is hidden from us — the wanted
bytes are known exactly. `fetch_plan.json` holds 218 sixteen-MiB blocks:

| phase | blocks | size | what |
|---|---|---|---|
| 1 — `$MFT` gaps | 120 | 2.01 GB | 10 MFT fragments past the boundary |
| 2 — known charts | 98 | 1.64 GB | the 18 charts still incomplete |
| **total** | **218** | **3.66 GB** | vs **1,005 GB** to reach the same depth sequentially |

**Phase 1 matters more than phase 2.** 1,790,560 MFT records — **31 % of the
volume's files** — sit in ten fragments that were never copied, so the 891-chart
survey covers only 69 % of `D:`. Those fragments are at known offsets
(1.637–2.519 TB) and cost 2 GB to fetch. Expect the chart count to grow.

## Order of operations

1. `python win_fetch.py verify` — elevated. Writes nothing; confirms the GUID
   and samples three tripwires.
2. **Regenerate `fetch_plan.json` first** if the iFly-native `.dat` chart work
   (`Sec_*.dat` under `/iFly/Backup/SDMMCDisk_*/VfrSec`) has targets to add —
   one trip to the machine beats two.
3. `python win_fetch.py targeted` — ~3.7 GB, a few minutes. Resumable; rerun
   skips stored blocks.
4. Copy the `sidecar/` folder back, `~/venv/bin/python rescan.py`, then re-run
   `scan_mft.py → analyze_charts.py → report.py → validate.py` with the sidecar
   attached.
5. Phase 2's block list only covers charts found in the 69 % MFT. After step 4
   the completed MFT will name more; generate a second block list and repeat
   step 3. Two short trips, not one long one.

Sequential fallback if a full forensic image is wanted:
`python win_fetch.py sequential --until 2.62TB` (1,005 GB, ~3 h at 100 MB/s;
`--until 3TB` for the whole disk, 1.385 TB, ~4 h). `E:` begins at 2.738 TB.

## What resuming will *not* fix

67 candidates failed content validation with **all** their clusters already
captured — deleted files whose clusters were handed to something else. The
bytes are present and are not charts. No amount of further copying recovers
them; only the 18 with genuinely uncaptured data are in scope.
