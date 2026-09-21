# openAIP archive mirror

Pulls the full openAIP export bucket into a local, de-duplicated, point-in-time archive.

## Quick start

Lives on the SD card. `-Root` defaults to an `archive\` folder beside these scripts, so
the data follows the card and never lands on C:.

```powershell
F:
cd \openaip-mirror
.\Sync-OpenAip.ps1 -ListOnly      # size it up, downloads nothing
.\Sync-OpenAip.ps1                # first full sync -> F:\openaip-mirror\archive
.\Register-OpenAipMirrorTask.ps1  # then daily at 06:00
```

## Simple downloader (plain files, no archive)

If you just want the current files on the card and don't need the de-duplicated
store or point-in-time snapshots below, use `Download-OpenAip.ps1` instead. It writes
plain files to `data\<country>\<file>` — GeoJSON by default — and skips anything already
up to date on re-run.

```powershell
.\Download-OpenAip.ps1 -ListOnly              # 992 GeoJSON files, ~424 MB
.\Download-OpenAip.ps1                        # all countries -> data\us\us_apt.geojson etc.
.\Download-OpenAip.ps1 -Country us,ca         # just North America
.\Download-OpenAip.ps1 -Format geojson,txt    # add OpenAIR airspace files
```

It has the same length check + `Range` resume as the full mirror (see "silently
truncates" below), so a half-received file is finished, not kept.

## What's in the bucket

Measured 2026-09-17: **5,275 objects, 1.59 GB** uncompressed. No API key, no account, no
signed URL — plain anonymous HTTPS GET.

| Format | Files | Size |
|---|---:|---:|
| geojson | 992 | 424 MB |
| ndgeojson | 992 | 423 MB |
| json | 992 | 405 MB |
| cupx | 537 | 149 MB |
| txt (OpenAIR) | 384 | 116 MB |
| xml | 578 | 85 MB |
| cup | 800 | 31 MB |

Keys are `<country>_<type>.<format>` — `us_apt.json`, `de_asp_v2.txt`. Types seen: `apt`
airports, `asp` airspace, `nav` navaids, `obs` obstacles, `rpp` reporting points, `hgl`
hang gliding, `hot` hotspots, `rca`, `raa`, plus `_v1`/`_v2` OpenAIR variants.

Rebuilt daily; objects carried `Last-Modified` around **03:14 UTC** on the day checked.

## Two things worth knowing about this bucket

**1. Your 403 was transient, not a permissions problem.** The object is public and
anonymous GET returns 200. A key that genuinely doesn't exist returns **404**, so a 403 on
a real file isn't "missing" or "needs auth" — it's the envoy front end refusing that
particular request, with an empty `<Message></Message>` where a real S3 ACL denial would
say something. Most likely edge throttling or a blip during the daily re-upload. The sync
script therefore treats **403 as retryable** with exponential backoff, which is not the
normal thing to do and is deliberate here.

**2. This bucket silently truncates large transfers.** Fetching `us_apt.json`
(`Content-Length: 41373726`) returned HTTP 200, exit code 0, and **35,655,188 bytes** —
the JSON ended mid-token at `"primary":tr`. Nothing in curl's output flagged it. An
archive quietly full of half-files is worse than no archive, so every download is checked
against `Content-Length` and resumed via HTTP `Range` (the bucket answers `206`) until the
byte count matches. A short file never reaches the store.

If you mirror this with something else, verify lengths. `curl -f -C -` alone will not
catch it.

## How the archive is laid out

```
<Root>\store\<ab>\<sha256>.gz     one gzipped copy per unique file content
<Root>\snapshots\2026-09-17.csv   what the bucket held that day
<Root>\logs\2026-09-17.log
<Root>\current\<country>\...      optional plain copy (-Materialize)
```

Content-addressed, so a file that didn't change between runs costs **zero** extra bytes —
the new snapshot just points at the blob already stored. Change detection is by ETag from
the bucket listing, so unchanged files aren't even downloaded.

Practically: first sync moves ~1.6 GB and stores roughly 300–400 MB gzipped. Later daily
runs typically transfer only the countries that actually changed.

## Restoring

```powershell
.\Restore-OpenAipSnapshot.ps1 -List
.\Restore-OpenAipSnapshot.ps1 -Date 2026-03-01 -Country us -Format json
```

Asking for a date with no snapshot gives you the nearest earlier one, so you can query by
calendar date without knowing the run history.

## Useful options

| Option | Effect |
|---|---|
| `-Country us,ca,mx` | Limit by country |
| `-Type apt,asp,nav` | Limit by object type |
| `-Format json,geojson` | Limit by format |
| `-ListOnly` | Report what would sync, download nothing |
| `-Materialize` | Also write a plain uncompressed tree |
| `-KeepSnapshots 90` | Keep 90 snapshots, garbage-collect unreferenced blobs |

Cutting to `-Format json` alone drops the job from 1.59 GB to ~405 MB, since geojson,
ndgeojson, cup and xml are the same data in other encodings.

Exit code is 1 if any object failed, so the scheduled task shows a failure. A failed
object keeps its previous good version in the new snapshot rather than leaving a hole.

## Running from the SD card

The card is 28.9 GB exFAT with 32 KB clusters. Three things follow from that:

- **Space is fine.** ~1.6 GB raw compresses to roughly 300-400 MB of blobs. Small files
  (many `.cup` exports are a few hundred bytes) each round up to a 32 KB cluster, so
  budget ~165 MB of slack at full 5,275-object coverage. Still nothing against 28.9 GB.
- **exFAT has no journaling.** Pulling the card mid-sync can corrupt the filesystem, not
  just the file being written. Let a run finish, or eject properly. Downloads land in
  `archive\.work` and only move into the store once complete, so an interrupted run
  costs you that run, not the existing archive.
- **Drive letters move.** The scripts locate their archive relative to themselves, so
  they work at any letter. The *scheduled task* does not � it stores the absolute path
  from registration time. If the card comes back as something other than `F:`,
  re-run `Register-OpenAipMirrorTask.ps1` from the card.

Because the task only runs when the card is present, check the log after a run:
`archive\logs\<date>.log`. A run that failed to find the card leaves no log at all.

## Licensing — check before this feeds anything shipping

openAIP's export page states the data is under **Creative Commons
Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**. Keeping a private archive is ordinary
use. Using it in a commercial product is what the NC term restricts, and iFly is
commercial — so if this is headed anywhere near shipping data, confirm the current terms
with openAIP directly and get written permission rather than relying on this note. I read
this off their site, not from a license agreement, and their `/terms` URL 404s.

Attribution is required even for non-commercial use.
