# Owner-run migration, frozen v1

Nothing in this runbook has been executed against production by this agent.
Run local steps in a checkout containing `redesign/data-contract` (after merging
with the other redesign sections). Use an external build volume, not gitignored
production catalogs copied into this repository. Keep old viewer URLs and old
Worker releases available throughout rollout. Do not overwrite any versioned key.

## Local setup and inputs

From the repository root, set these paths to the owner's actual local files.
The `/Volumes/...` values below are placeholders the owner replaces, not paths
this runbook verified:

```sh
export NEXT_BUILD=/Volumes/projects/archive-next
export NEXT_ERAS=/Volumes/drive/pmtiles
export NEXT_CHARTS=/Volumes/drive/pmtiles_charts
export NEXT_CATALOG=/Volumes/projects/archive-catalog
export NEXT_VECTOR=/Volumes/projects/protomaps_basemap/protomaps-20260826.pmtiles
mkdir -p "$NEXT_BUILD/plans" "$NEXT_BUILD/raw/sectionals/chart" "$NEXT_BUILD/overview/sectionals" "$NEXT_BUILD/versioned/sectionals" "$NEXT_BUILD/next"
python3 -m venv "$NEXT_BUILD/venv"
"$NEXT_BUILD/venv/bin/pip" install Pillow==12.1.0 boto3
```

`NEXT_CATALOG` must contain today's `airfields.json`, `timeline_data.json`, and
`coverage.json`. Rebuild those using the existing catalog pipeline if stale;
this section does not invent unavailable production inputs. The Python manifest
builder also requires Node on PATH for its checked-in schema validator.

For raster rendering, install the pinned packages outside the repository:

```sh
npm install --prefix "$NEXT_BUILD/render-runtime" @playwright/test@1.58.2 maplibre-gl@6.11.2 @protomaps/basemaps@5.7.2 pmtiles@3.0.6
"$NEXT_BUILD/render-runtime/node_modules/.bin/playwright" install chromium
git clone --depth 1 https://github.com/protomaps/basemaps-assets.git "$NEXT_BUILD/assets"
```

Verify that this checkout contains `assets/fonts/Noto Sans Regular/0-255.pbf`
and `assets/sprites/v4/dark.json` / `dark.png`. The renderer uses the entire fonts
and sprites trees for real city labels, not just the ASCII proof font.
Record the assets git commit alongside the output for reproducibility.

## 1. Versioned copies — preserve legacy keys

Build a local mirror whose paths map to the original R2 namespaces. Copy or
hard-link the era files into `raw/sectionals/` and the slug/date chart tree into
`raw/sectionals/chart/`. (Use a filesystem copy if the build volume differs.)
Keep basemap and airspace trees separate from the era tree. Then list what R2
actually holds and plan from that inventory; the mirror only supplies hashes:

```sh
rsync -a "$NEXT_ERAS/" "$NEXT_BUILD/raw/sectionals/"
rsync -a "$NEXT_CHARTS/" "$NEXT_BUILD/raw/sectionals/chart/"
rclone lsjson -R --files-only r2:charts --include 'sectionals/**' > "$NEXT_BUILD/plans/listing.json"
"$NEXT_BUILD/venv/bin/python" scripts/next_version_archives.py --listing "$NEXT_BUILD/plans/listing.json" --dir "$NEXT_BUILD/raw/sectionals" --prefix sectionals --out "$NEXT_BUILD/plans/originals.json"
```

`rclone lsjson` writes `Path` relative to the listed root, so with `r2:charts`
as the root it is the bucket key (`sectionals/...`). Listing records accept
`Path`, S3 `Key` or `key`; a `local` field can select a particular mirror
filename; keys outside `--prefix` are skipped, so a whole-bucket listing works.
Every record is `source: remote`: the object exists in R2 under `old`, and the
plan is a server-side copy to `{stem}.{sha256[:12]}.pmtiles`. Legacy chart
sources use extension-less `sectionals/chart/{slug}/{date}` keys; destinations
always include `.pmtiles`. A listed object with no mirror file fails the plan;
mirror files absent from the listing are not planned — diff them yourself.

The mirror must hold R2's current bytes. A stale mirror would stamp a copy with
a hash its bytes do not have and silently undo a same-key republish, so three
guards enforce it: the plan refuses a mirror file whose size differs from the
listing's `Size`; the `.sh` asserts each source's R2 size (`rclone size --json`)
before copying it; `--execute` asserts `ContentLength` and, for single-part
ETags, the mirror md5 recorded in the plan. Freeze source writers from listing
through copying. Without a local mirror, listing records must have a full
trusted `sha256` (ETag is insufficient), or the owner must explicitly add
`--read-remote` to stream/hash the listed objects. This agent did not use that
option. Output plans retain full SHA256, size, min/max zoom and outward-rounded
bounds.

### Without a mirror: stream from R2

When no mirror is mounted, hash the objects by streaming them (R2 egress is
free). `scripts/next_stream_hash.sh` splits the listing into shards, runs
`next_version_archives.py --read-remote --stubs` per shard with credentials
taken from the rclone `r2:` remote for that process only, then merges the shard
plans into `plan.json` + `plan.json.sh`:

```sh
scripts/next_stream_hash.sh "$NEXT_BUILD/plans/listing.json" "$NEXT_BUILD/plans/sectionals" "$HOME/archive-next-build/stubs" sectionals 12
```

Every object is read once; a dropped connection retries that object only. A
single stream from a home connection ran at about 8 MB/s on 2026-10-02 and six
streams at 29 MB/s, so shard. `--stubs` writes, under each versioned name, a
sparse file holding the archive's header, root, metadata and leaf directories
with a hole where the tile data would be: `next_build_manifest.py --dir` reads
exactly those bytes, so the manifest can be built with no local copy of any
archive. The stub directory must be on a local APFS disk (sparse files; the
apparent size equals the archives'). Plans from `--read-remote` carry the
object's `size` and `md5`, so the same `.sh` and `--execute` guards apply.

**Owner-only storage operation:** configure the rclone `r2:` remote, inspect
`originals.json.sh`, and execute the reviewed copy commands:

```sh
sh "$NEXT_BUILD/plans/originals.json.sh"
```

Alternatively the Python tool can execute copies only with `--execute` and
`R2_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` explicitly
present in the environment:

```sh
"$NEXT_BUILD/venv/bin/python" scripts/next_version_archives.py --listing "$NEXT_BUILD/plans/listing.json" --dir "$NEXT_BUILD/raw/sectionals" --prefix sectionals --out "$NEXT_BUILD/plans/originals.json" --execute
```

That path uses boto3's managed multipart copy for multi-GB archives. The Python path refuses existing destinations unless their full SHA256 metadata
matches, and guards source copies with the source ETag. The shell
path uses rclone server-side copy with `--immutable`. Copying existing bytes
creates new names and leaves the old viewer operational. Hash local sources at
storage speed; at 500 MB/s a 1 TB mirror takes roughly 33 minutes just to hash.
Remote hashing transfers all bytes and can be much slower.

## 2. Add overviews and raster basemap locally

Generate each era into the external overview directory. Do not run this over
per-chart artifacts: C3 requires overviews on era mosaics; solo charts advertise
their own zooms through pmz.

```sh
"$NEXT_BUILD/venv/bin/python" - <<'PY'
import os, subprocess
from pathlib import Path
source=Path(os.environ['NEXT_ERAS'])
out=Path(os.environ['NEXT_BUILD'])/'overview/sectionals'
for p in sorted(source.glob('*.pmtiles')):
    subprocess.run([os.environ['NEXT_BUILD']+'/venv/bin/python',
                    'scripts/next_add_overviews.py',str(p),str(out/p.name)],check=True)
PY
"$NEXT_BUILD/venv/bin/python" scripts/next_version_archives.py --dir "$NEXT_BUILD/overview/sectionals" --prefix sectionals --out "$NEXT_BUILD/plans/overviews.json"
```

Every original stored tile is retained byte-for-byte. New levels are RGBA
LANCZOS/WebP q80; new archive bytes require **new hashes**. Never upload overviews
onto the versioned copies made in step 1. The overview plan is a **local upload
plan** (`source: local`), not a server-side copy plan: its `old` names are the
legacy era keys, which do exist in R2 — with different bytes (no overviews). A
server-side copy would have put the legacy bytes under the overview hash with
`--immutable`, so the tool emits `rclone copyto <local file> r2:charts/<new>
--immutable` uploads for it, and `--execute` uploads with boto3. The basemap and
airspace plans below are local upload plans too.

Render a small real-cutout proof before the full basemap:

```sh
"$NEXT_BUILD/venv/bin/python" scripts/next_render_basemap.py --source "$NEXT_VECTOR" --assets "$NEXT_BUILD/assets" --modules "$NEXT_BUILD/render-runtime/node_modules" --out "$NEXT_BUILD/basemap-proof" --bbox -98 35 -97 36 --minzoom 7 --maxzoom 9
"$NEXT_BUILD/venv/bin/python" scripts/next_render_basemap.py --estimate
```

Inspect labels, neighboring seams, roads, terrain, coastline and dark colors.
For a full build, omit `--bbox` and use `--minzoom 0 --maxzoom 13` (the defaults):

```sh
"$NEXT_BUILD/venv/bin/python" scripts/next_render_basemap.py --source "$NEXT_VECTOR" --assets "$NEXT_BUILD/assets" --modules "$NEXT_BUILD/render-runtime/node_modules" --out "$NEXT_BUILD/basemap-full"
```

Use a fresh output directory per source/assets combination. Intermediate
`tiles/z/x/y.webp` and `render-jobs.ndjson` are build scratch, not publication
artifacts. Four-and-a-half million tiles can require substantial inode count,
disk space, packing memory and a week or more of serial rendering. The packer
spools contents to disk; its filename/directory/hash indexes remain in memory.
A interrupted render fails without publishing; rebuilding the same directory
rerenders requested tiles and replaces each output. Do not mix bbox runs there.

Stage the final basemap (choose the date/name) and a selected current local
**existing airspace** archive before versioning them. The airspace builder's
normal output is the source; this migration does not regenerate historical cycles.

```sh
mkdir -p "$NEXT_BUILD/overlay-archives/basemap" "$NEXT_BUILD/overlay-archives/airspace"
cp "$NEXT_BUILD/basemap-full/raster.pmtiles" "$NEXT_BUILD/overlay-archives/basemap/raster-20261003.pmtiles"
# Owner: copy the selected airspace .pmtiles into overlay-archives/airspace/.
"$NEXT_BUILD/venv/bin/python" scripts/next_version_archives.py --dir "$NEXT_BUILD/overlay-archives/basemap" --prefix basemap --out "$NEXT_BUILD/plans/basemap.json"
"$NEXT_BUILD/venv/bin/python" scripts/next_version_archives.py --dir "$NEXT_BUILD/overlay-archives/airspace" --prefix airspace --out "$NEXT_BUILD/plans/airspace.json"
```

Set a feature to null by leaving its archive out of the staging directory.
The builder refuses multiple basemap/airspace builds to avoid picking arbitrarily.

## 3. Stage final versioned files, overlays and manifest

This local staging snippet selects new overview eras, original per-chart
artifacts, and the selected overlay archives. It checks complete file hashes
again while copying. Empty airspace/basemap plans are allowed if disabled.

```sh
"$NEXT_BUILD/venv/bin/python" - <<'PY'
import os,json,shutil,hashlib
from pathlib import Path
root=Path(os.environ['NEXT_BUILD']); stage=root/'versioned'; final=[]
originals=json.loads((root/'plans/originals.json').read_text())
sets=[(root/'overview/sectionals',json.loads((root/'plans/overviews.json').read_text()),'sectionals/'),
      (root/'raw/sectionals',[r for r in originals if '/chart/' in r['old']],'sectionals/'),
      (root/'overlay-archives/basemap',json.loads((root/'plans/basemap.json').read_text()),'basemap/'),
      (root/'overlay-archives/airspace',json.loads((root/'plans/airspace.json').read_text()),'airspace/')]
for directory,records,prefix in sets:
    for r in records:
        rel=r['old'].removeprefix(prefix)
        p=directory/rel
        if not p.exists() and '/chart/' in r['old']: p=p.with_name(p.name+'.pmtiles')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
        assert h.hexdigest()==r['sha256'],str(p)
        dst=stage/r['new'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        final.append(r)
(root/'plans/final.json').write_text(json.dumps(final,indent=2))
PY
"$NEXT_BUILD/venv/bin/python" scripts/next_build_overlays.py --airfields "$NEXT_CATALOG/airfields.json" --timeline "$NEXT_CATALOG/timeline_data.json" --plan "$NEXT_BUILD/plans/final.json" --out "$NEXT_BUILD/next" --margin 2
"$NEXT_BUILD/venv/bin/python" scripts/next_build_manifest.py --dir "$NEXT_BUILD/versioned" --coverage "$NEXT_CATALOG/coverage.json" --overlays "$NEXT_BUILD/next/overlays.json" --out "$NEXT_BUILD/next"
```

The manifest reads headers/directories only, so the staging hash check is
necessary (or point `--dir` at the sparse stubs written by `--stubs`, whose
names already carry the streamed hashes). It computes z6 coverage from actual entries and validates the schema.
If it exceeds 80,000 bytes gzip, stop publication and resolve
[CHANGE_REQUEST_C2.md](CHANGE_REQUEST_C2.md) with the other sections. The temporary
`--allow-over-budget` option is explicit, not an automatic format change; the
validator needs the same option. No start-only dates are emitted. Existing
coverage.json contents are embedded verbatim, and all era records sort
chronologically. Pin building fails if any chart pm reference lacks a matching
version/zoom header. Correct incomplete mirrors/plans before retrying.

**Owner-only publication:** upload versioned files, then binary/details/shards,
then the final manifest. `--immutable` prevents silent replacement of a published
version. Do not publish `overlays.json` (it is a local build descriptor).

```sh
rclone copy "$NEXT_BUILD/versioned/" r2:charts/ --immutable
rclone copy "$NEXT_BUILD/next/" r2:charts/next/ --immutable --exclude overlays.json --exclude 'manifest.*.json'
rclone copy "$NEXT_BUILD/next/" r2:charts/next/ --immutable --include 'manifest.*.json'
```

A manifest is a commit point: every file it references must already be complete.
Record its URL, full SHA256, plans and local build measurements. New data names
are immutable; the existing file proxy's day-long cache policy remains unchanged
for ordinary file URLs. C1 tile/metadata responses have their explicit year TTL.

## 4. Worker staging, then owner production deploy

Confirm a **complete** object listing contains no keys beginning `t/`. Enforce
that reservation on future ingestion. That is the assumption that prevents
virtual `/t/` paths from shadowing real R2 objects. Keep the existing raw routes,
BUCKET and optional TILES bindings; no wrangler.toml route was added here.

Run the local acceptance checks from the merged repository root:

```sh
npm run test:workers
python3 -m unittest discover -s tests -p 'test_next_*.py'
node next/contract/fixtures/make_fixtures.mjs
node next/contract/fixtures/serve.mjs
```

In another terminal, use the path printed by the generator / fixture-index.json:

```sh
node next/contract/validate.mjs --manifest next/contract/fixtures/out/next/manifest.380d9b057640.json --base http://127.0.0.1:8765
```

That filename is deterministic with the default generator and may change when
fixtures change; `fixtures/out/fixture-index.json` is authoritative. The server
imports the Worker, not an independent tile implementation. It binds localhost.

Deploy this Worker to an owner-configured **staging** binding/domain first, using
the existing deployment workflow. There is no staging environment defined by this
change; use a separate account/bucket or a reviewed staging configuration.
Validate a staged manifest against that Worker:

```sh
node next/contract/validate.mjs --manifest /absolute/path/to/staged/manifest.HASH.json --base https://STAGING-DATA-DOMAIN/
```

Also curl a known gzip airspace tile with `--raw`: verify stored gzip magic,
Content-Encoding, content length, and that ordinary clients transparently decode
it. Verify HEAD bodylessness and unchanged old viewer range/ETag behavior using
both the old production viewer and a staging copy. Node tests alone cannot prove
Cloudflare's compression/cancellation/cache behavior.

Only after those checks, the owner deploys through the existing Worker directory:

```sh
cd worker
npx wrangler deploy
```

Then run the same smoke check on the production manifest URL and data domain.
The old viewer keeps using its original URLs. The shell section must separately
build the new page with the selected immutable manifest URL; do not repoint it
before the Worker and all referenced data are live. Follow the owning deployment
workflow for that frontend release.

## 5. Garbage collection and rollback

Keep every old version while **any published manifest** references it, including
manifests referenced by old frontend builds, rollback releases, cached pages and
preview deployments. Remove a manifest from the published/rollback registry only
when its release is formally retired. The default grace is **400 days after its
last published reference is retired**, covering the 365-day tile TTL plus 35 days.
The grace applies to era/chart/overlay archives, airfield pairs and pin folders.
Freeze collection during migration and rollback incidents. Compare against the
complete retained manifest registry and all legacy viewer/bundle references,
produce a deletion plan, review it, and only then delete. Do not use "last seen
request" alone as proof an object is unreferenced. The agent supplies no automatic
destructive collector.

Rollback the frontend to its recorded previous manifest/build URL. Old archives
remain usable because no versioned bytes changed. If the Worker must be rolled
back to a release without `/t/`, first restore the old viewer: the new viewer
requires `/t/`. Restore the previous Worker through the account's existing release
workflow. Preserve newly published objects for diagnosis; deleting them does not
help rollback and breaks cached/new frontend builds. Revert the next feature flag
or entry URL before removing any required endpoint.

## Cost and duration planning

Use measured bytes and elapsed times printed by the builders, not synthetic
fixture timing, to approve a full migration. Copying temporarily doubles storage
for the selected source bytes; overviews create another version until references
retire. Full-sheet per-chart originals remain required by old solo-view URLs.
R2 Standard's published rates are $0.015/GB-month, $4.50/million Class A operations
and $0.36/million Class B operations, before free tier and rounding; Internet
egress has no fee. Source: [R2 pricing](https://developers.cloudflare.com/r2/pricing/)
checked October 2, 2026. Multipart copies count their parts as well as completion.
At these rates one extra 1 TB retained for a month is about $15; an estimated
100 GB raster adds about $1.50/month. CPU/rendering hosts, Worker invocations,
Analytics Engine usage and temporary disk are separate costs.

The measured cold 30-tile viewport uses three R2 reads with both endpoints;
`/t/` reduces client requests and metadata payload, not block reads in this case.
The full basemap estimate is 4,559,534 tiles, about 100 GB / 190 serial hours at
22 KB / 0.15 s per tile. Sparse synthetic overviews took 0.179 s for 64 z8 tiles
and added 43.9% to that z8-only source; archives also containing z9–11 should be
measured independently. Manifest/overlays primarily parse directories and JSON;
reading hashes and decoding overview source tiles will dominate slow disks.
Do not schedule garbage collection or frontend cutover based on these estimates
without real-input validation and a resolved manifest budget.
