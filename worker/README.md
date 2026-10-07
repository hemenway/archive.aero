# archive-aero-tiles Worker

Proxies R2 range requests for the PMTiles archives on `data.archive.aero`,
logging tile-shaped reads to Analytics Engine for popularity analysis and every
failed response to a separate error dataset. `/t/` is a virtual namespace that
serves decoded tiles for the `next/` viewer (`src/tiles.js`).

## Bindings

- `BUCKET` — R2 bucket `charts`: era archives at `sectionals/<start>_to_<end>.pmtiles`,
  per-chart archives at `sectionals/chart/<slug>/<date>`, the metadata bundles,
  `basemap/` and `airspace/`
- `TILES` — Analytics Engine dataset `tile_logs` (sampled successful tile reads)
- `TILE_ERRORS` — Analytics Engine dataset `tile_errors` (every response ≥ 400
  and every uncaught exception, unsampled)

## Vars

- `ALLOWED_ORIGIN` (default `*`)
- `SAMPLE_RATE` (default `0.05` — fraction of qualifying reads logged)
- `METADATA_BYTES` (default `1024` — skip range reads ≤ this size to avoid logging PMTiles header/directory traversal)

## Deploy

Either via wrangler:

```
cd worker
npx wrangler@latest deploy
```

Or paste `src/index.js` into a new Worker in the Cloudflare dashboard
and configure bindings to match `wrangler.toml`.

## Query logs

Cloudflare dashboard → Analytics & Logs → Analytics Engine → SQL API.
Top charts by request volume:

```sql
SELECT blob1 AS key, sum(_sample_interval * double4) AS requests
FROM tile_logs
WHERE timestamp > NOW() - INTERVAL '7' DAY
GROUP BY key
ORDER BY requests DESC
LIMIT 50
```

`double4` is `1 / SAMPLE_RATE` at the time of the read: the Worker logs only that
fraction of qualifying reads, on top of Analytics Engine's own sampling. Rows
written before 2026-10 lack it; multiply their `sum(_sample_interval)` by 20
(`SAMPLE_RATE` was 0.05 throughout). `blob3` is the edge result: `HIT`, `MISS`,
`COALESCE` (joined another request's R2 read) or `BYPASS` (`Cache-Control:
no-cache`, typically a pmtiles reload after an ETag change).

Failures by status and path:

```sql
SELECT blob2 AS status, blob1 AS path, count() AS n
FROM tile_errors
WHERE timestamp > NOW() - INTERVAL '1' DAY
GROUP BY status, path
ORDER BY n DESC
LIMIT 50
```

`tile_errors` blobs: path, status, country, `Range`, `Cache-Control`,
user agent, referer; doubles: status, milliseconds.
