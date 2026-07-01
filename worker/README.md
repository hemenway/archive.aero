# archive-aero-tiles Worker

Proxies R2 range requests for the PMTiles archives on `tiles.archive.aero`,
logging tile-shaped reads to Analytics Engine for popularity analysis.

## Bindings

- `BUCKET` — R2 bucket `charts` (PMTiles live under `sectionals/pmtiles/`)
- `TILES` — Analytics Engine dataset `tile_logs`

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
SELECT blob1 AS key, sum(_sample_interval) AS requests
FROM tile_logs
WHERE timestamp > NOW() - INTERVAL '7' DAY
GROUP BY key
ORDER BY requests DESC
LIMIT 50
```
