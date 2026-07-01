// Cloudflare Worker: proxies R2 range requests for archive.aero PMTiles,
// logging each tile-shaped read to Analytics Engine (sampled).
//
// Bindings (configured in Cloudflare dashboard or wrangler.toml):
//   BUCKET  -> R2 bucket "charts"
//   TILES   -> Analytics Engine dataset "tile_logs"
//
// Vars:
//   ALLOWED_ORIGIN  default "*"
//   SAMPLE_RATE     default "0.05" (5% of qualifying reads)
//   METADATA_BYTES  default "1024" (range reads <= this many bytes are not logged)

const CORS_BASE = {
  "access-control-allow-methods": "GET, HEAD, OPTIONS",
  "access-control-allow-headers": "Range, If-Match, If-None-Match",
  "access-control-expose-headers":
    "Content-Length, Content-Range, ETag, Accept-Ranges",
  "access-control-max-age": "86400",
};

function parseRange(header) {
  if (!header) return null;
  const match = header.match(/^bytes=(\d+)-(\d+)?$/);
  if (!match) return null;
  const offset = parseInt(match[1], 10);
  const end = match[2] !== undefined ? parseInt(match[2], 10) : undefined;
  return { offset, end };
}

function withCors(response, origin) {
  const headers = new Headers(response.headers);
  headers.set("access-control-allow-origin", origin);
  for (const [k, v] of Object.entries(CORS_BASE)) headers.set(k, v);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

export default {
  async fetch(request, env, ctx) {
    const startedAt = Date.now();
    const origin = env.ALLOWED_ORIGIN || "*";

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "access-control-allow-origin": origin,
          ...CORS_BASE,
        },
      });
    }

    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method not allowed", { status: 405 });
    }

    const url = new URL(request.url);
    const key = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    if (!key) return new Response("Not found", { status: 404 });

    const range = parseRange(request.headers.get("range"));

    const cache = caches.default;
    let response = await cache.match(request);
    let cacheStatus = "MISS";

    if (response) {
      cacheStatus = "HIT";
    } else {
      const r2Range = range
        ? range.end !== undefined
          ? { offset: range.offset, length: range.end - range.offset + 1 }
          : { offset: range.offset }
        : undefined;

      const object = r2Range
        ? await env.BUCKET.get(key, { range: r2Range })
        : await env.BUCKET.get(key);

      if (!object) return new Response("Not found", { status: 404 });

      const headers = new Headers();
      object.writeHttpMetadata(headers);
      headers.set("etag", object.httpEtag);
      headers.set("accept-ranges", "bytes");
      // .pmtiles archives are immutable per filename; cache aggressively.
      headers.set("cache-control", "public, max-age=604800, immutable");

      if (range) {
        const total = object.size;
        const rangeStart = range.offset;
        const rangeLen = r2Range.length ?? total - rangeStart;
        const rangeEnd = rangeStart + rangeLen - 1;
        headers.set("content-range", `bytes ${rangeStart}-${rangeEnd}/${total}`);
        headers.set("content-length", String(rangeLen));
        response = new Response(object.body, { status: 206, headers });
      } else {
        headers.set("content-length", String(object.size));
        response = new Response(object.body, { status: 200, headers });
      }

      ctx.waitUntil(cache.put(request, response.clone()));
    }

    // Log to Analytics Engine (sampled, skip tiny header/dir reads).
    if (env.TILES) {
      const metadataBytes = parseInt(env.METADATA_BYTES || "1024", 10);
      const sampleRate = parseFloat(env.SAMPLE_RATE || "0.05");
      const rangeSize =
        range && range.end !== undefined ? range.end - range.offset + 1 : null;
      const isMetadata = rangeSize !== null && rangeSize <= metadataBytes;
      const isRangeOpen = range && range.end === undefined;

      if (!isMetadata && !isRangeOpen && Math.random() < sampleRate) {
        try {
          env.TILES.writeDataPoint({
            indexes: [key],
            blobs: [
              key,
              request.cf?.country || "XX",
              cacheStatus,
              (request.headers.get("user-agent") || "").slice(0, 256),
              request.headers.get("referer") || "",
            ],
            doubles: [
              rangeSize || 0,
              Date.now() - startedAt,
              range ? range.offset : 0,
            ],
          });
        } catch (_) {
          // Never let logging fail a tile request.
        }
      }
    }

    return withCors(response, origin);
  },
};
