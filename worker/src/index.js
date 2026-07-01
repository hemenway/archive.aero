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
    "Content-Length, Content-Range, ETag, Accept-Ranges, X-Cache",
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

// Files are re-uploaded under the same name while tuning, so cache only briefly
// (both edge and browser) and lean on ETag revalidation to refresh changed
// archives within ~5 min. stale-while-revalidate keeps panning smooth.
const CACHE_CONTROL = "public, max-age=300, stale-while-revalidate=60";

// Edge entries are stored as 200 surrogates because Cloudflare's Cache API will
// not store 206 responses. Flip back to 206 on the way out when the entry
// carries a content-range (i.e. it answered a byte-range request).
function fromCached(cached) {
  const status = cached.headers.get("content-range") ? 206 : 200;
  return new Response(cached.body, { status, headers: cached.headers });
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
    const ifNoneMatch = request.headers.get("if-none-match");

    const r2Range = range
      ? range.end !== undefined
        ? { offset: range.offset, length: range.end - range.offset + 1 }
        : { offset: range.offset }
      : undefined;

    // Cache key encodes the byte range as a query param (not a Range header, so
    // the entry is a plain cacheable request) — each distinct range caches on
    // its own and is shared across all visitors, draining R2 to ~1 read per
    // range per TTL window.
    const rangeTag = range ? `${range.offset}-${range.end ?? ""}` : "full";
    const cacheKey = new Request(`${url.origin}${url.pathname}?r=${rangeTag}`);

    const cache = caches.default;
    let response;
    let cacheStatus = "MISS";

    const cached = await cache.match(cacheKey);
    if (cached) {
      cacheStatus = "HIT";
      const cachedEtag = cached.headers.get("etag");
      // Revalidating client + unchanged file -> 304 straight from the edge,
      // no R2 read at all.
      if (ifNoneMatch && cachedEtag && ifNoneMatch === cachedEtag) {
        return withCors(
          new Response(null, {
            status: 304,
            headers: { etag: cachedEtag, "cache-control": CACHE_CONTROL },
          }),
          origin
        );
      }
      response = fromCached(cached);
    } else {
      // Miss: read from R2. Forward the client's validator so an unchanged file
      // costs a bodyless conditional read instead of a full range transfer.
      const getOptions = r2Range ? { range: r2Range } : {};
      if (ifNoneMatch) getOptions.onlyIf = { etagDoesNotMatch: ifNoneMatch };

      const object = await env.BUCKET.get(key, getOptions);

      if (!object) return new Response("Not found", { status: 404 });

      // onlyIf precondition failed -> the client's cached copy is still current.
      if (ifNoneMatch && !object.body) {
        return withCors(
          new Response(null, {
            status: 304,
            headers: { etag: object.httpEtag, "cache-control": CACHE_CONTROL },
          }),
          origin
        );
      }

      const headers = new Headers();
      object.writeHttpMetadata(headers);
      headers.set("etag", object.httpEtag);
      headers.set("accept-ranges", "bytes");
      headers.set("cache-control", CACHE_CONTROL);

      if (range) {
        const total = object.size;
        const rangeStart = range.offset;
        const rangeLen = r2Range.length ?? total - rangeStart;
        const rangeEnd = rangeStart + rangeLen - 1;
        headers.set("content-range", `bytes ${rangeStart}-${rangeEnd}/${total}`);
        headers.set("content-length", String(rangeLen));
      } else {
        headers.set("content-length", String(object.size));
      }

      // Store a 200 surrogate (206 is uncacheable), then serve the same bytes
      // flipped back to 206 for range requests via fromCached().
      const surrogate = new Response(object.body, { status: 200, headers });
      ctx.waitUntil(cache.put(cacheKey, surrogate.clone()));
      response = fromCached(surrogate);
    }

    // Surface the edge result so cache behaviour is visible in DevTools/curl.
    response.headers.set("x-cache", cacheStatus);

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
