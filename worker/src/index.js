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

// Charts change on a 56-day cycle, so cache aggressively at both edge and
// browser and lean on ETag revalidation after expiry. Trade-off: a file
// re-uploaded under the same name can be served stale for up to a day.
const CACHE_CONTROL = "public, max-age=86400, stale-while-revalidate=3600";

// Client validators are compared here in the Worker, never pushed down to R2:
// BUCKET.get() with onlyIf + range throws when the precondition fails (the
// normal file-unchanged case), which turned every browser revalidation that
// missed the edge cache into a 503. Accepts the header's list/weak forms.
function etagMatches(ifNoneMatch, httpEtag) {
  if (!ifNoneMatch || !httpEtag) return false;
  return ifNoneMatch.split(",").some((tag) => {
    tag = tag.trim();
    if (tag === "*") return true;
    if (tag.startsWith("W/")) tag = tag.slice(2);
    return tag === httpEtag;
  });
}

// Bounded ranges up to this size are buffered and shared across concurrent
// requests in the isolate: tile herds fan 80+ identical reads out in the same
// millisecond, far faster than the first response can populate the edge
// cache, so without this every one of them pays its own R2 read. Larger or
// unbounded reads (whole multi-GB archives) stream straight through.
const COALESCE_MAX_BYTES = 1 << 20;
const inflightRanges = new Map();

// 416 with the real object size lets range clients (pmtiles) recover by
// refetching the header instead of treating the archive as broken.
async function rangeNotSatisfiable(env, key, origin) {
  let size = "*";
  try {
    const head = await env.BUCKET.head(key);
    if (head) size = head.size;
  } catch (_) {}
  return withCors(
    new Response("Requested range not satisfiable", {
      status: 416,
      headers: { "content-range": `bytes */${size}`, "cache-control": "no-store" },
    }),
    origin
  );
}

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
    let key;
    try {
      key = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    } catch (_) {
      return withCors(new Response("Bad request", { status: 400 }), origin);
    }
    if (!key) return new Response("Not found", { status: 404 });

    // HEAD: answer from object metadata alone. Routing HEAD through the GET
    // path streams the whole object into the edge cache via waitUntil.
    if (request.method === "HEAD") {
      let head;
      try {
        head = await env.BUCKET.head(key);
      } catch (err) {
        console.error(
          `R2 head failed for ${key}: ${String((err && err.message) || err)}`
        );
        return withCors(
          new Response(null, {
            status: 503,
            headers: { "retry-after": "1", "cache-control": "no-store" },
          }),
          origin
        );
      }
      if (!head) return new Response("Not found", { status: 404 });
      const inm = request.headers.get("if-none-match");
      if (etagMatches(inm, head.httpEtag)) {
        return withCors(
          new Response(null, {
            status: 304,
            headers: { etag: head.httpEtag, "cache-control": CACHE_CONTROL },
          }),
          origin
        );
      }
      const headers = new Headers();
      head.writeHttpMetadata(headers);
      headers.set("etag", head.httpEtag);
      headers.set("accept-ranges", "bytes");
      headers.set("cache-control", CACHE_CONTROL);
      headers.set("content-length", String(head.size));
      headers.set("x-cache", "HEAD");
      return withCors(new Response(null, { status: 200, headers }), origin);
    }

    const range = parseRange(request.headers.get("range"));
    // An inverted range (end < start) can never be satisfied; R2 would throw.
    if (range && range.end !== undefined && range.end < range.offset) {
      return rangeNotSatisfiable(env, key, origin);
    }
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
      if (etagMatches(ifNoneMatch, cachedEtag)) {
        return withCors(
          new Response(null, {
            status: 304,
            headers: { etag: cachedEtag, "cache-control": CACHE_CONTROL },
          }),
          origin
        );
      }
      response = fromCached(cached);
    } else if (
      r2Range &&
      r2Range.length !== undefined &&
      r2Range.length <= COALESCE_MAX_BYTES
    ) {
      // Miss on a bounded, buffer-sized range: share one R2 read among every
      // concurrent request for the same range. The leader buffers the body
      // (tile reads are KBs, capped at COALESCE_MAX_BYTES) so joiners can each
      // build their own response from the same bytes.
      const inflightKey = cacheKey.url;
      let flight = inflightRanges.get(inflightKey);
      const isLeader = !flight;
      if (!flight) {
        flight = (async () => {
          const object = await env.BUCKET.get(key, { range: r2Range });
          if (!object) return null;
          const headers = new Headers();
          object.writeHttpMetadata(headers);
          headers.set("etag", object.httpEtag);
          headers.set("accept-ranges", "bytes");
          headers.set("cache-control", CACHE_CONTROL);
          // Clamp to the object size: R2 truncates a bounded range that
          // overruns EOF, and the headers must match the bytes delivered.
          const total = object.size;
          const rangeStart = range.offset;
          const rangeEnd = Math.min(rangeStart + r2Range.length, total) - 1;
          headers.set("content-range", `bytes ${rangeStart}-${rangeEnd}/${total}`);
          headers.set("content-length", String(rangeEnd - rangeStart + 1));
          const buffer = await object.arrayBuffer();
          return { buffer, headers, httpEtag: object.httpEtag };
        })();
        inflightRanges.set(inflightKey, flight);
        // Evict on settlement, success or failure — a rejected flight must
        // never be joinable later, or one transient error would fan out.
        flight.then(
          () => inflightRanges.delete(inflightKey),
          () => inflightRanges.delete(inflightKey)
        );
      }

      let result;
      try {
        result = await flight;
      } catch (err) {
        const msg = String((err && err.message) || err);
        if (/satisfiable|invalid range|10039/i.test(msg)) {
          return rangeNotSatisfiable(env, key, origin);
        }
        console.error(`R2 get failed for ${key}: ${msg}`);
        return withCors(
          new Response("Upstream storage error", {
            status: 503,
            headers: { "retry-after": "1", "cache-control": "no-store" },
          }),
          origin
        );
      }

      if (!result) return new Response("Not found", { status: 404 });

      // Only the leader fills the edge cache; N joiners doing N puts of the
      // same entry is wasted work.
      if (isLeader) {
        ctx.waitUntil(
          cache
            .put(
              cacheKey,
              new Response(result.buffer, { status: 200, headers: result.headers })
            )
            .catch(() => {})
        );
      }

      // Client's cached copy is still current -> bodyless 304.
      if (etagMatches(ifNoneMatch, result.httpEtag)) {
        return withCors(
          new Response(null, {
            status: 304,
            headers: { etag: result.httpEtag, "cache-control": CACHE_CONTROL },
          }),
          origin
        );
      }

      response = new Response(result.buffer, {
        status: 206,
        headers: result.headers,
      });
      if (!isLeader) cacheStatus = "COALESCE";
    } else {
      // Miss on a whole-file, open-ended, or oversized read: stream straight
      // from R2 without buffering.
      //
      // R2 rejections must not escape as uncaught exceptions (opaque 500s):
      // a range past EOF is the client's problem (416, recoverable), anything
      // else is transient storage trouble the client should retry (503).
      let object;
      try {
        object = await env.BUCKET.get(key, r2Range ? { range: r2Range } : {});
      } catch (err) {
        const msg = String((err && err.message) || err);
        if (/satisfiable|invalid range|10039/i.test(msg)) {
          return rangeNotSatisfiable(env, key, origin);
        }
        console.error(`R2 get failed for ${key}: ${msg}`);
        return withCors(
          new Response("Upstream storage error", {
            status: 503,
            headers: { "retry-after": "1", "cache-control": "no-store" },
          }),
          origin
        );
      }

      if (!object) return new Response("Not found", { status: 404 });

      // Client's cached copy is still current -> bodyless 304. The unread R2
      // body is dropped; deliberately no cache fill here (streaming a multi-GB
      // surrogate into the edge cache on a revalidation would be pathological).
      if (etagMatches(ifNoneMatch, object.httpEtag)) {
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
        // Clamp to the object size: R2 truncates a bounded range that overruns
        // EOF, and the headers must match the bytes actually delivered.
        const total = object.size;
        const rangeStart = range.offset;
        const rangeEnd =
          Math.min(rangeStart + (r2Range.length ?? total - rangeStart), total) - 1;
        headers.set("content-range", `bytes ${rangeStart}-${rangeEnd}/${total}`);
        headers.set("content-length", String(rangeEnd - rangeStart + 1));
      } else {
        headers.set("content-length", String(object.size));
      }

      // Store a 200 surrogate (206 is uncacheable), then serve the same bytes
      // flipped back to 206 for range requests via fromCached().
      const surrogate = new Response(object.body, { status: 200, headers });
      // A failed cache fill (e.g. client disconnect mid-stream) must not
      // surface as an invocation error; the response itself already went out.
      ctx.waitUntil(cache.put(cacheKey, surrogate.clone()).catch(() => {}));
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
