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

// Returns null for no Range header, { offset, end } for bytes=N-M / bytes=N-,
// { suffix } for bytes=-N, and { invalid: true } for anything else (multi-range,
// malformed). An unsupported form used to be treated as "no Range" and served
// the WHOLE object — one stray `bytes=-100` against a 4.4 GB era streamed all
// of it and tried to cache it; the RFC lets a server ignore Range, but not at
// that price. Unsupported forms are answered 416 instead.
function parseRange(header) {
  if (!header) return null;
  const match = header.match(/^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$/i);
  if (!match) return { invalid: true };
  const [, a, b] = match;
  if (a === "" && b === "") return { invalid: true };
  if (a === "") return { suffix: parseInt(b, 10) };
  const offset = parseInt(a, 10);
  const end = b !== "" ? parseInt(b, 10) : undefined;
  if (!Number.isSafeInteger(offset) || (end !== undefined && !Number.isSafeInteger(end))) {
    return { invalid: true };
  }
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

// Bounded reads are served from 1 MiB blocks aligned to the file, not from
// the exact byte range the client asked for. PMTiles stores tiles in Hilbert
// order, so a 30-tile viewport spans 2-5 blocks and a one-column pan almost
// never leaves them: caching per block turns most reads into edge hits, where
// caching per exact range (every viewport is a fresh set of ranges) hit ~6%.
// Each block is fetched from R2 once per colo per TTL, shared by every
// concurrent request in the isolate, and stored in the Cache API as a plain
// 200 keyed on the block index. Reads longer than a block, open-ended reads
// and whole-file reads stream straight through (see below).
const BLOCK_BYTES = 1 << 20;
const inflightBlocks = new Map();

function blockCacheKey(url, index) {
  return new Request(`${url.origin}${url.pathname}?b=${index}&bs=${BLOCK_BYTES}`);
}

// Fetch one block, from the edge cache or R2. Resolves to
//   { buffer, etag, total, contentType, source }   source: HIT | MISS | COALESCE
// or null when the object does not exist. Throws R2 errors through to the
// caller (a block start past EOF surfaces as R2's "not satisfiable").
async function getBlock(env, ctx, url, key, index, bypass = false) {
  const cache = caches.default;
  const cacheKey = blockCacheKey(url, index);
  // bypass: the client asked for fresh bytes (Cache-Control: no-cache, which
  // pmtiles sends via cache:"reload" when it detects an ETag mismatch). The
  // edge entry is version-less, so without this the retry re-read the same
  // stale block for up to a day after a same-key republish.
  const cached = bypass ? null : await cache.match(cacheKey);
  if (cached) {
    return {
      buffer: await cached.arrayBuffer(),
      etag: cached.headers.get("etag"),
      total: parseInt(cached.headers.get("x-object-size"), 10),
      contentType: cached.headers.get("content-type"),
      source: "HIT",
    };
  }

  const inflightKey = cacheKey.url;
  let flight = inflightBlocks.get(inflightKey);
  const isLeader = !flight;
  if (!flight) {
    flight = (async () => {
      const object = await env.BUCKET.get(key, {
        range: { offset: index * BLOCK_BYTES, length: BLOCK_BYTES },
      });
      if (!object) return null;
      const buffer = await object.arrayBuffer();
      const headers = new Headers();
      object.writeHttpMetadata(headers);
      return {
        buffer,
        etag: object.httpEtag,
        total: object.size,
        contentType: headers.get("content-type") || "application/octet-stream",
      };
    })();
    inflightBlocks.set(inflightKey, flight);
    // Evict on settlement, success or failure — a rejected flight must never
    // be joinable later, or one transient error would fan out.
    flight.then(
      () => inflightBlocks.delete(inflightKey),
      () => inflightBlocks.delete(inflightKey)
    );
  }
  const result = await flight;
  if (!result) return null;

  // Only the leader fills the edge cache; N joiners doing N puts is waste.
  if (isLeader) {
    const headers = new Headers({
      "content-type": result.contentType,
      "content-length": String(result.buffer.byteLength),
      etag: result.etag,
      "x-object-size": String(result.total),
      "cache-control": CACHE_CONTROL,
    });
    ctx.waitUntil(
      cache
        .put(cacheKey, new Response(result.buffer, { status: 200, headers }))
        .catch(() => {})
    );
  }
  return { ...result, source: isLeader ? "MISS" : "COALESCE" };
}

// Edge entries for streamed (non-block) reads are stored as 200 surrogates
// because Cloudflare's Cache API will not store 206 responses. Flip back to
// 206 on the way out when the entry carries a content-range.
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
      return withCors(new Response("Method not allowed", { status: 405, headers: { allow: "GET, HEAD, OPTIONS" } }), origin);
    }

    const url = new URL(request.url);
    let key;
    try {
      key = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    } catch (_) {
      return withCors(new Response("Bad request", { status: 400 }), origin);
    }
    if (!key) return withCors(new Response("Not found", { status: 404 }), origin);

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
      if (!head) return withCors(new Response("Not found", { status: 404 }), origin);
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

    let range = parseRange(request.headers.get("range"));
    if (range && range.invalid) return rangeNotSatisfiable(env, key, origin);
    if (range && range.suffix !== undefined) {
      // bytes=-N: the last N bytes. Resolve against the object size so it
      // takes the bounded block path like any other range.
      let head;
      try {
        head = await env.BUCKET.head(key);
      } catch (err) {
        console.error(`R2 head failed for ${key}: ${String((err && err.message) || err)}`);
        return withCors(new Response("Upstream storage error", {
          status: 503, headers: { "retry-after": "1", "cache-control": "no-store" } }), origin);
      }
      if (!head) return withCors(new Response("Not found", { status: 404 }), origin);
      if (range.suffix === 0) return rangeNotSatisfiable(env, key, origin);
      const offset = Math.max(0, head.size - range.suffix);
      range = { offset, end: head.size - 1 };
    }
    // An inverted range (end < start) can never be satisfied; R2 would throw.
    if (range && range.end !== undefined && range.end < range.offset) {
      return rangeNotSatisfiable(env, key, origin);
    }
    const ifNoneMatch = request.headers.get("if-none-match");
    const cc = (request.headers.get("cache-control") || "").toLowerCase();
    const bypassCache = /\bno-cache\b/.test(cc) || /\bno-cache\b/i.test(request.headers.get("pragma") || "");

    const r2Range = range
      ? range.end !== undefined
        ? { offset: range.offset, length: range.end - range.offset + 1 }
        : { offset: range.offset }
      : undefined;

    const cache = caches.default;
    let response;
    let cacheStatus = "MISS";

    if (r2Range && r2Range.length !== undefined && r2Range.length <= BLOCK_BYTES) {
      // Block path: a bounded read no longer than a block touches at most two
      // blocks. Fetch the first, clamp the range to the real object size, then
      // the second only if the clamped range still crosses into it.
      const firstIndex = Math.floor(range.offset / BLOCK_BYTES);
      let first;
      try {
        first = await getBlock(env, ctx, url, key, firstIndex, bypassCache);
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
      if (!first) return withCors(new Response("Not found", { status: 404 }), origin);

      const total = first.total;
      if (range.offset >= total) return rangeNotSatisfiable(env, key, origin);
      const rangeStart = range.offset;
      const rangeEnd = Math.min(range.end, total - 1);
      const lastIndex = Math.floor(rangeEnd / BLOCK_BYTES);

      let bytes;
      cacheStatus = first.source;
      if (lastIndex === firstIndex) {
        const from = rangeStart - firstIndex * BLOCK_BYTES;
        bytes = first.buffer.slice(from, from + (rangeEnd - rangeStart + 1));
      } else {
        let second;
        try {
          second = await getBlock(env, ctx, url, key, lastIndex, bypassCache);
        } catch (err) {
          console.error(`R2 get failed for ${key}: ${String((err && err.message) || err)}`);
          return withCors(
            new Response("Upstream storage error", {
              status: 503,
              headers: { "retry-after": "1", "cache-control": "no-store" },
            }),
            origin
          );
        }
        // Two blocks with different ETags straddle a republish. The bytes
        // would be a splice of two files; 416 makes the pmtiles client refetch
        // the header and start over instead.
        if (!second || second.etag !== first.etag) {
          return rangeNotSatisfiable(env, key, origin);
        }
        const from = rangeStart - firstIndex * BLOCK_BYTES;
        const head = first.buffer.slice(from);
        const tail = second.buffer.slice(0, rangeEnd - lastIndex * BLOCK_BYTES + 1);
        bytes = new Uint8Array(head.byteLength + tail.byteLength);
        bytes.set(new Uint8Array(head), 0);
        bytes.set(new Uint8Array(tail), head.byteLength);
        // Report the slower of the two block sources.
        const rank = { HIT: 0, COALESCE: 1, MISS: 2 };
        if (rank[second.source] > rank[cacheStatus]) cacheStatus = second.source;
      }

      // Client's cached copy is still current -> bodyless 304.
      if (etagMatches(ifNoneMatch, first.etag)) {
        return withCors(
          new Response(null, {
            status: 304,
            headers: { etag: first.etag, "cache-control": CACHE_CONTROL },
          }),
          origin
        );
      }

      response = new Response(bytes, {
        status: 206,
        headers: {
          "content-type": first.contentType,
          "content-length": String(rangeEnd - rangeStart + 1),
          "content-range": `bytes ${rangeStart}-${rangeEnd}/${total}`,
          etag: first.etag,
          "accept-ranges": "bytes",
          "cache-control": CACHE_CONTROL,
        },
      });
    } else {
      // Whole-file, open-ended, or longer-than-a-block read: stream straight
      // from R2 without buffering, keyed on the exact range in the edge cache.
      const rangeTag = range ? `${range.offset}-${range.end ?? ""}` : "full";
      const cacheKey = new Request(`${url.origin}${url.pathname}?r=${rangeTag}`);
      const cached = bypassCache ? null : await cache.match(cacheKey);
      if (cached) {
        cacheStatus = "HIT";
        const cachedEtag = cached.headers.get("etag");
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
      } else {
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

        if (!object) return withCors(new Response("Not found", { status: 404 }), origin);

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
      // Only bounded range reads that cannot be the archive's own metadata:
      // a whole-file GET (timeline_data.json, airfields.json, a bundle) and the
      // 16 KB header+root-directory probe at offset 0 are not tile views, and
      // they were being logged as such (2026-09-08 audit). Leaf-directory
      // reads deeper in the file cannot be told apart here without the
      // directory; scripts/analytics_heatmap.py classifies those.
      const isTileShaped = !!range && rangeSize !== null && range.offset > 0;

      if (isTileShaped && !isMetadata && !isRangeOpen && Math.random() < sampleRate) {
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
