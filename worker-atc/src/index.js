// atc worker — serves the flattened atchistory.org archive from R2 (atc-site)
// and, after cutover, 301s the old hostnames into archive.aero/atc/.
// Worklist: worklists/08_atchistory_migration.md
//
// Hostname behavior:
//   atc-staging.archive.aero   serve (X-Robots-Tag: noindex)
//   archive.aero/atc/*         serve
//   (www.)atchistory.org       MODE var: "serve" (cutover morning) or "redirect"
//
// URI scheme, canonical/alias resolution and the route table live in
// routes.js (W3C "Cool URIs don't change" — see that file's header).

import P_MAP from "./p_map.json" with { type: "json" };
import { PREFIX, keyCandidates, routeOldPath } from "./routes.js";

const NEW_ORIGIN = "https://archive.aero";
const OLD_HOSTS = new Set(["atchistory.org", "www.atchistory.org"]);

// Error pages share the archive.aero design language (Barlow display face,
// dark solid surfaces, plane mark — see design-de-ai conventions). Inline and
// self-contained: they must render even when R2 or the fonts host is down.
function pageShell(title, heading, inner) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${title}</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%231e90ff'><path d='M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z'/></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@600;700&display=swap">
<style>
  body { margin: 0; background: #0a0e12; color: #fff; line-height: 1.6;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 34rem; margin: 14vh auto 0; padding: 0 20px; }
  .brand { display: inline-flex; align-items: center; gap: 8px; color: #fff;
           font-family: Barlow, system-ui, sans-serif; font-weight: 700;
           text-decoration: none; font-size: 1.02rem; margin-bottom: 34px; }
  .brand svg { width: 18px; height: 18px; fill: #1e90ff; }
  h1 { font-family: Barlow, system-ui, sans-serif; font-weight: 700;
       font-size: 1.7rem; margin: 0 0 10px; }
  p { margin: 0 0 14px; color: rgba(255,255,255,0.75); }
  a { color: #1e90ff; text-decoration: none; }
  a:hover { color: #4da6ff; text-decoration: underline; }
  .links { margin-top: 26px; padding-top: 16px;
           border-top: 1px solid rgba(255,255,255,0.09); }
  .links a { display: inline-block; margin-right: 8px; margin-bottom: 8px;
             padding: 7px 13px; border: 1px solid rgba(255,255,255,0.22);
             border-radius: 8px; color: #fff; font-weight: 600;
             font-family: Barlow, system-ui, sans-serif; font-size: 0.9rem; }
  .links a:hover { text-decoration: none; border-color: #4da6ff; color: #4da6ff; }
</style>
</head>
<body>
<main>
  <a class="brand" href="${NEW_ORIGIN}/"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg>archive.aero</a>
  <h1>${heading}</h1>
  ${inner}
</main>
</body>
</html>`;
}

const PAGE_404 = pageShell(
  "Page not found — Air Traffic Control History",
  "Page not found",
  `<p>There's no page at this address in the Air Traffic Control History
collection.</p>
<p>If a link from the old <strong>atchistory.org</strong> brought you here, it
should have landed on the page's new home — tell
<a href="mailto:archive@atchistory.org">archive@atchistory.org</a> and it will
be fixed.</p>
<div class="links">
  <a href="${NEW_ORIGIN}/atc/">ATC History home</a>
  <a href="${NEW_ORIGIN}/">Chart viewer</a>
</div>`);

const PAGE_410 = pageShell(
  "Gone — atchistory.org",
  "This page is gone for good",
  `<p>The old atchistory.org forum and WordPress internals were retired when
the collection moved to its permanent home. The archive itself — every
article, photo, and document — is preserved at
<strong>archive.aero/atc</strong>.</p>
<div class="links">
  <a href="${NEW_ORIGIN}/atc/">ATC History home</a>
</div>`);

const OLD_ROBOTS = `User-agent: *
Allow: /

Sitemap: https://www.atchistory.org/wp-sitemap.xml
`;

function log(env, host, path, status, referer) {
  try {
    if (!env.ATC_LOGS) return;
    const always = status >= 400;
    if (!always && Math.random() > parseFloat(env.SAMPLE_RATE || "0.05")) return;
    env.ATC_LOGS.writeDataPoint({
      blobs: [host, path.slice(0, 96), (referer || "").slice(0, 96)],
      doubles: [status],
      indexes: [String(status)],
    });
  } catch (e) {
    // analytics must never break serving
  }
}

function html(body, status, extra = {}) {
  return new Response(body, {
    status,
    headers: { "content-type": "text/html; charset=utf-8", ...extra },
  });
}

function parseRange(header) {
  const m = /^bytes=(\d*)-(\d*)$/.exec(header || "");
  if (!m || (!m[1] && !m[2])) return null;
  if (!m[1]) return { suffix: parseInt(m[2], 10) };
  const offset = parseInt(m[1], 10);
  if (!m[2]) return { offset };
  const length = parseInt(m[2], 10) - offset + 1;
  // Inverted range (end < start): ignore it and serve the whole object rather
  // than handing R2 a negative length, which throws out of serveKey.
  if (length <= 0) return null;
  return { offset, length };
}

async function serveKey(env, key, request, headers) {
  const range = parseRange(request.headers.get("range"));
  let obj;
  try {
    obj = await env.BUCKET.get(key, range ? { range } : undefined);
  } catch (err) {
    // A range past EOF is the client's problem — a resumed `curl -C -` on an
    // already-complete file sends exactly bytes=<size>-. Answer 416 with the
    // real size instead of letting the throw escape as an opaque 500, which
    // also skips the log() call at the end of fetch().
    if (!/satisfiable|invalid range|10039/i.test(String((err && err.message) || err)))
      throw err;
    const head = await env.BUCKET.head(key).catch(() => null);
    if (!head) return null; // key really doesn't exist: keep probing candidates
    return new Response(null, {
      status: 416,
      headers: { ...headers, "content-range": `bytes */${head.size}`, "accept-ranges": "bytes" },
    });
  }
  if (obj === null) return null;

  const h = new Headers(headers);
  obj.writeHttpMetadata(h);
  h.set("etag", obj.httpEtag);
  h.set("accept-ranges", "bytes");
  const isHtml = /\.(html?|shtml)$/i.test(key);
  // FrontPage-era .htm pages are windows-1252 with meta tags. rclone stored
  // "text/html; charset=utf-8" (Go's mime table), and a header charset beats
  // the meta tag — strip it on .htm/.shtml so the page's own declaration wins.
  if (/\.(htm|shtml)$/i.test(key)) h.set("content-type", "text/html");
  h.set("cache-control", isHtml ? "public, max-age=3600" : "public, max-age=86400");
  if (key === "feed/index.html") h.set("content-type", "application/rss+xml");

  let status = 200;
  if (range && obj.range) {
    status = 206;
    const off = obj.range.offset ?? 0;
    const len = obj.range.length ?? obj.size - off;
    h.set("content-range", `bytes ${off}-${off + len - 1}/${obj.size}`);
    h.set("content-length", String(len));
  }
  if (request.method === "HEAD") {
    // the runtime omits Content-Length for null-body responses; clients
    // (and the parity harness) want the real size on HEAD
    if (status === 200) h.set("content-length", String(obj.size));
    return new Response(null, { status, headers: h });
  }
  return new Response(obj.body, { status, headers: h });
}

// Serve a canonical URI ("/atc/lewiston", "/atc/history/checklst") from R2 by
// probing its key candidates.
async function serveCanonical(env, canon, request, extraHeaders) {
  for (const k of keyCandidates(canon)) {
    const r = await serveKey(env, k, request, extraHeaders);
    if (r) return r;
  }
  return null;
}

async function redirectOldHost(env, url, request) {
  const rawPath = url.pathname; // keep original percent-encoding
  let path;
  try {
    path = decodeURIComponent(rawPath);
  } catch {
    return html("Bad request", 400);
  }

  // Exceptions that must keep answering 200 on the old hostname. Guard the R2
  // misses: a probed wp-sitemap variant that was never frozen must 404, not
  // crash the handler on a null response.
  if (path === "/robots.txt")
    return new Response(OLD_ROBOTS, { headers: { "content-type": "text/plain" } });
  if (path === "/google994516164ab8ab88.html")
    return (
      (await serveKey(env, "google994516164ab8ab88.html", request, {})) ??
      html(PAGE_404, 404)
    );
  if (/^\/wp-sitemap[\w-]*\.xml$/.test(path))
    return (
      (await serveKey(env, "_oldhost" + path, request, {
        "content-type": "application/xml",
      })) ?? html(PAGE_404, 404)
    );

  // Query-string shortlinks — resolved through the FULL route table, not just
  // canonicalOf: dropped slugs (junk pages, /home/) must also cost exactly one
  // hop instead of bouncing through /atc/<slug> a second time.
  const p = url.searchParams.get("p") || url.searchParams.get("page_id");
  if (p && P_MAP[p]) {
    const r = routeOldPath(P_MAP[p]);
    if (r?.status === 410) return html(PAGE_410, 410);
    return Response.redirect(NEW_ORIGIN + (r?.to ?? PREFIX + P_MAP[p]), 301);
  }
  if (url.searchParams.has("s")) return Response.redirect(NEW_ORIGIN + PREFIX + "/", 301);

  const route = routeOldPath(path);
  if (route?.status === 410) return html(PAGE_410, 410);
  if (route?.to) return Response.redirect(NEW_ORIGIN + route.to, 301);

  // Unknown path: prefix it unchanged (original encoding preserved) and let the
  // serving side 404 honestly.
  return Response.redirect(NEW_ORIGIN + PREFIX + rawPath, 301);
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const host = url.hostname;
    const referer = request.headers.get("referer");

    if (request.method !== "GET" && request.method !== "HEAD")
      return new Response("Method not allowed", {
        status: 405,
        headers: { allow: "GET, HEAD" },
      });

    let response;
    if (OLD_HOSTS.has(host) && env.MODE === "redirect") {
      response = await redirectOldHost(env, url, request);
    } else {
      // Serving. archive.aero owns only /atc/*; staging answers both that and
      // the bare site root, so pre-cutover click-through works either way.
      const staging = host !== "archive.aero";
      let rawPath = url.pathname;
      if (rawPath === PREFIX) return Response.redirect(NEW_ORIGIN + PREFIX + "/", 301);
      if (rawPath.startsWith(PREFIX + "/")) rawPath = rawPath.slice(PREFIX.length);
      else if (!staging) return html(PAGE_404, 404);
      let path;
      try {
        path = decodeURIComponent(rawPath);
      } catch {
        return html("Bad request", 400);
      }
      // `staging` above means "serve the bare site root, no /atc/ prefix" — a
      // shape the old hostnames share with atc-staging once the cutover 09:00
      // step routes them here. It must NOT imply noindex: atchistory.org in
      // serve mode IS the live indexed site, and noindexing it during the
      // 09:00→13:00 window would deindex the corpus the 301s are meant to move.
      const noindex =
        !OLD_HOSTS.has(host) && (staging || env.ATC_NOINDEX === "1");
      const extra = noindex ? { "x-robots-tag": "noindex, nofollow" } : {};
      const selfOrigin = staging ? url.origin : NEW_ORIGIN;
      const selfPrefix = staging ? "" : PREFIX;

      // An old spelling reaching the new host 301s to the canonical URI —
      // same map, same single hop, so a stale link costs one redirect whether
      // it arrives via atchistory.org or via archive.aero/atc/.
      const route = routeOldPath(path);
      if (route?.status === 410) response = html(PAGE_410, 410, extra);
      else if (route?.to)
        response = Response.redirect(
          route.to.startsWith(PREFIX)
            ? selfOrigin + selfPrefix + route.to.slice(PREFIX.length)
            : NEW_ORIGIN + route.to,
          301);
      else
        response =
          (await serveCanonical(env, PREFIX + path, request, extra)) ??
          html(PAGE_404, 404, extra);
    }

    ctx.waitUntil(Promise.resolve(log(env, host, url.pathname, response.status, referer)));
    return response;
  },
};
