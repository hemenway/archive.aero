// Routing assertions for the canonical/alias URI scheme (worklist 08).
// Pure — no R2, no network. Run: cd worker-atc && npm test
//
// The properties being defended, in W3C "Cool URIs" terms:
//   1. every canonical URI serves itself (never redirects to itself)
//   2. every old path reaches its canonical target in exactly ONE hop
//   3. no canonical URI contains wp-content, /category/, /author/, .htm,
//      a collision suffix, a space, or a /page/N segment
//   4. the alias space is injective enough that nothing chains
import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { PREFIX, canonicalOf, keyCandidates, routeOldPath } from "../src/routes.js";

const MAP = JSON.parse(readFileSync(new URL("../src/route_map.json", import.meta.url)));

test("old paths reach canonical in one hop", () => {
  const cases = {
    "/facility-photos/": "/atc/facility-photos",
    "/category/early-radio-years/light-houses/": "/atc/topics/light-houses",
    "/historical-maps/u-s-historical-airway-maps/historical-maps-2/":
      "/atc/us-historical-airway-maps",
    "/wp-content/uploads/2016/10/1954AirlineRoutes.jpg":
      "/atc/media/2016/10/1954AirlineRoutes.jpg",
    "/History/Western Airway Beacons List.xls":
      "/atc/history/western-airway-beacons-list.xls",
    "/classphotos/PhotoHome.htm": "/atc/class-photos/photo-home",
    "/History/checklst.htm": "/atc/how-the-pilots-checklist-came-about",
    "/index.htm": "/atc/",
    // author archives are the same posts at a forbidden address — the reader
    // keeps their page number, just not the author's name
    "/author/admin/page/12/": "/atc/archive/12",
    "/author/admin/": "/atc/",
    "/page/2/": "/atc/archive/2",
    "/category/early-radio-years/page/3/": "/atc/topics/early-radio-years/3",
    "/advertising-area-one/": "/atc/",
  };
  for (const [old, want] of Object.entries(cases))
    assert.equal(routeOldPath(old)?.to, want, old);
});

test("tombstones stay 410", () => {
  for (const p of ["/forum/", "/forum/index.php", "/wp-admin/", "/xmlrpc.php",
                   "/lewiston/feed/"])
    assert.equal(routeOldPath(p)?.status, 410, p);
});

test("site-feed variants 301 to the frozen feed, one hop", () => {
  for (const p of ["/feed/atom/", "/feed/atom", "/feed/rdf/", "/feed/rss/",
                   "/feed/rss2/"])
    assert.equal(routeOldPath(p)?.to, `${PREFIX}/feed`, p);
  // per-post comment feeds stay 410
  assert.equal(routeOldPath("/lewiston/feed/")?.status, 410);
});

test("canonical URIs serve themselves — no self-redirect", () => {
  for (const canon of Object.values(MAP.alias)) {
    const inner = canon.slice(PREFIX.length) || "/";
    const route = routeOldPath(inner);
    assert.ok(route === null || route.to === canon,
      `${canon} would redirect to ${route?.to ?? route?.status}`);
  }
});

test("no alias target is itself an alias (no chains)", () => {
  for (const [old, canon] of Object.entries(MAP.alias)) {
    const inner = canon.slice(PREFIX.length) || "/";
    const onward = MAP.alias[inner];
    assert.ok(onward === undefined || onward === canon,
      `${old} -> ${canon} -> ${onward}`);
  }
  for (const [old, target] of Object.entries(MAP.drop)) {
    if (target === "410" || !target.startsWith(PREFIX)) continue;
    const inner = target.slice(PREFIX.length) || "/";
    assert.ok(MAP.alias[inner] === undefined || MAP.alias[inner] === target,
      `drop ${old} -> ${target} -> ${MAP.alias[inner]}`);
  }
});

test("canonical space is free of the shapes the W3C doc warns about", () => {
  const banned = [
    [/\/wp-(content|includes|admin)\//, "software mechanism"],
    [/\/category\//, "subject classification route"],
    [/\/author\//, "author name"],
    [/\/page\/\d+/, "pagination machinery"],
    [/\.s?html?$/i, "server file extension"],
    [/ /, "literal space"],
    [/-[2-9]$/, "collision suffix"],
  ];
  // The only deliberate survivors are collision suffixes the generator could
  // not strip because a sibling still holds the bare slug — it publishes that
  // list rather than letting the exception hide.
  const kept = new Set(MAP.kept_suffix);
  for (const canon of new Set(Object.values(MAP.alias))) {
    for (const [rx, why] of banned) {
      if (rx.test(canon) && !kept.has(canon)) assert.fail(`${canon}: ${why}`);
    }
  }
  assert.ok(kept.size < 150, `too many unstrippable suffixes: ${kept.size}`);
});

test("key candidates resolve documents and files sensibly", () => {
  assert.deepEqual(keyCandidates("/atc/media/2016/10/x.jpg"),
    ["wp-content/uploads/2016/10/x.jpg"]);
  assert.deepEqual(keyCandidates("/atc/history/Pubs/early_communications.pdf"),
    ["History/Pubs/early_communications.pdf"]);
  assert.equal(keyCandidates("/atc/facility-photos")[0],
    "facility-photos/index.html");
  assert.ok(keyCandidates("/atc/").every((k) => /^index\.|^Default\./.test(k)));
});

test("canonicalOf leaves already-canonical paths alone", () => {
  for (const p of ["/media/2016/10/x.jpg", "/topics/light-houses",
                   "/history/western-airway-beacons-list.xls"])
    assert.equal(canonicalOf(p), null, p);
});

test("every ?p= shortlink slug resolves in one hop", () => {
  // index.js routes P_MAP[p] through routeOldPath; whatever URL that emits
  // must be stable — routing its inner path again must go nowhere new.
  // Guards the two 2026-08-20 finds: dropped junk slugs bouncing twice, and
  // /home/ (WP front-page slug, live 301s it to /) minting a dead /atc/home/.
  const P_MAP = JSON.parse(readFileSync(new URL("../src/p_map.json", import.meta.url)));
  for (const slug of Object.values(P_MAP)) {
    const r = routeOldPath(slug);
    if (r?.status === 410) continue;
    const target = r?.to ?? PREFIX + slug;
    const inner = target.slice(PREFIX.length) || "/";
    const onward = routeOldPath(inner);
    assert.ok(onward === null || onward.to === target,
      `?p= slug ${slug} -> ${target} -> ${onward?.to ?? onward?.status}`);
  }
  assert.equal(routeOldPath("/home/")?.to, "/atc/");
});
