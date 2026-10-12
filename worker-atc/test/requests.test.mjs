// Request-level edge cases through the real fetch handler: hostile shortlink
// ids, the query on the bare /atc prefix, and plain http on the apex.
import assert from "node:assert/strict";
import { test } from "node:test";
import worker from "../src/index.js";
import P_MAP from "../src/p_map.json" with { type: "json" };

function harness(t) {
  const previous = globalThis.caches;
  globalThis.caches = { default: { async match() { return undefined; }, async put() {} } };
  t.after(() => { globalThis.caches = previous; });
  const env = { MODE: "redirect", SAMPLE_RATE: "0", BUCKET: {
    async get() { return null; },
    async head() { return null; },
  } };
  const pending = [];
  return async (url, init) => {
    const res = await worker.fetch(new Request(url, init), env, { waitUntil(p) { pending.push(p); } });
    await Promise.all(pending.splice(0));
    return res;
  };
}

test("shortlink ids that name Object built-ins are not pages, and never 500", async (t) => {
  const request = harness(t);
  for (const id of ["constructor", "__proto__", "toString", "hasOwnProperty", "valueOf"]) {
    for (const url of [`https://archive.aero/atc/?p=${id}`, `https://archive.aero/atc/?page_id=${id}`,
                       `https://atchistory.org/?p=${id}`, `https://www.atchistory.org/?page_id=${id}`]) {
      const res = await request(url);
      assert.ok(res.status < 500, `${url} -> ${res.status}`);
      // Not resolved as a shortlink: no redirect to a page derived from a built-in.
      assert.doesNotMatch(res.headers.get("location") || "", /function|object/i, url);
    }
  }
});

test("the bare /atc prefix keeps its query on the way to /atc/", async (t) => {
  const request = harness(t);
  const id = Object.keys(P_MAP)[0];
  for (const query of [`?p=${id}`, "?state=AK&city=Juneau"]) {
    const res = await request("https://archive.aero/atc" + query);
    assert.equal(res.status, 301);
    assert.equal(res.headers.get("location"), "https://archive.aero/atc/" + query);
  }
  // ...and the shortlink then resolves to its page, not the landing page.
  const res = await request(`https://archive.aero/atc/?p=${id}`);
  assert.equal(res.status, 301);
  assert.notEqual(res.headers.get("location"), "https://archive.aero/atc/");
});

test("plain http on the apex is one 301 to the https spelling", async (t) => {
  const request = harness(t);
  const cases = {
    "http://archive.aero/atc/": "https://archive.aero/atc/",
    "http://archive.aero/atc/history/FacilityPhotos/?C=M": "https://archive.aero/atc/history/FacilityPhotos/?C=M",
    "http://archive.aero/atc": "https://archive.aero/atc/",
    "http://archive.aero/atc?state=AK": "https://archive.aero/atc/?state=AK",
  };
  for (const [url, want] of Object.entries(cases)) {
    for (const method of ["GET", "HEAD"]) {
      const res = await request(url, { method });
      assert.equal(res.status, 301, `${method} ${url}`);
      assert.equal(res.headers.get("location"), want, `${method} ${url}`);
    }
  }
  // https is not redirected by this rule (a 404 from the empty fake bucket).
  assert.equal((await request("https://archive.aero/atc/no-such-page")).status, 404);
});
