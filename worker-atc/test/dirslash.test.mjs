// DirectorySlash through the real fetch handler (worklist 08, 2026-09-20):
// a directory answered by its index file is served at the slashed URI only,
// so a listing's relative links resolve inside it. Fake R2, fake cache.
import assert from "node:assert/strict";
import { test } from "node:test";
import worker from "../src/index.js";

const LISTING = "<!-- generated directory listing --><html><body><a href=\"ak/\">ak/</a></body></html>";
const KEYS = {
  "History/index.html": LISTING,
  "History/FacilityPhotos/index.html": LISTING,
  "Images/index.html": LISTING,
  "classphotos/7711/index.html": LISTING,
  "classphotos/7711/Class7711.jpg": "jpeg-bytes",
  "History/checklst2.htm": "<html><body>checklist</body></html>",
};

function harness(t) {
  const previous = globalThis.caches;
  globalThis.caches = { default: { async match() { return undefined; }, async put() {} } };
  t.after(() => { globalThis.caches = previous; });
  const heads = [];
  const env = { MODE: "redirect", ATC_SHELL: "1", SAMPLE_RATE: "0", BUCKET: {
    async get(key) {
      if (!(key in KEYS)) return null;
      const bytes = new TextEncoder().encode(KEYS[key]);
      return { size: bytes.length, httpEtag: `"${key}"`, body: bytes,
        writeHttpMetadata(h) { h.set("content-type", "text/html; charset=utf-8"); },
        arrayBuffer: async () => bytes.buffer };
    },
    async head(key) { heads.push(key); return key in KEYS ? { size: 1 } : null; },
  } };
  const pending = [];
  const request = async (url, init) => {
    const res = await worker.fetch(new Request(url, init), env, { waitUntil(p) { pending.push(p); } });
    await Promise.all(pending.splice(0));
    return res;
  };
  return { request, heads };
}

test("slashless rule-derived directories 301 to the slashed spelling, same host", async (t) => {
  const { request } = harness(t);
  const cases = {
    "https://archive.aero/atc/history/FacilityPhotos": "https://archive.aero/atc/history/FacilityPhotos/",
    "https://archive.aero/atc/class-photos/7711": "https://archive.aero/atc/class-photos/7711/",
    // a rule that respells the directory itself: only the slashed probe finds it
    "https://archive.aero/atc/images": "https://archive.aero/atc/images/",
    // query rides along, as Apache does it
    "https://archive.aero/atc/history/FacilityPhotos?C=M": "https://archive.aero/atc/history/FacilityPhotos/?C=M",
    // staging serves the bare tree and redirects within it
    "https://atc-staging.archive.aero/history/FacilityPhotos": "https://atc-staging.archive.aero/history/FacilityPhotos/",
  };
  for (const [url, want] of Object.entries(cases)) {
    const res = await request(url);
    assert.equal(res.status, 301, url);
    assert.equal(res.headers.get("location"), want, url);
    const head = await request(url, { method: "HEAD" });
    assert.equal(head.status, 301, "HEAD " + url);
  }
});

test("slashed directories, files and map canonicals still serve", async (t) => {
  const { request } = harness(t);
  for (const url of [
    "https://archive.aero/atc/history/FacilityPhotos/",
    "https://archive.aero/atc/class-photos/7711/",
    "https://archive.aero/atc/class-photos/7711/Class7711.jpg",
    "https://archive.aero/atc/history/checklst2",   // .htm dropped in canonical space
    "https://archive.aero/atc/History",             // slashless BY THE MAP: permanent
  ]) {
    const res = await request(url);
    assert.equal(res.status, 200, url);
  }
});

test("unknown paths stay 404 and cost no extra probes unless a rule respells them", async (t) => {
  const { request, heads } = harness(t);
  assert.equal((await request("https://archive.aero/atc/wp")).status, 404);
  assert.equal((await request("https://archive.aero/atc/history/nowhere")).status, 404);
  assert.equal((await request("https://archive.aero/atc/.env")).status, 404);
  assert.deepEqual(heads, []);
  assert.equal((await request("https://archive.aero/atc/masters")).status, 404);
  assert.deepEqual(heads, ["Masters/index.html", "Masters/index.htm", "Masters/Default.htm"]);
});
