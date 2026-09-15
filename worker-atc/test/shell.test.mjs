// The site shell splice (src/shell.js). Pure — no R2, no network.
// Run: cd worker-atc && npm test
import assert from "node:assert/strict";
import { test } from "node:test";
import { injectShell, shellBody, shellHead } from "../src/shell.js";

const enc = new TextEncoder();
const latin1 = (b) => new TextDecoder("latin1").decode(b);

test("shell markup is pure ASCII (spliced into cp1252 and utf-8 pages alike)", () => {
  for (const s of [shellHead(), shellBody(), shellBody({ atcHome: "/" })])
    for (let i = 0; i < s.length; i++) assert.ok(s.charCodeAt(i) < 0x7f, `non-ASCII at ${i}`);
});

test("head part lands before </head>, body part right after <body>", () => {
  const page = "<html><head><title>x</title></head><body class=\"a\"><p>hi</p></body></html>";
  const out = latin1(injectShell(enc.encode(page), { atcHome: "/atc/" }));
  assert.match(out, /<\/title><!--aa-shell-->.*<\/style><!--\/aa-shell--><\/head>/s);
  assert.match(out, /<body class="a"><!--aa-shell--><div id="aa-shell">.*<\/div><!--\/aa-shell--><p>hi<\/p>/s);
  assert.ok(out.includes('href="/atc/" aria-current="page"'));
  assert.ok(out.includes('href="https://archive.aero/"'));
});

test("WordPress skip link keeps first place in the tab order", () => {
  const page = "<html><head></head><body>\n\t\n\t<a class=\"skip-link screen-reader-text\" href=\"#content\">Skip to content</a>\n\n<header>";
  const out = latin1(injectShell(enc.encode(page), {}));
  assert.match(out, /Skip to content<\/a><!--aa-shell-->/);
});

test("windows-1252 bytes either side of the splice are untouched", () => {
  // 0x93/0x94 = curly quotes in cp1252; invalid as utf-8, so a text round trip
  // would have replaced them
  const open = enc.encode("<html><head></head><body>");
  const before = Uint8Array.from([...open, 0x93, 0x94, ...enc.encode("</body>")]);
  const out = injectShell(before, {});
  assert.deepEqual([...out.subarray(-9)], [0x93, 0x94, ...enc.encode("</body>")]);
  assert.deepEqual([...out.subarray(0, 12)], [...enc.encode("<html><head>")]);
  assert.equal(out.length, before.length + enc.encode(shellHead()).length + enc.encode(shellBody({})).length);
});

test("pages without <body> (feeds, stubs) are left alone", () => {
  assert.equal(injectShell(enc.encode('<?xml version="1.0"?><rss><channel/></rss>'), {}), null);
  assert.equal(injectShell(enc.encode("google-site-verification: abc"), {}), null);
});

test("no </head>: both parts go in at the body tag", () => {
  const out = latin1(injectShell(enc.encode("<body bgcolor=\"#fff\"><table>"), {}));
  assert.match(out, /^<body bgcolor="#fff"><!--aa-shell--><link .*<\/style><!--\/aa-shell--><!--aa-shell--><div id="aa-shell">/s);
});
