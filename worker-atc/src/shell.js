// Site shell for the ATC collection: the archive.aero wordmark, the
// collections strip (which collection you are in), About and Support — the
// same header the chart viewer carries at archive.aero/, so the two
// collections read as one archive. The worker splices it into every HTML page
// it serves (see injectShell); the preserved pages underneath are untouched.
//
// ASCII ONLY, on purpose. The markup is inserted into the raw byte stream of
// pages that declare utf-8, iso-8859-1 and windows-1252, so every byte here
// must mean the same thing in all three: entities (&middot;) instead of
// typographic characters, no em dashes, no curly quotes.

export const SHELL_MARK = "aa-shell";
const OPEN = `<!--${SHELL_MARK}-->`;
const CLOSE = `<!--/${SHELL_MARK}-->`;

const MAIN = "https://archive.aero";
const PLANE =
  "M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z";

// Mirrors index.html/styles.css on the main site: .collections / .collection /
// .about-btn / .donate-btn, pixel-matched, id-prefixed so the WordPress and
// FrontPage themes underneath (global a:hover, .site-logo, .menu-toggle ...)
// cannot restyle it. Fixed 56px tall (52px on phones). The last rule is
// theme-aware: the WordPress theme (escapade, all 1,858 WP pages carry its
// body class) reserves its 250px fixed side masthead as body padding on
// desktop, so the shell pulls back to the viewport edge and the masthead
// starts under it.
export const SHELL_CSS = `#aa-shell{display:flex;align-items:center;justify-content:space-between;gap:16px;height:56px;margin:0;padding:0 24px;box-sizing:border-box;position:relative;z-index:10000;background:#0a0e12;border-bottom:1px solid rgba(255,255,255,.09);color:#fff;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:13px;line-height:1.2;text-align:left}
#aa-shell *{box-sizing:border-box;margin:0;padding:0;border:0}
#aa-shell a{color:inherit;text-decoration:none;background:none;outline-offset:2px}
#aa-shell a:hover{text-decoration:none}
#aa-shell a:focus-visible{outline:2px solid #1e90ff}
#aa-shell .aa-left{display:flex;align-items:center;gap:16px;min-width:0}
#aa-shell .aa-brand{display:inline-flex;align-items:center;gap:8px;font-family:Barlow,system-ui,sans-serif;font-weight:700;font-size:19px;letter-spacing:-.2px;color:#fff;white-space:nowrap}
#aa-shell .aa-brand:hover{color:#4da6ff}
#aa-shell .aa-brand svg{width:18px;height:18px;display:block;fill:#1e90ff}
#aa-shell .aa-coll{display:none;align-items:stretch;gap:18px;margin-left:4px;padding-left:20px;border-left:1px solid rgba(255,255,255,.14)}
#aa-shell .aa-coll-label{display:none;font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.6)}
#aa-shell .aa-c{display:flex;flex-direction:column;gap:1px;padding:3px 0;color:rgba(255,255,255,.6);border-bottom:2px solid transparent}
#aa-shell .aa-c:hover{color:#fff}
#aa-shell .aa-c[aria-current=page]{color:#fff;border-bottom-color:#1e90ff}
#aa-shell .aa-c-name{font-family:Barlow,system-ui,sans-serif;font-size:15px;font-weight:600;line-height:1.15;white-space:nowrap}
#aa-shell .aa-c-desc{display:none;font-size:11px;line-height:1.2;white-space:nowrap;color:rgba(255,255,255,.6)}
#aa-shell .aa-c[aria-current=page] .aa-c-desc{color:rgba(255,255,255,.72)}
#aa-shell .aa-right{display:flex;align-items:center;gap:12px;flex:none}
#aa-shell .aa-btn{display:inline-flex;align-items:center;gap:6px;padding:7px 13px;border:1px solid rgba(255,255,255,.25);border-radius:8px;font-size:13px;font-weight:600;color:#fff;white-space:nowrap}
#aa-shell .aa-btn:hover{background:rgba(255,255,255,.1);border-color:rgba(255,255,255,.45)}
#aa-shell .aa-btn svg{width:14px;height:14px;flex:none;fill:none;stroke:#1e90ff;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
#aa-shell .aa-cta{display:inline-flex;align-items:center;padding:8px 14px;border-radius:8px;background:#ff813f;color:#10161c;font-size:13px;font-weight:700;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.3)}
#aa-shell .aa-cta:hover{background:#ff9257;color:#10161c}
@media (min-width:768px){#aa-shell .aa-coll{display:flex}#aa-shell .aa-pill{display:none}}
@media (min-width:1100px){#aa-shell .aa-coll-label,#aa-shell .aa-c-desc{display:block}}
@media (max-width:640px){#aa-shell{height:52px;padding:0 12px;gap:8px}#aa-shell .aa-brand{font-size:17px}#aa-shell .aa-right{gap:6px}#aa-shell .aa-btn,#aa-shell .aa-cta{font-size:11px;padding:6px 9px}#aa-shell .aa-btn svg{display:none}}
@media (min-width:55.063em){body.wp-child-theme-escapade #aa-shell{margin-left:-250px}body.wp-child-theme-escapade .side-masthead{top:56px !important}}`;

// The <head> part: Barlow for the wordmark (already what the landing, error
// pages and main site load) and the shell's own rules.
export function shellHead() {
  return (
    `${OPEN}<link rel="preconnect" href="https://fonts.googleapis.com">` +
    `<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>` +
    `<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@600;700&display=swap">` +
    `<style>${SHELL_CSS}</style>${CLOSE}`
  );
}

// The <body> part. `atcHome` is where "ATC History" points: "/atc/" on
// archive.aero, "/" on atc-staging (which serves the bare tree).
export function shellBody({ atcHome = "/atc/" } = {}) {
  return (
    `${OPEN}<div id="aa-shell">` +
    `<div class="aa-left">` +
    `<a class="aa-brand" href="${MAIN}/" title="archive.aero home">` +
    `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${PLANE}"/></svg>archive.aero</a>` +
    `<nav class="aa-coll" aria-label="Collections">` +
    `<span class="aa-coll-label" aria-hidden="true">Collections</span>` +
    `<a class="aa-c" href="${MAIN}/">` +
    `<span class="aa-c-name">Sectional Charts</span>` +
    `<span class="aa-c-desc">U.S. aeronautical charts, 1930 to today</span></a>` +
    `<a class="aa-c" href="${atcHome}" aria-current="page">` +
    `<span class="aa-c-name">ATC History</span></a>` +
    `</nav></div>` +
    `<div class="aa-right">` +
    `<a class="aa-btn aa-pill" href="${MAIN}/" title="Sectional Charts: the historical chart viewer">` +
    `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10"/>` +
    `<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>Charts</a>` +
    `<a class="aa-btn" href="${MAIN}/about" title="About this archive &amp; how it works">About</a>` +
    `<a class="aa-cta" href="https://buymeacoffee.com/ryanhemenway" target="_blank" rel="noopener" title="Support this project">Support</a>` +
    `</div></div>${CLOSE}`
  );
}

const BODY_TAG = /<body\b[^>]*>/i;
const HEAD_END = /<\/head\s*>/i;
// WordPress pages open with a screen-reader skip link; it stays first in the
// tab order, the shell goes right after it.
const SKIP_LINK = /^\s*<a class="skip-link[^>]*>[^<]*<\/a>/;

const encoder = new TextEncoder();

function assertAscii(s) {
  for (let i = 0; i < s.length; i++)
    if (s.charCodeAt(i) > 0x7e) throw new Error(`shell markup is not ASCII at ${i}: ${s.slice(i, i + 20)}`);
  return s;
}

// Splice the shell into an HTML page's bytes. Returns null when the page has
// no <body> tag (feeds, verification stubs) so the caller serves it as-is.
// Works on bytes, not text: pages declaring iso-8859-1/windows-1252 are
// preserved byte-for-byte on either side of the ASCII inserts, and no
// transcoding pass ever sees them.
export function injectShell(bytes, opts) {
  // "latin1" is WHATWG's alias for windows-1252: a single-byte decoding, so
  // every char index below is a byte offset (what 0x80-0x9f decode TO is
  // irrelevant — those bytes are copied through untouched).
  const text = new TextDecoder("latin1").decode(bytes);
  const bm = BODY_TAG.exec(text);
  if (!bm) return null;
  let at = bm.index + bm[0].length;
  const skip = SKIP_LINK.exec(text.slice(at, at + 400));
  if (skip) at += skip[0].length;
  const hm = HEAD_END.exec(text.slice(0, bm.index));
  const head = encoder.encode(assertAscii(shellHead()));
  const body = encoder.encode(assertAscii(shellBody(opts)));

  const parts = [];
  if (hm) {
    parts.push(bytes.subarray(0, hm.index), head, bytes.subarray(hm.index, at), body, bytes.subarray(at));
  } else {
    parts.push(bytes.subarray(0, at), head, body, bytes.subarray(at));
  }
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}
