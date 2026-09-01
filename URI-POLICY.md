# archive.aero URI policy

Adopted 2026-08-21. Framework: W3C **"Cool URIs don't change"**
(<https://www.w3.org/Provider/Style/URI>) — the doctrine already applied to the
ATC collection's canonical space (worklist 08 §A3) and to the per-chart PMTiles
keys, promoted here to site law.

**A URI, once published, works forever.** Improvements overwrite content at the
same address; they never move it. Every workstream that mints URLs — generated
pages, new collections, redirects — follows this file.

## The covenant

1. **One canonical form per resource,** `https` + apex host only
   (`https://archive.aero/…`); `www` and `http` are permanent 301s to it.
2. **Documents are extensionless with no trailing slash** (`/about`). `.html`
   names the server's technology, not the resource; the `.html` spellings are
   permanent 301 aliases, never the canonical. Format extensions on real files
   (`.pdf`, `.jpg`, `.xls`) stay — they name the artifact's format.
3. **Collections are one short top-level prefix with a trailing slash**
   (`/atc/`, `/sectionals/`). Everything a collection publishes lives under its
   prefix; the prefix never encodes software, vendor, or hosting choices.
4. **Cross-collection entities** — things many collections describe, like
   airports — get their own top-level space (`/airports/…`) rather than living
   inside any one collection's prefix.
5. **Lowercase canonical in all newly minted spaces**; mixed-case forms are 301
   aliases. (Scar tissue, not theory: case-insensitive APFS silently destroyed
   four case-colliding files in 2026-08 — see worklist 08 §A4. Never mint a
   space where two spellings can collide.)
6. **Aliases are permanent.** A published 301 is never removed, and never
   retargeted except to shorten a chain. Alias maps live in
   `worker-atc/src/route_map.json` (the `/atc/` space) and in Cloudflare
   redirect rules on the zone (core pages, `www`).
7. **Keys are never renamed or deleted.** Republishing an improved artifact
   overwrites the same key — the per-chart PMTiles rule
   (`sectionals/chart/<slug>/<date>`, extension-less), applied site-wide.
8. **The viewer's query parameters are API.** `?date=&lat=&lng=&zoom=` on `/`
   are published permalinks (the share panel mints them); the names and
   semantics are permanent and never repurposed.

## The address plan

| Space | Meaning | Status |
|---|---|---|
| `/` | The sectional chart viewer — permanent. Never relocated, never demoted to a portal; the front-door job belongs to the nav and `/about`. | live |
| `/about` `/contribute` `/sources` | Site-level documents | live — extensionless canonical since 2026-08-21 |
| `/atc/…` | ATC History collection (designed canonical space, worklist 08 §A3) | live |
| `/sectionals/…` | Sectional-chart pages. Edition pages at `/sectionals/chart/<slug>/<date>`, mirroring the R2 artifact key on `data.archive.aero` — the human page and the machine artifact share one identifier on two hosts. | planned (worklist 09 B2) |
| `/airports/<icao>` | Cross-collection airport pages, lowercase (`/airports/kbos`; uppercase forms 301) | planned (worklist 09 B1) |
| future collections | one new prefix each (e.g. `/tac/`), added to this table when minted | — |

## Checklist for a new collection

Ship all of these on day one, none later:

1. The prefix, recorded in the table above with its mint date.
2. `sitemap-<name>.xml` slotted into the `sitemap.xml` index
   (`scripts/atc_gen_sitemaps.py` is the pattern).
3. A nav entry: viewer menu-panel "Collections" group + the shared header on
   the static pages.
4. Permanent 301 aliases from any prior home of the content (the whole 08
   migration is the worked example).
5. `rel=canonical` + `og:url` + `og:image` on every page.

## Redirect inventory — core site (Cloudflare Bulk Redirect List)

The `/atc/*` space handles its own aliases inside the worker; the core site's
aliases live in a Cloudflare **Bulk Redirect List** (live + verified
2026-08-21), uploaded from
`scripts/cf_bulk_redirects_core.csv` (the versioned source of truth — CSV
columns: `source,target,status,preserve_query,include_subdomains,
subpath_matching,preserve_path_suffix`; no header row). Enumerated exact URLs,
deliberately no wildcards, so nothing can ever intercept `/atc/*` paths ahead
of the worker:

- `www.archive.aero/` → `https://archive.aero/`, 301, subpath matching +
  preserve path suffix + preserve query (i.e. `www.archive.aero/anything` →
  `archive.aero/anything`; scheme-less source matches http and https).
  Requires the `www` DNS record (added 2026-08-21) — a hostname that doesn't
  resolve is worse than any redirect.
- `/index.html` → `/`, and `/{about,contribute,sources}.html` → the
  extensionless canonical, all 301 with query preserved.

**When a new core doc page is minted, add its `.html` alias row to the CSV
and re-upload the list** — the alias is part of shipping the page. The list
does nothing until an account-level **Bulk Redirect Rule** references it;
that rule stays enabled forever.

### Open gap — plain `http://` on the apex is not redirected (found 2026-08-23)

Covenant 1 says `http` is a permanent 301 to the `https` canonical. It isn't:
every canonical URL answers **200 over plain http** — verified on `/`, `/about`,
`/contribute`, `/sources`, and `/atc/`. Only the `.html` aliases redirect, and
only because those bulk-redirect rows are scheme-less. There is also no HSTS
header on the `https` responses.

So `http://archive.aero/about` and `https://archive.aero/about` are both live,
serving the same page — a duplicate-content split, and cleartext delivery of the
whole site. It should be closed **before the cutover-day sitemap submission**,
which is when Google starts hard-canonicalizing these URLs.

Fix is a dashboard change (not in this repo). Preferred: a **Single Redirect
rule** on the zone, `http.request.scheme eq "http" and http.host eq
"archive.aero"` -> 301 to the same path on `https`. That closes the apex without
touching `www`, whose `http` form already reaches the canonical in one hop via
the Bulk Redirect list (verified 08 §F).

Do **not** simply switch on **Always Use HTTPS**: it runs ahead of Bulk
Redirects, so `http://www.archive.aero/x` would become a two-hop chain
(`-> https://www/x -> https://apex/x`) — the exact chain 08 §F avoided. Add HSTS
(`max-age=31536000`, no preload at first) once the redirect is in place.
