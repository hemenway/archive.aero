# Canonical fixtures

From the repository root:

```sh
node next/contract/fixtures/make_fixtures.mjs
node next/contract/fixtures/serve.mjs
```

The generator prints the manifest path; `out/fixture-index.json` also identifies
it and example tile coordinates. The default manifest URL is
`http://127.0.0.1:8765/next/manifest.dfb66f478bc3.json`. All files in `out/` are
ignored and regenerable. Override destination and base URL with positional args:

```sh
node next/contract/fixtures/make_fixtures.mjs /tmp/archive-fixtures http://127.0.0.1:9000/
node next/contract/fixtures/serve.mjs /tmp/archive-fixtures 9000
```

Run `node next/contract/validate.mjs --manifest FILE --base URL` for HTTP smoke,
or add `--schema-only` for offline CI. `npm run test:workers` includes a full
fixture generation, hash check, schema check and smoke against in-memory doubles.
The HTTP server imports the actual Worker handler. No production endpoints are
used; the examples use synthetic data and example.com links.

The dataset has four overlapping eras, one with null antimeridian bounds,
PNG tiles at z4–11, compressed MVT class/efloor geometries for all eight style
codes and three regions (airspace metadata in the production
`archive_aero.regions` shape, with a cycle in effect in January 1951), 21 airfields with every status/year combination,
two z5 pin shards (391 and 392), a per-chart solo archive, and a basemap at z0–13.
C4 details are index-aligned, and all archive hashes are computed from their bytes.
This provides shared fixtures for the renderer, data plane, shell and data section.

`benchmark.mjs` compares cold viewport requests with the old range endpoint.
`make_basemap_proof.mjs DIR` creates a local vector source with land, water and
an English place label for exercising the Chromium renderer. Use mirrored
Protomaps fonts and sprites as described in `../MIGRATION.md`; it does not need
the production vector cutout.
