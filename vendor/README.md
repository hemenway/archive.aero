# Pinned browser dependencies

These are unmodified release files from the exact npm versions in package-lock.json:

| Library | Version | License |
| --- | --- | --- |
| Leaflet | 1.9.4 | BSD-2-Clause |
| PMTiles | 3.0.6 | BSD-3-Clause |
| Protomaps Leaflet | 5.0.0 | BSD-3-Clause |
| PapaParse | 5.4.1 | MIT |

Run `npm ci && npm run vendor:frontend` to reproduce them. `vendor/manifest.json`
records package paths, byte sizes and SHA-384 hashes. `npm run check:frontend`
verifies exact bytes, release versions, the five original script/CSS SRI pins,
and generated application modules without writing files or contacting a CDN.
Leaflet images and the upstream source maps stay at their relative URLs.

Keep versioned paths immutable. For an upgrade, update the exact package version,
lockfile, vendor script's version/SRI pins, HTML or `src/csv.js` URLs and integrity
attributes, and tests together; regenerate vendor files and frontend assets.
Commit the generated files. The deployed site does not need node_modules.

Each release retains its upstream license. PMTiles' npm 3.0.6 package omits the
license file; its BSD notice is copied from the
[upstream repository license](https://github.com/protomaps/PMTiles/blob/main/LICENSE).
The bundled dependencies' notices are in `THIRD_PARTY_LICENSES.txt`.

Modules use content-hashed filenames; vendor files use release-version directories.
The host controls HTTP caching. No Cloudflare settings are changed by this task.
Do not remove previously deployed module hashes while cached HTML can reference them.
