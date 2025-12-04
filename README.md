# AeroMap

Historical Aerial Maps — Accessible, Searchable, Viewable Anywhere

AeroMap is an open-source project for hosting, viewing, and distributing historical aerial maps and aeronautical chart scans. It provides a modern web interface that loads Cloud-Optimized GeoTIFFs (COGs) directly from object storage and map tile services.

This project powers [aeromap.io](https://aeromap.io), a simple, fast, and public viewer for aviation map history.

⸻

## About

AeroMap is designed to feel like a “ForeFlight Time Machine” — a way to explore how aeronautical charts and airspace depictions have evolved over time.

The project focuses on:

- **Webpage design**
  - Simple, fast, and readable layout
  - Easy navigation between charts and years
  - Works well on desktops, tablets, and minimal hardware
- **ForeFlight Time Machine–style experience**
  - Quickly compare historical charts to modern tools
  - See how procedures, airspace, and airports have changed
- **History of sectionals**
  - Browse chart revisions across decades
  - Understand how aviation cartography and navigation evolved
- **Run on minimal equipment**
  - Static-only deployment, no backend required
  - Optimized for low-resource servers and inexpensive hosting

### Goals

- Add coverage for **more cities** and regions
- Include **Terminal Area Charts (TACs)** alongside sectionals
- Highlight **interesting airports** and related resources (e.g. interestingairports.com–style curation)
- Keep the site **free to use** as long as server and storage costs remain sustainable

AeroMap is and will remain free to access as long as server and storage costs stay manageable. In the future, it may be necessary to introduce **aviation‑related ads** or sponsorships to help offset hosting costs.

If you’d like to support the project and help keep it ad‑light (or ad‑free), please consider **donating to the project**. (Donation details TBD / coming soon.)

⸻

## Features

- **Modern Leaflet-based viewer**  
  Smooth panning, zooming, and opacity controls for comparing chart revisions.

- **Supports Cloud-Optimized GeoTIFF (COG)**  
  Loads maps directly from Cloudflare R2, AWS S3, public datasets, or any CORS-enabled bucket.

- **Timeline interface**  
  Quickly navigate through map editions (1930s → 2000s) in a clean visual timeline.

- **COG tiling via TiTiler, RDNT tiles, or native Leaflet GeoTIFF**  
  Pluggable tile service support depending on your hosting environment.

- **Lightweight static deployment**  
  100% static HTML/JS/CSS. No backend required.  
  Deploy anywhere: Cloudflare Pages, GitHub Pages, Netlify, S3, etc.

- **Optional Worker proxy**  
  A Cloudflare Worker can mirror the site directly from GitHub for auto-updates.
