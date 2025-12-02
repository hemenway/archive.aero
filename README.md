✈️ AeroMap / AeroArchive

Historical Aerial Maps — Accessible, Searchable, Viewable Anywhere

AeroMap is an open-source project for hosting, viewing, and distributing historical aerial maps and aeronautical chart scans. It provides a modern web interface that loads Cloud-Optimized GeoTIFFs (COGs), GeoTIFFs, or tiled imagery directly from cloud storage and displays them in an intuitive timeline-style viewer.

This project powers aeromap.io, a simple, fast, and public viewer for aviation map history.

⸻

🚀 Features
	•	Modern Leaflet-based viewer
Smooth panning, zooming, and opacity controls for comparing chart revisions.
	•	Supports Cloud-Optimized GeoTIFF (COG)
Loads maps directly from Cloudflare R2, AWS S3, public datasets, or any CORS-enabled bucket.
	•	Timeline interface
Quickly navigate through map editions (1930s → 2000s) in a clean visual timeline.
	•	COG tiling via TiTiler, RDNT tiles, or native Leaflet GeoTIFF
Pluggable tile service support depending on your hosting environment.
	•	Lightweight static deployment
100% static HTML/JS/CSS. No backend required.
Deploy anywhere: Cloudflare Pages, GitHub Pages, Netlify, S3, etc.
	•	Optional Worker proxy
Cloudflare Worker can mirror the site directly from GitHub for auto-updates.
# AeroMap