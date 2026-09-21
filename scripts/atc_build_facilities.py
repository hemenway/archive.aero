#!/usr/bin/env python3
"""Build the public facility-location index from the preserved site.

Only location labels, canonical URLs, titles and preview-image URLs are kept;
the archived HTML and photographs remain in R2. Run after rebuilding site/:
  python3 scripts/atc_build_facilities.py [--site /path/to/site]
"""
import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def build(site):
    routes = json.loads((ROOT / "worker-atc/src/route_map.json").read_text())
    posts = json.loads((ROOT / "worker-atc/src/p_map.json").read_text())
    source = (site / "facility-photos/index.html").read_text()
    locations = re.findall(
        r'<h5 data-state="([^"]*)" data-ids="([^"]*)" data-city="([^"]*)"', source)
    if not locations:
        raise ValueError("No facility locations found in the preserved page")
    index = {}
    for state, ids, city in locations:
        entries = []
        for pid in ids.split(","):
            old = posts[pid]
            canonical = routes["alias"][old]
            page = (site / old.strip("/") / "index.html").read_text()
            title = re.search(r'<h1 class="entry-title"[^>]*>(.*?)</h1>', page, re.S)[1]
            title = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
            content = re.search(
                r'<div class="entry-content"[^>]*>(.*?)</div><!-- .entry-content -->',
                page, re.S)[1]
            image = re.search(r'<img\b[^>]*\bsrc="([^"]+)"', content)
            entry = {"href": canonical, "title": title}
            if image:
                url = urlsplit(html.unescape(image[1]))
                if url.netloc != "archive.aero" or not url.path.startswith("/atc/"):
                    raise ValueError(f"Unexpected preview URL: {image[1]}")
                entry["image"] = url.path
            entries.append(entry)
        index.setdefault(html.unescape(state), {})[html.unescape(city)] = entries
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path,
                        default=Path("/Volumes/projects/atchistory_build/site"))
    args = parser.parse_args()
    index = build(args.site)
    output = ROOT / "worker-atc/src/facilities.json"
    output.write_text(json.dumps(index, ensure_ascii=True, indent=2) + "\n")
    print(f"Wrote {len(index)} states/regions, "
          f"{sum(len(cities) for cities in index.values())} locations to {output}")
