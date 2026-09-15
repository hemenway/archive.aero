#!/usr/bin/env python3
"""Static dev server with HTTP Range support, for checking PMTiles-backed viewer
features against local files before they are published.

``python -m http.server`` ignores the Range header and answers a full 200,
which pmtiles.js rejects ("check that your storage backend supports HTTP Byte
Serving"), so an unpublished archive cannot be viewed through it. This server
answers single-range requests with 206 + Content-Range, adds CORS headers, and
sends Cache-Control: no-store so the preview pane never serves a stale
styles.css.

``--mount /url-prefix=/dir`` serves a directory outside the repo at a URL
prefix, e.g. the local PMTiles mirrors that live on /Volumes/projects:

    ~/venv/bin/python scripts/dev_server.py --port 8898 \
        --mount /airspace=/Volumes/projects/airspace_pmtiles

Everything else is served from the current directory (run it at the repo root).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class RangeHandler(SimpleHTTPRequestHandler):
    mounts: list[tuple[str, Path]] = []
    extensions_map = dict(SimpleHTTPRequestHandler.extensions_map,
                          **{".pmtiles": "application/octet-stream", ".geojson": "application/geo+json",
                             ".bundle": "application/octet-stream", ".mjs": "text/javascript"})

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, root in self.mounts:
            if clean == prefix or clean.startswith(prefix + "/"):
                rel = clean[len(prefix):].lstrip("/")
                return str(root / rel) if rel else str(root)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range, If-Match, If-None-Match")
        self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range, ETag, Accept-Ranges")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        m = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", rng or "")
        if not m or not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        a, b = m.group(1), m.group(2)
        if a == "" and b == "":
            return super().do_GET()
        if a == "":                      # suffix range: last N bytes
            start, end = max(0, size - int(b)), size - 1
        else:
            start = int(a)
            end = min(int(b), size - 1) if b else size - 1
        if start >= size or start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        st = os.stat(path)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("ETag", f'"{st.st_size:x}-{int(st.st_mtime):x}"')
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, fmt: str, *args) -> None:  # quieter: one line per request
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8898)))
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--mount", action="append", default=[], metavar="/prefix=/dir",
                    help="serve /dir at /prefix (repeatable)")
    args = ap.parse_args()
    mounts = []
    for spec in args.mount:
        prefix, _, d = spec.partition("=")
        root = Path(d).expanduser()
        if not root.is_dir():
            sys.exit(f"--mount {spec}: {root} is not a directory (volume mounted?)")
        mounts.append(("/" + prefix.strip("/"), root))
    RangeHandler.mounts = mounts
    httpd = ThreadingHTTPServer((args.bind, args.port), RangeHandler)
    print(f"serving {os.getcwd()} on http://{args.bind}:{args.port}/"
          + "".join(f"\n  {p}/ -> {r}" for p, r in mounts), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
