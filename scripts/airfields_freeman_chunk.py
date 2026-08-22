#!/usr/bin/env python3
"""Regenerate airfields_chunks.jsonl from a www.airfields-freeman.com mirror.

Reverse-engineered (2026-08-21) to byte-parity against the original
airfields_chunks.jsonl of the 2026-02-17 grab, whose generating script is lost:

  text  = "# {<title> stripped}\n\n" + "\n".join(soup.stripped_strings)
          (per-string strip; newlines internal to a source string survive)
  chunks: window 3000 chars from the raw start (no whitespace skip), cut at
          the last plain space ' ' inside the window (newlines are NOT cut
          points, space excluded), emit window.strip();
          next raw start = cut_position - 300
  images: every <img src> of the page resolved to an absolute local path,
          de-duplicated and sorted, identical list on every chunk
  id    = "<relpath>:::<chunk_index>"; url = https://www.airfields-freeman.com/<relpath>

Usage:
  airfields_freeman_chunk.py --tree .../snapshot-2026-08-21/www.airfields-freeman.com \
      --out .../airfields_chunks_2026-08-21.jsonl
"""
import argparse
import json
import os
import posixpath
import sys
import urllib.parse

from bs4 import BeautifulSoup

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300
BASE_URL = "https://www.airfields-freeman.com/"


def page_text(soup):
    title = soup.title.get_text(strip=True) if soup.title else ""
    body = "\n".join(soup.stripped_strings)
    return f"# {title}\n\n" + body


def window_chunks(text):
    n = len(text)
    out = []
    start = 0
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            sp = text.rfind(" ", start, end)
            if sp > start:
                end = sp
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        start = end - CHUNK_OVERLAP
    return out


def page_images(soup, page_abspath, tree_root):
    page_dir = os.path.dirname(page_abspath)
    seen = set()
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src or src.startswith(("http://", "https://", "data:")):
            continue
        local = os.path.normpath(os.path.join(page_dir, urllib.parse.unquote(src)))
        if not local.startswith(os.path.normpath(tree_root) + os.sep):
            continue
        seen.add(local)
    return sorted(seen)


def iter_pages(tree_root):
    for dirpath, dirnames, filenames in os.walk(tree_root):
        dirnames.sort()
        for f in sorted(filenames):
            if f.lower().endswith((".htm", ".html")):
                yield os.path.join(dirpath, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True, help="mirror root (the www.airfields-freeman.com dir)")
    ap.add_argument("--out", required=True, help="output jsonl path")
    args = ap.parse_args()

    tree = os.path.abspath(args.tree)
    n_pages = n_chunks = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in iter_pages(tree):
            rel = os.path.relpath(path, tree)
            rel_url = posixpath.join(*rel.split(os.sep))
            html = open(path, encoding="utf-8", errors="replace").read()
            soup = BeautifulSoup(html, "lxml")
            text = page_text(soup)
            images = page_images(soup, path, tree)
            for i, chunk in enumerate(window_chunks(text)):
                row = {
                    "id": f"{rel_url}:::{i}",
                    "url": BASE_URL + rel_url,
                    "file": path,
                    "chunk_index": i,
                    "text": chunk,
                    "images": images,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_chunks += 1
            n_pages += 1
    print(f"{n_pages} pages -> {n_chunks} chunks -> {args.out}")


if __name__ == "__main__":
    main()
