#!/Users/ryanhemenway/venv/bin/python
"""Generate static directory-listing pages for the /atc/ static trees.

The live site serves Apache autoindex listings for directories without index
files (raw-log evidence: /History/Pubs/faa_world/ etc. return 200 and get real
visitors). The atc worker serves R2 objects only, so every such directory gets
a pre-generated index.html at build time. Directories that already have an
index file (index.htm/index.html/Default.htm — FrontPage era) are untouched.
"""
import html
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

STATIC = Path("/Volumes/projects/atchistory_build/static")
TREES = ["History", "Images", "Masters", "classphotos", "pdf", "video"]
# Apache DirectoryIndex names only — curated pages like classphotos/PhotoHome.htm
# are NOT index files (the live site shows autoindex for those dirs).
INDEX_NAMES = {"index.htm", "index.html", "default.htm"}
MARKER = "<!-- generated directory listing: archive.aero /atc/ migration -->"

PAGE = """{marker}
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Index of {title}</title>
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; margin: 2rem auto;
       max-width: 52rem; padding: 0 1rem; color: #1a1a1a; background: #fdfdfb; }}
h1 {{ font-size: 1.15rem; font-weight: 600; border-bottom: 1px solid #999;
     padding-bottom: .4rem; }}
h1 a {{ color: #274e75; text-decoration: none; }}
table {{ border-collapse: collapse; width: 100%; font-size: .92rem; }}
td {{ padding: .22rem .6rem .22rem 0; vertical-align: top; }}
td.size {{ text-align: right; white-space: nowrap; color: #555; }}
a {{ color: #274e75; }}
p.note {{ color: #666; font-size: .8rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>Index of {crumbs}</h1>
<table>
{rows}
</table>
<p class="note">Part of the Air Traffic Control History collection, preserved
at <a href="/atc/">archive.aero/atc</a>.</p>
</body>
</html>
"""


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / 1:.1f} {unit}"
        n /= 1024


def crumb_links(rel):
    parts = rel.parts
    out = ['<a href="/atc/">atc</a>']
    for i, p in enumerate(parts):
        href = "/atc/" + quote("/".join(parts[: i + 1])) + "/"
        out.append(f'<a href="{href}">{html.escape(p)}</a>')
    return " / ".join(out)


def main():
    made, skipped = 0, 0
    for tree in TREES:
        base = STATIC / tree
        if not base.is_dir():
            continue
        for d in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
            names = {n.name.lower() for n in d.iterdir()}
            existing = names & INDEX_NAMES
            gen_target = d / "index.html"
            if existing and not (existing == {"index.html"}
                                 and gen_target.exists()
                                 and MARKER in gen_target.read_text(errors="replace")[:100]):
                skipped += 1
                continue
            rel = d.relative_to(STATIC)
            entries = sorted(d.iterdir(),
                             key=lambda p: (p.is_file(), p.name.lower()))
            rows = ['<tr><td><a href="../">[parent directory]</a></td>'
                    "<td></td><td></td></tr>"]
            for e in entries:
                if e.name == "index.html" or e.name.startswith("."):
                    continue
                name = e.name + ("/" if e.is_dir() else "")
                href = quote(e.name) + ("/" if e.is_dir() else "")
                size = "&mdash;" if e.is_dir() else fmt_size(e.stat().st_size)
                mtime = datetime.fromtimestamp(e.stat().st_mtime,
                                               tz=timezone.utc)
                rows.append(f'<tr><td><a href="{href}">{html.escape(name)}</a>'
                            f'</td><td class="size">{size}</td>'
                            f'<td class="size">{mtime:%Y-%m-%d}</td></tr>')
            gen_target.write_text(PAGE.format(
                marker=MARKER, title=html.escape(f"/{rel}/"),
                crumbs=crumb_links(rel), rows="\n".join(rows)))
            made += 1
    print(f"generated {made} listing pages; {skipped} dirs already had an index")


if __name__ == "__main__":
    main()
