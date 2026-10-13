"""Stage only explicitly published files, never the repository root, for Pages.

Besides site-files.json, the Leaflet viewer is staged a second time under
/classic/ (CLASSIC below): the next viewer took over / on 2026-10-12 and the
old one stays published at a stable path for its share links, as the fallback
of browsers without WebGL2, and as the rollback (next/HOSTING.md, rollout
step 4). The copy is marked noindex; its asset URLs are already absolute.
"""
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess


ASSET = re.compile(r"assets/(?:boot|viewer|csv)\.[a-f0-9]{16}\.js")
# Published path -> source file. index.html references these three by relative URL
# (styles.css, dates.csv as the CSV fallback, coverage.json), so they travel with it.
CLASSIC = {
    "classic/index.html": "index.html",
    "classic/styles.css": "styles.css",
    "classic/dates.csv": "dates.csv",
    "classic/coverage.json": "coverage.json",
}
NOINDEX = '<meta name="robots" content="noindex">'
CHARSET = re.compile(r'<meta charset="utf-8"\s*/?>', re.IGNORECASE)


def classic_page(html):
    """The viewer page for /classic/: one noindex meta after the charset, nothing else changed."""
    if NOINDEX in html:
        return html
    found = list(CHARSET.finditer(html))
    if len(found) != 1:
        raise ValueError("index.html must declare its charset exactly once")
    at = found[0].end()
    return html[:at] + NOINDEX + html[at:]


def build_site(root):
    root = Path(root).resolve()
    files = json.loads((root / "site-files.json").read_text())
    if not isinstance(files, list) or not all(isinstance(p, str) for p in files):
        raise ValueError("site-files.json must be an explicit list of file paths")
    # Retain every committed fingerprint: previously deployed HTML may use it.
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "assets/"], cwd=root
    ).decode().split("\0")
    files = sorted(set(files + [p for p in tracked if ASSET.fullmatch(p)]))
    for relative in files:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or str(path) != relative:
            raise ValueError(f"Invalid publication path: {relative}")
        if path.parts[0] == "_site":
            raise ValueError("Cannot publish build output as input")
        # Reject symlinks at every level, including a symlinked parent directory.
        if any((root / Path(*path.parts[:i])).is_symlink()
               for i in range(1, len(path.parts) + 1)):
            raise ValueError(f"Publication input is a symlink: {relative}")
        if not (root / relative).is_file():
            raise ValueError(f"Publication input is missing or not a file: {relative}")
    # The classic copy mirrors published files only: a source outside site-files.json is never staged.
    classic = {dst: src for dst, src in CLASSIC.items() if src in files} if "index.html" in files else {}
    if "index.html" in files and set(classic) != set(CLASSIC):
        raise ValueError("The classic viewer copy needs every CLASSIC source in site-files.json")
    output = root / "_site"
    if output.is_symlink():
        raise ValueError("_site must not be a symlink")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    for relative in files:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    for dst, src in classic.items():
        target = output / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        if src == "index.html":
            target.write_text(classic_page((root / src).read_text()))
        else:
            shutil.copyfile(root / src, target)
    (output / ".nojekyll").touch()
    return output, sorted(files + list(classic))


if __name__ == "__main__":
    output, files = build_site(Path(__file__).resolve().parents[1])
    print(f"Staged {len(files)} published files plus .nojekyll in {output.name}/")
