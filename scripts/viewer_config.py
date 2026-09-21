"""Keep historical --update-html callers working after module extraction."""
from pathlib import Path
import re
import subprocess


def viewer_config_source(path):
    path = Path(path)
    if path.suffix == ".html" and "data-viewer-entry" in path.read_text():
        return path.parent / "src" / "viewer.js"
    return path


def update_viewer_config(path, key, url):
    """Update a config URL and regenerate fingerprinted modules when applicable.

    Accepts both the old index.html entry point and the extracted source path;
    old single-file copies still work. Never edits generated assets in place.
    """
    target = viewer_config_source(path)
    source = target.read_text()
    pattern = rf"(\b{re.escape(key)}:\s*)(null|'[^']*')"
    if len(re.findall(pattern, source)) != 1:
        raise ValueError(f"{target}: expected one {key} config entry")
    quoted = "'" + url.replace("\\", "\\\\").replace("'", "\\'") + "'"
    target.write_text(re.sub(pattern, lambda m: m[1] + quoted, source, count=1))
    root = target.parent.parent
    if target == root / "src" / "viewer.js":
        builder = root / "scripts" / "build_frontend.mjs"
        if not builder.is_file():
            raise FileNotFoundError(f"Missing frontend builder: {builder}")
        subprocess.run(["node", str(builder), "--root", str(root.resolve())], check=True)
    return target
