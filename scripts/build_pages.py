"""Stage only explicitly published files, never the repository root, for Pages."""
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess


ASSET = re.compile(r"assets/(?:boot|viewer|csv)\.[a-f0-9]{16}\.js")


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
    (output / ".nojekyll").touch()
    return output, files


if __name__ == "__main__":
    output, files = build_site(Path(__file__).resolve().parents[1])
    print(f"Staged {len(files)} published files plus .nojekyll in {output.name}/")
