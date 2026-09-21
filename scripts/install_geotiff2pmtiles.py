#!/usr/bin/env python3
"""Install upstream main with native WebP; print the exact binary path to stdout."""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


MODULE = "github.com/pspoerri/geotiff2pmtiles"
REPOSITORY = f"https://{MODULE}.git"
BRANCH = "main"
TOOLS_DIR = (
    Path.home() / "Library" / "Caches" if sys.platform == "darwin"
    else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
) / "archive.aero"
DEFAULT_BINARY = TOOLS_DIR / "bin" / "geotiff2pmtiles"


def _run(args, *, env=None, cwd=None, timeout=60):
    try:
        return subprocess.run(
            args, check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, cwd=cwd, timeout=timeout,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"{args[0]} failed: {detail}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not run {args[0]}: {exc}") from exc


def ensure_geotiff2pmtiles(tools_dir=TOOLS_DIR):
    """Check GitHub on every call; build once per commit and return an immutable path.

    A failed lookup/build raises rather than silently using an older binary.
    Existing revision binaries remain available to already-running batches.
    """
    tools_dir = Path(tools_dir).resolve()
    print(f"Checking {MODULE}@{BRANCH}...", file=sys.stderr, flush=True)
    git_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    remote = _run(
        ["git", "ls-remote", "--exit-code", REPOSITORY, f"refs/heads/{BRANCH}"],
        env=git_env,
    ).split()
    if len(remote) != 2 or not re.fullmatch(r"[0-9a-f]{40}", remote[0]):
        raise RuntimeError(f"Could not resolve {REPOSITORY} branch {BRANCH}")
    commit = remote[0]
    binary = tools_dir / "geotiff2pmtiles" / commit / "geotiff2pmtiles"

    if not binary.is_file() or not os.access(binary, os.X_OK):
        _run(["pkg-config", "--exists", "libwebp"])
        tools_dir.mkdir(parents=True, exist_ok=True)
        print(f"Building g2p {commit[:12]} with native libwebp...", file=sys.stderr, flush=True)
        # GOBIN must be absolute. Staging on the same filesystem makes publishing
        # atomic; failed builds never replace a working executable.
        with tempfile.TemporaryDirectory(prefix="g2p-build-", dir=tools_dir) as stage:
            built_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            env = {
                **git_env, "GOBIN": stage, "CGO_ENABLED": "1",
                "GOWORK": "off", "GOPROXY": "direct",
            }
            _run([
                "go", "install", "-ldflags",
                f"-X main.version={BRANCH}-{commit[:12]} -X main.commit={commit} "
                f"-X main.buildDate={built_at}",
                f"{MODULE}/cmd/geotiff2pmtiles@{commit}",
            ], env=env, cwd=stage, timeout=1800)
            staged_binary = Path(stage) / "geotiff2pmtiles"
            version = _run([str(staged_binary), "--version"])
            metadata = {
                "repository": REPOSITORY, "branch": BRANCH, "commit": commit,
                "built_at": built_at, "cgo_enabled": True, "version": version,
            }
            staged_metadata = Path(stage) / "build.json"
            staged_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            binary.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_metadata, binary.parent / "build.json")
            os.replace(staged_binary, binary)

    # Convenience path for direct CLI use. The slicer uses the returned revision
    # path so another workflow updating this link cannot change its running batch.
    bin_dir = tools_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".g2p-link-", dir=bin_dir) as stage:
        link = Path(stage) / "geotiff2pmtiles"
        link.symlink_to(binary)
        os.replace(link, bin_dir / "geotiff2pmtiles")
    print(f"Using g2p {commit}", file=sys.stderr, flush=True)
    return binary


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        print(ensure_geotiff2pmtiles())
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
