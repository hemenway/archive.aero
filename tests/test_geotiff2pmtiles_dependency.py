"""Dependency updates must not change running batches or mask update failures."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import install_geotiff2pmtiles as g2p


class DependencyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tools = Path(directory.name)
        self.commit = "a" * 40
        self.installs = []
        self.fail_lookup = False
        self.fail_build = False

    def run_command(self, args, **kwargs):
        if args[:2] == ["git", "ls-remote"]:
            if self.fail_lookup:
                raise subprocess.CalledProcessError(1, args, stderr="GitHub unavailable")
            output = f"{self.commit}\trefs/heads/main\n"
        elif args[:2] == ["pkg-config", "--exists"]:
            output = ""
        elif args[:2] == ["go", "install"]:
            self.installs.append(args[-1])
            self.assertEqual(kwargs["env"]["CGO_ENABLED"], "1")
            self.assertEqual(kwargs["env"]["GOWORK"], "off")
            binary = Path(kwargs["env"]["GOBIN"]) / "geotiff2pmtiles"
            binary.write_text(self.commit)
            binary.chmod(0o755)
            if self.fail_build:
                raise subprocess.CalledProcessError(1, args, stderr="compiler failed")
            output = ""
        elif args[-1] == "--version":
            output = "geotiff2pmtiles main-test\nformats: jpeg, png, webp, terrarium"
        else:
            self.fail(f"Unexpected command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    def install(self):
        with patch.object(g2p.subprocess, "run", side_effect=self.run_command):
            return g2p.ensure_geotiff2pmtiles(self.tools)

    def test_checks_main_each_time_and_reuses_commit_build(self):
        first = self.install()
        self.assertEqual(self.install(), first)
        self.assertEqual(len(self.installs), 1)
        self.assertTrue(self.installs[0].endswith("@" + self.commit))
        metadata = json.loads((first.parent / "build.json").read_text())
        self.assertEqual(metadata["commit"], self.commit)
        self.assertTrue(metadata["cgo_enabled"])

        self.commit = "b" * 40
        second = self.install()
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_text(), "a" * 40)
        self.assertEqual(second.read_text(), "b" * 40)
        self.assertEqual((self.tools / "bin/geotiff2pmtiles").resolve(), second)
        self.assertEqual(len(self.installs), 2)

    def test_failed_lookup_never_falls_back_to_cached_binary(self):
        first = self.install()
        self.fail_lookup = True
        with self.assertRaisesRegex(RuntimeError, "GitHub unavailable"):
            self.install()
        self.assertEqual((self.tools / "bin/geotiff2pmtiles").resolve(), first)

    def test_failed_build_preserves_previous_install_and_cleans_staging(self):
        first = self.install()
        self.commit = "b" * 40
        self.fail_build = True
        with self.assertRaisesRegex(RuntimeError, "compiler failed"):
            self.install()
        self.assertEqual((self.tools / "bin/geotiff2pmtiles").resolve(), first)
        self.assertTrue(os.access(first, os.X_OK))
        self.assertFalse((self.tools / "geotiff2pmtiles" / self.commit).exists())
        self.assertEqual(list(self.tools.glob("g2p-build-*")), [])


if __name__ == "__main__":
    unittest.main()
