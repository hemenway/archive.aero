import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.viewer_config import update_viewer_config, viewer_config_source


class ViewerConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="archive-viewer-config-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        repo = Path(__file__).resolve().parents[1]
        for relative in ["index.html", "src/viewer.js", "src/csv.js", "src/boot.js", "scripts/build_frontend.mjs"]:
            dest = self.root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / relative, dest)

    def assert_generated(self):
        subprocess.run(["node", str(self.root / "scripts/build_frontend.mjs"), "--check"], check=True, capture_output=True)
        html = (self.root / "index.html").read_text()
        for relative in re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html):
            self.assertTrue((self.root / relative.lstrip("/")).is_file())

    def test_old_html_entry_updates_source_and_generated_modules(self):
        html = self.root / "index.html"
        old_html = html.read_text()
        source = update_viewer_config(html, "bundleUrl", "https://example.test/metadata-new.bundle")
        self.assertEqual(source, self.root / "src/viewer.js")
        self.assertIn("bundleUrl: 'https://example.test/metadata-new.bundle'", source.read_text())
        self.assertNotEqual(old_html, html.read_text())
        self.assert_generated()

    def test_direct_source_entry_rebuilds_too(self):
        source = self.root / "src/viewer.js"
        update_viewer_config(source, "airspaceUrl", "https://example.test/airspace.pmtiles")
        self.assert_generated()

    def test_legacy_single_file_and_missing_key(self):
        legacy = self.root / "legacy.html"
        legacy.write_text("<script>const CONFIG = {bundleUrl: null};</script>")
        self.assertEqual(viewer_config_source(legacy), legacy)
        update_viewer_config(legacy, "bundleUrl", "https://example.test/a.bundle")
        updated = legacy.read_text()
        self.assertIn("bundleUrl: 'https://example.test/a.bundle'", updated)
        with self.assertRaises(ValueError):
            update_viewer_config(legacy, "missingKey", "https://example.test/nope")
        self.assertEqual(legacy.read_text(), updated)


if __name__ == "__main__":
    unittest.main()
