import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "build_pages", Path(__file__).resolve().parents[1] / "scripts/build_pages.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PagesPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.write("site-files.json", json.dumps(["index.html", "dates.csv"]))
        self.write("index.html", "published viewer")
        self.write("dates.csv", "date_iso,url\n")

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def test_only_site_files_and_committed_fingerprints_are_published(self):
        old = "assets/viewer.0123456789abcdef.js"
        current = "assets/boot.abcdef0123456789.js"
        private = [".env", "worklists/README.md", "scripts/slicer.py",
                   "master_dole_v2.csv", "worklists/contacts.csv",
                   "assets/notes.md", "assets/unversioned.js"]
        for path in [old, current, *private]:
            self.write(path, "fixture")
        subprocess.run(["git", "add", "--", old, current, *private],
                       cwd=self.root, check=True)
        self.write("assets/csv.aaaaaaaaaaaaaaaa.js", "untracked output")
        self.write("_site/worklists/README.md", "stale exposed content")
        output, _ = builder.build_site(self.root)
        self.assertEqual(
            {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()},
            {"index.html", "dates.csv", old, current, ".nojekyll"},
        )

    def test_missing_input_stops_publication(self):
        (self.root / "dates.csv").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            builder.build_site(self.root)

    def test_rejects_escaping_paths_and_directories(self):
        for entry in ["../private.csv", "/etc/passwd", "assets/../.env", "_site/index.html"]:
            with self.subTest(entry=entry):
                self.write("site-files.json", json.dumps([entry]))
                with self.assertRaises(ValueError):
                    builder.build_site(self.root)

    def test_rejects_symlinked_input_and_output(self):
        (self.root / "dates.csv").unlink()
        (self.root / "dates.csv").symlink_to(self.root / "index.html")
        with self.assertRaisesRegex(ValueError, "symlink"):
            builder.build_site(self.root)
        (self.root / "dates.csv").unlink()
        self.write("dates.csv", "fixture")
        (self.root / "_site").symlink_to(self.root / "assets", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            builder.build_site(self.root)


if __name__ == "__main__":
    unittest.main()
