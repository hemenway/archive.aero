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
        self.write("site-files.json", json.dumps(["index.html", "dates.csv", "styles.css", "coverage.json"]))
        self.write("index.html", '<html><head><meta charset="utf-8" /><title>v</title></head><body>published viewer</body></html>')
        self.write("dates.csv", "date_iso,url\n")
        self.write("styles.css", "body{}")
        self.write("coverage.json", "{}")

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
            {"index.html", "dates.csv", "styles.css", "coverage.json", old, current, ".nojekyll",
             "classic/index.html", "classic/styles.css", "classic/dates.csv", "classic/coverage.json"},
        )

    def test_classic_copy_is_the_viewer_marked_noindex(self):
        output, files = builder.build_site(self.root)
        page = (output / "classic/index.html").read_text()
        self.assertEqual(page.count('<meta name="robots" content="noindex">'), 1)
        self.assertEqual(page.replace('<meta name="robots" content="noindex">', ""), (self.root / "index.html").read_text())
        self.assertIn('<meta charset="utf-8" /><meta name="robots" content="noindex"><title>', page)
        self.assertEqual((output / "classic/styles.css").read_text(), "body{}")
        self.assertIn("classic/index.html", files)
        # A viewer page that does not declare its charset once cannot be marked.
        self.write("index.html", "no charset")
        with self.assertRaisesRegex(ValueError, "charset"):
            builder.build_site(self.root)
        # The copy never pulls in a file that is not itself published.
        self.write("index.html", '<meta charset="utf-8">')
        self.write("site-files.json", json.dumps(["index.html", "dates.csv"]))
        with self.assertRaisesRegex(ValueError, "classic"):
            builder.build_site(self.root)
        self.write("site-files.json", json.dumps(["dates.csv"]))
        output, files = builder.build_site(self.root)
        self.assertFalse((output / "classic").exists())

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
