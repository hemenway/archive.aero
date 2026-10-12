"""Chart groups whose source files resolve to one chart key fail as a whole."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from slicer import ChartSlicer


class ChartCollisionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = ChartSlicer.__new__(ChartSlicer)
        s.logs = []
        s.log = s.logs.append
        s.chart_temp_dir = Path(tmp.name)
        s.chart_pmtiles_dir = Path(tmp.name) / 'pm'
        s.chart_collisions = []
        s.keep_chart_temp = False
        s._build_gcp_warp_context = lambda rec: None
        s._row_shapefile = lambda rec: None
        s._chart_manifest_load = lambda: {}
        s._is_valid_chart_pmtiles = lambda path: False
        s._is_pdf_payload = lambda path: False
        s.warped = []
        s.warp_and_cut = lambda src, shp, out, record=None, full_sheet=False: s.warped.append(src.name) and False
        self.slicer = s

    def run_group(self, names_by_stem):
        s = self.slicer
        s.find_files = lambda *args: [Path(f'/src/{stem}.tif') for stem in names_by_stem]
        s._chart_identity = lambda rec, file_stem=None: ('boston', names_by_stem[file_stem], '')
        rec = {'location': 'Boston', 'edition': '1', 'filename': 'Boston_1955.tif'}
        s._chart_group_worker('1955-10-13', {'records': [rec]})

    def test_colliding_files_are_reported_and_nothing_is_converted(self):
        self.run_group({'Boston_1955_a': '1955-10-13', 'Boston_1955_b': '1955-10-13'})
        self.assertEqual(self.slicer.warped, [])
        self.assertEqual(self.slicer.chart_collisions,
                         [('1955-10-13', 'Boston', '1955-10-13', 'Boston_1955_b.tif')])

    def test_distinct_halves_are_both_converted(self):
        self.run_group({'Boston_1955_N': '1955-10-13-N', 'Boston_1955_S': '1955-10-13-S'})
        self.assertEqual(self.slicer.warped, ['Boston_1955_N.tif', 'Boston_1955_S.tif'])
        self.assertEqual(self.slicer.chart_collisions, [])


if __name__ == '__main__':
    unittest.main()
