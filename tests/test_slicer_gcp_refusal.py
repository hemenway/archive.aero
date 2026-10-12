"""The slicer refuses catalog rows whose corner GCPs fail the fit check."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dole_v2
from slicer import ChartSlicer
from test_dole_gcp_fit import LCC_45_33, row_for, scan, sheet


class GcpRefusalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        old, new = sheet(44, 41, -72, -69), sheet(44, 40, -72, -69)
        rows = []
        r = row_for(scan(old, LCC_45_33), old, **LCC_45_33)
        r.update(filename='boston_1950.tif', date='1950-01-12', end_date='1950-07-06', cutline='extents/boston_ma')
        rows.append(r)
        # The 1953 format change under copied 41N corners, with a georeferenced
        # alternate download of the same edition.
        r = row_for(scan(new, LCC_45_33), old, **LCC_45_33)
        r.update(filename='boston_1953.tif', date='1953-12-02', end_date='1954-06-03', cutline='extents/boston_ma')
        rows.append(r)
        alt = {k: '' for k in dole_v2.V2_FIELDS}
        alt.update(filename='boston_1953_georef.tif', location='Boston', date='1953-12-02', end_date='1954-06-03')
        rows.append(alt)
        self.csv = Path(tmp.name) / 'dole.csv'
        with open(self.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=dole_v2.V2_FIELDS)
            w.writeheader()
            w.writerows(rows)

    def load(self, allow=False):
        tmp = self.csv.parent
        s = ChartSlicer(tmp, tmp / 'out', self.csv, tmp, temp_dir=tmp / 'temp')
        s.logs = []
        s.log = s.logs.append
        s.allow_gcp_misfit = allow
        s.load_dole_data()
        return s, sorted(r['filename'] for rows in s.dole_data.values() for r in rows)

    def test_misfit_row_is_refused_and_the_alternate_remains(self):
        s, names = self.load()
        self.assertEqual(names, ['boston_1950.tif', 'boston_1953_georef.tif'])
        self.assertEqual([r[0] for r in s.gcp_refused], ['boston_1953.tif'])
        self.assertTrue(any('refusing boston_1953.tif' in line for line in s.logs))
        self.assertTrue(any('GCP fit refused=1' in line for line in s.logs))

    def test_override_keeps_it_with_a_warning(self):
        s, names = self.load(allow=True)
        self.assertEqual(names, ['boston_1950.tif', 'boston_1953.tif', 'boston_1953_georef.tif'])
        self.assertEqual(s.gcp_refused, [])
        self.assertTrue(any('kept: --allow-gcp-misfit' in line for line in s.logs))


if __name__ == '__main__':
    unittest.main()
