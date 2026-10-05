"""The dole `crop` column: overscan outside the box never reaches a warp."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import dole_v2
from slicer import ChartSlicer

W, H = 400, 300                 # raw file, pixels
BOX = (60, 40, 340, 260)        # the "sheet" inside the overscan, raw frame
MARK = (200, 150)               # one distinctive pixel inside the sheet


def _write(path, arr):
    ds = gdal.GetDriverByName('GTiff').Create(str(path), arr.shape[2], arr.shape[1], 3, gdal.GDT_Byte)
    for i in range(3):
        ds.GetRasterBand(i + 1).WriteArray(arr[i])
    ds = None


def _row(name, rotation, pixels_display, crop=''):
    row = {k: '' for k in dole_v2.V2_FIELDS}
    row.update(filename=name, location='Test', date='1955-10-13', cutline='none', crop=crop,
               rotation=str(rotation or ''), lcc_lat1='33', lcc_lat2='45', lcc_lat0='34', lcc_lon0='-122.5')
    geo = [(-123, 46), (-120, 46), (-120, 44), (-123, 44)]
    for i, ((px, py), (lon, lat)) in enumerate(zip(pixels_display, geo), 1):
        row.update({f'gcp{i}_px': str(px), f'gcp{i}_py': str(py), f'gcp{i}_lon': str(lon), f'gcp{i}_lat': str(lat)})
    return row


def _mark_xy(path):
    """Georeferenced centre of the red marker in a warp output."""
    ds = gdal.Open(str(path))
    a = ds.ReadAsArray()
    ys, xs = np.nonzero((a[0] == 255) & (a[1] == 0) & (a[2] == 0) & (a[3] == 255))
    gt = ds.GetGeoTransform()
    return gt[0] + (xs.mean() + 0.5) * gt[1], gt[3] + (ys.mean() + 0.5) * gt[5]


class CropTests(unittest.TestCase):
    def setUp(self):
        gdal.UseExceptions()
        previous = gdal.GetConfigOption('GDAL_PAM_ENABLED')
        self.addCleanup(gdal.SetConfigOption, 'GDAL_PAM_ENABLED', previous)
        gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
        self.slicer = ChartSlicer.__new__(ChartSlicer)
        self.slicer.log = lambda message: None
        self.slicer.warp_multithread = False
        self.slicer.resample_alg = gdal.GRA_NearestNeighbour
        self.slicer.warp_output = 'tif'

    def test_crop_matches_a_physically_cropped_source(self):
        x0, y0, x1, y1 = BOX
        arr = np.full((3, H, W), 20, np.uint8)                 # overscan: near-black scanner bed
        arr[:, y0:y1, x0:x1] = 230                             # the sheet
        arr[:, MARK[1] - 2:MARK[1] + 3, MARK[0] - 2:MARK[0] + 3] = np.array([255, 0, 0], np.uint8)[:, None, None]
        for rotation in (0, 90, 180, 270):
            with self.subTest(rotation=rotation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.slicer.shape_dir = root
                full = root / 'full.tif'
                cut = root / 'cut.tif'
                _write(full, arr)
                _write(cut, arr[:, y0:y1, x0:x1])
                # GCPs on the sheet corners as the tool stores them: display frame.
                raw_corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                disp = [dole_v2.px_raw_to_display(x, y, rotation, W, H) for x, y in raw_corners]
                crop = dole_v2.format_crop(*dole_v2.crop_raw_to_display(BOX, rotation, W, H))
                cw, ch = x1 - x0, y1 - y0
                disp_cut = [dole_v2.px_raw_to_display(x - x0, y - y0, rotation, cw, ch) for x, y in raw_corners]

                out_crop, out_cut, out_full = root / 'o_crop.tif', root / 'o_cut.tif', root / 'o_full.tif'
                self.assertTrue(self.slicer.warp_and_cut_from_row_gcps(
                    full, _row(full.name, rotation, disp, crop), out_crop))
                self.assertTrue(self.slicer.warp_and_cut_from_row_gcps(
                    cut, _row(cut.name, rotation, disp_cut), out_cut))
                self.assertTrue(self.slicer.warp_and_cut_from_row_gcps(
                    full, _row(full.name, rotation, disp), out_full))

                a, b = gdal.Open(str(out_crop)), gdal.Open(str(out_cut))
                self.assertEqual(a.GetGeoTransform(), b.GetGeoTransform())
                np.testing.assert_array_equal(a.ReadAsArray(), b.ReadAsArray())
                # No scanner bed survives, and the sheet did not move.
                pix = a.ReadAsArray()
                self.assertFalse(((pix[0] == 20) & (pix[3] == 255)).any())
                full_pix = gdal.Open(str(out_full)).ReadAsArray()
                self.assertTrue(((full_pix[0] == 20) & (full_pix[3] == 255)).any())
                res = a.GetGeoTransform()[1]
                np.testing.assert_allclose(_mark_xy(out_crop), _mark_xy(out_full), atol=res)

    def test_bad_crop_fails_the_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.slicer.shape_dir = root
            full = root / 'full.tif'
            _write(full, np.full((3, H, W), 200, np.uint8))
            corners = [(0, 0), (W, 0), (W, H), (0, H)]
            for crop in ('10,10,5,5', 'junk', '5000,5000,6000,6000'):
                with self.subTest(crop=crop):
                    self.assertFalse(self.slicer.warp_and_cut_from_row_gcps(
                        full, _row(full.name, 0, corners, crop), root / 'o.tif'))

    def test_crop_is_part_of_the_warp_fingerprint(self):
        self.assertIn('crop', ChartSlicer.GEOREF_FIELDS)
        self.assertIn('crop', dole_v2.V2_FIELDS)


if __name__ == '__main__':
    unittest.main()
