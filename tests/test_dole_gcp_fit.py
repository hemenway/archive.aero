"""Corner-GCP plausibility (dole_v2.gcp_fit / gcp_fit_problems).

Synthetic scans: a sheet's true corners are projected into the row's chart
projection and mapped to pixels through a known affine (square pixels, a
small scan rotation), exactly as a correct scan of a correct chart would be.
"""
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import dole_v2

LCC_45_33 = dict(lcc_lat1='33', lcc_lat2='45', lcc_lat0='23', lcc_lon0='-96')


def scan(corners, row_proj, m_per_px=40.0, rot_deg=0.4, y_stretch=1.0, origin=(500, 300)):
    """Pixel positions of ground corners (lon, lat) on a synthetic scan."""
    project = dole_v2.row_projector(row_proj)
    ground = [project(lon, lat) for lon, lat in corners]
    gx0 = min(g[0] for g in ground)
    gy1 = max(g[1] for g in ground)
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    out = []
    for gx, gy in ground:
        e, n = (gx - gx0) / m_per_px, (gy1 - gy) / (m_per_px * y_stretch)
        out.append((origin[0] + c * e - s * n, origin[1] + s * e + c * n))
    return out


def row_for(pixels, corners, **proj):
    row = {k: '' for k in dole_v2.V2_FIELDS}
    row.update(filename='synthetic.tif', location='Boston', **proj)
    for i, ((px, py), (lon, lat)) in enumerate(zip(pixels, corners), 1):
        row.update({f'gcp{i}_px': f'{px:.2f}', f'gcp{i}_py': f'{py:.2f}',
                    f'gcp{i}_lon': str(lon), f'gcp{i}_lat': str(lat)})
    return row


def sheet(north, south, west, east):
    return [(west, north), (east, north), (east, south), (west, south)]  # TL TR BR BL


class GcpFitTests(unittest.TestCase):
    def test_a_correct_scan_fits_square_and_tight(self):
        corners = sheet(44, 41, -72, -69)
        fit = dole_v2.gcp_fit(row_for(scan(corners, LCC_45_33), corners, **LCC_45_33))
        self.assertAlmostEqual(fit['aspect'], 1.0, places=3)
        self.assertAlmostEqual(fit['scale_ratio'], 1.0, places=3)
        self.assertLess(fit['shear_deg'], 0.05)
        self.assertLess(fit['rms_m'], 50)   # pixel rounding only
        self.assertFalse(fit['mirrored'])
        self.assertAlmostEqual(fit['m_per_px'], 40.0, delta=0.2)
        self.assertEqual(dole_v2.gcp_fit_problems(row_for(scan(corners, LCC_45_33), corners, **LCC_45_33)),
                         ('ok', []))

    def test_boston_1953_rows_are_refused(self):
        # The 1953-12 -> 1970 Boston sheet is 44-40N; its rows kept the older
        # 44-41N corners. Worklist 04: pixel span y/x 1.73 vs 1.32 -> 0.76.
        printed = sheet(44, 40, -72, -69)
        catalog = sheet(44, 41, -72, -69)
        row = row_for(scan(printed, LCC_45_33), catalog, **LCC_45_33)
        fit = dole_v2.gcp_fit(row)
        self.assertAlmostEqual(fit['aspect'], 0.76, delta=0.02)
        level, reasons = dole_v2.gcp_fit_problems(row)
        self.assertEqual(level, 'error')
        self.assertIn('aspect 0.7', reasons[0])
        # ...and the fixed rows (gcp3/gcp4 lat 41 -> 40) pass.
        self.assertEqual(dole_v2.gcp_fit_problems(row_for(scan(printed, LCC_45_33), printed, **LCC_45_33))[0], 'ok')

    def test_paper_shrinkage_is_tolerated_up_to_the_warning_band(self):
        corners = sheet(44, 41, -72, -69)
        for stretch, want in ((1.015, 'ok'), (1.05, 'warn'), (1.15, 'error')):
            row = row_for(scan(corners, LCC_45_33, y_stretch=stretch), corners, **LCC_45_33)
            self.assertEqual(dole_v2.gcp_fit_problems(row)[0], want, stretch)

    def test_the_check_does_not_depend_on_scan_rotation(self):
        # A sheet far from the projection's central meridian, scanned upright
        # or turned: the same squeeze reads the same.
        printed, catalog = sheet(44, 40, -72, -69), sheet(44, 41, -72, -69)
        aspects = [dole_v2.gcp_fit(row_for(scan(printed, LCC_45_33, rot_deg=r), catalog, **LCC_45_33))['aspect']
                   for r in (-16.0, 0.0, 7.5, 90.0)]
        self.assertLess(max(aspects) - min(aspects), 1e-6)

    def test_mirror_order_is_an_error(self):
        corners = sheet(44, 41, -72, -69)
        pixels = scan(corners, LCC_45_33)
        swapped = [corners[1], corners[0], corners[3], corners[2]]   # TL<->TR, BL<->BR
        level, reasons = dole_v2.gcp_fit_problems(row_for(pixels, swapped, **LCC_45_33))
        self.assertEqual(level, 'error')
        self.assertTrue(any('mirror' in r for r in reasons))

    def test_one_misplaced_corner_shows_as_residual(self):
        corners = sheet(44, 41, -72, -69)
        pixels = scan(corners, LCC_45_33)
        typo = list(corners)
        typo[2] = (-68.0, 41.0)            # BR lon typo: 69 -> 68
        fit = dole_v2.gcp_fit(row_for(pixels, typo, **LCC_45_33))
        self.assertGreater(fit['rms_m'], GCP_ERROR_RMS)
        self.assertEqual(dole_v2.gcp_fit_problems(row_for(pixels, typo, **LCC_45_33))[0], 'error')

    def test_wrong_standard_parallels_warn_through_the_residual(self):
        # A wide WASP-style sheet printed with the modern 33 20'/38 40'
        # parallels but cataloged as 45/33: ~1 km at the corners, the band
        # worklist 04 measured on the Dallas WASP rows (237 m vs 25 m was the
        # milder case). Mercator is far worse.
        corners = sheet(42, 37, -120, -110.4)
        printed_in = dict(lcc_lat1='33.333333', lcc_lat2='38.666667', lcc_lat0='37', lcc_lon0='-115')
        old_sp = dict(lcc_lat1='33', lcc_lat2='45', lcc_lat0='37', lcc_lon0='-115')
        merc = dict(proj='merc', lcc_lon0='-115')
        pixels = scan(corners, printed_in)
        self.assertEqual(dole_v2.gcp_fit_problems(row_for(pixels, corners, **printed_in))[0], 'ok')
        self.assertEqual(dole_v2.gcp_fit_problems(row_for(pixels, corners, **old_sp))[0], 'warn')
        self.assertEqual(dole_v2.gcp_fit_problems(row_for(pixels, corners, **merc))[0], 'error')

    def test_mercator_rows_fit_in_mercator(self):
        corners = sheet(25.5, 24.0, -83.0, -80.0)   # a Key West strip
        merc = dict(proj='merc', lcc_lon0='-81.5')
        self.assertEqual(dole_v2.gcp_fit_problems(row_for(scan(corners, merc), corners, **merc)), ('ok', []))

    def test_rows_without_gcps_or_projection_have_nothing_to_check(self):
        row = {k: '' for k in dole_v2.V2_FIELDS}
        self.assertIsNone(dole_v2.gcp_fit(row))
        self.assertEqual(dole_v2.gcp_fit_problems(row), ('ok', []))
        corners = sheet(44, 41, -72, -69)
        no_proj = row_for(scan(corners, LCC_45_33), corners)
        self.assertIsNone(dole_v2.gcp_fit(no_proj))

    def test_repeated_pixels_are_degenerate(self):
        corners = sheet(44, 41, -72, -69)
        row = row_for([(10, 10)] * 4, corners, **LCC_45_33)
        self.assertEqual(dole_v2.gcp_fit_problems(row)[0], 'error')

    def test_unknown_proj_is_an_error_not_a_crash(self):
        corners = sheet(44, 41, -72, -69)
        row = row_for(scan(corners, LCC_45_33), corners, proj='polyconic', lcc_lon0='-70')
        self.assertEqual(dole_v2.gcp_fit_problems(row)[0], 'error')


GCP_ERROR_RMS = dole_v2.GCP_FIT_ERROR['rms_m']


class ProjectionAgreesWithProj(unittest.TestCase):
    def test_against_gdal(self):
        try:
            from osgeo import osr
        except ImportError:
            self.skipTest('GDAL Python bindings not installed')
        osr.UseExceptions()
        rng = random.Random(7)
        for row in (LCC_45_33, dict(proj='merc', lcc_lon0='-81.5'),
                    dict(lcc_lat1='50', lcc_lat2='58', lcc_lat0='52', lcc_lon0='178')):
            src, dst = osr.SpatialReference(), osr.SpatialReference()
            src.ImportFromEPSG(4269)
            dst.SetFromUserInput(dole_v2.row_lcc_crs(row))
            for s in (src, dst):
                s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            t = osr.CoordinateTransformation(src, dst)
            project = dole_v2.row_projector(row)
            lon0 = float(row['lcc_lon0'])
            for _ in range(100):
                lon = (lon0 + rng.uniform(-10, 10) + 180) % 360 - 180
                lat = rng.uniform(20, 30) if row.get('proj') else rng.uniform(18, 68)
                x, y, _ = t.TransformPoint(lon, lat)
                px, py = project(lon, lat)
                self.assertLess(max(abs(x - px), abs(y - py)), 0.001, (row, lon, lat))


if __name__ == '__main__':
    unittest.main()
