"""Sidebar completeness must inspect source metadata, never infer it from a cutline."""
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

from osgeo import gdal, osr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from source_georef import SourceGeorefIndex

# Loading the Flask module must not create a session-secret file in the repo.
_open = open
def _test_open(path, *args, **kwargs):
    if str(path).endswith('/georef_session_secret'):
        return io.StringIO('a' * 64)
    return _open(path, *args, **kwargs)

tool = None
if (ROOT / 'scripts/1georef_toolv10.py').is_file():
    spec = importlib.util.spec_from_file_location('georef_tool', ROOT / 'scripts/1georef_toolv10.py')
    tool = importlib.util.module_from_spec(spec)
    with mock.patch('builtins.open', side_effect=_test_open):
        spec.loader.exec_module(tool)


class SourceFixture(unittest.TestCase):
    def setUp(self):
        gdal.UseExceptions()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.index = None
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4269)
        self.wkt = srs.ExportToWkt()

    def tif(self, name, *, transform=False, crs=False, gcps=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = gdal.GetDriverByName('GTiff').Create(str(path), 20, 20, 1, gdal.GDT_Byte)
        if transform:
            ds.SetGeoTransform((-123, .01, 0, 46, 0, -.01))
        if crs:
            ds.SetProjection(self.wkt)
        if gcps:
            ds.SetGCPs([gdal.GCP(-123, 46, 0, 0, 0), gdal.GCP(-122, 46, 0, 20, 0),
                        gdal.GCP(-122, 45, 0, 20, 20), gdal.GCP(-123, 45, 0, 0, 20)],
                       self.wkt)
        ds = None
        return path

    def row(self, filename, **fields):
        return dict(filename=filename, cutline='sectional/test', **fields)

    def reindex(self):
        self.index = SourceGeorefIndex(self.root)
        for path in self.root.rglob('*'):
            if path.is_file():
                self.index.add_file(path)

    def status(self, row):
        if self.index is None:
            self.reindex()
        return 'georef' if self.index.has_georef(row['filename'], row.get('src_crs', '')) else 'incomplete'


class SourceGeorefTests(SourceFixture):
    def test_cutline_and_crs_alone_never_georeference_a_scan(self):
        self.tif('scan.tif')
        self.tif('crs_only.tif', crs=True)
        for row in [self.row('scan.tif'), self.row('scan.tif', src_crs='EPSG:4269'),
                    self.row('crs_only.tif'), self.row('missing.tif')]:
            with self.subTest(row=row):
                self.assertEqual(self.status(row), 'incomplete')

    def test_native_georef_and_invalid_crs_override(self):
        self.tif('native.tif', transform=True, crs=True)
        self.assertEqual(self.status(self.row('native.tif')), 'georef')
        self.assertEqual(self.status(self.row('native.tif', src_crs='invalid-crs')), 'incomplete')

    def test_world_file_requires_crs_and_invalidates_on_changes(self):
        Image.new('RGB', (20, 20)).save(self.root / 'scan.jpg')
        row = self.row('scan.jpg', src_crs='EPSG:4269')
        self.assertEqual(self.status(row), 'incomplete')
        world = self.root / 'scan.jgw'
        world.write_text('.01\n0\n0\n-.01\n-123\n46\n')
        self.assertEqual(self.status(row), 'georef')
        self.assertEqual(self.status(self.row('scan.jpg')), 'incomplete')
        # World file still exists, but no invertible pixel-to-ground mapping.
        world.write_text('0\n0\n0\n0\n-123\n46\n')
        self.assertEqual(self.status(row), 'incomplete')
        world.unlink()
        self.assertEqual(self.status(row), 'incomplete')

    def test_embedded_gcps_are_valid_source_georef(self):
        self.tif('gcps.tif', gcps=True)
        self.assertEqual(self.status(self.row('gcps.tif')), 'georef')

    @unittest.skipUnless(gdal.GetDriverByName('PDF'), 'GDAL PDF driver is unavailable')
    def test_tiff_row_uses_georeferenced_same_stem_pdf(self):
        source = self.tif('source.tif', transform=True, crs=True)
        dataset = gdal.Open(str(source))
        pdf = gdal.GetDriverByName('PDF').CreateCopy(str(self.root / 'sheet.pdf'), dataset)
        pdf = dataset = None
        self.assertEqual(self.status(self.row('sheet.tif')), 'georef')
        # An actual TIFF row wins even when a georeferenced PDF is beside it.
        self.tif('sheet.tif')
        self.reindex()
        self.assertEqual(self.status(self.row('sheet.tif')), 'incomplete')

    def test_zip_checks_every_face_without_extracting(self):
        north = self.tif('north.tif', transform=True, crs=True)
        south = self.tif('south.tif')
        for filename, members in [('mixed.zip', [north, south]), ('good.zip', [north])]:
            with zipfile.ZipFile(self.root / filename, 'w') as bundle:
                for path in members:
                    bundle.write(path, 'nested/' + path.name)
        self.reindex()
        before = sorted(self.root.rglob('*'))
        self.assertEqual(self.status(self.row('good.zip')), 'georef')
        self.assertEqual(self.status(self.row('mixed.zip')), 'incomplete')
        self.assertEqual(sorted(self.root.rglob('*')), before)

    def test_extracted_zip_faces_take_precedence_and_include_nested_members(self):
        self.tif('bundle/north.tif', transform=True, crs=True)
        self.tif('bundle/nested/south.tif')
        self.assertEqual(self.status(self.row('bundle.zip')), 'incomplete')
        self.tif('bundle/nested/south.tif', transform=True, crs=True)
        self.assertEqual(self.status(self.row('bundle.zip')), 'georef')

    def test_metadata_probe_is_cached_and_source_replacement_is_detected(self):
        self.tif('native.tif', transform=True, crs=True)
        row = self.row('native.tif')
        with mock.patch.object(gdal, 'OpenEx', wraps=gdal.OpenEx) as opened:
            self.assertEqual(self.status(row), 'georef')
            self.assertEqual(self.status(row), 'georef')
            self.assertEqual(opened.call_count, 1)
        self.tif('native.tif')
        self.assertEqual(self.status(row), 'incomplete')

    def test_persisted_cache_is_reused_and_still_checks_source_signature(self):
        path = self.tif('sources/native.tif', transform=True, crs=True)
        cache = self.root / 'metadata-cache.json'
        self.index = SourceGeorefIndex(self.root, cache)
        self.index.add_file(path)
        self.assertTrue(self.index.has_georef(path.name))
        self.index.save_cache()
        self.index = SourceGeorefIndex(self.root, cache)
        self.index.add_file(path)
        with mock.patch.object(gdal, 'OpenEx', side_effect=AssertionError('should reuse cache')):
            self.assertTrue(self.index.has_georef(path.name))
        self.tif('sources/native.tif')
        self.assertFalse(self.index.has_georef(path.name))


@unittest.skipUnless(tool, 'local georef tool is intentionally gitignored')
class GeorefToolStatusTests(SourceFixture):
    def setUp(self):
        super().setUp()
        self.saved_state = tool.state
        self.addCleanup(setattr, tool, 'state', self.saved_state)
        tool.state = tool.AppState()
        tool.state.root_path = str(self.root)
        tool.state.csv_path = str(self.root / 'absent.csv')
        patcher = mock.patch.object(tool, 'SOURCE_GEOREF_CACHE', str(self.root / 'metadata-cache.json'))
        patcher.start()
        self.addCleanup(patcher.stop)

    def status(self, row):
        return tool.row_status(row)

    def test_native_georef_accepts_all_cutline_decisions(self):
        self.tif('native.tif', transform=True, crs=True)
        for cutline in [{'cutline': 'sectional/test'}, {'cutline': 'extents/test'},
                        {'cutline': 'none'}]:
            with self.subTest(cutline=cutline):
                self.assertEqual(self.status(dict(filename='native.tif', **cutline)), 'georef')
        self.assertEqual(self.status({'filename': 'native.tif'}), 'incomplete')
        self.assertEqual(self.status({'filename': 'native.tif', 'cutline_wkt':
                                      'POLYGON((-123 46,-122 46,-122 45,-123 46))'}), 'incomplete')

    def test_ready_gcps_do_not_require_source_probe(self):
        row = self.row('not-mounted.tif', lcc_lat1='33', lcc_lat2='45',
                       lcc_lat0='39', lcc_lon0='-122')
        for i, (px, py, lon, lat) in enumerate(
                [(0, 0, -123, 46), (20, 0, -122, 46),
                 (20, 20, -122, 45), (0, 20, -123, 45)], 1):
            row.update({f'gcp{i}_px': str(px), f'gcp{i}_py': str(py),
                        f'gcp{i}_lon': str(lon), f'gcp{i}_lat': str(lat)})
        with mock.patch.object(tool, 'build_file_index', side_effect=AssertionError('unexpected probe')):
            self.assertEqual(self.status(row), 'ready')

    def test_timeline_keeps_unreferenced_scans_in_the_incomplete_list(self):
        self.tif('scan.tif')
        self.tif('native.tif', transform=True, crs=True)
        rows = [self.row('scan.tif'), self.row('native.tif')]
        with mock.patch.object(tool.analyze_dole, 'analyze_csv', return_value=([], {'Test': rows})):
            response = tool.app.test_client().get('/timeline_data')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['counts'], {'incomplete': 1, 'georef': 1, 'ready': 0})
        self.assertEqual([r['filename'] for r in data['incomplete']], ['scan.tif'])


if __name__ == '__main__':
    unittest.main()
