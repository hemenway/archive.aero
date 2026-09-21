"""Original UInt16 scans stay intact while GCP warps render as Byte."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import dole_v2
from slicer import ChartSlicer


class HighBitSourceTests(unittest.TestCase):
    def test_tiff_and_persistent_vrt_match_8bit_rendering_without_changing_master(self):
        gdal.UseExceptions()
        previous_pam=gdal.GetConfigOption('GDAL_PAM_ENABLED')
        self.addCleanup(gdal.SetConfigOption,'GDAL_PAM_ENABLED',previous_pam)
        gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            master=root/'original.tif'; old=root/'old_8bit.tif'
            w,h=384,256
            values=np.random.default_rng(42).integers(0,65536,(3,h,w),dtype=np.uint16)
            ds=gdal.GetDriverByName('GTiff').Create(str(master),w,h,3,gdal.GDT_UInt16)
            for i in range(3):
                ds.GetRasterBand(i+1).WriteArray(values[i])
                ds.GetRasterBand(i+1).SetColorInterpretation((gdal.GCI_RedBand,gdal.GCI_GreenBand,gdal.GCI_BlueBand)[i])
            ds=None
            original_hash=hashlib.sha256(master.read_bytes()).hexdigest()
            ds=gdal.GetDriverByName('GTiff').Create(str(old),w,h,3,gdal.GDT_Byte)
            reduced=((values.astype(np.uint32)+128)//257).astype(np.uint8)
            for i in range(3):
                ds.GetRasterBand(i+1).WriteArray(reduced[i])
                ds.GetRasterBand(i+1).SetColorInterpretation((gdal.GCI_RedBand,gdal.GCI_GreenBand,gdal.GCI_BlueBand)[i])
            ds=None
            row={k:'' for k in dole_v2.V2_FIELDS}
            row.update(filename=master.name,location='Test',date='1955-10-13',cutline='none',lcc_lat1='33',lcc_lat2='45',lcc_lat0='34',lcc_lon0='-122.5')
            for i,(px,py,lon,lat) in enumerate([(0,0,-123,46),(w,0,-120,46),(w,h,-120,44),(0,h,-123,44)],1):
                row.update({f'gcp{i}_px':str(px),f'gcp{i}_py':str(py),f'gcp{i}_lon':str(lon),f'gcp{i}_lat':str(lat)})
            slicer=ChartSlicer.__new__(ChartSlicer)
            slicer.shape_dir=root;slicer.log=lambda message:None
            slicer.warp_multithread=False;slicer.resample_alg=gdal.GRA_NearestNeighbour
            for mode in ('tif','vrt'):
                with self.subTest(mode=mode):
                    slicer.warp_output=mode
                    original_render=root/f'original_render.{mode}';old_render=root/f'old_render.{mode}'
                    self.assertTrue(slicer.warp_and_cut_from_row_gcps(master,row,original_render))
                    self.assertTrue(slicer.warp_and_cut_from_row_gcps(old,row,old_render))
                    # Reopen after the method returns: VRT source lifetime matters.
                    a=gdal.Open(str(original_render));b=gdal.Open(str(old_render))
                    self.assertEqual(a.GetRasterBand(1).DataType,gdal.GDT_Byte)
                    np.testing.assert_array_equal(a.ReadAsArray(),b.ReadAsArray())
                    self.assertEqual(a.GetGeoTransform(),b.GetGeoTransform())
                    a=b=None
                    self.assertEqual(hashlib.sha256(master.read_bytes()).hexdigest(),original_hash)
            ds=gdal.Open(str(master));self.assertEqual(ds.GetRasterBand(1).DataType,gdal.GDT_UInt16)


if __name__=='__main__':unittest.main()
