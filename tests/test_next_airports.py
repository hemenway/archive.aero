import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from next_build_airports import (PILOT, point_in_ring, point_in_rings, point_in_box, covers, airport_charts, decades,
                                 airport_record, sitemap_xml, read_idents, build)

CSV_HEAD='"id","ident","type","name","latitude_deg","longitude_deg","elevation_ft","continent","iso_country","iso_region","municipality","scheduled_service","icao_code","iata_code","gps_code","local_code","home_link","wikipedia_link","keywords"\n'
CSV_ROWS=('3422,"KBOS","large_airport","Boston Logan International Airport",42.36197,-71.0079,20,"NA","US","US-MA","Boston","yes","KBOS","BOS","KBOS","BOS",,,\n'
          '5453,"PHNL","large_airport","Daniel K. Inouye International Airport",21.318387,-157.92567,13,"OC","US","US-HI","Honolulu, Oahu","yes","PHNL","HNL","PHNL","HNL",,,\n'
          '1,"PADK","medium_airport","Adak Airport",51.878,-176.646,18,"NA","US","US-AK","Adak Island","yes","PADK","ADK","PADK","ADK",,,\n')
BOSTON_RING=[[[-72,44],[-69,44],[-69,41],[-72,41],[-72,44]]]
AM_RING=[[[178,53],[-172.4,53],[-172.6,51],[180,51],[178,51],[178,53]]]


class NextAirportsTests(unittest.TestCase):
    def test_point_in_ring_multi_ring_and_antimeridian(self):
        self.assertTrue(point_in_ring(-71,42.36,BOSTON_RING[0])); self.assertFalse(point_in_ring(-74,42.36,BOSTON_RING[0]))
        self.assertFalse(point_in_ring(-71,40.9,BOSTON_RING[0]))
        two=[[[-72,44],[-69,44],[-69,41],[-72,41],[-72,44]],[[10,10],[12,10],[12,12],[10,12],[10,10]]]
        self.assertTrue(point_in_rings(11,11,two)); self.assertTrue(point_in_rings(-70,42,two)); self.assertFalse(point_in_rings(0,0,two))
        # Adak (-176.6) and Attu (173) both sit on a ring whose lons straddle 180; Honolulu does not.
        self.assertTrue(point_in_ring(-176.646,51.878,AM_RING[0])); self.assertTrue(point_in_ring(179,52,AM_RING[0]))
        self.assertFalse(point_in_ring(-157.9,21.3,AM_RING[0])); self.assertFalse(point_in_ring(175,50,AM_RING[0]))
        self.assertTrue(point_in_box(-176.6,52,[177,51,-172,53])); self.assertFalse(point_in_box(-157.9,52,[177,51,-172,53]))
    def test_pmb_and_ring_together(self):
        # The warped artifact stops short of the ring: the ring says yes, the pmb says no, the pmb excludes.
        tiled={'d':'1950-01-01','pm':'x','pmb':[-72,42.5,-69,44]}
        self.assertFalse(covers(-71,42.36,tiled,BOSTON_RING)); self.assertTrue(covers(-71,43,tiled,BOSTON_RING))
        # The artifact box runs a degree past the ring (the collar): a box-only hit is not coverage.
        collar={'d':'1950-01-01','pm':'x','pmb':[-73,40,-68,45]}
        self.assertTrue(covers(-71,42.36,collar,BOSTON_RING)); self.assertFalse(covers(-72.5,42.36,collar,BOSTON_RING))
        self.assertTrue(covers(-72.5,42.36,collar,None),'no ring: the box decides alone')
        self.assertTrue(covers(-71,42.36,{'d':'1950-01-01'},BOSTON_RING)); self.assertFalse(covers(-74,42.36,{'d':'1950-01-01'},BOSTON_RING))
        self.assertIsNone(covers(-71,42.36,{'d':'1950-01-01'},None),'untiled and no ring: unplaceable')
        self.assertTrue(covers(-176.6,52,{'pm':'x','pmb':[-180,-85.0512,180,85.0512]},AM_RING),'the conservative world box defers to the ring')
        self.assertFalse(covers(-157.9,21.3,{'pm':'x','pmb':[-180,-85.0512,180,85.0512]},AM_RING))
    def test_decades_first_last_and_untiled_flag(self):
        bos={'ident':'kbos','icao':'KBOS','lat':42.36,'lon':-71.0}
        index=[{'slug':'boston_ma','names':['Boston, MA'],'ref':'extents/boston_ma','type':'sectional','charts':[
                    {'d':'1939-10-01','e':'1940-04-01','ed':'9','pm':'p','pmb':[-72.4,40.4,-68.6,44.3],'og':'sectionals/chart/boston_ma/1939-10-01.jpg'},
                    {'d':'1933-03-01','e':'1933-08-01','ed':'1','pm':'p','pmb':[-72.4,40.4,-68.6,44.3]},
                    {'d':'1957-06-01','e':'1957-11-21','ed':'84'},
                    {'d':'1960-01-01','e':'1960-06-01','ed':'90','pm':'p','pmb':[-72.4,43,-68.6,44.3]}]},
               {'slug':'new_york','names':['New York'],'ref':'sectional/new_york','type':'sectional','charts':[
                    {'d':'1957-06-01','e':'1957-12-01','ed':'3','pm':'p','pmb':[-77,40,-68.6,44.3]}]},
               {'slug':'los_angeles','names':['Los Angeles'],'ref':'sectional/los_angeles','type':'sectional','charts':[
                    {'d':'1950-01-01','e':'1950-06-01','ed':'1','pm':'p','pmb':[-122,32,-114.5,36.1]}]},
               {'slug':'boston_tac','names':['Boston TAC'],'ref':None,'type':'tac','charts':[{'d':'1980-01-01','e':'1980-07-01','ed':'1'}]}]
        warnings=[]; charts=airport_charts(bos,index,{'extents/boston_ma':BOSTON_RING},warnings.append)
        self.assertEqual([(c['slug'],c['d'],c['tiled']) for c in charts],
                         [('boston_ma','1933-03-01',True),('boston_ma','1939-10-01',True),('boston_ma','1957-06-01',False),('new_york','1957-06-01',True)],
                         'sorted by date then slug; the 1960 artifact box misses Logan; untiled 1957 placed by ring')
        self.assertEqual(charts[1]['og'],'sectionals/chart/boston_ma/1939-10-01.jpg'); self.assertNotIn('og',charts[0])
        self.assertEqual((charts[0]['name'],charts[0]['type'],charts[0]['e'],charts[0]['ed']),('Boston, MA','sectional','1933-08-01','1'))
        self.assertEqual(warnings,['KBOS: 1 unplaceable editions (no pmb, no ring) in [\'boston_tac\']'])
        self.assertEqual(decades(charts),[{'decade':1930,'editions':2,'tiled':2,'first':'1933-03-01','last':'1939-10-01'},
                                          {'decade':1950,'editions':2,'tiled':1,'first':'1957-06-01','last':'1957-06-01'}])
        rec=airport_record(bos,charts)
        self.assertEqual((rec['first'],rec['last'],rec['editions'],rec['tiled']),('1933-03-01','1957-06-01',4,3))
        empty=airport_record(bos,[]); self.assertEqual((empty['first'],empty['last'],empty['editions'],empty['decades']),(None,None,0,[]))
    def test_build_hash_folder_index_summary_and_sitemap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); (tmp/'airports.csv').write_text(CSV_HEAD+CSV_ROWS)
            charts=tmp/'charts.abc'; charts.mkdir()
            (charts/'boston_ma.json').write_text(json.dumps({'slug':'boston_ma','names':['Boston, MA'],'ref':'extents/boston_ma','type':'sectional','bbox':[-72,41,-69,44],'am':False,'charts':[
                {'d':'1933-03-01','e':'1933-08-01','ed':'1','pm':'sectionals/chart/boston_ma/1933-03-01.aaaaaaaaaaaa','pmz':[0,11],'pmb':[-72.4,40.4,-68.6,44.3]},
                {'d':'1957-06-01','e':'1957-11-21','ed':'84'}]}))
            (charts/'western_aleutian_islands.json').write_text(json.dumps({'slug':'western_aleutian_islands','names':['Western Aleutian Islands'],'ref':'sectional/wai','type':'sectional','bbox':None,'am':True,'charts':[
                {'d':'2024-12-26','e':'2025-02-20','ed':'Unknown'}]}))
            (tmp/'timeline.json').write_text(json.dumps({'rings':{'extents/boston_ma':BOSTON_RING,'sectional/wai':AM_RING}}))
            (tmp/'idents.txt').write_text('# pilot\nKBOS\nPHNL # Honolulu\nPADK\n'); self.assertEqual(read_idents(tmp/'idents.txt'),['KBOS','PHNL','PADK'])
            warnings=[]; result=build(tmp/'airports.csv',charts,tmp/'timeline.json',tmp/'out',['KBOS','PHNL','PADK'],'https://archive.aero',warnings.append)
            self.assertRegex(result['airports']['base'],r'^next/airports\.[0-9a-f]{12}/$'); self.assertEqual(result['airports']['count'],3)
            self.assertEqual(result['sitemap'],{'file':'sitemap-airports.xml','urls':3}); self.assertEqual(warnings,['PHNL: no edition covers it'])
            folder=tmp/'out'/result['airports']['base'][len('next/'):]
            self.assertEqual(sorted(p.name for p in folder.iterdir()),['index.json','kbos.json','padk.json','phnl.json'])
            bos=json.loads((folder/'kbos.json').read_text())
            self.assertEqual({k:bos[k] for k in ('ident','icao','iata','name','city','region','lat','lon','elev_ft','first','last','editions','tiled')},
                             {'ident':'kbos','icao':'KBOS','iata':'BOS','name':'Boston Logan International Airport','city':'Boston','region':'US-MA','lat':42.362,'lon':-71.0079,'elev_ft':20,
                              'first':'1933-03-01','last':'1957-06-01','editions':2,'tiled':1})
            self.assertEqual([c['tiled'] for c in bos['charts']],[True,False]); self.assertEqual(len(bos['decades']),2)
            self.assertEqual(json.loads((folder/'padk.json').read_text())['charts'][0]['slug'],'western_aleutian_islands','placed by the antimeridian ring')
            idx=json.loads((folder/'index.json').read_text()); self.assertEqual([r['ident'] for r in idx],['kbos','padk','phnl'])
            self.assertEqual(idx[2]['editions'],0); self.assertNotIn('charts',idx[0]); self.assertNotIn('decades',idx[0])
            self.assertEqual(json.loads((tmp/'out'/'airports.json').read_text()),result)
            self.assertEqual((folder/'kbos.json').read_bytes(),json.dumps(bos,separators=(',',':'),ensure_ascii=False).encode(),'compact JSON')
            xml=(tmp/'out'/'sitemap-airports.xml').read_text(); self.assertEqual(xml.count('<url>'),3); self.assertIn('<loc>https://archive.aero/airports/kbos</loc>',xml)
            self.assertTrue(xml.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'))
            # Same inputs, same hash; a changed input, a new folder beside the old one.
            again=build(tmp/'airports.csv',charts,tmp/'timeline.json',tmp/'out',['KBOS','PHNL','PADK'],'https://archive.aero',lambda m:None); self.assertEqual(again,result)
            other=build(tmp/'airports.csv',charts,tmp/'timeline.json',tmp/'out',['KBOS'],'https://archive.aero',lambda m:None); self.assertNotEqual(other['airports']['base'],result['airports']['base'])
            self.assertRaises(ValueError,build,tmp/'airports.csv',charts,tmp/'timeline.json',tmp/'out',['KBOS','KZZZ'])
            self.assertRaises(ValueError,build,tmp/'airports.csv',charts,tmp/'timeline.json',tmp/'out',['KBOS','kbos'])
            self.assertRaises(ValueError,sitemap_xml,[],'http://archive.aero')
        self.assertEqual(len(PILOT),50); self.assertEqual(len(set(PILOT)),50)


if __name__=='__main__':unittest.main()
