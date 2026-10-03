import gzip
import hashlib
import io
import json
import os
import random
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from PIL import Image
from next_pmtiles import Archive, id_to_zxy, write_archive, zxy_to_id, decompress
from next_add_overviews import add_overviews
import next_version_archives
from next_version_archives import plan, execute, shell_commands, bounds
from next_build_overlays import airfields, pins, shard_indices, build as build_overlays
from next_build_manifest import build, emit, estimate


def webp(color):
    out=io.BytesIO();Image.new('RGBA',(32,32),color).save(out,'WEBP',lossless=True);return out.getvalue()


class FakeS3:
    """The slice of a boto3 S3 client that execute() touches; boto3 itself is never imported."""
    class exceptions:
        class ClientError(Exception):
            def __init__(self,code):super().__init__(code);self.response={'Error':{'Code':code}}
    def __init__(self,objects):self.objects=objects;self.copies=[];self.uploads=[]
    def head_object(self,Bucket,Key):
        if Key not in self.objects:raise self.exceptions.ClientError('404')
        return self.objects[Key]
    def copy(self,source,bucket,key,ExtraArgs,Config):self.copies.append((source['Key'],key,ExtraArgs['CopySourceIfMatch']))
    def upload_file(self,filename,bucket,key,Config,ExtraArgs):self.uploads.append((filename,key,ExtraArgs))


class NextDataTests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.dir=Path(self.tmp.name)
    def archive(self,name='original.pmtiles',tiles=None,**options):
        path=self.dir/name
        h={'tile_type':4,'tile_compression':1,'min_lon_e7':-1020000000,'min_lat_e7':300000000,
           'max_lon_e7':-780000000,'max_lat_e7':420000000,**options}
        tiles=tiles or [(zxy_to_id(8,x,y),webp(color)) for x,y,color in
            [(60,100,(255,0,0,255)),(61,100,(0,255,0,255)),(60,101,(0,0,255,255)),(61,101,(255,255,255,255))]]
        write_archive(path,sorted(tiles),h,{'fixture':True});return path
    def test_hilbert_roundtrip(self):
        for z in range(8):
            for x in range(2**z):
                for y in range(2**z):self.assertEqual(id_to_zxy(zxy_to_id(z,x,y)),(z,x,y))
    def test_overviews_original_bytes_and_known_pixels(self):
        source=self.archive();output=self.dir/'overview.pmtiles';stats=add_overviews(source,output)
        a,b=Archive(source),Archive(output);original=dict(a.tiles());new=dict(b.tiles())
        for tid,v in original.items():self.assertEqual(a.read(*v),b.read(*new[tid]))
        self.assertEqual(b.h['min_zoom'],4);self.assertEqual(b.h['max_zoom'],8);self.assertEqual(b.metadata(),a.metadata())
        for z in range(4,8):self.assertTrue(any(id_to_zxy(tid)[0]==z for tid in new))
        img=Image.open(io.BytesIO(b.read(*new[zxy_to_id(7,30,50)]))).convert('RGBA')
        for xy,expected in [((8,8),(255,0,0)),((24,8),(0,255,0)),((8,24),(0,0,255)),((24,24),(255,255,255))]:
            actual=img.getpixel(xy);self.assertGreater(actual[3],250)
            self.assertTrue(all(abs(a-e)<15 for a,e in zip(actual[:3],expected)),(actual,expected))
        self.assertGreater(stats['new_tiles'],0);self.assertGreater(stats['seconds'],0)
    def test_missing_children_transparent_and_antimeridian(self):
        source=self.archive(tiles=[(zxy_to_id(8,0,100),webp((255,0,0,255))),
                                   (zxy_to_id(8,255,100),webp((0,255,0,255)))],min_lon_e7=1700000000,max_lon_e7=-1700000000)
        output=self.dir/'overview.pmtiles';add_overviews(source,output);b=Archive(output);tiles=dict(b.tiles())
        for z in range(4,8):
            xs={id_to_zxy(tid)[1] for tid in tiles if id_to_zxy(tid)[0]==z};self.assertEqual(xs,{0,2**z-1})
        img=Image.open(io.BytesIO(b.read(*tiles[zxy_to_id(7,0,50)]))).convert('RGBA')
        self.assertEqual(img.getpixel((25,25))[3],0);self.assertGreater(img.getpixel((5,5))[3],250)
    def test_gzip_overview_payloads_preserve_original_encoding(self):
        raw=gzip.compress(webp((220,30,60,255)),mtime=0)
        source=self.archive(tiles=[(zxy_to_id(8,60,100),raw)],tile_compression=2)
        out=self.dir/'gzip-overview.pmtiles';add_overviews(source,out)
        archive=Archive(out);tiles=dict(archive.tiles())
        self.assertEqual(archive.h['tile_compression'],2)
        self.assertEqual(archive.read(*tiles[zxy_to_id(8,60,100)]),raw)
        new=archive.read(*tiles[zxy_to_id(7,30,50)])
        self.assertTrue(new.startswith(b'\x1f\x8b'))
        self.assertEqual(Image.open(io.BytesIO(gzip.decompress(new))).format,'WEBP')
    def test_manifest_reads_directories_but_no_payloads(self):
        source=self.archive('sectionals/1950-01-01_to_1951-01-01.0123456789ab.pmtiles')
        tile_offset=Archive(source).h['tile_data_offset'];read=Archive.read
        def checked(archive,offset,length):
            self.assertLess(offset,tile_offset)
            self.assertLessEqual(offset+length,tile_offset)
            return read(archive,offset,length)
        with unittest.mock.patch.object(Archive,'read',checked):
            self.assertEqual(build(self.dir,{})['eras'][0]['c'],[25*64+15])
    def test_dedup_and_leaf_directories(self):
        rng=random.Random(1);tiles=[];tid=0
        for i in range(15000):
            tid+=rng.randrange(1,1000);tiles.append((tid,rng.randbytes(rng.randrange(80,500))))
        source=self.archive(tiles=tiles)
        a=Archive(source);self.assertEqual(len(list(a.tiles())),15000)
        self.assertGreater(a.h['leaf_directory_length'],0)
        dedup=self.archive('dedup.pmtiles',[(i,webp((1,2,3,255))) for i in range(100)])
        self.assertEqual(Archive(dedup).h['tile_contents_count'],1)
    def test_versioning_full_sha_and_chart_paths(self):
        self.archive('era.pmtiles');chart_file=self.archive('chart/test/1950-01-01.pmtiles')
        records=plan(self.dir);self.assertEqual(len(records),2)
        chart=next(r for r in records if '/chart/' in r['old']);self.assertTrue(chart['path'].startswith('sectionals/chart/test/1950-01-01.'))
        self.assertEqual(len(chart['sha256']),64);self.assertTrue(chart['new'].endswith('.pmtiles'));self.assertEqual(chart['z'],[8,8])
        # Scanned files are uploads: their bytes may differ from whatever R2 holds under `old`.
        self.assertEqual((chart['source'],chart['local'],chart['size']),('local',str(chart_file.resolve()),chart_file.stat().st_size));self.assertNotIn('md5',chart)
        sh=shell_commands(records,'charts');self.assertTrue(sh.startswith('#!/bin/sh\nset -eu\n'));self.assertNotIn('rclone size',sh);self.assertNotIn('rclone copyto r2:',sh)
        self.assertIn('\nrclone copyto '+shlex.quote(chart['local'])+' r2:charts/'+chart['new']+' --immutable\n',sh)
        listing=self.dir/'listing.json';listing.write_text(json.dumps([{'key':'sectionals/chart/test/1950-01-01','sha256':chart['sha256'],'z':[8,11],'b':None}]))
        [trusted]=plan(listing=listing);self.assertEqual((trusted['path'],trusted['source']),(chart['path'],'remote'));self.assertNotIn('size',trusted)
        self.assertEqual(shell_commands([trusted],'charts'),'#!/bin/sh\nset -eu\nrclone copyto r2:charts/sectionals/chart/test/1950-01-01 r2:charts/'+trusted['new']+' --immutable\n')
        listing.write_text(json.dumps([{'key':'sectionals/a.pmtiles','sha256':'etag'}]));self.assertRaises(ValueError,plan,listing=listing)
        with unittest.mock.patch.dict(os.environ,{},clear=True):self.assertRaises(ValueError,execute,records,'charts')
    def test_versioning_listing_mirror_guards_and_prefix(self):
        era=self.archive('era.pmtiles');size=era.stat().st_size;md5=hashlib.md5(era.read_bytes()).hexdigest()
        listing=self.dir/'listing.json'
        listing.write_text(json.dumps([{'Path':'sectionals/era.pmtiles','Size':size,'IsDir':False},{'Path':'basemap/x.pmtiles','Size':1},{'Path':'sectionals/timeline_data.json','Size':5},{'Path':'sectionals/chart/boston','Size':-1,'IsDir':True}]))
        [r]=plan(self.dir,listing)  # --prefix drops the basemap object; the non-archive and the rclone directory entry are skipped
        self.assertEqual((r['old'],r['source'],r['size'],r['md5'],r['local']),('sectionals/era.pmtiles','remote',size,md5,str(era.resolve())))
        sh=shell_commands([r],'charts');self.assertIn('rclone size --json "$1"',sh);self.assertIn('exit 1;',sh);self.assertNotIn(str(self.dir),sh)
        self.assertIn(f'\nexpect_size r2:charts/sectionals/era.pmtiles {size}\nrclone copyto r2:charts/sectionals/era.pmtiles r2:charts/{r["new"]} --immutable\n',sh)
        self.assertRaises(ValueError,plan,self.dir,listing,'basemap')  # under that prefix the basemap object is planned, and has no mirror file
        # A listing size the mirror does not match means a stale mirror: refuse before hashing anything.
        listing.write_text(json.dumps([{'Path':'sectionals/era.pmtiles','Size':size+1}]));self.assertRaises(ValueError,plan,self.dir,listing)
        listing.write_text(json.dumps([{'Path':'sectionals/missing.pmtiles'}]));self.assertRaises(ValueError,plan,self.dir,listing)
    def test_execute_uploads_local_and_guards_remote_sources(self):
        era=self.archive('era.pmtiles');[local]=plan(self.dir)
        listing=self.dir/'listing.json';listing.write_text(json.dumps([{'Path':'sectionals/era.pmtiles'}]));[remote]=plan(self.dir,listing)
        def run(records,objects):
            client=FakeS3(objects)
            with unittest.mock.patch.object(next_version_archives,'remote_client',lambda:client),unittest.mock.patch.object(next_version_archives,'transfer_config',lambda:'cfg'):
                execute(records,'charts')
            return client
        client=run([local],{});self.assertEqual(client.copies,[])
        self.assertEqual(client.uploads,[(local['local'],local['new'],{'ContentType':'application/vnd.pmtiles','Metadata':{'sha256':local['sha256']}})])
        self.assertEqual(run([local],{local['new']:{'Metadata':{'sha256':local['sha256']}}}).uploads,[])
        self.assertRaises(ValueError,run,[local],{local['new']:{'Metadata':{}}})
        good={'ETag':'"'+remote['md5']+'"','ContentLength':remote['size'],'Metadata':{}}
        client=run([remote],{remote['old']:good});self.assertEqual((client.copies,client.uploads),([(remote['old'],remote['new'],good['ETag'])],[]))
        self.assertRaises(ValueError,run,[remote],{remote['old']:{**good,'ContentLength':remote['size']+1}})
        self.assertRaises(ValueError,run,[remote],{remote['old']:{**good,'ETag':'"'+'0'*32+'"'}})
        multipart={**good,'ETag':'"'+'0'*32+'-3"'};self.assertEqual(run([remote],{remote['old']:multipart}).copies,[(remote['old'],remote['new'],multipart['ETag'])])
        era.write_bytes(era.read_bytes()+b'\0');self.assertRaises(ValueError,run,[local],{})  # changed since the plan hashed it
    def test_bounds_round_outward_in_plan_and_manifest(self):
        h={'min_lon_e7':-981234321,'min_lat_e7':300000900,'max_lon_e7':-950000900,'max_lat_e7':419999100}
        self.assertEqual(bounds(h),[-98.1235,30,-95,42])  # nearest would give [-98.1234,30.0001,-95.0001,41.9999]
        self.assertIsNone(bounds({**h,'min_lon_e7':1700000000}));self.assertEqual(bounds({**h,'max_lat_e7':860000000})[3],86)
        self.archive('sectionals/1950-01-01_to_1951-01-01.0123456789ab.pmtiles',**h)
        self.assertEqual(build(self.dir,{},generated='2026-10-03T00:00:00Z')['eras'][0]['b'],[-98.1235,30,-95,42])
        self.archive('era.pmtiles',**h);self.assertEqual(next(r for r in plan(self.dir) if r['old']=='sectionals/era.pmtiles')['b'],[-98.1235,30,-95,42])
    def test_airfield_layout_alignment_details_and_year_rules(self):
        features=[]
        for i in range(21):features.append({'geometry':{'type':'Point','coordinates':[-98,35]},'properties':{'name':str(i),'status':('open','gone','unknown')[i%3], 'start_year':1940 if i%2 else None,'end_year':1970 if i%3 else None,'last_known_year':1988,'url':'https://example.com','oa':'FIX'}})
        data,details=airfields({'features':features});self.assertEqual(data[:8],b'AAAF1\0\0\0');self.assertEqual(struct.unpack_from('<II',data,8),(21,0))
        start=16+8*21;end=(start+2*21+3)&~3;status=(end+2*21+3)&~3
        self.assertEqual(struct.unpack_from('<H',data,start+2), (1940,));self.assertEqual(data[status:status+3],bytes([0,1,2]))
        self.assertEqual(data[start+42:end],b'\0'*2);self.assertEqual(details[0]['last_known_year'],1988);self.assertEqual(details[0]['url'],'https://example.com')
        def visible(index,year,mask=7):
            s=struct.unpack_from('<H',data,start+index*2)[0];e=struct.unpack_from('<H',data,end+index*2)[0];st=data[status+index]
            if not mask&(1<<st):return False
            if year is None or (s==0 and e==0):return True
            if s and year<s:return False
            if st==0:return True
            if e and year>e:return False
            return True
        self.assertFalse(visible(1,1939));self.assertFalse(visible(1,1971));self.assertTrue(visible(3,2026));self.assertTrue(visible(1,None));self.assertFalse(visible(1,None,1))
    def test_pin_margin_antimeridian_and_versioned_artifact(self):
        rings=[[[179,50],[-179,50],[-179,51],[179,51],[179,50]]]
        ids=shard_indices(rings);xs={i%32 for i in ids};self.assertEqual(xs,{0,31})
        self.assertEqual(set(shard_indices([[[0,0],[0,0]]],margin=0)),{15*32+15,15*32+16,16*32+15,16*32+16})
        td={'locations':{'Aleutians':{'era':'old','ref':'am','charts':[{'d':'1950-01-01','e':'1951-01-01','pm':['chart/test/1950-01-01']}]}},'rings':{'am':rings}}
        shards=pins(td,[{'old':'sectionals/chart/test/1950-01-01','path':'sectionals/chart/test/1950-01-01.0123456789ab','z':[8,11],'b':None}])
        chart=next(iter(shards.values()))['locations']['Aleutians']['charts'][0]
        self.assertEqual(chart['pm'],['sectionals/chart/test/1950-01-01.0123456789ab']);self.assertEqual(chart['pmz'],[8,11]);self.assertEqual(chart['pmb'],[-180,-85.0512,180,85.0512])
        self.assertRaises(ValueError,pins,td,[])
    def test_manifest_real_directory_coverage_sorted_bounds_schema_budget(self):
        for k in ['1950-01-01_to_1951-01-01','1940-01-01_to_1941-01-01']:
            self.archive('sectionals/'+k+'.0123456789ab.pmtiles')
        self.archive('sectionals/1930-01-01.0123456789ab.pmtiles')
        coverage={'segments':[['1940-01-01','1941-01-01',1,25]],'note':'unchanged'}
        m=build(self.dir,coverage,generated='2026-10-03T00:00:00Z');self.assertEqual(m['coverage'],coverage)
        self.assertEqual(len(m['eras']),2)
        self.assertTrue(m['eras'][0]['k'].startswith('1940'));self.assertEqual(m['eras'][0]['c'],[25*64+15]);self.assertEqual(m['eras'][0]['b'],[-102,30,-78,42])
        path=emit(m,self.dir/'out');self.assertTrue(path.exists());self.assertEqual(json.loads(path.read_text()),m)
        m['eras'][0]['c']=[4096];self.assertRaises(subprocess.CalledProcessError,emit,m,self.dir/'bad')
        m['eras'][0]['c']=[25*64+15]
        m['coverage']={'note':os.urandom(100000).hex()};self.assertRaises(ValueError,emit,m,self.dir/'huge')
        self.assertTrue(emit(m,self.dir/'explicit',True).exists())
    def test_estimate_production_is_over_budget(self):
        stats=estimate(ROOT/'dates.csv');self.assertEqual(stats['eras'],3761);self.assertTrue(stats['over_budget'])


if __name__=='__main__':unittest.main()
