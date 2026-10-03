#!/usr/bin/env python3
"""Plan content-versioned server-side copies. No writes unless --execute.

Listing JSON is an array of {key, sha256, optional local, z, b}. A complete
SHA256 is required: multipart ETags are not content hashes. With a local mirror,
--dir computes hashes and headers; paths mirror the bucket's directory layout.
Extension-less chart keys can use local=<relative .pmtiles filename> in listing.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from next_pmtiles import Archive


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def bounds(h):
    w,s,e,n = [h[k]/1e7 for k in ('min_lon_e7','min_lat_e7','max_lon_e7','max_lat_e7')]
    return [round(v,4) for v in (w,s,e,n)] if -180 <= w < e <= 180 and -86 <= s < n <= 86 else None


def plan(directory=None, listing=None, prefix='sectionals', read_remote=False, bucket='charts'):
    records = json.loads(Path(listing).read_text()) if listing else []
    if isinstance(records, dict): records = records.get('objects', records.get('Contents'))
    if records is None: raise ValueError('listing must contain objects or Contents')
    remote = remote_client() if read_remote else None
    if directory and not listing:
        records = []
        for p in sorted(Path(directory).rglob('*.pmtiles')):
            key = f'{prefix}/{p.relative_to(directory).as_posix()}'
            if '/chart/' in key and not re.search(r'\.[a-f0-9]{12}\.pmtiles$', key):
                key = key.removesuffix('.pmtiles')  # legacy full-sheet keys are extension-less
            records.append({'key':key,'local':p.relative_to(directory).as_posix()})
    result = []; seen = set()
    for record in records:
        old = record.get('key') or record.get('Key') or record.get('Path'); local = record.get('local')
        if not old: raise ValueError('listing entry needs key, Key, or Path')
        if not old.endswith('.pmtiles') and '/chart/' not in old: continue
        stem = old.removesuffix('.pmtiles'); stem = re.sub(r'\.[a-f0-9]{12}$', '', stem)
        if old.startswith('t/') or '..' in old.split('/') or old.startswith('/'):
            raise ValueError('unsafe or reserved key')
        if directory:
            source = Path(directory) / (local or old)
            if not source.exists() and not local and old.startswith(prefix+'/'):
                source = Path(directory) / old[len(prefix)+1:]
            if not source.exists() and '/chart/' in old and not source.suffix:
                source = source.with_name(source.name+'.pmtiles')
            if not source.resolve().is_relative_to(Path(directory).resolve()): raise ValueError('local path escapes directory')
            digest = digest_file(source); a = Archive(source)
            z = [a.h['min_zoom'],a.h['max_zoom']]; b = bounds(a.h)
        elif remote:
            response = remote.get_object(Bucket=bucket, Key=old)
            hasher = hashlib.sha256(); first = bytearray()
            try:
                for chunk in response['Body'].iter_chunks(8*1024*1024):
                    hasher.update(chunk)
                    if len(first) < 127: first.extend(chunk[:127-len(first)])
            finally: response['Body'].close()
            from next_pmtiles import header
            ah = header(bytes(first)); digest = hasher.hexdigest()
            z = [ah['min_zoom'], ah['max_zoom']]; b = bounds(ah)
        else:
            digest = record.get('sha256',''); z = record.get('z'); b = record.get('b')
        if not re.fullmatch('[a-f0-9]{64}',digest): raise ValueError(f'{old}: complete SHA256 required')
        path = stem+'.'+digest[:12]; new = path+'.pmtiles'
        if not re.fullmatch(r'(?:[A-Za-z0-9_-]+/)+[A-Za-z0-9_.-]+\.[a-f0-9]{12}', path):
            raise ValueError('archive path does not satisfy C1: '+path)
        if new in seen: raise ValueError(f'duplicate destination {new}')
        seen.add(new); result.append({'old':old,'new':new,'path':path,'sha256':digest,'z':z,'b':b})
    return result


def remote_client():
    # Explicit invocation plus environment credentials. Never use a default profile.
    required = ('R2_ENDPOINT_URL','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY')
    if any(not os.environ.get(k) for k in required): raise ValueError('execution needs '+', '.join(required))
    import boto3
    client = boto3.client('s3',endpoint_url=os.environ['R2_ENDPOINT_URL'],
                         aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                         aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    return client


def execute(records, bucket):
    client = remote_client()
    from boto3.s3.transfer import TransferConfig
    for r in records:
        if r['old'] == r['new']: continue
        try:
            existing = client.head_object(Bucket=bucket, Key=r['new'])
        except client.exceptions.ClientError as error:
            if str(error.response.get('Error', {}).get('Code')) not in ('404', 'NoSuchKey', 'NotFound'): raise
            existing = None
        if existing is not None:
            if existing.get('Metadata', {}).get('sha256') == r['sha256']: continue
            raise ValueError('destination already exists without matching full SHA256 metadata: '+r['new'])
        source = client.head_object(Bucket=bucket, Key=r['old'])
        # Managed copy switches to multipart for archives above CopyObject's 5GiB limit.
        client.copy({'Bucket':bucket,'Key':r['old']}, bucket, r['new'],
                    ExtraArgs={'CopySourceIfMatch':source['ETag'], 'MetadataDirective':'REPLACE',
                               'Metadata':{**source.get('Metadata', {}), 'sha256':r['sha256']},
                               'ContentType':source.get('ContentType','application/vnd.pmtiles')},
                    Config=TransferConfig(multipart_threshold=512*1024**2,
                                          multipart_chunksize=128*1024**2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir'); ap.add_argument('--listing'); ap.add_argument('--prefix',default='sectionals')
    ap.add_argument('--read-remote',action='store_true',help='explicitly stream listing objects to hash them; needs environment credentials')
    ap.add_argument('--out',required=True); ap.add_argument('--bucket',default='charts'); ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    if not args.dir and not args.listing: ap.error('need --dir and/or --listing')
    records=plan(args.dir,args.listing,args.prefix,args.read_remote,args.bucket)
    Path(args.out).write_text(json.dumps(records,indent=2)+'\n')
    commands=[]
    for r in records:
        if r['old'] != r['new']:
            # rclone copyto uses server-side copy on the same remote, including multipart copies.
            commands.append('rclone copyto '+shlex.quote('r2:'+args.bucket+'/'+r['old'])+' '+
                            shlex.quote('r2:'+args.bucket+'/'+r['new'])+' --immutable')
    Path(args.out+'.sh').write_text('#!/bin/sh\nset -eu\n'+ '\n'.join(commands)+'\n')
    print(f'{len(records)} archives; wrote {args.out} and {args.out}.sh')
    if args.execute: execute(records,args.bucket)

if __name__ == '__main__': main()
