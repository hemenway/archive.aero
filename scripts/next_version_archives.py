#!/usr/bin/env python3
"""Plan content-versioned archive keys ({stem}.{sha256[:12]}.pmtiles). No writes unless --execute.

Two input modes, told apart by each record's `source`:
  --listing L.json [--dir MIRROR]  "remote": the records are objects that exist in R2 under `old`
      and the plan is a server-side copy. The mirror (or --read-remote) only supplies the hash, so it
      must hold R2's current bytes: a mirror file whose size differs from the listing's Size fails the
      plan, and `size`/`md5` of the hashed file are recorded so the .sh (rclone size) and --execute
      (ContentLength, single-part ETag) refuse a source that no longer matches.
  --dir D --prefix P (no listing)  "local": the records are files scanned under D whose bytes may
      differ from whatever R2 holds under `old` (overview builds reuse the legacy era names), so the
      plan is an upload of `local`, never a server-side copy.
`local` is the absolute path of the file that was hashed. Listing JSON is an array of
{key|Key|Path, optional Size, sha256, local, z, b}; keys outside --prefix are skipped. A complete
SHA256 is required: multipart ETags are not content hashes. Extension-less chart keys can use
local=<relative .pmtiles filename> in the listing.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from next_pmtiles import Archive, header


def digest_file(path, with_md5=False):
    """(sha256 hex, md5 hex or None) in one pass; md5 is what a single-part R2 ETag holds."""
    h = hashlib.sha256(); m = hashlib.md5() if with_md5 else None
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
            if m: m.update(chunk)
    return h.hexdigest(), (m.hexdigest() if m else None)


def bounds(h):
    """[w, s, e, n] rounded OUTWARD to 4 decimals (floor west/south, ceil east/north) so culling
    never trims an edge tile; None (never cull) for antimeridian-wrapping or implausible headers."""
    w,s,e,n = [h[k]/1e7 for k in ('min_lon_e7','min_lat_e7','max_lon_e7','max_lat_e7')]
    if not (-180 <= w < e <= 180 and -86 <= s < n <= 86): return None
    return [h['min_lon_e7']//1000/1e4, h['min_lat_e7']//1000/1e4, -(-h['max_lon_e7']//1000)/1e4, -(-h['max_lat_e7']//1000)/1e4]


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
        if record.get('IsDir') or (listing and not old.startswith(prefix+'/')): continue  # rclone dirs; basemap/airspace in a whole-bucket listing
        if not old.endswith('.pmtiles') and '/chart/' not in old: continue
        stem = old.removesuffix('.pmtiles'); stem = re.sub(r'\.[a-f0-9]{12}$', '', stem)
        if old.startswith('t/') or '..' in old.split('/') or old.startswith('/'):
            raise ValueError('unsafe or reserved key')
        size = record.get('size', record.get('Size')); size = size if isinstance(size, int) and size >= 0 else None
        md5 = record.get('md5'); hashed = None
        if directory:
            file = Path(directory) / (local or old)
            if not file.exists() and not local and old.startswith(prefix+'/'):
                file = Path(directory) / old[len(prefix)+1:]
            if not file.exists() and '/chart/' in old and not file.suffix:
                file = file.with_name(file.name+'.pmtiles')
            if not file.resolve().is_relative_to(Path(directory).resolve()): raise ValueError('local path escapes directory')
            if not file.is_file(): raise ValueError(f'{old}: no local file at {file}')
            if size is not None and file.stat().st_size != size:
                raise ValueError(f'{old}: mirror file is {file.stat().st_size} bytes, listing says {size}; refresh the mirror')
            digest, md5 = digest_file(file, bool(listing)); size = file.stat().st_size; hashed = str(file.resolve())
            a = Archive(file); z = [a.h['min_zoom'],a.h['max_zoom']]; b = bounds(a.h)
        elif remote:
            response = remote.get_object(Bucket=bucket, Key=old)
            hasher = hashlib.sha256(); m = hashlib.md5(); first = bytearray()
            try:
                for chunk in response['Body'].iter_chunks(8*1024*1024):
                    hasher.update(chunk); m.update(chunk)
                    if len(first) < 127: first.extend(chunk[:127-len(first)])
            finally: response['Body'].close()
            ah = header(bytes(first)); digest = hasher.hexdigest(); md5 = m.hexdigest(); size = response['ContentLength']
            z = [ah['min_zoom'], ah['max_zoom']]; b = bounds(ah)
        else:
            digest = record.get('sha256',''); z = record.get('z'); b = record.get('b')
        if not re.fullmatch('[a-f0-9]{64}',digest): raise ValueError(f'{old}: complete SHA256 required')
        path = stem+'.'+digest[:12]; new = path+'.pmtiles'
        if not re.fullmatch(r'(?:[A-Za-z0-9_-]+/)+[A-Za-z0-9_.-]+\.[a-f0-9]{12}', path):
            raise ValueError('archive path does not satisfy C1: '+path)
        if new in seen: raise ValueError(f'duplicate destination {new}')
        seen.add(new); out = {'old':old,'new':new,'path':path,'sha256':digest,'z':z,'b':b,'source':'remote' if listing else 'local'}
        if size is not None: out['size'] = size
        if md5: out['md5'] = md5
        if hashed: out['local'] = hashed
        result.append(out)
    return result


def shell_commands(records, bucket):
    """POSIX sh for a plan: uploads for local records, size-guarded server-side copies for remote ones."""
    lines = []
    for r in records:
        if r['old'] == r['new']: continue
        new = shlex.quote(f'r2:{bucket}/{r["new"]}')
        if r.get('source','remote') == 'local':
            lines.append(f'rclone copyto {shlex.quote(r["local"])} {new} --immutable'); continue
        old = shlex.quote(f'r2:{bucket}/{r["old"]}')
        if r.get('size') is not None: lines.append(f'expect_size {old} {r["size"]}')
        # rclone copyto uses server-side copy on the same remote, including multipart copies.
        lines.append(f'rclone copyto {old} {new} --immutable')
    head = ['#!/bin/sh', 'set -eu']
    if any(line.startswith('expect_size ') for line in lines):
        head += ['# expect_size KEY BYTES: refuse to copy an R2 object that is not the size the plan hashed.',
                 'expect_size() {',
                 '  n=$(rclone size --json "$1" | sed -n \'s/.*"bytes":\\([0-9][0-9]*\\).*/\\1/p\')',
                 '  [ "$n" = "$2" ] || { echo "$1: R2 holds ${n:-?} bytes, plan hashed $2 (stale mirror or republished key)" >&2; exit 1; }',
                 '}']
    return '\n'.join(head+lines)+'\n'


def remote_client():
    # Explicit invocation plus environment credentials. Never use a default profile.
    required = ('R2_ENDPOINT_URL','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY')
    if any(not os.environ.get(k) for k in required): raise ValueError('execution needs '+', '.join(required))
    import boto3
    client = boto3.client('s3',endpoint_url=os.environ['R2_ENDPOINT_URL'],
                         aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                         aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    return client


def transfer_config():
    # Managed transfers switch to multipart above 512 MiB: CopyObject caps at 5 GiB, uploads stream.
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(multipart_threshold=512*1024**2, multipart_chunksize=128*1024**2)


def execute(records, bucket):
    client = remote_client(); config = transfer_config()
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
        if r.get('source','remote') == 'local':
            if r.get('size') is not None and os.path.getsize(r['local']) != r['size']:
                raise ValueError(f"{r['local']}: {os.path.getsize(r['local'])} bytes now, plan hashed {r['size']}")
            client.upload_file(r['local'], bucket, r['new'], Config=config,
                               ExtraArgs={'ContentType':'application/vnd.pmtiles','Metadata':{'sha256':r['sha256']}})
            continue
        source = client.head_object(Bucket=bucket, Key=r['old']); etag = source['ETag'].strip('"')
        if r.get('size') is not None and source['ContentLength'] != r['size']:
            raise ValueError(f"{r['old']}: R2 object is {source['ContentLength']} bytes, plan hashed {r['size']}")
        if r.get('md5') and '-' not in etag and etag != r['md5']:
            raise ValueError(f"{r['old']}: R2 ETag {etag} is not the mirror md5 {r['md5']}")
        client.copy({'Bucket':bucket,'Key':r['old']}, bucket, r['new'],
                    ExtraArgs={'CopySourceIfMatch':source['ETag'], 'MetadataDirective':'REPLACE',
                               'Metadata':{**source.get('Metadata', {}), 'sha256':r['sha256']},
                               'ContentType':source.get('ContentType','application/vnd.pmtiles')},
                    Config=config)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir',help='local directory: scanned for uploads without --listing, or the mirror that hashes a listing')
    ap.add_argument('--listing',help='R2 inventory (rclone lsjson or S3 Contents): plan server-side copies of these objects')
    ap.add_argument('--prefix',default='sectionals',help='bucket prefix: prepended to scanned files; listing keys outside it are skipped')
    ap.add_argument('--read-remote',action='store_true',help='explicitly stream listing objects to hash them; needs environment credentials')
    ap.add_argument('--out',required=True); ap.add_argument('--bucket',default='charts'); ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    if not args.dir and not args.listing: ap.error('need --dir and/or --listing')
    records=plan(args.dir,args.listing,args.prefix,args.read_remote,args.bucket)
    Path(args.out).write_text(json.dumps(records,indent=2)+'\n')
    Path(args.out+'.sh').write_text(shell_commands(records,args.bucket))
    uploads=sum(r['source']=='local' for r in records)
    print(f'{len(records)} archives ({uploads} uploads, {len(records)-uploads} server-side copies); wrote {args.out} and {args.out}.sh')
    if args.execute: execute(records,args.bucket)

if __name__ == '__main__': main()
