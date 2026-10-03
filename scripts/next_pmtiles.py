"""Small local PMTiles v3 codec. No network I/O; tile payloads remain opaque."""
import gzip
import json
import struct
from pathlib import Path

FIELDS = ('root_offset root_length metadata_offset metadata_length leaf_directory_offset '
          'leaf_directory_length tile_data_offset tile_data_length addressed_tiles_count '
          'tile_entries_count tile_contents_count').split()


def header(data):
    if len(data) != 127 or data[:8] != b'PMTiles\x03':
        raise ValueError('expected a complete PMTiles v3 header')
    h = dict(zip(FIELDS, struct.unpack_from('<11Q', data, 8)))
    h.update(zip(('clustered', 'internal_compression', 'tile_compression', 'tile_type',
                  'min_zoom', 'max_zoom'), data[96:102]))
    h.update(zip(('min_lon_e7', 'min_lat_e7', 'max_lon_e7', 'max_lat_e7'),
                 struct.unpack_from('<4i', data, 102)))
    h.update(center_zoom=data[118], center_lon_e7=struct.unpack_from('<i', data, 119)[0],
             center_lat_e7=struct.unpack_from('<i', data, 123)[0])
    return h


def encode_header(h):
    b = bytearray(127); b[:8] = b'PMTiles\x03'
    struct.pack_into('<11Q', b, 8, *(h.get(k, 0) for k in FIELDS))
    b[96:102] = bytes(h.get(k, 0) for k in ('clustered', 'internal_compression',
                      'tile_compression', 'tile_type', 'min_zoom', 'max_zoom'))
    struct.pack_into('<4i', b, 102, *(h.get(k, v) for k, v in zip(
        ('min_lon_e7', 'min_lat_e7', 'max_lon_e7', 'max_lat_e7'),
        (-1800000000, -850000000, 1800000000, 850000000))))
    b[118] = h.get('center_zoom', h['min_zoom'])
    struct.pack_into('<2i', b, 119, h.get('center_lon_e7', 0), h.get('center_lat_e7', 0))
    return bytes(b)


def decompress(data, compression):
    if compression == 1: return data
    if compression == 2: return gzip.decompress(data)
    raise ValueError(f'unsupported PMTiles compression {compression}')


def varint(n):
    if n < 0: raise ValueError('negative varint')
    out = bytearray()
    while n > 127: out.append((n & 127) | 128); n >>= 7
    out.append(n); return bytes(out)


def directory(data, compression=2):
    data = decompress(data, compression); pos = 0
    def read():
        nonlocal pos
        n = 0
        for shift in range(0, 70, 7):
            if pos >= len(data): raise ValueError('truncated directory')
            b = data[pos]; pos += 1; n |= (b & 127) << shift
            if b < 128: return n
        raise ValueError('oversized varint')
    count = read()
    if count > len(data) // 4: raise ValueError('invalid directory count')
    ids = []; acc = 0
    for _ in range(count): acc += read(); ids.append(acc)
    runs = [read() for _ in ids]; lengths = [read() for _ in ids]; entries = []
    for i, tid in enumerate(ids):
        off = read()
        off = entries[-1][1] + entries[-1][2] if off == 0 and i else off - 1
        if off < 0 or lengths[i] == 0: raise ValueError('invalid directory entry')
        entries.append((tid, off, lengths[i], runs[i]))
    if pos != len(data): raise ValueError('trailing directory bytes')
    return entries


def encode_directory(entries):
    out = bytearray(varint(len(entries))); prev = 0
    for tid, _, _, _ in entries: out += varint(tid-prev); prev = tid
    for e in entries: out += varint(e[3])
    for e in entries: out += varint(e[2])
    for i, e in enumerate(entries):
        out += varint(0 if i and e[1] == entries[i-1][1]+entries[i-1][2] else e[1]+1)
    return gzip.compress(out, mtime=0)


def zxy_to_id(z, x, y):
    if not (0 <= z <= 31 and 0 <= x < 2**z and 0 <= y < 2**z):
        raise ValueError('coordinate outside zoom')
    acc = (4**z-1)//3; s = 2**(z-1) if z else 0
    while s:
        rx, ry = int(bool(x & s)), int(bool(y & s))
        acc += s*s*((3*rx)^ry)
        if not ry:
            if rx: x, y = s-1-x, s-1-y
            x, y = y, x
        s //= 2
    return acc


def id_to_zxy(tid):
    if tid < 0: raise ValueError('negative tile id')
    z = ((3*tid+1).bit_length()-1)//2
    t = tid-(4**z-1)//3; x = y = 0; s = 1
    while s < 2**z:
        rx, ry = (t//2)&1, (t ^ (t//2))&1
        if not ry:
            if rx: x, y = s-1-x, s-1-y
            x, y = y, x
        x += s*rx; y += s*ry; t //= 4; s *= 2
    return z, x, y


class Archive:
    def __init__(self, path):
        self.path = Path(path); self.size = self.path.stat().st_size
        self.h = header(self.read(0, 127))
        for field in ('root', 'metadata', 'leaf_directory', 'tile_data'):
            if self.h[field+'_offset']+self.h[field+'_length'] > self.size:
                raise ValueError(f'{field} outside archive')
    def read(self, off, length):
        with self.path.open('rb') as f: f.seek(off); b = f.read(length)
        if len(b) != length: raise ValueError('short archive read')
        return b
    def metadata(self):
        h = self.h
        return json.loads(decompress(self.read(h['metadata_offset'], h['metadata_length']),
                                     h['internal_compression']))
    def entries(self):
        h = self.h
        def visit(off, length, depth=0):
            if depth > 3: raise ValueError('directory nesting exceeds v3 limit')
            for e in directory(self.read(off, length), h['internal_compression']):
                if e[3]:
                    if e[1]+e[2] > h['tile_data_length']: raise ValueError('tile outside section')
                    yield e
                else:
                    if e[1]+e[2] > h['leaf_directory_length']: raise ValueError('leaf outside section')
                    yield from visit(h['leaf_directory_offset']+e[1], e[2], depth+1)
        yield from visit(h['root_offset'], h['root_length'])
    def tiles(self):
        for tid, off, length, run in self.entries():
            for i in range(run): yield tid+i, (self.h['tile_data_offset']+off, length)
    def coverage(self, zoom=6):
        cells = set()
        for tid, _ in self.tiles():
            z, x, y = id_to_zxy(tid)
            if z >= zoom: cells.add((x >> (z-zoom), y >> (z-zoom)))
        return sorted(x+2**zoom*y for x, y in cells)


def write_archive(path, tiles, h, metadata=None):
    """tiles: Hilbert-sorted (id, bytes) iterator. Spools payloads to disk.

    SHA256 dedup uses content hashes, never Python's randomized hash().
    Directory index and content hash index occupy memory proportional to entries.
    """
    import hashlib
    import shutil
    import tempfile
    h = dict(h); entries = []; seen = {}; total = addressed = 0; previous = -1
    with tempfile.TemporaryFile() as payload:
        for tid, data in tiles:
            if tid <= previous: raise ValueError('tiles must have unique increasing ids')
            if not data: raise ValueError('empty tile payload')
            previous = tid; addressed += 1
            digest = hashlib.sha256(data).digest()
            if digest in seen:
                off, length = seen[digest]
                payload.seek(off)
                if payload.read(length) != data: raise ValueError('SHA256 collision')
            else:
                off = total; payload.seek(total); payload.write(data); total += len(data)
                seen[digest] = (off, len(data))
            if entries and tid == entries[-1][0]+entries[-1][3] and off == entries[-1][1]:
                e = entries[-1]; entries[-1] = (*e[:3], e[3]+1)
            else: entries.append((tid, off, len(data), 1))
        if not entries: raise ValueError('cannot write empty archive')
        root = encode_directory(entries); leaves = b''
        if len(root) > 16384-127:
            leaf_size = 4096
            while True:
                roots = []; chunks = []; offset = 0
                for start in range(0, len(entries), leaf_size):
                    chunk = encode_directory(entries[start:start+leaf_size])
                    roots.append((entries[start][0], offset, len(chunk), 0))
                    chunks.append(chunk); offset += len(chunk)
                root = encode_directory(roots)
                if len(root) <= 16384-127: leaves = b''.join(chunks); break
                leaf_size *= 2
        meta = gzip.compress(json.dumps(metadata or {}, separators=(',', ':')).encode(), mtime=0)
        h.update(root_offset=127, root_length=len(root), metadata_offset=127+len(root),
                 metadata_length=len(meta), leaf_directory_offset=127+len(root)+len(meta),
                 leaf_directory_length=len(leaves), tile_data_offset=127+len(root)+len(meta)+len(leaves),
                 tile_data_length=total, addressed_tiles_count=addressed,
                 tile_entries_count=len(entries), tile_contents_count=len(seen), clustered=1,
                 internal_compression=2, min_zoom=id_to_zxy(entries[0][0])[0],
                 max_zoom=id_to_zxy(previous)[0])
        path = Path(path); part = path.with_name(path.name+'.part')
        path.parent.mkdir(parents=True, exist_ok=True)
        with part.open('wb') as out:
            out.write(encode_header(h)); out.write(root); out.write(meta); out.write(leaves)
            payload.seek(0); shutil.copyfileobj(payload, out)
        part.replace(path)
