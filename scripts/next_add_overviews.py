#!/usr/bin/env python3
"""Add WebP quality-80 z7..4 overviews, preserving every existing tile byte."""
import argparse
import gzip
import io
import time
from pathlib import Path
from PIL import Image
from next_pmtiles import Archive, decompress, id_to_zxy, write_archive, zxy_to_id


def add_overviews(source, output):
    start = time.monotonic(); archive = Archive(source); h = archive.h
    if Path(source).resolve() == Path(output).resolve(): raise ValueError('output must be a new archive')
    if h['tile_type'] != 4: raise ValueError('overviews require a WebP archive (one tile type per PMTiles)')
    if h['tile_compression'] not in (1, 2): raise ValueError('unsupported tile compression')
    original = dict(archive.tiles()); generated = {}; level = {}
    for tid, (off, length) in original.items():
        z, x, y = id_to_zxy(tid)
        if z == 8: level[x, y] = (off, length)
    if not level: raise ValueError('z8 source tiles are required')
    size = None
    for z in range(7, 3, -1):
        next_level = {}
        for px, py in sorted({(x//2, y//2) for x, y in level}):
            children = []
            for dy in range(2):
                for dx in range(2):
                    val = level.get((px*2+dx, py*2+dy))
                    if val is None: continue
                    raw = archive.read(*val) if isinstance(val, tuple) else val
                    img = Image.open(io.BytesIO(decompress(raw, h['tile_compression']))).convert('RGBA')
                    if size is None: size = img.width
                    if img.size != (size, size): raise ValueError('inconsistent tile dimensions')
                    children.append((dx, dy, img))
            canvas = Image.new('RGBA', (size*2, size*2))
            for dx, dy, img in children: canvas.paste(img, (dx*size, dy*size))
            reduced = canvas.resize((size, size), Image.Resampling.LANCZOS)
            out = io.BytesIO(); reduced.save(out, 'WEBP', quality=80)
            raw = out.getvalue()
            if h['tile_compression'] == 2: raw = gzip.compress(raw, mtime=0)
            tid = zxy_to_id(z, px, py)
            # Existing lower zooms also remain byte-identical and feed the next level.
            if tid in original: val = original[tid]
            else: generated[tid] = raw; val = raw
            next_level[px, py] = val
        level = next_level
    def tiles():
        for tid in sorted(original.keys() | generated.keys()):
            yield tid, archive.read(*original[tid]) if tid in original else generated[tid]
    write_archive(output, tiles(), h, archive.metadata())
    result = {'original_bytes': archive.size, 'output_bytes': Path(output).stat().st_size,
              'new_tiles': len(generated), 'seconds': round(time.monotonic()-start, 3)}
    result['overhead_bytes'] = result['output_bytes']-result['original_bytes']
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('source'); ap.add_argument('output')
    args = ap.parse_args(); print(add_overviews(args.source, args.output))

if __name__ == '__main__': main()
