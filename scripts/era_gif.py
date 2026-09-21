#!/usr/bin/env python3
"""Animate one spot through time from the published era PMTiles.

  ~/venv/bin/python scripts/era_gif.py LAT LON ZOOM START END STEPS [out.gif]
  e.g.  ... 32.78 -96.80 9 1940-01-01 2026-01-01 40 dallas.gif

Each frame is the 3x3 tile block around the point, compositing every era whose
[start, end) covers the date and whose bounds touch the block, oldest first
(same rule as index.html). Era bounds + headers/directories come from the
metadata bundle index.html points at, so only tile bodies hit the network.
"""
import gzip, io, json, math, re, struct, sys, datetime as dt, urllib.request
from PIL import Image, ImageDraw
import pmtiles.reader as pr
from viewer_config import viewer_config_source

lat, lon, z = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
d0, d1 = (dt.date.fromisoformat(s) for s in sys.argv[4:6])
steps, out = int(sys.argv[6]), (sys.argv[7] if len(sys.argv) > 7 else "era.gif")

def http(url, off=None, n=None):
    h = {"User-Agent": "era_gif"}
    if off is not None: h["Range"] = f"bytes={off}-{off+n-1}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h)).read()

bundle_url = re.search(r"bundleUrl:\s*'([^']+)'", viewer_config_source("index.html").read_text()).group(1)
bundle = http(bundle_url)
glen = struct.unpack("<I", bundle[8:12])[0]
index = json.loads(gzip.decompress(bundle[16:16 + glen]))
blobs, base = 16 + glen, index["baseUrl"]
eras = sorted(index["eras"], key=lambda e: e["k"])

n = 2 ** z
xc = (lon + 180) / 360 * n
yc = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
x0, y0 = int(xc) - 1, int(yc) - 1
def lon_of(x): return x / n * 360 - 180
def lat_of(y): return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
box = (lon_of(x0), lat_of(y0 + 3), lon_of(x0 + 3), lat_of(y0))  # w, s, e, n

def reader(era):
    url, off, ln = f"{base}{era['k']}.pmtiles", blobs + era["off"], era["len"]
    def get(o, k):
        if o + k <= ln: return bundle[off + o: off + o + k]   # header/dirs from bundle
        return http(url, o, k)
    return pr.Reader(get)

readers, tiles = {}, {}
def tile(era, x, y):
    key = (era["k"], x, y)
    if key not in tiles:
        rd = readers.setdefault(era["k"], reader(era))
        zz = min(z, rd.header()["max_zoom"]); s = 2 ** (z - zz)
        raw, im = rd.get(zz, x // s, y // s), None
        if raw:
            im = Image.open(io.BytesIO(raw)).convert("RGBA")
            if s > 1:  # overzoom: crop the parent tile's quadrant and upscale
                w = 256 // s; cx, cy = (x % s) * w, (y % s) * w
                im = im.crop((cx, cy, cx + w, cy + w)).resize((256, 256), Image.BILINEAR)
        tiles[key] = im
    return tiles[key]

frames = []
for i in range(steps):
    day = (d0 + (d1 - d0) * i / max(steps - 1, 1)).isoformat()
    frame = Image.new("RGBA", (768, 768), (20, 20, 20, 255))
    for era in eras:
        s, e = era["k"].split("_to_")
        b = era.get("b")
        if not (s <= day < e): continue
        if b and (b[2] < box[0] or b[0] > box[2] or b[3] < box[1] or b[1] > box[3]): continue
        for dx in range(3):
            for dy in range(3):
                t = tile(era, x0 + dx, y0 + dy)
                if t: frame.alpha_composite(t, (dx * 256, dy * 256))
    ImageDraw.Draw(frame).text((10, 10), day, fill="white")
    frames.append(frame.convert("P", palette=Image.ADAPTIVE)); print(day, flush=True)
frames[0].save(out, save_all=True, append_images=frames[1:], duration=300, loop=0)
print("wrote", out)
