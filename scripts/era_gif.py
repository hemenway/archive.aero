#!/usr/bin/env python3
"""Animate one spot through time from the published era PMTiles.

  ~/venv/bin/python scripts/era_gif.py LAT LON ZOOM START END STEPS [out.gif] [--skip-empty]
  e.g.  ... 32.78 -96.80 9 1940-01-01 2026-01-01 40 dallas.gif

Each frame is the 3x3 tile block around the point, compositing every era whose
[start, end) covers the date and whose bounds touch the block, oldest first
(same rule as index.html). Era bounds + headers/directories come from the
metadata bundle index.html points at, so only tile bodies hit the network.
Dates no era covers render as dark frames; --skip-empty drops them instead.
Every frame carries a small date label in its top-left corner.
"""
import gzip, io, json, math, re, struct, sys, datetime as dt, urllib.request
from PIL import Image, ImageChops, ImageDraw, ImageFont
import pmtiles.reader as pr
from viewer_config import viewer_config_source

skip_empty = "--skip-empty" in sys.argv
argv = [a for a in sys.argv[1:] if a != "--skip-empty"]
lat, lon, z = float(argv[0]), float(argv[1]), int(argv[2])
d0, d1 = (dt.date.fromisoformat(s) for s in argv[3:5])
steps, out = int(argv[5]), (argv[6] if len(argv) > 6 else "era.gif")

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

def label_font(size=16):
    """Small readable face: Pillow's bundled scalable default, else a system TTF, else the bitmap default."""
    try: return ImageFont.load_default(size=size)
    except TypeError: pass
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try: return ImageFont.truetype(f, size)
        except OSError: pass
    return ImageFont.load_default()
FONT = label_font()

def stamp(frame, text, pad=5, margin=8):
    """Date label, top-left: white text on a translucent dark box so it reads on any chart."""
    over = Image.new("RGBA", frame.size, (0, 0, 0, 0)); d = ImageDraw.Draw(over)
    l, t, r, b = d.textbbox((margin + pad, margin + pad), text, font=FONT)
    d.rectangle((margin, margin, r + pad, b + pad), fill=(20, 20, 20, 200))
    d.text((margin + pad, margin + pad), text, font=FONT, fill="white")
    frame.alpha_composite(over)

BG = (20, 20, 20, 255)
blank = Image.new("RGB", (768, 768), BG[:3])  # RGB: getbbox() on RGBA would test alpha only
frames = []
for i in range(steps):
    day = (d0 + dt.timedelta(days=round((d1 - d0).days * i / max(steps - 1, 1)))).isoformat()  # nearest day
    frame = Image.new("RGBA", (768, 768), BG)
    for era in eras:
        s, e = era["k"].split("_to_")
        b = era.get("b")
        if not (s <= day < e): continue
        if b and (b[2] < box[0] or b[0] > box[2] or b[3] < box[1] or b[1] > box[3]): continue
        for dx in range(3):
            for dy in range(3):
                t = tile(era, x0 + dx, y0 + dy)
                if t: frame.alpha_composite(t, (dx * 256, dy * 256))
    if skip_empty and ImageChops.difference(frame.convert("RGB"), blank).getbbox() is None:
        print(day, "empty, skipped", flush=True); continue
    stamp(frame, day)
    frames.append(frame.convert("P", palette=Image.ADAPTIVE)); print(day, flush=True)
if not frames: sys.exit("no frames: nothing published covers that spot in that date range")
frames[0].save(out, save_all=True, append_images=frames[1:], duration=300, loop=0)
print("wrote", out, f"({len(frames)} frames)")
