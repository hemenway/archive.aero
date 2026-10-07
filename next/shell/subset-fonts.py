"""Regenerate committed UI fonts: python3 next/shell/subset-fonts.py.

Requires fonttools[woff] and the pinned @fontsource/barlow npm package.
"""
import json
from pathlib import Path
from fontTools import subset
from fontTools.ttLib import TTFont

here = Path(__file__).resolve().parent
package = here.parents[1] / 'node_modules/@fontsource/barlow'
destination = here / 'fonts'
destination.mkdir(exist_ok=True)
metadata = {'version': json.loads((package / 'package.json').read_text())['version'], 'ranges': {}}
# UI labels, Latin names, typography and navigation symbols. Other scripts
# use the system fallback without downloading another display font.
characters = set(range(0x20, 0x7f)) | set(range(0xa0, 0x100)) | set(range(0x2000, 0x2070)) | set(range(0x2190, 0x2200)) | {0x2212}
for weight in (500, 600, 700):
    font = TTFont(package / f'files/barlow-latin-{weight}-normal.woff2', recalcTimestamp=False)
    options = subset.Options()
    options.flavor = 'woff2'
    options.recalc_timestamp = False
    sub = subset.Subsetter(options=options)
    sub.populate(unicodes=characters)
    sub.subset(font)
    metadata['ranges'][str(weight)] = ','.join(f'U+{c:X}' for c in sorted(font.getBestCmap()))
    font.save(destination / f'barlow-{weight}.woff2')
(destination / 'subsets.json').write_text(json.dumps(metadata, indent=2) + '\n')
