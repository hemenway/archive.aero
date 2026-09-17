#!/usr/bin/env python3
"""Diff two FAA NASR 28-day subscription cycles and rank what changed.

Exploratory / test script: how should NASR churn be visualised?  Reads the
legacy fixed-width .txt files straight out of the two subscription zips under
/Volumes/projects/aisdata/us_faa/nasr/<effective>/ (pulled by aisdata_pull.py),
pairs records by their identifying key, and emits every difference as an
*event* with a significance score, then prints them most-significant first.

What "significant" means here (all tunable at the top of the file):

    score = FILE_WEIGHT[file] * RECTYPE_MOD[record type] * OP_MOD

  FILE_WEIGHT   what the entity is: an airport (100) outranks a navaid (90)
                outranks an ILS/tower (60) outranks a fix (30) outranks a
                preferred route (10).  Tuned for a map overlay: things that
                appear on / vanish from / move on a chart score high.
  RECTYPE_MOD   base record (the facility itself) 1.0, runway 0.6, sub-data
                (ILS localizer, airway point, tower frequency...) 0.4-0.5,
                remark records 0.1.
  OP_MOD        removed 1.0, added 0.9, changed = the largest factor among the
                changed fields: status/ownership 0.9, frequency/altitude 0.8,
                name/identifier 0.7, runway physicals 0.6, dates/phones 0.15,
                remark text 0.1, anything else 0.3.  A lat/lon change is
                scaled by how far it moved (<10 m survey nudge 0.15 ... >500 m
                relocation 1.0); elevation caps at 0.6 and scales likewise by feet.

  Entities (an airport with all its runways/remarks, a navaid with its
  holding patterns...) roll up as headline + 0.25 * sum(rest) / sqrt(n), so
  breadth counts but a 90-line tower frequency rework does not outrank a
  closed airport.

  Systemic events sit above all of that and are reported once, with the
  per-record copies cut to 10%:
    layout   a shipped Layout_Data/*_rf.txt whose field positions moved, or a
             record length change (2021-05 APT, 2021-06 ATS/AWY/WXL, 2021-09
             NAV, 2026-09 ATS/AWY/PFR + the APT PCN column) -- field diffs for
             the affected record types are suppressed, score 500
    mass     one field changing uniformly in >= MASS_MIN records of a type
             (152 navaids Y->N voice; 1,102 runways blank->PCR/...), or
             >= 20% of a record type added/removed at once (675 RCOs gone),
             score 200 + 10*sqrt(n)

Closure signals.  Private strips are 98% of APT deletions, and the FAA
  deletes them when it can no longer *verify* them, not when the runway is
  plowed -- so the record says so a year ahead.  Measured over the 13 cycles
  to 2026-10-01: a private-use row whose owner had not answered the 5010-2
  mailout in 20+ years ("last date information request was completed",
  col 893) was deleted at 2.2% per cycle, 7.8x the base rate; 10-20 yrs
  0.8%; a row flipped to status CI goes 6-12 cycles later.  The human-event
  tells -- a manager/owner name change after an inspection, a NOT MAINTAINED /
  UNSAFE / FOR SALE remark, the owner field going blank -- arrive about a
  year out.  The main report therefore prints every removed airport's last
  known state (REMOVED AIRPORTS: status, inspection, owner-response age,
  owner/manager, closure-flavoured remarks) and --watch lists the rows that
  are showing those signals now, before they drop.

Traps handled:
  * Every record carries the cycle's effective date (09/03/2026, 20260903,
    03 SEP 2026); those tokens are masked before comparing (own date plus
    the two cycles before, since change-notice sets re-ship older enroute
    files) and date fields the layout names are ignored structurally.
  * Sub-records that repeat under one key (remarks, airway points, ILS
    markers) are compared as multisets, leftover lines paired by positional
    similarity so a reworded remark reads as one change, not remove + add.
  * STARDP carries the procedure code only on a group's first record and
    renumbers its internal sequence on every insertion; the code is copied
    down into the point records at load and the sequence number ignored.
  * 28-day "change notice" cycles re-ship the previous 56-day enroute files
    (ARB ATS AWY CDR MTR PFR PJA STARDP WXL) unchanged; zero churn there is
    real, not a bug -- and the README's own 28/56 label is unreliable in
    2026, so reports state whether enroute files actually changed.
  * Field names come from the layout files, which are parsed loosely (tabs,
    AN0004, missing element ids); CDR (comma-delimited) has no layout.

Not covered: the Class_Airspace shapefile, the CSV_Data zip (2022-11 on) and
the AIXM tree.  The .txt files sunset 2026-12-24 -- after that the CSV set is
the only continuing series.

Usage:
  ~/venv/bin/python scripts/nasr_diff.py                   # newest two cycles
  ~/venv/bin/python scripts/nasr_diff.py 2026-08-06 2026-09-03
  ~/venv/bin/python scripts/nasr_diff.py --files APT,NAV --top 60 --per-entity 8
  ~/venv/bin/python scripts/nasr_diff.py --json out.json   # every event, for plotting
  ~/venv/bin/python scripts/nasr_diff.py --series [--files APT,NAV,ILS,TWR]
                                                           # one row per consecutive pair
  ~/venv/bin/python scripts/nasr_diff.py --show-layout APT # parsed layout sections
  ~/venv/bin/python scripts/nasr_diff.py --news [--news-min 40] [OLD NEW]
                                # newsletter sift: notable airport/runway/service changes, tower
                                # hours, ILS/navaids/weather/comms, routes and activity areas;
                                # bulk record churn is summarized or suppressed
  ~/venv/bin/python scripts/nasr_diff.py --watch [OLD NEW]
                                # closure watch: airports still in NASR that are showing the
                                # pre-deletion signals (status -> CI, private owner/manager changed,
                                # new closure-flavoured remark) plus the size of the stale pool
"""
import argparse, collections, io, json, math, os, re, sys, time, zipfile
from datetime import date

ROOT = '/Volumes/projects/aisdata/us_faa/nasr'

FILES = ['APT', 'NAV', 'ILS', 'TWR', 'AWY', 'ATS', 'ARB', 'FIX', 'AFF', 'AWOS', 'COM',
         'MTR', 'STARDP', 'FSS', 'HPF', 'MAA', 'PJA', 'LID', 'PFR', 'WXL', 'CDR']

FILE_DESC = {
    'APT': 'airports, runways, remarks', 'NAV': 'navaids', 'ILS': 'ILS components',
    'TWR': 'towers / terminal comms', 'AWY': 'airways (Part 95)', 'ATS': 'ATS routes (non-Part 95)',
    'ARB': 'ARTCC boundary points', 'FIX': 'fixes / waypoints', 'AFF': 'ARTCC sites, RCAG freqs',
    'AWOS': 'weather sensors', 'COM': 'comm outlets (RCO)', 'MTR': 'military training routes',
    'STARDP': 'SIDs / STARs (legacy)', 'FSS': 'flight service stations', 'HPF': 'holding patterns',
    'MAA': 'misc activity areas', 'PJA': 'parachute jump areas', 'LID': 'location identifiers',
    'PFR': 'preferred routes', 'WXL': 'weather reporting locations', 'CDR': 'coded departure routes',
}

FILE_WEIGHT = {
    'APT': 100, 'NAV': 90, 'ILS': 60, 'TWR': 60, 'AWY': 50, 'ATS': 40, 'ARB': 40, 'FIX': 30,
    'AFF': 30, 'AWOS': 25, 'COM': 25, 'MTR': 20, 'STARDP': 20, 'FSS': 20, 'HPF': 15, 'MAA': 15,
    'PJA': 15, 'LID': 10, 'PFR': 10, 'WXL': 10, 'CDR': 5,
}

# the record type that *is* the entity (one per facility)
BASE_RT = {'APT': 'APT', 'NAV': 'NAV1', 'FIX': 'FIX1', 'AWY': 'AWY1', 'ATS': 'ATS1', 'ILS': 'ILS1',
           'TWR': 'TWR1', 'AFF': 'AFF1', 'MTR': 'MTR1', 'PFR': 'PFR1', 'PJA': 'PJA1', 'MAA': 'MAA1',
           'HPF': 'HP1', 'AWOS': 'AWOS1'}
SINGLE_RT = {'ARB', 'COM', 'FSS', 'LID', 'WXL', 'CDR', 'STARDP'}      # no record-type prefix
# only refreshed on the 56-day major cycle; a change-notice set re-ships them
ENROUTE = {'ARB', 'ATS', 'AWY', 'CDR', 'MTR', 'PFR', 'PJA', 'STARDP', 'WXL'}

REMARK_RT = {'RMK', 'NAV2', 'FIX4', 'FIX5', 'AWY4', 'AWY5', 'ATS4', 'ATS5', 'TWR6', 'ILS6', 'AFF2',
             'AFF4', 'MTR2', 'MTR3', 'MTR4', 'PJA2', 'PJA3', 'PJA5', 'MAA3', 'MAA4', 'MAA5', 'MAA6',
             'MAA7', 'HP4', 'AWOS2'}
RECTYPE_MOD = {'RWY': 0.6, 'ATT': 0.3, 'ARS': 0.3, 'TWR3': 0.5, 'TWR7': 0.5, 'AFF3': 0.5}

OP_MOD = {'removed': 1.0, 'added': 0.9}
# first matching rule wins, so the cheap-and-noisy ones come first
FIELD_FACTORS = [
    (('REMARK', 'TEXT', 'DESCRIPTION'), 0.1),
    (('DATE', 'PHONE', 'ADDRESS', "MANAGER'S", "OWNER'S", 'INSPECT', 'ACTIVATION', 'SEQUENCE',
      'ACCURACY', 'SOURCE'), 0.15),
    (('LATITUDE', 'LONGITUDE', 'COORDINATE'), 1.0),          # scaled by metres moved, see position_factor
    (('ELEVATION',), 0.6),                                   # scaled by feet; doesn't move the dot
    (('MAGNETIC VARIATION',), 0.5),
    (('STATUS', 'COMMISSION', 'OWNERSHIP', 'USE ', 'CLOSED', 'CLASS '), 0.9),
    (('FREQUENCY', 'FREQ', 'CHANNEL', 'MEA', 'MOCA', 'ALTITUDE', 'MINIMUM'), 0.8),
    (('NAME', 'IDENTIFIER', 'ICAO', 'IDENT'), 0.7),
    (('LENGTH', 'WIDTH', 'SURFACE', 'LIGHT', 'MARKING', 'DISPLACED', 'THRESHOLD', 'GRADIENT',
      'WEIGHT', 'PCN', 'PAVEMENT', 'BEARING', 'RADIAL', 'DISTANCE'), 0.6),
]
DEFAULT_FIELD_FACTOR = 0.3
# fields that tick every cycle without meaning anything (the effective date is
# also string-masked, for files whose layout can't be parsed)
IGNORE_FIELDS = ('DATE INFORMATION EXTRACTED', 'INFORMATION EFFECTIVE DATE', 'EFFECTIVE DATE',
                 'INTERNAL SEQUENCE NUMBER')
LAYOUT_BREAK_MOD = 0.05     # per-record events in a file whose record length changed
LAYOUT_BREAK_SCORE = 500    # the break itself, so it tops the list
# one field changing in this many records of one type is a systemic (data-model /
# administrative) event, reported once; the per-record copies keep 10% weight
MASS_MIN = 50
MASS_SCORE = lambda n: 200 + 10 * math.sqrt(n)

# ---------------------------------------------------------------- record keys

def rectype(f, line):
    if f == 'APT':
        return line[:3]
    if f == 'AWOS':
        return line[:5]
    if f in SINGLE_RT:
        return ''
    return line[:4].rstrip()


def record_key(f, rt, line):
    """The identifying prefix; equal keys pair old with new."""
    if f == 'APT':
        return line[:{'APT': 14, 'RWY': 23, 'ATT': 18, 'ARS': 35, 'RMK': 29}.get(rt, 14)]
    if f == 'NAV':
        return line[:28]
    if f == 'FIX':
        return line[:64]
    if f == 'AWY':
        return line[:15]
    if f == 'ATS':
        return line[:25]
    if f == 'ILS':
        return line[:30] if rt == 'ILS5' else line[:28]
    if f == 'TWR':
        return line[:8]
    if f == 'AFF':
        return line[:8] + line[48:78] + line[128:133] if rt == 'AFF1' else line[:43]
    if f == 'ARB':
        return line[:12]
    if f == 'AWOS':
        return line[:19]
    if f == 'COM':
        return line[:11]
    if f == 'FSS':
        return line[:4]
    if f == 'HPF':
        return line[:87]
    if f == 'LID':
        return line[:9]
    if f == 'MAA':
        return line[:10]
    if f == 'MTR':
        return line[:12]
    if f == 'PFR':
        return line[:22] if rt == 'PFR2' else line[:19]
    if f == 'PJA':
        return line[:10]
    if f == 'STARDP':
        return line[38:51] + line[30:36] + line[10:12]      # procedure, fix, point type
    if f == 'WXL':
        return line[:5]
    if f == 'CDR':
        return line.split(',', 1)[0]
    return line[:12]


def entity_id(f, rt, line):
    """The facility a record belongs to, for roll-up."""
    if f == 'APT':
        return line[3:14]
    if f == 'NAV':
        return line[4:28]
    if f == 'FIX':
        return line[4:64]
    if f == 'AWY':
        return line[4:10]
    if f == 'ATS':
        return line[4:20]
    if f == 'ILS':
        return line[4:28]
    if f == 'TWR':
        return line[4:8]
    if f == 'AFF':
        return line[4:8] + '|' + (line[48:78] if rt == 'AFF1' else line[8:38])
    if f == 'ARB':
        return line[:7]
    if f == 'AWOS':
        return line[5:19]
    if f == 'STARDP':
        return line[38:51]
    if f == 'CDR':
        return line.split(',', 1)[0]
    return record_key(f, rt, line)[:{'COM': 11, 'FSS': 4, 'HPF': 87, 'LID': 9, 'MAA': 10,
                                     'MTR': 12, 'PFR': 19, 'PJA': 10, 'WXL': 5}.get(f, 12)]


def entity_label(f, rt, line):
    """Human-readable name, from the base record only (None otherwise)."""
    s = lambda a, b: line[a:b].strip()
    if f == 'APT' and rt == 'APT':
        return f"{s(27, 31)} {s(133, 183)}, {s(48, 50)} ({s(14, 27).lower()}, {s(93, 133).title()})"
    if f == 'NAV' and rt == 'NAV1':
        return f"{s(4, 8)} {s(8, 28)} {s(42, 72)}, {s(142, 144)}"
    if f == 'FIX' and rt == 'FIX1':
        return f"{s(4, 34)} {s(34, 64).title()} ({s(213, 228).lower()})"
    if f == 'AWY' and rt == 'AWY1':
        return f"airway {s(4, 9)}"
    if f == 'ATS' and rt == 'ATS1':
        return f"ATS route {s(4, 6)}{s(6, 18)}"
    if f == 'ILS' and rt == 'ILS1':
        return f"{s(28, 34)} {s(18, 28)} rwy {s(15, 18)} {s(44, 94)}"
    if f == 'TWR' and rt == 'TWR1':
        return f"{s(4, 8)} {s(104, 154)}, {s(62, 64)}"
    if f == 'AFF' and rt == 'AFF1':
        return f"{s(4, 8)} {s(8, 48)} / {s(48, 78)} {s(128, 133)}"
    if f == 'AWOS' and rt == 'AWOS1':
        return f"{s(5, 9)} {s(9, 19)}"
    if f == 'COM':
        return f"{s(0, 4)} {s(4, 11)}"
    if f == 'FSS':
        return f"{s(0, 4)} {s(4, 30)}"
    if f == 'HPF' and rt == 'HP1':
        return s(4, 84)
    if f == 'LID':
        return f"{s(4, 9)} {s(14, 54).title()}"
    if f == 'MAA' and rt == 'MAA1':
        return f"{s(4, 10)} {s(10, 35)}"
    if f == 'MTR' and rt == 'MTR1':
        return f"{s(4, 7)}{s(7, 12)}"
    if f == 'PFR' and rt == 'PFR1':
        return f"{s(4, 9)}>{s(9, 14)} {s(14, 17)} #{s(17, 19)}"
    if f == 'PJA' and rt == 'PJA1':
        return f"{s(4, 10)} near {s(10, 14)}"
    if f == 'WXL':
        return f"{s(0, 5)} {s(22, 62).title()}, {s(62, 64)}"
    if f == 'ARB':
        return f"{s(0, 3)} {s(12, 52).title()} {s(52, 62).lower()} pt {s(7, 12)}"
    if f == 'STARDP':
        return f"{s(38, 51)} {s(51, 161)}" if s(51, 161) else None
    if f == 'CDR':
        return line.split(',', 1)[0]
    return None


def sub_id(f, rt, line):
    """What distinguishes a sub-record within its entity (runway id, remark element...)."""
    key = record_key(f, rt, line)
    ent = entity_id(f, rt, line)
    tail = key
    if ent and ent in key:
        tail = key.replace(ent, '', 1)
    if rt and tail.startswith(rt):
        tail = tail[len(rt):]
    return tail.strip()


def payload(f, rt, line, n=70):
    """The record beyond its key, whitespace collapsed -- what an add/remove shows."""
    body = ' '.join(line[len(record_key(f, rt, line)):].replace('##/##/####', '').replace('## ### ####', '').split())
    return body if len(body) <= n else body[:n - 2] + '..'

# ------------------------------------------------------------------- layouts

FIELD_RE = re.compile(r'^\s*([LR])\s*(AN|N)\s*(\d+)\s+(\d+)\s+(\S*)\s*(.*?)\s*$')
ELEM_RE = re.compile(r'^(N/A|NONE|DLID|TRAN|GEN|[A-Z]{1,4}\d{1,3}[A-Z]?)$')   # else it's the description
HDR_RE = re.compile(r"'([A-Z0-9]{2,6})'\s+RECORD\s+TYPE")
HINT_RE = re.compile(r'^\s*([A-Z0-9]{2,6}):\s')


def parse_layout(text):
    """-> [(label, [(start, length, name)])], best effort across the 21 layout dialects."""
    lines = text.splitlines()
    sections, pending = [], None
    for i, l in enumerate(lines):
        if '\t' in l:                       # maa_rf.txt is tab-delimited
            parts = [p for p in l.split('\t')]
            if len(parts) >= 5 and parts[0].strip() in ('L', 'R') and parts[2].strip().isdigit():
                just, typ, length, start = (p.strip() for p in parts[:4])
                desc = ' '.join(p.strip() for p in parts[4:] if p.strip())
                m = (just, typ, length, start, '', desc)
            else:
                m = None
        else:
            hm = HDR_RE.search(l)
            if hm and not FIELD_RE.match(l):
                pending = hm.group(1)
            mm = FIELD_RE.match(l)
            m = mm.groups() if mm else None
        if not m:
            continue
        _, _, length, start, elem, desc = m
        elem = elem.rstrip(',.')
        if elem and not ELEM_RE.match(elem):      # no element column: first word is description
            desc = (elem + ' ' + desc).strip()
        start, length = int(start), int(length)
        if start == 1 and (not sections or sections[-1][1]):
            label = None
            for j in range(i + 1, min(i + 4, len(lines))):
                h = HINT_RE.match(lines[j])
                if h:
                    label = h.group(1)
                    break
            sections.append((label or pending or f'sec{len(sections) + 1}', []))
            pending = None
        if not sections:
            sections.append(('sec1', []))
        sections[-1][1].append((start, length, desc.rstrip('. ')))
    return sections


def section_for(sections, rt):
    if not sections:
        return []
    for label, fields in sections:
        if label == rt:
            return fields
    if rt and rt[-1].isdigit() and int(rt[-1]) - 1 < len(sections):
        return sections[int(rt[-1]) - 1][1]
    return max(sections, key=lambda s: len(s[1]))[1]


def field_diff(fields, a, b):
    """-> [(field name, old, new)] over the columns that differ."""
    n = max(len(a), len(b))
    a, b = a.ljust(n), b.ljust(n)
    runs, i = [], 0
    while i < n:
        if a[i] != b[i]:
            j = i
            while j < n and a[j] != b[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    out, seen = [], set()
    for i, j in runs:
        covered = False
        for start, length, name in fields:
            s0, e0 = start - 1, start - 1 + length
            if s0 < j and i < e0:
                covered = True
                if (s0, e0) in seen:
                    continue
                seen.add((s0, e0))
                if not ignorable(name):
                    out.append((name, a[s0:e0].strip(), b[s0:e0].strip()))
        if not covered:
            out.append((f'cols {i + 1}-{j}', a[i:j].strip(), b[i:j].strip()))
    return out


DMS_RE = re.compile(r'^(\d{1,3})-(\d{2})-(\d{2}(?:\.\d+)?)([NSEW])$')
SEC_RE = re.compile(r'^(\d+(?:\.\d+)?)([NSEW])$')


def arcsec(v):
    m = DMS_RE.match(v)
    if m:
        d, mi, se, h = m.groups()
        x = int(d) * 3600 + int(mi) * 60 + float(se)
    else:
        m = SEC_RE.match(v)
        if not m:
            return None
        x, h = float(m.group(1)), m.group(2)
    return -x if h in 'SW' else x


def position_factor(a, b):
    """Scale a lat/lon change by how far it moved: survey nudges are not relocations."""
    x, y = arcsec(a), arcsec(b)
    if x is None or y is None:
        return DEFAULT_FIELD_FACTOR        # named like a coordinate, isn't one
    metres = abs(x - y) * 30.87
    return 0.15 if metres < 10 else 0.3 if metres < 50 else 0.6 if metres < 500 else 1.0


def elevation_factor(a, b):
    try:
        d = abs(float(a) - float(b))
    except ValueError:
        return DEFAULT_FIELD_FACTOR
    return 0.15 if d < 3 else 0.4 if d < 20 else 1.0


def field_factor(name, a='', b=''):
    u = name.upper()
    for keys, fac in FIELD_FACTORS:
        if any(k in u for k in keys):
            if fac == 1.0 and ('LATITUDE' in u or 'LONGITUDE' in u or 'COORDINATE' in u):
                return position_factor(a, b)
            if 'ELEVATION' in u:
                return fac * elevation_factor(a, b)
            return fac
    return DEFAULT_FIELD_FACTOR


def ignorable(name):
    u = name.upper()
    return any(k in u for k in IGNORE_FIELDS)

# --------------------------------------------------------------------- load

class Cycle:
    def __init__(self, eff):
        self.eff = eff
        self.data, self.labels, self.reclen, self.layouts = {}, {}, {}, {}
        self.counts = {}                    # file -> Counter(record type)
        self.readme = ''


def date_masks(effs):
    """Tokens to blank: the cycle's own effective date in its three spellings.
    Keep this list short -- every token is a substring scan of every line."""
    out = []
    for eff in effs:
        d = date.fromisoformat(eff)
        out += [(d.strftime('%m/%d/%Y'), '##/##/####'),
                (d.strftime('%Y%m%d'), '########'),
                (d.strftime('%d %b %Y').upper(), '## ### ####')]
    return out


def masks_for(cycles, effs, back=2):
    """The dates of these cycles plus the `back` cycles before each: a 28-day
    change-notice set re-ships the previous major cycle's enroute files with
    the older date still stamped in them."""
    want = set()
    for eff in effs:
        i = cycles.index(eff)
        want.update(cycles[max(0, i - back):i + 1])
    return date_masks(sorted(want))


def zip_path(eff):
    return os.path.join(ROOT, eff, f'28DaySubscription_Effective_{eff}.zip')


def stardp_inherit(line, state):
    """STARDP groups (one internal sequence number) carry the procedure code /
    name only on their first record; copy it into the point records so every
    record keys on the procedure rather than on a sequence number that
    renumbers whenever anything is inserted."""
    seq, code, name = line[:5], line[38:51], line[51:161].strip()
    if seq != state.get('seq'):
        state['seq'] = seq
        state['code'] = code if code.strip() else (name[:13] if name else '').ljust(13)
    if not code.strip() and state['code'].strip():
        line = line[:38] + state['code'] + line[51:]
    return line


def load_cycle(eff, files, masks, quiet=False):
    t0 = time.time()
    cyc = Cycle(eff)
    z = zipfile.ZipFile(zip_path(eff))
    names = {n.upper(): n for n in z.namelist()}
    for f in files:
        n = names.get(f + '.TXT')
        if not n:
            continue
        groups, labels, first = collections.defaultdict(list), {}, None
        counts = collections.Counter()
        base = BASE_RT.get(f)
        state = {}
        with z.open(n) as fh:
            for raw in io.TextIOWrapper(fh, encoding='latin-1', newline=None):
                line = raw.rstrip('\r\n')
                if not line:
                    continue
                if first is None:
                    first = len(line)
                for tok, mask in masks:
                    if tok in line:
                        line = line.replace(tok, mask)
                if f == 'STARDP':
                    line = stardp_inherit(line, state)
                rt = rectype(f, line)
                counts[rt or f] += 1
                groups[record_key(f, rt, line)].append(line)
                if f in SINGLE_RT or rt == base:
                    lab = entity_label(f, rt, line)
                    if lab:
                        labels[entity_id(f, rt, line)] = lab
        cyc.data[f], cyc.labels[f], cyc.reclen[f], cyc.counts[f] = groups, labels, first, counts
        ln = names.get(f'LAYOUT_DATA/{f}_RF.TXT')
        if ln:
            cyc.layouts[f] = z.read(ln)
    if 'README.TXT' in names:
        cyc.readme = z.read(names['README.TXT']).decode('latin-1')
    if not quiet:
        print(f'  loaded {eff} in {time.time() - t0:.1f}s', file=sys.stderr)
    return cyc

# --------------------------------------------------------------------- diff

def pair_leftovers(old, new, cap=80):
    """Greedy pairing of unmatched lines under one key by positional similarity.

    Records are fixed-width, so the right metric is 'same character in the same
    column' -- Jaccard over the non-blank (column, char) pairs, which is all
    C-speed set ops (difflib's ratio() was quadratic on 1,600-char tower lines).
    """
    if len(old) == 1 and len(new) == 1:
        return [(old[0], new[0])], [], []
    if len(old) > cap or len(new) > cap:
        return [], old, new
    so = [{(i, c) for i, c in enumerate(a) if c != ' '} for a in old]
    sn = [{(i, c) for i, c in enumerate(b) if c != ' '} for b in new]
    cand = []
    for i, a in enumerate(so):
        for j, b in enumerate(sn):
            inter = len(a & b)
            if inter:
                r = inter / (len(a) + len(b) - inter)
                if r >= 0.5:
                    cand.append((r, i, j))
    cand.sort(reverse=True)
    used_i, used_j, pairs = set(), set(), []
    for r, i, j in cand:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j)
        pairs.append((old[i], new[j]))
    return (pairs, [a for i, a in enumerate(old) if i not in used_i],
            [b for j, b in enumerate(new) if j not in used_j])


def rectype_mod(f, rt):
    if f in SINGLE_RT or rt == BASE_RT.get(f):
        return 1.0
    if rt in RECTYPE_MOD:
        return RECTYPE_MOD[rt]
    if rt in REMARK_RT:
        return 0.1
    return 0.4


def broken_rectypes(old_layout, new_layout):
    """Record types whose field positions moved between the two shipped layouts."""
    if old_layout is None or new_layout is None or old_layout == new_layout:
        return set()
    old = {lab: [(st, ln) for st, ln, _ in fl] for lab, fl in parse_layout(old_layout.decode('latin-1'))}
    new = {lab: [(st, ln) for st, ln, _ in fl] for lab, fl in parse_layout(new_layout.decode('latin-1'))}
    if set(old) != set(new):
        return {'*'}
    return {lab for lab in new if old[lab] != new[lab]}


def diff_file(f, old_cyc, new_cyc, sections, layout_broken):
    """layout_broken: set of record types whose field diffs are meaningless ('*' = all)."""
    old, new = old_cyc.data.get(f, {}), new_cyc.data.get(f, {})
    labels = dict(old_cyc.labels.get(f, {}))
    labels.update(new_cyc.labels.get(f, {}))
    events = []
    w = FILE_WEIGHT.get(f, 10)

    def emit(op, rt, line, fields=None, line_new=None):
        ent = entity_id(f, rt, line)
        broken = '*' in layout_broken or rt in layout_broken
        mod = rectype_mod(f, rt)
        if op == 'changed':
            if not fields:
                return
            opm = max(field_factor(nm, a, b) for nm, a, b in fields)
        else:
            opm = OP_MOD[op]
        score = w * mod * opm * (LAYOUT_BREAK_MOD if broken else 1.0)
        events.append({
            'file': f, 'rt': rt or f, 'op': op, 'entity': ent.strip(),
            'label': labels.get(ent) or ent.strip(), 'sub': sub_id(f, rt, line),
            'score': round(score, 2), 'fields': fields or [],
            'snippet': payload(f, rt, line_new or line),
        })

    for k in set(old) | set(new):
        o, n = old.get(k, []), new.get(k, [])
        if o == n:
            continue
        oc, nc = collections.Counter(o), collections.Counter(n)
        o_rest, n_rest = list((oc - nc).elements()), list((nc - oc).elements())
        if not o_rest and not n_rest:
            continue                        # same records, different order
        rt = rectype(f, (o_rest or n_rest)[0])
        pairs, o_rest, n_rest = pair_leftovers(o_rest, n_rest)
        for a, b in pairs:
            if '*' in layout_broken or rt in layout_broken:
                fields = [('record layout changed', '', '')]
            else:
                fields = field_diff(section_for(sections, rt), a, b)
            emit('changed', rt, a, fields, line_new=b)
        for a in o_rest:
            emit('removed', rt, a)
        for b in n_rest:
            emit('added', rt, b)
    return events


def diff_cycles(old_cyc, new_cyc, files):
    """-> (events, layout_notes)"""
    events, notes = [], []
    for f in files:
        if f not in new_cyc.data or f not in old_cyc.data:
            if f in new_cyc.data or f in old_cyc.data:
                notes.append((f, 'file only in one cycle', True))
            continue
        broken = set()
        if f != 'CDR' and old_cyc.reclen[f] != new_cyc.reclen[f]:
            notes.append((f, f'record length {old_cyc.reclen[f]} -> {new_cyc.reclen[f]}', True))
            broken = {'*'}
        elif old_cyc.layouts.get(f) != new_cyc.layouts.get(f):
            broken = broken_rectypes(old_cyc.layouts.get(f), new_cyc.layouts.get(f))
            if broken:
                notes.append((f, f"field positions moved in {' '.join(sorted(broken))} records", True))
            else:
                notes.append((f, 'Layout_Data/*_rf.txt wording differs, positions unchanged', False))
        sections = parse_layout(new_cyc.layouts[f].decode('latin-1')) if f in new_cyc.layouts else []
        events += diff_file(f, old_cyc, new_cyc, sections, broken)
        if broken:
            events.append({'file': f, 'rt': f, 'op': 'layout', 'entity': '*', 'label': f'{f}.txt record layout changed',
                           'sub': notes[-1][1], 'score': LAYOUT_BREAK_SCORE, 'fields': [], 'snippet': ''})
    return fold_mass_changes(fold_mass_addremove(events, old_cyc, new_cyc)), notes


def fold_mass_addremove(events, old_cyc, new_cyc):
    """>= MASS_MIN and >= 20% of a record type added or removed at once is one event."""
    per = collections.Counter((e['file'], e['rt'], e['op']) for e in events if e['op'] in ('added', 'removed'))
    mass = {}
    for (f, rt, op), n in per.items():
        total = (old_cyc if op == 'removed' else new_cyc).counts.get(f, {}).get(rt, 0)
        if n >= MASS_MIN and total and n >= 0.2 * total:
            mass[(f, rt, op)] = (n, total)
    if not mass:
        return events
    out = []
    for e in events:
        if (e['file'], e['rt'], e['op']) in mass:
            e = dict(e, score=round(e['score'] * 0.1, 2), mass_fields=[e['op']])
        out.append(e)
    for (f, rt, op), (n, total) in mass.items():
        out.append({'file': f, 'rt': rt, 'op': 'mass', 'entity': '*',
                    'label': f'{n} of {total} {rt} records {op} ({100 * n / total:.0f}%)',
                    'sub': f'{FILE_DESC.get(f, f)}', 'score': round(MASS_SCORE(n), 1),
                    'fields': [], 'snippet': '', 'n': n})
    return out


def fold_mass_changes(events):
    """Collapse a field that changed in >= MASS_MIN records of one type into one event."""
    per = collections.defaultdict(list)
    for e in events:
        if e['op'] == 'changed':
            for name, a, b in e['fields']:
                per[(e['file'], e['rt'], name)].append((a, b))
    def uniform(pairs):
        """Same old->new in 30%+, or 60%+ blank before, or 60%+ landing on one value."""
        n = len(pairs)
        top_pair = collections.Counter(pairs).most_common(1)[0][1]
        blank = sum(1 for a, _ in pairs if not a)
        top_new = collections.Counter(b for _, b in pairs).most_common(1)[0][1]
        return top_pair >= 0.3 * n or blank >= 0.6 * n or top_new >= 0.6 * n

    mass = {k: v for k, v in per.items()
            if len(v) >= MASS_MIN and k[2] != 'record layout changed' and uniform(v)}
    if not mass:
        return events
    out = []
    for e in events:
        if e['op'] == 'changed':
            keep = [(n, a, b) for n, a, b in e['fields'] if (e['file'], e['rt'], n) not in mass]
            if len(keep) != len(e['fields']):
                e = dict(e)
                e['mass_fields'] = [n for n, _, _ in e['fields'] if (e['file'], e['rt'], n) in mass]
                if keep:
                    opm = max(field_factor(n, a, b) for n, a, b in keep)
                    full = max(field_factor(n, a, b) for n, a, b in e['fields'])
                    e['score'] = round(e['score'] * opm / full, 2) if full else e['score']
                else:
                    e['score'] = round(e['score'] * 0.1, 2)
        out.append(e)
    for (f, rt, name), pairs in mass.items():
        pat = collections.Counter((a or '(blank)', b or '(blank)') for a, b in pairs)
        top = ', '.join(f'{a[:28]} -> {b[:28]} x{n}' for (a, b), n in pat.most_common(2))
        out.append({'file': f, 'rt': rt, 'op': 'mass', 'entity': '*',
                    'label': f'{len(pairs)} {rt} records: {name.lower()}',
                    'sub': top[:110], 'score': round(MASS_SCORE(len(pairs)), 1),
                    'fields': [], 'snippet': '', 'n': len(pairs)})
    return out


def roll_up(events):
    ents = collections.defaultdict(list)
    for e in events:
        systemic = e['op'] in ('mass', 'layout')
        ents[(e['file'], e['label'] if systemic else e['entity'])].append(e)
    out = []
    for (f, ent), evs in ents.items():
        evs.sort(key=lambda e: -e['score'])
        head = evs[0]['score']
        rest = evs[1:]
        score = head + 0.25 * sum(e['score'] for e in rest) / math.sqrt(max(1, len(rest)))
        out.append({'file': f, 'entity': ent, 'label': evs[0]['label'], 'score': round(score, 1),
                    'n': len(evs), 'events': evs})
    out.sort(key=lambda x: -x['score'])
    return out

# ------------------------------------------------------------------- report

def cycle_kind(readme):
    """What the README claims -- unreliable in 2026 (says 28-day on cycles whose
    enroute files changed), so reports pair it with the empirical enroute flag."""
    m = re.search(r'cycle is an? (.+?) subscriber set', readme, re.I)
    if not m:
        return '?'
    return '56-day major' if '56' in m.group(1) else '28-day change notice' if '28' in m.group(1) else m.group(1)


def fmt_change(e, width):
    if e['op'] == 'changed':
        parts = []
        for name, a, b in e['fields'][:4]:
            a, b = a or '(blank)', b or '(blank)'
            if len(a) + len(b) > 70:
                a, b = a[:33] + '..', b[:33] + '..'
            parts.append(f'{name.lower()}: {a} -> {b}')
        if len(e['fields']) > 4:
            parts.append(f'+{len(e["fields"]) - 4} more')
        s = f"{e['rt']} {e['sub']}  " + ' | '.join(parts)
    elif e['op'] in ('mass', 'layout'):
        s = f"{e['rt']} {e['op']}  {e['sub']}"
    else:
        s = f"{e['rt']} {e['op']}  {e['sub']}  {e['snippet']}".replace('   ', '  ')
    return s if len(s) <= width else s[:width - 2] + '..'


def print_report(old_cyc, new_cyc, events, notes, top, per_entity, width):
    enroute = sorted({e['file'] for e in events if e['file'] in ENROUTE})
    print(f"NASR {old_cyc.eff} -> {new_cyc.eff}    README says {cycle_kind(new_cyc.readme)}; "
          f"enroute files {'changed: ' + ' '.join(enroute) if enroute else 'unchanged'}")
    systemic = [e for e in events if e['op'] in ('mass', 'layout')]
    print(f"{len(events) - len(systemic):,} record events across {len(set(e['file'] for e in events))} files, "
          f"{len(systemic)} systemic, total score {sum(e['score'] for e in events):,.0f}")

    if notes:
        print('\nLAYOUT CHANGES')
        for f, msg, broken in notes:
            print(f"  {f:7} {msg}{'   (field diffs suppressed there)' if broken else ''}")

    print('\nBY FILE                                    added  removed  changed  systemic  entities     score')
    per = collections.defaultdict(lambda: collections.Counter())
    ents = collections.defaultdict(set)
    for e in events:
        per[e['file']][e['op']] += 1
        per[e['file']]['score'] += e['score']
        ents[e['file']].add(e['entity'])
    for f in sorted(per, key=lambda f: -per[f]['score']):
        c = per[f]
        print(f"  {f:7}{FILE_DESC.get(f, ''):32} {c['added']:6} {c['removed']:8} {c['changed']:8} "
              f"{c['mass'] + c['layout']:9} {len(ents[f] - {'*'}):9} {c['score']:9,.0f}")
    quiet = [f for f in FILES if f in new_cyc.data and f in old_cyc.data and f not in per]
    if quiet:
        print(f"  unchanged: {' '.join(quiet)}")

    rolled = roll_up(events)
    print(f'\nTOP {top} ENTITIES  (score = headline event + 25% * sum(rest) / sqrt(n))')
    for i, r in enumerate(rolled[:top], 1):
        print(f"{i:3}. {r['score']:7.1f}  {r['file']:6} {r['label'][:70]}   [{r['n']} event{'s' if r['n'] > 1 else ''}]")
        for e in r['events'][:per_entity]:
            print(f"               {fmt_change(e, width)}")
        if r['n'] > per_entity:
            print(f"               .. {r['n'] - per_entity} more")

    if 'APT' in old_cyc.data and 'APT' in new_cyc.data:
        print_removed(old_cyc, new_cyc, removed_airports(old_cyc, new_cyc), width)

    fields = collections.Counter()
    for e in events:
        if e['op'] == 'changed':
            for name, _, _ in e['fields']:
                if name != 'record layout changed':
                    fields[(e['file'], e['rt'], name)] += 1
    if fields:
        folded = {(e['file'], e['rt'], n) for e in events for n in e.get('mass_fields', [])}
        print('\nMOST-CHANGED FIELDS   (* = folded into one systemic event above)')
        for (f, rt, name), n in fields.most_common(25):
            print(f"  {n:7,}{'*' if (f, rt, name) in folded else ' '} {f:6} {rt:5} {name[:70]}")


def series(cycles, files, all_cycles, jpath):
    """One row per consecutive pair: per-file score + counts.  Loads each zip once."""
    rows = []
    prev = None
    hdr = ['pair', 'kind', 'enr', 'total'] + files
    print('  '.join(f'{h:>7}' if i > 2 else f'{h:<24}' if i == 0 else f'{h:>4}' for i, h in enumerate(hdr)))
    for eff in cycles:
        cur = load_cycle(eff, files, masks_for(all_cycles, [eff]), quiet=True)
        if prev is not None:
            events, notes = diff_cycles(prev, cur, files)
            per = collections.Counter()
            cnt = collections.Counter()
            for e in events:
                per[e['file']] += e['score']
                cnt[e['file']] += 1
            total = sum(per.values())
            kind = cycle_kind(cur.readme)
            kind = '28d' if '28' in kind else ('56d' if '56' in kind else '?')
            enr = any(cnt[f] for f in files if f in ENROUTE)
            row = {'old': prev.eff, 'new': cur.eff, 'kind': kind, 'enroute_changed': enr, 'total': round(total),
                   'score': {f: round(per[f]) for f in files}, 'events': {f: cnt[f] for f in files},
                   'layout': [f for f, _, b in notes if b]}
            rows.append(row)
            print(f"{prev.eff}>{cur.eff:<11}  {kind:>4}  {'yes' if enr else '-':>4}  {total:7,.0f}" +
                  ''.join(f"  {per[f]:7,.0f}" for f in files) +
                  (f"   layout: {' '.join(row['layout'])}" if row['layout'] else ''))
            sys.stdout.flush()
        prev = cur
    if jpath:
        json.dump(rows, open(jpath, 'w'), indent=1)
        print(f'wrote {jpath}', file=sys.stderr)



# --------------------------------------------------------------------- news
#
# Newsletter sifting.  Airport *prominence* is independent of the change
# scoring above: it says how much anyone cares about this airport at all,
# from fields the APT record carries.  Public use alone clears the bar; a
# private field needs a tower, a certificate or real traffic to make it.

NEWS_MIN = 40
NEWS_TYPE_MULT = {'airport': 1.0, 'seaplane': 0.5, 'heli': 0.3}   # else 0.2 (ultralight, glider, balloon)


def apt_profile(base, rwys):
    s = lambda a, b: base[a:b].strip()
    num = lambda a, b: int(s(a, b)) if s(a, b).isdigit() else 0
    based = sum(num(a, a + 3) for a in (1004, 1007, 1010, 1013, 1016, 1019, 1022))
    ops = sum(num(a, a + 6) for a in (1025, 1031, 1037, 1043, 1049, 1055))
    commercial = num(1025, 1031) + num(1031, 1037)
    longest, paved = 0, False
    for r in rwys:
        ln = r[23:28].strip()
        longest = max(longest, int(ln) if ln.isdigit() else 0)
        if r[32:44].strip().upper().startswith(('ASPH', 'CONC', 'PEM')):
            paved = True
    typ = s(14, 27).lower()
    p = dict(site=base[3:14].strip(), ident=s(27, 31), name=s(133, 183), city=s(93, 133).title(),
             st=s(48, 50), typ=typ, use=s(185, 187), own=s(183, 185), tower=s(980, 981) == 'Y',
             arff=s(842, 857), npias=s(857, 864), based=based, ops=ops, commercial=commercial,
             longest=longest, paved=paved, icao=s(1210, 1217), act=s(833, 840), status=s(840, 842),
             lat=s(523, 538), lon=s(550, 565),
             # the verification trail: who answers for this record, and when they last did
             insp_method=s(881, 883), insp_agency=s(883, 884), last_insp=s(884, 892),
             last_info=s(892, 900), owner=s(187, 222), mgr=s(355, 390))
    prom = ((40 if p['use'] == 'PU' else 0) + (40 if p['tower'] else 0) + (50 if p['arff'] else 0)
            + (20 if p['npias'] else 0) + min(30, based // 2) + min(30, ops // 2000)
            + (15 if commercial else 0) + min(20, longest // 500) + (5 if paved else 0)
            + (10 if p['icao'] else 0))
    mult = next((m for k, m in NEWS_TYPE_MULT.items() if k in typ), 0.2)
    p['prominence'] = round(prom * mult)
    return p


def describe(p):
    bits = [p['typ'], 'public use' if p['use'] == 'PU' else 'private use']
    if p['tower']:
        bits.append('towered')
    if p['arff']:
        bits.append('Part 139')
    if p['longest']:
        bits.append(f"{p['longest']:,} ft {'paved' if p['paved'] else 'unpaved'}")
    if p['based']:
        bits.append(f"{p['based']} based aircraft")
    if p['ops']:
        bits.append(f"{p['ops']:,} ops/yr")
    if p['act']:
        bits.append(f"since {p['act']}")
    return ', '.join(bits)


def apt_index(cyc):
    """site -> (base line, [runway lines]) for one cycle."""
    bases, rwys = {}, collections.defaultdict(list)
    for k, lines in cyc.data.get('APT', {}).items():
        if k.startswith('APT'):
            bases[k[3:14].strip()] = lines[0]
        elif k.startswith('RWY'):
            rwys[k[3:14].strip()] += lines
    return bases, rwys


def apt_remarks(cyc):
    """site -> [(element, text)] for one cycle."""
    out = collections.defaultdict(list)
    for k, lines in cyc.data.get('APT', {}).items():
        if k.startswith('RMK'):
            for l in lines:
                out[k[3:14].strip()].append((l[16:31].strip(), l[31:].strip()))
    return out


def records_of(cyc, f, rt=None):
    """Yield every record in a file, optionally restricted to one record type."""
    for lines in cyc.data.get(f, {}).values():
        for line in lines:
            if rt is None or rectype(f, line) == rt:
                yield line


def entity_lines(cyc, f, rt=None):
    """One representative line per entity (base records when *rt* is supplied)."""
    out = {}
    for line in records_of(cyc, f, rt):
        r = rectype(f, line)
        out.setdefault(entity_id(f, r, line).strip(), line)
    return out


def entity_labels(cyc, f):
    return {k.strip(): v for k, v in cyc.labels.get(f, {}).items()}

# ------------------------------------------------------------ closure signals
#
# See the docstring: deletions are the FAA giving up on verifying a record, and
# the APT base record carries the trail -- status, inspection, the owner's last
# 5010 response, the owner/manager names -- plus the remarks a human wrote.

STALE_YEARS = 20            # owner-response age past which a private strip is on the purge list
INSPECT_BY = {'F': 'FAA insp', 'S': 'state insp', 'C': 'contract insp',
              '1': '5010-1 mailout', '2': '5010-2 mailout'}
CLOSURE_RE = re.compile(r'NOT MAINTAINED|UNMAINTAINED|UNSAFE|UNUSABLE|CLSD INDEF|CLOSED INDEF|CLSD PERM'
                        r'|CLOSED PERM|PERM(?:ANENTLY)? CLSD|DEACTIVAT|ABANDON|FOR SALE|\bSOLD\b|DECEASED'
                        r'|NO LONGER|NOT IN USE|OUT OF SERVICE|\bARPT (?:IS )?CLSD\b')
# the same words in a remark about hours, seasons, taxiways or the approach control
NOT_CLOSURE_RE = re.compile(r'\b(?:TWY|TAXIWAY|APCH|ATCT|TWR|FSS|UNICOM|WHEN|HOL|HOLS|WKEND|WEEKEND|NIGHT'
                            r'|DUSK|DAWN|WINTER|SEASON|SNOW|EXC|EXCP|PPR|DURG|DALGT|TO PUB|TO PUBLIC|TO ACR'
                            r'|TO TRANSIENT|EACH YEAR|ANNUALLY|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC'
                            r'|\d{4}-\d{4})\b')
TAG_ORDER = ['converted', 'was CP', 'was CI', 'unverified', 'stale', 'remark', 'young', '-']


def closure_remark(text):
    u = text.upper()
    return bool(CLOSURE_RE.search(u)) and not NOT_CLOSURE_RE.search(u)


def same_name(a, b):
    """Punctuation, spacing and a ', Inc' / ' - Hospital' tail are not a new owner."""
    x, y = (re.sub(r'[^A-Z0-9]', '', s.upper()) for s in (a, b))
    return bool(x) and bool(y) and (x in y or y in x)


def years_since(mmddyyyy, eff):
    """Age of an MMDDYYYY field at a cycle's effective date, or None."""
    if not mmddyyyy or len(mmddyyyy) != 8 or not mmddyyyy.isdigit():
        return None
    try:
        d = date(int(mmddyyyy[4:]), int(mmddyyyy[:2]), int(mmddyyyy[2:4]))
    except ValueError:
        return None
    return (date.fromisoformat(eff) - d).days / 365.25


def mmyyyy(mmddyyyy):
    return f'{mmddyyyy[:2]}/{mmddyyyy[4:]}' if mmddyyyy and len(mmddyyyy) == 8 else (mmddyyyy or '-')


def response_text(p, eff):
    age = years_since(p['last_info'], eff)
    if age is None:
        return 'no owner response on record'
    return f"owner response {mmyyyy(p['last_info'])} ({age:.1f}y)"


def facts(p, eff, act=True):
    """The verification trail on one airport record, one line."""
    m = p['insp_method']
    insp = INSPECT_BY.get(m, f'insp {m}' if m else 'no insp')
    if p['last_insp']:
        insp += f" {mmyyyy(p['last_insp'])}"
    who = f"owner {p['owner'].title()}" if p['owner'] else 'owner blank'
    if p['mgr'] and p['mgr'] != p['owner']:
        who += f", mgr {p['mgr'].title()}"
    return (f"act {p['act'] or '-'}  " if act else '') + f"{insp}  {response_text(p, eff)}  {who}"


def head(p):
    return f"{p['ident']:5} {p['name'].title()}, {p['city']} {p['st']}".ljust(50)[:50] + f"  {p['typ'][:8]:8} {p['use']} {p['status'] or 'O':2}"


def death_tag(p, rmks, eff, converted=False):
    """What kind of deletion this was.  'was CI' = already closed; 'unverified 35y'
    = the purge of rows nobody answers for; 'stale 7y' = same, milder; 'remark' =
    a human wrote the reason; 'young' = activated <5 yrs ago (a paperwork strip);
    'converted' = the ident re-appears on a new site number."""
    if converted:
        return 'converted'
    if p['status'] in ('CI', 'CP'):
        return f"was {p['status']}"
    age = years_since(p['last_info'], eff)
    if age is not None and age > STALE_YEARS:
        return f'unverified {age:.0f}y'
    if age is not None and age > 5:
        return f'stale {age:.0f}y'
    if any(closure_remark(tx) for _, tx in rmks):
        return 'remark'
    m = re.match(r'(\d\d)/(\d{4})$', p['act'])
    if m and (date.fromisoformat(eff) - date(int(m.group(2)), int(m.group(1)), 1)).days < 5 * 365.25:
        return 'young'
    return '-'


def removed_airports(old_cyc, new_cyc):
    """Every APT base record in old and not in new, with its last known state."""
    ob, orw = apt_index(old_cyc)
    nb, nrw = apt_index(new_cyc)
    orm = apt_remarks(old_cyc)
    new_idents = {apt_profile(l, [])['ident'] for s, l in nb.items() if s not in ob}
    out = []
    for site, line in ob.items():
        if site in nb:
            continue
        p = apt_profile(line, orw.get(site, []))
        rmks = orm.get(site, [])
        # the CI status echo is implied by the tag; keep the remarks a human wrote
        closure = [(el, tx) for el, tx in rmks if closure_remark(tx)
                   and not (p['status'] in ('CI', 'CP') and tx.upper().startswith('(ARPT STATUS)'))]
        tag = death_tag(p, rmks, old_cyc.eff, converted=p['ident'] in new_idents)
        out.append(dict(p, site=site, tag=tag, facts=facts(p, old_cyc.eff),
                        response_age=years_since(p['last_info'], old_cyc.eff), closure_remarks=closure))
    rank = lambda t: next((i for i, k in enumerate(TAG_ORDER) if t.startswith(k)), len(TAG_ORDER))
    out.sort(key=lambda r: (rank(r['tag']), -(r['response_age'] or 0), r['ident']))
    return out


def print_removed(old_cyc, new_cyc, rows, width):
    if 'APT' not in old_cyc.data or 'APT' not in new_cyc.data:
        return
    n_ci = sum(1 for r in rows if r['status'] in ('CI', 'CP'))
    n_stale = sum(1 for r in rows if (r['response_age'] or 0) > 5)
    n_pu = sum(1 for r in rows if r['use'] == 'PU')
    print(f"\nREMOVED AIRPORTS  ({len(rows)}; last known state in {old_cyc.eff}: {n_ci} already CI/CP, "
          f"{n_stale} with no owner response in 5+ yrs, {n_pu} public use)")
    print("  tag: was CI = already closed; unverified Ny = owner never answered the 5010 mailout in N yrs "
          "(the purge); young = activated <5 yrs ago; converted = ident moved to a new site number")
    clip = lambda s: s if len(s) <= width else s[:width - 2] + '..'
    for r in rows:
        print(clip(f"  {head(r)}   [{r['tag']}]"))
        print(clip(f"        {r['facts']}"))
        for el, tx in r['closure_remarks']:
            print(clip(f"        rmk {el}: {tx}"))


def watch(old_cyc, new_cyc):
    """Airports still in NASR that are showing the pre-deletion signals.
    -> (pool counts, [(section, profile, what)])"""
    ob, orw = apt_index(old_cyc)
    nb, nrw = apt_index(new_cyc)
    orm, nrm = apt_remarks(old_cyc), apt_remarks(new_cyc)
    items = []
    pool = collections.Counter()
    pool_state = collections.Counter()
    for site, line in nb.items():
        p = apt_profile(line, nrw.get(site, []))
        age = years_since(p['last_info'], new_cyc.eff)
        if p['status'] in ('CI', 'CP'):
            pool['closed'] += 1
        elif p['use'] == 'PR' and age is None:
            pool['never'] += 1
        elif p['use'] == 'PR' and age > STALE_YEARS:
            pool['stale'] += 1
            pool_state[p['st']] += 1
        if site not in ob:
            continue
        q = apt_profile(ob[site], orw.get(site, []))
        if p['status'] in ('CI', 'CP') and q['status'] not in ('CI', 'CP'):
            items.append(('STATUS -> CLOSED', p, f"status {q['status'] or 'O'} -> {p['status']}"))
        # a name change only means sold / died where a private owner can do either;
        # at a city-owned field it is staff turnover
        if p['own'] == 'PR':
            what = []
            if p['owner'] != q['owner'] and not same_name(p['owner'], q['owner']):
                what.append(f"owner {q['owner'].title() or '(blank)'} -> {p['owner'].title() or '(blank)'}")
            if p['mgr'] != q['mgr'] and not same_name(p['mgr'], q['mgr']):
                what.append(f"mgr {q['mgr'].title() or '(blank)'} -> {p['mgr'].title() or '(blank)'}")
            if what:
                items.append(('OWNER / MANAGER CHANGED', p, '; '.join(what)))
        old_tx = {tx for _, tx in orm.get(site, [])}
        for el, tx in nrm.get(site, []):
            if tx not in old_tx and closure_remark(tx) and not (
                    p['status'] in ('CI', 'CP') and tx.upper().startswith('(ARPT STATUS)')):
                items.append(('NEW CLOSURE-FLAVOURED REMARK', p, f'{el}: {tx}'))
    # closures and remarks: the prominent airport first; ownership: the stalest record first
    stale = lambda p: years_since(p['last_info'], new_cyc.eff) or 0
    items.sort(key=lambda t: ((-stale(t[1]), t[1]['use'] != 'PU') if t[0] == 'OWNER / MANAGER CHANGED'
                              else (-t[1]['prominence'],)) + (t[1]['ident'],))
    pool['by_state'] = pool_state
    return pool, items


WATCH_SECTIONS = ['STATUS -> CLOSED', 'OWNER / MANAGER CHANGED', 'NEW CLOSURE-FLAVOURED REMARK']


def print_watch(old_cyc, new_cyc, pool, items, width):
    print(f"CLOSURE WATCH  {old_cyc.eff} -> {new_cyc.eff}   (airports still in NASR that are showing the pre-deletion signals)")
    print(f"at-risk pool in {new_cyc.eff}: {pool['closed']:,} rows closed (CI/CP); {pool['stale']:,} private-use rows "
          f"with no owner response in {STALE_YEARS}+ yrs (~2%/cycle deleted); {pool['never']:,} private-use with none on record")
    print("  stale pool by state: " + ', '.join(f'{st} {n}' for st, n in pool['by_state'].most_common(12)))
    clip = lambda s: s if len(s) <= width else s[:width - 2] + '..'
    for section in WATCH_SECTIONS:
        rows = [t for t in items if t[0] == section]
        print(f"\n{section} ({len(rows)})" + ('   -- privately owned fields only, stalest record first'
                                                if section == 'OWNER / MANAGER CHANGED' else ''))
        for _, p, what in rows:
            print(clip(f"  {head(p)}   {what}"))
            print(clip(f"        {facts(p, new_cyc.eff)}"))


NEWS_SECTIONS = ('BY THE NUMBERS', 'CLOSED / REMOVED', 'OPENED / NEW',
                 'CHANGES AT NOTABLE AIRPORTS', 'TOWERS', 'COMMUNICATIONS / WEATHER',
                 'INSTRUMENT LANDING SYSTEMS', 'NAVAIDS', 'AIRWAYS / ROUTES',
                 'AIRSPACE / ACTIVITY AREAS')


def news(old_cyc, new_cyc, events, minimum=NEWS_MIN):
    """Return a conservative newsletter shortlist, not a dump of all diffs.

    Airport-specific items must clear the prominence floor.  Infrastructure
    with value beyond one airport (navaids, weather, ATC communications and
    routes) does not.  Repeated record churn is paired, grouped, or suppressed.
    """
    ob, orw = apt_index(old_cyc)
    nb, nrw = apt_index(new_cyc)
    orm = apt_remarks(old_cyc)
    prof_cache = {}

    def prof(site):
        site = (site or '').strip()
        if site not in prof_cache:
            if site in nb:
                prof_cache[site] = apt_profile(nb[site], nrw.get(site, []))
            elif site in ob:
                prof_cache[site] = apt_profile(ob[site], orw.get(site, []))
            else:
                prof_cache[site] = None
        return prof_cache[site]

    out, skipped, seen = [], collections.Counter(), set()
    where = lambda p: f"{p['name'].title()} ({p['ident']}), {p['city']}, {p['st']}"
    shown = lambda v: v or '(none)'
    norm = lambda v: re.sub(r'[^A-Z0-9]', '', v.upper())

    def add_raw(rank, section, headline, detail=''):
        key = (section, headline)
        if key not in seen:
            seen.add(key)
            out.append((rank, section, headline, detail))

    def add(rank, section, p, what, extra='', gate=True):
        if p and gate and p['prominence'] < minimum:
            skipped[section] += 1
            return
        headline = f"{where(p)} -- {what}" if p else what
        detail = (describe(p) + ('. ' + extra if extra else '')) if p else extra
        add_raw(rank + (p['prominence'] if p else 0), section, headline, detail)

    def is_mass(n, total):
        return n >= MASS_MIN and total and n >= 0.2 * total

    apt_ev = collections.defaultdict(list)
    for e in events:
        if e['file'] == 'APT' and e['op'] in ('added', 'removed', 'changed'):
            apt_ev[e['entity']].append(e)

    ident_of = lambda site: (prof(site) or {}).get('ident')
    added_sites, removed_sites = set(nb) - set(ob), set(ob) - set(nb)
    added_idents = {ident_of(site): site for site in added_sites if ident_of(site)}
    removed_idents = {ident_of(site): site for site in removed_sites if ident_of(site)}
    converted = set(added_idents) & set(removed_idents)

    # A useful numerator survives the prominence filter: show the actual amount
    # of facility churn while keeping the dozens of private pads out of the list.
    if added_sites or removed_sites:
        pure_add = [prof(s) for s in added_sites if ident_of(s) not in converted]
        pure_rem = [prof(s) for s in removed_sites if ident_of(s) not in converted]

        def tally(rows):
            c = collections.Counter(p['typ'] for p in rows if p)
            plural = {'airport': 'airports', 'heliport': 'heliports',
                      'seaplane base': 'seaplane bases', 'gliderport': 'gliderports'}
            return ', '.join(f"{n} {t if n == 1 else plural.get(t, t + 's')}"
                             for t, n in sorted(c.items(), key=lambda x: (-x[1], x[0]))) or 'none'

        apu = sum(p['use'] == 'PU' for p in pure_add if p)
        rpu = sum(p['use'] == 'PU' for p in pure_rem if p)
        headline = (f"{len(pure_add)} landing facilities entered NASR ({apu} public-use); "
                    f"{len(pure_rem)} left ({rpu} public-use)")
        detail = f"Added: {tally(pure_add)}. Removed: {tally(pure_rem)}."
        if converted:
            detail += (f" {len(converted)} identifier{'s were' if len(converted) != 1 else ' was'} "
                       "re-established on a different site record: " + ', '.join(sorted(converted)) + '.')
        add_raw(1000, 'BY THE NUMBERS', headline, detail)

    for site, evs in apt_ev.items():
        p = prof(site)
        if not p:
            continue
        base = [e for e in evs if e['rt'] == 'APT']
        # Do not mistake every runway belonging to a newly added/deleted airport
        # for a runway project.
        rwy = [e for e in evs if e['rt'] == 'RWY'] if all(e['op'] == 'changed' for e in base) else []
        for e in base:
            if e['op'] == 'removed':
                if p['ident'] in converted:
                    continue
                rmks = orm.get(site, [])
                tail = f"[{death_tag(p, rmks, old_cyc.eff)}] {facts(p, old_cyc.eff, act=False)}"
                closure = [f'{el}: {tx}' for el, tx in rmks if closure_remark(tx)
                           and not tx.upper().startswith('(ARPT STATUS)')]
                if closure:
                    tail += '. rmk ' + ' | '.join(closure)
                add(100, 'CLOSED / REMOVED', p, 'removed from NASR' +
                    ('' if p['status'] == 'O' else f" (was already {p['status']})"), tail)
            elif e['op'] == 'added':
                old_site = removed_idents.get(p['ident'])
                if p['ident'] in converted and old_site != site:
                    q = prof(old_site)
                    add(60, 'CHANGES AT NOTABLE AIRPORTS', p,
                        f"re-established as {p['typ']} (was {q['typ']})")
                else:
                    add(100, 'OPENED / NEW', p, 'new in NASR')
            else:
                for name, a, b in e['fields']:
                    u = name.upper()
                    if u.startswith('AIRPORT STATUS CODE'):
                        if b in ('CI', 'CP') and a not in ('CI', 'CP'):
                            add(100, 'CLOSED / REMOVED', p,
                                f"status {shown(a)} -> {b} (closed {'indefinitely' if b == 'CI' else 'permanently'})")
                        elif a in ('CI', 'CP') and b == 'O':
                            add(90, 'OPENED / NEW', p, f'reopened (status {a} -> O)')
                    elif u.startswith('FACILITY USE'):
                        add(80, 'CHANGES AT NOTABLE AIRPORTS', p,
                            'opened to public use' if b == 'PU' else 'closed to public use (now private)')
                    elif u.startswith('OFFICIAL FACILITY NAME'):
                        if norm(a) != norm(b):
                            add(30, 'CHANGES AT NOTABLE AIRPORTS', p,
                                f"renamed: {a.title()} -> {b.title()}")
                    elif u.startswith('LOCATION IDENTIFIER'):
                        add(40, 'CHANGES AT NOTABLE AIRPORTS', p,
                            f"identifier {shown(a)} -> {shown(b)}")
                    elif u.startswith('LANDING FACILITY TYPE'):
                        add(60, 'CHANGES AT NOTABLE AIRPORTS', p,
                            f"facility type {shown(a).lower()} -> {shown(b).lower()}")
                    elif u.startswith('ICAO IDENTIFIER'):
                        add(35, 'CHANGES AT NOTABLE AIRPORTS', p,
                            f"ICAO identifier {shown(a)} -> {shown(b)}")
                    elif u.startswith('AIR TRAFFIC CONTROL TOWER'):
                        if (a == 'Y') != (b == 'Y'):
                            add(80, 'CHANGES AT NOTABLE AIRPORTS', p,
                                'control tower ' + ('established' if b == 'Y' else 'closed'))
                    elif u.startswith('AIRPORT ARFF CERTIFICATION'):
                        m = re.search(r'(\d\d)/(\d{4})$', b)
                        if b and m and int(m.group(2)) < int(new_cyc.eff[:4]) - 1:
                            continue        # old certificate backfill, not current news
                        add(60, 'CHANGES AT NOTABLE AIRPORTS', p,
                            f"Part 139 certificate {'issued: ' + b if b else 'dropped (was ' + a + ')'}")
                    elif u.startswith('AIRPORT OWNERSHIP TYPE'):
                        add(40, 'CHANGES AT NOTABLE AIRPORTS', p,
                            f"ownership {shown(a)} -> {shown(b)}")
                    elif u.startswith('FUEL TYPES AVAILABLE') and re.sub(r'\s+', '', a) != re.sub(r'\s+', '', b):
                        add(45, 'CHANGES AT NOTABLE AIRPORTS', p,
                            f"public fuel {shown(a)} -> {shown(b)}")
                    elif u.startswith(('UNICOM FREQUENCY', 'COMMON TRAFFIC ADVISORY FREQUENCY')):
                        add(45, 'CHANGES AT NOTABLE AIRPORTS', p,
                            f"{'CTAF' if 'TRAFFIC' in u else 'UNICOM'} {shown(a)} -> {shown(b)}")
                    elif u.startswith('NPIAS/FEDERAL AGREEMENTS CODE'):
                        add(35, 'CHANGES AT NOTABLE AIRPORTS', p,
                            f"NPIAS/federal-agreement code {shown(a)} -> {shown(b)}")
                    elif u.startswith('FACILITY HAS MILITARY/CIVIL JOINT USE') and (a == 'Y') != (b == 'Y'):
                        add(60, 'CHANGES AT NOTABLE AIRPORTS', p,
                            'military/civil joint use ' + ('established' if b == 'Y' else 'ended'))
                    elif u.startswith('LANDING FEE CHARGED') and (a == 'Y') != (b == 'Y'):
                        add(25, 'CHANGES AT NOTABLE AIRPORTS', p,
                            'landing fee ' + ('introduced' if b == 'Y' else 'removed'))

        # A removed+added pair at an existing airport is normally magnetic
        # redesignation, not a runway closure and construction in one cycle.
        r_add = [e for e in rwy if e['op'] == 'added']
        r_rem = [e for e in rwy if e['op'] == 'removed']
        if r_add and r_rem and len(r_add) == len(r_rem):
            for x, y in zip(sorted(r_rem, key=lambda e: e['sub']), sorted(r_add, key=lambda e: e['sub'])):
                add(20, 'CHANGES AT NOTABLE AIRPORTS', p,
                    f"runway {x['sub'][2:]} redesignated {y['sub'][2:]}")
        else:
            for e in r_add:
                add(50, 'CHANGES AT NOTABLE AIRPORTS', p,
                    f"new runway {e['sub'][2:]}: {e['snippet'][:40]}")
            for e in r_rem:
                add(50, 'CHANGES AT NOTABLE AIRPORTS', p,
                    f"runway {e['sub'][2:]} removed: {e['snippet'][:40]}")
        for e in rwy:
            if e['op'] != 'changed':
                continue
            rid = e['sub'][2:]
            for name, a, b in e['fields']:
                u = name.upper()
                if u.startswith('PHYSICAL RUNWAY LENGTH') and a.isdigit() and b.isdigit() \
                        and abs(int(a) - int(b)) >= 500:
                    add(50, 'CHANGES AT NOTABLE AIRPORTS', p,
                        f"runway {rid} {'extended' if int(b) > int(a) else 'shortened'} "
                        f"{int(a):,} -> {int(b):,} ft")
                elif u.startswith('RUNWAY SURFACE TYPE'):
                    pa = a.upper().startswith(('ASPH', 'CONC', 'PEM'))
                    pb = b.upper().startswith(('ASPH', 'CONC', 'PEM'))
                    if pa != pb:
                        add(50, 'CHANGES AT NOTABLE AIRPORTS', p,
                            f"runway {rid} {'paved' if pb else 'no longer listed as paved'} ({shown(a)} -> {shown(b)})")
                elif u.startswith('RUNWAY LIGHTS EDGE INTENSITY') and a != b \
                        and ({a, b} & {'LOW', 'MED', 'HIGH'}):
                    add(35, 'CHANGES AT NOTABLE AIRPORTS', p,
                        f"runway {rid} edge lighting {shown(a)} -> {shown(b)}")
                elif u.startswith('APPROACH LIGHT SYSTEM') and a != b:
                    add(40, 'CHANGES AT NOTABLE AIRPORTS', p,
                        f"runway {rid} approach lights {shown(a)} -> {shown(b)}")
                elif u.startswith('RUNWAY END IDENTIFIER LIGHTS') and (a == 'Y') != (b == 'Y'):
                    add(40, 'CHANGES AT NOTABLE AIRPORTS', p,
                        f"runway {rid} {'gained' if b == 'Y' else 'lost'} REIL")
                elif u.startswith('RUNWAY CENTERLINE LIGHTS') and (a == 'Y') != (b == 'Y'):
                    add(40, 'CHANGES AT NOTABLE AIRPORTS', p,
                        f"runway {rid} {'gained' if b == 'Y' else 'lost'} centerline lights")
                elif u.startswith('RUNWAY END TOUCHDOWN LIGHTS') and (a == 'Y') != (b == 'Y'):
                    add(40, 'CHANGES AT NOTABLE AIRPORTS', p,
                        f"runway {rid} {'gained' if b == 'Y' else 'lost'} touchdown-zone lights")
                elif u.startswith('INSTRUMENT LANDING SYSTEM (ILS) TYPE') and a != b:
                    add(45, 'CHANGES AT NOTABLE AIRPORTS', p,
                        f"runway {rid} ILS capability {shown(a)} -> {shown(b)}")

    # Towers: only genuine ATCT/TRACON records.  The TWR file also carries many
    # NON-ATCT airport communication listings which must not become openings.
    tower_events = []
    for e in events:
        if e['file'] != 'TWR' or e['rt'] != 'TWR1':
            continue
        cyc = old_cyc if e['op'] == 'removed' else new_cyc
        line = cyc.data.get('TWR', {}).get('TWR1' + e['entity'].ljust(4), [''])[0]
        if not line:
            continue
        ftype = line[238:250].strip()
        if not (ftype.startswith('ATCT') or 'TRACON' in ftype):
            continue
        tower_events.append((e, line))

    # A tower record is keyed by airport ident.  Pair an ident change at the
    # same airport site instead of claiming that one tower closed and another
    # opened.  APT already reports the ident change when that file is loaded.
    tower_site_ops = collections.defaultdict(lambda: {'added': [], 'removed': []})
    for e, line in tower_events:
        if e['op'] in ('added', 'removed') and line[18:29].strip():
            tower_site_ops[line[18:29].strip()][e['op']].append((e, line))
    tower_rekeys = set()
    for site, d in tower_site_ops.items():
        if len(d['added']) == len(d['removed']) == 1:
            old_e, _ = d['removed'][0]
            new_e, line = d['added'][0]
            tower_rekeys.update((('removed', old_e['entity']), ('added', new_e['entity'])))
            if 'APT' not in old_cyc.data or 'APT' not in new_cyc.data:
                place = f"{line[104:154].strip().title()}, {line[62:64].strip()}"
                add_raw(60, 'TOWERS', f"{place} -- tower identifier {old_e['entity']} -> {new_e['entity']}")

    for e, line in tower_events:
        p = prof(line[18:29])
        place = where(p) if p else f"{line[104:154].strip().title()}, {line[62:64].strip()}"
        ftype = line[238:250].strip()
        if e['op'] in ('added', 'removed'):
            if (e['op'], e['entity']) in tower_rekeys:
                continue
            add_raw(80 + (p['prominence'] if p else 0), 'TOWERS',
                    f"{place} -- {ftype} {e['entity']} {'opened' if e['op'] == 'added' else 'closed'}",
                    describe(p) if p else '')
        elif e['op'] == 'changed':
            hours = [(a, b) for name, a, b in e['fields']
                     if name.upper().startswith('NUMBER OF HOURS OF DAILY OPERATION')
                     and a.isdigit() and b.isdigit() and abs(int(a) - int(b)) >= 4]
            if hours:
                a, b = hours[0]
                add_raw(55 + (p['prominence'] if p else 0), 'TOWERS',
                        f"{place} -- tower hours changed {a} -> {b} hours/day", describe(p) if p else '')
            for name, a, b in e['fields']:
                if name.upper().startswith('FACILITY TYPE') and a != b:
                    add_raw(65 + (p['prominence'] if p else 0), 'TOWERS',
                            f"{place} -- facility type {shown(a)} -> {shown(b)}", describe(p) if p else '')

    # Whole ILS systems, then material component changes.  Positional surveys,
    # operator names and remark rewrites stay out.
    ils_ev = collections.defaultdict(set)
    for e in events:
        if e['file'] == 'ILS' and e['rt'] == 'ILS1' and e['op'] in ('added', 'removed'):
            ils_ev[e['entity']].add(e['op'])

    ils_labels = entity_labels(old_cyc, 'ILS') | entity_labels(new_cyc, 'ILS')

    def add_ils(rank, site, what, ent=''):
        p = prof(site)
        if p:
            add(rank, 'INSTRUMENT LANDING SYSTEMS', p, what)
        else:
            label = ils_labels.get(ent, '')
            add_raw(rank, 'INSTRUMENT LANDING SYSTEMS', f"{label} -- {what}" if label else what)

    ils_type_ops = collections.defaultdict(lambda: {'added': [], 'removed': []})
    for ent, ops in ils_ev.items():
        if len(ops) != 1:
            continue
        site, rwy, typ = ent[:11].strip(), ent[11:14].strip(), ent[14:].strip()
        ils_type_ops[(site, rwy)][next(iter(ops))].append((ent, typ))
    ils_type_pairs = set()
    for (site, rwy), d in ils_type_ops.items():
        if len(d['added']) == len(d['removed']) == 1:
            old_ent, old_type = d['removed'][0]
            new_ent, new_type = d['added'][0]
            if old_type != new_type:
                ils_type_pairs.update((old_ent, new_ent))
                add_ils(40, site, f"rwy {rwy} landing-system type {old_type} -> {new_type}", new_ent)

    by_ils = collections.defaultdict(lambda: {'added': [], 'removed': []})
    for ent, ops in ils_ev.items():
        if len(ops) != 1 or ent in ils_type_pairs:
            continue                        # remove+add means rewritten in place
        site, rwy, typ = ent[:11].strip(), ent[11:14].strip(), ent[14:].strip()
        by_ils[(site, typ)][next(iter(ops))].append(rwy)
    for (site, typ), d in by_ils.items():
        rem, new_ = sorted(d['removed']), sorted(d['added'])
        if rem and new_ and len(rem) == len(new_):
            for a, b in zip(rem, new_):
                add_ils(10, site, f"{typ} rwy {a} redesignated {b}")
            continue
        for rwy in rem:
            add_ils(30, site, f"{typ} rwy {rwy} decommissioned")
        for rwy in new_:
            add_ils(30, site, f"{typ} rwy {rwy} commissioned")

    component = {'ILS2': 'localizer', 'ILS3': 'glide slope', 'ILS4': 'DME', 'ILS5': 'marker beacon'}
    for e in events:
        if e['file'] != 'ILS' or e['rt'] not in component or e['entity'] in ils_ev:
            continue
        site, rwy, typ = e['entity'][:11].strip(), e['entity'][11:14].strip(), e['entity'][14:].strip()
        comp = component[e['rt']]
        if e['rt'] == 'ILS5' and e['sub']:
            comp = {'IM': 'inner marker', 'MM': 'middle marker', 'OM': 'outer marker'}.get(e['sub'], comp)
        if e['op'] in ('added', 'removed'):
            add_ils(35, site, f"{typ} rwy {rwy} {comp} "
                    f"{'added' if e['op'] == 'added' else 'removed'}", e['entity'])
            continue
        status = [(a, b) for name, a, b in e['fields']
                  if (name.upper().startswith('OPERATIONAL STATUS')
                      or name.upper().startswith('LOW POWERED NDB STATUS')) and a != b]
        freq = [(a, b) for name, a, b in e['fields']
                if ('FREQUENCY' in name.upper() or 'CHANNEL' in name.upper()) and a != b]
        if status:
            a, b = status[0]
            add_ils(45, site, f"{typ} rwy {rwy} {comp} status {shown(a)} -> {shown(b)}", e['entity'])
        if freq:
            a, b = freq[0]
            add_ils(35, site, f"{typ} rwy {rwy} {comp} frequency/channel "
                    f"{shown(a)} -> {shown(b)}", e['entity'])

    # Navaid bases only.  Sub-record fix lists, holding patterns and restriction
    # prose account for most NAV churn and are deliberately ignored here.
    nav_events = collections.defaultdict(list)
    for e in events:
        if e['file'] != 'NAV' or e['rt'] != 'NAV1':
            continue
        cyc = old_cyc if e['op'] == 'removed' else new_cyc
        line = cyc.data.get('NAV', {}).get('NAV1' + e['entity'].ljust(24), [''])[0]
        if line:
            nav_events[line[4:8].strip()].append((e, line))
    for ident, rows in nav_events.items():
        added = [(e, l) for e, l in rows if e['op'] == 'added']
        removed = [(e, l) for e, l in rows if e['op'] == 'removed']
        paired = len(added) == len(removed) == 1
        if paired:
            a_line, b_line = removed[0][1], added[0][1]
            a_typ, b_typ = a_line[8:28].strip(), b_line[8:28].strip()
            if a_typ != b_typ:
                name, st = b_line[42:72].strip().title(), b_line[142:144].strip()
                add_raw(90, 'NAVAIDS', f"{name} ({ident}), {st} -- facility type {a_typ} -> {b_typ}")
        for e, line in rows:
            typ, name, st = line[8:28].strip(), line[42:72].strip().title(), line[142:144].strip()
            title = f"{name} {typ} ({ident}), {st}"
            vor = any(t in typ for t in ('VOR', 'TACAN', 'DME'))
            if e['op'] in ('added', 'removed'):
                if paired:
                    continue
                rank = (80 if vor else 40) + (30 if e['op'] == 'removed' else 20)
                add_raw(rank, 'NAVAIDS',
                        f"{title} -- {'new in' if e['op'] == 'added' else 'removed from'} NASR")
                continue
            for fname, a, b in e['fields']:
                u = fname.upper()
                if u.startswith('NAVIGATION AID STATUS') and a != b:
                    add_raw(90 if vor else 55, 'NAVAIDS',
                            f"{title} -- status {shown(a)} -> {shown(b)}")
                elif u.startswith('NAME OF NAVAID') and norm(a) != norm(b):
                    add_raw(55, 'NAVAIDS', f"{title} -- renamed {a.title()} -> {b.title()}")
                elif u.startswith('NAVAID PUBLIC USE') and (a == 'Y') != (b == 'Y'):
                    add_raw(60, 'NAVAIDS',
                            f"{title} -- {'opened to' if b == 'Y' else 'closed to'} public use")
            freq = [(name, a, b) for name, a, b in e['fields']
                    if ('FREQUENCY' in name.upper() or 'CHANNEL' in name.upper()) and a != b]
            if freq:
                changes = ', '.join(f"{shown(a)} -> {shown(b)}" for _, a, b in freq)
                add_raw(70 if vor else 45, 'NAVAIDS', f"{title} -- frequency/channel {changes}")

    # Weather sensors are keyed by ident+type in the source.  Pair by ident so
    # AWOS-3 -> AWOS-3P is one upgrade rather than a removal plus an opening.
    def awos_index(cyc):
        d = {}
        for line in records_of(cyc, 'AWOS'):
            if not line.startswith('AWOS1'):
                continue
            ident = line[5:9].strip()
            d[ident] = {'ident': ident, 'typ': line[9:19].strip(), 'status': line[19:20].strip(),
                        'date': line[20:30].strip(), 'freq': tuple(x for x in (line[68:75].strip(),
                        line[75:82].strip()) if x), 'site': line[110:121].strip(),
                        'city': line[121:161].strip().title(), 'st': line[161:163].strip()}
        return d

    def weather_place(w):
        p = prof(w['site'])
        return where(p) if p else f"{w['city'] or w['ident']}, {w['st']}"

    def canon_freqs(freqs):
        vals = []
        for v in freqs:
            try:
                vals.append(f'{float(v):g}')
            except ValueError:
                vals.append(v)
        return tuple(vals)

    oaw, naw = awos_index(old_cyc), awos_index(new_cyc)
    aw_add, aw_rem = set(naw) - set(oaw), set(oaw) - set(naw)

    # Sensor ident usually follows the airport ident.  Pair a simultaneous
    # re-identification by the stable landing-facility site number.
    old_sites, new_sites = collections.defaultdict(list), collections.defaultdict(list)
    for ident in aw_rem:
        if oaw[ident]['site']:
            old_sites[oaw[ident]['site']].append(ident)
    for ident in aw_add:
        if naw[ident]['site']:
            new_sites[naw[ident]['site']].append(ident)
    aw_rekeys = []
    for site in set(old_sites) & set(new_sites):
        if len(old_sites[site]) == len(new_sites[site]) == 1:
            old_ident, new_ident = old_sites[site][0], new_sites[site][0]
            aw_rekeys.append((old_ident, new_ident, oaw[old_ident], naw[new_ident]))
            aw_rem.remove(old_ident)
            aw_add.remove(new_ident)

    for old_ident, new_ident, a, b in aw_rekeys:
        place = weather_place(b)
        if a['typ'] != b['typ']:
            add_raw(50, 'COMMUNICATIONS / WEATHER',
                    f"{place} -- weather sensor {old_ident} {a['typ']} -> {new_ident} {b['typ']}")
        elif 'APT' not in old_cyc.data or 'APT' not in new_cyc.data:
            add_raw(35, 'COMMUNICATIONS / WEATHER',
                    f"{place} -- {b['typ']} identifier {old_ident} -> {new_ident}")
        if a['status'] != b['status']:
            state = {'Y': 'commissioned', 'N': 'decommissioned'}.get(b['status'], shown(b['status']))
            add_raw(60, 'COMMUNICATIONS / WEATHER', f"{place} -- {new_ident} {b['typ']} {state}")
        if canon_freqs(a['freq']) != canon_freqs(b['freq']):
            add_raw(45, 'COMMUNICATIONS / WEATHER',
                    f"{place} -- {new_ident} {b['typ']} frequency "
                    f"{shown('/'.join(canon_freqs(a['freq'])))} -> {shown('/'.join(canon_freqs(b['freq'])))}")

    def weather_presence(idents, data, op, total):
        if not idents or is_mass(len(idents), total):
            return
        if len(idents) <= 5:
            for ident in sorted(idents):
                w = data[ident]
                add_raw(55, 'COMMUNICATIONS / WEATHER',
                        f"{weather_place(w)} -- {ident} {w['typ']} {op} NASR")
        else:
            names = [f"{ident} ({data[ident]['city']})" for ident in sorted(idents)]
            tail = ', '.join(names[:12]) + (f", and {len(names) - 12} more" if len(names) > 12 else '')
            add_raw(60, 'COMMUNICATIONS / WEATHER', f"{len(idents)} weather sensors {op} NASR", tail)

    weather_presence(aw_add, naw, 'added to', len(naw))
    weather_presence(aw_rem, oaw, 'removed from', len(oaw))
    type_changes = collections.defaultdict(list)
    for ident in sorted(set(oaw) & set(naw)):
        if oaw[ident]['typ'] != naw[ident]['typ']:
            type_changes[(oaw[ident]['typ'], naw[ident]['typ'])].append(ident)
    for (old_type, new_type), idents in type_changes.items():
        if len(idents) >= 4:
            names = ', '.join(f"{ident} ({naw[ident]['city']})" for ident in idents)
            add_raw(55, 'COMMUNICATIONS / WEATHER',
                    f"{len(idents)} weather sensors changed type {old_type} -> {new_type}", names)
        else:
            for ident in idents:
                add_raw(50, 'COMMUNICATIONS / WEATHER',
                        f"{weather_place(naw[ident])} -- {ident} weather sensor type {old_type} -> {new_type}")
    weather_freq = collections.defaultdict(list)
    for ident in sorted(set(oaw) & set(naw)):
        a, b = oaw[ident], naw[ident]
        place = weather_place(b)
        if a['status'] != b['status']:
            state = {'Y': 'commissioned', 'N': 'decommissioned'}.get(b['status'], shown(b['status']))
            add_raw(60, 'COMMUNICATIONS / WEATHER', f"{place} -- {ident} {b['typ']} {state}")
        if a['freq'] != b['freq']:
            weather_freq[(place, canon_freqs(a['freq']), canon_freqs(b['freq']))].append((ident, b['typ']))
    for (place, old_freq, new_freq), stations in weather_freq.items():
        ident = '/'.join(i for i, _ in stations)
        typ = stations[0][1] if len(stations) == 1 else 'weather sensor'
        suffix = ' records' if len(stations) > 1 else ''
        add_raw(45, 'COMMUNICATIONS / WEATHER',
                f"{place} -- {ident} {typ}{suffix} frequency "
                f"{shown('/'.join(old_freq))} -> {shown('/'.join(new_freq))}")

    # RCOs are also paired by ident.  A moderate batch becomes one useful line;
    # a >=20% mass rewrite is already represented as a systemic event and is
    # omitted from the newsletter entirely.
    def com_index(cyc):
        d = {}
        for line in records_of(cyc, 'COM'):
            ident = line[:4].strip()
            city = line[117:143].strip() or line[17:43].strip()
            state = line[143:163].strip() or line[43:63].strip()
            d[ident] = {'ident': ident, 'typ': line[4:11].strip(), 'city': city.title(),
                        'state': state.title(), 'call': line[214:240].strip().title(),
                        'freq': tuple(sorted(set(re.findall(r'\d{3}\.\d{1,3}[A-Z]?', line[240:384])))),
                        'status': line[662:682].strip()}
        return d

    def com_place(c):
        return f"{c['call'] or c['city'] or c['ident']}, {c['state']}".rstrip(', ')

    oc, nc = com_index(old_cyc), com_index(new_cyc)

    def com_presence(idents, data, op, total):
        if not idents or is_mass(len(idents), total):
            return
        verb = 'added to' if op == 'added' else 'removed from'
        if len(idents) <= 5:
            for ident in sorted(idents):
                c = data[ident]
                add_raw(45, 'COMMUNICATIONS / WEATHER',
                        f"{com_place(c)} -- {ident} {c['typ']} {verb} NASR")
        else:
            names = [f"{ident} ({data[ident]['city'] or data[ident]['call']})" for ident in sorted(idents)]
            tail = ', '.join(names[:12]) + (f", and {len(names) - 12} more" if len(names) > 12 else '')
            add_raw(50, 'COMMUNICATIONS / WEATHER',
                    f"{len(idents)} remote communications outlets {verb} NASR", tail)

    com_presence(set(nc) - set(oc), nc, 'added', len(nc))
    com_presence(set(oc) - set(nc), oc, 'removed', len(oc))
    for ident in sorted(set(oc) & set(nc)):
        a, b = oc[ident], nc[ident]
        place = com_place(b)
        if a['typ'] != b['typ']:
            add_raw(45, 'COMMUNICATIONS / WEATHER',
                    f"{place} -- {ident} outlet type {a['typ']} -> {b['typ']}")
        if a['status'] != b['status']:
            add_raw(55, 'COMMUNICATIONS / WEATHER',
                    f"{place} -- {ident} {b['typ']} status {shown(a['status'])} -> {shown(b['status'])}")
        if a['freq'] != b['freq']:
            add_raw(45, 'COMMUNICATIONS / WEATHER',
                    f"{place} -- {ident} {b['typ']} frequencies "
                    f"{shown('/'.join(a['freq']))} -> {shown('/'.join(b['freq']))}")

    # Report only routes that wholly enter or leave the file; point and altitude
    # edits within a surviving route are publication churn, not a clean headline.
    for f in ('AWY', 'ATS'):
        if f not in old_cyc.data or f not in new_cyc.data:
            continue
        oe = set(entity_lines(old_cyc, f))
        ne = set(entity_lines(new_cyc, f))
        labs = entity_labels(old_cyc, f) | entity_labels(new_cyc, f)
        noun = 'airways' if f == 'AWY' else 'ATS routes'
        for op, changed, total in (('added to', ne - oe, len(ne)), ('removed from', oe - ne, len(oe))):
            if is_mass(len(changed), total):
                continue
            if len(changed) > 10:
                names = ', '.join((labs.get(x) or x) for x in sorted(changed)[:10])
                add_raw(50, 'AIRWAYS / ROUTES', f"{len(changed)} {noun} {op} NASR", names + ', ...')
            else:
                for ent in sorted(changed):
                    add_raw(50, 'AIRWAYS / ROUTES', f"{labs.get(ent) or ent} {op} NASR")

    # New/deleted parachute, aerobatic/glider and military-training areas are
    # sparse and meaningful.  Ignore their contact, remark and point churn.
    def activity_title(f, line):
        if f == 'MAA':
            code, typ = line[4:10].strip(), line[10:35].strip()
            name = line[300:420].strip().title() or typ.title()
            airport = line[227:277].strip().title()
            if airport and norm(name) == norm(typ):
                kind = {'APA': 'aerobatic practice area', 'GLIDER': 'glider area',
                        'OTHER': 'activity area'}.get(typ.upper(), typ.lower() + ' area')
                name = f'{airport} {kind}'
            place = ', '.join(x for x in (line[140:170].strip().title(), line[108:110].strip()) if x)
            return f"{name} ({code})" + (f", {place}" if place else '')
        if f == 'PJA':
            code = line[4:10].strip()
            name = line[261:311].strip().title() or 'Parachute jump area'
            airport = line[200:250].strip().title()
            if airport and re.fullmatch(r'[A-Z0-9]{2,5}', name.upper()):
                name = f'{airport} jump area'
            place = ', '.join(x for x in (line[117:147].strip().title(), line[85:87].strip()) if x)
            return f"{name} ({code})" + (f", {place}" if place else '')
        return entity_label('MTR', 'MTR1', line) or entity_id('MTR', 'MTR1', line).strip()

    activity_fields = {
        'MAA': ('MAA TYPE', 'MAA AREA NAME', 'MAA MAXIMUM ALTITUDE', 'MAA MINIMUM ALTITUDE',
                'MAA AREA RADIUS', 'SHOW ON VFR CHART', 'MAA USE'),
        'PJA': ('PJA DROP ZONE NAME', 'PJA MAXIMUM ALTITUDE', 'PJA AREA RADIUS',
                'SECTIONAL CHARTING REQUIRED', 'PJA USE'),
        'MTR': (),
    }
    for f in ('MAA', 'PJA', 'MTR'):
        rt = BASE_RT[f]
        old_base, new_base = entity_lines(old_cyc, f, rt), entity_lines(new_cyc, f, rt)
        for op, changed, data, total in (('new in', set(new_base) - set(old_base), new_base, len(new_base)),
                                         ('removed from', set(old_base) - set(new_base), old_base, len(old_base))):
            if is_mass(len(changed), total):
                continue
            for ent in sorted(changed):
                add_raw(45, 'AIRSPACE / ACTIVITY AREAS',
                        f"{activity_title(f, data[ent])} -- {op} NASR")
        for e in events:
            if e['file'] != f or e['rt'] != rt or e['op'] != 'changed':
                continue
            changes = [(name, a, b) for name, a, b in e['fields']
                       if any(name.upper().startswith(k) for k in activity_fields[f]) and a != b]
            if not changes:
                continue
            line = new_base.get(e['entity']) or old_base.get(e['entity'])

            def activity_change(name, a, b):
                u = name.upper()
                label = ('maximum altitude' if 'MAXIMUM ALTITUDE' in u else
                         'minimum altitude' if 'MINIMUM ALTITUDE' in u else
                         'radius' if 'AREA RADIUS' in u else
                         'sectional charting' if 'CHART' in u else
                         'name' if 'AREA NAME' in u or 'DROP ZONE NAME' in u else
                         'type' if u.endswith('TYPE') else 'use')
                def value(v):
                    m = re.match(r'^(\d+)(MSL|AGL)$', v)
                    if m:
                        return f"{int(m.group(1)):,} {m.group(2)}"
                    if label == 'radius' and v.replace('.', '', 1).isdigit():
                        return f'{v} NM'
                    return shown(v)
                return f"{label} {value(a)} -> {value(b)}"

            what = '; '.join(activity_change(name, a, b) for name, a, b in changes[:3])
            add_raw(35, 'AIRSPACE / ACTIVITY AREAS', f"{activity_title(f, line)} -- {what}")

    out.sort(key=lambda t: -t[0])
    return out, skipped


def print_news(old_cyc, new_cyc, items, skipped, minimum):
    print(f"NEWSLETTER CANDIDATES  {old_cyc.eff} -> {new_cyc.eff}")
    print(f"airport-specific items shown only if prominence >= {minimum} (public use = 40; tower 40, Part 139 50, "
          f"NPIAS 20, based aircraft, ops, runway length; heliports x0.3)")
    for section in NEWS_SECTIONS:
        rows = [t for t in items if t[1] == section]
        if not rows and not skipped.get(section):
            continue
        print(f"\n{section}")
        for rank, _, head, detail in rows:
            print(f"  - {head}")
            if detail:
                print(f"      {detail}")
        if skipped.get(section):
            print(f"  ({skipped[section]} below the bar -- --news-min 0 to see them)")


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('old', nargs='?', help='older effective date (default: second newest)')
    ap.add_argument('new', nargs='?', help='newer effective date (default: newest)')
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--files', help='comma list, default all 21 .txt files')
    ap.add_argument('--top', type=int, default=40)
    ap.add_argument('--per-entity', type=int, default=4)
    ap.add_argument('--width', type=int, default=130, help='report line width')
    ap.add_argument('--json', help='write every event (or every series row) here')
    ap.add_argument('--series', action='store_true', help='diff every consecutive pair of cycles')
    ap.add_argument('--since', help='--series: first cycle to include')
    ap.add_argument('--show-layout', metavar='FILE', help='dump parsed layout sections and exit')
    ap.add_argument('--news', action='store_true',
                    help='newsletter sift: high-signal facility, infrastructure, route and activity-area changes')
    ap.add_argument('--news-min', type=int, default=NEWS_MIN, help='airport prominence floor for --news')
    ap.add_argument('--watch', action='store_true',
                    help='closure watch: airports still in NASR showing the pre-deletion signals')
    args = ap.parse_args()

    ROOT = args.root
    files = [f.strip().upper() for f in args.files.split(',')] if args.files else FILES
    if args.news and not args.files:
        files = ['APT', 'TWR', 'ILS', 'NAV', 'AWOS', 'COM', 'AWY', 'ATS', 'MAA', 'PJA', 'MTR']
    elif args.watch and not args.files:
        files = ['APT']
    cycles = sorted(d for d in os.listdir(ROOT) if re.match(r'\d{4}-\d\d-\d\d$', d)
                    and os.path.exists(zip_path(d)))
    if len(cycles) < 2:
        sys.exit(f'need two cycles under {ROOT}')

    if args.show_layout:
        f = args.show_layout.upper()
        cyc = load_cycle(args.new or cycles[-1], [], [], quiet=True)
        z = zipfile.ZipFile(zip_path(cyc.eff))
        names = {n.upper(): n for n in z.namelist()}
        text = z.read(names[f'LAYOUT_DATA/{f}_RF.TXT']).decode('latin-1')
        for label, fields in parse_layout(text):
            print(f'[{label}] {len(fields)} fields')
            for start, length, name in fields:
                print(f'   {start:5} {length:5}  {name}')
        return

    if args.series:
        sel = [c for c in cycles if not args.since or c >= args.since]
        series(sel, files, cycles, args.json)
        return

    old, new = args.old or cycles[-2], args.new or cycles[-1]
    for c in (old, new):
        if c not in cycles:
            sys.exit(f'no cycle {c}; have {cycles[0]} .. {cycles[-1]}')
    masks = masks_for(cycles, [old, new])
    old_cyc = load_cycle(old, files, masks)
    new_cyc = load_cycle(new, files, masks)
    if args.watch:
        pool, items = watch(old_cyc, new_cyc)
        print_watch(old_cyc, new_cyc, pool, items, args.width)
        if args.json:
            json.dump({'old': old, 'new': new, 'pool': {**pool, 'by_state': dict(pool['by_state'])},
                       'items': [dict(section=sec, what=what, **p) for sec, p, what in items]},
                      open(args.json, 'w'), indent=1)
        return
    events, notes = diff_cycles(old_cyc, new_cyc, files)
    events.sort(key=lambda e: -e['score'])
    if args.news:
        items, skipped = news(old_cyc, new_cyc, events, args.news_min)
        print_news(old_cyc, new_cyc, items, skipped, args.news_min)
        if args.json:
            json.dump([dict(rank=r, section=sec, headline=h, detail=d) for r, sec, h, d in items],
                      open(args.json, 'w'), indent=1)
        return
    print_report(old_cyc, new_cyc, events, notes, args.top, args.per_entity, args.width)
    if args.json:
        json.dump({'old': old, 'new': new, 'layout_notes': notes, 'events': events,
                   'entities': roll_up(events),
                   'removed_airports': removed_airports(old_cyc, new_cyc) if 'APT' in files else []},
                  open(args.json, 'w'), indent=1)
        print(f'\nwrote {args.json}  ({len(events)} events)')


if __name__ == '__main__':
    main()
