#!/usr/bin/env python3
"""Pull the Tier-1 structured aeronautical datasets into /Volumes/projects/aisdata/.

Sibling of rawtiffs/ for *vector / structured* AIS data (the raw material for the
time-aware overlays: aerodromes, navaids, airspace, routes, obstacles).  Every
source gets its own directory, a README.txt the script maintains, and a
manifest.jsonl with one line per fetched file (url, bytes, sha256, fetched,
source-reported date).  Snapshots are kept per date; a file is re-fetched only
when the remote size/checksum differs, so the tree accumulates a time series.

Sources (all free; see the README each writes):

  fr_sia    France, SIA aeronautical database export (AIXM 4.5 + SIA XML), one
            zip per AIRAC cycle.
              * historical 2019-2023: mirrored from data.cquest.org/dgac/aip/
                (Christian Quest's Licence Ouverte mirror, incl. GeoJSON conversions)
              * current cycles: the SIA shop lists them at 0 EUR but delivers
                through a (free) customer account -- download them yourself and
                drop the zip in ~/Downloads (or --sia-drop DIR); this script files
                it under fr_sia/cycles/<effective-date>/ and records it.
              The script also scrapes the shop index so the README lists which
              cycles exist upstream and which are still missing here.
  ch_bazl   Switzerland, FOCA (BAZL) open government data via the geo.admin.ch
            STAC API: aerodromes/heliports, mountain landing sites, air
            navigation obstacles (AIXM), UAS zones, plus the ICAO 1:500k and
            glider chart rasters (COG GeoTIFF).
  br_anac   Brazil, ANAC open data: the aerodrome register CSVs (public, private,
            *deleted* aerodromes, runways, aprons, heliports) crawled from
            sistemas.anac.gov.br/dadosabertos/Aerodromos/.
  br_geoaisweb  Brazil, DECEA GeoAISWEB WFS (geoaisweb.decea.mil.br/geoserver):
            airports, heliports, runways, VOR/NDB/DME, waypoints, airways, FIRs,
            TMAs, CTAs, CTRs, ATZs, restricted/prohibited/danger areas, as
            GeoJSON.  No key needed - the primary Brazilian vector source.
  br_decea  Brazil, DECEA AISWEB API (ROTAER, charts, supplements...). Needs a
            free API key: set AISWEB_API_KEY / AISWEB_API_PASS; skipped otherwise.
  us_faa    United States, FAA:
              * nasr/<effective>/  the NASR 28-day subscription zip for every
                cycle still online at nfdc.faa.gov (probed backwards 28 days at a
                time from the current cycle until --nasr-miss consecutive misses;
                2020 cycles were still there on 2026-09-14) - airports, runways,
                navaids, fixes, airways, airspace, ARTCC boundaries, as TXT/XML/CSV.
              * adds/<modified>/  every dataset in the Aeronautical Data Delivery
                Service open-data catalog (adds-faa.opendata.arcgis.com) that offers
                a GeoJSON download: airports, runways, NAVAIDs, ILS, designated
                points, ATS routes, class/special-use/route airspace, MTRs,
                obstacles, plus the "Pending" next-cycle previews.

Usage:
  ~/venv/bin/python scripts/aisdata_pull.py [--only fr,ch,br,us] [--dry-run]
      [--root /Volumes/projects/aisdata] [--sia-drop DIR] [--sia-ip 95.143.78.56]
"""
import argparse, hashlib, html, json, os, re, shutil, subprocess, sys, time, zipfile
from urllib.parse import quote, urljoin, unquote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

UA = 'Mozilla/5.0 (archive.aero aisdata mirror; ryan@archive.aero)'
TODAY = time.strftime('%Y-%m-%d')
STATE = {'dry': False}


# ----------------------------------------------------------------- helpers
def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


BROWSER_UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36'


def get(url, timeout=120, binary=True, ua=None):
    r = urlopen(Request(url, headers={'User-Agent': ua or UA, 'Accept': '*/*'}), timeout=timeout)
    data = r.read()
    return data if binary else data.decode('utf-8', 'replace')


def head_size(url):
    try:
        r = urlopen(Request(url, method='HEAD', headers={'User-Agent': UA}), timeout=60)
        return int(r.headers.get('content-length') or -1)
    except Exception:
        return -1


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest(srcdir):
    p = os.path.join(srcdir, 'manifest.jsonl')
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def already_have(manifest, url=None, file=None, size=None):
    """Latest manifest record for this url/file, if the local copy still exists."""
    for m in reversed(manifest):
        if (url and m.get('url') == url) or (file and m.get('file') == file):
            if size is not None and m.get('bytes') != size:
                return None
            if os.path.exists(os.path.join(m['_root'], m['file'])):
                return m
            return None
    return None


def record(srcdir, rec):
    rec = dict(rec)
    rec['fetched'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(os.path.join(srcdir, 'manifest.jsonl'), 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def fetch_to(srcdir, manifest, url, relpath, meta=None, size_hint=None, ua=None, timeout=600):
    """Download url -> srcdir/relpath unless the manifest says we already have it."""
    dest = os.path.join(srcdir, relpath)
    have = already_have(manifest, url=url, size=size_hint if size_hint and size_hint > 0 else None)
    if have:
        return have
    if STATE['dry']:
        log('  [dry] would fetch', url, '->', relpath)
        return None
    for attempt in range(3):
        try:
            data = get(url, timeout=timeout, ua=ua)
            break
        except (HTTPError, URLError) as e:
            if isinstance(e, HTTPError) and e.code in (403, 404):
                log('  MISSING', e.code, url)
                record(srcdir, {'url': url, 'file': relpath, 'status': e.code, **(meta or {})})
                return None
            log('  retry', attempt + 1, url, str(e)[:100]); time.sleep(3 * (attempt + 1))
    else:
        record(srcdir, {'url': url, 'file': relpath, 'status': None, 'error': 'gave up', **(meta or {})})
        return None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as f:
        f.write(data)
    rec = {'url': url, 'file': relpath, 'status': 200, 'bytes': len(data), 'sha256': sha256(data), **(meta or {})}
    record(srcdir, rec)
    log(f'  got {relpath} ({len(data):,} B)')
    rec['_root'] = srcdir
    manifest.append(rec)
    return rec


def write_readme(srcdir, title, body):
    p = os.path.join(srcdir, 'README.txt')
    with open(p, 'w') as f:
        f.write(f'{title}\n{"=" * len(title)}\n{body.strip()}\n\nLast run of scripts/aisdata_pull.py: {TODAY}\n')


def prep(root, name):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    man = load_manifest(d)
    for m in man:
        m['_root'] = d
    return d, man


# ------------------------------------------------------------- France / SIA
CQUEST = 'http://data.cquest.org/dgac/aip/'
SIA_INDEX = 'https://www.sia.aviation-civile.gouv.fr/produits-numeriques-en-libre-disposition/les-bases-de-donnees-sia.html'


def listing(url):
    """Parse an nginx/apache autoindex into [(href, date, size)]."""
    s = get(url, binary=False)
    out = []
    for m in re.finditer(r'<a href="([^"?]+)">[^<]*</a>\s*(\d{2}-\w{3}-\d{4} \d{2}:\d{2}|\S+ \S+)?\s*(\d+|-)?', s):
        href = m.group(1)
        if href.startswith('../') or href.startswith('/'):
            continue
        out.append((href, m.group(2), int(m.group(3)) if m.group(3) and m.group(3).isdigit() else None))
    return out


def mirror_dir(srcdir, manifest, base_url, rel, depth=0):
    for href, date, size in listing(base_url):
        if href.endswith('/'):
            if depth < 3:
                mirror_dir(srcdir, manifest, urljoin(base_url, href), os.path.join(rel, unquote(href.rstrip('/'))), depth + 1)
        else:
            fetch_to(srcdir, manifest, urljoin(base_url, href), os.path.join(rel, unquote(href)),
                     meta={'source_date': date, 'source': 'data.cquest.org mirror'}, size_hint=size)


def sia_curl(url, ip):
    """SIA's host fails DNSSEC on some home resolvers; --sia-ip pins it (curl --resolve keeps SNI right)."""
    cmd = ['curl', '-sS', '-L', '-A', UA, '--max-time', '60']
    if ip:
        cmd += ['--resolve', f'www.sia.aviation-civile.gouv.fr:443:{ip}']
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200])
    return r.stdout


def sia_upstream_cycles(ip):
    """Scrape the shop index: [(cycle label 'MM/YY', product url, effective range, size)]."""
    out = []
    try:
        idx = sia_curl(SIA_INDEX, ip)
    except Exception as e:
        log('  SIA index unreachable:', e); return out
    pages = sorted(set(re.findall(r'href="(https://www\.sia\.aviation-civile\.gouv\.fr/[^"]*donnees-aeronautiques-xml[^"]*\.html)"', idx)))
    for p in pages:
        try:
            t = sia_curl(p, ip)
        except Exception as e:
            log('  SIA page unreachable:', p, e); continue
        t = html.unescape(re.sub(r'<[^>]+>', ' ', re.sub(r'<script.*?</script>', '', t, flags=re.S)))
        t = re.sub(r'\s+', ' ', t)
        lab = re.search(r'AIRAC (\d\d/\d\d)', t)
        rng = re.search(r'En vigueur du (\d\d/\d\d/\d{4}) au (\d\d/\d\d/\d{4})', t)
        size = re.search(r'\((\d+(?:,\d+)? ?[MK]o)\)', t)
        out.append({'cycle': lab.group(1) if lab else '?', 'url': p,
                    'effective_from': rng.group(1) if rng else None, 'effective_to': rng.group(2) if rng else None,
                    'size': size.group(1) if size else None})
    return out


def sia_export_date(path):
    """Effective date of an SIA database export zip, read from its members
    (XML_SIA_<date>.xml / AIXM4.5_all_FR_OM_<date>.xml), or None when the zip
    is something else (the shop names its downloads opaquely, and the eAIP
    zip is 30x bigger and holds HTML, not exports)."""
    if os.path.getsize(path) > 200 * 2**20:
        return None
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                m = re.search(r'(?:XML_SIA|AIXM4\.5_all_FR_OM)[^/]*?(\d{4}-\d{2}-\d{2})\.xml$', name)
                if m:
                    return m.group(1)
    except zipfile.BadZipFile:
        return None
    return None


def sia_ingest_drops(srcdir, manifest, drops):
    """File any manually downloaded SIA database export zip under
    cycles/<date>/export_xml_bd_sia_<date>.zip. Recognised by content, not
    by name: the shop delivers the product under an opaque filename."""
    n = 0
    for d in drops:
        if not d or not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            src = os.path.join(d, fn)
            if fn.startswith('.'):
                continue
            # Safari expands the zip into a folder and deletes it: re-zip the
            # folder's members (stored deflated, sorted, fixed timestamps from
            # the files themselves) so the filed artefact is a zip like the rest.
            if os.path.isdir(src):
                eff = sia_export_dir_date(src)
                if not eff:
                    continue
                rel = os.path.join('cycles', eff, f'export_xml_bd_sia_{eff}.zip')
                dest = os.path.join(srcdir, rel)
                if os.path.exists(dest):
                    log(f'  {fn}/: cycle {eff} already filed as {rel}, left in place'); continue
                if STATE['dry']:
                    log('  [dry] would zip + file', src, '->', rel); continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for member in sorted(os.listdir(src)):
                        if not member.startswith('.'):
                            zf.write(os.path.join(src, member), member)
                data = open(dest, 'rb').read()
                record(srcdir, {'url': f'file://{src}', 'file': rel, 'status': 200, 'bytes': len(data), 'sha256': sha256(data),
                                'source': 'SIA shop, manual download (free customer account); re-zipped from the folder Safari expanded',
                                'effective': eff, 'original_name': fn})
                shutil.rmtree(src)
                log('  zipped + filed', rel, f'(was folder {fn}/)'); n += 1
                continue
            if not os.path.isfile(src):
                continue
            eff = sia_export_date(src)
            if not eff:
                continue
            rel = os.path.join('cycles', eff, f'export_xml_bd_sia_{eff}.zip')
            dest = os.path.join(srcdir, rel)
            if os.path.exists(dest):
                log(f'  {fn}: cycle {eff} already filed as {rel}, left in place'); continue
            data = open(src, 'rb').read()
            if STATE['dry']:
                log('  [dry] would file', src, '->', rel); continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(src, dest)
            record(srcdir, {'url': f'file://{src}', 'file': rel, 'status': 200, 'bytes': len(data), 'sha256': sha256(data),
                            'source': 'SIA shop, manual download (free customer account)', 'effective': eff,
                            'original_name': fn})
            log('  filed', rel, f'(was {fn})'); n += 1
    return n


MONTHS_FR_EN = {m: i for i, m in enumerate(
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 1)}
MONTHS_FR_EN.update({'fev': 2, 'avr': 4, 'mai': 5, 'jui': 6, 'aou': 8, 'dec': 12})


def sia_eaip_date(name):
    """'eaip_03_sep_2026' (zip or folder) -> '2026-09-03'."""
    m = re.match(r'eaip_(\d{2})_([a-z]{3})_(\d{4})(?:\.zip)?$', name, re.I)
    if not m or m.group(2).lower() not in MONTHS_FR_EN:
        return None
    return f'{m.group(3)}-{MONTHS_FR_EN[m.group(2).lower()]:02d}-{m.group(1)}'


def sia_ingest_eaip(srcdir, manifest, drops):
    """File a hand-downloaded eAIP (the shop's "eAIP" product: the HTML AIP
    with every chart and PDF, ~1.4 GB per cycle) under eaip/<date>/eaip_<date>.zip.
    Safari expands the zip into a folder; that is re-zipped, one artefact
    per cycle. Not read by any build - kept because the shop drops each
    cycle when the next one comes."""
    n = 0
    for d in drops:
        if not d or not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            eff = sia_eaip_date(fn)
            if not eff:
                continue
            src = os.path.join(d, fn)
            rel = os.path.join('eaip', eff, f'eaip_{eff}.zip')
            dest = os.path.join(srcdir, rel)
            if os.path.exists(dest):
                log(f'  {fn}: eAIP {eff} already filed as {rel}, left in place'); continue
            if STATE['dry']:
                log('  [dry] would file eAIP', src, '->', rel); continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.isdir(src):
                count = 0
                with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(src):
                        dirs.sort()
                        for f in sorted(files):
                            if f.startswith('.'):
                                continue
                            full = os.path.join(root, f)
                            zf.write(full, os.path.relpath(full, src)); count += 1
                how = f're-zipped from the folder Safari expanded ({count} files)'
            else:
                shutil.move(src, dest); how = 'as downloaded'
            h = hashlib.sha256()
            with open(dest, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    h.update(chunk)
            record(srcdir, {'url': f'file://{src}', 'file': rel, 'status': 200, 'bytes': os.path.getsize(dest),
                            'sha256': h.hexdigest(), 'source': f'SIA shop eAIP, manual download; {how}',
                            'effective': eff, 'original_name': fn})
            if os.path.isdir(src):
                shutil.rmtree(src)
            log('  filed', rel, f'(was {fn}, {how})'); n += 1
    return n


def sia_export_dir_date(path):
    """Effective date of an SIA export that Safari already expanded into a
    folder: it must hold XML_SIA_<date>.xml (the AIXM twin is optional)."""
    try:
        names = os.listdir(path)
    except OSError:
        return None
    for name in names:
        m = re.match(r'XML_SIA[^/]*?(\d{4}-\d{2}-\d{2})\.xml$', name)
        if m:
            return m.group(1)
    return None


def pull_fr(root, args):
    srcdir, man = prep(root, 'fr_sia')
    log('fr_sia: mirroring', CQUEST)
    mirror_dir(srcdir, man, CQUEST, 'cquest_mirror')
    drops = [args.sia_drop, os.path.expanduser('~/Downloads')]
    n = sia_ingest_drops(srcdir, man, drops)
    n_eaip = sia_ingest_eaip(srcdir, man, drops)
    eaip_dates = sorted({m['file'].split('/')[1] for m in load_manifest(srcdir) if m['file'].startswith('eaip/')})
    ups = sia_upstream_cycles(args.sia_ip)
    have_dates = sorted({m['file'].split('/')[1] for m in load_manifest(srcdir) if m['file'].startswith('cycles/')}
                        | {re.search(r'(\d{4}-\d{2}-\d{2})', m['file']).group(1)
                           for m in load_manifest(srcdir) if 'export_xml_bd_sia' in m['file'] and re.search(r'\d{4}-\d{2}-\d{2}', m['file'])})
    lines = [f'  {u["cycle"]}  {u["effective_from"]} -> {u["effective_to"]}  {u["size"] or ""}  {u["url"]}' for u in ups]
    write_readme(srcdir, 'fr_sia - France, SIA aeronautical database exports (AIXM 4.5 + SIA XML)', f'''
What: the SIA (DGAC) export of the whole French AIP database, France + overseas,
one zip per AIRAC cycle: AIXM4.5_all_FR_OM_<date>.xml (Eurocontrol AIXM 4.5)
and XML_SIA_<date>.xml (SIA's own schema).  Aerodromes, runways, navaids,
airspace, routes, obstacles, with the effective date in the filename.
Licence: Licence Ouverte / Open Licence v2.0 (Etalab) - see cquest_mirror/Licence-Ouverte-v2.0.pdf.

cquest_mirror/   byte-for-byte mirror of http://data.cquest.org/dgac/aip/ -
                 per-cycle zips 2019-02 .. 2023-10, the AIXM 4.5 XSDs, SIA's
                 user guide (documentation/), and cquest's GeoJSON conversions
                 (geojson/, made with github.com/cquest/aixmParser).
cycles/<date>/   zips downloaded from the SIA shop by hand.  The shop
                 ({SIA_INDEX})
                 lists each cycle at 0 EUR but delivers through a free
                 customer account, so this script cannot fetch them; download
                 the "Donnees aeronautiques XML AIRAC mm/yy" product (the 5.6 MB
                 zip, NOT the multi-GB eAIP), leave it in ~/Downloads under
                 whatever name the shop gave it (zip, or the folder Safari
                 expands it into), and re-run: it is recognised by its
                 XML_SIA_<date>.xml member, filed here as a zip and recorded.
eaip/<date>/     the shop's "eAIP" product (HTML AIP + every chart and PDF,
                 ~1.4 GB per cycle), same manual route, filed as one zip per
                 cycle; not read by any build, kept because each cycle
                 disappears from the shop when the next one comes.
oaci_500k/<yr>/  the SIA "Carte aeronautique OACI 1:500 000" sheets (Nord-Ouest,
                 Nord-Est, Sud-Ouest, Sud-Est), one folder per yearly edition,
                 as received - raster PDFs, no georef, SIA/IGN copyright (NOT
                 open data; reference material, never publish).  Not pulled by
                 this script; see oaci_500k/README.txt for what each edition is.

Cycles the shop lists right now ({len(ups)} pages scraped {TODAY}):
{chr(10).join(lines) or "  (index not reachable this run)"}

Effective dates held locally: {", ".join(have_dates) or "none"}
Manually filed this run: {n} export(s), {n_eaip} eAIP(s)
eAIP cycles held: {", ".join(eaip_dates) or "none"}
manifest.jsonl: one line per file (url, bytes, sha256, fetched, source_date).
''')


# ------------------------------------------------------- Switzerland / BAZL
STAC = 'https://data.geo.admin.ch/api/stac/v0.9/collections/'
CH_COLLECTIONS = [
    'ch.bazl.flugplaetze-heliports', 'ch.bazl.gebirgslandeplaetze', 'ch.bazl.luftfahrthindernis',
    'ch.bazl.einschraenkungen-drohnen', 'ch.bazl.uas-aktivitaetszonen', 'ch.bazl.hindernisbegrenzungsflaechen-kataster',
    'ch.bazl.sicherheitszonenplan', 'ch.bazl.luftfahrtkarten-icao', 'ch.bazl.segelflugkarte',
]


def pull_ch(root, args):
    srcdir, man = prep(root, 'ch_bazl')
    summary = []
    for cid in CH_COLLECTIONS:
        try:
            col = json.loads(get(STAC + cid, binary=False))
            items = json.loads(get(STAC + cid + '/items?limit=100', binary=False)).get('features', [])
        except Exception as e:
            log('  STAC error', cid, e); continue
        for it in items:
            upd = (it.get('properties') or {}).get('updated') or (it.get('properties') or {}).get('datetime') or ''
            snap = upd[:10] or TODAY
            for aname, a in it.get('assets', {}).items():
                href = a['href']
                # geo.admin publishes a multihash checksum: skip when unchanged, whatever the snapshot date
                mh = a.get('checksum:multihash')
                prev = next((m for m in reversed(man) if m.get('url') == href and m.get('status') == 200), None)
                if prev and mh and prev.get('multihash') == mh and os.path.exists(os.path.join(srcdir, prev['file'])):
                    continue
                rel = os.path.join(cid, snap, aname)
                fetch_to(srcdir, man, href, rel, meta={'collection': cid, 'item': it['id'], 'updated': upd,
                                                       'type': a.get('type'), 'multihash': mh,
                                                       'title': col.get('title'), 'license': (col.get('license') or '')})
        summary.append(f'  {cid}: {col.get("title")} [{col.get("license")}] {len(items)} item(s)')
    write_readme(srcdir, 'ch_bazl - Switzerland, FOCA (BAZL) open geodata via data.geo.admin.ch STAC', f'''
What: the Federal Office of Civil Aviation datasets published as open government
data on the federal geoportal.  Vector layers come as GeoPackage / Shapefile /
GeoJSON (LV95 EPSG:2056 and WGS84 variants), obstacles as AIXM, and the ICAO
1:500 000 and glider charts as cloud-optimized GeoTIFF (the two rasters are a
chart family in their own right - Swiss ICAO chart, current edition only).
Airspace itself is not on the geoportal (skyguide publishes it through the
AIP / EAD); the ICAO chart raster carries it graphically.
Licence: OGD "open use" terms of geo.admin.ch (per-collection license field in manifest.jsonl).
Layout: <collection id>/<item updated date>/<asset>.  A new dated folder appears
only when the STAC checksum of an asset changes, so old folders are prior editions.

Collections pulled:
{chr(10).join(summary)}
''')


# --------------------------------------------------------- Brazil / ANAC
ANAC = 'https://sistemas.anac.gov.br/dadosabertos/Aerodromos/'


def pull_br_anac(root, args):
    srcdir, man = prep(root, 'br_anac')
    log('br_anac: crawling', ANAC)
    snap = os.path.join('snapshots', TODAY)

    def crawl(url, rel, depth=0):
        for href, date, size in listing(url):
            if href.endswith('/'):
                if depth < 4:
                    crawl(urljoin(url, href), os.path.join(rel, unquote(href.rstrip('/'))), depth + 1)
            else:
                # keep one copy per distinct content: reuse the last snapshot's file when size matches
                prev = next((m for m in reversed(man) if m.get('url') == urljoin(url, href) and m.get('status') == 200), None)
                if prev and size and prev.get('bytes') == size and os.path.exists(os.path.join(srcdir, prev['file'])):
                    continue
                fetch_to(srcdir, man, urljoin(url, href), os.path.join(snap, rel, unquote(href)),
                         meta={'source_date': date, 'source': 'ANAC dados abertos'})
    crawl(ANAC, '')
    write_readme(srcdir, 'br_anac - Brazil, ANAC open data: aerodrome register', f'''
What: the civil aviation agency's aerodrome register as CSV - public and private
aerodromes (with ICAO code, coordinates, elevation, runways), runway and taxiway
tables, apron stands, heliports, and "Aerodromos Excluidos" (aerodromes removed
from the register, with dates - the lifespan signal).  Crawled from
{ANAC} (directory listing).
Licence: Brazilian open-data (dados.gov.br terms; ODbL-like attribution).
Layout: snapshots/<date>/<original folder tree>/<file>.  A file is re-fetched
only when the remote size changed, so each snapshot holds just what moved.
''')


GEOAISWEB = 'https://geoaisweb.decea.mil.br/geoserver/ows'
GEOAISWEB_LAYERS = [
    'airport', 'airport_heliport', 'heliport', 'runway', 'runway_v2', 'rwydirection',
    'vor', 'ndb', 'dme', 'navaids', 'waypoint', 'waypoint_aisweb',
    'airway', 'vw_aerovia_alta', 'vw_aerovia_baixa', 'vw_aerovia_alta_v2', 'vw_aerovia_baixa_v2', 'rotas_diretas',
    'airspace', 'fir', 'SETOR_FIR', 'setores_tma', 'TMA', 'CTA', 'CTR', 'ATZ', 'zida', 'fiz', 'fis', 'opea',
    'eac_d', 'eac_p', 'eac_r', 'aga_jurisdicao',
]


def pull_br_geoaisweb(root, args):
    srcdir, man = prep(root, 'br_geoaisweb')
    log('br_geoaisweb: WFS GetFeature x', len(GEOAISWEB_LAYERS), 'layers')
    caps_url = f'{GEOAISWEB}?service=WFS&version=2.0.0&request=GetCapabilities'
    fetch_to(srcdir, man, caps_url + f'&_={TODAY}', os.path.join('snapshots', TODAY, 'GetCapabilities.xml'),
             meta={'source': 'DECEA GeoAISWEB WFS'})
    summary = []
    for layer in GEOAISWEB_LAYERS:
        url = (f'{GEOAISWEB}?service=WFS&version=2.0.0&request=GetFeature&typeNames=ICA:{layer}'
               f'&outputFormat=application/json&srsName=EPSG:4326')
        rel = os.path.join('snapshots', TODAY, f'{layer}.geojson')
        if STATE['dry']:
            log('  [dry] would fetch layer', layer); continue
        try:
            data = get(url, timeout=300)
            feats = json.loads(data).get('features', [])
        except Exception as e:
            log('  WFS error', layer, str(e)[:120]); record(srcdir, {'url': url, 'file': rel, 'status': None, 'error': str(e)[:200]}); continue
        h = sha256(data)
        prev = next((m for m in reversed(man) if m.get('layer') == layer and m.get('status') == 200), None)
        if prev and prev.get('sha256') == h and os.path.exists(os.path.join(srcdir, prev['file'])):
            summary.append(f'  {layer}: {len(feats)} features (unchanged since {prev["file"].split("/")[1]})'); continue
        os.makedirs(os.path.dirname(os.path.join(srcdir, rel)), exist_ok=True)
        open(os.path.join(srcdir, rel), 'wb').write(data)
        record(srcdir, {'url': url, 'file': rel, 'status': 200, 'bytes': len(data), 'sha256': h, 'layer': layer,
                        'features': len(feats), 'source': 'DECEA GeoAISWEB WFS'})
        log(f'  got {layer}: {len(feats)} features ({len(data):,} B)')
        summary.append(f'  {layer}: {len(feats)} features')
    write_readme(srcdir, 'br_geoaisweb - Brazil, DECEA GeoAISWEB WFS (ICA aeronautical layers)', f'''
What: the Instituto de Cartografia Aeronautica's GeoServer, public WFS/WMS, no key:
{GEOAISWEB}  (workspace ICA; 400+ layers, most are per-aerodrome chart sheets and
obstacle-surface polygons - only the country-wide aeronautical layers are pulled).
Features carry an 'effectived' (effective date) attribute where DECEA maintains one.
Licence: DECEA "Politica de Uso" of AISWEB (official public AIS data, attribution).
Layout: snapshots/<date>/<layer>.geojson (WGS84).  A layer is re-saved only when
its content hash changed since the last snapshot.

Layers this run:
{chr(10).join(summary)}
''')


AISWEB = 'https://aisweb.decea.mil.br/api/'
AISWEB_AREAS = ['rotaer', 'cartas', 'suplementos', 'infotemp', 'aerodromos']


def pull_br_decea(root, args):
    key, pw = os.environ.get('AISWEB_API_KEY'), os.environ.get('AISWEB_API_PASS')
    srcdir, man = prep(root, 'br_decea')
    note = ''
    if key and pw:
        for area in AISWEB_AREAS:
            url = f'{AISWEB}?apiKey={quote(key)}&apiPass={quote(pw)}&area={area}'
            if STATE['dry']:
                log('  [dry] would query area', area); continue
            try:
                data = get(url)
            except Exception as e:
                log('  AISWEB error', area, e); continue
            if b'Erro' in data[:300]:
                log('  AISWEB refused', area, data[:120]); continue
            rel = os.path.join('snapshots', TODAY, f'{area}.xml')
            os.makedirs(os.path.dirname(os.path.join(srcdir, rel)), exist_ok=True)
            open(os.path.join(srcdir, rel), 'wb').write(data)
            record(srcdir, {'url': f'{AISWEB}?area={area}', 'file': rel, 'status': 200, 'bytes': len(data),
                            'sha256': sha256(data), 'source': 'DECEA AISWEB API'})
            log(f'  got {rel} ({len(data):,} B)')
    else:
        note = ('\nNOT PULLED: set AISWEB_API_KEY and AISWEB_API_PASS.  There is no self-service signup: '
                'the API page (https://aisweb.decea.mil.br/?i=publicacoes&p=api, "solicite sua chave de acesso") '
                'routes the request through a SAC-DECEA help-desk ticket (https://ajuda.decea.gov.br/). '
                'Docs: https://documenter.getpostman.com/view/7201070/SzKQyg3H .  The vector data is on the '
                'keyless GeoAISWEB WFS (br_geoaisweb/) anyway; this API mainly adds ROTAER text, NOTAM and chart PDFs.\n')
        log('br_decea: skipped (no AISWEB_API_KEY / AISWEB_API_PASS in env)')
    write_readme(srcdir, 'br_decea - Brazil, DECEA AISWEB API', f'''
What: DECEA's AIS web service.  Query form {AISWEB}?apiKey=..&apiPass=..&area=<area>
returning XML per area: {", ".join(AISWEB_AREAS)} (rotaer = the aerodrome
manual, cartas = chart index with PDF links, suplementos = AIP SUP list).
Layout: snapshots/<date>/<area>.xml.{note}
''')


# ------------------------------------------------------------- USA / FAA
NASR_ANCHOR = '2026-09-03'          # a known 28-day cycle effective date
NASR_URL = 'https://nfdc.faa.gov/webContent/28DaySub/28DaySubscription_Effective_{d}.zip'
ADDS_CATALOG = 'https://adds-faa.opendata.arcgis.com/api/feed/dcat-us/1.1.json'


def nasr_exists(d):
    """nfdc 503s HEAD and non-browser UAs; a 1-byte range GET is the reliable probe."""
    try:
        r = urlopen(Request(NASR_URL.format(d=d), headers={'User-Agent': BROWSER_UA, 'Range': 'bytes=0-0'}), timeout=60)
        cr = r.headers.get('content-range', '')
        return int(cr.split('/')[-1]) if '/' in cr else -1
    except HTTPError as e:
        return None if e.code in (403, 404) else -1
    except URLError:
        return -1


def pull_us(root, args):
    import datetime as dt
    srcdir, man = prep(root, 'us_faa')
    # --- NASR: walk cycles backwards from the current one, and one forward
    anchor = dt.date.fromisoformat(NASR_ANCHOR)
    today = dt.date.today()
    cur = anchor + dt.timedelta(days=28 * ((today - anchor).days // 28))
    dates = [cur + dt.timedelta(days=28)] + [cur - dt.timedelta(days=28 * i) for i in range(0, args.nasr_max_cycles)]
    misses, held, seen_dates = 0, [], set()
    for d in dates:
        ds = d.isoformat()
        rel = os.path.join('nasr', ds, f'28DaySubscription_Effective_{ds}.zip')
        if any(m.get('file') == rel and m.get('status') == 200 and os.path.exists(os.path.join(srcdir, m['file'])) for m in man):
            held.append(ds); misses = 0; continue
        size = nasr_exists(ds)
        if size is None:
            if d <= cur:
                misses += 1
                if misses >= args.nasr_miss:
                    log(f'  NASR: {args.nasr_miss} consecutive missing cycles before {ds}; stopping back-fill'); break
            continue
        misses = 0
        if STATE['dry']:
            log(f'  [dry] would fetch NASR {ds} ({size:,} B)'); continue
        log(f'  NASR {ds} ({size/1e6:.0f} MB)')
        r = fetch_to(srcdir, man, NASR_URL.format(d=ds), rel, meta={'effective': ds, 'source': 'FAA NASR 28-day subscription'},
                     ua=BROWSER_UA, timeout=1800)
        if r:
            held.append(ds)
    # --- ADDS open-data catalog
    adds_lines = []
    try:
        cat = json.loads(get(ADDS_CATALOG, binary=False, ua=BROWSER_UA))
    except Exception as e:
        log('  ADDS catalog error', e); cat = {'dataset': []}
    snap = os.path.join('adds', TODAY)
    if not STATE['dry']:
        os.makedirs(os.path.join(srcdir, snap), exist_ok=True)
        open(os.path.join(srcdir, snap, 'catalog_dcat.json'), 'w').write(json.dumps(cat, indent=1))
    for ds in cat.get('dataset', []):
        title = ds.get('title', '').strip()
        gj = next((x.get('downloadURL') or x.get('accessURL') for x in ds.get('distribution', [])
                   if (x.get('format') or '').lower() == 'geojson'), None)
        if not gj:
            continue
        mod = (ds.get('modified') or '')[:10] or TODAY
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', title).strip('_')
        rel = os.path.join('adds', mod, f'{safe}.geojson')
        prev = next((m for m in reversed(man) if m.get('title') == title and m.get('status') == 200), None)
        if prev and prev.get('modified') == mod and os.path.exists(os.path.join(srcdir, prev['file'])):
            adds_lines.append(f'  {title}: unchanged ({mod})'); continue
        if STATE['dry']:
            log('  [dry] would fetch ADDS', title, mod); continue
        r = fetch_to(srcdir, man, gj, rel, meta={'title': title, 'modified': mod, 'identifier': ds.get('identifier'),
                                                 'source': 'FAA ADDS open data'}, ua=BROWSER_UA, timeout=900)
        if r:
            try:
                n = len(json.load(open(os.path.join(srcdir, rel))).get('features', []))
            except Exception:
                n = '?'
            adds_lines.append(f'  {title}: {n} features ({mod})')
    write_readme(srcdir, 'us_faa - United States, FAA NASR 28-day subscription + ADDS open data', f'''
nasr/<effective>/28DaySubscription_Effective_<date>.zip
    The National Airspace System Resources subscriber file for each 28-day
    cycle: APT (airports/runways), NAV, FIX, AWY, ARB (ARTCC boundaries), ATS,
    CLS (class airspace), SUA, MTR, ILS, COM, TWR, HPF, PJA, PFR, STARDP, and
    since 2022 the same as CSV plus AIXM 5.1 XML.  Fetched from
    {NASR_URL.format(d='<date>')}
    The script probes backwards from the current cycle 28 days at a time until
    {args.nasr_miss} consecutive cycles are missing upstream, so each run also picks up
    the next cycle when it appears.  Public domain (US Government work).
    Cycles held: {len(held)} ({held[-1] if held else '-'} .. {held[0] if held else '-'})

adds/<modified>/<Dataset>.geojson
    Every dataset in the Aeronautical Data Delivery Service catalog
    ({ADDS_CATALOG}) that offers a GeoJSON download, WGS84, keyed by the
    catalog's "modified" date (= the cycle it was published for).  "Pending *"
    datasets are the next cycle's preview.  catalog_dcat.json is the catalog as seen.
    Datasets this run:
{chr(10).join(adds_lines)}
''')


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/Volumes/projects/aisdata')
    ap.add_argument('--only', default='fr,ch,br,us', help='comma list of fr,ch,br,us')
    ap.add_argument('--nasr-max-cycles', type=int, default=400, help='how many 28-day cycles back to probe at most')
    ap.add_argument('--nasr-miss', type=int, default=6, help='stop the NASR back-fill after this many consecutive missing cycles')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--sia-drop', default=None, help='extra folder to look for manually downloaded SIA zips')
    ap.add_argument('--sia-ip', default=None, help='pin www.sia.aviation-civile.gouv.fr (local resolver SERVFAILs it)')
    a = ap.parse_args()
    STATE['dry'] = a.dry_run
    if not os.path.isdir(os.path.dirname(a.root)):
        sys.exit(f'{os.path.dirname(a.root)} not mounted')
    os.makedirs(a.root, exist_ok=True)
    rp = os.path.join(a.root, 'README.txt')
    if not os.path.exists(rp):
        open(rp, 'w').write('''aisdata - structured / vector aeronautical data (sibling of rawtiffs)
=====================================================================
Raw material for archive.aero's time-aware overlays (aerodromes, navaids,
airspace, routes, obstacles): national AIS database exports and open-data
registers, kept as downloaded, one snapshot per effective date, so that a
time series accumulates.  Never derived files: conversions to GeoJSON /
PMTiles are built elsewhere from these.  Each source directory has its own
README.txt (what, licence, layout) and manifest.jsonl (one line per file:
url, bytes, sha256, fetched, source date).  Pulled by scripts/aisdata_pull.py.
''')
    only = {s.strip() for s in a.only.split(',')}
    if 'fr' in only:
        pull_fr(a.root, a)
    if 'ch' in only:
        pull_ch(a.root, a)
    if 'br' in only:
        pull_br_anac(a.root, a)
        pull_br_geoaisweb(a.root, a)
        pull_br_decea(a.root, a)
    if 'us' in only:
        pull_us(a.root, a)
    log('done')


if __name__ == '__main__':
    main()
