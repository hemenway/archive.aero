#!/usr/bin/env python3
"""Publish a slicerd run's new eras and chart artifacts the way the October 13 release did.

Four phases, each idempotent, all driven by one candidate directory (worklist 18,
"Repeat the production data build"):

  plan      hash the run's era archives and the chart artifacts of the given era keys
            (scripts/next_version_archives.py's local records: old/new/path/sha256/z/b/size)
            -> CANDIDATE/work/upload-plan.json. No network.
  upload    put every planned object under its content-hashed key in r2:charts-beta
            (skips objects already there with matching sha256 metadata), head-verify size
            and sha256, append the verified records to RUN/hashed/plan.jsonl (the shape
            publish_chart_pmtiles_from_reslice.py reads) and
            CANDIDATE/publication/verified-objects.jsonl.
  permanent era archives: server-side copy charts-beta/<hashed> -> charts/sectionals/<key>.pmtiles
            (refuses a permanent key that exists with different bytes);
            chart artifacts: merge the run's manifest lines into
            worklists/data/chart_pmtiles/manifest.jsonl and run
            publish_chart_pmtiles_from_reslice.py --copy for the run's keys (receipts).
  stage     CANDIDATE/stage mirrors PROD/stage by symlink plus sparse stubs for the new
            eras; CANDIDATE/work/plan.json = PROD/work/plan.json superseded by the new
            records; timeline_data.json / coverage.json rebuilt, the timeline filtered to
            artifacts in the plan (-> work/timeline.json) for next_build_overlays.py.

Then (by hand, see worklist 18): overlays -> cards -> overlays again -> airports ->
manifest -> upload next/ + cards + timeline -> shell build + deploy -> legacy bundle.
R2 credentials come from the rclone `r2:` remote for this process only.
"""
import argparse, datetime, json, os, re, shlex, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
import next_version_archives as nva          # noqa: E402
from next_pmtiles import header as pm_header  # noqa: E402

CHART_PM_DIR = REPO / 'worklists' / 'data' / 'chart_pmtiles'
HASHED_BUCKET, PERMANENT_BUCKET = 'charts-beta', 'charts'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def read_jsonl(path):
    if not Path(path).exists(): return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def append_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        for r in records: f.write(json.dumps(r, separators=(',', ':')) + '\n')


def r2_env():
    """Export the rclone r2 remote's credentials, as next_execute_plan.sh does."""
    if os.environ.get('R2_ENDPOINT_URL'): return
    dump = json.loads(subprocess.check_output(['rclone', 'config', 'dump']))['r2']
    os.environ['AWS_ACCESS_KEY_ID'] = dump['access_key_id']
    os.environ['AWS_SECRET_ACCESS_KEY'] = dump['secret_access_key']
    os.environ['R2_ENDPOINT_URL'] = dump['endpoint']


def stub_bytes(path):
    """Bytes of header + root directory + metadata + leaf directories: what a sparse manifest stub holds."""
    with open(path, 'rb') as f: h = pm_header(f.read(127))
    return max(h['root_offset'] + h['root_length'], h['metadata_offset'] + h['metadata_length'],
               h['leaf_directory_offset'] + h['leaf_directory_length'], 127)


def run_chart_entries(run, keys):
    """Chart manifest lines the run emitted for these era keys (the run seeds its manifest with the canonical one)."""
    manifest = run / 'charts' / 'manifest.jsonl'
    seed = int((run / 'charts' / 'manifest.seed_lines').read_text().strip() or 0) if (run / 'charts' / 'manifest.seed_lines').exists() else 0
    lines = read_jsonl(manifest)[seed:]
    wanted = {}
    for e in lines:
        if e.get('date_key') in keys and (run / 'charts' / e['slug'] / (e['name'] + '.pmtiles')).is_file():
            wanted[e['key']] = e   # the last line per key is the final artifact
    return wanted


# ---------------------------------------------------------------- plan

def cmd_plan(a):
    run, cand, keys = a.run, a.candidate, set(a.keys)
    (cand / 'work').mkdir(parents=True, exist_ok=True)
    missing = [k for k in keys if not (run / 'pmtiles' / f'{k}.pmtiles').is_file()]
    if missing: sys.exit(f'no pmtiles in {run}/pmtiles for: {missing}')
    eras = [r for r in nva.plan(directory=run / 'pmtiles', prefix='sectionals')
            if re.sub(r'^sectionals/|\.pmtiles$', '', r['old']) in keys]
    charts = run_chart_entries(run, keys)
    chart_records = [r for r in nva.plan(directory=run / 'charts', prefix='sectionals/chart')
                     if r['old'].removeprefix('sectionals/') in charts]
    for r in eras + chart_records:
        r['stub_bytes'] = stub_bytes(r['local']); r['run'] = run.name
    for r in eras:
        key = re.sub(r'^sectionals/|\.pmtiles$', '', r['old'])
        log = run / 'logs' / f'align_{key}.log'
        r['align'] = log.read_text().strip().splitlines()[-1] if log.exists() else 'no alignment log'
    records = eras + chart_records
    (cand / 'work' / 'upload-plan.json').write_text(json.dumps(records, indent=1))
    (cand / 'work' / 'chart-manifest.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in charts.values()))
    print(f'{len(eras)} era archives, {len(chart_records)} chart artifacts -> {cand}/work/upload-plan.json')
    for r in eras: print(f"  {r['old']:50} z{r['z']} b={r['b']} {r['size']:>12,} B  {r['align']}")
    for r in chart_records: print(f"  {r['old']:50} z{r['z']} {r['size']:>12,} B")


# ---------------------------------------------------------------- upload (hashed)

def head(client, bucket, key):
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.ClientError as error:
        if str(error.response.get('Error', {}).get('Code')) in ('404', 'NoSuchKey', 'NotFound'): return None
        raise


def verify(client, bucket, key, record):
    h = head(client, bucket, key)
    if h is None: return 'missing'
    if h['ContentLength'] != record['size']: return f"size {h['ContentLength']} != {record['size']}"
    if h.get('Metadata', {}).get('sha256') != record['sha256']: return 'sha256 metadata differs'
    return 'exact'


def cmd_upload(a):
    cand, run = a.candidate, a.run
    records = json.loads((cand / 'work' / 'upload-plan.json').read_text())
    r2_env(); client = nva.remote_client()
    before = {r['new']: verify(client, HASHED_BUCKET, r['new'], r) for r in records}
    todo = [r for r in records if before[r['new']] != 'exact']
    for r in todo:
        if before[r['new']] != 'missing': sys.exit(f"{r['new']}: exists in {HASHED_BUCKET} but {before[r['new']]}; not overwriting")
    if a.dry_run:
        print(f"dry run: {len(todo)} to upload, {len(records) - len(todo)} already exact"); return
    nva.execute(todo, HASHED_BUCKET)
    verified, bad = [], []
    for r in records:
        state = verify(client, HASHED_BUCKET, r['new'], r)
        action = 'already exact' if before[r['new']] == 'exact' else ('uploaded and exact' if state == 'exact' else state)
        print(f'  {action:20} {HASHED_BUCKET} {r["new"]}')
        (verified if state == 'exact' else bad).append({**r, 'remote': f'r2:{HASHED_BUCKET}', 'uploaded_at': now(),
                                                       'verified_at': now(), 'verification_action': action})
    if bad: sys.exit(f'{len(bad)} objects failed verification; nothing recorded for them')
    done = {r['new'] for r in read_jsonl(run / 'hashed' / 'plan.jsonl')}
    append_jsonl(run / 'hashed' / 'plan.jsonl', [r for r in verified if r['new'] not in done])
    append_jsonl(cand / 'publication' / 'verified-objects.jsonl',
                 [{'bucket': HASHED_BUCKET, 'key': r['new'], 'size': r['size'], 'sha256': r['sha256'],
                   'verified_at': r['verified_at'], 'action': r['verification_action']} for r in verified])
    (cand / 'work' / 'upload-plan.json').write_text(json.dumps(verified, indent=1))
    print(f'{len(verified)} objects exact in {HASHED_BUCKET}; plan recorded in {run}/hashed/plan.jsonl')


# ---------------------------------------------------------------- permanent keys

def cmd_permanent(a):
    cand, run = a.candidate, a.run
    records = json.loads((cand / 'work' / 'upload-plan.json').read_text())
    if any(not r.get('verified_at') for r in records): sys.exit('run `upload` first: plan records carry no verification')
    r2_env(); client = nva.remote_client(); config = nva.transfer_config()
    eras = [r for r in records if '/chart/' not in r['old']]
    for r in eras:
        key = r['old']  # sectionals/<era>.pmtiles
        state = verify(client, PERMANENT_BUCKET, key, r)
        if state == 'exact': print(f'  already exact      {PERMANENT_BUCKET} {key}'); continue
        if state != 'missing' and not a.replace: sys.exit(f'{key}: exists in {PERMANENT_BUCKET} with {state}; pass --replace to overwrite in place')
        if a.dry_run: print(f'  would copy         {PERMANENT_BUCKET} {key}  <- {HASHED_BUCKET}/{r["new"]}'); continue
        source = client.head_object(Bucket=HASHED_BUCKET, Key=r['new'])
        client.copy({'Bucket': HASHED_BUCKET, 'Key': r['new']}, PERMANENT_BUCKET, key, Config=config,
                    ExtraArgs={'CopySourceIfMatch': source['ETag'], 'MetadataDirective': 'REPLACE',
                               'Metadata': {**source.get('Metadata', {}), 'sha256': r['sha256']},
                               'ContentType': 'application/vnd.pmtiles'})
        state = verify(client, PERMANENT_BUCKET, key, r)
        print(f'  {"copied and exact" if state == "exact" else "FAILED " + state:18} {PERMANENT_BUCKET} {key}')
        if state != 'exact': sys.exit(1)
        append_jsonl(cand / 'publication' / 'verified-objects.jsonl',
                     [{'bucket': PERMANENT_BUCKET, 'key': key, 'size': r['size'], 'sha256': r['sha256'],
                       'verified_at': now(), 'action': 'copied and exact', 'source': f'{HASHED_BUCKET}/{r["new"]}'}])
    # Chart artifacts: canonical manifest lines, then the reslice publisher does the copies + receipts.
    new_entries = read_jsonl(cand / 'work' / 'chart-manifest.jsonl')
    canonical = CHART_PM_DIR / 'manifest.jsonl'
    have = {l.strip() for l in canonical.read_text().splitlines()}
    add = [e for e in new_entries if json.dumps(e) not in have and json.dumps(e, separators=(',', ':')) not in have]
    if add and not a.dry_run:
        with open(canonical, 'a') as f:
            for e in add: f.write(json.dumps(e) + '\n')
    print(f'{len(add)} chart manifest line(s) appended to {canonical}')
    keys = cand / 'work' / 'chart-keys.txt'
    keys.write_text(''.join(e['key'] + '\n' for e in new_entries))
    cmd = [sys.executable, str(REPO / 'scripts' / 'publish_chart_pmtiles_from_reslice.py'), '--run', str(run),
           '--keys', str(keys), '--workers', '4'] + ([] if a.dry_run else ['--copy'])
    if a.replace: cmd.append('--refresh')
    print('+', ' '.join(shlex.quote(c) for c in cmd)); subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- stage

def cmd_stage(a):
    cand, prod = a.candidate, a.prod
    records = json.loads((cand / 'work' / 'upload-plan.json').read_text())
    stage = cand / 'stage'
    for sub in ('sectionals', 'basemap', 'airspace'):
        (stage / sub).mkdir(parents=True, exist_ok=True)
        for p in sorted((prod / 'stage' / sub).iterdir()):
            if p.name.startswith('.'): continue
            t = stage / sub / p.name
            if not t.exists() and not t.is_symlink(): t.symlink_to(p.resolve() if p.is_symlink() else p)
    for r in records:
        if '/chart/' in r['old']: continue
        t = stage / r['new']
        if t.exists(): continue
        with open(r['local'], 'rb') as f: prefix = f.read(r['stub_bytes'])
        with open(t, 'wb') as f: f.write(prefix); f.truncate(r['size'])
        print(f'  stub {t.name} ({r["stub_bytes"]} of {r["size"]} bytes)')
    (cand / 'work').mkdir(exist_ok=True)
    plan = json.loads((prod / 'work' / 'plan.json').read_text())
    latest = {p['old']: p for p in plan}
    for r in records: latest[r['old']] = r
    merged = list(latest.values())
    (cand / 'work' / 'plan.json').write_text(json.dumps(merged, separators=(',', ':')))
    print(f'plan.json: {len(plan)} -> {len(merged)} records')
    subprocess.run([sys.executable, str(REPO / 'scripts' / 'build_timeline_data.py')], check=True, cwd=REPO)
    subprocess.run([sys.executable, str(REPO / 'scripts' / 'build_coverage.py')], check=True, cwd=REPO)
    full = json.loads((REPO / 'timeline_data.json').read_text())
    (cand / 'work' / 'timeline-full.json').write_text(json.dumps(full, separators=(',', ':')))
    have = set(latest)
    dropped = 0
    for loc in full['locations'].values():
        for c in loc.get('charts', []):
            pm = c.get('pm')
            if not pm: continue
            if isinstance(pm, list):
                keep = [p for p in pm if 'sectionals/' + p in have]
                if keep: c['pm'] = keep
                else: del c['pm']; dropped += 1
            elif 'sectionals/' + pm not in have:
                del c['pm']; dropped += 1
    (cand / 'work' / 'timeline.json').write_text(json.dumps(full, separators=(',', ':')))
    for f in ('coverage.json',): (cand / 'work' / f).write_bytes((REPO / f).read_bytes())
    (cand / 'work' / 'og-manifest.jsonl').write_bytes((prod / 'work' / 'og-manifest.jsonl').read_bytes())
    print(f'timeline.json: {dropped} chart link(s) without a plan artifact dropped; stage holds '
          f'{len(list((stage / "sectionals").iterdir()))} era archives')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True, help='slicerd run dir, e.g. /Volumes/projects/slicer-runs/<run>')
    ap.add_argument('--candidate', type=Path, required=True, help='release candidate dir, e.g. ~/archive-next-build/<name>')
    ap.add_argument('--prod', type=Path, default=Path.home() / 'archive-next-build' / 'prod')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan'); p.add_argument('keys', nargs='+'); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser('upload'); p.add_argument('--dry-run', action='store_true'); p.set_defaults(fn=cmd_upload)
    p = sub.add_parser('permanent'); p.add_argument('--dry-run', action='store_true')
    p.add_argument('--replace', action='store_true', help='overwrite permanent keys whose bytes differ (a republish)')
    p.set_defaults(fn=cmd_permanent)
    p = sub.add_parser('stage'); p.set_defaults(fn=cmd_stage)
    a = ap.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
