#!/usr/bin/env python3
"""Expand the compact per-authority harvest dumps into index_rows.jsonl.

Each <root>/harvest/ead_harvest_<tag>.json (saved by the browser) is {"harvested", "auths": {CODE: {"prefix", "rows": [[eff,name,eaip,airac,heading,tail,fname_override],...]}}}
written from the in-browser harvester. link = prefix + '/' + tail + '/' + (fname_override or name-minus-.pdf + '_' + eff + '.pdf').
Re-running replaces that authority's rows in index_rows.jsonl (links carry a rotating token, so the newest harvest wins).
"""
import glob, json, os, sys

root = sys.argv[1] if len(sys.argv) > 1 else '/Volumes/projects/rawtiffs_attic/ead_aip_2026-09'
idx = os.path.join(root, 'index_rows.jsonl')
old = {}
if os.path.exists(idx):
    for l in open(idx):
        if l.strip():
            r = json.loads(l)
            old.setdefault(r['code'], []).append(r)
for p in sorted(glob.glob(os.path.join(root, 'harvest', '*.json')), key=os.path.getmtime):
    d = json.load(open(p))
    for code, a in d['auths'].items():
        rows = []
        for eff, name, eaip, airac, heading, tail, fo in a['rows']:
            fname = fo or (name[:-4] if name.lower().endswith('.pdf') else name) + '_' + eff + '.pdf'
            link = f"{a['prefix']}/{tail}/{fname}" if tail and a['prefix'] else None
            rows.append({'code': code, 'eff': eff, 'name': name, 'eaip': eaip, 'airac': airac,
                         'heading': heading, 'link': link, 'auth': code, 'harvested': d.get('harvested')})
        old[code] = rows
n = 0
with open(idx, 'w') as f:
    for code in sorted(old):
        for r in old[code]:
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); n += 1
print(f'{len(old)} authorities, {n} rows -> {idx}')
