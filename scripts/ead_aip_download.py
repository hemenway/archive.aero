#!/usr/bin/env python3
"""Download every AIP document indexed from EAD Basic into the attic.

Input : <root>/index_rows.jsonl  (one line per AIP Library result row, written by
        the in-browser harvester: code, eff, name, eaip, airac, heading, link, auth)
Output: <root>/<AUTH>/<LANG>/<TYPE>/<PART>/<filename>   (filename as served, which
        already carries the effective date, e.g. LQ_GEN_0_2_en_2026-07-09.pdf)
        <root>/manifest.jsonl  one line per attempted file (idempotent; re-run to resume)

The pamslight PDF URLs need no session cookie (verified 2026-09-14 with curl), so
only the *index* needs the logged-in browser; the bytes are fetched here.
"""
import argparse, hashlib, json, os, sys, time, threading, queue
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE = 'https://www.ead.eurocontrol.int'
UA = 'Mozilla/5.0 (archive.aero AIP mirror; ryan@archive.aero)'


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def dest_for(root, row):
    # link: /eadbasic/pamslight-<hash>/<key>/<LANG>/<TYPE>/<PART>/<file>
    parts = row['link'].strip('/').split('/')
    tail = parts[-4:] if len(parts) >= 4 else parts
    lang, typ, part, fname = (tail + ['', '', '', ''])[:4] if len(tail) == 4 else ('_', '_', '_', parts[-1])
    return os.path.join(root, row['code'], lang, typ, part, fname)


def fetch(url, tries=4):
    last = None
    for i in range(tries):
        try:
            r = urlopen(Request(url, headers={'User-Agent': UA, 'Accept': '*/*'}), timeout=120)
            data = r.read()
            return r.status, r.headers.get('content-type', ''), data
        except HTTPError as e:
            if e.code in (404, 403, 410):
                return e.code, e.headers.get('content-type', ''), e.read()[:200]
            last = e
        except URLError as e:
            last = e
        time.sleep(2 * (i + 1))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/Volumes/projects/rawtiffs_attic/ead_aip_2026-09')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--delay', type=float, default=0.25, help='per-worker pause between files')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    rows = load_jsonl(os.path.join(a.root, 'index_rows.jsonl'))
    done, missing = set(), {}
    for m in load_jsonl(os.path.join(a.root, 'manifest.jsonl')):
        if m.get('status') == 200:
            done.add(m['file'])
        elif m.get('status') in (404, 403, 410):
            missing[m['file']] = m['link']   # retry only under a fresh (re-harvested) link
    seen, todo = set(), []
    for r in rows:
        if not r.get('link'):
            continue
        f = os.path.relpath(dest_for(a.root, r), a.root)
        if f in seen:
            continue
        seen.add(f)
        if f in done or missing.get(f) == r['link']:
            continue
        todo.append(r)
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(rows)} rows, {len(seen)} unique links, {len(done)} already done, {len(todo)} to fetch', flush=True)

    q = queue.Queue()
    for r in todo:
        q.put(r)
    lock = threading.Lock()
    mf = open(os.path.join(a.root, 'manifest.jsonl'), 'a')
    stats = {'ok': 0, 'missing': 0, 'err': 0, 'bytes': 0}

    def worker():
        while True:
            try:
                r = q.get_nowait()
            except queue.Empty:
                return
            url = BASE + quote(r['link'])
            dest = dest_for(a.root, r)
            rec = {k: r.get(k) for k in ('code', 'eff', 'name', 'eaip', 'airac', 'heading', 'link', 'auth')}
            rec['file'] = os.path.relpath(dest, a.root)
            try:
                status, ctype, data = fetch(url)
                rec['status'] = status
                rec['content_type'] = ctype
                if status == 200:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, 'wb') as f:
                        f.write(data)
                    rec['bytes'] = len(data)
                    rec['sha256'] = hashlib.sha256(data).hexdigest()
                    rec['verification'] = 'pdf-magic' if data[:5] == b'%PDF-' else f'NOT-PDF head={data[:8]!r}'
                    with lock:
                        stats['ok'] += 1; stats['bytes'] += len(data)
                else:
                    rec['verification'] = 'server-missing'
                    with lock:
                        stats['missing'] += 1
            except Exception as e:
                rec['status'] = None
                rec['error'] = str(e)[:200]
                with lock:
                    stats['err'] += 1
            rec['fetched'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            with lock:
                mf.write(json.dumps(rec, ensure_ascii=False) + '\n'); mf.flush()
                n = stats['ok'] + stats['missing'] + stats['err']
                if n % 100 == 0:
                    print(f"{n}/{len(todo)} ok={stats['ok']} missing={stats['missing']} err={stats['err']} {stats['bytes']/1e9:.2f} GB", flush=True)
            time.sleep(a.delay)

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    mf.close()
    print(f"done: ok={stats['ok']} missing={stats['missing']} err={stats['err']} {stats['bytes']/1e9:.2f} GB", flush=True)


if __name__ == '__main__':
    main()
