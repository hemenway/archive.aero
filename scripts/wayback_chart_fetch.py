#!/usr/bin/env python3
"""Fetch FAA chart zips from the Wayback Machine into rawtiffs/.

Follows the existing rawtiffs convention: the downloaded zip is kept verbatim
under its flattened Wayback URL, and the .tif inside is extracted beside it as
<Name>_<TYPE>_<edition>.tif.  Sidecars (.tfw/.htm) stay in the retained zip.

Resumable: a target whose zip is already present and passes `unzip -t` is
skipped.  Every fetch is verified before it counts (HTTP 200, size within
tolerance of the CDX-recorded length, zip integrity, a .tif member present,
TIFF magic).  Nothing is written to rawtiffs until a download passes.
"""
import argparse, io, json, os, re, shutil, subprocess, sys, tempfile, time, zipfile
import urllib.request, urllib.error, urllib.parse

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")
DEST = "/Volumes/projects/rawtiffs"


def flat_name(ts, original):
    """web/<ts>/<url-without-scheme> with / -> _ , matching what is on disk."""
    bare = re.sub(r"^https?://", "", original)
    return ("web.archive.org_web_%s_%s" % (ts, bare)).replace("/", "_")


def cdx_captures(original, limit=8):
    """Other Wayback captures of the same URL (newest first), for hopping
    past a capture whose body the Wayback frontend truncates at an arbitrary
    offset (the 2026-07 lesson: retrying the same timestamp never helps)."""
    q = ("https://web.archive.org/cdx/search/cdx?url=%s&output=json&fl=timestamp,length,statuscode"
         "&filter=statuscode:200&collapse=digest&limit=%d&sort=reverse" % (urllib.parse.quote(original, safe=""), limit))
    try:
        req = urllib.request.Request(q, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            rows = json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return []
    return [{"ts": r[0], "size": int(r[1]) if r[1].isdigit() else None} for r in rows[1:]]


def zip_ok(path):
    """True when an on-disk zip passes a full integrity test (the docstring
    promise; existence alone let a truncated .part-less file count as done)."""
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def fetch(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def verify(blob, expect):
    """Return (tif_members, err). Verifies zip integrity and TIFF magic.

    A zip may hold several sheets (the helicopter-route zips carry East/West
    halves), so every .tif member is returned -- taking only the first would
    silently drop sheets.
    """
    if len(blob) < 500_000:
        return None, "short read (%d B)" % len(blob)
    if expect and abs(len(blob) - expect) > max(4096, expect * 0.02):
        return None, "size %d != CDX %d" % (len(blob), expect)
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as e:
        return None, "bad zip: %s" % e
    if zf.testzip() is not None:
        return None, "crc failure in %s" % zf.testzip()
    tifs = sorted(n for n in zf.namelist() if n.lower().endswith(".tif"))
    if not tifs:
        return None, "no .tif member (%s)" % ",".join(zf.namelist()[:4])
    for n in tifs:
        with zf.open(n) as fh:
            if fh.read(2) not in (b"II", b"MM"):
                return None, "member %s is not a TIFF" % n
    return tifs, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    targets = json.load(open(a.targets))
    if a.limit:
        targets = targets[:a.limit]

    done = set()
    if os.path.exists(a.manifest):
        for line in open(a.manifest):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok":
                done.add(r["zip"])

    mf = open(a.manifest, "a")
    ok = skip = fail = 0
    for i, t in enumerate(targets, 1):
        # Strip any Wayback prefix shape (http/https, with or without the
        # id_ flag) so the flat name never carries the archive host.
        original = re.sub(r"^https?://web\.archive\.org/web/\d+(?:[a-z]{2}_)?/", "", t["url"])
        zname = flat_name(t["ts"], original)
        zpath = os.path.join(a.dest, zname)
        label = "%s %s %d" % (t["loc_disp"].replace("_", " "), t["type"], t["edition"])

        if zname in done or (os.path.exists(zpath) and zip_ok(zpath)):
            skip += 1
            continue
        if os.path.exists(zpath):
            print("[%d/%d] %s: on-disk zip fails integrity test; refetching" % (i, len(targets), label))
            os.remove(zpath)
        if a.dry_run:
            print("would fetch %-40s (%6.1f MB)" % (os.path.basename(original),
                                                    t.get("size", 0) / 2**20))
            continue

        # Try the planned capture, then hop to other CDX captures of the same
        # URL: Wayback truncates responses at arbitrary offsets and a retry
        # of the same timestamp tends to truncate at the same place.
        err = None
        got = None
        attempts = [{"ts": t["ts"], "size": t.get("size"), "url": t["url"]}]
        for cap in cdx_captures(original):
            if cap["ts"] != t["ts"]:
                attempts.append({"ts": cap["ts"], "size": cap["size"],
                                 "url": "https://web.archive.org/web/%sid_/%s" % (cap["ts"], original)})
        for cap in attempts:
            for attempt in range(1, a.retries + 1):
                try:
                    st, blob = fetch(cap["url"])
                    if st != 200:
                        err = "HTTP %s" % st
                    else:
                        member, err = verify(blob, cap.get("size"))
                        if not err:
                            got = cap
                            break
                except Exception as e:                       # noqa: BLE001
                    err = "%s: %s" % (type(e).__name__, e)
                time.sleep(min(30, 2 ** attempt))
            if got:
                break
            print("[%d/%d] %s: capture %s failed (%s); trying another capture" % (i, len(targets), label, cap["ts"], err))
        if not got:
            fail += 1
            mf.write(json.dumps({"status": "fail", "zip": zname, "url": t["url"],
                                 "error": err, "captures_tried": [c["ts"] for c in attempts]}) + "\n"); mf.flush()
            print("[%d/%d] FAIL %s - %s" % (i, len(targets), label, err), flush=True)
            continue
        if got["ts"] != t["ts"]:
            zname = flat_name(got["ts"], original)
            zpath = os.path.join(a.dest, zname)

        # write only after verification passed
        tmp = tempfile.NamedTemporaryFile(dir=a.dest, delete=False, suffix=".part")
        tmp.write(blob); tmp.close()
        os.replace(tmp.name, zpath)
        wrote = []
        with zipfile.ZipFile(zpath) as zf:
            for m in member:
                out_name = os.path.basename(m).replace(" ", "_")
                op = os.path.join(a.dest, out_name)
                if os.path.exists(op):
                    # A same-named member from another capture/edition must
                    # not overwrite what is already on disk: disambiguate by
                    # the capture timestamp, never by clobbering.
                    stem, ext = os.path.splitext(out_name)
                    out_name = "%s__%s%s" % (stem, got["ts"], ext)
                    op = os.path.join(a.dest, out_name)
                with zf.open(m) as fh, open(op, "wb") as out:
                    shutil.copyfileobj(fh, out)
                wrote.append({"tif": out_name, "bytes": os.path.getsize(op)})
        ok += 1
        mf.write(json.dumps({"status": "ok", "zip": zname, "url": got["url"], "capture": got["ts"],
                             "bytes": len(blob), "extracted": wrote,
                             "loc": t["loc_disp"], "type": t["type"],
                             "edition": t["edition"]}) + "\n"); mf.flush()
        print("[%d/%d] ok   %-34s %6.1f MB  -> %s" % (
            i, len(targets), label, len(blob) / 2**20,
            ", ".join(w["tif"] for w in wrote)), flush=True)
        time.sleep(a.sleep)

    print("\nok=%d skipped=%d failed=%d" % (ok, skip, fail))


if __name__ == "__main__":
    main()
