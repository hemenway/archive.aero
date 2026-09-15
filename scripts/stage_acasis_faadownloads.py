#!/usr/bin/env python3
"""Stage only-copy FAA chart originals off the ACASIS FAADownloads tree.

The tree is a file-level copy of the iFly build machine's D: volume -- the same
volume the paused `image.imgcv2` acquisition covers, byte-verified against the
carved set.  Charts that are the only copy ever seen AND slicer-ready as-is go
into rawtiffs/ per the admission test in CLAUDE.md; everything else stays in
the attic.  Sidecars (.tfw/.htm) travel with the .tif because here there is no
retained container to recover them from -- the .htm carries the FAA's own
Beginning_Date/Ending_Date, which become the row's date/end_date.
"""
import argparse, hashlib, json, os, re, shutil, sys

SRC_ROOT = "/Volumes/ACASIS/FAADownloads"
DEST     = "/Volumes/projects/rawtiffs/acasis_faadownloads"
DATE_RE  = re.compile(r"(Publication|Beginning|Ending)_Date:\s*(\d{8})")


def sha256(path, buf=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def htm_dates(path):
    if not os.path.exists(path):
        return None, None
    txt = re.sub(r"<[^>]*>", " ", open(path, encoding="utf-8", errors="replace").read())
    d = {}
    for k, v in DATE_RE.findall(txt):
        d.setdefault(k, v)
    fmt = lambda s: f"{s[:4]}-{s[4:6]}-{s[6:]}" if s else None
    return fmt(d.get("Beginning") or d.get("Publication")), fmt(d.get("Ending"))


def tiff_structure(path):
    """'ok' when the file has a TIFF magic, a readable first IFD with entries,
    and a non-zero tail (a captured-but-reallocated file often ends in zeros)."""
    import struct
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(8)
            if head[:4] in (b"II*\x00", b"MM\x00*"):
                le = head[:2] == b"II"
                off = struct.unpack("<I" if le else ">I", head[4:8])[0]
                f.seek(off)
                n = f.read(2)
                if len(n) < 2 or struct.unpack("<H" if le else ">H", n)[0] == 0:
                    return "empty first IFD"
            elif head[:4] in (b"II+\x00", b"MM\x00+"):
                pass  # BigTIFF: magic only
            else:
                return "no TIFF magic"
            f.seek(max(0, size - 4096))
            if not f.read(4096).strip(b"\0"):
                return "zero tail"
        return "ok"
    except OSError as e:
        return str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_list")
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    items = json.load(open(a.stage_list))
    if not a.dry_run:
        os.makedirs(a.dest, exist_ok=True)
    mpath = os.path.join(a.dest, "manifest.jsonl")
    seen = set()
    if os.path.exists(mpath):
        for line in open(mpath):
            try:
                seen.add(json.loads(line)["dest"])
            except (ValueError, KeyError):
                pass

    mf = None if a.dry_run else open(mpath, "a")
    ok = skip = bad = 0
    seen_sources = {}  # basename -> source rel path, to catch same-name collisions within a run
    for it in items:
        rel = it["path"][2:] if it["path"].startswith("./") else it["path"]
        src_tif = os.path.join(SRC_ROOT, rel)
        stem = os.path.splitext(src_tif)[0]
        # Original filename VERBATIM (salvage convention): no space->underscore
        # renaming. Two sources sharing a basename are disambiguated by
        # appending the source directory, never by overwriting the first.
        dest_name = os.path.basename(src_tif)
        if dest_name in seen_sources and seen_sources[dest_name] != rel:
            dest_name = "%s__%s%s" % (os.path.splitext(dest_name)[0],
                                      os.path.dirname(rel).replace("/", "_") or "root",
                                      os.path.splitext(dest_name)[1])
        dest_tif = os.path.join(a.dest, dest_name)

        if dest_name in seen and os.path.exists(dest_tif):
            skip += 1
            continue
        if not os.path.exists(src_tif):
            print("MISSING SOURCE %s" % src_tif); bad += 1; continue
        if a.dry_run:
            print("would stage %-40s (%6.1f MB)" % (dest_name, it["size"] / 2**20)); continue

        tiff_check = tiff_structure(src_tif)
        if tiff_check != "ok":
            print("TIFF STRUCTURE FAILED (%s): %s" % (tiff_check, src_tif)); bad += 1; continue
        src_hash = sha256(src_tif)
        shutil.copy2(src_tif, dest_tif)
        if sha256(dest_tif) != src_hash:
            os.remove(dest_tif)
            print("HASH MISMATCH after copy: %s" % dest_name); bad += 1; continue
        seen_sources[os.path.basename(src_tif)] = rel

        sidecars = []
        for ext in (".tfw", ".htm"):
            s = stem + ext
            if os.path.exists(s):
                d = os.path.join(a.dest, os.path.splitext(dest_name)[0] + ext)
                shutil.copy2(s, d)
                sidecars.append(os.path.basename(d))
        beg, end = htm_dates(stem + ".htm")
        st = os.stat(src_tif)
        mf.write(json.dumps({
            "dest": dest_name,
            "source_path": "D:\\FAADownloads\\" + rel.replace("/", "\\"),
            "source_volume": "ACASIS copy of iFly build machine D: (WD-WCC1T0799404)",
            "source_inode": st.st_ino, "source_mtime": int(st.st_mtime),
            "state": "live",
            "bytes": st.st_size, "sha256": src_hash,
            "verified": "TIFF magic + first IFD + non-zero tail checked on source; sha256 match after copy",
            "sidecars": sidecars,
            "place": it["place"], "type": it["type"], "edition": it["edition"],
            "faa_date": beg, "faa_end_date": end,
            "rescue_from_unrecoverable": bool(it.get("rescue")),
        }) + "\n"); mf.flush()
        ok += 1
        print("staged %-42s %6.1f MB  %s..%s  %s" % (
            dest_name, st.st_size / 2**20, beg or "?", end or "?", ",".join(sidecars)), flush=True)

    print("\nstaged=%d skipped=%d failed=%d" % (ok, skip, bad))


if __name__ == "__main__":
    main()
