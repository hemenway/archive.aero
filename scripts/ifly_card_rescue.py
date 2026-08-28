#!/usr/bin/env python3
"""Hunt deleted/hidden chart files on the iFly EFB card via a ddrescue image.

Companion to scripts/ifly_extract.py. The card has been rewritten by years of
iFly updates, so unallocated exFAT space may still hold chart editions that
were deleted by updaters and survive nowhere else. Workflow:

    1. image    ddrescue the raw device into an image file (+ mapfile).
                Raw device reads need root on macOS:
                  sudo ~/venv/bin/python scripts/ifly_card_rescue.py image \
                      --device /dev/rdisk6 --out DIR
    2. scan     parse the exFAT volume in the image: walk the live directory
                tree, enumerate DELETED entry sets (full UTF-16 names, sizes,
                timestamps, first cluster, NoFatChain flag survive deletion),
                descend into deleted directories, and signature-carve
                unallocated clusters for iFly artifacts (.gti "ImageName:",
                .fil index lines, .gtj headers). Writes scan_report.json.
    3. recover  extract every recoverable deleted file (NoFatChain files are
                contiguous by definition; FAT-chained files follow the chain
                when it survives, else contiguous fallback), flag clusters
                since reused by live files, validate chart bundles (.fil
                offsets must land on JPEG SOI markers in the .dat), and lay
                out recovered files card-style for ifly_extract.py --card-dir.

exFAT only (the card is exFAT). Pure stdlib. Reads the image, never the card
(except `image`, which only reads the device and only writes to --out).
"""

import argparse
import datetime
import json
import mmap
import os
import re
import struct
import subprocess
import sys

ENTRY = 32
TYPE_EOD = 0x00
# in-use types and their deleted (bit7-cleared) twins
T_BITMAP, T_UPCASE, T_LABEL, T_FILE, T_GUID, T_STREAM, T_NAME = (
    0x81, 0x82, 0x83, 0x85, 0xA0, 0xC0, 0xC1)
PLAUSIBLE_TYPES = {0x81, 0x82, 0x83, 0x85, 0xA0, 0xC0, 0xC1,
                   0x01, 0x02, 0x03, 0x05, 0x20, 0x40, 0x41, 0x00}


def dos_ts(v):
    if not v:
        return ""
    try:
        return datetime.datetime(
            1980 + (v >> 25), (v >> 21) & 0xF, (v >> 16) & 0x1F,
            (v >> 11) & 0x1F, (v >> 5) & 0x3F, (v & 0x1F) * 2).isoformat()
    except ValueError:
        return ""


class ExFat:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.base = self._find_boot()
        b = self.mm[self.base:self.base + 512]
        self.bps = 1 << b[108]
        self.spc = 1 << b[109]
        self.cs = self.bps * self.spc
        (self.fat_off,) = struct.unpack_from("<I", b, 80)
        (self.heap_off,) = struct.unpack_from("<I", b, 88)
        (self.cluster_count,) = struct.unpack_from("<I", b, 92)
        (self.root_cluster,) = struct.unpack_from("<I", b, 96)
        self.fat_base = self.base + self.fat_off * self.bps
        self.heap_base = self.base + self.heap_off * self.bps
        self.bitmap = None  # bytes; bit i = cluster i+2

    def _find_boot(self):
        def is_boot(off):
            return (self.mm[off + 3:off + 11] == b"EXFAT   "
                    and self.mm[off + 510:off + 512] == b"\x55\xaa")
        if is_boot(0):
            return 0
        # MBR partition entries
        if self.mm[510:512] == b"\x55\xaa":
            for i in range(4):
                (lba,) = struct.unpack_from("<I", self.mm, 446 + 16 * i + 8)
                if lba and lba * 512 + 512 <= len(self.mm) and is_boot(lba * 512):
                    return lba * 512
        # brute scan the first 64 MB on 512B alignment
        for off in range(0, min(len(self.mm), 64 << 20), 512):
            if is_boot(off):
                return off
        raise SystemExit("no exFAT boot sector found in image")

    def cl_off(self, cl):
        return self.heap_base + (cl - 2) * self.cs

    def cl_ok(self, cl):
        return 2 <= cl < 2 + self.cluster_count

    def fat(self, cl):
        (v,) = struct.unpack_from("<I", self.mm, self.fat_base + 4 * cl)
        return v

    def chain(self, first, size, nofat, max_clusters=1 << 20):
        """Cluster list for a file: contiguous if nofat, else follow FAT
        (contiguous fallback when the chain was wiped)."""
        n = max(1, -(-size // self.cs))
        if nofat:
            return list(range(first, first + n)), "contig"
        cls, cl = [], first
        for _ in range(min(n, max_clusters)):
            if not self.cl_ok(cl):
                break
            cls.append(cl)
            nxt = self.fat(cl)
            if nxt == 0xFFFFFFFF:  # end-of-chain marker: chain is authoritative
                return cls, "fat-chain"
            if len(cls) >= n:
                break
            if not self.cl_ok(nxt):  # wiped/invalid chain -> contiguous guess
                return list(range(first, first + n)), "contig-fallback"
            cl = nxt
        if len(cls) < n:
            return list(range(first, first + n)), "contig-fallback"
        return cls, "fat-chain"

    def read_clusters(self, cls, size):
        out = bytearray()
        for cl in cls:
            if not self.cl_ok(cl):
                out += b"\x00" * self.cs
                continue
            o = self.cl_off(cl)
            out += self.mm[o:o + self.cs]
        return bytes(out[:size])

    def allocated(self, cl):
        i = cl - 2
        return bool(self.bitmap[i >> 3] & (1 << (i & 7))) if self.bitmap else None


def parse_dir_clusters(fs, data):
    """Yield entry sets (live + deleted) from raw directory bytes.

    Returns list of dicts: {deleted, is_dir, name, size, first_cluster,
    nofat, mtime, ctime, attrs} plus orphan streams. Scans every 32-byte
    slot, including past the 0x00 end-of-directory marker (stale tails)."""
    out = []
    i, n = 0, len(data) - ENTRY + 1
    while i < n:
        t = data[i]
        if t in (T_FILE, 0x05):
            sec_count = data[i + 1]
            attrs = struct.unpack_from("<H", data, i + 4)[0]
            ctime = struct.unpack_from("<I", data, i + 8)[0]
            mtime = struct.unpack_from("<I", data, i + 12)[0]
            name, stream, j = "", None, i + ENTRY
            for _ in range(min(sec_count, 20)):
                if j >= len(data) - ENTRY + 1:
                    break
                st = data[j]
                if st in (T_STREAM, 0x40):
                    flags = data[j + 1]
                    name_len = data[j + 3]
                    vdl, = struct.unpack_from("<Q", data, j + 8)
                    fc, = struct.unpack_from("<I", data, j + 20)
                    dl, = struct.unpack_from("<Q", data, j + 24)
                    stream = {"first_cluster": fc, "size": dl or vdl,
                              "nofat": bool(flags & 2), "name_len": name_len}
                elif st in (T_NAME, 0x41):
                    name += data[j + 2:j + 32].decode("utf-16-le", "replace")
                else:
                    break
                j += ENTRY
            if stream:
                nl = stream.pop("name_len", 0)
                if nl:
                    name = name[:nl]
                out.append({
                    "deleted": t == 0x05, "is_dir": bool(attrs & 0x10),
                    "name": name.rstrip("\x00"), "attrs": attrs,
                    "mtime": dos_ts(mtime), "ctime": dos_ts(ctime), **stream})
            i = j
            continue
        if t in (T_STREAM, 0x40) and data[i - ENTRY] not in (T_FILE, 0x05):
            fc, = struct.unpack_from("<I", data, i + 20)
            dl, = struct.unpack_from("<Q", data, i + 24)
            if fs.cl_ok(fc) and dl:
                out.append({"deleted": True, "is_dir": False, "orphan": True,
                            "name": f"_orphan_stream_cl{fc}", "attrs": 0,
                            "mtime": "", "ctime": "", "first_cluster": fc,
                            "size": dl, "nofat": bool(data[i + 1] & 2)})
        i += ENTRY
    return out


def looks_like_dir(fs, cl):
    if not fs.cl_ok(cl):
        return False
    o = fs.cl_off(cl)
    head = fs.mm[o:o + ENTRY * 8]
    types = head[::ENTRY]
    return all(t in PLAUSIBLE_TYPES for t in types) and any(t for t in types)


def walk(fs, first, size, nofat, path, seen, report, depth=0):
    if depth > 12 or first in seen:
        return
    seen.add(first)
    cls, how = fs.chain(first, size or fs.cs * 256, nofat)
    # a directory's true size is unknown for the root; stop at unallocated
    # clusters that no longer parse as directory data
    data = bytearray()
    for cl in cls:
        if not fs.cl_ok(cl):
            break
        if fs.bitmap is not None and not fs.allocated(cl) and not looks_like_dir(fs, cl):
            break
        data += fs.mm[fs.cl_off(cl):fs.cl_off(cl) + fs.cs]
    for e in parse_dir_clusters(fs, bytes(data)):
        e["path"] = f"{path}/{e['name']}"
        e["walk"] = how
        report.append(e)
        if e["is_dir"] and fs.cl_ok(e["first_cluster"]):
            if not e["deleted"] or looks_like_dir(fs, e["first_cluster"]):
                walk(fs, e["first_cluster"], e["size"], e["nofat"],
                     e["path"], seen, report, depth + 1)


def find_bitmap(fs):
    cls, _ = fs.chain(fs.root_cluster, fs.cs * 256, False)
    for cl in cls:
        if not fs.cl_ok(cl):
            break
        o = fs.cl_off(cl)
        data = fs.mm[o:o + fs.cs]
        for i in range(0, len(data) - ENTRY + 1, ENTRY):
            if data[i] == T_BITMAP:
                fc, = struct.unpack_from("<I", data, i + 20)
                dl, = struct.unpack_from("<Q", data, i + 24)
                bcls, _ = fs.chain(fc, dl, True)
                fs.bitmap = fs.read_clusters(bcls, dl)
                return
        if data[0] == 0:
            break


GTI_RE = re.compile(rb"ImageName:[^\r\n]{1,120}\r?\n")
FIL_RE = re.compile(rb"(?:\d{1,2}-\d{2,3}-\d{2,3}\.JPG,\d{1,12},\d{1,9}\r\n){4,}")
GTJ_RE = re.compile(rb"(?:Sec_|Tac_|Wac_|ENR_)[A-Za-z0-9_]{2,40}\r\n\d{3,5}\r\n\d{3,5}\r\n")


def carve_unallocated(fs):
    hits = []
    step = fs.cs
    for cl in range(2, 2 + fs.cluster_count):
        if fs.allocated(cl):
            continue
        o = fs.cl_off(cl)
        chunk = fs.mm[o:o + step]
        if not chunk.strip(b"\x00"):
            continue
        for kind, rx in (("gti", GTI_RE), ("fil", FIL_RE), ("gtj", GTJ_RE)):
            m = rx.search(chunk)
            if m:
                ctx = chunk[m.start():m.start() + 700]
                hits.append({"cluster": cl, "offset": o + m.start(),
                             "kind": kind,
                             "preview": ctx.decode("ascii", "replace")[:400]})
                break
    return hits


def cmd_image(args):
    dev = args.device
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    img = os.path.join(out_dir, "card.img")
    mapf = os.path.join(out_dir, "card.map")
    disk = re.sub(r"^/dev/r?", "/dev/", dev).rstrip("0123456789") \
        if re.search(r"s\d+$", dev) else re.sub(r"^/dev/r?", "/dev/", dev)
    if os.geteuid() != 0:
        print("Raw device reads need root. Run:\n"
              f"  sudo {sys.executable} {os.path.abspath(__file__)} "
              f"image --device {dev} --out {out_dir}")
        return 1
    info = subprocess.run(["diskutil", "info", disk], capture_output=True,
                          text=True).stdout
    print("".join(l + "\n" for l in info.splitlines()
                  if re.search(r"Device Node|Media Name|Disk Size|Protocol|Removable", l)))
    if not args.yes:
        if input(f"Image {dev} (read-only) to {img}? [y/N] ").strip().lower() != "y":
            print("aborted")
            return 1
    subprocess.run(["diskutil", "unmountDisk", disk], check=False)
    # ddrescue INFILE OUTFILE MAPFILE -- device is strictly read-only input
    rc = subprocess.run(["ddrescue", "-b", "512", "-r3", dev, img, mapf]).returncode
    subprocess.run(["diskutil", "mountDisk", disk], check=False)
    if args.chown and rc == 0:
        subprocess.run(["chown", args.chown, img, mapf], check=False)
    return rc


def cmd_scan(args):
    fs = ExFat(args.image)
    print(f"exFAT @ {fs.base:#x}: cluster {fs.cs//1024} KB x {fs.cluster_count}, "
          f"root cl {fs.root_cluster}")
    find_bitmap(fs)
    alloc = sum(bin(b).count("1") for b in fs.bitmap) if fs.bitmap else -1
    print(f"allocation bitmap: {alloc}/{fs.cluster_count} clusters allocated")
    entries = []
    walk(fs, fs.root_cluster, 0, False, "", set(), entries)
    live = [e for e in entries if not e["deleted"]]
    dele = [e for e in entries if e["deleted"]]
    print(f"directory walk: {len(live)} live entries, {len(dele)} deleted entries")
    print("carving unallocated clusters for iFly signatures...")
    carves = carve_unallocated(fs)
    print(f"{len(carves)} signature hits in unallocated space")
    rep = {"image": os.path.abspath(args.image), "exfat_base": fs.base,
           "cluster_size": fs.cs, "cluster_count": fs.cluster_count,
           "allocated_clusters": alloc, "entries": entries, "carves": carves}
    out = args.report or os.path.join(os.path.dirname(os.path.abspath(args.image)),
                                      "scan_report.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"report -> {out}")
    for e in sorted(dele, key=lambda e: -e["size"])[:40]:
        reuse = ""
        if fs.bitmap and fs.cl_ok(e["first_cluster"]):
            cls, _ = fs.chain(e["first_cluster"], e["size"], True)
            hit = sum(1 for c in cls if fs.cl_ok(c) and fs.allocated(c))
            reuse = f" reused={hit}/{len(cls)}"
        print(f"  DEL {e['path']}  {e['size']:,}B  cl{e['first_cluster']}"
              f" {'nofat' if e['nofat'] else 'fat'} mtime={e['mtime']}{reuse}")
    return 0


def jpeg_ok(buf, off, ln):
    return buf[off:off + 2] == b"\xff\xd8" if off + ln <= len(buf) else False


def cmd_recover(args):
    fs = ExFat(args.image)
    find_bitmap(fs)
    rep = json.load(open(args.report))
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    recovered, manifest = 0, []
    for e in rep["entries"]:
        if not e["deleted"] or e["is_dir"] or not e["size"]:
            continue
        if not fs.cl_ok(e["first_cluster"]):
            continue
        if args.max_bytes and e["size"] > args.max_bytes:
            continue
        cls, how = fs.chain(e["first_cluster"], e["size"], e["nofat"])
        reused = sum(1 for c in cls if fs.cl_ok(c) and fs.allocated(c))
        data = fs.read_clusters(cls, e["size"])
        rel = e["path"].lstrip("/").replace("..", "_")
        dst = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(data)
        manifest.append({**{k: e[k] for k in
                            ("path", "size", "first_cluster", "nofat", "mtime")},
                         "chain": how, "reused_clusters": reused,
                         "n_clusters": len(cls), "out": dst})
        recovered += 1
    # validate chart triples: .fil offsets must hit JPEG SOI in the .dat
    by_stem = {}
    for m in manifest:
        stem, ext = os.path.splitext(m["out"])
        by_stem.setdefault(stem, {})[ext.lower()] = m
    for stem, exts in by_stem.items():
        if ".fil" in exts and ".dat" in exts:
            fil = open(exts[".fil"]["out"], "rb").read().decode("ascii", "replace")
            dat = open(exts[".dat"]["out"], "rb").read()
            offs = []
            for line in fil.splitlines():
                p = line.strip().split(",")
                if len(p) == 3 and p[1].isdigit() and p[2].isdigit():
                    offs.append((int(p[1]), int(p[2])))
            good = sum(1 for o, l in offs if jpeg_ok(dat, o, l))
            exts[".dat"]["tiles_valid"] = f"{good}/{len(offs)}"
            print(f"  {os.path.basename(stem)}: {good}/{len(offs)} tile offsets "
                  f"land on JPEG markers"
                  + (f" (reused clusters: {exts['.dat']['reused_clusters']})"
                     if exts[".dat"]["reused_clusters"] else ""))
    with open(os.path.join(out_root, "recover_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"recovered {recovered} deleted files -> {out_root}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("image", help="ddrescue the card to an image (needs sudo)")
    p.add_argument("--device", required=True, help="raw device, e.g. /dev/rdisk6")
    p.add_argument("--out", required=True, help="directory for card.img + card.map")
    p.add_argument("--chown", help="user[:group] to chown the image to afterwards")
    p.add_argument("--yes", action="store_true", help="skip the confirm prompt")
    p = sub.add_parser("scan", help="find deleted files + carve unallocated space")
    p.add_argument("--image", required=True)
    p.add_argument("--report", help="output JSON (default: beside image)")
    p = sub.add_parser("recover", help="extract deleted files from the image")
    p.add_argument("--image", required=True)
    p.add_argument("--report", required=True, help="scan_report.json from scan")
    p.add_argument("--out", required=True)
    p.add_argument("--max-bytes", type=int, default=0, help="skip files larger than this")
    args = ap.parse_args()
    sys.exit({"image": cmd_image, "scan": cmd_scan, "recover": cmd_recover}[args.cmd](args))


if __name__ == "__main__":
    main()
