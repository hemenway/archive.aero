#!/usr/bin/env python3
"""Walk/extract FAT32 volumes inside raw SD-card images (read-only).

    fatwalk.py list IMG PATH [PATH...]      recursive listing (live + deleted)
    fatwalk.py extract IMG DIRPATH OUTDIR [--match REGEX]
    fatwalk.py carve IMG DIRPATH OUTDIR       recover DELETED files (contiguity)

Listing columns: L/D (live/deleted), size, write date, first cluster, path.
Extract copies live files (original names verbatim), follows FAT chains,
and refuses files whose chains run past the end of a truncated image.
"""
import os, re, struct, sys


class Fat32:
    def __init__(self, img_path):
        self.f = open(img_path, "rb")
        self.img_size = os.path.getsize(img_path)
        mbr = self.f.read(512)
        self.poff = struct.unpack_from("<I", mbr, 446 + 8)[0] * 512
        self.f.seek(self.poff)
        v = self.f.read(512)
        self.bps = struct.unpack_from("<H", v, 11)[0]
        self.spc = v[13]
        reserved = struct.unpack_from("<H", v, 14)[0]
        nfats = v[16]
        self.fatsz = struct.unpack_from("<I", v, 36)[0]
        self.root = struct.unpack_from("<I", v, 44)[0]
        totsec = struct.unpack_from("<I", v, 32)[0]
        self.clb = self.spc * self.bps
        self.data0 = self.poff + (reserved + nfats * self.fatsz) * self.bps
        self.nclus = (totsec - reserved - nfats * self.fatsz) // self.spc + 2
        self.f.seek(self.poff + reserved * self.bps)
        self.fat = self.f.read(self.fatsz * self.bps)

    def ent(self, c):
        return struct.unpack_from("<I", self.fat, c * 4)[0] & 0x0FFFFFFF

    def chain(self, start):
        c, seen = start, 0
        while 2 <= c < 0x0FFFFFF8 and seen < self.nclus:
            yield c
            c = self.ent(c)
            seen += 1

    def clus_off(self, c):
        return self.data0 + (c - 2) * self.clb

    def read_clus(self, c):
        off = self.clus_off(c)
        if off + self.clb > self.img_size:
            return None
        self.f.seek(off)
        return self.f.read(self.clb)

    def listdir(self, start):
        """Yield (name, attr, first_cluster, size, wdate, live) entries."""
        lfn, dlfn = [], []
        for c in self.chain(start):
            data = self.read_clus(c)
            if data is None:
                yield ("<DIR-TRUNCATED-BY-IMAGE>", 0, 0, 0, "", False)
                return
            for i in range(0, self.clb, 32):
                e = data[i:i + 32]
                if e[0] == 0:
                    return
                attr = e[11]
                if attr == 0x0F:
                    part = (e[1:11] + e[14:26] + e[28:32]).decode(
                        "utf-16le", "replace").split("\x00")[0]
                    (dlfn if e[0] == 0xE5 else lfn).insert(0, part)
                    continue
                first = (struct.unpack_from("<H", e, 20)[0] << 16
                         | struct.unpack_from("<H", e, 26)[0])
                size = struct.unpack_from("<I", e, 28)[0]
                wd = struct.unpack_from("<H", e, 24)[0]
                wdate = f"{1980 + (wd >> 9)}-{(wd >> 5) & 15:02d}-{wd & 31:02d}"
                if e[0] == 0xE5:
                    n83 = ("?" + e[1:8].decode("ascii", "replace").strip()
                           + ("." + e[8:11].decode("ascii", "replace").strip()
                              if e[8:11].strip() != b"" else ""))
                    yield ("".join(dlfn) or n83, attr, first, size, wdate, False)
                    dlfn = []
                    continue
                name = "".join(lfn) or (
                    e[0:8].decode("ascii", "replace").strip()
                    + ("." + e[8:11].decode("ascii", "replace").strip()
                       if e[8:11].strip() != b"" else ""))
                lfn = []
                if name not in (".", ".."):
                    yield (name, attr, first, size, wdate, True)

    def resolve(self, path):
        cur = self.root
        for part in [p for p in path.strip("/").split("/") if p]:
            hit = None
            for name, attr, first, *_ , live in self.listdir(cur):
                if live and name.lower() == part.lower() and attr & 0x10:
                    hit = first
                    break
            if hit is None:
                raise SystemExit(f"path component not found: {part!r}")
            cur = hit
        return cur

    def walk(self, start, prefix=""):
        for name, attr, first, size, wdate, live in self.listdir(start):
            p = f"{prefix}/{name}"
            if attr & 0x10 and live and first >= 2:
                yield (p + "/", attr, first, 0, wdate, live)
                yield from self.walk(first, p)
            else:
                yield (p, attr, first, size, wdate, live)

    def extract(self, first, size, out_path):
        need = size
        with open(out_path, "wb") as out:
            for c in self.chain(first):
                if need <= 0:
                    break
                data = self.read_clus(c)
                if data is None:
                    raise IOError("chain runs past end of truncated image")
                out.write(data[:min(self.clb, need)])
                need -= self.clb
        if need > 0:
            raise IOError(f"chain ended {need} bytes short of directory size")


def content_ok(name, blob):
    """'ok' when the carved bytes carry the format's own structure."""
    ext = os.path.splitext(name)[1].lower()
    if not blob:
        return "empty"
    if ext in (".jpg", ".jpeg"):
        return "ok" if blob[:2] == b"\xff\xd8" and blob.rstrip(b"\0")[-2:] == b"\xff\xd9" else "no JPEG SOI/EOI"
    if ext in (".tif", ".tiff"):
        if blob[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            return "no TIFF magic"
        if blob[:4] in (b"II*\x00", b"MM\x00*"):
            le = blob[:2] == b"II"
            off = struct.unpack("<I" if le else ">I", blob[4:8])[0]
            if off + 2 > len(blob) or struct.unpack("<H" if le else ">H", blob[off:off + 2])[0] == 0:
                return "first IFD unreadable"
        return "ok"
    if ext == ".zip":
        return "ok" if b"PK\x05\x06" in blob[-65557:] else "no ZIP end-of-central-directory"
    if ext == ".png":
        return "ok" if blob[:8] == b"\x89PNG\r\n\x1a\n" and blob.rstrip(b"\0")[-8:-4] == b"IEND" else "no PNG signature/IEND"
    if ext in (".gti", ".gtj", ".fil", ".fil2", ".txt", ".csv"):
        head = blob[:4096]
        return "ok" if all(32 <= b < 127 or b in (9, 10, 13) for b in head) else "non-text bytes in a text file"
    return "ok"  # unknown format: contiguity is all we can check


def main():
    cmd, img = sys.argv[1], sys.argv[2]
    fs = Fat32(img)
    if cmd == "list":
        for path in sys.argv[3:]:
            start = fs.resolve(path)
            for p, attr, first, size, wdate, live in fs.walk(start, "/" + path.strip("/")):
                kind = "D" if live else "x"
                print(f"{kind} {size:>11} {wdate} c{first:<8} {p}")
    elif cmd == "extract":
        dirpath, outdir = sys.argv[3], sys.argv[4]
        pat = re.compile(sys.argv[sys.argv.index("--match") + 1], re.I) \
            if "--match" in sys.argv else None
        os.makedirs(outdir, exist_ok=True)
        start = fs.resolve(dirpath)
        n = err = 0
        for name, attr, first, size, wdate, live in fs.listdir(start):
            if not live or attr & 0x10 or (pat and not pat.search(name)):
                continue
            try:
                fs.extract(first, size, os.path.join(outdir, name))
                n += 1
            except IOError as e:
                err += 1
                print(f"  SKIP {name}: {e}", file=sys.stderr)
        print(f"extracted {n} files to {outdir}" + (f" ({err} skipped)" if err else ""))

    elif cmd == "carve":
        # deleted files: FAT chain is zeroed on delete, so recover by
        # contiguity from the surviving (first_cluster, size) -- but only
        # when no cluster in that run has been reallocated to a live file,
        # AND the carved bytes pass the format's own structure check:
        # zeroed FAT entries prove nothing about content (the ACASIS lesson).
        # Windows FAT32 drivers zero DIR_FstClusHI on delete, so a deleted
        # entry whose first cluster reads < 65536 while the volume has more
        # clusters than that may be pointing 64K-cluster-multiples short of
        # its real data: try each candidate high word and keep the one whose
        # bytes validate.
        dirpath, outdir = sys.argv[3], sys.argv[4]
        os.makedirs(outdir, exist_ok=True)
        start = fs.resolve(dirpath)
        carved = total = 0
        for name, attr, first, size, wdate, live in fs.listdir(start):
            if live or attr & 0x10 or first < 2 or size == 0 or name.startswith("<"):
                continue
            total += 1
            n = -(-size // fs.clb)
            candidates = [first]
            if first < 0x10000 and fs.nclus > 0x10000:
                candidates += [first | (hi << 16) for hi in range(1, (fs.nclus >> 16) + 1)]
            result = None
            for cand in candidates:
                if cand + n > fs.nclus:
                    continue
                reused = sum(1 for c in range(cand, cand + n) if fs.ent(c) != 0)
                if reused:
                    result = result or ("reused", cand, reused)
                    continue
                chunks, need, truncated = [], size, False
                for c in range(cand, cand + n):
                    data = fs.read_clus(c)
                    if data is None:
                        truncated = True
                        break
                    chunks.append(data[:min(fs.clb, need)])
                    need -= fs.clb
                if truncated:
                    result = result or ("truncated", cand, 0)
                    continue
                blob = b"".join(chunks)
                verdict = content_ok(name, blob)
                if verdict == "ok":
                    result = ("ok", cand, blob)
                    break
                result = result if result and result[0] == "ok" else ("bad-content", cand, verdict)
            if result and result[0] == "ok":
                carved += 1
                cand, blob = result[1], result[2]
                note = "" if cand == first else f" (FstClusHI restored: c{first} -> c{cand})"
                print(f"{name:<36} {size:>10} {wdate} c{cand:<8} clean, content verified{note}")
                with open(os.path.join(outdir, name), "wb") as out:
                    out.write(blob)
            elif result:
                why = {"reused": f"REUSED {result[2]}/{n} clusters",
                       "truncated": "runs past end of image",
                       "bad-content": f"contiguous but content check failed: {result[2]}"}[result[0]]
                print(f"{name:<36} {size:>10} {wdate} c{first:<8} {why}")
            else:
                print(f"{name:<36} {size:>10} {wdate} c{first:<8} no cluster candidate fits the image")
        print(f"{carved}/{total} deleted files carved with verified content to {outdir}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
