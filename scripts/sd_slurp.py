#!/usr/bin/env python3
"""Slurp a stack of SD cards to ddrescue images — hot-swap, hands-free.

Insert cards, run this (auto re-execs under sudo — raw reads need root),
confirm the first batch, and from then on it's fully automatic: each card
images in its own parallel ddrescue (per-card .img + .map + .info.txt +
.log), and the moment ONE card finishes it is ejected and announced safe to
swap — while the other slot keeps imaging. Newly inserted cards are
auto-detected (two consecutive scans, ~5 s debounce) and start immediately.
No Enter needed after the first round.

Ending it: q+Enter stops accepting new cards and exits once running images
finish; Ctrl-C interrupts immediately (ddrescue saves its mapfile, and the
same physical card — matched by volume UUID — resumes next time).

Safety: only external physical disks no bigger than --max-gb (default 1024)
are ever considered, the disk hosting the output folder is always excluded,
and a card whose finished image already exists in the output folder is
skipped with a note (delete its .map to force a re-slurp).

Usage:
    sd_slurp.py                       # default output folder
    sd_slurp.py --out DIR             # elsewhere
    sd_slurp.py --dry-run             # show what would happen, no root needed
"""

import argparse
import datetime
import os
import plistlib
import pwd
import re
import select
import shutil
import subprocess
import sys
import time

DEFAULT_OUT = "/Volumes/ACASIS/All SD cards/2026-08-20"
DDRESCUE = shutil.which("ddrescue") or "/opt/homebrew/bin/ddrescue"
DEBOUNCE_SCANS = 2      # consecutive sightings before a new card starts
SCAN_EVERY = 3.0        # seconds between diskutil scans
TICK = 1.0              # seconds between progress/stdin polls


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def plist(*args):
    out = subprocess.run(args, capture_output=True).stdout
    return plistlib.loads(out) if out.strip() else {}


def out_parent_disk(out_dir):
    """Whole-disk identifier hosting the output folder (e.g. 'disk7')."""
    probe = out_dir
    while not os.path.exists(probe):
        probe = os.path.dirname(probe) or "/"
    dev = sh("df", "-P", probe).stdout.strip().split("\n")[-1].split()[0]
    m = re.match(r"/dev/(disk\d+)", dev)
    return re.sub(r"s\d+$", "", m.group(1)) if m else ""


def detect_cards(out_dir, max_bytes):
    exclude = out_parent_disk(out_dir)
    cards = []
    for d in plist("diskutil", "list", "-plist", "external", "physical") \
            .get("AllDisksAndPartitions", []):
        dev = d.get("DeviceIdentifier", "")
        size = d.get("Size", 0)
        if not dev or dev == exclude or size > max_bytes or size == 0:
            continue
        labels, uuid = [], ""
        for p in d.get("Partitions", []):
            if p.get("VolumeName"):
                labels.append(p["VolumeName"])
            uuid = uuid or p.get("VolumeUUID", "")
        info = plist("diskutil", "info", "-plist", dev)
        cards.append({
            "dev": dev, "size": size,
            "label": " ".join(labels) or "UNTITLED",
            "uuid": uuid.replace("-", "").lower(),
            "media": info.get("MediaName", ""),
        })
    return cards


def gb(size):
    return f"{size/1e9:.1f}GB"


def sanitize(s):
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9.-]+", "_", s)).strip("_") or "UNTITLED"


def next_seq(out_dir):
    seqs = [int(m.group(1)) for f in os.listdir(out_dir)
            if (m := re.match(r"(\d{3})_", f))]
    return max(seqs, default=0) + 1


def _maps_for(out_dir, card):
    if not card["uuid"]:
        return []
    tag = f"_{card['uuid'][:8]}"
    return [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))
            if f.endswith(".map") and tag in f]


def find_resume(out_dir, card):
    """An unfinished image of this same card (matched by volume UUID)."""
    for mapf in _maps_for(out_dir, card):
        if "Finished" not in open(mapf, errors="replace").read():
            return mapf[:-4]
    return None


def find_finished(out_dir, card):
    """A COMPLETED image of this same card, to avoid auto re-slurping."""
    for mapf in _maps_for(out_dir, card):
        if "Finished" in open(mapf, errors="replace").read():
            return mapf[:-4]
    return None


def card_basename(out_dir, card, seq):
    name = f"{seq:03d}_{sanitize(card['label'])}_{gb(card['size'])}"
    if card["uuid"]:
        name += f"_{card['uuid'][:8]}"
    return os.path.join(out_dir, name)


def chown_to_user(paths):
    su = os.environ.get("SUDO_USER")
    if not su or os.geteuid() != 0:
        return
    u = pwd.getpwnam(su)
    for p in paths:
        try:
            os.chown(p, u.pw_uid, u.pw_gid)
        except OSError:
            pass


def start_job(card, out_dir):
    base = find_resume(out_dir, card)
    resumed = base is not None
    if base is None:
        base = card_basename(out_dir, card, next_seq(out_dir))
    img, mapf = base + ".img", base + ".map"
    with open(base + ".info.txt", "w") as fh:
        fh.write(f"# sd_slurp {datetime.datetime.now().isoformat()}\n")
        fh.write(sh("diskutil", "info", card["dev"]).stdout)
    sh("diskutil", "unmountDisk", card["dev"])
    log = open(base + ".log", "a")
    proc = subprocess.Popen(
        [DDRESCUE, "-b", "512", "-r3", f"/dev/r{card['dev']}", img, mapf],
        stdout=log, stderr=subprocess.STDOUT)
    print(f"\n  + {card['dev']} {card['label']} -> {os.path.basename(img)}"
          + (" (resuming unfinished image)" if resumed else ""))
    return {"card": card, "proc": proc, "img": img, "mapf": mapf,
            "base": base, "log": log,
            "tag": f"{card['dev']} {card['label']}"}


def reap_job(j):
    """Handle a finished ddrescue: report, eject, chown. Returns success."""
    j["log"].close()
    rc = j["proc"].returncode
    finished = os.path.exists(j["mapf"]) and \
        "Finished" in open(j["mapf"], errors="replace").read()
    if rc == 0 and finished:
        sh("diskutil", "eject", j["card"]["dev"])
        print(f"\n  ✓ {j['tag']}: complete — ejected, SWAP CARD NOW "
              f"(next card auto-starts)")
    elif not finished:
        print(f"\n  ✗ {j['tag']}: interrupted — will resume when this card "
              f"is next inserted")
    else:
        print(f"\n  ✗ {j['tag']}: ddrescue exit {rc} — check {j['base']}.log")
    chown_to_user([j["img"], j["mapf"], j["base"] + ".info.txt",
                   j["base"] + ".log"])
    return rc == 0 and finished


def progress_line(active):
    parts = []
    for j in active:
        try:
            pct = min(100.0, os.path.getsize(j["img"]) * 100.0
                      / j["card"]["size"])
        except OSError:
            pct = 0.0
        parts.append(f"{j['tag']}: {pct:5.1f}%")
    return "  " + "   ".join(parts) if parts else "  (idle — insert a card)"


def pick(cards):
    for i, c in enumerate(cards, 1):
        print(f"  [{i}] {c['dev']:<8} {c['label']:<20} {gb(c['size']):>8}"
              f"  {c['media']}")
    while True:
        ans = input("Enter/all=slurp all, numbers (e.g. 1,2)=subset, "
                    "q=quit: ").strip().lower()
        if ans in ("q", "quit"):
            return "quit"
        if not ans or "all" in ans:
            return cards
        try:
            return [cards[int(n) - 1] for n in re.split(r"[,\s]+", ans)]
        except (ValueError, IndexError):
            print("didn't understand — try again")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output folder (default: {DEFAULT_OUT})")
    ap.add_argument("--max-gb", type=float, default=1024,
                    help="ignore disks bigger than this (default 1024)")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect + print what would start; no root needed")
    args = ap.parse_args()
    max_bytes = int(args.max_gb * 1e9)

    if not args.dry_run and os.geteuid() != 0:
        print("raw device reads need root — re-running under sudo...")
        os.execvp("sudo", ["sudo", sys.executable, os.path.abspath(__file__)]
                  + sys.argv[1:])

    vol = "/" + "/".join(args.out.strip("/").split("/")[:2])
    if args.out.startswith("/Volumes/") and not os.path.ismount(vol):
        sys.exit(f"output volume {vol} is not mounted")
    os.makedirs(args.out, exist_ok=True)
    chown_to_user([args.out, os.path.dirname(args.out)])

    print("=== scanning for cards ===")
    cards = detect_cards(args.out, max_bytes)
    if args.dry_run:
        for c in cards:
            print(f"  would slurp {c['dev']} {c['label']} {gb(c['size'])}")
        return

    if cards:
        sel = pick(cards)
        if sel == "quit":
            return
    else:
        print("no cards inserted yet — going straight to auto mode")
        sel = []

    print("\nAUTO MODE: swap cards anytime, new cards start automatically.\n"
          "q+Enter = finish running images then exit; Ctrl-C = interrupt "
          "(resumable).")

    active = [start_job(c, args.out) for c in sel]
    done_keys = {}      # uuid/dev -> reason, to skip re-slurping this session
    sightings = {}      # dev -> consecutive scans seen (debounce)
    quitting = False
    last_scan = 0.0
    try:
        while True:
            # reap finished jobs
            for j in [j for j in active if j["proc"].poll() is not None]:
                active.remove(j)
                reap_job(j)
                key = j["card"]["uuid"] or j["card"]["dev"]
                done_keys[key] = "done this session"
            if quitting and not active:
                break
            # stdin: q to quit
            r, _, _ = select.select([sys.stdin], [], [], TICK)
            if r:
                if sys.stdin.readline().strip().lower() in ("q", "quit"):
                    quitting = True
                    if active:
                        print("\n  q: finishing running image(s), no new "
                              "cards will start...")
                    else:
                        break
            # scan for new cards
            if not quitting and time.monotonic() - last_scan >= SCAN_EVERY:
                last_scan = time.monotonic()
                seen_now = set()
                for card in detect_cards(args.out, max_bytes):
                    dev = card["dev"]
                    seen_now.add(dev)
                    if any(j["card"]["dev"] == dev for j in active):
                        continue
                    key = card["uuid"] or dev
                    if key in done_keys:
                        if done_keys[key] != "notified":
                            print(f"\n  - {dev} {card['label']}: already "
                                  f"slurped ({done_keys[key]}) — remove it")
                            done_keys[key] = "notified"
                        continue
                    fin = find_finished(args.out, card)
                    if fin:
                        print(f"\n  - {dev} {card['label']}: finished image "
                              f"already exists ({os.path.basename(fin)}) — "
                              f"skipping (delete its .map to re-slurp)")
                        done_keys[key] = "notified"
                        continue
                    sightings[dev] = sightings.get(dev, 0) + 1
                    if sightings[dev] >= DEBOUNCE_SCANS:
                        sightings.pop(dev, None)
                        active.append(start_job(card, args.out))
                # reset debounce for devices that vanished
                for dev in list(sightings):
                    if dev not in seen_now:
                        sightings.pop(dev, None)
            print(f"\r{progress_line(active)} ", end="", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — waiting for ddrescue to save mapfiles...")
        for j in active:
            j["proc"].wait()
            reap_job(j)
    print(f"\ndone — images in {args.out}")


if __name__ == "__main__":
    main()
