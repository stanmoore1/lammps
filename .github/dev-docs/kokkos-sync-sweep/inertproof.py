#!/usr/bin/env python3
"""Prove which remaining sites cannot matter.

A sync only matters if the wrapper ever copies that array in that direction; a
claim only matters because it makes such a copy happen later.  The census counts
those copies on clean runs, so a site whose array never copied in its direction
is inert whatever a mutation would show -- no sampling, and no coverage given up.

Conservative on purpose.  The census is keyed by array and direction, not by
source line, so several sites collapse onto one entry and a single live one
keeps them all.  Anything whose array cannot be identified is left to the sweep.
"""
import re, collections
SRC = "/home/user/lammps/src/KOKKOS/"
SP  = "/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad/"

h2d, d2h, seen = set(), set(), set()
for ln in open(SP + "census.raw"):
    m = re.match(r'\[copies\] (\S+) h2d=(\d+) d2h=(\d+)', ln)
    if not m: continue
    lab, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    seen.add(lab)
    if a: h2d.add(lab)
    if b: d2h.add(lab)

# label -> the k_<name> spellings that could refer to it
bysuffix = collections.defaultdict(set)
for lab in seen:
    tail = lab.split(":")[-1]
    bysuffix[tail].add(lab)
    bysuffix[tail[2:] if tail.startswith("k_") else "k_" + tail].add(lab)

DIR = {"sync_device": h2d, "modify_host": h2d, "sync_host": d2h, "modify_device": d2h}

def site_info(path, line):
    try:
        t = open(SRC + path, errors="replace").readlines()[line - 1]
    except (OSError, IndexError):
        return None, None
    t = t.replace("; // INJECTED-BUG ", "")
    op = None
    for k in ("sync_host", "sync_device", "modify_host", "modify_device"):
        if k in t: op = k; break
    if op is None:
        m = re.search(r'\.(sync|modify)<\s*(\w+)', t)
        if m:
            dev = m.group(2) not in ("LMPHostType", "Host")
            op = m.group(1) + ("_device" if dev else "_host")
    v = re.search(r'\bk_(\w+)', t)
    return op, ("k_" + v.group(1)) if v else None

verdict = {}
for ln in open(SP + "campaign.state"):
    p = ln.split()
    if len(p) >= 2: verdict[p[0]] = p[1]

todo = [s.strip() for s in open(SP + "sites_reached_filemajor.txt")
        if s.strip() and not s.startswith("#") and s.strip() not in verdict]

proven, live, unknown = [], [], []
for s in todo:
    path, num = s.rsplit(":", 1)
    op, var = site_info(path, int(num))
    if not op or not var or var not in bysuffix:
        unknown.append(s); continue
    labs = bysuffix[var]
    (live if any(l in DIR[op] for l in labs) else proven).append(s)

print(f"remaining sites {len(todo)}")
print(f"  provably inert, no copy ever in that direction : {len(proven)}")
print(f"  live, the sweep still has to test them         : {len(live)}")
print(f"  array not identifiable, left to the sweep      : {len(unknown)}")
print(f"  => sweep shrinks to {len(live) + len(unknown)} sites "
      f"({100 * (len(live) + len(unknown)) // max(1, len(todo))}% of what is left)")

with open(SP + "sites_live.txt", "w") as f:
    for s in live + unknown: f.write(s + "\n")
with open(SP + "sites_proven_inert.txt", "w") as f:
    for s in proven: f.write(s + "\n")

# Does the proof agree with the mutations already run?  Every already-tested
# site the proof calls inert must have come back INERT, or the proof is wrong.
agree = disagree = 0
for s, v in verdict.items():
    path, num = s.rsplit(":", 1)
    op, var = site_info(path, int(num))
    if not op or not var or var not in bysuffix: continue
    if any(l in DIR[op] for l in bysuffix[var]): continue
    if v == "INERT": agree += 1
    else: disagree += 1; print(f"  CONTRADICTS: {s} proved inert but the sweep found {v}")
print(f"check against {agree + disagree} already-tested sites the proof calls inert: "
      f"{agree} agree, {disagree} contradict")
