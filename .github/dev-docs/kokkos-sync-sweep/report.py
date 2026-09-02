#!/usr/bin/env python3
"""Rebuild the sweep table from the saved detector output.
   report.py <sweep.log> <sites.txt>"""
import sys, re, os
SP = "/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad"
LAB = re.compile(r'^\[stale\] +([A-Za-z_/][A-Za-z_0-9/]*:[A-Za-z_0-9]*)')
ABORT = re.compile(r'DualView "([^"]+)"')

def labels(path):
    if not os.path.exists(path): return None
    out = set()
    for line in open(path, errors="replace"):
        m = LAB.match(line)
        if m: out.add(m.group(1))
    return out

sites = [l.strip() for l in open(sys.argv[2]) if l.strip()]
log = open(sys.argv[1], errors="replace").read().split("\n")
cur = None
res = {}
for line in log:
    m = re.match(r'===== INJECTION (\d+) : (\S+)', line)
    if m: cur = int(m.group(1)); res[cur] = []
    m = re.match(r'  (\S+) : (\S+)', line)
    if m and cur: res[cur].append((m.group(1), m.group(2)))

for n, site in enumerate(sites, 1):
    rows = res.get(n, [])
    broke = [(t, r) for t, r in rows if r in ("DIVERGED", "CRASH")]
    if not broke:
        print(f"{n:2}. {site:44} inert in all 16 cases")
        continue
    detected = []
    for t, r in broke:
        base = labels(f"{SP}/inj/base.{t}.w.err")
        inj  = labels(f"{SP}/inj/n{n}.{t}.w.err")
        w    = 0
        ab   = set()
        p = f"{SP}/inj/n{n}.{t}.w.err"
        if os.path.exists(p):
            for l in open(p, errors="replace"):
                if l.startswith("[watch]"): w += 1
                m2 = ABORT.search(l)
                if m2 and "concurrent modification" in l: ab.add(m2.group(1))
        new = sorted(inj - base) if (base is not None and inj is not None) else []
        detected.append((t, r, w, new, sorted(ab)))
    kinds = sorted({r for _, r in broke})
    print(f"{n:2}. {site:44} {len(broke)}/16 broke ({','.join(kinds)})")
    for t, r, w, new, ab in detected:
        found = []
        if w: found.append(f"watch={w}")
        if new: found.append("stale:" + ",".join(new))
        if ab: found.append("abort:" + ",".join(ab))
        print(f"      {t:34} {r:9} {'  '.join(found) or 'NOT DETECTED'}")
