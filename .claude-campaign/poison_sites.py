#!/usr/bin/env python3
"""Summarise the distinct stale accesses in an ASan survey log: one row per
   (fault kind, first LAMMPS frame), with a count."""
import sys, re, collections
FRAME = re.compile(r'^\s*#\d+\s+0x[0-9a-f]+\s+in\s+(.+?)\s+(/\S+|\(.*)')
rows = collections.Counter()
cur = None
frames = []
def flush():
    global cur, frames
    if cur:
        site = "unknown"
        for f in frames:
            if "LAMMPS_NS" in f or "/src/" in f:
                site = f
                break
        rows[(cur, site[:110])] += 1
    cur, frames = None, []
for line in open(sys.argv[1], errors="replace"):
    if "ERROR: AddressSanitizer" in line:
        flush()
        m = re.search(r'AddressSanitizer:\s*(\S+)', line)
        cur = m.group(1) if m else "error"
    elif cur is not None:
        m = FRAME.match(line)
        if m: frames.append(m.group(1))
        elif line.strip() == "" and frames: flush()
flush()
for (kind, site), n in rows.most_common():
    print(f"{n:6}  {kind:22} {site}")
print(f"# {len(rows)} distinct sites, {sum(rows.values())} reports")
