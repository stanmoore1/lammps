#!/usr/bin/env python3
"""Compare the thermo output of two LAMMPS logs.

Two things this gets right that a plain diff of the two files does not, both of
which produced false findings before:

  - wall-clock columns are dropped.  S/CPU and friends differ on every pair of
    runs, so any input whose thermo_style names one looks like a divergence and
    is not one.
  - only complete thermo rows are compared, matched to their header, so a log
    that stopped early is reported as such rather than silently comparing equal.

Prints SAME, DIFFER or NO-OUTPUT.  Pair it with a run of each build twice: an
input that does not reproduce itself cannot be compared against anything, and
several of those were written up as divergences before this was checked.
"""

import sys, re

TIMING={'S/CPU','CPU','CPULeft','T/CPU','Elapsed','WallTime','CPUleft'}
def rows(fn):
    out=[]; cols=None
    try: lines=open(fn,errors='ignore').read().split('\n')
    except OSError: return None,None
    for i,l in enumerate(lines):
        if l.strip().startswith('Step') and re.match(r'^\s+Step\s',l):
            cols=l.split(); keep=[j for j,c in enumerate(cols) if c not in TIMING]; continue
        if cols and re.match(r'^\s+[0-9]+\s+[-0-9.]',l):
            p=l.split()
            if len(p)==len(cols): out.append(tuple(p[j] for j in keep))
    return out,cols
a,_=rows(sys.argv[1]); b,_=rows(sys.argv[2])
if a is None or b is None or not a or not b: print("NO-OUTPUT"); sys.exit()
print("SAME" if a==b else "DIFFER")
