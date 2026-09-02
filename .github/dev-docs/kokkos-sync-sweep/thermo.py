#!/usr/bin/env python3
"""Print the thermo rows of a LAMMPS screen dump: a leading integer step
   followed by at least two more numbers, skipping timing and histogram lines."""
import sys, re
num = re.compile(r'^\s*\d+((\s+-?\d+\.?\d*([eE][-+]?\d+)?){2,})\s*$')
for line in open(sys.argv[1], errors="replace"):
    if "%" in line or "CPU" in line: continue
    if num.match(line): print(" ".join(line.split()))
