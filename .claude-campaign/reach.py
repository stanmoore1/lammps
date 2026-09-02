#!/usr/bin/env python3
"""Split the site list into reached / unreached using gcov data from a
   --coverage build that has run the core cases.
   reach.py <cov-build-dir> <sites-file>"""
import subprocess, sys, os, json, collections

BUILD, SITES = sys.argv[1], sys.argv[2]
SP = "/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad"

sites = collections.defaultdict(list)
for l in open(SITES):
    l = l.strip()
    if not l or l.startswith("#"): continue
    f, n = l.rsplit(":", 1)
    sites[f].append(int(n))

objdir = None
for root, dirs, files in os.walk(BUILD):
    if any(f.endswith(".gcda") for f in files):
        objdir = root
        break
if not objdir:
    sys.exit("no .gcda found -- run the cases on the coverage build first")

reached, unreached, nodata = [], [], []
for fname, lines in sorted(sites.items()):
    gcda = os.path.join(objdir, fname + ".gcda")
    if not os.path.exists(gcda):
        nodata.extend(f"{fname}:{n}" for n in lines)
        continue
    r = subprocess.run(["gcov", "-i", "-t", gcda], capture_output=True, text=True, cwd=objdir)
    counts = {}
    for chunk in r.stdout.split("\n"):
        if not chunk.strip(): continue
        try: data = json.loads(chunk)
        except json.JSONDecodeError: continue
        for fl in data.get("files", []):
            if os.path.basename(fl["file"]) != fname: continue
            for ln in fl.get("lines", []):
                counts[ln["line_number"]] = ln["count"]
    for n in lines:
        # the call may span lines; count the site line and the next two
        c = max(counts.get(n, 0), counts.get(n + 1, 0), counts.get(n + 2, 0))
        (reached if c > 0 else unreached).append(f"{fname}:{n}")

open(SP + "/sites_reached.txt", "w").write("\n".join(reached) + "\n")
open(SP + "/sites_unreached.txt", "w").write("\n".join(unreached + nodata) + "\n")
print(f"reached {len(reached)}  unreached {len(unreached)}  no-gcda {len(nodata)}")
