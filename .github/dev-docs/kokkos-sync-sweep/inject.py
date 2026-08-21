#!/usr/bin/env python3
"""Comment out one sync/modify call, or put the file back.

   inject.py <file> <line>      comment the call out
   inject.py --restore <file>   restore that one file

   Restoring only the file that was touched matters: a blanket
   "git checkout -- src/KOKKOS/" also throws away any uncommitted work on the
   debugging tools themselves, which is how two rounds of this were measured
   against the wrong binary.
"""
import sys, subprocess
SRC = "src/KOKKOS/"
REPO = "/home/user/lammps"
if sys.argv[1] == "--restore":
    subprocess.run(["git", "checkout", "--", SRC + sys.argv[2]], cwd=REPO, check=True)
    sys.exit(0)
f, n = sys.argv[1], int(sys.argv[2])
p = REPO + "/" + SRC + f
lines = open(p).readlines()
tgt = lines[n - 1]
assert "sync" in tgt or "modif" in tgt, tgt
# An empty statement instead of a bare comment: the target can be the body of
# a brace-less if/else, where deleting the whole line leaves "if (...) else"
# and the build fails (atom_map_kokkos.cpp:374/376 went SKIP this way).
lines[n - 1] = "; // INJECTED-BUG " + tgt
open(p, "w").writelines(lines)
print("removed:", tgt.strip())
