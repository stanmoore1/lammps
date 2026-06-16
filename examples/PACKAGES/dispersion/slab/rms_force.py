#!/usr/bin/env python3
"""RMS force calculator + estimator verification for the compact-switch (CSB)
ewald/disp/slab.

For a sweep of kmax: run CSB, dump total forces, and read the estimator's
predicted RMS force error.  The measured per-atom RMS force error is taken vs a
converged reference (very high kmax).  Verifies predicted ~ measured.
"""
import subprocess, re, os, glob
import numpy as np

LMP = "/home/user/lammps-csb/build-mpi/lmp"
RC, D = "3.0", "0.5"
KREF = 1500
KMAXES = [12, 16, 24, 32, 48, 64, 96, 128, 192, 256]

def run(km):
    out = subprocess.run(
        ["mpirun", "--allow-run-as-root", "-np", "1", LMP, "-in", "in.csbforce",
         "-var", "rc", RC, "-var", "D", D, "-var", "km", str(km),
         "-log", "none", "-screen", "/dev/stdout"],
        capture_output=True, text=True).stdout
    m = re.search(r"estimated absolute RMS force accuracy = (\S+)", out)
    est = float(m.group(1)) if m else float("nan")
    f = read_forces(f"force_{km}.dump")
    return est, f

def read_forces(fn):
    lines = open(fn).read().splitlines()
    i = lines.index("ITEM: ATOMS id fx fy fz") + 1
    data = np.array([[float(x) for x in ln.split()] for ln in lines[i:]])
    data = data[data[:, 0].argsort()]
    return data[:, 1:4]

print(f"reference kmax={KREF} ...")
_, fref = run(KREF)
N = len(fref)

print(f"\n{'kmax':>5} {'measured RMS':>14} {'estimated RMS':>14} {'est/meas':>9}")
print("-" * 48)
rows = []
for km in KMAXES:
    est, f = run(km)
    df = f - fref
    rms = np.sqrt((df ** 2).sum() / N)         # per-atom RMS force error (vector magnitude)
    rows.append((km, rms, est))
    print(f"{km:>5} {rms:>14.4e} {est:>14.4e} {est/rms:>9.2f}")

# cleanup dumps
for fn in glob.glob("force_*.dump"):
    os.remove(fn)
