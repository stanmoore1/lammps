#!/bin/bash
# Run after in.runA completes (needs traj.dump). Recomputes the IK profile
# (short cutoff + dispersion kspace) and the long-cutoff reference (plain
# lj/cut 8.0, no kspace) over the SAME dumped frames, then post-processes.
set -e
cd "$(dirname "$0")"
LMP="mpirun -np 4 --allow-run-as-root ../../build/lmp"
echo "=== rerun A (short cutoff + ewald/disp/planar) ==="
$LMP -in in.rerunA > rerunA.log 2>&1
echo "=== rerun B (plain lj/cut 8.0, no kspace) ==="
$LMP -in in.rerunB > rerunB.log 2>&1
echo "=== post-processing ==="
python3 verify_pressure.py
