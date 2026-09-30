#!/bin/bash
# Evaluate every configuration in cases/ with standalone oxDNA and LAMMPS and print the
# hydrogen-bonding (hb), cross-stacking (xstk) and coaxial-stacking (cx) energies.
#   OXDNA=/path/to/oxDNA LMP=/path/to/lmp ./run_cases.sh [case ...]
# LAMMPS needs the CG-DNA, ASPHERE and MOLECULE packages. oxDNA3 cases run LAMMPS twice: with
# the shipped potentials/oxdna3_lj.cgdna and with oxdna3_lj_ts.cgdna (smoothing points
# sqrt(0.81225/a) as standalone oxDNA3, made by tools/patch_potfile.py), which isolates Q1.
set -e
H=$(cd "$(dirname "$0")" && pwd)
: "${OXDNA:?set OXDNA to the standalone oxDNA binary}" "${LMP:?set LMP to the LAMMPS binary}"
cases=${*:-$(cd $H/cases && ls)}
printf '%-36s %-6s %12s %12s %12s   (N = nucleotides; total energies, oxDNA units)\n' case code hb xstk cx
for c in $cases; do
  cd $H/cases/$c
  N=$(head -1 top.top | awk '{print $1}')
  "$OXDNA" input_oxdna > log_oxdna.txt 2>&1
  read -r fene bexc stck nexc hb xs cx dh < pe_ox.dat
  awk -v N=$N -v c=$c -v hb=$hb -v xs=$xs -v cx=$cx 'BEGIN{printf "%-36s %-6s %12.6f %12.6f %12.6f\n", c, "oxDNA", hb*N, xs*N, cx*N}'
  if grep -q oxdna3 in.lammps; then vars="lmp:oxdna3_lj.cgdna lmp_ts:oxdna3_lj_ts.cgdna"; else vars="lmp:"; fi
  for v in $vars; do
    tag=${v%%:*}; pot=${v#*:}; extra=""
    if [ -n "$pot" ]; then cp $H/lammps/$pot .; extra="-var potfile $pot"; fi
    "$LMP" -in in.lammps -var data data.lmp -var tag $tag $extra -log log_$tag.lammps -screen none
    awk -v c=$c -v t=$tag '{printf "%-36s %-6s %12.6f %12.6f %12.6f\n", c, t, $6, $7, $8}' energy_$tag.txt
  done
done
