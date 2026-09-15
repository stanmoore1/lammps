#!/bin/bash
# Compare the thermo output of the two memory models, input by input.
SP=/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad/sync
thermo() {
  # the thermo block only: drop timings, paths, wall clock and version banners
  zcat "$1" 2>/dev/null | sed -n '/^ *Step /,/^Loop time/p' \
    | grep -vE "Loop time|^ *Step |WARNING|MPI task" 
}
n=0; d=0
while read -r f; do
  name=$(echo "$f" | sed 's|examples/||; s|/|_|g')
  a=$SP/plain845/$name.out.gz; b=$SP/split845/$name.out.gz
  [ -f "$a" ] && [ -f "$b" ] || continue
  n=$((n+1))
  if ! diff -q <(thermo "$a") <(thermo "$b") >/dev/null 2>&1; then
    d=$((d+1))
    echo "DIVERGES $f"
    diff <(thermo "$a") <(thermo "$b") | head -6 | sed 's/^/    /'
  fi
done < $SP/all-inputs.txt
echo
echo "compared $n inputs, $d diverge"
