#!/bin/bash
# Compare the thermo output of two run-sweep.sh tags, input by input.
#
#   A=<tag> B=<tag> LIST=<input-list> diff-divergence.sh
#
# This is the one check that does not go through the sync protocol: it looks at
# the numbers the run produced, so it catches a wrong value that every detector
# agreed was properly synced.
#
# The comparison itself is compare-thermo.py and nothing else.  There used to be
# a second extractor here, and when the wall-clock column problem was fixed in
# one and not the other, PACKAGES/fep/CH4-CF4/bar10 was written up as a bug it
# never was.  One implementation, so it cannot drift again.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SP=${SP:-/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad/sync}
A=${A:-plain845}; B=${B:-split845}; LIST=${LIST:-$SP/all-inputs.txt}

n=0; d=0; u=0; m=0
while read -r f; do
  [ -n "$f" ] || continue
  name=$(echo "$f" | sed 's|examples/||; s|/|_|g')
  a=$SP/$A/$name.out.gz; b=$SP/$B/$name.out.gz
  if [ ! -f "$a" ] || [ ! -f "$b" ]; then m=$((m+1)); continue; fi
  detail=$(python3 "$HERE/compare-thermo.py" "$a" "$b" 2>&1 >/dev/null)
  verdict=$(python3 "$HERE/compare-thermo.py" "$a" "$b" 2>/dev/null)
  case $verdict in
    NO-OUTPUT) u=$((u+1)); echo "UNCOMPARED $f ($detail)" ;;
    SAME)      n=$((n+1)) ;;
    *)         n=$((n+1)); d=$((d+1)); echo "DIVERGES $f ($detail)" ;;
  esac
done < "$LIST"
echo
echo "compared $n inputs, $d diverge, $u uncompared, $m missing a log on one side"
