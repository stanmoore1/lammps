#!/bin/bash
# Compare the shared-memory build against the split-memory one over a list of
# inputs, with the control that matters: each build runs twice, so an input that
# does not reproduce itself is reported as such instead of being compared
# against one arbitrary run of the other side.  Several entries in FINDINGS.md
# were written up as memory-model divergences before this control existed and
# were nothing of the kind -- one differed only in its S/CPU column, and two are
# not reproducible on the same binary at all, with or without KOKKOS.
#
#   compare-builds.sh <input-list> [outdir]
#
# The list holds one input path per line, relative to the repository root.
# One line per input is printed: whether each side reproduced itself (det /
# PLAIN-NONDET / SYNC-NONDET), and whether the two agree (cross=SAME|DIFFER).

set -u
LIST=${1:?usage: compare-builds.sh <input-list> [outdir]}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${2:-${TMPDIR:-/tmp}/compare-builds}
mkdir -p "$OUT"

PK="-k on t 1 -sf kk -pk kokkos neigh full newton off comm device sort device atom/map device gpu/aware on"
NP=${NP:-4}
TIMEOUT=${TIMEOUT:-900}

thermo_same() { python3 "$HERE/compare-thermo.py" "$1" "$2"; }

while read -r f <&3; do
  [ -n "$f" ] || continue
  d=$(dirname "$f"); b=$(basename "$f")
  for bl in plain sync; do
    for r in 1 2; do
      (cd "$ROOT/$d" && timeout "$TIMEOUT" mpirun --allow-run-as-root --oversubscribe \
          -np "$NP" "$ROOT/build-$bl/lmp" $PK -in "$b" \
          -log "$OUT/$bl.$r.$b.log" -screen none > /dev/null 2>&1 < /dev/null)
    done
  done
  rows=$(grep -cE "^ +[0-9]+ +[-0-9.]" "$OUT/plain.1.$b.log" 2>/dev/null)
  det="det"
  [ "$(thermo_same "$OUT/plain.1.$b.log" "$OUT/plain.2.$b.log")" = SAME ] || det="PLAIN-NONDET"
  [ "$(thermo_same "$OUT/sync.1.$b.log"  "$OUT/sync.2.$b.log")"  = SAME ] || det="$det SYNC-NONDET"
  cross=$(thermo_same "$OUT/plain.1.$b.log" "$OUT/sync.1.$b.log")
  echo "CMP $b rows=$rows $det cross=$cross"
done 3< "$LIST"
echo "ALLDONE"
