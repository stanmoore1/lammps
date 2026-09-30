#!/bin/bash
# survey.sh -- count the flagged locations of one TU in both precisions
#
# Usage: survey.sh <TU.cpp> [more TUs...]
#
# Prints one line per TU: "<single> <mixed> <errors> <basename>", where
# single/mixed are the numbers of distinct flagged src/KOKKOS locations
# (including headers reached by the TU) reported by chk.sh and errors is the
# number of ERROR lines in either precision.  Use it to build the baseline
# counts that go into the subagent prompts, e.g.
#   for f in src/KOKKOS/fix_*_kokkos.cpp; do survey.sh "$f"; done | sort -rn

D="$(dirname "${BASH_SOURCE[0]}")"
[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 1; }
for f in "$@"; do
  so="$("$D/chk.sh" single "$f" 2> /dev/null)"
  mo="$("$D/chk.sh" mixed "$f" 2> /dev/null)"
  s=$(echo "$so" | grep -cE '^src/')
  m=$(echo "$mo" | grep -cE '^src/')
  e=$(printf '%s\n%s\n' "$so" "$mo" | grep -c '^ERROR')
  echo "$s $m $e $(basename "$f")"
done
