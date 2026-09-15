#!/bin/bash
# Group a stale/watch sweep by the routine that reported.  $1 = sweep tag
D=/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad/sync/$1
echo "inputs run: $(wc -l < $D/index.txt 2>/dev/null || echo 0), with reports: $(ls $D/*.rep 2>/dev/null | wc -l)"
tmp=$(mktemp)
for f in $D/*.rep; do [ -e "$f" ] || continue
  # only the report header lines, one per (routine) per input
  grep -aoP '^\[(stale|watch)\].*? from \K[^,]*' "$f" | sed -E 's/\(.*//' | sort -u | sed "s|\$|\t$(basename $f .rep)|"
done > $tmp
echo "distinct routines: $(cut -f1 $tmp | sort -u | wc -l)"
echo
echo "=== routines by number of inputs (rare first) ==="
cut -f1 $tmp | sort | uniq -c | sort -n
rm -f $tmp
