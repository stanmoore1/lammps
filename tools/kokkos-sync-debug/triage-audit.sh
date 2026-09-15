#!/bin/bash
# Pull the audit detail out of a sweep's compressed output and group it.
# $1 = sweep tag (e.g. audit845)
SP=/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad/sync/$1
for g in $SP/*.out.gz; do
  n=$(basename "$g" .out.gz)
  zcat "$g" 2>/dev/null | sed -n '/undeclared changes to per-atom arrays/,/^[^ ]/p' \
    | grep -a '^  ' | sed -E 's/ on [0-9]+ step\(s\)//' | sort -u | sed "s|^|$n\t|"
done > $SP/../$1.claims.tsv
echo "inputs with claims: $(cut -f1 $SP/../$1.claims.tsv | sort -u | wc -l)"
echo "distinct claims:    $(cut -f2 $SP/../$1.claims.tsv | sort -u | wc -l)"
echo
echo "=== claims that are NOT the 'declares every array' / 'reads stale' caveat ==="
cut -f2 $SP/../$1.claims.tsv | sort | uniq -c | sort -rn \
  | grep -v "declares every array" | grep -v "reads stale" | head -40
echo
echo "=== styles named, by how many inputs ==="
awk -F'\t' '{split($2,a," "); print a[1]"\t"$1}' $SP/../$1.claims.tsv | sort -u | cut -f1 | sort | uniq -c | sort -rn | head -25
