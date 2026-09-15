#!/bin/bash
# Group a poison/ASan sweep by distinct fault site.  $1 = sweep tag
D=/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad/sync/$1
echo "inputs run:      $(wc -l < $D/index.txt 2>/dev/null || echo 0)"
echo "with ASan report:$(ls $D/*.asan 2>/dev/null | wc -l)"
echo "non-zero exits:"; cut -f1 $D/index.txt 2>/dev/null | sort | uniq -c | sort -rn | head
echo
echo "=== distinct fault sites (kind | first LAMMPS frame | count | example input) ==="
for f in $D/*.asan; do
  [ -e "$f" ] || continue
  kind=$(grep -m1 -oP 'ERROR: AddressSanitizer: \K[a-z-]+' "$f")
  site=$(grep -m1 -oP '#[0-9]+ 0x[0-9a-f]+ in \K.*lammps/src/[^ ]*' "$f" | sed -E 's/.*(src\/[^ ]*)/\1/')
  frame=$(grep -m1 -oP '#[0-9]+ 0x[0-9a-f]+ in \K[^ ]+' "$f")
  echo -e "$kind\t$site\t$frame\t$(basename $f .asan)"
done | sort | awk -F'\t' '{k=$1"\t"$2"\t"$3; c[k]++; if(!(k in ex)) ex[k]=$4} END {for (k in c) printf "%3d\t%s\t%s\n", c[k], k, ex[k]}' | sort -rn
