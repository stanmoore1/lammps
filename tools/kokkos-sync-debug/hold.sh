#!/bin/bash
# Generic foreground hold: block on the FIFO until $2 appears in file $1, or the
# duration runs out.  Same mechanism as wd.sh -- read -t on a FIFO opened
# read-write, no sleep, no spin -- but keyed to an arbitrary completion marker
# so it can wait on something other than the sweep pipeline.
SP=/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad
WATCH=$1; DONEPAT=${2:-DONE}; DUR=${3:-560}
F=$SP/sync/hold.fifo; [ -p $F ] || { rm -f $F; mkfifo $F; }
exec 8<>$F
prev=$(wc -l < "$WATCH" 2>/dev/null || echo 0)
end=$((SECONDS + DUR)); news=""
while [ $SECONDS -lt $end ]; do
  if grep -q "$DONEPAT" "$WATCH" 2>/dev/null; then news="COMPLETE"; break; fi
  now=$(wc -l < "$WATCH" 2>/dev/null || echo 0)
  if [ "$now" -gt "$prev" ]; then news=$(tail -n +$((prev+1)) "$WATCH"); prev=$now; fi
  read -t 30 -u 8 x
done
if [ "$news" = "COMPLETE" ]; then echo "COMPLETE ($(grep -c . "$WATCH") lines)"
elif [ -n "$news" ]; then echo "$news"
else echo "(quiet, held ${SECONDS}s of ${DUR}s; $(wc -l < "$WATCH" 2>/dev/null || echo 0) done so far)"; fi
