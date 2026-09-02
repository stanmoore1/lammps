#!/bin/bash
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
touch $SP/blind.seen
while :; do
  if [ -f $SP/campaign.results ]; then
    grep "NOT-DETECTED" $SP/campaign.results 2>/dev/null | awk '{print $1, $3}' | sort -u \
      | while read site tag; do
          grep -qx "$site $tag" $SP/blind.seen || { echo "BLIND: $site on $tag"; echo "$site $tag" >> $SP/blind.seen; }
        done
  fi
  grep -q "CAMPAIGN COMPLETE" $SP/campaign.log 2>/dev/null && { echo "CAMPAIGN COMPLETE"; break; }
  pgrep -f "campaign.sh" >/dev/null || { echo "CAMPAIGN STOPPED"; break; }
  sleep 240
done
