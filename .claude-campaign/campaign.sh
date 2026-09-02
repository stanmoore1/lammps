#!/bin/bash
# Exhaustive mutation campaign with pooled screening.
#
#   campaign.sh <sites-file> <pool-size>
#
# Sites are batched (never two from one file per batch), each batch is one
# incremental rebuild and one pass over the cheap screen cases.  A clean batch
# marks all its sites inert; a failing batch is bisected to the site, which is
# then diagnosed with the poison build first and the watch machinery second.
# Progress checkpoints to campaign.state so the campaign resumes after a stop.
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
STATE=$SP/campaign.state
RESULTS=$SP/campaign.results
POOL=${2:-6}
cd /home/user/lammps
# Refuse to run over uncommitted work: the restore at the end is a plain
# git checkout of the injected file, which has destroyed uncommitted edits
# three separate times in this project.
if [ -n "$(git status --porcelain src/KOKKOS cmake doc)" ]; then
  echo "REFUSING: uncommitted changes under src/KOKKOS -- commit them first"; exit 2
fi

# One campaign at a time: two of them share the source tree, so each would be
# screening the other's injected faults as well as its own, and the verdicts
# would be worthless.  This already happened once.
exec 9>$SP/campaign.lock
if ! flock -n 9; then
  echo "REFUSING: another campaign holds $SP/campaign.lock"; exit 3
fi

touch $STATE $RESULTS

labels() { sed -n 's/^\[stale\] \+\([A-Za-z_/][A-Za-z_0-9/]*:[A-Za-z_0-9]*\).*/\1/p' "$1" 2>/dev/null | sort -u; }

classified() { grep -q "^$1 " $STATE 2>/dev/null; }
mark() { echo "$1 $2" >> $STATE; echo "$1 $2 $3" >> $RESULTS; }

rebuild() { (cd $SP/build-rigdbg && cmake --build . -j 4 >/dev/null 2>&1); }

# The poison binary lives in its own build directory and needs its own rebuild,
# or the diagnosis runs the unmodified code and can never trap.  Only the sites
# that actually manifest pay for this.
rebuild_poison() { (cd $SP/build-poison && cmake --build . -j 4 >/dev/null 2>&1); }

# cheap screen over the core cases; echoes "PASS" or the failing tag + kind
screen() {
  while read d i np pk; do
    [ "$pk" = "-" ] && pk="" || pk=$(echo "$pk" | tr ':' ' ')
    tag="$i.$np${pk:+.dev}"
    o=/tmp/camp.$tag
    bash $SP/runcase.sh $d $i $np $o "$pk"; rc=$?
    if [ $rc -ne 0 ]; then echo "FAIL $tag CRASH"; return 1; fi
    if ! diff -q <(python3 $SP/thermo.py $SP/inj/base.$tag 2>/dev/null) \
                 <(python3 $SP/thermo.py $o 2>/dev/null) >/dev/null 2>&1; then
      echo "FAIL $tag DIVERGED"; return 1
    fi
  done < $SP/cases_core.txt
  echo PASS
}

# diagnose one manifesting site: poison build in survey mode first, then the
# watch/stale diff for the unclaimed-write class
diagnose() {
  local site=$1 tag=$2 kind=$3
  local d i np pk
  read d i np pk < <(awk -v t="$tag" '{pk=$4; gsub(":"," ",pk); tg=$2"."$3 (($4=="-")?"":".dev"); if (tg==t) {print $1,$2,$3,($4=="-"?"":pk); exit}}' $SP/cases.txt)
  local verdict=""
  # Say so when the poison build cannot be produced.  A truncated liblammps.a,
  # left behind by killing a link, once made this fail on every site and the
  # diagnosis fell through to the weaker checks without a word.
  if [ -x $SP/build-poison/lmp ] && ! rebuild_poison; then
    echo "  WARNING: poison build failed, diagnosing $site without it" >&2
    echo "$site POISON-BUILD-FAILED" >> $SP/campaign.warnings
  fi
  if [ -x $SP/build-poison/lmp ] && rebuild_poison; then
    # per-rank log files: four ranks writing to one stderr interleave into
    # reports that belong to no single process
    rm -rf /tmp/diaglog.$tag; mkdir -p /tmp/diaglog.$tag
    LMP_KOKKOS_POISON=1 \
      ASAN_OPTIONS=detect_leaks=0:halt_on_error=0:log_path=/tmp/diaglog.$tag/a \
      bash $SP/runcase_bin.sh $SP/build-poison/lmp $d $i $np /tmp/diag.$tag "$pk"
    local hit=$(cat /tmp/diaglog.$tag/a.* 2>/dev/null | grep -m1 -A12 "use-after-poison" \
      | grep -m1 "in LAMMPS_NS\|in AtomVec" | sed 's/.* in //;s/ \/.*//' | cut -c1-90)
    [ -n "$hit" ] && verdict="poison:$hit"
  fi
  if [ -z "$verdict" ]; then
    LMP_KOKKOS_WATCH= LMP_KOKKOS_STALE= LMP_KOKKOS_STALE_STRICT=1 \
      LMP_KOKKOS_WATCH_SKIP=comm:k_count,comm:k_buf_send \
      bash $SP/runcase.sh $d $i $np /tmp/diagw.$tag "$pk"
    local new=$(comm -13 <(labels $SP/inj/base.$tag.w.err) <(labels /tmp/diagw.$tag.err) | tr '\n' ',')
    local w=$(grep -c '^\[watch\]' /tmp/diagw.$tag.err 2>/dev/null)
    [ -n "$new" ] && verdict="stale:$new"
    [ "$w" -gt 0 ] && verdict="$verdict watch:$w"
    grep -q "concurrent modification" /tmp/diagw.$tag.err 2>/dev/null && \
      verdict="$verdict abort:$(grep -m1 -o 'DualView "[^"]*"' /tmp/diagw.$tag.err)"
  fi
  mark "$site" "MANIFESTS-$kind" "$tag ${verdict:-NOT-DETECTED}"
}

# test a set of sites as one pool; recurse on failure
test_pool() {
  local sites=("$@")
  [ ${#sites[@]} -eq 0 ] && return
  for s in "${sites[@]}"; do
    python3 $SP/inject.py ${s%%:*} ${s##*:} >/dev/null || mark "$s" SKIP inject-failed
  done
  if ! rebuild; then
    for s in "${sites[@]}"; do python3 $SP/inject.py --restore ${s%%:*}; done
    if [ ${#sites[@]} -eq 1 ]; then mark "${sites[0]}" SKIP build-failed; return; fi
  else
    local out=$(screen)
    for s in "${sites[@]}"; do python3 $SP/inject.py --restore ${s%%:*}; done
    if [ "$out" = "PASS" ]; then
      for s in "${sites[@]}"; do mark "$s" INERT ""; done
      echo "  pool of ${#sites[@]}: inert"
      return
    fi
    echo "  pool of ${#sites[@]}: $out"
    if [ ${#sites[@]} -eq 1 ]; then
      set -- $out
      python3 $SP/inject.py ${sites[0]%%:*} ${sites[0]##*:} >/dev/null
      rebuild
      diagnose "${sites[0]}" "$2" "$3"
      python3 $SP/inject.py --restore ${sites[0]%%:*}
      rebuild
      return
    fi
  fi
  # bisect; a single site that got this far has nothing left to split
  if [ ${#sites[@]} -le 1 ]; then mark "${sites[0]}" SKIP unresolved; return; fi
  local mid=$(( ${#sites[@]} / 2 ))
  test_pool "${sites[@]:0:$mid}"
  test_pool "${sites[@]:$mid}"
}

mapfile -t all < <(grep -v '^#' ${1:-$SP/sites_reached.txt})
todo=()
for s in "${all[@]}"; do classified "$s" || todo+=("$s"); done
echo "campaign: ${#all[@]} sites, ${#todo[@]} to do, pool size $POOL"

pool=(); declare -A infile
for s in "${todo[@]}"; do
  f=${s%%:*}
  if [ -n "${infile[$f]}" ] || [ ${#pool[@]} -ge $POOL ]; then
    test_pool "${pool[@]}"; pool=(); unset infile; declare -A infile
  fi
  pool+=("$s"); infile[$f]=1
done
test_pool "${pool[@]}"
echo "CAMPAIGN COMPLETE: $(grep -c ' INERT' $STATE) inert, $(grep -c ' MANIFESTS' $STATE) manifest, $(grep -c ' SKIP' $STATE) skipped"
