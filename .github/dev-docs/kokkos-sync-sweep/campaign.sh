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

# Key the stale diff on (label <- reading function), not the bare label: the
# clean baseline's strict-mode noise (grow_pointers reads every array) shares
# labels with real findings, and a bare-label diff then swallows the one line
# that names the root cause -- atom:bond_atom read from bond_all() went
# NOT-DETECTED exactly this way.  Lines without a "from" fall back to the label.
labels() {
  { sed -n 's/^\[stale\] \+\([A-Za-z_/][A-Za-z_0-9/]*:[A-Za-z_0-9]*\): .*, from \(.*\)$/\1<-\2/p' "$1";
    sed -n '/, from /!s/^\[stale\] \+\([A-Za-z_/][A-Za-z_0-9/]*:[A-Za-z_0-9]*\).*/\1/p' "$1"; } 2>/dev/null \
    | sed 's/ /_/g' | sort -u
}

# Watch reports get the same treatment as stale labels: keyed on the view and
# the rule that fired, subtracted against the clean baseline, so a rule that
# turns out to speak on clean runs cannot convert every manifesting site into
# a false watch verdict.
# The key carries the element where the sides part as well as the view: a
# buffer whose unused tail differs reports the same view on clean runs, and
# keying on the view alone would subtract a real finding away with it.
wlabels() {
  awk '/written without a claim/ {
         view = $2; sub(/:$/, "", view);
         side = (index($0, "the host side") > 0) ? "host" : "device";
         idx = "?";
         if ((getline line) > 0 && match(line, /element [0-9]+/))
           idx = substr(line, RSTART + 8, RLENGTH - 8);
         print view "=" side "@" idx }' "$1" 2>/dev/null | sort -u
}

# The audit's end-of-run report, one line per style and array, with the step
# count stripped so two runs can be compared.  This is the only detector that
# reads the bytes rather than the coherence flags, so it is the only one left
# when the injection is inside AtomVec*Kokkos::modified() itself: removing a
# claim there breaks the very oracle the other detectors consult.
alabels() {
  sed -n 's/^  \(.*\) on [0-9]* step(s)$/\1/p' "$1" 2>/dev/null | sed 's/ /_/g' | sort -u
}

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
  # A verdict of "the tools said nothing" is only worth having if the fault was
  # there to be found.  The screen phase decided this site manifests; check the
  # diagnosis binary carries the injection and reproduces it, or the site goes
  # back on the queue instead of being written down as a blind spot.
  if ! grep -q "INJECTED-BUG" src/KOKKOS/${site%%:*}; then
    echo "  WARNING: $site not injected at diagnosis time"; return 1
  fi

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
      LMP_KOKKOS_WATCH_SKIP= \
      bash $SP/runcase.sh $d $i $np /tmp/diagw.$tag "$pk"
    local new=$(comm -13 <(labels $SP/inj/base.$tag.w.err) <(labels /tmp/diagw.$tag.err) | tr '\n' ',')
    local neww=$(comm -13 <(wlabels $SP/inj/base.$tag.w.err) <(wlabels /tmp/diagw.$tag.err) | tr '\n' ',')
    [ -n "$new" ] && verdict="stale:$new"
    [ -n "$neww" ] && verdict="$verdict watch:$neww"
    grep -q "concurrent modification" /tmp/diagw.$tag.err 2>/dev/null && \
      verdict="$verdict abort:$(grep -m1 -o 'DualView "[^"]*"' /tmp/diagw.$tag.err)"

    # Last, the audit.  It costs a copy of every per-atom array around every
    # style call, so it only runs where the other two are silent.
    if [ -z "$verdict" ] && [ -f $SP/inj/base.$tag.a ]; then
      LMP_KOKKOS_AUDIT=1 bash $SP/runcase.sh $d $i $np /tmp/diaga.$tag "$pk"
      local newa=$(comm -13 <(alabels $SP/inj/base.$tag.a) <(alabels /tmp/diaga.$tag) | tr '\n' ',')
      [ -n "$newa" ] && verdict="audit:$newa"
    fi

    # Poison is the detector that names the root cause, and its binary is a
    # large one that the container restart has now twice thrown away.  A site
    # that nothing else caught while poison was missing is not a blind spot, it
    # is a site that was never fully diagnosed: leave it unclassified so a later
    # pass takes it again, rather than writing down a verdict it did not earn.
    if [ -z "$verdict" ] && [ ! -x $SP/build-poison/lmp ]; then
      echo "  $site: poison binary absent, leaving unclassified"
      echo "$site POISON-ABSENT" >> $SP/campaign.warnings
      return 1
    fi

    # Did this run actually go wrong?  A detector cannot report a fault that
    # did not happen, and the watch run is a different binary and a different
    # environment from the screen, so it does not always reproduce.  Silence
    # from a run that behaved itself says nothing about the tools.
    if [ -z "$verdict" ]; then
      if diff -q <(python3 $SP/thermo.py $SP/inj/base.$tag 2>/dev/null) \
                 <(python3 $SP/thermo.py /tmp/diagw.$tag 2>/dev/null) >/dev/null 2>&1; then
        verdict="NOT-REPRODUCED"
      fi
    fi
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
      # An ignored build failure here diagnoses the previous, clean binary and
      # every detector then correctly says nothing, which reads as a blind
      # spot.  Leave the site unclassified instead so a later pass retries it.
      if rebuild; then
        diagnose "${sites[0]}" "$2" "$3" || echo "  $s left unclassified"
      else
        echo "  pool of 1: build failed at diagnosis, leaving ${sites[0]} unclassified"
      fi
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

# Pools take sites in file order and may take several from the same file: the
# rebuild then compiles one or two files instead of six, and the archive and
# link dominate instead of being multiplied.  Injecting several lines of one
# file together is fine -- restore is per file and the bisect re-injects
# subsets from a clean file each time.
# Size the pool to what the region is turning out to be.  Screening dominates
# the cost and the bisect multiplies it: a pool of six where every site
# manifests costs eleven screen passes to resolve six sites, where testing them
# one at a time costs six.  In an inert region the same pooling is six times
# better.  So follow the recent verdicts -- shrink where sites are manifesting,
# grow again where they are not.
recent=""
adapt_pool() {
  local m=${recent//[^M]/}
  if [ ${#m} -ge 5 ]; then echo 1
  elif [ ${#m} -ge 3 ]; then echo 2
  elif [ ${#m} -ge 1 ]; then echo 3
  else echo $POOL; fi
}

pool=()
for s in "${todo[@]}"; do
  if [ ${#pool[@]} -ge $(adapt_pool) ]; then
    before=$(grep -c ' MANIFESTS' $STATE 2>/dev/null)
    test_pool "${pool[@]}"; pool=()
    after=$(grep -c ' MANIFESTS' $STATE 2>/dev/null)
    [ "$after" -gt "$before" ] && recent="${recent}M" || recent="${recent}i"
    recent=${recent: -8}
  fi
  pool+=("$s")
done
test_pool "${pool[@]}"
echo "CAMPAIGN COMPLETE: $(grep -c ' INERT' $STATE) inert, $(grep -c ' MANIFESTS' $STATE) manifest, $(grep -c ' SKIP' $STATE) skipped"
