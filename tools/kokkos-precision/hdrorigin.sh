#!/bin/bash
# hdrorigin.sh -- find which translation unit(s) produce a header's warnings
#
# Usage: hdrorigin.sh [--log <build.log>] [--prec <p>] [--max N] <header.h> [line]
#
# A header warns only when an including .cpp is compiled, and different TUs
# instantiate different templates, so a header can be clean through one TU
# and still warn through another.  Two modes:
#
#   --log <build.log>  scan a ninja build log (e.g. from fullsurvey.sh or a
#                      full build) and print every KOKKOS TU whose compile
#                      output contains an in-scope warning in <header.h>
#                      (optionally only at <line>), with the number of
#                      distinct locations.  Fast; needs a complete log.
#   (default)          compile the including TUs syntax-only (flags from
#                      mkcmd.sh) and print "single=N mixed=N <TU>".  The TUs
#                      are the src/KOKKOS/*.cpp files that include the header
#                      directly, plus those that include it through one
#                      intermediate src/KOKKOS header.  --prec selects single,
#                      mixed, or both (default both); --max limits the number
#                      of TUs compiled (default 20; each takes 5-20 s, so run
#                      long scans in the background).
#
# Use the result to name the triggering TU explicitly in agent prompts and in
# vc.sh ("header.h@tu1.cpp,tu2.cpp"), and to confirm that a previously clean
# TU stays clean after a header fix.

source "$(dirname "${BASH_SOURCE[0]}")/kkp-env.sh"
LOG=""; PRECS="single mixed"; MAX=20
while [ $# -gt 0 ]; do
  case "$1" in
    --log)  LOG="$2"; shift 2 ;;
    --prec) PRECS="$2"; shift 2 ;;
    --max)  MAX="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    --*) kkp_die "unknown option $1" ;;
    *) break ;;
  esac
done
[ $# -ge 1 ] || { sed -n '2,26p' "$0"; exit 1; }
H="$(basename "$1")"; LINE="${2:-[0-9]+}"
PAT="src/KOKKOS/${H//./\\.}:${LINE}:[0-9]+: warning:.*\\[-(${KKP_WFLAGS})\\]"

if [ -n "$LOG" ]; then
  [ -f "$LOG" ] || kkp_die "no build log $LOG"
  awk -v pat="$PAT" '
    /Building CXX object .*KOKKOS\/[A-Za-z0-9_]+\.cpp\.o/ {
      tu = $0; sub(/\.cpp\.o.*/, ".cpp", tu); sub(/.*KOKKOS\//, "src/KOKKOS/", tu) }
    $0 ~ pat { loc = $0; sub(/: warning:.*/, "", loc); sub(/.*src\/KOKKOS\//, "", loc)
               if (!((tu, loc) in seen)) { seen[tu, loc] = 1; n[tu]++ } }
    END { for (t in n) printf("%6d %s\n", n[t], t) }' "$LOG" | sort -k1,1nr -k2,2
  exit 0
fi

kkp_check_repo; kkp_check_clang
tus="$(cd "$KKP_REPO" && grep -lE "#include +\"$H\"" src/KOKKOS/*.cpp 2> /dev/null)"
for h in $(cd "$KKP_REPO" && grep -lE "#include +\"$H\"" src/KOKKOS/*.h 2> /dev/null); do
  tus+=$'\n'"$(cd "$KKP_REPO" && grep -lE "#include +\"$(basename "$h")\"" src/KOKKOS/*.cpp 2> /dev/null)"
done
tus="$(echo "$tus" | grep . | sort -u)"
[ -n "$tus" ] || kkp_die "no src/KOKKOS/*.cpp includes $H (directly or through one header)"
total=$(echo "$tus" | grep -c .)
[ "$total" -gt "$MAX" ] && echo "# $total including TUs; checking the first $MAX (use --max)" 1>&2
for tu in $(echo "$tus" | head -n "$MAX"); do
  line=""
  for p in $PRECS; do
    c=$(kkp_compile "$p" "$tu" | grep -E "$PAT" | grep -oE "src/KOKKOS/[A-Za-z0-9_.]+:[0-9]+:[0-9]+" \
          | sort -u | grep -c .)
    line+="$p=$c "
  done
  echo "$line $tu"
done
