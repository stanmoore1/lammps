#!/bin/bash
# fullsurvey.sh -- per-file in-scope warning counts from a FULL build
#
# Usage: fullsurvey.sh [--rebuild [--touch]] <build-dir> [build-log]
#
# Parses a complete ninja build log and prints, for every src/KOKKOS file, the
# number of distinct flagged locations (-Wimplicit-float-conversion and
# -Wdouble-promotion; headers are counted once even if many TUs report them),
# sorted by count, followed by a TOTAL line and the number of compile errors.
#
#   build-log   log to parse (default: <build-dir>/build.log)
#   --rebuild   first run "ninja -C <build-dir> -k 0" and write its output to
#               $KKP_WORK/fullbuild_<dirname>.log, then parse that log
#   --touch     with --rebuild: touch src/KOKKOS/kokkos_type.h first, so that
#               EVERY KOKKOS TU is recompiled.  An incremental build only
#               reports warnings of the TUs it recompiles; a survey from an
#               incomplete build silently misses whole style families.
#
# This is the authoritative check: the per-TU checkers are fast but see only
# the TUs you ask about.

source "$(dirname "${BASH_SOURCE[0]}")/kkp-env.sh"
REB=0; TOUCH=0
while [ "${1:-}" = "--rebuild" ] || [ "${1:-}" = "--touch" ]; do
  [ "$1" = "--rebuild" ] && REB=1; [ "$1" = "--touch" ] && TOUCH=1; shift
done
[ $# -ge 1 ] || { sed -n '2,20p' "$0"; exit 1; }
B="$1"; LOG="${2:-$B/build.log}"
if [ $REB -eq 1 ]; then
  [ -f "$B/build.ninja" ] || kkp_die "$B is not a configured Ninja build directory"
  if [ $TOUCH -eq 1 ]; then
    kkp_check_repo; touch "$KKP_REPO/src/KOKKOS/kokkos_type.h"
  fi
  mkdir -p "$KKP_WORK"
  LOG="$KKP_WORK/fullbuild_$(basename "$(realpath "$B")").log"
  echo "building $B (log: $LOG) ..." 1>&2
  ninja -C "$B" -k 0 > "$LOG" 2>&1; echo "ninja exit status $?" 1>&2
fi
[ -f "$LOG" ] || kkp_die "no build log $LOG (use --rebuild or give the log path)"

FILT="src/KOKKOS/[A-Za-z0-9_]+\.(cpp|h)"
grep -E "${FILT}:[0-9]+:[0-9]+: warning:.*\[-(${KKP_WFLAGS})\]" "$LOG" \
  | grep -oE "${FILT}:[0-9]+:[0-9]+" | sort -u | cut -d: -f1 | sed 's#.*src/KOKKOS/##' \
  | sort | uniq -c | sort -k1,1nr -k2,2
tot=$(grep -E "${FILT}:[0-9]+:[0-9]+: warning:.*\[-(${KKP_WFLAGS})\]" "$LOG" \
  | grep -oE "${FILT}:[0-9]+:[0-9]+" | sort -u | grep -c .)
err=$(grep -E "error:" "$LOG" | grep -vc "\.o\.d'")
echo "TOTAL $tot locations; $err error lines in $LOG"
