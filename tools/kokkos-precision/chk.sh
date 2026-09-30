#!/bin/bash
# chk.sh -- per-TU checker for silent fp32/fp64 conversions in the KOKKOS package
#
# Usage: chk.sh <single|mixed|double> <TU.cpp> [path-regex]
#
# Compiles one translation unit syntax-only with the flags captured by mkcmd.sh
# and prints every flagged location (src/KOKKOS/<file>:line:col) for the two
# in-scope warnings (-Wimplicit-float-conversion, -Wdouble-promotion), in the
# .cpp file AND in every src/KOKKOS header it reaches, followed by any real
# compile errors as "ERROR: ..." lines.
#
# EMPTY OUTPUT = CLEAN.  A file is not clean while ERROR lines are printed,
# even if no warnings are listed (an edit that breaks compilation also makes
# the warnings disappear).
#
# The optional third argument restricts the warning list to paths matching an
# extended regex, e.g. a header basename:
#   chk.sh single src/KOKKOS/pair_lj_cut_tip4p_long_kokkos.cpp pair_tip4p_kokkos.h
# ERROR lines are always printed.
#
# "double" compiles with -DLMP_KOKKOS_DOUBLE_DOUBLE; use it to show that a
# remaining warning also exists in the default build (then it is out of scope).
# Set KKP_WFLAGS to change the warning set, e.g.
#   KKP_WFLAGS='Wimplicit-int-float-conversion' chk.sh double <TU.cpp>

source "$(dirname "${BASH_SOURCE[0]}")/kkp-env.sh"
[ $# -ge 2 ] || { sed -n '2,25p' "$0"; exit 1; }
P="$1"; ONLY="${3:-}"
kkp_check_repo; kkp_check_clang
SRC="$(kkp_relpath "$2")" || exit 1
OUT="$(kkp_compile "$P" "$SRC")" || exit 1  # fails only without flags
FILT="src/KOKKOS/[A-Za-z0-9_]+\.(cpp|h)"
echo "$OUT" | grep -E "${FILT}:[0-9]+:[0-9]+: warning:.*\[-(${KKP_WFLAGS})\]" \
  | grep -oE "${FILT}:[0-9]+:[0-9]+" | grep -E "${ONLY:-.}" | sort -u -t: -k1,1 -k2,2n -k3,3n
echo "$OUT" | grep -E "error:" | grep -v "\.o\.d'" | sed 's/^/ERROR: /'
exit 0
