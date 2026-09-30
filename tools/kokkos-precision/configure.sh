#!/bin/bash
# configure.sh -- configure a clang KOKKOS (Serial) build that shows silent
#                 fp32/fp64 conversions as warnings
#
# Usage: configure.sh <single|mixed> <build-dir> [extra cmake arguments...]
#
# Runs cmake from the repository's cmake/ folder with
#   -C cmake/presets/clang.cmake -C cmake/presets/kokkos-serial.cmake
#   -C cmake/presets/kokkos-packages.cmake
#   -D KOKKOS_PREC=<single|mixed>
#   -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wimplicit-float-conversion -Wdouble-promotion -pedantic"
#   -D PKG_REAXFF=off -G Ninja
# ML-IAP is double-only and is already left out by the kokkos-packages preset.
# REAXFF is switched off because it was out of scope for the original work;
# pass -D PKG_REAXFF=on as an extra argument to include it.
#
# Before configuring, the script regenerates the list of packages with KOKKOS
# styles (the command documented in kokkos-packages.cmake) and warns about any
# package the preset lacks (new ports tend to arrive before the preset is
# updated); add those with extra -D PKG_<NAME>=on arguments.
#
# Afterwards: cmake --build <build-dir> -- -k 0 2>&1 | tee <build-dir>/build.log
#             mkcmd.sh <single|mixed> <build-dir>

source "$(dirname "${BASH_SOURCE[0]}")/kkp-env.sh"
[ $# -ge 2 ] || { sed -n '2,23p' "$0"; exit 1; }
P="$1"; B="$2"; shift 2
case "$P" in single|mixed) ;; *) kkp_die "precision must be single or mixed" ;; esac
kkp_check_repo; kkp_check_clang
PRE="$KKP_REPO/cmake/presets"
for p in clang kokkos-serial kokkos-packages; do
  [ -f "$PRE/$p.cmake" ] || kkp_die "missing preset $PRE/$p.cmake"
done

have="$(sed -n '/set(ALL_PACKAGES/,/)/p' "$PRE/kokkos-packages.cmake" | tr -d '()' \
        | tr ' ' '\n' | grep -E '^[A-Z0-9-]+$' | grep -v ALL_PACKAGES | sort -u)"
need="$(cd "$KKP_REPO/src/KOKKOS" && for f in *_kokkos.h; do b=${f/_kokkos/}; \
        [ -f ../$b ] || ls ../*/$b 2> /dev/null; done | cut -d/ -f2 | sort -u)"
missing="$(comm -13 <(echo "$have") <(echo "$need") | grep -vE '^(ML-IAP|REAXFF)$')"
if [ -n "$missing" ]; then
  echo "WARNING: kokkos-packages.cmake lacks packages with KOKKOS styles:" 1>&2
  echo "  $missing" | tr '\n' ' ' 1>&2; echo 1>&2
  echo "  add: $(for m in $missing; do printf -- '-D PKG_%s=on ' "$m"; done)" 1>&2
fi

cmake -S "$KKP_REPO/cmake" -B "$B" \
  -C "$PRE/clang.cmake" -C "$PRE/kokkos-serial.cmake" -C "$PRE/kokkos-packages.cmake" \
  -D KOKKOS_PREC="$P" \
  -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wimplicit-float-conversion -Wdouble-promotion -pedantic" \
  -D PKG_REAXFF=off -G Ninja "$@"
