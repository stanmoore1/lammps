#!/bin/bash
# configure.sh -- configure a clang KOKKOS (Serial) build that shows silent
#                 fp32/fp64 conversions as warnings
#
# Usage: configure.sh <single|mixed> <build-dir> [extra cmake arguments...]
#
# Runs cmake from the repository's cmake/ folder with the settings of the
# original work:
#   -C cmake/presets/clang.cmake -C cmake/presets/kokkos-serial.cmake
#   -C cmake/presets/kokkos-packages.cmake
#   -D CMAKE_BUILD_TYPE=RelWithDebInfo -D KOKKOS_PREC=<single|mixed>
#   -D BUILD_MPI=off -D FFT=KISS
#   -D PKG_REAXFF=off -D PKG_ML-IAP=off -D PKG_ML-PACE=off
#   -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wimplicit-float-conversion -Wdouble-promotion -pedantic"
#   -G Ninja
#
# Package defaults (all can be overridden with extra arguments, which are
# passed last, so they take precedence):
#   REAXFF    off: out of scope for the original work (by instruction).
#   ML-IAP    off: its KOKKOS styles require KOKKOS_PREC=double; the preset
#             already leaves it out, the explicit setting guards stale caches.
#   ML-PACE   off: configuring ML-PACE downloads an external library; that
#             download was blocked in the original environment, so ML-PACE
#             was never part of the original scope.  With network access,
#             pass -D PKG_ML-PACE=on to include it (or set KKP_ML_PACE=on).
#
# Always pass the FULL set of options when re-configuring an existing build
# directory; do not rely on values cached from an earlier configuration.
#
# Before configuring, the script regenerates the list of packages with KOKKOS
# styles (the command documented in kokkos-packages.cmake) and warns about any
# package the preset lacks (new ports tend to arrive before the preset is
# updated); add those with extra -D PKG_<NAME>=on arguments.
#
# Afterwards (a full build takes 20-30 minutes on 4 cores; run it in the
# background and wait for the EXIT marker, see README.md):
#   (ninja -C <build-dir> -k 0; echo "EXIT=$?") > <build-dir>/build.log 2>&1
#   mkcmd.sh <single|mixed> <build-dir>

source "$(dirname "${BASH_SOURCE[0]}")/kkp-env.sh"
[ $# -ge 2 ] || { sed -n '2,38p' "$0"; exit 1; }
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
missing="$(comm -13 <(echo "$have") <(echo "$need") | grep -vE '^(ML-IAP|REAXFF|ML-PACE)$')"
if [ -n "$missing" ]; then
  echo "WARNING: kokkos-packages.cmake lacks packages with KOKKOS styles:" 1>&2
  echo "  $missing" | tr '\n' ' ' 1>&2; echo 1>&2
  echo "  add: $(for m in $missing; do printf -- '-D PKG_%s=on ' "$m"; done)" 1>&2
fi

cmake -S "$KKP_REPO/cmake" -B "$B" \
  -C "$PRE/clang.cmake" -C "$PRE/kokkos-serial.cmake" -C "$PRE/kokkos-packages.cmake" \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo -D KOKKOS_PREC="$P" \
  -D BUILD_MPI=off -D FFT=KISS \
  -D PKG_REAXFF=off -D PKG_ML-IAP=off -D PKG_ML-PACE="${KKP_ML_PACE:-off}" \
  -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wimplicit-float-conversion -Wdouble-promotion -pedantic" \
  -G Ninja "$@"
