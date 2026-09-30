#!/bin/bash
# mkcmd.sh -- capture the compile command of a configured KOKKOS build
#
# Usage: mkcmd.sh <single|mixed|double> <build-dir>
#
# Writes $KKP_WORK/full_cmd_<prec>.txt with the complete clang++ command that
# ninja uses for src/KOKKOS/fix_nve_kokkos.cpp in <build-dir> (a CMake/Ninja
# build configured with configure.sh).  chk.sh, survey.sh, and vc.sh strip the
# source, output, and dependency options from it and reuse the flags (defines,
# include paths, warning flags) for any other KOKKOS TU.
#
# Re-run it after re-configuring the build (for example after enabling more
# packages): the include paths and defines change with the package set.
# The build must be configured from the same checkout you are editing, or the
# checkers will compile against the wrong headers.
# This is the generalized form of the original generator:
#   ninja -C build-kk-$P -t commands <fix_nve_kokkos.cpp.o> | tail -1 > full_cmd_$P.txt

source "$(dirname "${BASH_SOURCE[0]}")/kkp-env.sh"
[ $# -eq 2 ] || { sed -n '2,19p' "$0"; exit 1; }
P="$1"; B="$2"
case "$P" in
  single) DEF=LMP_KOKKOS_SINGLE_SINGLE ;;
  mixed)  DEF=LMP_KOKKOS_SINGLE_DOUBLE ;;
  double) DEF=LMP_KOKKOS_DOUBLE_DOUBLE ;;
  *) kkp_die "precision must be single, mixed, or double" ;;
esac
[ -f "$B/build.ninja" ] || kkp_die "$B is not a configured Ninja build directory"
command -v ninja > /dev/null || kkp_die "ninja not found"

OBJ="$(ninja -C "$B" -t targets all 2> /dev/null | grep -oE '^[^:]*src/KOKKOS/fix_nve_kokkos\.cpp\.o' | head -1)"
[ -n "$OBJ" ] || kkp_die "no object for src/KOKKOS/fix_nve_kokkos.cpp in $B (is PKG_KOKKOS on?)"
mkdir -p "$KKP_WORK"
OUTF="$KKP_WORK/full_cmd_$P.txt"
ninja -C "$B" -t commands "$OBJ" 2> /dev/null | tail -1 > "$OUTF"

grep -q -- "-D$DEF" "$OUTF" || echo "WARNING: $OUTF does not define $DEF; check KOKKOS_PREC of $B" 1>&2
for w in -Wimplicit-float-conversion -Wdouble-promotion; do
  grep -q -- "$w" "$OUTF" || echo "WARNING: $OUTF lacks $w; configure with configure.sh" 1>&2
done
grep -q -- "clang++" "$OUTF" || echo "WARNING: $B does not use clang++" 1>&2
home="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$B/CMakeCache.txt" 2> /dev/null)"
if [ -n "$KKP_REPO" ] && [ -n "$home" ] && [ "$(realpath "$home")" != "$(realpath "$KKP_REPO/cmake")" ]; then
  echo "WARNING: $B was configured from $home, not from $KKP_REPO/cmake" 1>&2
fi
echo "wrote $OUTF ($(wc -c < "$OUTF") bytes, $P)"
