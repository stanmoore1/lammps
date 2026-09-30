#!/bin/bash
# kkp-env.sh -- common settings for the kokkos-precision scripts (sourced, not run).
#
# Environment variables (all optional):
#   KKP_REPO    LAMMPS checkout to operate on.  Default: the git top level of the
#               current directory, or else of the directory containing this script.
#   KKP_WORK    work directory for the captured compile commands and logs.
#               Default: $HOME/.cache/kk-precision
#   CXX_CLANG   clang++ executable.  Default: $(command -v clang++)
#   KKP_WFLAGS  extended regex of the warning flags that are in scope.
#               Default: Wimplicit-float-conversion|Wdouble-promotion

KKP_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${KKP_REPO:-}" ]; then
  KKP_REPO="$(git rev-parse --show-toplevel 2> /dev/null)"
  [ -n "$KKP_REPO" ] || KKP_REPO="$(git -C "$KKP_SCRIPTS" rev-parse --show-toplevel 2> /dev/null)"
fi
KKP_WORK="${KKP_WORK:-$HOME/.cache/kk-precision}"
CXX_CLANG="${CXX_CLANG:-$(command -v clang++)}"
KKP_WFLAGS="${KKP_WFLAGS:-Wimplicit-float-conversion|Wdouble-promotion}"

kkp_die() { echo "ERROR: $*" 1>&2; exit 1; }

kkp_check_repo() {
  [ -n "$KKP_REPO" ] && [ -f "$KKP_REPO/src/KOKKOS/kokkos_type.h" ] \
    || kkp_die "cannot find a LAMMPS checkout (set KKP_REPO or run inside the repository)"
}

kkp_check_clang() {
  [ -n "$CXX_CLANG" ] && [ -x "$CXX_CLANG" ] || kkp_die "clang++ not found (set CXX_CLANG)"
}

# print the compiler flags (without compiler, -o, -c, and dependency options)
# for precision $1 (single|mixed|double), from $KKP_WORK/full_cmd_$1.txt.
# If full_cmd_double.txt does not exist, it is derived from the single command
# by switching the precision define to LMP_KOKKOS_DOUBLE_DOUBLE.
kkp_flags() {
  local p="$1" f="$KKP_WORK/full_cmd_$1.txt" extra=""
  case "$p" in
    single|mixed|double) ;;
    *) kkp_die "precision must be single, mixed, or double (got '$p')" ;;
  esac
  if [ "$p" = double ] && [ ! -s "$f" ]; then
    f="$KKP_WORK/full_cmd_single.txt"
    extra='s#-DLMP_KOKKOS_SINGLE_SINGLE#-DLMP_KOKKOS_DOUBLE_DOUBLE#'
  fi
  [ -s "$f" ] || kkp_die "missing $f; run mkcmd.sh $p <build-dir> first"
  sed -E "s# -o [^ ]+ -c [^ ]+##; s#^[^ ]*clang\+\+ ##; s# -MD##; s# -MT [^ ]+##; s# -MF [^ ]+##; ${extra}" "$f"
}

# resolve a KOKKOS file argument (basename, repo-relative, or absolute path)
# to a path relative to the repository top level
kkp_relpath() {
  local a="$1"
  if [ -f "$a" ]; then
    a="$(realpath "$a")"
    echo "${a#"$KKP_REPO"/}"
  elif [ -f "$KKP_REPO/$a" ]; then
    echo "$a"
  elif [ -f "$KKP_REPO/src/KOKKOS/$a" ]; then
    echo "src/KOKKOS/$a"
  else
    kkp_die "cannot find file '$1'"
  fi
}

# compile TU $2 (repo-relative) syntax-only with precision $1 and print all
# diagnostics.  Returns 0 even when the TU has compile errors (those are part
# of the output); returns 2 only if the compile flags are unavailable.
kkp_compile() {
  local flags
  flags="$(kkp_flags "$1")" || return 2
  # shellcheck disable=SC2086
  (cd "$KKP_REPO" && "$CXX_CLANG" $flags -fsyntax-only "$2" 2>&1)
  return 0
}
