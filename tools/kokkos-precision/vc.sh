#!/bin/bash
# vc.sh -- verify-and-commit gate for a finished batch of KOKKOS files
#
# Usage: vc.sh [--commit "<message>"] file1 [file2 ...]
#
# For each file, compiles it syntax-only in BOTH the mixed and the single
# precision configuration (flags from mkcmd.sh) and counts in-scope warnings
# attributed to that file and compile errors:
#   <file>: mixed(w=N e=N) single(w=N e=N)
# The batch passes only if every count is zero.  Without --commit nothing else
# happens.  With --commit and a passing batch, exactly these files are staged
# and committed with the given message (no trailers are added; the commit is
# not signed unless KKP_SIGN=1 is set).  A failing batch is never committed.
#
# Files may be given as basenames (looked up in src/KOKKOS), repo-relative, or
# absolute paths.  A header is compiled through an including TU: give it as
# "header.h@including_tu.cpp", or the first src/KOKKOS/*.cpp that includes it
# is used.  Headers can warn differently in different TUs (template
# instantiation), so the final full rebuild remains the authoritative check.
#
# Exit status: 0 = all clean (and committed if requested), 1 = failures.

source "$(dirname "${BASH_SOURCE[0]}")/kkp-env.sh"
MSG=""
if [ "${1:-}" = "--commit" ]; then
  [ $# -ge 3 ] || kkp_die "--commit needs a message and at least one file"
  MSG="$2"; shift 2
fi
[ $# -ge 1 ] || { sed -n '2,24p' "$0"; exit 1; }
kkp_check_repo; kkp_check_clang

count() {  # count <diagnostics> <repo-relative file>
  local w e
  w=$(echo "$1" | grep -E "${2}:[0-9]+:[0-9]+: warning:.*\[-(${KKP_WFLAGS})\]" \
        | grep -oE "${2}:[0-9]+:[0-9]+" | sort -u | grep -c .)
  e=$(echo "$1" | grep -E "error:" | grep -vc "\.o\.d'")
  echo "w=$w e=$e"
}

ok=1; files=()
for arg in "$@"; do
  f="${arg%%@*}"; tu=""
  [ "$f" != "$arg" ] && tu="${arg#*@}"
  rel="$(kkp_relpath "$f")" || exit 1
  files+=("$rel")
  if [ -z "$tu" ]; then
    case "$rel" in
      *.cpp) tu="$rel" ;;
      *.h)
        b="$(basename "$rel")"
        tu="$(cd "$KKP_REPO" && grep -l "#include \"$b\"" src/KOKKOS/*.cpp 2> /dev/null | head -1)"
        [ -n "$tu" ] || { echo "  $rel: no including src/KOKKOS/*.cpp found; use $b@<tu.cpp>"; ok=0; continue; }
        ;;
      *) echo "  $rel: not a .cpp or .h file"; ok=0; continue ;;
    esac
  fi
  tu="$(kkp_relpath "$tu")" || exit 1
  mo="$(kkp_compile mixed "$tu")" || exit 1
  so="$(kkp_compile single "$tu")" || exit 1
  mr="$(count "$mo" "$rel")"; sr="$(count "$so" "$rel")"
  via=""; [ "$tu" != "$rel" ] && via=" (via $(basename "$tu"))"
  echo "  $(basename "$rel")$via: mixed($mr) single($sr)"
  [ "$mr" = "w=0 e=0" ] && [ "$sr" = "w=0 e=0" ] || ok=0
done

if [ $ok -ne 1 ]; then
  echo "  FAILED${MSG:+ -- NOT COMMITTED}"; exit 1
fi
if [ -n "$MSG" ]; then
  sign="--no-gpg-sign"; [ "${KKP_SIGN:-0}" = 1 ] && sign=""
  (cd "$KKP_REPO" && git add -- "${files[@]}" && git commit -q $sign -m "$MSG" -- "${files[@]}") \
    && echo "  COMMITTED $(git -C "$KKP_REPO" rev-parse --short HEAD)" || { echo "  COMMIT FAILED"; exit 1; }
else
  echo "  CLEAN (not committed; pass --commit \"<msg>\" to commit)"
fi
