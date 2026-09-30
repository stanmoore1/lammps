#!/bin/bash
# vc.sh -- verify-and-commit gate for a finished batch of KOKKOS files
#
# Usage: vc.sh [--commit "<message>"] [--style] [--tus N] file1 [file2 ...]
#
# For each file, compiles it syntax-only in BOTH the mixed and the single
# precision configuration (flags from mkcmd.sh) and counts the in-scope
# warnings attributed to that file and the compile errors:
#   <file>: mixed(w=N e=N) single(w=N e=N)
# The batch passes only if every count is zero.  Without --commit nothing else
# happens.  A failing batch is never committed.
#
# Headers are compiled through SEVERAL including TUs, because a header warns
# differently in different TUs (template instantiation):
#   header.h@tu1.cpp,tu2.cpp   check through exactly these TUs
#   header.h                   check through the first N src/KOKKOS/*.cpp files
#                              that include it (N = --tus or KKP_HDR_TUS,
#                              default 3; if no .cpp includes it directly, the
#                              TUs including a header that includes it are used)
# Use hdrorigin.sh to find the TU that produced a particular header warning,
# and name that TU explicitly.  The final full rebuild remains authoritative.
#
# --commit "<msg>"  stage and commit exactly these files if the batch passes.
#                   Author AND committer are KKP_AUTHOR_NAME/KKP_AUTHOR_EMAIL
#                   (default: git user.name/user.email); the commit is not
#                   signed unless KKP_SIGN=1; no trailers are added, and the
#                   message is checked for trailers afterwards.
# --style           before committing, also run "make check-whitespace" and
#                   "make check-permissions" in src/ (or set KKP_STYLE=1).
#
# Timing: each TU costs two clang runs of 5-20 s each, a header N times that.
# A foreground tool call with a 2-minute limit fits about 4-6 .cpp files; run
# larger batches in the background, e.g.
#   vc.sh --commit "..." f1 ... f8 > $KKP_WORK/vc.log 2>&1; echo "EXIT=$?" >> $KKP_WORK/vc.log
#
# Files may be given as basenames (looked up in src/KOKKOS), repo-relative, or
# absolute paths.  Exit status: 0 = all clean (and committed if requested),
# 1 = failures.

source "$(dirname "${BASH_SOURCE[0]}")/kkp-env.sh"
MSG=""; STYLE="${KKP_STYLE:-0}"; NTUS="${KKP_HDR_TUS:-3}"
while [ $# -gt 0 ]; do
  case "$1" in
    --commit) [ $# -ge 2 ] || kkp_die "--commit needs a message"; MSG="$2"; shift 2 ;;
    --style)  STYLE=1; shift ;;
    --tus)    [ $# -ge 2 ] || kkp_die "--tus needs a number"; NTUS="$2"; shift 2 ;;
    -h|--help) sed -n '2,38p' "$0"; exit 0 ;;
    --*) kkp_die "unknown option $1" ;;
    *) break ;;
  esac
done
[ $# -ge 1 ] || { sed -n '2,38p' "$0"; exit 1; }
kkp_check_repo; kkp_check_clang

count() {  # count <diagnostics> <repo-relative file>
  local w e
  w=$(echo "$1" | grep -E "${2}:[0-9]+:[0-9]+: warning:.*\[-(${KKP_WFLAGS})\]" \
        | grep -oE "${2}:[0-9]+:[0-9]+" | sort -u | grep -c .)
  e=$(echo "$1" | grep -E "error:" | grep -vc "\.o\.d'")
  echo "w=$w e=$e"
}

# print up to $2 src/KOKKOS/*.cpp files that include header $1 (basename),
# directly or through one intermediate src/KOKKOS header
including_tus() {
  local b="$1" n="$2" tus
  tus="$(cd "$KKP_REPO" && grep -lE "#include +\"$b\"" src/KOKKOS/*.cpp 2> /dev/null)"
  if [ -z "$tus" ]; then
    local h
    for h in $(cd "$KKP_REPO" && grep -lE "#include +\"$b\"" src/KOKKOS/*.h 2> /dev/null); do
      tus+="$(cd "$KKP_REPO" && grep -lE "#include +\"$(basename "$h")\"" src/KOKKOS/*.cpp 2> /dev/null)"$'\n'
    done
  fi
  echo "$tus" | grep . | sort -u | head -n "$n"
}

ok=1; files=()
for arg in "$@"; do
  f="${arg%%@*}"; tulist=""
  [ "$f" != "$arg" ] && tulist="$(echo "${arg#*@}" | tr ',' ' ')"
  rel="$(kkp_relpath "$f")" || exit 1
  files+=("$rel")
  if [ -z "$tulist" ]; then
    case "$rel" in
      *.cpp) tulist="$rel" ;;
      *.h)
        tulist="$(including_tus "$(basename "$rel")" "$NTUS" | tr '\n' ' ')"
        [ -n "${tulist// /}" ] || { echo "  $rel: no including src/KOKKOS/*.cpp found; use $(basename "$rel")@<tu.cpp>"; ok=0; continue; }
        ;;
      *) echo "  $rel: not a .cpp or .h file"; ok=0; continue ;;
    esac
  fi
  for tu in $tulist; do
    tu="$(kkp_relpath "$tu")" || exit 1
    mo="$(kkp_compile mixed "$tu")" || exit 1
    so="$(kkp_compile single "$tu")" || exit 1
    mr="$(count "$mo" "$rel")"; sr="$(count "$so" "$rel")"
    via=""; [ "$tu" != "$rel" ] && via=" (via $(basename "$tu"))"
    echo "  $(basename "$rel")$via: mixed($mr) single($sr)"
    { [ "$mr" = "w=0 e=0" ] && [ "$sr" = "w=0 e=0" ]; } || ok=0
  done
done

if [ $ok -ne 1 ]; then
  echo "  FAILED${MSG:+ -- NOT COMMITTED}"; exit 1
fi
if [ -z "$MSG" ]; then
  echo "  CLEAN (not committed; pass --commit \"<msg>\" to commit)"; exit 0
fi
if [ "$STYLE" = 1 ]; then
  (cd "$KKP_REPO/src" && make -s check-whitespace && make -s check-permissions) \
    || { echo "  STYLE CHECK FAILED -- NOT COMMITTED (try make fix-whitespace / fix-permissions)"; exit 1; }
fi
(cd "$KKP_REPO" && git add -- "${files[@]}") || { echo "  git add FAILED"; exit 1; }
kkp_git_commit -m "$MSG" -- "${files[@]}" || { echo "  COMMIT FAILED"; exit 1; }
echo "  COMMITTED $(git -C "$KKP_REPO" log -1 --format='%h  %an <%ae> / %cn <%ce> / sig %G?')"
kkp_check_trailers HEAD || { echo "  WARNING: the commit message contains trailer lines (a commit hook?); amend it"; exit 1; }
exit 0
