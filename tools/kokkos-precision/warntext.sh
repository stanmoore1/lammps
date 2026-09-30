#!/bin/bash
# warntext.sh -- print the full clang text of the in-scope warnings of one TU
#
# Usage: warntext.sh <single|mixed|double> <TU.cpp> [file-regex] [-q]
#
# chk.sh prints only file:line:col.  To choose the direction of a cast you
# need the types clang reports, e.g.
#   src/KOKKOS/pair_foo_kokkos.cpp:123:27: warning: implicit conversion
#   increases floating-point precision: 'KK_FLOAT' (aka 'float') to 'double'
#   [-Wdouble-promotion]
# followed by the source line and the caret marking the column.  This script
# compiles the TU syntax-only with the flags captured by mkcmd.sh and prints,
# for every distinct flagged location, the warning line plus clang's source
# and caret lines, sorted by file and line.
#
#   file-regex  only show warnings in paths matching this extended regex
#               (default: the TU itself; use "." for every src/KOKKOS file
#               reached, or a header basename such as pair_kokkos.h)
#   -q          print only the warning lines, without source and caret
#
# Remember: through atomic and scatter views clang may name the target type
# as plain 'double' instead of KK_ACC_FLOAT; the declared view type in the
# header decides (t_kkacc_* = KK_ACC_FLOAT, t_kkfloat_* = KK_FLOAT).

source "$(dirname "${BASH_SOURCE[0]}")/kkp-env.sh"
Q=0; args=()
for a in "$@"; do [ "$a" = "-q" ] && Q=1 || args+=("$a"); done
set -- "${args[@]}"
[ $# -ge 2 ] || { sed -n '2,23p' "$0"; exit 1; }
P="$1"
kkp_check_repo; kkp_check_clang
SRC="$(kkp_relpath "$2")" || exit 1
ONLY="${3:-$(basename "$SRC")}"
OUT="$(kkp_compile "$P" "$SRC")" || exit 1
echo "$OUT" | awk -v wf="\\\\[-(${KKP_WFLAGS})\\\\]" -v only="$ONLY" -v q="$Q" '
  # a diagnostic starts with path:line:col: ; keep in-scope warnings in src/KOKKOS
  /^[^ ]*:[0-9]+:[0-9]+: / { keep = 0 }
  /src\/KOKKOS\/[A-Za-z0-9_]+\.(cpp|h):[0-9]+:[0-9]+: warning:/ {
    if ($0 ~ wf) {
      loc = $0; sub(/: warning:.*/, "", loc); sub(/.*src\/KOKKOS\//, "src/KOKKOS/", loc)
      if (loc ~ only && !(loc in seen)) {
        seen[loc] = 1; keep = 1; n++
        split(loc, p, ":"); key[n] = sprintf("%s:%08d:%08d", p[1], p[2], p[3])
        txt[n] = $0; sub(/^.*src\/KOKKOS\//, "src/KOKKOS/", txt[n]); ctx[n] = 0
        next
      }
    }
  }
  keep && q == 0 && ctx[n] < 2 && !/^In file included/ && !/: note: / {
    txt[n] = txt[n] "\n" $0; ctx[n]++
  }
  END {
    # simple insertion sort by file:line:col
    for (i = 2; i <= n; i++) { k = key[i]; t = txt[i]; j = i - 1
      while (j > 0 && key[j] > k) { key[j+1] = key[j]; txt[j+1] = txt[j]; j-- }
      key[j+1] = k; txt[j+1] = t }
    for (i = 1; i <= n; i++) print txt[i]
    printf("# %d distinct flagged locations\n", n) > "/dev/stderr"
  }'
echo "$OUT" | grep -E "error:" | grep -v "\.o\.d'" | sed 's/^/ERROR: /'
exit 0
