#!/bin/bash
# squashverify.sh -- squash <base>..HEAD into ONE commit and prove nothing changed
#
# Usage: squashverify.sh <base> (-F <message-file> | -m "<message>") [--dry-run]
#
# <base> is the tip of the pull request target (e.g. origin/develop after a
# fresh "git fetch"); it must be an ancestor of HEAD, i.e. already merged
# into the branch, otherwise the squashed commit would contain its changes.
#
# Steps:
#   1. require a clean work tree (no staged or unstaged changes to tracked files)
#   2. record OLD=HEAD and TREE=HEAD^{tree}; create the backup branch
#      kkp-backup-pre-squash-<short OLD>
#   3. git reset --soft <base>; commit the whole range as one commit with the
#      configured identity (KKP_AUTHOR_NAME/KKP_AUTHOR_EMAIL, default git
#      user.name/user.email) as author AND committer, unsigned unless
#      KKP_SIGN=1, and with exactly the given message (no trailers added)
#   4. verify: the tree hash is unchanged, "git diff OLD HEAD" is empty,
#      exactly one commit in <base>..HEAD, its parent is <base>, no trailer
#      lines (Co-Authored-By, Claude-Session, Generated with, ...), author and
#      committer are the configured identity, signature status N (unsigned)
#   On any failure HEAD is moved back to OLD ("git reset --soft OLD", which is
#   lossless because the trees are identical) and the backup branch is kept.
#
# --dry-run only runs the checks of step 1 and prints what would happen.
#
# The script never pushes.  It prints the push command (force-with-lease
# pinned to the SHA the remote branch had) and the post-push check.  Delete
# the backup branch only after the push has been verified.

source "$(dirname "${BASH_SOURCE[0]}")/kkp-env.sh"
[ $# -ge 3 ] || { sed -n '2,29p' "$0"; exit 1; }
BASE="$1"; shift; MSGARGS=(); DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    -F) [ -f "$2" ] || kkp_die "no message file $2"; MSGARGS=(-F "$(realpath "$2")"); shift 2 ;;
    -m) MSGARGS=(-m "$2"); shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) kkp_die "unknown argument $1" ;;
  esac
done
[ ${#MSGARGS[@]} -eq 2 ] || kkp_die "give the commit message with -F <file> or -m <msg>"
kkp_check_repo; kkp_identity
g() { git -C "$KKP_REPO" "$@"; }

BASESHA="$(g rev-parse --verify -q "$BASE^{commit}")" || kkp_die "unknown base $BASE"
OLD="$(g rev-parse HEAD)"; TREE="$(g rev-parse "HEAD^{tree}")"
BR="$(g symbolic-ref --short -q HEAD)" || kkp_die "HEAD is detached; check out the branch first"
[ -z "$(g status --porcelain --untracked-files=no)" ] || kkp_die "work tree has uncommitted changes"
g merge-base --is-ancestor "$BASESHA" "$OLD" || kkp_die "$BASE is not an ancestor of HEAD; merge it first"
N="$(g rev-list --count "$BASESHA..$OLD")"
[ "$N" -ge 1 ] || kkp_die "nothing to squash in $BASE..HEAD"
echo "branch $BR: $N commits in $BASE..HEAD, tree $TREE"
echo "identity: $KKP_AUTHOR_NAME <$KKP_AUTHOR_EMAIL>, signing: $([ "${KKP_SIGN:-0}" = 1 ] && echo allowed || echo off)"
[ $DRY -eq 1 ] && { echo "dry run: no changes made"; exit 0; }

BACKUP="kkp-backup-pre-squash-$(g rev-parse --short "$OLD")"
g branch -f "$BACKUP" "$OLD" || kkp_die "cannot create backup branch"
fail() { echo "FAILED: $*" 1>&2; g reset -q --soft "$OLD"
         echo "HEAD restored to $OLD; backup branch $BACKUP kept" 1>&2; exit 1; }

g reset -q --soft "$BASESHA" || fail "reset --soft"
kkp_git_commit "${MSGARGS[@]}" || fail "commit"
NEW="$(g rev-parse HEAD)"

[ "$(g rev-parse "HEAD^{tree}")" = "$TREE" ] || fail "tree hash changed"
[ -z "$(g diff "$OLD" "$NEW")" ] || fail "diff against the old HEAD is not empty"
[ "$(g rev-list --count "$BASESHA..$NEW")" = 1 ] || fail "not exactly one commit on top of $BASE"
[ "$(g rev-parse "$NEW^")" = "$BASESHA" ] || fail "parent is not $BASE"
kkp_check_trailers "$NEW" || fail "commit message contains trailer lines"
id="$(g log -1 --format='%an <%ae>|%cn <%ce>' "$NEW")"
want="$KKP_AUTHOR_NAME <$KKP_AUTHOR_EMAIL>"
[ "$id" = "$want|$want" ] || fail "author/committer is '$id', expected '$want' for both"
sig="$(g log -1 --format=%G? "$NEW")"
if [ "${KKP_SIGN:-0}" != 1 ]; then
  [ "$sig" = N ] || fail "commit is signed (status $sig); check commit.gpgsign / gpg.format"
fi

echo "OK: $(g log -1 --format='%h %s' "$NEW")"
echo "    tree $TREE unchanged; 1 commit on $BASE; $want as author and committer; signature $sig"
echo "backup branch: $BACKUP (delete after the push is verified: git branch -D $BACKUP)"
up="$(g rev-parse -q --verify "refs/remotes/origin/$BR" 2> /dev/null)"
echo "push with:  git push --force-with-lease=$BR:${up:-<remote sha>} origin $BR"
echo "then check: git fetch origin $BR && test \"\$(git rev-parse HEAD)\" = \"\$(git rev-parse origin/$BR)\""
