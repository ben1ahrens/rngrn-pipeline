#!/usr/bin/env bash
# c2_queue.sh — run a queue of C2 cells back to back, COMMITTING AFTER EACH ONE.
#
# Three background runs have already been lost to session exits. A cell that finishes and
# is not committed is a cell that will be re-run. This wrapper makes the commit part of the
# cell rather than a thing done afterwards, so a kill costs one cell, never the queue.
#
#   c2_queue.sh <queuefile>
# where each non-comment line of <queuefile> is:
#   <root_name>|<seeds_csv>|<targets_csv>|<workers>|<override string>
set -uo pipefail
WT=/home/benja/projects/personal/rngrn/worktrees/c-tune-nc1
cd "$WT"

while IFS='|' read -r name seeds targets workers overrides; do
  case "$name" in ''|\#*) continue ;; esac
  root="experiments/$name"
  echo "### $(date -Is) START $name  [$overrides]"
  # shellcheck disable=SC2086
  bash scripts/c2_cell.sh "$root" "$seeds" "$targets" "$workers" -- $overrides
  rc=$?
  echo "### $(date -Is) END $name rc=$rc"
  git add -A
  git commit -q -m "C2 cell $name: nc1, seeds $seeds, targets $targets, overrides: $overrides

Run records and target reports for one measurement cell. Committed per cell so a
session exit costs one cell rather than the unit.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" || echo "### nothing to commit for $name"
done < "$1"
echo "### QUEUE DONE"
