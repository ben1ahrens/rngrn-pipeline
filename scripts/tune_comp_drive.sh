#!/usr/bin/env bash
# tune_comp_drive.sh — run the C1 tuning cells SEQUENTIALLY, in pre-registered priority
# order, writing one target-report JSON per (cell, sample) as it completes so that a
# process exit costs one cell rather than the whole unit.
#
# Each cell differs from `baseline` by EXACTLY ONE override, so a difference between two
# cells is attributable. The shared block below is never varied.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # THIS worktree, never a hardcoded one (D-EVID-16)
cd "$WT"

run_cell () {   # run_cell <cell> "<samples>" "<seeds>" [extra -o ...]
  local cell=$1; shift
  local samples=$1; shift
  local seeds=$1; shift
  WORKERS=${WORKERS:-3} ./scripts/tune_comp_cell.sh "$cell" "$samples" "$seeds" "$@" \
    >> "$WT/experiments/tune_comp_reports/drive.log" 2>&1
  echo "$(date +%H:%M:%S) finished cell=$cell" >> "$WT/experiments/tune_comp_reports/drive.log"
}

T2="sample_0000 sample_0001"
K8="0 1 2 3 4 5 6 7"

run_cell baseline "$T2" "$K8"
run_cell detach   "$T2" "$K8" -o loss.detach_xstar=true
echo ALLDONE >> "$WT/experiments/tune_comp_reports/drive.log"
