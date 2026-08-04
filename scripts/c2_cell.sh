#!/usr/bin/env bash
# c2_cell.sh — run ONE tuning cell for unit C2 (nc1): the same -o override list over
# several three_gene_qvar targets, K seeds each, one target-report per target.
#
#   c2_cell.sh <runs_root> <seeds_csv> <targets_csv> <workers> -- -o k=v -o k=v ...
#
# The compute shape from the unit brief is applied here so no cell can silently omit it.
# NOT -e: a target that times out must be RECORDED and the cell must continue to the next
# target, rather than aborting the queue on one pathological sample.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # THIS worktree, never a hardcoded one (D-EVID-16)
VENV="$WT/.venv/bin/python"
export PYTHONPATH=$WT/src KMP_AFFINITY=disabled OMP_NUM_THREADS=1

ROOT="$1"; SEEDS_CSV="$2"; TARGETS_CSV="$3"; WORKERS="$4"; shift 4
[ "${1:-}" = "--" ] && shift
SEEDS=${SEEDS_CSV//,/ }
# PER-TARGET TIMEOUT. C2 measured a ~30x per-target cost spread at identical settings
# (sample_0002: >54 min for 4 of 8 seeds against ~90 s for all 8 of sample_0000). Without a
# cap one pathological target eats the whole unit. A timed-out target is REPORTED as such,
# never silently dropped from a denominator.
# 3600 s, and DELIBERATELY IDENTICAL to competitive's (scripts/c1_queue5.sh in c-tune-comp).
# The two units had chosen independently -- nc1 1800 s defaulted and overridden to 2400 s at
# launch, competitive 7200 s -- for no principled reason, and it biased the comparison
# between the forms. A timed-out target yields rc=124 and NO target report: a MISSING
# measurement, not a measured failure. PREREGISTRATION section 2 forbids a seed being a
# silent drop from the denominator, and a target dropped for cost is the same error one level
# up, with section 3 reading ">= 4 of 6 targets". nc1 is BOTH the binding form AND the more
# expensive one -- the ~30x cost spread tracks non-patterning and nc1 patterns less -- so the
# smaller budget fell on the form that needed it most, and nc1/sample_0003 was already lost
# that way. 3600 s is ~3.5x the ~1030 s observed for well-behaved targets (legacy 370 s,
# qvar/sample_0003 1024 s, nc1/sample_0001 1053 s), so a timeout now means GENUINELY
# pathological rather than unlucky.
TIMEOUT=${C2_TIMEOUT:-3600}

cd "$WT"
for t in ${TARGETS_CSV//,/ }; do
  echo "=== CELL $ROOT target=$t overrides: $* ==="
  t0=$(date +%s)
  # EVERY trainer invocation goes through scripts/guarded_run.sh. Five sessions were ended
  # by the WSL2 VM's global OOM killer because two agents each ran a 5-process pool; the
  # guard takes one flock shared across ALL worktrees, waits for MemAvailable >= 8 GiB and
  # raises its own oom_score_adj so a TRAINER is killed rather than the session.
  # The timeout is INSIDE the guard on purpose: waiting for the lock is the guard WORKING,
  # and must not be charged against a budget meant for pathological TRAINING cost.
  bash scripts/guarded_run.sh timeout "$TIMEOUT" \
      $VENV -m rngrn.cli --runs-root "$ROOT" target-report \
      --config configs/nc1_m3_registry.yaml \
      --dataset-id three_gene_qvar --sample-key "$t" --form nc1 \
      --seeds $SEEDS --workers "$WORKERS" \
      -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
      -o train.n_restarts=64 -o train.lbfgs_steps=0 "$@" > /dev/null < /dev/null
  rc=$?
  echo "=== TARGET $ROOT $t rc=$rc secs=$(( $(date +%s) - t0 )) ==="
done
echo "=== CELL DONE $ROOT ==="
