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
WT=/home/benja/projects/personal/rngrn/worktrees/c-tune-nc1
VENV=/home/benja/projects/personal/rngrn/worktrees/turing-training/.venv/bin/python
export PYTHONPATH=$WT/src KMP_AFFINITY=disabled OMP_NUM_THREADS=1

ROOT="$1"; SEEDS_CSV="$2"; TARGETS_CSV="$3"; WORKERS="$4"; shift 4
[ "${1:-}" = "--" ] && shift
SEEDS=${SEEDS_CSV//,/ }
# PER-TARGET TIMEOUT. C2 measured a ~30x per-target cost spread at identical settings
# (sample_0002: >54 min for 4 of 8 seeds against ~90 s for all 8 of sample_0000). Without a
# cap one pathological target eats the whole unit. A timed-out target is REPORTED as such,
# never silently dropped from a denominator.
TIMEOUT=${C2_TIMEOUT:-1800}

cd "$WT"
for t in ${TARGETS_CSV//,/ }; do
  echo "=== CELL $ROOT target=$t overrides: $* ==="
  t0=$(date +%s)
  timeout "$TIMEOUT" $VENV -m rngrn.cli --runs-root "$ROOT" target-report \
      --config configs/nc1_m3_registry.yaml \
      --dataset-id three_gene_qvar --sample-key "$t" --form nc1 \
      --seeds $SEEDS --workers "$WORKERS" \
      -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
      -o train.n_restarts=64 -o train.lbfgs_steps=0 "$@" > /dev/null < /dev/null
  rc=$?
  echo "=== TARGET $ROOT $t rc=$rc secs=$(( $(date +%s) - t0 )) ==="
done
echo "=== CELL DONE $ROOT ==="
