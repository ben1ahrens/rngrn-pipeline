#!/usr/bin/env bash
# c2_cell.sh — run ONE tuning cell for unit C2 (nc1): the same -o override list over
# several three_gene_qvar targets, K seeds each, one target-report per target.
#
#   c2_cell.sh <runs_root> <seeds_csv> <targets_csv> <workers> -- -o k=v -o k=v ...
#
# The compute shape from the unit brief is applied here so no cell can silently omit it.
set -euo pipefail
WT=/home/benja/projects/personal/rngrn/worktrees/c-tune-nc1
VENV=/home/benja/projects/personal/rngrn/worktrees/turing-training/.venv/bin/python
export PYTHONPATH=$WT/src KMP_AFFINITY=disabled OMP_NUM_THREADS=1

ROOT="$1"; SEEDS_CSV="$2"; TARGETS_CSV="$3"; WORKERS="$4"; shift 4
[ "${1:-}" = "--" ] && shift
SEEDS=${SEEDS_CSV//,/ }

cd "$WT"
for t in ${TARGETS_CSV//,/ }; do
  echo "=== CELL $ROOT target=$t overrides: $* ==="
  $VENV -m rngrn.cli --runs-root "$ROOT" target-report \
      --config configs/nc1_m3_registry.yaml \
      --dataset-id three_gene_qvar --sample-key "$t" --form nc1 \
      --seeds $SEEDS --workers "$WORKERS" \
      -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
      -o train.n_restarts=64 -o train.lbfgs_steps=0 "$@" > /dev/null
done
echo "=== CELL DONE $ROOT ==="
