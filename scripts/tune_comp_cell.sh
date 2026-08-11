#!/usr/bin/env bash
# tune_comp_cell.sh — run ONE tuning cell (one override set) over N targets x K seeds.
#
#   ./tune_comp_cell.sh <cell_name> "<sample_key ...>" "<seed ...>" [extra -o args ...]
#
# Writes one target-report JSON per target under $OUT/<cell_name>/<sample_key>.json and
# the run records under experiments/tune_comp/. Every cell shares the batched-CUDA block
# below so cells differ ONLY by the extra overrides passed in.
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # THIS worktree, never a hardcoded one (D-EVID-16)
VENV="$WT/.venv/bin/python"
export PYTHONPATH=$WT/src KMP_AFFINITY=disabled OMP_NUM_THREADS=1
cd "$WT"

CELL=$1; shift
SAMPLES=$1; shift
SEEDS=$1; shift

OUT=$WT/experiments/tune_comp_reports/$CELL
mkdir -p "$OUT"

BASE_OV=(-o data.dataset_id=three_gene_qvar
         -o train.batched=true -o train.device=cuda
         -o model.dispersion_backend=cubic
         -o train.n_restarts=64 -o train.lbfgs_steps=0
         -o train.adam_steps=400)

for S in $SAMPLES; do
  if [ -s "$OUT/$S.json" ]; then echo "skip $CELL/$S (exists)"; continue; fi
  echo "=== $CELL / $S ==="
  $VENV -m rngrn.cli --runs-root experiments/tune_comp target-report \
      --config configs/m3_registry.yaml --dataset-id three_gene_qvar \
      --sample-key "$S" --form competitive --seeds $SEEDS --workers "${WORKERS:-4}" \
      "${BASE_OV[@]}" "$@" > "$OUT/$S.json"
done
echo "DONE $CELL"
