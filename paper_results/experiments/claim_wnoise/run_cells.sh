#!/usr/bin/env bash
# paper-wnoise unit (Unit A): sequential driver for the four weight-noise cells.
# Design + pre-registered analysis: docs/DECISIONS.md::D-WNOISE-1.
# Each cell queues on the repo-global trainer flock via guarded_run.sh; run them one
# at a time here rather than backgrounding all four, since a second concurrent pool
# would blow the memory ceiling even after the flock clears (CLAUDE.md 7a).
# Order is control first, then the extreme, so the informative pair exists if time
# runs out: 0p00, 0p20, 0p048, 0p10.
set -uo pipefail
cd /home/benja/projects/personal/rngrn/worktrees/paper-wnoise

COMMON="--config configs/nc1_m3_registry.yaml --dataset-id three_gene_qvar --sample-key sample_0001 \
  --form nc1 --seeds 0 1 2 3 4 5 6 7 --workers 4 \
  -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
  -o train.n_restarts=64 -o train.lbfgs_steps=0 -o train.adam_steps=400 \
  -o loss.weights.param_prior=1.0 -o loss.weights.turing=8.0 -o loss.weights.kstar=8.0"

run_cell() {
  local tag="$1"; shift
  local extra="$1"; shift
  local log="experiments/claim_wnoise/${tag}.driver.log"
  echo "=== $(date -Iseconds) starting cell ${tag} ===" | tee -a "$log"
  eval bash scripts/guarded_run.sh timeout 2400 .venv/bin/python -m rngrn.cli \
    --runs-root "experiments/claim_wnoise/sigma_w_${tag}" target-report \
    $COMMON $extra >> "$log" 2>&1
  local rc=$?
  echo "=== $(date -Iseconds) cell ${tag} exit ${rc} ===" | tee -a "$log"
  return $rc
}

# control: NO weight-noise overrides at all -- exercises the untouched default path
run_cell "0p00" ""
run_cell "0p20" "-o train.weight_noise_sigma=0.2 -o train.weight_noise_seed=5320"
run_cell "0p048" "-o train.weight_noise_sigma=0.048 -o train.weight_noise_seed=5348"
run_cell "0p10" "-o train.weight_noise_sigma=0.1 -o train.weight_noise_seed=5310"

echo "=== $(date -Iseconds) ALL CELLS DONE ===" | tee -a experiments/claim_wnoise/driver_overall.log
