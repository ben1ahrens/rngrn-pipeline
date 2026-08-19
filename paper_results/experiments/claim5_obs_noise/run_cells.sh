#!/usr/bin/env bash
# claim-5 unit: sequential driver for the four noise-level cells.
# Each cell queues on the repo-global trainer flock via guarded_run.sh; run
# them one at a time here rather than backgrounding all four, since a second
# concurrent pool would blow the memory ceiling even after the flock clears
# (see run-training skill / CLAUDE.md 7a).
set -uo pipefail
cd /home/benja/projects/personal/rngrn/worktrees/paper-claim5

COMMON="--config configs/nc1_m3_registry.yaml --dataset-id three_gene_qvar --sample-key sample_0001 \
  --form nc1 --seeds 0 1 2 3 4 5 6 7 --workers 4 \
  -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
  -o train.n_restarts=64 -o train.lbfgs_steps=0 -o train.adam_steps=400 \
  -o loss.weights.param_prior=1.0 -o loss.weights.turing=8.0 -o loss.weights.kstar=8.0"

run_cell() {
  local tag="$1"; shift
  local extra="$1"; shift
  local log="experiments/claim5_obs_noise/${tag}.driver.log"
  echo "=== $(date -Iseconds) starting cell ${tag} ===" | tee -a "$log"
  eval bash scripts/guarded_run.sh timeout 2400 .venv/bin/python -m rngrn.cli \
    --runs-root "experiments/claim5_obs_noise/sigma_${tag}" target-report \
    $COMMON $extra >> "$log" 2>&1
  local rc=$?
  echo "=== $(date -Iseconds) cell ${tag} exit ${rc} ===" | tee -a "$log"
  return $rc
}

run_cell "0p00" ""
run_cell "0p01" "-o data.obs_noise_sigma=0.01 -o data.obs_noise_seed=4201"
run_cell "0p05" "-o data.obs_noise_sigma=0.05 -o data.obs_noise_seed=4205"
run_cell "0p20" "-o data.obs_noise_sigma=0.2 -o data.obs_noise_seed=4220"

echo "=== $(date -Iseconds) ALL CELLS DONE ===" | tee -a experiments/claim5_obs_noise/driver_overall.log
