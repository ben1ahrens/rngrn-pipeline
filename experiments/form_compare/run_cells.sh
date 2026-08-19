#!/usr/bin/env bash
# form-robustness unit: sequential driver for the two competitive-form cells
# (sample_0001 then sample_0004), mirroring experiments/claim5_obs_noise/run_cells.sh.
# Hyperparameters are copied verbatim from the nc1 c2_P_t8k8_consol winning cell
# (docs/C2_NC1_TUNING.md §2) with --form switched to competitive and no obs-noise
# overrides. See docs/DECISIONS.md D-FORMCOMP-1 for the design and the disclosed
# nc1-tuned-hyperparameter confound.
set -uo pipefail
cd /home/benja/projects/personal/rngrn/worktrees/paper-formcomp

COMMON="--config configs/m3_registry.yaml --dataset-id three_gene_qvar \
  --form competitive --seeds 0 1 2 3 4 5 6 7 --workers 4 \
  -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
  -o train.n_restarts=64 -o train.lbfgs_steps=0 -o train.adam_steps=400 \
  -o loss.weights.param_prior=1.0 -o loss.weights.turing=8.0 -o loss.weights.kstar=8.0"

run_cell() {
  local tag="$1"; shift
  local log="experiments/form_compare/${tag}.driver.log"
  echo "=== $(date -Iseconds) starting cell ${tag} ===" | tee -a "$log"
  bash scripts/guarded_run.sh timeout 2400 .venv/bin/python -m rngrn.cli \
    --runs-root "experiments/form_compare/comp_${tag#sample_}" target-report \
    $COMMON --sample-key "$tag" >> "$log" 2>&1
  local rc=$?
  echo "=== $(date -Iseconds) cell ${tag} exit ${rc} ===" | tee -a "$log"
  return $rc
}

run_cell "sample_0001"
run_cell "sample_0004"

echo "=== $(date -Iseconds) ALL CELLS DONE ===" | tee -a experiments/form_compare/driver_overall.log
