#!/usr/bin/env bash
# paper-wnoise unit (Unit A): top driver. Smoke first (plumbing check, never a number
# source), then an automated gate, then the four real cells via run_cells.sh.
# Detached with setsid/nohup by the caller; all output to driver_top.log.
set -uo pipefail
cd /home/benja/projects/personal/rngrn/worktrees/paper-wnoise

echo "=== $(date -Iseconds) SMOKE starting (sigma_w=0.048, 2 seeds, 50 steps) ==="
bash scripts/guarded_run.sh timeout 2400 .venv/bin/python -m rngrn.cli \
  --runs-root experiments/claim_wnoise/smoke target-report \
  --config configs/nc1_m3_registry.yaml --dataset-id three_gene_qvar --sample-key sample_0001 \
  --form nc1 --seeds 0 1 --workers 2 \
  -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
  -o train.n_restarts=64 -o train.lbfgs_steps=0 -o train.adam_steps=50 \
  -o loss.weights.param_prior=1.0 -o loss.weights.turing=8.0 -o loss.weights.kstar=8.0 \
  -o train.weight_noise_sigma=0.048 -o train.weight_noise_seed=5348 \
  > experiments/claim_wnoise/smoke.driver.log 2>&1
rc=$?
echo "=== $(date -Iseconds) SMOKE exit ${rc} ==="

# ---- gate: the knobs must be in the frozen config, and both seeds must have run ----
gate_ok=1
fc=$(ls experiments/claim_wnoise/smoke/runs/*/config/frozen_config.yaml 2>/dev/null | head -1)
if [ "$rc" -ne 0 ]; then echo "GATE FAIL: smoke exit ${rc}"; gate_ok=0; fi
if [ -z "$fc" ] || ! grep -q "weight_noise_sigma: 0.048" "$fc" || ! grep -q "weight_noise_seed: 5348" "$fc"; then
  echo "GATE FAIL: weight-noise knobs not in frozen_config (${fc:-none})"; gate_ok=0
fi
nruns=$(wc -l < experiments/claim_wnoise/smoke/runs.jsonl 2>/dev/null || echo 0)
if [ "$nruns" -ne 2 ]; then echo "GATE FAIL: expected 2 smoke run rows, got ${nruns}"; gate_ok=0; fi

if [ "$gate_ok" -ne 1 ]; then
  echo "=== $(date -Iseconds) SMOKE GATE FAILED -- real cells NOT launched ==="
  exit 1
fi
echo "=== $(date -Iseconds) SMOKE GATE PASSED (frozen config carries the knobs, ${nruns} rows) ==="

bash experiments/claim_wnoise/run_cells.sh
echo "=== $(date -Iseconds) TOP DRIVER DONE ==="
