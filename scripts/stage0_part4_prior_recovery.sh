#!/usr/bin/env bash
# stage0_part4_prior_recovery.sh — PART 4 of the Stage-0 biological-viability check.
#
# Recover five REAL targets (three_gene_qvar sample_0000..0004 — all inside the §1a TUNING
# range, so nothing held out is burned) TWICE: once with the biological prior ACTIVE
# (loss.weights.param_prior=1.0, i.e. the D-ratio log-normal centred at the literature
# 7.5 plus the soft alpha/delta box hinges) and once with it OFF. The OFF arm is the
# control: without it, "the prior made this implausible/unpatterned" has nothing to be
# measured against.
#
# NOTE the prior only became reachable in this branch. Before it, loss.weights.param_prior
# was a NO-OP on the path recover.py runs (losses/total.py never called terms.param_prior),
# so the ON arm below would have been byte-identical to the OFF arm.
#
# HARD CONSTRAINTS, all measured, all in model.py / recover.py docstrings:
#   * dispersion_backend=cubic is REQUIRED on CUDA (eig has no batched cuSOLVER kernel for
#     small non-symmetric problems: 816 ms per restart-step, FLAT in B, 2500x worse).
#   * batched=true REQUIRES lbfgs_steps=0, resid weight 0 and m==N.
#   * NEVER float32 (45x slower at B=128, and torch 2.13's linalg.det backward builds its
#     LU at the global default dtype so it does not even run).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" KMP_AFFINITY=disabled OMP_NUM_THREADS=1
PY=/home/benja/projects/personal/rngrn/worktrees/turing-training/.venv/bin/python
ROOT=experiments/stage0_bioviab

for s in 0000 0001 0002 0003 0004; do
  for arm in prior_on prior_off; do
    w=1.0; [ "$arm" = prior_off ] && w=0.0
    echo "=== sample_${s}  ${arm}  (param_prior weight ${w}) ==="
    "$PY" -m rngrn.cli --runs-root "$ROOT" train --config configs/m3_registry.yaml \
      -o data.dataset_id=three_gene_qvar \
      -o "data.sample_key=sample_${s}" \
      -o train.batched=true -o train.device=cuda \
      -o model.dispersion_backend=cubic \
      -o train.n_restarts=64 -o train.lbfgs_steps=0 -o train.adam_steps=400 \
      -o "loss.weights.param_prior=${w}" \
      -o "tracking.run_name=stage0_${arm}_${s}"
  done
done
