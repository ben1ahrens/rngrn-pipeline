#!/usr/bin/env bash
# c1_queue4.sh — the C1 queue, reordered on the C2 cross-unit measurement, and the FIRST
# C1 queue to run every trainer invocation under scripts/guarded_run.sh.
#
# TWO CHANGES FROM c1_queue3.sh, both forced by evidence:
#
#  1. THE GUARD. c1_queue3.sh called the trainer directly. Five sessions have been ended by
#     the WSL2 VM's global OOM killer, and the overcommit is the SUM over concurrently
#     running agents, which no single agent can see. Every job below goes through
#     scripts/guarded_run.sh, which takes ONE flock shared across all worktrees, waits for
#     MemAvailable >= 8192 MB, and raises its own oom_score_adj so the kernel kills a
#     trainer rather than the session. A job that blocks on the lock is the guard WORKING.
#
#  2. `loss.weights.turing` LEADS. It is not in this unit's original axis list and C1 has
#     never tried it. C2 measured `-o loss.weights.turing=8.0` taking nc1 on qvar
#     sample_0000 from turing_frac 0.0625 to 0.750 at K=8 -- the largest rate movement
#     measured anywhere in this project, and larger than anything C1's own eight axes have
#     produced. The axis is proven LIVE here on both execution paths before any cell is
#     believed (docs/C1_COMPETITIVE_TUNING.md sec 1).
#
# ORDERING RULE. Every cell that could plausibly become `best_config` runs BOTH targets --
# sample_0000 (hard, exposes the rate problem) and sample_0003 (patterns, so criteria
# 3.2-3.4 are computable at all) -- because criterion 3.1's cross-target control needs >= 2
# distinct targets to exist. Pure curve points (the turing weight sweep) run sample_0000
# only; they are read for the shape of the rate curve, not for 3.1.
#
# `legacy_control` gains a SECOND legacy target. That is a criterion-3.1 instrument, not a
# rate cell: on legacy data all 8 seeds pattern and agree on k* to 1.5 %, so a within- vs
# cross-target comparison computed ENTIRELY INSIDE the legacy runs-root isolates the
# J-degeneracy from the Turing rate completely. PREREGISTRATION section 1 forbids legacy
# data from supporting any claim; it is a labelled control in a separate runs-root and its
# numbers are never pooled with three_gene_qvar numbers.
#
#   setsid nohup bash scripts/c1_queue4.sh > experiments/tune_comp_reports/queue4.log 2>&1 </dev/null &
set -uo pipefail
WT=/home/benja/projects/personal/rngrn/worktrees/c-tune-comp
VENV=/home/benja/projects/personal/rngrn/worktrees/turing-training/.venv/bin/python
export PYTHONPATH=$WT/src KMP_AFFINITY=disabled OMP_NUM_THREADS=1
cd "$WT"

REPORTS=$WT/experiments/tune_comp_reports
WORKERS=${WORKERS:-4}
TIMEOUT=${TIMEOUT:-7200}

SHARED=(-o train.batched=true -o train.device=cuda
        -o model.dispersion_backend=cubic
        -o train.n_restarts=64 -o train.lbfgs_steps=0
        -o train.adam_steps=400)

job () {
  local cell=$1; shift
  local root=$1; shift
  local dsid=$1; shift
  local skey=$1; shift
  local seeds=$1; shift
  local out=$REPORTS/$cell
  mkdir -p "$out"
  if [ -s "$out/$skey.json" ]; then
    echo "$(date -Is) SKIP  $cell/$skey (non-empty report exists)"
    return 0
  fi
  echo "$(date -Is) START $cell/$skey ds=$dsid seeds=[$seeds] timeout=${TIMEOUT}s extra=[$*]"
  local t0=$SECONDS
  timeout -k 60 "$TIMEOUT" bash scripts/guarded_run.sh \
      $VENV -m rngrn.cli --runs-root "$root" target-report \
      --config configs/m3_registry.yaml --dataset-id "$dsid" \
      --sample-key "$skey" --form competitive --seeds $seeds --workers "$WORKERS" \
      -o data.dataset_id="$dsid" "${SHARED[@]}" "$@" > "$out/$skey.json.tmp"
  local rc=$?
  local dt=$((SECONDS - t0))
  # A report lands ONLY on rc=0 AND non-empty, via tmp+mv: a killed process must never
  # leave a zero-byte file that a later reader takes for a measurement. Several partial
  # cells have already been mistaken for results in this project.
  if [ $rc -eq 0 ] && [ -s "$out/$skey.json.tmp" ]; then
    mv "$out/$skey.json.tmp" "$out/$skey.json"
    echo "$(date -Is) OK    $cell/$skey  ${dt}s"
  else
    rm -f "$out/$skey.json.tmp"
    echo "$(date -Is) FAIL  $cell/$skey  rc=$rc ${dt}s"
  fi
}

R=experiments/tune_comp
RL=experiments/tune_comp_legacy
Q=three_gene_qvar
V=three_gene_val
K8="0 1 2 3 4 5 6 7"

# --- 1. THE RATE LEVER C1 HAS NEVER TRIED. Both targets, in priority order. ---
job turing8  $R $Q sample_0000 "$K8" -o loss.weights.turing=8.0
job baseline $R $Q sample_0003 "$K8"
job turing8  $R $Q sample_0003 "$K8" -o loss.weights.turing=8.0

# --- 2. JOB B instrument: a SECOND legacy target so 3.1's cross-target control can be
#        computed inside a set where the Turing rate is 8/8 and cannot be the confound. ---
job legacy_control $RL $V sample_0001 "$K8"
job legacy_control $RL $V sample_0002 "$K8"

# --- 3. the init axis, and its combination with the rate lever ---
job lowbasal        $R $Q sample_0000 "$K8" -o model.init=low_basal
job turing8_lowbasal $R $Q sample_0000 "$K8" -o loss.weights.turing=8.0 -o model.init=low_basal

# --- 4. the turing-weight CURVE. sample_0000 only: read for shape, not for 3.1. ---
job turing2  $R $Q sample_0000 "$K8" -o loss.weights.turing=2.0
job turing4  $R $Q sample_0000 "$K8" -o loss.weights.turing=4.0
job turing16 $R $Q sample_0000 "$K8" -o loss.weights.turing=16.0

# --- 5. the second target for whichever of 3 wins, plus the remaining rate axes ---
job lowbasal         $R $Q sample_0003 "$K8" -o model.init=low_basal
job turing8_lowbasal $R $Q sample_0003 "$K8" -o loss.weights.turing=8.0 -o model.init=low_basal
job dinit    $R $Q sample_0000 "$K8" -o model.d_init_from_kstar=true
job prior1p0 $R $Q sample_0000 "$K8" -o loss.weights.param_prior=1.0

echo "$(date -Is) ALLDONE"
