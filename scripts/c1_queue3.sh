#!/usr/bin/env bash
# c1_queue3.sh — the C1 queue, reordered a SECOND time on measured evidence. Same job
# mechanics as c1_queue.sh (tmp+mv, skip-if-non-empty, per-job timeout); order only.
#
# WHAT CHANGED AND WHY (all three reasons are measurements, recorded in
# docs/C1_COMPETITIVE_TUNING.md sec 6 and sec 7):
#
#  1. `detach` is a MEASURED NULL on the rate: turing_frac 0.125 -> 0.125, the same seed,
#     2/512 -> 2/512 pooled restarts, and the seven collapsed seeds return kstar_model
#     0.00587 to five significant figures in BOTH arms. It drops to the back.
#  2. The legacy control refutes "3.1 is downstream of the Turing rate": 8/8 seeds Turing,
#     kstar_spread 0.0147, and topology_consistency STILL 0.125 with 8 distinct structures.
#     So an axis that only lifts the rate cannot fix the PRIMARY criterion, and an axis that
#     constrains the PARAMETERS becomes a 3.1 axis, not just a 3.4 one. `param_prior` is
#     promoted on that reasoning: Stage 0 measured it pinning the scored d_ratio to 7.50-7.69
#     on 5/5 targets, i.e. it demonstrably collapses part of the solution manifold.
#  3. Every promising cell now runs BOTH targets before the next cell starts, because
#     criterion 3.1's cross-target control needs >= 2 distinct targets to exist at all.
#
# The adam_steps curve keeps its budget but is scoped by measured cost: 400 steps costs
# ~640 s/seed on qvar sample_0000, so 8000 steps is ~3.5 h for a 2-seed cell. 2000 runs on
# both targets; 8000 runs on sample_0000 only, with its own longer timeout.
#
#   setsid nohup bash scripts/c1_queue3.sh > experiments/tune_comp_reports/queue3.log 2>&1 </dev/null &
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
  echo "$(date -Is) START $cell/$skey seeds=[$seeds] timeout=${TIMEOUT}s extra=[$*]"
  local t0=$SECONDS
  timeout -k 60 "$TIMEOUT" $VENV -m rngrn.cli --runs-root "$root" target-report \
      --config configs/m3_registry.yaml --dataset-id "$dsid" \
      --sample-key "$skey" --form competitive --seeds $seeds --workers "$WORKERS" \
      -o data.dataset_id="$dsid" "${SHARED[@]}" "$@" > "$out/$skey.json.tmp"
  local rc=$?
  local dt=$((SECONDS - t0))
  if [ $rc -eq 0 ] && [ -s "$out/$skey.json.tmp" ]; then
    mv "$out/$skey.json.tmp" "$out/$skey.json"
    echo "$(date -Is) OK    $cell/$skey  ${dt}s"
  else
    rm -f "$out/$skey.json.tmp"
    echo "$(date -Is) FAIL  $cell/$skey  rc=$rc ${dt}s"
  fi
}

R=experiments/tune_comp
Q=three_gene_qvar
K8="0 1 2 3 4 5 6 7"

# --- 1. the init axis: the one aimed at the 2/512 reachability number ---
job lowbasal $R $Q sample_0000 "$K8" -o model.init=low_basal
job baseline $R $Q sample_0003 "$K8"
job lowbasal $R $Q sample_0003 "$K8" -o model.init=low_basal

# --- 2. the parameter prior: now a criterion-3.1 axis, not only a 3.4 one ---
job prior1p0 $R $Q sample_0000 "$K8" -o loss.weights.param_prior=1.0
job prior1p0 $R $Q sample_0003 "$K8" -o loss.weights.param_prior=1.0

# --- 3. the D init ---
job dinit    $R $Q sample_0000 "$K8" -o model.d_init_from_kstar=true
job dinit    $R $Q sample_0003 "$K8" -o model.d_init_from_kstar=true

# --- 4. the adam_steps CURVE, 2 seeds x target, scoped by measured cost ---
job steps2000 $R $Q sample_0000 "0 1" -o train.adam_steps=2000
job steps2000 $R $Q sample_0003 "0 1" -o train.adam_steps=2000
TIMEOUT=20000     # the 8000-step cell only; restored on the next line's scope by hand
job steps8000 $R $Q sample_0000 "0 1" -o train.adam_steps=8000
TIMEOUT=7200

# --- 5. a lower-weight prior, only once 1.0 has been read ---
job prior0p3 $R $Q sample_0000 "$K8" -o loss.weights.param_prior=0.3
job prior0p3 $R $Q sample_0003 "$K8" -o loss.weights.param_prior=0.3

# --- 6. the never-swept inherited weights ---
job anchor4   $R $Q sample_0000 "$K8" -o loss.weights.anchor=4.0
job anchor0p5 $R $Q sample_0000 "$K8" -o loss.weights.anchor=0.5
job staging005 $R $Q sample_0000 "$K8" -o loss.staging_off_frac=0.05 -o loss.staging_ramp_frac=0.05

# --- 7. detach on the patterning target, LAST: it is a measured null on sample_0000 ---
job detach   $R $Q sample_0003 "$K8" -o loss.detach_xstar=true

echo "$(date -Is) ALLDONE"
