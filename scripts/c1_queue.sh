#!/usr/bin/env bash
# c1_queue.sh — the DETACHED C1 tuning queue.
#
# Runs one (cell, sample) job at a time, in pre-registered priority order, writing the
# target-report JSON only once the command has SUCCEEDED (tmp file + mv), so a killed
# process can never leave a zero-byte file that looks like a result. Each job is skipped
# if a NON-EMPTY report already exists, so re-launching resumes rather than redoing.
#
# Launch detached:
#   setsid nohup bash scripts/c1_queue.sh > experiments/tune_comp_reports/queue.log 2>&1 </dev/null &
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # THIS worktree, never a hardcoded one (D-EVID-16)
VENV="$WT/.venv/bin/python"
export PYTHONPATH=$WT/src KMP_AFFINITY=disabled OMP_NUM_THREADS=1
cd "$WT"

REPORTS=$WT/experiments/tune_comp_reports
WORKERS=${WORKERS:-4}
TIMEOUT=${TIMEOUT:-7200}          # per-job wall-clock ceiling, seconds

SHARED=(-o train.batched=true -o train.device=cuda
        -o model.dispersion_backend=cubic
        -o train.n_restarts=64 -o train.lbfgs_steps=0
        -o train.adam_steps=400)

# job <cell> <runs_root> <dataset_id> <sample_key> "<seeds>" [extra -o ...]
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
  echo "$(date -Is) START $cell/$skey seeds=[$seeds] extra=[$*]"
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
RL=experiments/tune_comp_legacy

# ---- 1. the top axis: detach_xstar, against the committed baseline on sample_0000 ----
job detach   $R  three_gene_qvar sample_0000 "0 1 2 3 4 5 6 7" -o loss.detach_xstar=true

# ---- 2. the legacy CONTROL (one-off; interprets the 0.125 vs 0.3684 gap) ----
job legacy_control $RL three_gene_val sample_0000 "0 1 2 3 4 5 6 7"

# ---- 3. finish the baseline / detach pair on the patterning target ----
job baseline $R  three_gene_qvar sample_0003 "0 1 2 3 4 5 6 7"
# NOTE the override. Without it this line runs a BASELINE and writes the result into the
# detach cell's report, and the skip-if-non-empty guard makes that permanent. It was the only
# `job detach` line in any queue script missing it (:60 above, and c1_queue2.sh:67/:75 and
# c1_queue3.sh:103, all carry it). Nothing in the record is wrong -- detach/ holds only
# sample_0000.json -- but baseline/sample_0003.json now exists, so on relaunch line 66 SKIPs
# and this one fires immediately.
job detach   $R  three_gene_qvar sample_0003 "0 1 2 3 4 5 6 7" -o loss.detach_xstar=true

# ---- 4. the adam_steps curve: 2 seeds x 2 targets only, per the unit spec ----
job steps2000 $R three_gene_qvar sample_0000 "0 1" -o train.adam_steps=2000
job steps2000 $R three_gene_qvar sample_0003 "0 1" -o train.adam_steps=2000
job steps8000 $R three_gene_qvar sample_0000 "0 1" -o train.adam_steps=8000
job steps8000 $R three_gene_qvar sample_0003 "0 1" -o train.adam_steps=8000

# ---- 5. the remaining single-override axes, sample_0000 first (the hard target) ----
job lowbasal $R  three_gene_qvar sample_0000 "0 1 2 3 4 5 6 7" -o model.init=low_basal
job dinit    $R  three_gene_qvar sample_0000 "0 1 2 3 4 5 6 7" -o model.d_init_from_kstar=true
job prior0p3 $R  three_gene_qvar sample_0000 "0 1 2 3 4 5 6 7" -o loss.weights.param_prior=0.3
job anchor4  $R  three_gene_qvar sample_0000 "0 1 2 3 4 5 6 7" -o loss.weights.anchor=4.0
job anchor0p5 $R three_gene_qvar sample_0000 "0 1 2 3 4 5 6 7" -o loss.weights.anchor=0.5
job staging005 $R three_gene_qvar sample_0000 "0 1 2 3 4 5 6 7" -o loss.staging_off_frac=0.05 -o loss.staging_ramp_frac=0.05

# ---- 6. the same axes on the patterning target, only after every sample_0000 arm ----
job lowbasal $R  three_gene_qvar sample_0003 "0 1 2 3 4 5 6 7" -o model.init=low_basal
job dinit    $R  three_gene_qvar sample_0003 "0 1 2 3 4 5 6 7" -o model.d_init_from_kstar=true
job prior0p3 $R  three_gene_qvar sample_0003 "0 1 2 3 4 5 6 7" -o loss.weights.param_prior=0.3
job anchor4  $R  three_gene_qvar sample_0003 "0 1 2 3 4 5 6 7" -o loss.weights.anchor=4.0

echo "$(date -Is) ALLDONE"
