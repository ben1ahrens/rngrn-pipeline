#!/usr/bin/env bash
# c1_queue2.sh — the C1 tuning queue, REORDERED on evidence measured after c1_queue.sh
# was launched. Identical job mechanics (tmp+mv, skip-if-non-empty, per-job timeout);
# only the order differs, and every job already completed by c1_queue.sh is skipped.
#
# WHY THE ORDER CHANGED. scripts/c1_analyse.py's restart block, run on the committed
# baseline cell, measures only **2 of 512** pooled restarts (8 seeds x 64 restarts) with
# sig_max_pos > 0 on sample_0000, p90 = -0.0413. So the k*->0 degeneracy is NOT the
# loss-argmin selection rule discarding Turing restarts -- the Turing regime is essentially
# never REACHED from any of 512 inits. STATE_OF_THE_SCIENCE.md sec 10 has the root cause
# already measured: the RNGRN's Jacobian diagonal is negative at EVERY default init
# (0/200 converged inits had a positive diagonal; 88/88 real three_gene systems do), and
# Turing patterning needs self-activation. The same section measures a low-basal init
# taking Turing-unstable inits from 0 % to 82 %. `model.init=low_basal` is therefore the
# axis aimed straight at the statistic that is failing, and it is promoted to run second.
#
#   setsid nohup bash scripts/c1_queue2.sh > experiments/tune_comp_reports/queue2.log 2>&1 </dev/null &
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # THIS worktree, never a hardcoded one (D-EVID-16)
VENV="$WT/.venv/bin/python"
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
K8="0 1 2 3 4 5 6 7"

# --- the two axes aimed at the Turing RATE, on the hard target, against the baseline ---
job detach   $R  three_gene_qvar sample_0000 "$K8" -o loss.detach_xstar=true
job lowbasal $R  three_gene_qvar sample_0000 "$K8" -o model.init=low_basal

# --- the one-off legacy CONTROL: is 0.125 "qvar is harder" or "something regressed"? ---
job legacy_control $RL three_gene_val sample_0000 "$K8"

# --- the same three arms on the PATTERNING target, so 3.2-3.4 are computable at all ---
job baseline $R  three_gene_qvar sample_0003 "$K8"
job detach   $R  three_gene_qvar sample_0003 "$K8" -o loss.detach_xstar=true
job lowbasal $R  three_gene_qvar sample_0003 "$K8" -o model.init=low_basal

# --- the adam_steps CURVE: 2 seeds x 2 targets only, per the unit spec ---
job steps2000 $R three_gene_qvar sample_0000 "0 1" -o train.adam_steps=2000
job steps2000 $R three_gene_qvar sample_0003 "0 1" -o train.adam_steps=2000
job steps8000 $R three_gene_qvar sample_0000 "0 1" -o train.adam_steps=8000
job steps8000 $R three_gene_qvar sample_0003 "0 1" -o train.adam_steps=8000

# --- the remaining single-override axes, hard target first ---
job dinit    $R  three_gene_qvar sample_0000 "$K8" -o model.d_init_from_kstar=true
job prior0p3 $R  three_gene_qvar sample_0000 "$K8" -o loss.weights.param_prior=0.3
job anchor4  $R  three_gene_qvar sample_0000 "$K8" -o loss.weights.anchor=4.0
job anchor0p5 $R three_gene_qvar sample_0000 "$K8" -o loss.weights.anchor=0.5
job staging005 $R three_gene_qvar sample_0000 "$K8" -o loss.staging_off_frac=0.05 -o loss.staging_ramp_frac=0.05

job dinit    $R  three_gene_qvar sample_0003 "$K8" -o model.d_init_from_kstar=true
job prior0p3 $R  three_gene_qvar sample_0003 "$K8" -o loss.weights.param_prior=0.3
job anchor4  $R  three_gene_qvar sample_0003 "$K8" -o loss.weights.anchor=4.0

echo "$(date -Is) ALLDONE"
