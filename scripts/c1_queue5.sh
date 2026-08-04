#!/usr/bin/env bash
# c1_queue5.sh — the TRIMMED queue, with the timeout defect from c1_queue4.sh fixed.
#
# TWO CHANGES FROM c1_queue4.sh, both about the timeout, and the first one is a real bug.
#
# 1. THE TIMEOUT NOW SITS INSIDE THE GUARD.
#        c1_queue4.sh:  timeout 7200 bash scripts/guarded_run.sh <trainer>   # WRONG
#        here:          bash scripts/guarded_run.sh timeout 3600 <trainer>   # right
#    guarded_run.sh takes a flock shared with the OTHER worktree, so only one trainer pool
#    runs anywhere at a time. With the timeout OUTSIDE, C1's budget started ticking while it
#    was BLOCKED WAITING for C2 to finish a target — so C1 could burn most of its budget
#    queuing, get killed mid-training having computed nothing, and log a `timeout`, which
#    reads as "this target is pathological" when it means "this target never got to start".
#    That would have been a sixth silent failure of the kind this project keeps producing.
#    C2's scripts/c2_cell.sh had it right and says why: waiting for the lock is the guard
#    WORKING, and must not be charged against a budget meant for pathological training cost.
#
# 2. THE BUDGET IS NOW 3600 s AND IS THE SAME FOR BOTH FORMS.
#    c1 (competitive) used 7200 s; c2 (nc1) used 1800 s defaulted, overridden to 2400 s.
#    Nothing principled — the two units chose independently. It matters because a target
#    that times out yields rc=124 and NO target report: a MISSING measurement, not a
#    measured failure. PREREGISTRATION §2 forbids a seed being a silent drop from the
#    denominator; a target dropped for cost is the same error one level up, and §3 reads
#    "≥ 4 of 6 targets". nc1 is BOTH the binding form AND the more expensive one (the ~30x
#    cost spread tracks non-patterning, and nc1 patterns less), so the smaller budget fell
#    on the form that needed it most — nc1/sample_0003 has already been lost this way.
#    3600 s is ~3.5x the ~1030 s observed for well-behaved targets (legacy 370 s,
#    qvar/sample_0003 1024 s, nc1/sample_0001 1053 s), so a timeout now means GENUINELY
#    pathological rather than unlucky. A target that still hits it is reported as a MEASURED
#    cost failure with its wall clock, never as a silent absence.
#
# THE TRIM. c1_queue4.sh queued 14 cells; this queues 4. Dropped: the turing-weight curve
# (2/4/16), d_init_from_kstar, param_prior, and two more legacy controls. They refine
# numbers but cannot change a conclusion — C1's §8 mechanism result already establishes that
# the objective constrains J only through sigma(k), fixing 7 of 9 functions of J and pinning
# the diagonal while leaving the off-diagonal pair senses unconstrained, so NO reweighting of
# existing terms can move criterion 3.1. What is kept is the one untested REACHABILITY lever:
# low_basal, which STATE_OF_THE_SCIENCE §10 measures taking Turing-unstable inits from 0 % to
# 82 %, against a Jacobian diagonal that is negative at every default init while 88/88 real
# three_gene systems are positive. It is settable from config only because of this branch's
# no-op fix, and it has never run under training on the post-B3 solver.
#
# Completed cells are SKIPped on a non-empty report, so re-running this is free.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # THIS worktree, never a hardcoded one (D-EVID-16)
VENV="$WT/.venv/bin/python"
export PYTHONPATH=$WT/src KMP_AFFINITY=disabled OMP_NUM_THREADS=1
cd "$WT"

REPORTS=$WT/experiments/tune_comp_reports
WORKERS=${WORKERS:-4}
TIMEOUT=${TIMEOUT:-3600}

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
  # timeout INSIDE guarded_run.sh — see note 1 above.
  bash scripts/guarded_run.sh timeout -k 60 "$TIMEOUT" \
      $VENV -m rngrn.cli --runs-root "$root" target-report \
      --config configs/m3_registry.yaml --dataset-id "$dsid" \
      --sample-key "$skey" --form competitive --seeds $seeds --workers "$WORKERS" \
      -o data.dataset_id="$dsid" "${SHARED[@]}" "$@" > "$out/$skey.json.tmp"
  local rc=$?
  local dt=$((SECONDS - t0))
  # A report lands ONLY on rc=0 AND non-empty, via tmp+mv: a killed process must never
  # leave a zero-byte file that a later reader takes for a measurement.
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

# The one untested reachability lever, alone and combined with the rate lever that worked.
job lowbasal         $R $Q sample_0000 "$K8" -o model.init=low_basal
job turing8_lowbasal $R $Q sample_0000 "$K8" -o loss.weights.turing=8.0 -o model.init=low_basal
job lowbasal         $R $Q sample_0003 "$K8" -o model.init=low_basal
job turing8_lowbasal $R $Q sample_0003 "$K8" -o loss.weights.turing=8.0 -o model.init=low_basal

echo "$(date -Is) ### QUEUE5 DONE"
