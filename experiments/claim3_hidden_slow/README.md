# claim3_hidden_slow — paper claim 3: recovery from partial observation

**Purpose.** Test paper claim 3, "recovery of a 3N GRN from partial observations that
patterns robustly," on `three_gene_qvar` / `sample_0001`, seeds 0-7. Design fixed by the
controller: hide species 0, the slow channel (see below). Full detail and the measured
results table are in `docs/DECISIONS.md` § D-PAPER-2 — this file is the pointer plus the
exact commands.

**Branch:** `feature/paper-claim3-hidden-channel`, off `main@48441e4`.

## Arms

Run in this order, each through `bash scripts/guarded_run.sh` (mandatory — a repo hook
refuses unguarded trainer launches). Runs-roots are subdirectories of this directory.

### (a) `replica_fullobs` — code-drift check

The exact `experiments/c2_P_t8k8_consol` invocation (CUDA batched, `n_restarts=64`,
`resid=0`, full observation), run on `main@48441e4` to check for drift against the
historical baseline recorded at git `4a61201`. Not restart- or objective-matched to (b)/(c)
— it exists only to confirm the training code has not changed.

```bash
bash scripts/guarded_run.sh timeout 2400 .venv/bin/python -m rngrn.cli \
  --runs-root experiments/claim3_hidden_slow/replica_fullobs target-report \
  --config configs/nc1_m3_registry.yaml --dataset-id three_gene_qvar --sample-key sample_0001 \
  --form nc1 --seeds 0 1 2 3 4 5 6 7 --workers 4 \
  -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
  -o train.n_restarts=64 -o train.lbfgs_steps=0 -o train.adam_steps=400 \
  -o loss.weights.param_prior=1.0 -o loss.weights.turing=8.0 -o loss.weights.kstar=8.0
```

**Result:** bit-identical to the `4a61201` baseline (`kstar_fft_rel_err_mean =
0.03522518377119767` in both). Clean drift check. Wall time ~18 min.

### (b) `ctrl_fullobs_resid` — full-observation control, matched to (c)

Full observation, serial CPU path (forced for (c) below), `resid=0.3`. This is the control
that (c) is read against — same objective, same restart budget, same seeds, only the
observation differs.

```bash
bash scripts/guarded_run.sh timeout 9000 .venv/bin/python -m rngrn.cli \
  --runs-root experiments/claim3_hidden_slow/ctrl_fullobs_resid target-report \
  --config configs/nc1_m3_registry.yaml --dataset-id three_gene_qvar --sample-key sample_0001 \
  --form nc1 --seeds 0 1 2 3 4 5 6 7 --workers 4 \
  -o train.batched=false -o train.device=cpu -o model.dispersion_backend=cubic \
  -o train.n_restarts=32 -o train.lbfgs_steps=0 -o train.adam_steps=400 \
  -o loss.weights.param_prior=1.0 -o loss.weights.turing=8.0 -o loss.weights.kstar=8.0 \
  -o loss.weights.resid=0.3
```

**Result:** 8/8 recovered, 8/8 Turing-unstable, `kstar_fft_rel_err` mean 0.0527 / median
0.0503, morphology 7/8 compared matched (seed 6 Turing-unstable but morphology mismatch —
kept as two distinct claims, see `reporting-numbers.md`). Wall time ~9 min.

### (c) `hidden_slow` — the claim-3 experiment: species 0 hidden

Same as (b) plus `model.m=2`, `model.observed_idx=[1,2]` — species 0 is latent.

```bash
bash scripts/guarded_run.sh timeout 9000 .venv/bin/python -m rngrn.cli \
  --runs-root experiments/claim3_hidden_slow/hidden_slow target-report \
  --config configs/nc1_m3_registry.yaml --dataset-id three_gene_qvar --sample-key sample_0001 \
  --form nc1 --seeds 0 1 2 3 4 5 6 7 --workers 4 \
  -o train.batched=false -o train.device=cpu -o model.dispersion_backend=cubic \
  -o train.n_restarts=32 -o train.lbfgs_steps=0 -o train.adam_steps=400 \
  -o loss.weights.param_prior=1.0 -o loss.weights.turing=8.0 -o loss.weights.kstar=8.0 \
  -o loss.weights.resid=0.3 -o 'model.m=2' -o 'model.observed_idx=[1,2]'
```

**Result:** 8/8 recovered, 7/8 Turing-unstable (turing_frac 0.875, seed 6 the exception),
`kstar_fft_rel_err` mean 0.0743 / median 0.0814, morphology 7/7 compared matched (seed 6
excluded — no model rollout to compare when the recovery isn't Turing-unstable). Wall time
~11 min, after a ~17 min queue wait on the shared trainer lock (other units' cells were
running concurrently).

`-o key=[1,2]` parses correctly as JSON (`config.py::apply_overrides` uses `json.loads` on
the value) — no separate YAML config file was needed; this was verified in pre-flight
before any launch (`config.py:322-325`).

## Why species 0 is "the slow channel"

From `sample_0001`'s ground truth (`payload.h5` `params_json`):
- `D = [1.0, 190.293, 133.179]` — species 0 diffuses ~130-190x slower than species 1 or 2.
- `mu = [0.776, 1.321, 2.595]` — species 0 also has the lowest reaction/degradation rate.

Both criteria independently identify species 0, so the choice is unambiguous.

## Caveat: hidden slow channel is also hidden hub

`interaction_matrix = [[1,-1,-1],[1,0,0],[1,0,0]]`. Species 0 is self-activating and
inhibited by both other species; species 1 and 2 each carry only a self-term. Species 0 is
the *only* species with any cross-species coupling. This design cannot separate "hidden slow
channel" from "hidden hub" — any recovery degradation in (c) vs (b) could be either effect,
or both. See `docs/DECISIONS.md` § D-PAPER-2 for the full discussion, the `resid=0.3`
UNCALIBRATED disclosure (exp06 measured this residual as harmful to Turing recovery at other
sweep points — `recover.py:387-389`), and the measured comparison table.

## Honest summary

`recovered_turing` (dispersion) and `morphology_match` (rollout) are different claims and
are reported separately throughout, per `reporting-numbers.md`. Recovery under the hidden
slow channel/hub (c) is weaker than the matched full-observation control (b) on every
`kstar` error metric and loses one seed's Turing recovery, but still recovers 8/8 and
matches morphology on all 7 seeds where a rollout comparison was possible. Nothing here
supports a NOT-SUPPORTABLE verdict for claim 3 — but the `resid=0.3` / hub confound means the
*size* of the partial-observation cost is not cleanly attributable to observation alone.
