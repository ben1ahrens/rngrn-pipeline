# experiments/claim_wnoise/ — train-time weight noise vs recovery robustness

Four cells, one target (`three_gene_qvar/sample_0001`), 8 seeds each, at four train-time
weight-noise levels (per-step lognormal multiplicative on the positive physical parameters,
smoothed-objective estimator — mechanism and pre-registered analysis in
`docs/DECISIONS.md::D-WNOISE-1`). Deliverable is the measured curve against the σ_w=0
control, not a pass/fail verdict. **The measured result is negative for the
noise-helps-robustness claim at these levels — see the Outcome block of D-WNOISE-1 and
`docs/PAPER_CLAIM_WNOISE.md`.**

## Headline table (from `summary_table.json`, medians unless stated)

| σ_w | Turing-unstable | vol_20pct median | k*_fft rel-err median | morphology match | wall |
|---|---|---|---|---|---|
| 0.0 (control) | 8/8 | 0.935 | 0.031 | 7/7 compared | ~43.5 min |
| 0.048 | 6/8 | 0.992 (survivors only) | 0.059 | 4/5 compared | ~3.2 min |
| 0.10 | 1/8 | 0.880 (1 seed) | 0.621 | 0 compared | ~3.0 min |
| 0.20 | 0/8 | n/a (no survivors) | 0.977 | 0 compared | ~8.2 min |

All 8 seeds at every level *produce* a model (`n_recovered` 8/8); what collapses is
Turing instability of the recovered model, then rollout patterning (0 of 8 seeds pattern
at σ_w ≥ 0.10, including the one Turing-unstable survivor at 0.10).

## Exact commands

Common flags, run through `scripts/guarded_run.sh` (mandatory):

```bash
bash scripts/guarded_run.sh timeout 2400 .venv/bin/python -m rngrn.cli \
  --runs-root experiments/claim_wnoise/sigma_w_XXXX target-report \
  --config configs/nc1_m3_registry.yaml --dataset-id three_gene_qvar --sample-key sample_0001 \
  --form nc1 --seeds 0 1 2 3 4 5 6 7 --workers 4 \
  -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
  -o train.n_restarts=64 -o train.lbfgs_steps=0 -o train.adam_steps=400 \
  -o loss.weights.param_prior=1.0 -o loss.weights.turing=8.0 -o loss.weights.kstar=8.0 \
  [-o train.weight_noise_sigma=SIGMA -o train.weight_noise_seed=SEED]
```

The bracketed overrides are omitted entirely for the σ_w=0 control (untouched default
code path, not a sigma=0 override).

| runs-root | σ_w | weight_noise_seed |
|---|---|---|
| `sigma_w_0p00` | 0.0 (control, no override) | n/a |
| `sigma_w_0p048` | 0.048 | 5348 |
| `sigma_w_0p10` | 0.10 | 5310 |
| `sigma_w_0p20` | 0.20 | 5320 |

Driver: `run_cells.sh` (sequential — cells queue on the repo-global trainer flock).
**Provenance note:** the 0p20 cell's first attempt on 2026-08-19 never ran — it timed out
after 7200 s waiting for the trainer flock behind an unrelated R3-lane run (exit 4 in
`0p20.driver.log`) — and was re-launched with the identical invocation at 17:11 the same
day, completing normally (exit 0). Same command, same seeds; nothing about the cell
differs except lock-wait timing.

Analysis: `analyze.py` (pre-registered in D-WNOISE-1 before any cell ran). Outputs
`summary_table.{json,csv}`, `seed_level.csv` (tracked) and `figures/wnoise_curves.png`
(gitignored; regenerate with the script).

## Seeds

Restart seeds `0 1 2 3 4 5 6 7`, identical across all four levels. `weight_noise_seed`
necessarily differs per level (it seeds the noise stream itself).

## Caveats

- **One target, 8 seeds per level** — not a general claim about weight-noise training;
  the direction at these levels on this target is, however, unambiguous.
- **Probe levels UNCALIBRATED by design** (D-WNOISE-1): {0.048, 0.10, 0.20} are probe
  points for a curve, not calibrated thresholds. Levels below 0.048 were not probed;
  this experiment cannot rule out a benefit at much smaller σ_w.
- **Survivor bias at 0.048:** the vol_20pct median 0.992 is over the 6 Turing-unstable
  survivors; the control's 0.935 is over all 8. Read alongside the 2/8 failures and the
  ~2× k* error cost, not in isolation.
- **Volume columns are NaN by construction where no Turing-unstable model exists**
  (`turing_volume_*` is defined only for a Turing-unstable recovery); morphology is
  "0 compared" at σ_w ≥ 0.10 because no seed produced a patterning rollout to compare.
- **The wall-time collapse (43 min → ~3 min) is itself a symptom**, consistent with the
  documented kill-on-first-failure contract: noise-induced Newton steady-state failures
  kill batch members early, so the batched pool shrinks and the run finishes fast with
  little surviving optimisation. It is not a speedup.
- Identical `kstar_fft_rel_err` values recurring across seeds/levels are k-grid
  quantisation (k*_model snaps to one of 400 grid points), not duplicated runs; 0.977 is
  the grid-extreme value that marks a fully failed k* fit.
