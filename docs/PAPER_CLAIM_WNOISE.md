# Paper claim — train-time weight noise and recovery robustness

**Status: measured, and NOT SUPPORTED as originally framed.** The hypothesis was that
injecting noise into the weights during training makes training more robust and the
learned GRN pattern more robustly. At the probed levels on this target, the measured
effect runs the other way: weight noise monotonically destroys recovery. An honest
negative is a result (D-WNOISE-1 said so in advance); this section reports it as one.

## Design (pre-registered)

Mechanism, noise model, probe levels and the analysis plan were fixed in
`docs/DECISIONS.md::D-WNOISE-1` **before any cell ran**. Per-step lognormal multiplicative
noise on the positive physical parameters (exact in log/softplus space), loss evaluated at
θ+ε, gradient applied to clean θ — the classic smoothed-objective estimator. Chosen to
match the evaluation perturbation model (`eval/analysis._draw_JD_cloud`), with the two
disclosed mismatches documented there. Four cells on `three_gene_qvar/sample_0001`, the
claim-5 c2_P invocation (nc1, batched CUDA, cubic dispersion, 64 restarts, 400 Adam
steps), 8 seeds per cell, σ_w ∈ {0 (control), 0.048, 0.10, 0.20}.

## Measured curve

Runs: `experiments/claim_wnoise/sigma_w_{0p00,0p048,0p10,0p20}/` (branch
`feature/paper-weight-noise`). Aggregates: `experiments/claim_wnoise/summary_table.json`;
seed level: `seed_level.csv`; figure: `analyze.py` → `figures/wnoise_curves.png`.

| σ_w | Turing-unstable | patterns (rollout) | vol_20pct median | k*_fft rel-err median |
|---|---|---|---|---|
| 0.0 | **8/8** | 7/8 | 0.935 | 0.031 |
| 0.048 | 6/8 | 5/8 | 0.992 † | 0.059 |
| 0.10 | 1/8 | 0/8 | 0.880 (1 seed) | 0.621 |
| 0.20 | 0/8 | 0/8 | n/a — NaN by construction | 0.977 (grid-extreme = failed fit) |

† survivors only (6 seeds) — see the survivor-bias caveat below.

Wording discipline: "Turing-unstable" and "patterns" are different claims and both
columns are shown. Every seed at every level produces *a* model; what noise destroys is
Turing instability of the recovered model, then patterning — at σ_w = 0.10 even the one
Turing-unstable survivor does not pattern in the rollout.

## What IS supportable

1. **Recovery reliability degrades monotonically with train-time weight noise** on this
   target: 8/8 → 6/8 → 1/8 → 0/8 Turing-unstable across σ_w = 0 → 0.048 → 0.10 → 0.20,
   with k* error rising in step (0.031 → 0.977 median) and rollout patterning gone by
   σ_w = 0.10. Under the smoothed-objective estimator at these levels, the noise term
   dominates the Turing/k* objective rather than regularising it.
2. **A weak, confounded hint at σ_w = 0.048**: the six surviving seeds' 20%-perturbation
   Turing volume (median 0.992) sits above the control's all-seed median (0.935). This is
   survivor-biased (the two failed seeds are excluded from the survivors' column by
   construction) and is bought at ~2× the k* error and 2/8 outright failures. It is a
   hypothesis for a finer sweep at σ_w < 0.048, not a claim.

## What is NOT supportable

- "Training is more robust with weight noise" — contradicted at every probed level.
- "The learned GRN patterns more robustly" — no noise level improves patterning; 0.048
  loses patterning on 2 seeds the control had (7/8 → 5/8), and ≥0.10 loses all of it.
- Any statement about levels below 0.048, other targets, other forms, or the
  per-restart-held (randomised-prior) noise variant — none were run.

## Caveats

- One target, 8 seeds per level; probe levels UNCALIBRATED by design (D-WNOISE-1).
- The vol_20pct comparison at 0.048 is survivor-biased (stated above and in the README).
- The control row is consistent with the claim-4/claim-5 baseline for this config
  (8/8 Turing-unstable, vol at ceiling for ≤10%, median 0.935 at 20%), so the collapse is
  attributable to the noise, not to a broken harness — the σ_w=0 path is the untouched
  default code path and the knob's identity at σ=0 is bit-exact (tested,
  `tests/test_weight_noise.py`).
- Wall-time per cell collapsed from ~43 min (control) to ~3 min at σ_w ≥ 0.048 —
  consistent with noise-induced steady-state failures killing batch members early
  (documented kill-on-first-failure contract). Corroborates the mechanism of failure;
  reported so the fast runs are not mistaken for healthy ones.

## If the paper needs a positive robustness lever

The measured, supportable robustness levers in this pack remain: the recovered models'
own perturbation robustness (claim 4: at ceiling to 10%, median 0.935 at 20% for this
config, population baseline 0.755) and the NC-1 vs competitive comparison
(`docs/PAPER_CLAIM_FORMCOMP.md`: nc1 0.935 vs competitive 0.8325 at 20%, shared-tuning
confound disclosed). A finer weight-noise sweep below σ_w = 0.048, or the per-restart
held-noise variant, are the two follow-ups D-WNOISE-1 already names; neither has been run.
