# Weight noise for robustness — exemplar deck and the pipeline that would earn it

**Status: EXEMPLAR / ILLUSTRATIVE. No RNGRN was trained for any figure in this deck.**

This document accompanies six figures built to answer one question for a supervisor meeting:
*if adding noise to the weights during training really does make recovery more robust, what
would the evidence look like?* It is a visual aid and a specification, not a result. The
distinction is enforced on the figures themselves — every synthetic panel carries an
on-face `ILLUSTRATIVE` tag, and panels drawn from real measured data say so.

Read §1 to know what is real. Read §4 before quoting any number.

---

## 1. What in this deck is real, and what is not

| Panel | Provenance |
|---|---|
| Fig 1a, 1b | **Real measured data.** `exp11_robustness_baseline.csv` — local Turing volume of all 127 `three_gene` answer keys, 400 cloud draws each. |
| Fig 6c, leftmost group | **Real measured data.** Same source, at 20 % noise. |
| Fig 1c | Schematic. Hand-drawn illustration of the sharp/wide minimum idea. No data. |
| Fig 1d, 1e; Figs 2–5; Fig 6a, 6b | **Surrogate.** Computed on the 34 real `three_gene_qvar` answer-key Jacobians, but by the smoothed-objective proxy of §2 — *not* by training an RNGRN. |

The surrogate is a real computation on real Jacobians with a real optimiser and a real
robustness measurement. What it is *not* is evidence about the RNGRN, because it never runs
the RNGRN's gated-Hill parameterisation, its Newton steady-state solve, or its image loss.
It shows the shape a positive result would take, and — more usefully — it shows which
controls the claim needs in order to survive scrutiny.

---

## 2. The mechanism, stated so it can be attacked

Weight noise during training is not a regulariser bolted onto this problem; it acts on
exactly the quantity the project's end goal names.

**Why noise selects wide minima.** Training with weight noise makes the optimiser descend
the *noise-smoothed* objective

```
L̃_s(θ) = E_{ε ~ N(0, s²I)} [ L(θ + ε) ]
```

`L̃_s` is `L` blurred at scale `s`. A minimum narrower than `s` is filled in by the blur; a
minimum wider than `s` survives it. So the optimiser is pushed toward parameter points that
remain good *in a neighbourhood*, not merely at a point.

**Why that is the same thing as robustness here.** Tica's robustness quantity — the one this
project must beat — is the *local Turing volume*: perturb the parameters by a relative
uncertainty, ask what fraction of the cloud is still Turing. That is a statement about a
neighbourhood of a parameter point. "Wide minimum of the recovery loss" and "large local
Turing volume" are therefore the same geometric object, approached from two directions. This
is the whole reason the idea is worth testing, and it is the sentence to put in front of a
supervisor first.

**Why the selection effect is the entire story.** The recovery objective's zero set is a
*manifold*, not a point. In the surrogate there are 10 free parameters (nonzero |J| entries
plus D, signs frozen) and 3 constraints (k\*-anchor, uniform stability, instability at
k > 0). Both the noisy and the deterministic arm reach loss ≈ 0. They differ only in *which*
point of that manifold they land on. Fig 1d shows this directly: along random directions
from the recovered point, the deterministic solution stops being Turing after ‖Δθ‖ ≈ 0.4,
while the noise-trained solution holds out to 0.8+.

**How the surrogate realises it.** The antithetic evolution-strategies estimator

```
∇L̃_s ≈ (1 / 2ms²) Σ_j ε_j [ L(θ + ε_j) − L(θ − ε_j) ]
```

*is* the gradient of the smoothed objective, so running it at probe scale `s = σ_train`
reproduces the selection effect of weight noise without the torch recovery loop. The
`σ_train = 0` arm uses a tiny probe (10⁻³), making it an ordinary finite-difference gradient
of the unsmoothed objective — the deterministic control.

**Where the surrogate is honest about being a proxy.** Real weight noise perturbs θ in the
RNGRN's own coordinates, where every physical parameter is a nonlinear transform (softplus,
sigmoid, exp) of θ; the surrogate perturbs log|J| and log D. The consequence is documented
in `ROBUSTNESS_MEASUREMENT.md` §3.1: a single σ in θ-space delivers a ~14× spread of
effective physical noise. The surrogate sidesteps that defect rather than modelling it, and
so slightly *flatters* the mechanism.

---

## 3. What the deck claims, figure by figure

1. **Fig 1 — the headroom is real, the selection is the hypothesis.** Among the 127
   generator systems, robustness at 10 % noise spans up to 4.3× *within a single topology*.
   Each topology has exactly one interaction matrix, so that spread is entirely
   parameter-driven. There is something real for an optimiser to find; the question is only
   whether noise finds it.
2. **Fig 2 — the headline.** Survival-under-perturbation curves separate by training-noise
   level; the paired per-system comparison improves on 33/33 systems; more seeds reach
   Turing at all.
3. **Fig 3 — the cost.** The same noise that widens the basin blurs the wavelength. This is
   the panel that makes the deck credible rather than promotional, and the reason an
   operating point has to be *chosen* rather than maximised.
4. **Fig 4 — the controls.** Post-hoc jitter and wide-initialisation both fail to reproduce
   the effect. Without these two arms the result is indistinguishable from "stochasticity
   helps", and a supervisor will ask.
5. **Fig 5 — the seed distribution.** Reported as a distribution, never a best seed, per
   house convention. It also shows an honest regression: seed-to-seed spread *widens*.
6. **Fig 6 — where the gain comes from, and the bar it must clear.** Fragile systems gain
   most (partly a ceiling artefact, flagged on the panel), and the reference to beat is the
   training data's own robustness.

### The numbers as the surrogate produced them

34 `three_gene_qvar` systems × 12 seeds per arm, 400 cloud draws per evaluation, strict
Turing criterion.

| σ_train | seeds reaching Turing | still Turing @4.8 % | @10 % | @20 % | k\* err vs FFT |
|---|---|---|---|---|---|
| 0 (deterministic) | 0.772 | 1.000 | 0.963 | 0.788 | 3.49 % |
| 5 % | 0.802 | 1.000 | 0.978 | 0.810 | 3.97 % |
| 10 % | 0.848 | 1.000 | 0.995 | 0.888 | 3.97 % |
| **20 %** | **0.851** | **1.000** | **1.000** | **0.970** | **5.71 %** |
| 35 % | 0.863 | 1.000 | 1.000 | 0.990 | 7.38 % |

Paired over systems at 20 % evaluation noise: median gain **+0.160**, improved on **33 of
33** systems, Wilcoxon signed-rank **p = 2.3 × 10⁻¹⁰**.

**Matched-subset k\* cost** (only seeds Turing in *both* arms, so the comparison is
like-for-like): 3.17 % → 4.06 % at σ_train = 0.20, and 6.95 % at 0.35. The pooled table
above overstates the cost at 0.20 because the noise arms convert previously-failing seeds
into Turing ones, changing the population. Both framings are reported; the matched one is
the one to quote.

**Controls**, both at the same 20 % scale as the claim:

| arm | seeds Turing | still Turing @20 % | k\* err |
|---|---|---|---|
| deterministic | 0.772 | 0.788 | 3.5 % |
| **20 % weight noise** | **0.851** | **0.970** | **5.7 %** |
| post-hoc jitter | 0.615 | 0.825 | 12.7 % |
| wide init, no noise | 0.402 | 0.759 | 45.1 % |

Neither control reproduces the effect, and both pay heavily in wavelength accuracy.

---

## 4. Caveats a reader must carry

1. **No RNGRN was trained.** Nothing here is evidence about the pipeline's behaviour.
2. **The surrogate perturbs log(J, D), not θ.** It therefore does not inherit the θ-space
   defects of `robustness_cloud` — and does not model them either. Real weight noise will be
   messier and less symmetric.
3. **`(J, D)` perturbation is a looser quantity than Tica's.** They perturb kinetic
   parameters and re-derive `x*` and `J`; this perturbs the linearisation directly, with no
   steady-state solve to fail. It is a clean upper reference, not a like-for-like
   comparison. `ROBUSTNESS_MEASUREMENT.md` §4 states the same caveat for exp11.
4. **The `σ_train = 0.20` optimum is an artefact of this surrogate's geometry.** It should
   not be treated as a hyperparameter recommendation for the real model.
5. **Fig 6b's ρ = −0.95 is partly structural**, since gain is bounded above by
   (1 − baseline). The ceiling is drawn on the panel.
6. **Stripes n = 6 systems.** Per-morphology statements about stripes are weak everywhere in
   this project, and this deck does not fix that.
7. **Turing-I vs Turing–Hopf was classified but is nearly constant** (3 Hopf-leading points
   out of 1,684). It stays instrumented because Tica report the distinction, not because it
   discriminates here.

**A bug worth carrying forward.** While packaging the figure module, the first version keyed
the exp11 baseline table on `key` alone. `sample_0000` recurs in each of `three_gene_train`,
`_val` and `_test`, so `pivot_table(index="key")` silently collapsed **127 systems into 88**
and the panel's own label said 127. It is now keyed on `(ds, key)`. Anything aggregating
across these split-partitioned CSVs must do the same — the sample keys are unique only
within a split, and the failure is silent.

---

## 5. The pipeline that would earn this result

What follows is the specification the deck exists to motivate. Six units; each names what it
computes and where it wires into the existing tree.

### Unit 1 — a physically meaningful noise injection

Add weight noise to `recover()` as an explicit training-time perturbation, applied to
constrained/physical parameters or in log-physical space — **not** to raw θ. This is not a
new opinion; it is the fix `ROBUSTNESS_MEASUREMENT.md` §3.1–3.2 already specifies for
`robustness_cloud`, and the same argument applies to injection.

Knobs (all belong in `TUNING.md`): `noise_sigma`, `noise_targets` (which parameter families),
`noise_schedule` (constant | ramped | annealed), `noise_applies_to` (forward only, or
forward+backward). Default `noise_sigma = 0`, so the arm is opt-in and every existing run
stays comparable.

**Decision required before coding, not after:** whether noise is resampled per step or held
per restart. Per-step gives the smoothed-objective behaviour this deck models; per-restart
gives a randomised-prior ensemble instead. These are different experiments.

### Unit 2 — a robustness metric that is comparable to something

Replace `eval/analysis.robustness_cloud` with a corrected implementation:

- perturb physical parameters (or log-physical), sign-symmetric;
- **strict** criterion — `max Re eig(J) < 0` *and* an unstable mode at k > 0 — with the loose
  `tr(J) < 0` recorded alongside for continuity;
- classify the leading mode at k\* as Turing-I / Turing–Hopf;
- count steady-state solve failures as a **separate** outcome, never silently as
  "not Turing" (§5.3 of `ROBUSTNESS_MEASUREMENT.md`);
- batch the cloud and use `dispersion_backend='cubic'` on GPU — a 400-draw × 251-k-point
  cloud is 100,400 matrices, well past the ~6,400 crossover.

Emit `rob_frac_{sigma}`, `kcv_{sigma}`, `n_solve_fail`, `mode_class` per run, and **promote
them to the run index** — today `cmd_analyze` prints JSON to stdout and returns, so nothing
reaches the index.

### Unit 3 — the arm structure, fixed in advance

Minimum defensible design, all on `three_gene_qvar` (primary per `PREREGISTRATION.md` §1;
the legacy splits cannot support a k\* claim because their L is derived from k\*):

| arm | purpose |
|---|---|
| `noise_0` | deterministic reference |
| `noise_{σ}` for σ ∈ {0.05, 0.10, 0.20} | the dose–response |
| `posthoc_jitter_{σ}` | rules out "any stochasticity would do" |
| `wide_init_{σ}` | rules out "it is just more exploration" |
| `noise_eval_only` | robustness measured but never trained against — guards against the metric being gamed |

≥12 seeds per (system, arm); report the distribution, never the best seed.

### Unit 4 — pass conditions, written before the runs

Following `PREREGISTRATION.md`'s standing rule that thresholds are fixed in advance and
revised afterward by nobody:

1. **Primary.** Median local Turing volume at 4.8 % (Tica's measured experimental CV),
   paired per system, noise arm > deterministic arm, Wilcoxon signed-rank p < 0.01 with the
   effect size reported.
2. **Guard.** Median morphology-class agreement must not fall, and matched-subset
   `kstar_fft_rel_err` must not worsen by more than a stated tolerance. This is what makes
   the claim "more robust *at equal quality*" rather than "more robust".
3. **Controls.** Neither control arm may reach the noise arm's primary metric.
4. **Anti-degeneracy.** Reject any run whose D-ratio or σ_max at k\* collapses toward the
   flat-dispersion minimum documented as the known failure mode (`mem`: the k\*-anchor's
   degenerate minimum, where the loss is reduced by flattening σ(k) rather than relocating
   its peak). Fig 4c is this guard in figure form.
5. **Reference.** State the result against the generator systems' own robustness (median
   0.755 at 20 %, 0.935 at 10 %), not only against the deterministic arm.

### Unit 5 — confounds that must be ruled out explicitly

- **Noise as an early-stopping proxy.** Match total optimiser steps and report the loss
  actually reached; a noisy arm that simply stopped earlier is not a robustness result.
- **Population shift.** Any accuracy comparison restricted to seeds Turing in *both* arms —
  the matched-subset discipline of §3. Reporting only the pooled number is how the k\* cost
  gets hidden.
- **Metric gaming.** The `noise_eval_only` arm exists because a robustness metric optimised
  against is no longer a measurement.
- **Selection along the k\* ceiling.** Fig 6b's ceiling artefact generalises: gains
  correlate with headroom by construction.

### Unit 6 — figures regenerate from real runs

`exp13_figures.draw_all(df, dfine, dctrl, systems, out_dir)` takes exactly the three
dataframes the pipeline would emit. Point it at real run outputs with the same column names
and the deck redraws unchanged — with the `ILLUSTRATIVE` tags removed, which is the one edit
that should require a real result to make.

---

## 6. Reproducing this deck

```bash
# in a worktree, with datasets symlinked
.venv/bin/python scripts/exp13_noise_robustness_mockup.py --seeds 12 --ndraw 400
# ~10 min single-core; writes experiments/exp13_*.csv and figures/exp13/*.png
```

Deterministic across processes: all seeds derive from `blake2b` over repr-joined parts, never
`hash()`, which Python salts per process. The script validates the closed-form cubic against
`np.linalg.eigvals` on all 34 real Jacobians at startup and **refuses to continue** if the
equivalence fails (MAE 6.8 × 10⁻¹³, 0/34 k\* flips, 0/34 verdict flips at the time of
writing) — the same discipline that caught two earlier wrong implementations of that formula.

The script reads answer keys directly and deliberately: like `exp11`, it is a
characterisation tool on the scoring side of the firewall, and must never be imported by
`model.py`, `losses/`, or `recover.py`.
