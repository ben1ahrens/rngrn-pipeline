# State of the science — measured results, dead ends, and superseded claims

**Status: information, not instruction.** This is the evidence ledger. Every entry is
something that was measured, with the configuration it was measured under. Several
entries exist specifically to record that an earlier, more optimistic number was wrong
— those are marked **SUPERSEDED** and the corrected reading is given.

Branch `feature/turing-training`, HEAD `d8070ca`. Experiment scripts in `scripts/`,
raw results in `experiments/*.json` (gitignored tree, so the JSONs are the record).

---

## 1. The one rule: the firewall

No analytic quantity of a ground-truth system — x\*, J, D, σ(k), k\*, or any generating
coefficient — may reach recovery, including initialisation. The only truth-derived
inputs are `(frame, L, observed_idx)`. Everything else lives in `gate.AnswerKey` and is
consumed only by `validate.score_recovery`.

**What the guard actually enforces** (audited; `docs/DATA_INTO_MODEL.md` §5.1):
`tests/test_firewall.py` is a *substring match over import lines* against five names —
`rd_models`, `data.solver`, `data.cache`, `AnswerKey`, `answer_key`. Empirically:
`from .data.gate import AnswerKey` is caught; `from .data import gate` then
`gate.from_registry(...)` is **not**; `from .data.gate import from_registry` is **not**;
`from .data import registry` is **not**; `import h5py` and reading `payload.h5`
directly is **not**. The tree is currently clean — verified three ways — but the guard
would not stop a new loss term that reached for truth.

**The real exposure is not an import.** `L = clip(6·2π/k*, 18, 220)` in every
generator, so **k\*_true ≡ 6·2π/L to within 1e-6 for 94.8 % of all 287 registered
samples**. `L` is a legal recovery input; the loss anchors to `kstar_obs = FFT(image, L)`,
which tracks truth at r = +0.920. An image-blind predictor using `L` alone scores
0.000 % median error on `kstar_rel_err`. Changing the metric does not fix this — and
note that switching the headline target to `k_star_fft` (§8.1) does **not** fix it
either, since `kstar_obs` is itself computed from the image *and* `L`. Only regenerating
with `L` decorrelated from k\* does.

**Firewall rule for any new prior:** its constants must come from outside the datasets —
published values, or a stated modelling assumption. Centring a D-ratio prior on the
measured median true D-ratio, or sweeping the prior centre and keeping the value that
minimises `kstar_rel_err` on samples you then report against, routes `AnswerKey.D` into
recovery through judgement rather than through an import. The static test cannot see
either.

---

## 2. Recovery: what works and where it stops

### 2.1 The recipe that currently performs best

Assembled inline in `scripts/exp05_pixel_minibatch.py::fit` (the library
`losses/total.py` path is *not* what the experiments run — see §2.6):

```
loss = kstar_anchor(m, x*, kgrid, kstar_obs)                    # data term
     + anchor_w * mean((log(obs_scale) - log(x*))**2)           # frame-scale anchor, anchor_w=2.0
     + w(step) * split_hinges(m, x*, kgrid)                     # Turing hinges, disjoint support
```

with `w(step) = 0` for the first 25 % of steps, then ramped linearly to 1.0 over the
next 25 % — Endres-style **data-first** staging. `stationarity_residual` is **off**.

Two independent fixes were required, each addressing a different failure (measured,
40 seeds × 400 steps, N=3):

| configuration | converged | Turing |
|---|---|---|
| original hinges, no anchor | 2/40 | 0 % |
| original hinges + frame-scale anchor | 37/40 | 0 % |
| split hinges, no anchor | 2/40 | 0 % |
| **split hinges + frame-scale anchor** | **38/40** | **36.8 %** (14 Turing, 11 self-activating) |

The frame-scale anchor fixes *convergence*; the split-support objective converts
convergence into *instability*. Neither alone does anything.

### 2.2 Why the original hinges were self-defeating

`losses/terms.turing_hinges` computes `L_unstable` from `sig.max()` and `L_uniform`
from `sig[0]`. At 60/60 random inits, `argmax σ(k)` **is** the k≈0 grid point — so the
same scalar receives opposing gradients. The only escape, an interior peak, is punished
on approach. Fix: give the two conditions disjoint support —
`L_uniform` from k=0, `L_unstable` from `max` over `k ≥ k_min` with
`i_min = max(1, int(0.1·len(kgrid)))`. Validated as `turing_hinges_split` in
`scripts/exp02_objective_fix.py`; **not yet promoted into `losses/terms.py`**.

### 2.3 Why the frame-scale anchor is needed

Gradient ascent on `diag(J)` raises α, which raises x\*, which drives Hill occupancy
toward 1 where the curve is flat and its derivative vanishes; saturation `KA·x^n` runs
to 1e2–1e5 when self-activation requires `1 + KA·x^n < n`. The anchor pins x\* at the
observed mean intensity and thus in the sub-saturated regime. Self-activation goes
0 % → 96 % over 60 seeds. `obs_scale = frame.mean()` is an image observable, so the
anchor is firewall-legal. `anchor_w = 2.0` works and has not been swept.

### 2.4 THE OPEN PROBLEM — generalisation failure (exp09)

6 `three_gene_val` samples, 8 seeds each, 4000 steps:

| sample | k\*_true | k\*_obs (FFT) | anchor err | Turing seeds | k\* err |
|---|---|---|---|---|---|
| sample_0000 | 0.4121 | 0.4320 | 4.8 % | 7/8 | 3.4 % |
| sample_0001 | 0.3419 | 0.2202 | 35.6 % | **0/8** | — |
| sample_0002 | 0.4020 | 0.4132 | 2.8 % | 7/8 | 2.9 % |
| sample_0003 | 0.7529 | 0.6900 | 8.4 % | **0/8** | — |
| sample_0004 | 0.3619 | 0.2766 | 23.6 % | **0/8** | — |
| sample_0005 | 0.4622 | 0.4583 | 0.8 % | **0/8** | — |

**2 of 6 samples, 0 of 8 seeds on the other four.** Earlier single-sample results do
not generalise. Anchor quality correlates with success (corr = −0.494, n=6) but is
**not sufficient** — sample_0005 has the *best* anchor in the set (0.8 %) and still
fails 0/8.

**Root cause, traced not assumed.** The k\*-anchor has a degenerate minimum. While the
model is non-Turing, σ(k) peaks at k=0, so
`L = logsumexp(temp·σ)/temp − σ(k_obs)` can be reduced by **flattening σ** — killing
D separation and ‖J‖ — rather than by relocating its peak. Trace on sample_0005:
`kstar_model = 0.0000` from step 0; anchor loss frozen at 0.0692 for 1000 steps; the
Newton steady-state solve dies at step 1123, exactly when the hinges ramp in at 25 % of
budget. **The Newton death is a symptom, not the cause.**

**Two fixes tested and rejected:**
- Warm-starting the Newton solve from the previous step's x\* (continuation): no effect,
  still 0/6 on all three failing samples.
- `terms.anticollapse`, which exists precisely to exclude this flat-dispersion minimum
  and had never been enabled: rescues nothing, and *harms* the working sample
  (6/6 → 2/6 at weight 2.0).

### 2.5 Statistics corrections that matter

- **SUPERSEDED — the "0.30 % ensemble" result.** exp08 reported trivial-baseline
  6·2π/L 0.00 % / fft_only 6.60 % / recovery 0.30 % / shuffled-L 52.38 % on 4 samples ×
  4 seeds, described at the time as a breakthrough. At 6 samples the per-seed median
  (3.1 %) and the ensemble median (3.14 %) are **identical**. The 0.30 % was a
  two-sample artefact; seed averaging does **not** cancel error.
- **What survives from exp08** — and it is the reason any k\* number is readable at all:
  recovery (3.1 %) beats its own FFT loss target (6.6 %), so it is not merely tracking
  `kstar_obs`; and given another sample's L it degrades to ~52 % rather than returning
  `6·2π/L_wrong`, so it is not inverting the L↔k\* identity either.
- **The FFT anchor has a hard resolution floor.** One FFT bin is 16.7 % of k\*, giving
  `k_star_fft` a ±8.3 % offset from the analytic value; observed fft-vs-linear deviations
  (median 8.4 %, max 25.1 %) are steps of ±0.5 and ±1.5 bins, not physical disagreement.
  §8.3 shows this offset is **deterministic, not random** — every sample's `k_star_fft`
  sits within 0.011 of a half-integer bin while the analytic k\* sits on bin 6.000. The
  loss tolerance `tau = 0.12` is only 0.72 of one FFT bin, *below the estimator's own
  resolution*. The exact 6.00-bin identity holds for the `three_gene` family only; the
  classical sets break it where the `clip(18, 220)` bound fired.

### 2.6 Library path vs experiment path — they are not the same objective

`recover.recover` (library) and `scripts/exp0*.py::fit` (what actually produced every
result above) are different code with different objectives. Library defaults:
`n_restarts=4, adam_steps=1500, adam_lr=0.05, lbfgs_steps=50, tau=0.12, jac_floor=1.0,
grad_clip=10.0, form='competitive'`, weights
`{kstar:1.0, turing:1.0, resid:0.3, anticollapse:0.5, morphology:0.0}` — i.e. the
library still uses the **original** (self-defeating) hinges, has `resid` **on** at 0.3,
`anticollapse` **on** at 0.5, and no frame-scale anchor. None of those defaults reflect
what was learned. Nothing is persisted mid-run, so an interrupted restart loses the
whole 4×1550-step budget.

---

### 2.7 The Endres / Matas-Gil method this recipe is derived from

The data-first staging in §2.1 is borrowed, and the source is worth knowing in detail
because two of its details were misread earlier in the project.

Matas-Gil A & Endres RG, *Unraveling biochemical spatial patterns: machine learning
approaches to the inverse problem of Turing patterns*.
[arXiv:2309.06339](https://arxiv.org/abs/2309.06339) (12 Sep 2023); published as
iScience, [10.1016/j.isci.2024.109822](https://doi.org/10.1016/j.isci.2024.109822).
Author list and DOI verified against the arXiv API record; the values below were
extracted from the source in an earlier session.

**Their data generation.** 50×50 discrete grid, **zero-flux (no-flux)** boundaries,
centre-difference Laplacian turning the PDE into an ODE system, forward-integrated to
steady state from **random** initial conditions. Models: Schnakenberg, FitzHugh–Nagumo,
Brusselator. They deliberately **fix the initial conditions across patterns** so
comparisons are like-for-like. Pattern *type* and *wavelength* are set by the model;
*position* and *shape* by the initial conditions — which is why they compare in the
**frequency domain** (RAPS), never in pixel space.

**Their training loop.** Batches of **128 elements**, **200,000 Adam iterations**, under
an hour of compute. The first **10,000–20,000 iterations minimise ONLY the approximation
(data) loss**; then both losses, with weights chosen so the approximation is not
destroyed by the new term. Reported errors <10 % on inferred parameters.

**CRITICAL READING CORRECTION — what "batches of 128" means.** They are **128 spatial
pixel locations drawn from the ONE image** — not 128 images and not 128 samples. From the
source: *"When training the approximation, we use information on the whole pattern, but
when we switch to the LPDE, we only use interior points"*; their Fig. 3 shows N randomly
selected points on a single Turing pattern, and a **3×3 pixel region is sufficient** to
recover parameters, with relative error falling as a power law of exponent ≈ −1/2. An
earlier reading in this project concluded no minibatch dimension existed; that was wrong.
This is what `scripts/exp05_pixel_minibatch.py` implements and tests.

**So the "200,000 iterations from a single image" figure is real, and it is the reason
this project ever tested long budgets** — but note §3: our own hit rate **saturates at
4000 steps**, so their budget does not transfer. Two structural reasons it need not:
they observe **all** species and assume the candidate model **is** the true model, so
their PDE-residual loss can reach zero; neither holds here.

**Two more transferable points.**

1. They hit the same **unknown intensity scale** problem (experimental image pixel values
   23–255) and solved it with a **free rescaling variable** — functionally the same device
   as our frame-scale anchor (§2.3, and §2.8 for what that scale actually encodes).
2. They treat the method's **stochasticity as a feature**, providing multiple parameter
   alternatives. That matches the "engine for biologically plausible values" framing
   directly, and it is why results here are reported as a **seed distribution**, never
   the best seed.

**Differences from our setup, to keep in view:** 50×50 zero-flux vs our 96×96 periodic;
all species observed vs our m ≤ N; candidate model == true model vs our gated-Hill RNGRN
which *cannot* represent the generators' kinetics.

### 2.8 Does the power spectrum's amplitude encode the concentration field?

**Yes, exactly — and the pipeline throws it away on purpose.** Measured by
`scripts/exp12_spectral_amplitude.py`; three findings.

**(1) The amplitude IS the concentration field, by Parseval.** For an (N,N) field,
verified to ≤2.4e-7 on all 127 `three_gene` samples (channel 0):

```
|F[0,0]|² / N⁴        ==  mean(field)²      the DC bin IS the mean concentration
Σ|F_sub|² / N⁴        ==  var(field)        integrated mean-subtracted power IS the variance
```

where `F_sub` is the transform of `field − field.mean()`. RAPS peak power scales exactly
as amplitude² (a 3× field rescale gives 9.00× peak power).

**(2) Every scored spectral quantity is amplitude-blind by construction.** Both
`observables.raps()` and `scoring/morphology._spectral_block()` subtract the mean —
annihilating the DC bin — and `_spectral_block` then divides by the block sum. Probe:
rescale a real field by 3× and

| quantity | self-distance under a 3× rescale |
|---|---|
| `spectral_distance_2d` | 2.2e-11 |
| `morphology_distance` | 3.8e-9 |
| k\*, skew, kurtosis | bit-identical |

That invariance is a **feature**, not an oversight: it is why morphology comparison
survives the unknown fluorescence gain of an experimental image — the same problem
Endres/Matas-Gil solve with a free rescaling variable (§2.7).

**(3) But the discarded channel is informative, and one loss term already exploits it.**
Spearman ρ against true dynamical quantities, n = 127:

| discarded channel | vs true x\*₀ | vs σ_max | vs k\* |
|---|---|---|---|
| `frame.mean()` | **+0.950** | −0.117 | −0.134 |
| `frame.var()` | +0.881 | +0.097 | −0.154 |
| CV = std/mean | +0.070 | +0.357 | −0.048 |

`frame.mean()` tracks the true homogeneous steady state x\*₀ at ρ = 0.95, with ratio
median 0.921 (IQR 0.796–1.036) — i.e. an ~8 %-biased estimator of x\*₀ available from the
image alone. **That is precisely what the frame-scale anchor is**: `anchor_w ·
mean((log(obs_scale) − log(x*))²)` with `obs_scale = frame.mean()` is a firewall-legal
estimator of x\*₀ read off the DC amplitude, and it is the term that took convergence
from 2/40 to 37/40 (§2.1).

Two consequences worth carrying:

- The amplitude channels carry almost nothing about k\* (|ρ| ≤ 0.16), which justifies
  normalising them away for **spatial-mode** scoring while using them for **scale**
  anchoring. The pipeline is, correctly, doing both.
- If an experimental image ever replaces a simulated frame, `frame.mean()` is in
  arbitrary fluorescence units and this estimator breaks. Endres' free-rescaling device
  is the known remedy; the anchor as written assumes the frame is in concentration units.

## 3. Rejected approaches — do not re-attempt without new information

Each of these was measured, not assumed. They are recorded so the same ground is not
covered twice.

**`stationarity_residual` — settled OFF.** Swept pixel batch {64,128,512} × weight
{1,3,10} (exp06, 8 seeds each): **all nine** configs collapse the Turing hit rate to
1/8 versus 5/10 without the term; best median k\* error 11.8 % vs 0.4 % without. Not a
weight-tuning problem. Structural reason: Endres *can* use a PDE-residual term because
they assume the candidate model **is** the true model; the gated-Hill RNGRN cannot
represent the generators' kinetics, so a zero residual is unreachable and the term
pulls toward a wrong model. Pixel minibatching genuinely *improved* the residual
(full-field 30.0 % → batch-128 23.9 %) — the batching idea was sound; the term is the
wrong constraint here.

**"Turing-first" homotopy — rejected.** Start from the low-basal prior that reaches the
Turing manifold 82 % of the time, then anneal from "stay Turing" toward "match k\*".
Convergence collapses to 1/16 versus 10/16 for the plain baseline, k\* rel err 99.8 %.
Reaching the Turing manifold and being able to *move along* it are different
properties: β ~ 1e-4 with steep Hill responses is a numerically stiff corner where the
Newton solve fails once the k\* term pulls. The low-basal init remains useful as a
*diagnostic*, not as a starting point.

**Shifted power iteration for the dispersion — wrong, not merely slow.** Matmul-only
and superficially ideal for GPU, but **49/127 Turing verdicts flipped** at 60
iterations. Measured cause: 11/200 k-points have a *complex* dominant eigenvalue, and a
real power iteration cannot converge inside a 2-D invariant subspace — it rotates. The
large Frobenius shift was a red herring (convergence ratio ~0.32, i.e. fast).

**Endres-scale iteration budgets — no longer motivated.** Turing hit rate on
`three_gene_val/sample_0000` over 16 seeds: 8/16 @400 steps, 13/16 @2000, 15/16 @4000,
15/16 @8000. It **saturates at 4000 steps** (~26 s/seed). k\* median is 2.7 % at 400
steps and 3.4 % at 2000/4000/8000 — flat across the three longer budgets and slightly
*worse* than the 400-step value, though the 400-step median is over only 8 Turing seeds
versus 13–15 at longer budgets, so the populations differ and the comparison is not
like-for-like. Consequence: there is no long-run regime to unlock, which removes the
motivation for a 200k-iteration budget.

**`anticollapse` — enabling it does not help.** See §2.4.

**Newton continuation (warm-start from previous x\*) — no effect.** See §2.4.

---

## 4. What is validated, and what is only repackaged

| module | status |
|---|---|
| `model.py`, `observables.py` | VALIDATED (autodiff J vs finite-diff to 1e-11; general-eig vs 2×2 closed form; KA+KR≡s to 1e-6) |
| `_sigma_max_cubic` (N=3 dispersion) | VALIDATED against `eigvals` on all 127 real answer-key Jacobians |
| `losses/terms.py`, `eval/*`, `validate.py`, `data/*` | repackaged from scaffold; wiring correct, **recovery science unproven** |
| `eval/analysis.turing_ok`, `robustness_cloud` | repackaged, **never validated**; `robustness_cloud` never run on a recovery |
| `eval/numerics.integrate_bdf1_newton_krylov` | **stub** — delegates to ETDRK4 |
| `losses/weighting.{GradNorm,NTK}Weighting.combine` | **stubs** — run with fixed weights |
| `losses/terms.morphology_consistency` | non-differentiable numpy diagnostic, **not in `compute_terms`**; the `morphology` weight is inert |

### 4.1 The exact cubic dispersion (the GPU unlock)

For N=3 the characteristic polynomial of `(J − k²D)` is a cubic whose coefficients
(trace, sum of principal 2×2 minors, determinant) are differentiable and whose roots
are closed-form — pure arithmetic, no eig/svd/solve. Validated on all 127 real
Jacobians: σ_max MAE **9.2e-13** (max 7.0e-11), k\* MAE **exactly 0**, Turing verdict
flips **0/127**, `dσ_max/dJ` agreeing with `eigvals` to 2e-16.

Speed: CPU 0.29 ms vs eigvals 0.18 ms (0.6×, slightly slower) but **CUDA 0.97 ms vs
156.61 ms (162× faster)**. So: eigvals on CPU, cubic on GPU.
`RNGRN(dispersion_backend='cubic')` exists; default stays `'eig'`. Exact for **N ≤ 3
only** — N=4's closed form is numerically poor, N≥5 has none.

**Sign trap, do not "simplify":** in the trigonometric branch the argument is
`(3q)/(2p)·sqrt(-3/p)` with `p < 0`. Using `|p|` instead gives a plausible-looking
function with MAE ~2e+2. Both this and the power-iteration failure were caught **only**
by the equivalence check against `eigvals` on real Jacobians.

### 4.2 Where the GPU earns its place

The axis is **parallel work**, not serial iterations. Measured cubic dispersion,
CPU vs CUDA, by batch of 3×3 matrices:

| batch | CPU (ms) | CUDA (ms) | speedup |
|---|---|---|---|
| 200 (1 seed) | 0.18 | 1.54 | 0.12× |
| 3,200 (16 seeds) | 0.98 | 1.26 | 0.78× |
| 12,800 (64 seeds) | 2.91 | 0.98 | 2.97× |
| 60,800 (16 seeds × 19 samples) | 14.09 | 1.00 | **14.1×** |
| 243,200 (64 × 19) | 78.30 | 2.57 | 30.5× |
| 1e6 | 359.45 | 9.42 | 38.1× |

**Crossover ~6,400 matrices.** A full val-split sweep is 60,800 ⇒ 14×. And a sweep is
the scientifically correct unit anyway, since results are reported as a seed
distribution.

**Bottleneck after wiring the cubic (commit d8070ca):** a single training step is
CPU 138.3 ms vs CUDA 417.6 ms — GPU still **3× slower per seed**, because
`terms.steady_state` is a damped Newton with up to 100 **sequential** 3×3 solves
(CPU 2.5 ms vs CUDA 8.8 ms), launch-bound exactly as `eigvals` was. The cubic alone
took CUDA from 632.6 → 417.6 ms/step. The remaining move is to batch the restarts —
give the model a leading batch dim over parameters and batch the Newton over B — which
is the same change that makes the whole sweep GPU-resident.

### 4.3 Device portability

`RNGRN` has `.device` / `.dtype` properties (from `theta_s`). Callers creating tensors
for the model **must** use `model.device` / `model.dtype`, or `.to('cuda')` breaks with
a cross-device RuntimeError. `recover()` takes `device=`.

---

## 5. Morphology scoring — measured accuracy

`src/rngrn/scoring/morphology.py`. Wired into `validate.score_recovery`.

| metric | accuracy |
|---|---|
| 4 observables (skew, kurtosis, ang_conc, ang_entropy), leave-one-out 1-NN on 127 `three_gene` | 92.1 % pooled / **81.8 % balanced** (majority baseline 47.2 %) |
| bank from train only, on 39 held-out val+test | 89.7 % pooled / **74.1 % balanced** — spots 94.4 %, labyrinth 94.4 %, **stripes 33.3 %** (only 3 held-out stripes) |
| 2D spectral block, n=24 | 83.5 % |
| 1D radially-averaged RAPS | 54.3 % (barely above the 47.2 % baseline — radial averaging destroys the anisotropy that defines stripes) |

`skew` alone carries 92.1 % pooled; the angular pair is what lifts stripes.
Z-scoring by `three_gene_train` SDs is used: unscaled Euclidean scores higher pooled
(94.5 %) but only by abandoning stripes (42.9 % vs 57.1 % recall).

**Ordering matters:** normalise-then-log1p gives 83.5 % and is exactly
intensity-invariant (3× rescale self-distance 6.6e-18); log1p-then-normalise gives
70.1 % and lets brightness leak in (1.0e-3 under the same rescale).

**Stripes is the weak class everywhere** — 7 of 127 samples, 4 of 88 in train. Balanced
accuracy is the honest number, and stripes is too weakly characterised to be a
pass/fail condition. **This is an open decision** (options put to the user: balanced
accuracy excluding stripes; the continuous `morphology_distance` instead of a class
match; or generate more stripes samples first).

Note the collision with the goal: Tica's pattern is **stripes** (concentric rings), the
class our morphology scorer is weakest on.

---

## 6. Rollout cost — the trap

`eval.rollout.simulate` at 96×96 ETDRK4 runs ~4.2 ms/step, and `nsteps` is derived from
the model's own σ_max (`T = horizon_growth_times / σ_max`) — **~128k steps for an
untrained N=3 model, i.e. ~9 minutes for one simulated field**, with a 200k-step clip
giving a ~14 min ceiling. An earlier brief assumed ~1.9 s per rollout: off by three
orders of magnitude.

Consequence: `fit()` scores the *target* frame's morphology for free (it is the image
recovery already trains on) but must never roll out on every run.
`validate.score_recovery` therefore takes `target_frame` and `model_frame` separately
with a three-state `morphology_scored` ∈ {`compared`, `target_only`, `not_scored`}.

---

## 7. Dataset inventory and its defects

Three families × 3 splits, **stratified by morphology only** — nothing stratifies on L
or k\*. 70/15/15 from a shuffled pool, rng seed 7.

| family | n | train/val/test | species | keys per sample |
|---|---|---|---|---|
| `three_gene` | 127 | 88/19/20 | 3 | D, final_frame, interaction_matrix, jacobian, times, trajectory, x_star |
| `three_gene_classical` | 85 | 58/11/16 | 3 | D, final_frame, jacobian, x_star |
| `two_gene_classical` | 75 | 52/10/13 | 2 | D, final_frame, jacobian, x_star |

Only `three_gene` carries `interaction_matrix` and a `trajectory` (6 timepoints).
Frames are `(N, 96, 96)`.

`three_gene` composition: 5 topologies, **one interaction matrix each** (verified — no
within-topology wiring variation), Hill n ∈ {2, 3, 4}, reaction `multiplicative` (118)
or `additive` (9):

| topology | n | interaction matrix (rows = target, cols = source) |
|---|---|---|
| `cross_repress` | 27 | `[[1,-1,-1],[1,0,0],[1,-1,0]]` |
| `selfinhib` | 28 | `[[1,-1,0],[1,-1,0],[1,0,-1]]` |
| `substrate_depl` | 28 | `[[1,-1,0],[1,0,-1],[0,1,0]]` |
| `double_inhibitor` | 26 | `[[1,-1,-1],[1,0,0],[1,0,0]]` |
| `relay_chain` | 18 | `[[1,-1,0],[0,0,1],[1,0,0]]` |

All 127 have exactly one positive diagonal entry (self-activation on node 0), 4–6 edges,
and `D[0] = 1` as the slowest species with the other two 10–250× faster.

Morphology is imbalanced throughout — `three_gene_train` is stripes 4 / spots 42 /
labyrinth 42. Val splits of 10–19 samples are thin for calibrating any threshold.

| family | D-ratio (min/median/max) | L (min/median/max) | k\* (min/median/max) | periods per box |
|---|---|---|---|---|
| `three_gene` | 16.6 / 134.9 / 249.3 | 40.0 / 98.7 / 207.7 | 0.181 / 0.382 / 0.943 | **exactly 6.000** |
| `three_gene_classical` | 12.3 / 86.5 / 197.1 | 18.0 / 68.2 / 149.8 | 0.252 / 0.552 / 2.898 | 6.0–8.30 |
| `two_gene_classical` | 10.0 / 44.5 / 191.3 | 18.0 / 44.7 / 129.2 | 0.292 / 0.843 / 2.808 | 6.0–8.04 |

### 7.1 The domain-size defect and the open decision

Every generator — 2-gene and 3-gene, character-for-character the same line — sets
`lam = 2π/k_star; L = clip(6·lam, 18, 220); dx = L/grid`. So **L is derived from
k\*_true** and each pattern holds exactly 6 wavelengths. See §1 for why this defeats the
firewall in spirit.

The user stated they "wanted all of the patterns for my training data to have the same
domain size", which the current data violates. Three options were measured and put to
the user, **still open**:

- **(a) fixed L** — matches the stated intent, makes L carry zero information, but caps
  k\* to a 5.3× span at grid=96 (L=100 gives k\* ∈ [0.19, 1.01]) versus the generators'
  0.08–3.0 screening band, and forecloses any domain-size generalisation claim.
- **(b) vary L but draw periods-per-box randomly** (e.g. 3–14) to decorrelate it from
  k\* — keeps the full band and enables splitting on L to test extrapolation.
- **(c) fixed L with grid raised to 192/256** — wider band at 4–7× simulation cost.

The user also asked whether the model should generalise across domain sizes, which
option (a) forecloses.

### 7.2 A related fixed bug worth knowing

`gate.from_registry` used to take L as a caller *parameter* while seven file-backed
configs hardcoded `L=100.0`, and read `attrs['kstar']` — a name no payload carries — so
`AnswerKey.kstar` was permanently `None` and `kstar_rel_err` permanently NaN. Since
`kstar_obs` scales as 1/L, **the loss target itself** was wrong by 0.72×–2.5× on 3-gene
data and up to 5.5× on 2-gene. Fixed in `d76378a`: the gate now reads each sample's own
`L`, `k_star` and `k_star_fft` attrs and raises if absent.

---

## 8. Scoring decisions

### 8.1 THE VALIDATION TARGET IS `k_star_fft` — reversed 2026-07-29

**Current decision (user, 2026-07-29): validation uses `k_star_fft`, the FFT estimate
from the image, NOT the analytic `k_star`.** This **reverses** the decision of
2026-07-26, which made analytic `k_star` the headline target and `k_star_fft` a
clearly-labelled secondary diagnostic. Both metrics are already computed and both already
reach the run index (`kstar_rel_err` and `kstar_fft_rel_err`), so the change is which one
is the headline — but every k\* number recorded before this date was reported against the
analytic target and is **not** directly comparable to numbers reported after it.

**The argument for the reversal** (and it is a good one): the analytic `k_star` is
`argmax_k σ(k)` from the *true J and D*. It is a property of the generating equations
that no experiment can observe. `k_star_fft` is a property of the *image*, which is what
a real inverse problem actually has. Grading against the analytic value measures
agreement with a latent quantity; grading against the FFT value measures agreement with
something measurable on a micrograph. For a pipeline whose end goal involves
experimentally obtained patterns, the observable target is the defensible one.

**The counter-argument, recorded because it does not disappear:** the loss already
anchors to an FFT estimate of k\*, so validating against an FFT estimate partly grades
the model on reproducing its own input. §8.2 measures exactly how much of a problem this
is, and the answer is "less than it sounds, but not zero".

### 8.2 Measured: `k_star_fft` is NOT the same number the loss anchors to

This had to be checked before the reversal could be assessed, because if
`k_star_fft == kstar_obs` then validation would be perfectly circular.

Measured on 128 samples across four splits, comparing the stored `k_star_fft` attr with
`observables.kstar_of(frame[0], L)` — the actual loss anchor:

| split | n | median \|fft − obs\|/fft | max |
|---|---|---|---|
| `three_gene_train` | 88 | 3.67 % | 39.3 % |
| `three_gene_val` | 19 | 3.13 % | 14.0 % |
| `three_gene_classical_val` | 11 | 2.62 % | 7.0 % |
| `two_gene_classical_val` | 10 | 1.56 % | 5.3 % |

**Exact matches: 0 of 128. Within 1 %: 22 of 128.** They correlate at r = 0.998 but they
are different estimators — the stored attr is rounded to 2–3 decimals and derived
differently from the live `raps()` power-weighted centroid. So validation against
`k_star_fft` is **not** literally circular; it is a comparison against an independently
computed FFT estimate of the same underlying quantity. The residual concern is
methodological rather than arithmetic: both estimators share the same *bias structure*
(§8.3), so a model that reproduces the bias is rewarded.

### 8.3 The ±8.3 % offset is deterministic, not random error

Worth knowing precisely, because it changes how the new target should be read.

Expressing each estimator in FFT bin units (bin width `2π/L`):

- the analytic `k_star` sits at **exactly 6.000 bins** (median, and `three_gene` is exact
  by the `L = 6λ` generator construction);
- the stored `k_star_fft` sits at a **half-integer** bin — every one of the 128 samples is
  within **0.011 bins of a half-integer** (median 0.004).

Half a bin out of six is 0.5/6 = **8.33 %**, which is exactly the offset measured between
the two: median signed deviation +8.2 % to +8.4 % on three of the four splits and −8.2 %
on `three_gene_val`. The sign splits 74 positive / 54 negative.

So the "±8.3 % quantisation floor" recorded earlier is not noise — it is a **systematic
half-bin offset whose direction varies by sample**. Consequences for the new target:

1. A model that reproduces `k_star_fft` to within ~8 % may be exactly reproducing the
   analytic k\*, or may be off by a full bin in the other direction. The metric cannot
   distinguish those without also reporting the analytic value.
2. `tau = 0.12` is 0.72 of one FFT bin — still **below the estimator's own resolution**.
   With `k_star_fft` now the headline, `tau` is the knob most in need of revisiting.
3. **Keep emitting `kstar_rel_err` against the analytic value as the secondary
   diagnostic** — the roles simply swap. Dropping it would make the half-bin ambiguity
   unresolvable.

### 8.4 Unchanged decisions

- **Parameter values are deliberately not scored** as a fitting target. See
  `GOAL_tica_equivalent.md` §3 — but note the carve-out in its §2.3: Experiment B's
  success criterion *is* a statement about recovered coupling magnitudes.
- **The spare-species inertness threshold** (`scoring/overparam.py`,
  `DEFAULT_COUPLING_THRESHOLD = 0.05`, relative) is an explicitly **uncalibrated**
  starting point. Calibrate from the distribution of `max_abs_coupling_*` in the
  `expB_control_matched` arm, where no spare species exists by construction. Metrics use
  the strongest *single* edge to/from the spare species, not a mean, because a mean
  hides one strong spurious edge behind many near-zero ones.
- The analytic `k_star` is reproducible from the stored `J` and `D` to a median 0.032 %
  via `eval/analysis.turing_ok` (verified on all 127), so it remains a trustworthy
  secondary reference.
- **Parameter values are deliberately not scored.** See `GOAL_tica_equivalent.md` §3.
- **The spare-species inertness threshold** (`scoring/overparam.py`,
  `DEFAULT_COUPLING_THRESHOLD = 0.05`, relative) is an explicitly **uncalibrated**
  starting point. Calibrate from the distribution of `max_abs_coupling_*` in the
  `expB_control_matched` arm, where no spare species exists by construction. Metrics use
  the strongest *single* edge to/from the spare species, not a mean, because a mean
  hides one strong spurious edge behind many near-zero ones.

---

## 9. Identifiability experiments — what they test and one degeneracy

Both feed the model 2 observed channels and ask for a 3×3 GRN, inferring a (1,H,W)
latent field for the unobserved species. They test **opposite** things:

- **A** (`expA_hidden_channel`, N=3 truth, one gene genuinely hidden): is the hidden
  gene recoverable? Success = the 3×3 correct up to relabelling. Control:
  `expA_control_full` (same data, m=3).
- **B** (`expB_overparam`, N=2 truth, model given N=3): does spare capacity invent a
  gene? Success = the spare species stays **inert** plus the observed 2×2 sub-block
  matches truth; a confident third gene is a false positive. Control:
  `expB_control_matched` (same data, correctly-sized N=2 model).

N=3 is an assumption imposed via config, never inferred — the firewall means the model
is never told the true species count. Recovery is per-frame variational fitting to one
image, not amortised training across the dataset.

**The degeneracy to know:** with N=3 and m=2 there is exactly **one** unobserved
species, so permutation-invariant scoring is a **no-op** — `n_permutations_searched == 1`,
aligned score == identity score, delta 0. Alignment only does work at ≥2 hidden species.
The earlier `sign_match_frac ≈ 0.44` was therefore never a permutation problem. The real
labelling freedom in this configuration is a **sign/scale gauge** on the single latent
field, so `latent_field_quality` using `|r|` is the honest identifiability read. A test
asserts the degeneracy so it can never be misreported as work done.

Degradation is always read against the matched control, never against zero.
`benchmark.degradation_table()` (via `rngrn benchmark --degradation`) groups by
`(arm, dataset, n_true, n_model)` with controls first. `subblock_sign_match_mean` is the
**only** column comparable across every arm; `sign_match_aligned_mean` applies only where
a same-size true J exists, and `spare_inert_frac` only to the over-parameterised arm.
NaN generally means not-applicable-by-construction, not failure.

---

## 10. Init distribution — solved but deliberately unadopted

Root cause of 0/300 Turing-unstable inits: **the RNGRN's Jacobian diagonal is always
negative at init.** `∂f_i/∂x_i = (production sensitivity) − δ_i`, and at default init
α≈0.82, KA≈0.14, δ≈0.82, so the saturating Hill term's derivative cannot exceed δ.
Measured: 0/200 converged inits have any positive diagonal (mean diag ≈ −0.63, max
seen −0.27), while 88/88 true `three_gene` systems do (range −10.8 to +3.1).

Turing patterning requires self-activation. Forcing the gate to full self-activation is
**not** enough (diagonal still negative); α sweeps 1–100 at default δ stay negative and
merely inflate x\*; D ratios 3–250 give 0 % on their own. The diagonal only goes positive
when δ drops to ~0.05–0.1, an order of magnitude below the init default. **So the fix is
the α/δ/KA balance, not the gate and not D separation.**

A firewall-safe init sampling low basal production (β ~ 1e-4 to 1e-2 vs the old
0.2–0.47), sub-saturated binding (KA 1e-2 to 10^-0.3), α 10^0.3–10^1.5, δ 1e-1–10^0.3,
gate ~ N(0, 2.5), D spread 10^0.9–10^2.4 takes Turing-unstable inits from **0 % to 82 %**.
Ladder by β upper bound (400 seeds each): 0.5 → 69.7 %; 0.05 → 80.3 %; 0.01 → 81.4 %;
0.001 → 82.0 %.

**Deliberately not adopted.** The init distribution biases which solutions are found,
and near-zero basal expression is a modelling assumption rather than a neutral prior.
It is a stop-and-confirm point.

**And the base rate is not a defect.** Scholes et al. report ~60 % of 2- and 3-node
topologies can pattern for *some* parameters while the patterning parameter volume is
<0.1 % (Tica's own linear-stability figure: 0.022 %). Random sampling therefore cannot
find the Turing regime — the *optimiser* must, which is what the 10/16 result
demonstrates. The same literature notes larger networks are more robust, so N=3 is an
easier regime than N=2.

---

## 11. The D-ratio prior — designed, unwired, and with an unresolved tension

The failing exp09 runs collapse the D-ratio toward ~8 while the generators sit at
108–140. That motivated a soft prior. It is **not implemented**.

**Literature values, and the tension they create.** Absolute morphogen diffusion shows a
local-to-global slowdown of nearly two orders of magnitude, from ~60 µm²/s (FCS, local)
to ~1–2 µm²/s (FRAP, global). The canonical measured *ratio* is Nodal/Lefty in live
zebrafish: the activator Nodal diffuses **~7.5× slower** than the inhibitor Lefty. That
is ~7.5 against the `three_gene` generators' 108–140 — a **~15× gap**.

Not a contradiction: the same literature reports that in extended networks Nodal and
Lefty need not have different diffusivities *at all* to pattern, and that differential
diffusivity instead contributes **robustness**. Larger ratios buy robustness while
biology gets away with less via richer topology. **This is exactly Tica's stated reason
for node C** — see `GOAL_tica_equivalent.md` §4.4 for what our own data says when a node
is made non-diffusible.

**Open design choice:** centre the prior on ~7.5 (biologically measured, far from the
data-generating regime) or ~100 (matches the generator but is grounded in the user's own
simulation choices); a wide spread spanning both keeps the prior informative but honest.

**Notation:** do **not** use `mu` for the prior's centre. In this codebase `mu` already
means (a) the QSS lift's fast-variable timescale in `eval/dynamical.py` and (b) the
generator's degradation rate. Use `log_D_ratio_target` / `log_D_ratio_spread`.

**Shape, as designed:** a soft log-normal penalty `w·(log D_ratio − target)²/(2·spread²)`,
**not a clamp**, with the prior centre sampled per restart (`target_s ~ N(target, s)`,
the randomised-prior trick) so training stays stochastic and the seed distribution
survives.

**Four attachment points** (`docs/DATA_INTO_MODEL.md` §4.4), noting that the experiment
scripts assemble their loss inline so a `losses/total.py`-only change reaches nothing:

1. `model.py` θ_D init scale — re-centre/widen, or draw an *ordered* θ_D. Init-side
   prior. **Reaches every script**, since all construct `RNGRN(N=3, seed=seed)`.
2. `model.py` the `D` property — reparameterise so the *ratio* is the learned quantity.
   Hard structural prior. **Reaches every script.**
3. A new term in `losses/terms.py` + `total.py` + a config weight — the clean,
   config-switchable soft prior. **Does not reach the experiment scripts** unless also
   added there.
4. Directly in the script's inline loss — reaches the numbers, but script-local and
   unaudited.

**And a scale note:** recovery never integrates time. It solves `f(x*) = 0` for the
model's own homogeneous steady state and works on σ(k) from the Jacobian, so nothing
time-dependent is trained.

---

## 12. Where the recorded knowledge lives

Prior sessions' findings are in this repo's docs and in the Claude Science project
memory (133 rows). Anything in `CLAUDE.md` is authoritative for conventions. Where a
doc and the source disagree, **the source wins** — two such discrepancies were found
while writing `CLAUDE.md` and are logged in it as known gaps.
