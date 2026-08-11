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
**Corrected 2026-08-11 — the list has grown from 5 names to 12 and the escape routes below
are mostly closed.** `FORBIDDEN` now also carries `data.gate` and `data.registry` (added
2026-08-04, precisely because `gate.from_registry` returns the full `(RecoveryInput,
AnswerKey)` pair) plus the payload-reading scripts `td_figures`, `gen_tg3`, `canon_select`,
`canon_generate`, `canon_annotate` (2026-08-10). So of the bypasses listed below, the three
via `gate`/`registry` are **now caught**; reading `payload.h5` directly through `h5py`
remains uncaught, and that is the live hole. `RECOVERY_SIDE` also gained `history.py`, and
`test_every_loss_and_eval_module_is_classified` now fails on any unclassified module under
`losses/`/`eval/` — but it globs `src/rngrn` only, so `scripts/` stays a manual allowlist.
**Read `tests/test_firewall.py` for the current lists, not this paragraph.**

*Historical, as of 2026-08-04:* `tests/test_firewall.py` is a *substring match over import
lines* against five names — `rd_models`, `data.solver`, `data.cache`, `AnswerKey`,
`answer_key`. Empirically:
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
`scripts/exp02_objective_fix.py`; **promoted into `losses/terms.py` by unit 1** — it is
`terms.turing_hinges_split` (`terms.py:186`) and is the library default
(`config.py:105`, `split_hinges=True`). *(Corrected 2026-08-04; this line previously said
"not yet promoted".)*

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

### 2.6 Library path vs experiment path — same objective now, still different code

`recover.recover` (library) and `scripts/exp0*.py::fit` (what actually produced every
result above) are different **code**, and were once different **objectives**.

> **As of `d8070ca` — SUPERSEDED, retained because it dates the numbers above.** Library
> weights were `{kstar:1.0, turing:1.0, resid:0.3, anticollapse:0.5, morphology:0.0}`, with
> the original self-defeating hinges, `resid` on at 0.3, and no frame-scale anchor. Results
> measured under those defaults are **not comparable** to results measured after.

**Corrected 2026-08-04.** Unit 1 promoted the learned objective into the library. Defaults
are now `{kstar:1.0, turing:1.0, resid:0.0, anticollapse:0.5, anchor:2.0, morphology:0.1,
param_prior:0.0}` with `split_hinges=True` and data-first staging (`config.py:99-105`) —
split hinges, the frame-scale anchor, staging, and the residual **off** (settled by exp06,
not merely untuned). Other defaults unchanged: `n_restarts=4, adam_steps=1500, adam_lr=0.05,
lbfgs_steps=50, jac_floor=1.0, grad_clip=10.0, form='competitive'`. Note `tau=0.12` is
**inert** — it reaches `kstar_anchor` and is never read (`TUNING.md:37`).

What is still true: the scripts assemble their loss **inline**, so the two paths agree today
only because someone checked, and nothing enforces it. Nothing is persisted mid-run, so an
interrupted restart loses the whole 4×1550-step budget.

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
destroyed by the new term. Matas-Gil & Endres never quote a percentage parameter error
for the PINN at all — their only quantitative anchor is comparative ("10–20x more noise
than LS", up to 40x for Brusselator). **"Reported errors <10 % on inferred parameters"
was a misattribution in an earlier version of this section** — that figure belongs to
Kho et al. (NTU/A\*STAR, *Design of Turing Systems with Physics-Informed Neural
Networks*, reference [32] in Endres), not to Endres/Matas-Gil. The local file that was
misread is `papers/endres_pinn_turing_design.pdf`, which is itself mis-named — it is
Kho et al., not Endres (see PR notes; renaming is local hygiene, not part of this diff
since `papers/` is gitignored).

**CRITICAL READING CORRECTION — what "batches of 128" means.** They are **128 spatial
pixel locations drawn from the ONE image** — not 128 images and not 128 samples. From the
source: *"When training the approximation, we use information on the whole pattern, but
when we switch to the LPDE, we only use interior points"*; their Fig. 3 shows N randomly
selected points on a single Turing pattern. An earlier reading in this project concluded
no minibatch dimension existed; that was wrong. This is what
`scripts/exp05_pixel_minibatch.py` implements and tests. Confirmed and strengthened: this
is now also proven from the actual code
(`Endres-group/IPTP-paper-code`, `models/RBFPINNs_ChemPat.py` lines 531–538), which is
stronger evidence than the paper text alone.

**MISATTRIBUTION CORRECTED — the "3×3 pixel region is sufficient" / power-law claim was
never a PINN result.** The paper contains **no PINN pixel-count study**. "A 3×3 pixel
region is sufficient" and "error falls as a power law with exponent ≈ −1/2" are
**least-squares (LS) results**, not PINN results — the −1/2 is ordinary sqrt(N)
noise-averaging of a *linear* estimator, and it does not license any inference about PINN
data efficiency. The paper is also self-inconsistent on this point (3×3 in the Fig. 3
caption vs 4×4 in the body); the real minimum pixel count is a **rank condition** — 2
pixels for Schnakenberg/Brusselator, 3 for FitzHugh–Nagumo. Do not cite "3×3 suffices" or
the −1/2 exponent as evidence about PINN (or RNGRN) pixel efficiency.

**So the "200,000 iterations from a single image" figure is real, and it is the reason
this project ever tested long budgets** — but note §3: our own hit rate **saturates at
4000 steps**, so their budget does not transfer. Two structural reasons it need not:
they observe **all** species (in their synthetic benchmarks — see below) and assume the
candidate model **is** the true model, so their PDE-residual loss can reach zero; neither
holds here. One more nuance worth carrying: their 200,000 iterations are ~10,000
**epochs** over a 2,500-pixel image with only ~900 trainable parameters — comparing raw
step counts against a differently-sized model (ours) is not meaningful on its own.

**Two more transferable points.**

1. They hit the same **unknown intensity scale** problem (experimental image pixel values
   23–255) and solved it with a **free rescaling variable** — functionally the same device
   as our frame-scale anchor (§2.3, and §2.8 for what that scale actually encodes).
2. **MISATTRIBUTION CORRECTED.** "They treat the method's stochasticity as a feature,
   providing multiple parameter alternatives" is **also Kho et al.**, verbatim from their
   abstract, not Endres/Matas-Gil. Endres treat estimator spread as **error**, not as a
   feature. The "engine for biologically plausible values" framing was therefore resting
   on the wrong citation; it is re-attributed to Kho et al. here, which genuinely does
   support it. Results in this project are still reported as a **seed distribution**,
   never the best seed — that practice stands on its own merits regardless of citation.

**Differences from our setup, to keep in view:** 50×50 zero-flux vs our 96×96 periodic;
candidate model == true model vs our gated-Hill RNGRN which *cannot* represent the
generators' kinetics; **and, MISATTRIBUTION CORRECTED, "all species observed vs our
m ≤ N" is true for their synthetic benchmarks but FALSE for their experimental CDIMA
case**, where exactly one species is observed and the second is reconstructed via a
trainable free-scale affine map of the observed image (adopted for RNGRN's own m<N case
in `recover.py::FreeScaleLatent`, unit 13).

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
Turing manifold 82 % of the time (**that premise is withdrawn — see §10; the strict figure
is 0 %**, so this homotopy had no Turing manifold to start from in the first place), then
anneal from "stay Turing" toward "match k\*".
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
| `eval/analysis.turing_ok`, `robustness_cloud` | **Corrected 2026-08-11 — both claims are now false.** `turing_ok` is VALIDATED: D-EVID-11 re-measured the strict criterion on 400 seeds against `_perturb_cloud`'s independent implementation, and `BIO_VIABILITY.md` §1.3 re-checks it 19/19. `robustness_cloud` HAS been run on recoveries and reaches every run row via `robustness_volumes` (`validate.py:64,386` → `turing_volume_*`); see `C1_COMPETITIVE_TUNING.md` §9.2 (`robustness_n_used = 8`) |
| `eval/numerics.integrate_bdf1_newton_krylov` | **stub** — delegates to ETDRK4 |
| `losses/weighting.{GradNorm,NTK}Weighting` | **Corrected 2026-08-04:** both `raise NotImplementedError` in `__init__` (`weighting.py:69`, `:87`), naming the missing estimator — they do **not** fall back to fixed weights, and neither defines a `combine`. `RatioWeighting` (`:94`) is implemented |
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

> **RESOLVED — corrected 2026-08-11. This section said the decision was "still open"; it was
> already closed when that was written.** Option **(b)** was adopted: `three_gene_qvar`
> (committed 2026-07-29, a week before this section's last edit) draws periods-per-box
> ~U{3..14} to decouple L from k\*, per `PREREGISTRATION.md` §1a. It was extended again on
> 2026-08-10 by `turing_spots`/`turing_labyrinth`, which use a geometric periods-per-box
> ladder {8..40} and are now PRIMARY (D-CANON-4, `CANONICAL_DATASETS.md`).
>
> **§7's dataset inventory above is therefore stale by omission** — it lists only the three
> legacy families and none of `three_gene_qvar`, `three_gene_multiL`, `turing_spots`,
> `turing_labyrinth`. For the current inventory read `docs/CANONICAL_DATASETS.md` and
> `PREREGISTRATION.md` §1, not this section. The analysis below is kept because the reasoning
> is what justified option (b).

The user stated they "wanted all of the patterns for my training data to have the same
domain size", which the legacy data violates. Three options were measured and put to
the user — *(b) was chosen; see the box above)*:

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

> **⚠ WITHDRAWN 2026-08-04 — the whole ladder above is an artefact.** See
> `docs/DECISIONS.md` D-EVID-11. Those percentages came from `eval/analysis.turing_ok`
> testing `tr(J) < 0`, which a uniformly **unstable** system satisfies whenever one
> positive eigenvalue is outweighed by the others. Every qualifying draw was uniformly
> unstable with k\* pinned to the scan floor (1e-3), so the "Turing-unstable" reading was
> the uniform instability being counted as a structured one.
>
> **Re-measured under the strict criterion (max Re eig(J) < 0, unstable at some k > 0),
> N=3, 400 seeds:**
>
> | init | converged | any positive J diagonal | STRICT Turing | loose (superseded) |
> |---|---|---|---|---|
> | `default` | 400 | 0/400 = 0.000 | **0/400 = 0.000** | 0/400 |
> | `low_basal` | 398 | 114/398 = **0.286** | **0/398 = 0.000** | 206/398 = 0.518 |
>
> Note the loose figure is 51.8 %, not 82 %, even before the criterion change: unit B3's
> steady-state multistart raised converged inits from 255 to 398 while the loose-Turing
> count stayed at exactly 206.
>
> **What is still true:** low_basal produces the **positive Jacobian diagonal** that Turing
> instability requires and the default init never produces (0/400). That is necessary, not
> sufficient, and it is the only claim this init supports.

**The honest statement:** the low-basal init changes the *sign structure* of the Jacobian
diagonal, which is a Turing prerequisite. It does **not** place inits on the Turing
manifold.

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

## 11. The D-ratio prior — implemented, opt-in, with an unresolved tension

The failing exp09 runs collapse the D-ratio toward ~8 while the generators sit at
108–140. That motivated a soft prior.

**Corrected 2026-08-04: it is implemented.** `terms.py::param_prior` (`terms.py:415`), bounds
in `configs/bio_box.yaml`, switched on by `loss.weights.param_prior` (default **0.0** —
opt-in, wired at `recover.py:390-400`). The scoring-side counterpart is
`scoring/plausibility.py`. This section previously said "not implemented"; that was true at
`d8070ca` and stopped being true with unit 5.

The tension below is **not** resolved by implementing it, and is the reason the prior is
opt-in rather than on by default.

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

**Design choice — SETTLED 2026-07-29 at ~7.5** (`DECISIONS.md` D1; `config.py::dratio_centre`,
with `dratio_spread=1.0` in natural-log units). The alternatives and the reasoning are kept
below because the choice is contestable and the ~15× gap is real, not because it is still open:
centre on ~7.5 (biologically measured, far from the data-generating regime) or ~100 (matches
the generator but is grounded in the user's own simulation choices); a wide spread spanning
both keeps the prior informative but honest.

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

---

## 13. Restart seeds were NOT independent draws — fixed 2026-07-29

**THE DEFECT.** `recover()` built restart `r` as `RNGRN(seed=model_seed + r)`. With
`n_restarts=R`, run seed `s` and run seed `s+1` therefore shared `R-1` of their `R`
model inits — a sliding window, not independent draws.

**MEASURED EVIDENCE (pre-fix).** `three_gene_val/sample_0000`, 8 seeds x 2 arms, 200
Adam steps, `n_restarts=2`, `lbfgs_steps=0`: seed pairs `(0,1)`, `(3,4)`, `(5,6)` in one
arm and `(1,2)`, `(4,5)`, `(6,7)` in the other returned **bit-identical** loss, `D` and
sign structure to 16 digits. 8 seeds were approximately 4 independent draws.

**WHY THIS MATTERED.** Reproducibility across seeds is the project's primary success
metric (see the project brief: "the model must consistently learn the SAME TOPOLOGY
across seeds"). With overlapping restart windows, a "7 of 8 seeds agree" claim was
inflated by roughly the overlap factor, and no seed-count in the record before this fix
is trustworthy on that axis.

**THE FIX (`src/rngrn/recover.py`, `_restart_seed`).** Each restart's model-init seed
is now derived from a stable hash (`hashlib.blake2b`, matching the `stable_seed()`
pattern already used in `scripts/exp11_robustness_baseline.py` — NOT builtin `hash()`,
which Python salts per process via `PYTHONHASHSEED`) of the pair `(model_seed, r)`
rather than `model_seed + r`. Determinism is preserved exactly: the same
`(model_seed, r)` pair always yields the same derived seed and hence the same init,
across processes. Distinct pairs give unrelated draws.

**MEASURED EVIDENCE (post-fix).** Re-ran the identical paired experiment (same
sample, same 8 seeds, `n_restarts=2`, 200 Adam steps, `lbfgs_steps=0`) after the fix.
Of the 6 seed pairs that were bit-identical before the fix, **0 of 6 remain
bit-identical** (loss, `D`, and sign structure all differ for every pair). Also
verified directly: for adjacent run seeds `s` and `s+1` (`s` = 5, 6, 7; `n_restarts=4`),
the 4 derived restart seeds for `s` never intersect the 4 derived restart seeds for
`s+1`, and the corresponding model inits (`KA` at construction) are never equal —
see `tests/test_determinism.py::test_restart_seeds_independent_across_run_seeds`.

**CONSEQUENCE FOR THE RECORD — READ BEFORE COMPARING ANY SEED-COUNT.** Every
"N seeds agree" / "N of M seeds converged to the same topology" number recorded
**before this change** was drawn from an overlapping restart window and is **not
comparable** to numbers produced after it. Do not restate an old seed-agreement count
as if it survived this fix — it did not; it was measured against a different
(non-independent) sampling scheme. Any reproducibility claim going forward must state
whether it predates or postdates 2026-07-29's `_restart_seed` fix.

## 13. nc1 training instability — diagnosed, and it was the SOLVER (unit B3)

`nc1` (non-competitive) recovery was numerically correct at init but lost its steady state
part-way through Adam training, which blocked half of "both models must work". The cause is
now measured, and **it is not a property of the nc1 reaction**: it is a globalisation
failure of the single fixed Newton seed `x0 = ones` in `losses/terms.steady_state`.

### 13.1 Reproduction (real data, `recover()` path, 8 independent seeds)

`lbfgs_steps=0`, `n_restarts=4`, `base.yaml` weights. Counted per RESTART: how many of the
4×8 = 32 restarts `recover()` abandoned because `compute_terms` raised `SteadyStateError`.

| sample | N | adam_steps | nc1 seeds ok | nc1 restarts lost | comp. seeds ok | comp. restarts lost |
|---|---|---|---|---|---|---|
| `two_gene_classical_val/sample_0000`  | 2 | 250  | 8/8 | 4/32  | 8/8 | 1/32 |
| `two_gene_classical_val/sample_0000`  | 2 | 2000 | **1/8** | **31/32** | 8/8 | 6/32 |
| `three_gene_val/sample_0000`          | 3 | 250  | 8/8 | 3/32  | 8/8 | 3/32 |
| `three_gene_val/sample_0000`          | 3 | 2000 | 8/8 | **17/32** | 8/8 | 5/32 |

Two corrections to the phase-A framing fall out of this table. First, the defect is
**step-budget dependent**: at 250 steps nc1 and competitive are indistinguishable on the
N=3 sample (3/32 vs 3/32); the collapse only appears at the configured 2000-step budget.
Second, it is nc1-specific **in rate, not in kind** — competitive fails the same way, 5–6
times in 32, and that had not been noticed.

**Reproduction caveat, stated because it changes what was measurable.** The phase-A defect
was reported on `configs/nc1_milestone1_schnak.yaml`. That config CANNOT run at this commit:
with `data.source=reference`, `resolution=128`, `L=100`, `data/solver.py`'s explicit
spectral diffusion is unstable (Schnakenberg `Dv=40` needs `dt < 2/(D·k²max) ≈ 1.6e-3`, but
`dt_eff = 0.2/max|eig J| = 2.0e-3`), so `simulate_to_attractor` raises
`FloatingPointError: solver diverged at step 29` — before any model is built.
`nc1_milestone1_gm.yaml` fails the same way at step 133. That is a **pre-existing
generator defect, untouched by B3** (its `dt` cap accounts only for the reaction rate, never
for the diffusion CFL), and it is why the numbers above are on registered real data rather
than on the reference systems.

### 13.2 Mechanism — the hypothesised veto collapse is REFUTED

The standing hypothesis was that nc1's multiplicative repression veto `prod_j (1 - thetaR_ij)`
collapses toward 0, flattening the reaction and making `J` ill-conditioned. At the first
failure (nc1, N=2, `two_gene_classical_val/sample_0000`, seed 0, restart 0, Adam step 880):

* veto product = **[0.971, 0.9987]** — nowhere near collapse (`KR ≤ 0.573`).
* `cond(J)` at the state the solver returned = **2.19**; at the true root, **1.98**. The
  steady state is not ill-conditioned.

What actually happens is a **fold**. The Newton iterate from `ones` walks to
`x = [0.745589, 0.746393]` and sticks there:

* `|f| = 1.6755e-2`, `svd(J) = [2.179, 1.699e-6]`, `det J = 3.70e-6`, `cond(J) = 1.28e6`
  — the iterate is on the `det J = 0` manifold.
* the Newton step therefore explodes (`|step|` 0.18 → 100 → 5.0e3) while the damping
  collapses (`lam` 1.0 → 9.3e-10). The line-search accept test is "any decrease in |f|", so
  these no-op steps are ACCEPTED and all 100 iterations are consumed with `|f|` pinned at
  `1.675518e-2`.
* the relaxation fallback then restarts from the **same** `x0 = ones` and crawls in the same
  near-fold bottleneck, ending at `|f| = 1.31e-2` ≫ the `1e-4` bar → `converged=False`.

And yet the reaction has **exactly one** positive root (67 `least_squares` starts spanning
1e-3..10 per component all land on it): `x* = [0.0611113, 0.2228224]`, `|f| = 5.6e-17`,
`eig J = (-0.703, -1.355)`, `cond 1.98`. Newton reaches it from `beta/delta` at
`|f| = 2.0e-15`, from `0.1·ones` at `8.4e-17`, from `10·ones` at `5.7e-17`. The old code was
reporting "no steady state" for a model whose steady state exists, is unique and is benign —
and `recover.py`'s fail-loud then threw the whole restart away.

So this is **option (i)** in the unit brief — fix the solve — and no model change (option
(ii)) is needed or adopted. Nothing about the objective, the veto, or the nc1 form was
altered.

### 13.3 The fix, and why it is safe on by default

`terms.steady_state(..., multistart=True)`:

* **attempt 1 is the pre-B3 algorithm verbatim** (damped Newton from `x0`, then the
  relaxation fallback from `x0`) and short-circuits on success;
* only on failure is Newton re-run from the analytic bracket
  `terms.steady_state_bracket`: `lo_i = beta_i/delta_i` (production OFF),
  `hi_i = (beta_i + Σ_j alpha_ij)/delta_i` (production SATURATED), and `sqrt(lo·hi)`.
  The bracket is exact for **both** forms because at any root
  `x_i = (beta_i + prod_i)/delta_i` with `0 ≤ prod_i ≤ Σ_j alpha_ij` (nc1: `thetaA ≤ 1` and
  veto `≤ 1`; competitive: `prod_i` is a weighted average of the `alpha_ij` with weights
  summing to `< 1`). The seeds read only the model's own parameters — firewall-clean, no
  frame, no answer key, no randomness.

Because attempt 1 is unchanged and wins whenever it used to, **every call that converged
before returns a bit-identical `x*`**. Measured, difference exactly `0.0` (not "small"):

* along real 400-step training trajectories, `multistart` off vs on at every step:
  competitive N=2 400/400 legacy-converged steps identical; nc1 N=2 396/396 identical with
  4 steps rescued; competitive N=3 400/400; nc1 N=3 400/400. **1596/1596, max|Δx| = 0.0.**
* `tests/test_science.py::test_b3_multistart_is_bit_identical_wherever_the_legacy_solve_converged`
  pins this on a 48-model stressed ensemble; the trap itself is pinned at full float64
  precision in `test_b3_nc1_newton_fold_trap_is_rescued`.

`multistart=False` restores the exact legacy solve for reproducing pre-B3 numbers.

### 13.4 Outcome, on the same samples and seeds as 13.1 (adam_steps=2000)

| sample | N | form | restarts lost BEFORE | AFTER |
|---|---|---|---|---|
| `two_gene_classical_val/sample_0000` | 2 | nc1         | 31/32 | **0/32** |
| `two_gene_classical_val/sample_0000` | 2 | competitive |  6/32 | **0/32** |
| `three_gene_val/sample_0000`         | 3 | nc1         | 17/32 | **0/32** |
| `three_gene_val/sample_0000`         | 3 | competitive |  5/32 | **0/32** |

Seed-level completion is 8/8 for both forms at both N. **nc1 is no longer the blocker.**

### 13.5 What this does NOT establish, and one measurement hazard

* This is a **completion** rate, not a recovery-quality result. Whether nc1 recovers the
  same topology as competitive, or the right one, is untouched here — no topology,
  robustness or reproducibility claim is made or implied.
* Numbers recorded before B3 that involved a FAILED steady state are not comparable: those
  restarts existed and were discarded, and now they participate. Converged-path numbers are
  bit-identical and remain comparable.
* **Seed independence hazard, found while measuring this and relevant to every multi-seed
  claim in this project.** `recover(seed=s)` builds restart `r` with `model_seed = s + r`,
  so consecutive seeds with `n_restarts=4` SHARE 3 of their 4 initialisations — seeds 0..3
  are not 4 independent samples. All 13.1/13.4 numbers use `seed = 4·i` (disjoint blocks).
  The N=3 / 250-step row is the one exception: it was measured before this was noticed and
  uses consecutive seeds, so treat its 3/32 vs 3/32 as ~2 effective seeds, not 8.
