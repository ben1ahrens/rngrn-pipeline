# SPEC — Fourier-space training for RNGRN

Written 2026-08-11, at the end of a design discussion with the owner. Every decision
below was made explicitly in that discussion; nothing here is a placeholder unless it
is marked UNCALIBRATED, and every UNCALIBRATED item carries the rule that will
calibrate it. Companion document: `docs/PLAN_fourier_training.md` (stages, gates,
diagnostics, file-change map). Audience: a reader with no context beyond the repo.

Two ground rules carried over from the discussion, restated because they bind every
section below:

- **No threshold, tolerance, test, or gate may be weakened to make something pass.**
  Changing one is an owner-level decision made in conversation, with a stated reason.
  (Updating `tests/test_losses.py`'s *enumeration* of term keys when a term is added
  is not a weakening — it is the pinned list doing its job. Weakening would be
  loosening a numeric bar or deleting an assertion.)
- **The firewall (`tests/test_firewall.py`) may not be removed or bypassed.** Any
  restructuring it needs goes to the owner first.

---

## 1. The stopping criterion

We iterate until BOTH hold, and we stop when they do. Loss values, spectral
distances, and training curves are diagnostics — they are never evidence that we are
done.

**Criterion 1 — RECOVERY (revised in discussion, 2026-08-11).** Given held-out
pattern data generated from a known Turing patterning network, RNGRN recovers **a**
GRN — not necessarily the generator's — whose forward simulation satisfies
Criterion 2, and does so **reproducibly**:

- **R1 (binding now): sign-structure reproducibility.** Over 5 independent training
  runs (5 seeds) on the same pattern, all 5 recovered sign structures are identical:
  every pairwise comparison of the 5 recovered N×N sign matrices agrees on all
  entries, under the **identity** species mapping (channels are observed and ordered,
  so species are not anonymous; permutation slack would be unearned). The zero-call
  threshold that decides "this edge is absent" must be **calibrated** (§9.3) before
  this gate is meaningful — the inherited 5%-of-max rule is UNCALIBRATED.
- **R2 (binding once calibrated): parameter similarity.** The 5 runs' parameters
  agree on **time-scale-invariant combinations** (§9.4) to within a threshold set by
  the pre-registered calibration rule of §9.4. Until that rule has produced a number
  (recorded in `docs/DECISIONS.md`), R2 is reported, not gated.
- **Reported, not gated:** agreement of the recovered sign structure with the
  generator's ground-truth circuit. The owner's decision: the learned GRN does not
  need the target's topology. Ground-truth comparison remains in every report as a
  diagnostic and as input to the identifiability story.

**Criterion 2 — FORWARD VALIDATION.** The recovered network, simulated forward with
its recovered parameters (the recovered parameter vector: `KA, KR, alpha, delta,
beta, D` — what `RecoveryResult.params` holds; note the codebase's `mu` names other
things and is NOT this vector), runs to completion and produces a genuine Turing
pattern:

- **F1: Turing-unstable**, verified numerically by the strict dispersion test
  already in the repo (`eval/analysis.turing_ok`: `max Re eig(J) < 0` at the
  homogeneous steady state AND `max_{k>0} sigma(k) > 0`).
- **F2: stationary, non-homogeneous.** The QSS rollout (§8) reaches a saturated
  state (its spatial-CV trace flat within tolerance) whose pattern amplitude is
  above the floor — not decayed to homogeneous, not blown up.
- **F3: wavelength — the primary objective.** The simulated pattern's measured
  wavenumber matches the observed pattern's within **one radial bin**:
  `|k*_sim − k*_obs| ≤ 2π/L`, both measured by the SAME estimator
  (`observables.raps`) on the same grid and L (§9.1).
- **Morphology: reported only, for now.** The rollout field's morphology class
  (via `scripts/phase_topology.measure`, the canonical sets' own annotator) and its
  distances are recorded in every report. Whether morphology becomes a binding gate
  is a decision point scheduled at Stage 2 of the plan — deferred by the owner, not
  forgotten. Orientation/rotation of the pattern is explicitly irrelevant
  everywhere: all comparisons are rotation-invariant.
- **Simulation regime: QSS gates; the finite-μ dynamical lift is a later milestone.**
  Forward validation runs the existing quasi-steady-state reaction (what
  `eval/rollout.simulate` integrates). Simulating the μ-lifted system in space
  (explicit promoter-occupancy variables) is a separate, later milestone with the
  preconditions `docs/FUTURE_WORK.md` demands (dt-convergence; reproduction of the
  QSS field at μ ≤ 1e-4) — its result is additional evidence, never the DoD gate.

**Stage 0 cannot satisfy this criterion, by construction.** A train/test split
within a single sample has no independent held-out data: a Turing pattern is
statistically homogeneous, so any partition of one sample has nearly identical
statistics to any other. Stage 0 is an overfitting sanity check with its own weaker
gates (`PLAN_fourier_training.md` §2). A Stage-0 result must never be reported as
evidence of recovery. The stopping criterion can only be evaluated at Stage 3, on
the pre-registered held-out samples.

---

## 2. The premise, corrected

The original motivation — "real-space pixel losses spend most of their gradient on
phase" — does not describe this codebase. The default objective has **no** pixel
loss: the one pixel-level term (`stationarity_residual`, key `resid`) was measured
harmful and settled OFF (exp06; `losses/terms.py` DEFAULT_WEIGHTS). What training
consumes from the observed frame today is exactly **two scalars**: `k*_obs` from the
radially averaged power spectrum of channel 0 (`recover.py`), and `frame.mean()`.

The real problem is therefore **data starvation**, not phase waste: ~3N²+3N
parameters constrained by two numbers plus priors. The measured failure mode matches
(C1 tuning: 7/8 seeds never reach the Turing regime). Fourier-space training widens
the data channel — full RAPS shape, 2-D spectral structure, per-channel amplitudes —
while still discarding phase, which for a phase-arbitrary Turing pattern carries no
circuit information. The conclusion of the original premise survives; the mechanism
is different, and the difference matters: we are ADDING information to the
objective, not removing phase from an existing pixel loss.

---

## 3. The forward map: implicit differentiation at the patterned steady state

Training gains, for the first time, a model-side predicted pattern. The linear
dispersion cannot provide one (it predicts which modes grow, not the saturated
spectrum), so:

1. **Forward solve, gradient-free.** With the current parameters θ, integrate the
   QSS reaction–diffusion system `∂u/∂t = D∇²u + f(u; θ)` from `x* + ε·noise` to a
   stationary patterned state `u*(θ)` — using the existing spectral ETDRK4
   integrator (`eval/numerics.integrate_etdrk4_rfft`), under `torch.no_grad()`.
   `u*` satisfies `F(u*, θ) := D∇²u* + f(u*; θ) = 0`.
2. **Backward, one adjoint solve.** For any scalar loss `L(u*)` of the predicted
   pattern, the implicit function theorem gives
   `dL/dθ = −λᵀ ∂F/∂θ` where `(∂F/∂u)ᵀ λ = ∂L/∂u`, evaluated at `u*`. One linear
   solve replaces backprop through thousands of PDE steps. `∂F/∂u` is the
   linearisation about the PATTERNED state (block: pointwise Jacobians of f plus
   the diagonal spectral Laplacian), applied matrix-free; the solve is Krylov
   (GMRES/CG on the normal equations), preconditioned by the Fourier-diagonal
   diffusion part.
3. **Zero modes.** The patterned state is translation-degenerate: `∂F/∂u` has (at
   least) two zero eigenvalues along `∂u*/∂x, ∂u*/∂y`. Every loss in this spec is
   built from `|F(u)|²` magnitudes and spatial moments, hence translation-invariant,
   so `∂L/∂u` is orthogonal to the zero modes and the adjoint solve is consistent;
   the solver still projects the Krylov space off the numerically-computed zero
   modes for conditioning. **This orthogonality claim is verified, not assumed** —
   diagnostic D1 of the plan compares the IFT gradient against finite differences
   before any training code is accepted.

   > **Superseded in part, 2026-08-12 (D1 measured; D-FFT-10 binds).** The solver
   > description in items 2–3 — projected Krylov (GMRES/CG on the normal equations,
   > space projected off the zero modes) — is exactly the scheme D1 measured to be
   > BIASED: on the grid the translations are only near-null, so the projected
   > residual converges (1e-13) while the TRUE residual stalls (5.5e-4–5.7e-3) and
   > every gradient inherits that error. The adjoint is instead the MINIMAL-NORM
   > least-squares solution — right-preconditioned LSMR, `M = (γ + Dk²)⁻¹`, with
   > true-residual refinement — and the Newton polish carries a 2×2
   > translation-subspace correction. The orthogonality verification of item 3
   > (measured ≤1e-18) is precisely what makes the minimal-norm solution the correct
   > adjoint. See `docs/DECISIONS.md` D-FFT-10 and `docs/DIAGNOSTICS_fft.md`
   > F-D1-2/F-D1-3; implemented in `src/rngrn/forward.py`.

**Ignition, and when the forward map exists at all.** A randomly initialised RNGRN
is almost never Turing-unstable; its forward solve relaxes to homogeneous and no
spectral loss exists. Therefore (owner decision):

- **Every existing objective term stays on, unchanged, for the whole run.** The
  k*-anchor, Turing hinges, frame-scale anchor, and anticollapse are linear-theory
  constraints defined for ANY θ; they are the ignition system that pushes θ into
  the Turing regime, and the recovery net when a model drifts out of it mid-run.
- **Spectral terms activate on DETECTED pattern existence, not on a schedule.** The
  forward solve runs only when the model is currently Turing-unstable
  (`sig_max_pos > margin` from the hinge diagnostics, steady state converged); the
  spectral terms contribute only when the solve actually patterned (amplitude above
  a floor — calibrated in diagnostic D2). When absent they are **omitted, never
  zeroed**, with a `spectral_skipped` flag in `parts`, following the repo's
  `resid_skipped` precedent, so a log can never confuse "no pattern yet" with
  "spectral loss = 0".
- **The current default objective is preserved verbatim as a permanent baseline
  config** (`configs/` arm), for the ablation ladder and for continuity of every
  recorded number. "Spectral-only after ignition" (dropping the k*-anchor once lit)
  is a later ablation arm, not a design commitment.

This change makes `CLAUDE.md` §7c's "training never simulates" false. The branch
amends that section in the same change that lands the forward solve (source wins,
doc gets fixed — §8 of CLAUDE.md).

---

## 4. The spectral representation

All spectral quantities are computed with `torch.fft` (currently absent from the
entire training path — this is greenfield), in double precision, in the objective's
own length units (`L_model = 1.0` on the nondim path, physical L otherwise; every
band is defined RELATIVE to `k*_obs`, so the construction is unit-free and does not
reintroduce L on the nondim path).

For a field `v` (one channel, H×W, periodic):

- `P(k) = |FFT2(v − mean(v))|²` — the 2-D power spectrum. Phase is discarded here
  and nowhere recovered: this is deliberate (Turing phase is arbitrary).
- **RAPS**: radial average of `P` into bins of width `Δk = 2π/L`, matching
  `observables.raps` bin-for-bin (the torch implementation is tested against the
  numpy one to tolerance; the numpy `raps` remains the measurement-side reference).
- **Bands**, all relative to `k*_obs` (measured once from the observed frame,
  exactly as today):
  - `B_train` — the fitting annulus around the fundamental:
    `[b_lo·k*_obs, b_hi·k*_obs]`, provisionally `b_lo = 0.5, b_hi = 1.5`
    (UNCALIBRATED until diagnostic D3 fixes the edges from the measured spectrum's
    support; recorded in DECISIONS.md when set).
    *Closed 2026-08-12:* D3 measured the edges at `b_lo = 0.60, b_hi = 1.55`
    (the ≥1%-of-peak contiguous RAPS support, 8 bins) — see DECISIONS.md
    D-FFT-9 closure 1. The provisional values above are kept for the record.
  - `B_low` — the low-k shoulder `(0, b_lo·k*_obs)`, excluding the DC bin.
  - `B_harm` — the harmonic band `[b_hi·k*_obs, 3·k*_obs]`.
  `B_low ∪ B_harm` are the HELD-OUT bands of the Stage-0 gate; at later stages all
  bands may enter the fit (per-stage masks are config, `loss.spectral.fit_bands`).
- **Normalisation — pinned to the training band.** Shape quantities are normalised
  by `Σ_{k∈B_train} P(k)` — never by total power, never by any statistic that
  includes held-out bands (that would leak held-out content into the fit) and never
  by any ground-truth quantity. This makes shape terms gain-invariant (a scalar
  rescale of the field cancels), preserving the repo's documented invariance for
  everything except the explicitly amplitude-aware terms below.
- **2-D structure.** The angular content that radial averaging destroys — and the
  repo has MEASURED that destruction: 1-D RAPS classifies morphology at 54.3% vs
  83.5% for a 2-D spectral block (`scoring/morphology.py`) — is captured by the
  band-limited 2-D log-power block, compared rotation-invariantly (over the
  rotation group of the square lattice, min over rotations; orientation must not
  matter, per the owner). The existing comparator `scoring.morphology.
  spectral_distance_2d` CANNOT be imported recovery-side (firewall: no
  recovery-side module may import `rngrn.scoring`); its arithmetic is PORTED to a
  side-neutral home (`rngrn/utils.py` precedent: `d_ratio_of`) with a drift test
  binding the two, exactly as `tests/test_plausibility.py` does for the duplicated
  box loader.

---

## 5. Loss terms

There is **no loss-term registry** in this codebase — terms are hard-wired in
`losses/total.py::compute_terms` and weighted via `LossConfig.weights`. "Term key"
below is the key in `term_vals` / `loss.weights.<key>` / the logged `L_<key>`.
Adding a term touches six places (terms.py + batched twin or loud refusal; total.py
serial + batched; `config.py::LossConfig.weights`; `configs/base.yaml`; run-index
`w_<key>` in train.py + export.py; history picks up `L_<key>` by name but the
ARRAY_SCHEMA_VERSION must bump when the column set changes).

Existing terms (unchanged, on throughout): `kstar`, `turing`, `anchor`,
`anticollapse`, opt-in `resid` (default 0, settled off) and `param_prior`
(default 0, opt-in). New terms, all computed on the predicted pattern `u*` vs the
observed frame, all only when ignited (§3):

**`spec_shape`** — gain-invariant spectral shape, per fitted channel c and band set:

    S_c(k) = P_c(k) / Σ_{k' ∈ B_train} P_c(k')        (model and observed alike)
    L_spec_shape = Σ_c Σ_{k ∈ fit bands} ( log S_c^mod(k) − log S_c^obs(k) )² · w(k)

  on the RAPS bins (radial part) — captures band shape and peak position;
  log-power, band-limited, train-band-normalised.

**`spec_aniso`** — the 2-D block distance of §4 (ported `spectral_distance_2d`
  arithmetic, band-limited, rotation-minimised) between model and observed
  channel-c spectra — captures the angular structure that separates morphologies.

**`spec_amp_mean`** — per-channel mean (DC) match, log-space:
  `Σ_c ( log mean(u*_c) − log mean(frame_c) )²` — generalises the existing
  frame-scale anchor to per-species means of the PREDICTED PATTERN (the anchor
  itself stays, unchanged, anchoring x*).

**`spec_amp_fluct`** — per-channel fluctuation scale, log-space:
  `Σ_c ( log std(u*_c) − log std(frame_c) )²` — pattern amplitude per species.

The two `spec_amp_*` terms are the ONLY amplitude-aware spectral terms, by owner
decision: shape stays gain-invariant; amplitude is separate, **ON for Stage 0**
(synthetic, noise-free, absolute concentrations — amplitude is legal clean signal),
individually ablatable, and switch-off-able wholesale for future data with unknown
gain. They are also what makes the Stage-0 channel co-gate (§below) meaningful.
Channel masks (`loss.spectral.fit_channels`) and band masks are config: Stage 0
fits species 0 only on `B_train`; the amplitude terms at Stage 0 read channels 1–2
ONLY through the co-gate evaluation, never through the fitted loss.

Real-space restoration terms (§6) share the ignition gate: **`real_moments`** —
match of low-order spatial moments (at minimum the skewness, sign-sensitively) of
`u*_c` to the observed channel: `Σ_c ( skew(u*_c) − skew(frame_c) )²` on fitted
channels. Differentiable, translation- and rotation-invariant, phase-free.

Weights for all new terms: UNCALIBRATED at birth, swept at Stage 0 (that is what a
tuning-role sample is for), recorded in DECISIONS.md when settled.

---

## 6. What the spectrum deliberately discards, and how it is restored

- **Phase** — discarded everywhere, restored nowhere. For statistically homogeneous
  Turing patterns phase carries position, not mechanism. This is the point of the
  design.
- **Odd-moment / contrast-inversion structure.** |F|² is blind to `v → −v` about the
  mean: spots and holes are near-identical in power. The repo's own classifier
  needs real-space skewness to separate them, and the canonical labyrinth set
  itself contained 3 mislabelled holes samples. Restored by `real_moments` (§5) in
  training, and by the morphology classification of the rollout (reported) in
  validation.
- **Absolute scale** — deliberately absent from shape terms, restored by the two
  explicit amplitude terms while they are on, and by the frame-scale anchor
  throughout.
- **Cross-channel phase relationships** (which species is high where) — discarded
  by per-channel spectra. Partially restored by per-channel amplitude ratios (the
  eigenvector structure of `J − k²D` fixes them; the Stage-0 channel co-gate tests
  exactly this) and, if later stages need it, by cross-spectra — noted as a future
  option, not built now.

---

## 7. Leakage boundaries

What may and may not be computed from what. A within-sample split makes leakage
easier to commit and harder to notice; this table is the audit line. The
firewall test (`tests/test_firewall.py`) enforces the import-level half; the rest
is design discipline, checked at review by the `firewall-auditor` agent.

LEGAL (recovery side may read):
- The observed frame; any statistic of it: `frame.mean()`, `kstar_obs` via
  `observables.raps`, band powers, moments. All existing legality arguments carry
  over unchanged.
- `L` — legal but LOADED: on the legacy registered datasets the generator set
  `L = clip(6·2π/k*_true, 18, 220)`, so k*_true is recoverable from L alone for
  94.8% of those samples. On the canonical sets the periods-per-box were drawn
  against an oracle-leak bar (the L-only trivial predictor errs 33–37%), but the
  rule stands: any L-derived quantity is within the firewall yet must NEVER be
  offered as evidence of recovery; the trivial-predictor control ships with every
  k* number (D-EVID-7).
- The k-grid (`recover._kgrid_for`), anchored to `kstar_obs` — unchanged.

FORBIDDEN (must never reach the recovery path):
- Everything already in `FORBIDDEN` (`tests/test_firewall.py`): the generator
  modules, `data.gate`/`AnswerKey`, the payload-reading scripts.
- The payload attrs `k_star`, `k_star_fft`, `params_json`, `x_star`, `jacobian`,
  `D`, `interaction_matrix` — the recovery path re-measures `k*_obs` from the frame
  and never reads the stored annotation. (The stored `k_star_fft` differing from
  the live `raps()` value by 1.6–3.7% is today the only thing keeping the headline
  validation metric non-circular — see the circularity note below.)
- **Normalisation constants computed over held-out content.** The Stage-0 spectral
  normaliser is `Σ_{B_train} P` of the TRAINING band of the TRAINING channel only.
  Normalising over all k, or over channels 1–2, would leak held-out structure into
  the fit. Same rule for any windowing/detrending statistic.
- The held-out k-bands and channels themselves, in any term that is fitted at
  Stage 0. They are evaluated exactly once per gate evaluation, by scoring code.

CIRCULARITY (not leakage, but a validity hazard — stated so it is planned for,
not discovered): once training fits the observed spectrum, `kstar_fft_rel_err` — the
current pre-registered headline — is partially self-fulfilling: the model was
TOLD the spectrum. It remains reported (pre-registration binds), but the
**independent** evidence of the stopping criterion is Criterion 2: the k* of a
FORWARD-SIMULATED pattern is an emergent property of the recovered dynamics, not a
fitted quantity, and the Stage-0 held-out bands/channels are the within-sample
analogue. Any report quoting `kstar_fft_rel_err` for a spectrally-trained run says
this in the same sentence.

New scripts that open `payload.h5` (diagnostics will need at least one) are added
to `FORBIDDEN` at birth — the completeness test cannot see `scripts/`
(CLAUDE.md §5, learned 2026-08-10).

---

## 8. Real-space validation procedure (implementable detail)

Runs once per recovery (per seed), after training, exactly as `train.fit` does
today via `_morphology_rollout` → `score_recovery`, extended as follows:

1. **Rollout**: `eval/rollout.simulate(model, L=ri.L, n=<target grid>, D=D_phys,
   seed=fixed)` — IC `x* + 1%·x*·N(0,1)`, spectral ETDRK4, existing saturation/
   collapse/blow-up detection. Grid: the target frame's grid (512²) is the
   reference; a cheaper validation grid is permitted ONLY after diagnostic D2 shows
   the measured k* and class calls agree between that grid and 512² on this sample
   (recorded in DECISIONS.md).
2. **F1**: `turing_ok(J_rec, D_phys)` — strict criterion, unchanged.
3. **F2**: `stopped_reason == "saturated"` (CV-trace flat within the existing
   tolerance) AND `patterned == True` (amplitude above the existing floor) AND NOT
   `blew_up`. A rollout that hits the step budget unsaturated is a FAIL of F2, not
   a retry-until-pass; the budget may be raised only as a recorded decision.
4. **F3**: `k*_sim = observables.kstar_of(field_0, L)` vs `k*_obs`, one-bin rule
   (§9.1). Reported with bin width and trivial-predictor control per D-EVID-7/8.
5. **Reported**: morphology class of `field_0` via `phase_topology.measure` at the
   matched grid/L, its margin, `morphology_distance`, the 2-D spectral distance,
   per-channel amplitude ratios of the rollout vs the target frame, ground-truth
   sign comparison (identity and best-permutation), and the full null-ensemble and
   reproducibility summaries of the stage's gates.

The 5-seed reproducibility gate (R1) consumes the 5 recovered sign matrices from 5
independent `train.fit` runs (distinct `train.seed`), identical config otherwise.

---

## 9. Thresholds and tolerances, with reasoning

### 9.1 Wavelength: one radial bin (BINDING)
`|k*_sim − k*_obs| ≤ 2π/L` — equivalently rel-err ≤ 1/p for a sample with p periods
per box: **11.96%** for `turing_labyrinth/sample_0000`, 2.9% at p=35.

> **Corrected 2026-08-14 (documentation audit).** This line read "12.5% … (p=8)",
> which is `1/8` from the *declared* ladder value `periods_per_box: 8` in
> `data/canonical_selection.json`. The *measured* value is `p = 8.3587`
> (`experiments/diag_fft/d3/results.json`, `channel_0.kstar_in_bins_p`), giving
> `Δk/k* = 0.033962/0.283878 = 11.96%` — matching `docs/DIAGNOSTICS_fft.md:174`,
> which had it right. The binding rule `|k*_sim − k*_obs| ≤ 2π/L` is unchanged and
> unaffected; only the illustrative percentage was wrong, and the correction
> **tightens** it. `docs/DECISIONS.md:2606-2607` still carries the old 12.5% and is
> pending the same fix (that file had uncommitted changes when this audit ran).
Reasoning: the RAPS estimator's resolution IS Δk = 2π/L; the pre-registered 8.3%
was derived on the three_gene sets and D-EVID-8 records it is not
dataset-independent; at p=8 it demands sub-bin precision the measurement does not
carry. One bin claims exactly the precision the estimator supports, per sample, and
tightens automatically with p. It clears the L-only trivial predictor (33–37% on
canonical sets) by ~3×, so it is meaningful against its control. Always reported
with `kstar_fft_bin_width` and the trivial-predictor control.

### 9.2 Forward validation F1/F2 (BINDING)
Strict Turing criterion and saturation/amplitude floors as already implemented
(`turing_ok` tol 1e-9; rollout `patterned` floor `max(1e-3, 0.02·|x*_0|)`).
The saturation-tolerance knobs are inherited [TUNE] items; if Stage-0 diagnostics
show them mis-set for 512² canonical fields, changing them is a recorded decision.

### 9.3 Sign reproducibility R1 (BINDING) and the zero-call threshold
5/5 identical sign structures under identity mapping. The zero-call rule (edge
absent if magnitude < 5% of max, `recover._topology`; scoring rtol 0.05) is
UNCALIBRATED. Calibration rule (pre-registered here): from the Stage-0 ≥10-seed
run set, fit the magnitude distributions of consistently-present vs
consistently-absent edges across seeds; place the threshold at the separation
point; record in DECISIONS.md; then R1 becomes binding with that number. If the
distributions do not separate, that is a finding (the sign structure is not stable)
and gates R1 fails honestly rather than being tuned until it passes.

### 9.4 Parameter similarity R2 (REPORTED until calibrated)
Statistic: per-quantity log-space spread (max pairwise |Δlog|) across the 5 runs,
computed on TIME-SCALE-INVARIANT combinations only — D-ratios, α/δ, β/δ, K's,
D·k*²/δ — never raw rates. Reasoning: a single stationary frame is invariant under
jointly scaling (f, D) by any constant, so absolute rates are pinned only by
priors; gating on them would fail for reasons carrying no information.
Calibration rule (pre-registered): run Stage 0 with ≥10 seeds; if sign structures
cluster, set the threshold at the measured within-cluster spread plus margin;
record in DECISIONS.md; R2 then binds.

### 9.5 Stage-0 gate 1: held-out k-bands (BINDING at Stage 0)
The recovered model's held-out-band prediction error (the `spec_shape` distance
evaluated on `B_low ∪ B_harm`, which the fit never saw) must **beat every member**
of a ≥6-member null ensemble: alternative sign structures (the distinct structures
arising across the multistart restarts, plus single-edge-flip variants of the
winner), each refit to `B_train` with gates frozen to that structure under the
identical budget. Reasoning: with ground-truth topology dropped from the DoD,
"wrong topology" must mean "alternative mechanism fitted under identical
conditions"; beat-all is the strictest form and every outcome is informative —
pass (unseen bands select the mechanism), fail-with-spread (bands discriminate,
training picked wrong), fail-no-spread (the measured identifiability verdict for
this data: reported as a finding, the gate redesigned with the owner, never
lowered).

### 9.6 Stage-0 gate 2: channel amplitude ratios (BINDING at Stage 0)
Predicted per-channel ratios `std(u*_c)/std(u*_0)` and `mean(u*_c)/mean(u*_0)`
(c = 1, 2) vs the observed frame's, in log space. These are forced by the
eigenvector structure of `J − k²D` and the saturation — mechanism the
single-channel fit never saw (channels 1–2 are masked from every fitted term at
Stage 0). Threshold: calibrated from the same null ensemble as 9.5 (the recovered
model must beat every null member's ratio error); reported alongside as raw
log-errors. Caveat recorded: channels 1–2 of sample_0000 are near-flat
(cv ≈ 0.075 vs 0.81), so the DYNAMIC RANGE is small — predicting near-flatness
correctly is itself the test, and the frames carry no observation noise, so the
signal is clean.

### 9.7 Patch control (REPORTED, never gated)
Spectral distance between held-out spatial quadrant and training quadrants of the
observed frame itself — the estimation-noise floor of the within-sample statistics.
Reasoning: statistical homogeneity makes a patch gate near-trivial; its only use is
as the yardstick "fit error should not be far below the frame's own patch-to-patch
variability" (that would indicate fitting estimation noise).

### 9.8 Band edges, ignition floor, spectral weights (UNCALIBRATED at birth)
`b_lo, b_hi` (§4), the ignition amplitude floor (§3), and all new term weights are
UNCALIBRATED until the plan's diagnostics D2/D3 and the Stage-0 sweeps set them;
each lands in DECISIONS.md with evidence when set. They are marked in code with the
repo's UNCALIBRATED convention at introduction.
