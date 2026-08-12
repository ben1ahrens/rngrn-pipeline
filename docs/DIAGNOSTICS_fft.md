# DIAGNOSTICS — Fourier-space training, milestone M0

Results of diagnostics D1–D6 (`docs/PLAN_fourier_training.md` §3), run 2026-08-12 on
branch `feature/fft-training`. Written for a zero-context reader; every number names the
artefact behind it. Scripts live in `scripts/diag_fft_d*.py`; machine-readable results in
`experiments/diag_fft/d*/results*.json` (tracked). Payload-opening scripts are in
`tests/test_firewall.py::FORBIDDEN` per CLAUDE.md §5.

None of these numbers is a recovery result. M0 licenses exactly one sentence class:
"the machinery works and the knobs are measured" (PLAN §10).

---

## D1 — IFT feasibility (GO/NO-GO)                      [status: **PASS** — 96², 10 directions × 5 terms, worst rel err 6.4e-8 (tol 1e-4)]

`scripts/diag_fft_d1.py`, `experiments/diag_fft/d1/results.json`.

Fixture: the recovered checkpoint
`experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/checkpoints/model.pt`
(N=3 competitive, dimensional path) — chosen because its run recorded
`recovered_turing=True` AND `morphology_match=True` (its rollout actually patterns as a
labyrinth). Re-verified in-script, never trusted from the label: sig_max = 0.3567 > 0 at
k*_lin = 0.176, max Re eig(J) = −0.4367 < 0. The fixture is a recovery OUTPUT — no
payload, no answer key; D1 needs no FORBIDDEN entry.

Setup: 96² grid, L = 8·2π/k*_lin = 285.5, dt = 0.185 (0.2/jac_rate, rollout.py's policy).
Patterned base state by ETDRK4 relaxation (amplitude+k* flat to 1e-4 over 5×500-step
chunks) then projected-Newton polish to ‖F(u*)‖/‖u*‖ = 2.1e-12. Linearisation A = ∂F/∂u
matrix-free via torch JVP/VJP of F(u) = D∇²u + f(u) (spectral Laplacian); adjoint solve
by lgmres, Fourier-diagonal (γ + Dk²)⁻¹ preconditioner, Krylov space projected off the
translational zero modes. FD reference: central differences with the SAME u*(θ) map,
warm-started Newton-ONLY from the base pattern (an ETDRK4 re-relaxation would drift the
translation phase — measured to contaminate FD of the moment term at 64²).

Acceptance, stated before the run: per-direction relative error of the directional
derivative ≤ 1e-4 at the best ε of a {1e-3, 1e-4, 1e-5} sweep, for the four spectral
terms; `real_moments` reported separately (see finding F-D1-1).

**VERDICT: PASS.** Full record (`experiments/diag_fft/d1/results.json`, run.log): all
10 random θ-directions × all 5 term forms agree with central finite differences to
worst-case 6.4e-8 relative (spectral terms) and 8.1e-8 (`real_moments`), against the
pre-stated 1e-4 tolerance — after two solver defects that D1 itself caught were fixed
(F-D1-2, F-D1-3 below). The IFT forward map is GO; M0's gate condition holds.

Measured at 96²:

- Translational zero modes: ‖A t‖/‖A v_rand‖ = 3.6e-4 (both directions), halved from
  8.3e-4 at 64² — consistent with spectral-truncation origin, i.e. the modes are exact
  in the continuum limit.
- ∂L/∂u ⊥ zero modes: spec_shape/spec_aniso/spec_amp_* all ≤ 2e-17 (exact in float64 —
  FFT-power losses are analytically phase-blind). `real_moments`: 1.2e-6 / 9.2e-8.
- Adjoint solve residuals: 4–7e-13 for all five terms.

**Finding F-D1-1 (real_moments finite-grid phase sensitivity).** Pixel-space moments
(skewness) are NOT exactly invariant under sub-pixel translation of a band-limited field
(the v³ spectrum aliases), so ∂L/∂u for `real_moments` has a small zero-mode component:
1.3e-4 at 64² → ~1e-6 at 96². It converges away with grid but is structurally nonzero.
With the corrected adjoint (F-D1-3) the term's gradient nevertheless passes the FD check
at 1.0e-5 — the leakage is real but harmless at these grids.

**Finding F-D1-2 (grid pinning defeats a translation-projected Newton).** On the grid the
translations are only NEAR-null (‖At‖/‖Av‖ ≈ 8e-4 at 64², 3.6e-4 at 96²), so a
θ-perturbed fixed point sits at a slightly different pinned phase and Newton steps
projected off the translations stall at a measured floor of 2.4e-8 — the residual
component along the modes is exactly what the projection refuses to correct. Fixed by a
2×2 mode-subspace solve (G_ij = ⟨t̂_i, A t̂_j⟩) added to each Newton step; solves then
reach ≤1e-11 in ~2 s at 64².

**Finding F-D1-3 — THE ADJOINT MUST BE THE MINIMAL-NORM SOLUTION; a projected-GMRES
adjoint is measurably WRONG. This is the load-bearing result of D1 and binds M1's
`forward.py`.** The first implementation solved P AᵀP λ = P g (Krylov space projected off
the translations). Its PROJECTED residual converged to 1e-13 while its TRUE residual
‖Aᵀλ − g‖/‖g‖ stalled at 5.5e-4–5.7e-3 (per term, measured) — invisible to the solver,
and it biased every gradient by exactly that order: FD-vs-IFT rel err 1e-5 (amp terms) to
5.5e-2 (spec_aniso), CONSTANT across ε over three decades (the systematic-bias
signature; the u*(θ) map itself is smooth — drift scales exactly linearly with ε). A
tangent-mode cross-check agreed with the adjoint and disagreed with FD, proving the
transpose/∂F∂θ assembly correct and localising the error to the solve. The mathematics:
for a translation-invariant loss (∂L/∂u ⊥ t, measured ≤1e-18) the pinning response drops
out of dL/dθ identically and the correct adjoint is the MINIMAL-NORM least-squares
solution of Aᵀλ = g. Implemented as right-preconditioned LSMR (M = (γ + Dk²)⁻¹,
symmetric) with true-residual refinement; true residuals then reach 1.4–3.6e-12 and the
FD check passes every term: 1.2e-8 / 2.2e-7 / 1.8e-6 / 1.0e-5 / 2.8e-5
(amp_fluct / amp_mean / shape / moments / aniso), against tolerance 1e-4, with the
textbook ε-signature (truncation at 1e-3, FD roundoff at 1e-6). 64² probe:
`/…/tmp/d1_probe2` artefacts; the committed `experiments/diag_fft/d1/results.json` is
the full 96², 10-direction record.

---

## D2 — forward-solve characterisation                   [status: DONE]

`scripts/diag_fft_d2.py --full` via `guarded_run.sh` → `experiments/diag_fft/d2/
results_full.json` (D1's single nice'd thread ran concurrently; timings are
factor-level planning numbers, stated as such). Fixture = the D1 checkpoint,
L = 8·2π/k*_lin, 3 seeds/grid.

- **Rollout cost to horizon (606 steps, etdrk4_rfft, CPU)**: 0.81 s (96²), 1.33 s
  (128²), 6.8 s (256²), **39.2 s (512²)**. All runs stop at 'horizon'; the saturation
  rule never fires (consistent with rollout.py's own documentation of its [TUNE] knobs).
- **Ignition floor**: Turing saturated amplitude 0.170–0.172 vs non-Turing decay
  ≤6.7e-17 (contrast model verified sig_max = −0.433) — separation 2.6e15. The existing
  `pattern_floor = max(1e-3, 0.02·|x*₀|)` (= 9.7e-3 here) sits mid-gap → KEPT, now
  measured (D-FFT-9 closure 2).
- **Grid fidelity — NEGATIVE: no cheaper validation grid is licensed** (D-FFT-9
  closure 3). Morphology class vs 512²: flips on 2/3 seeds at 256² (spots vs labyrinth);
  k* one-bin agreement fails 2/6 at 96²/128². All class calls and F-gates run at 512².
  Band-limited spectral quantities did agree at 256² (D4's pilot remains valid for its
  band-distance purpose).
- **Torch ETDRK4 port**: bit-equivalent to numpy (max Δ 1.1e-13 after 100 steps).
  **Batched CUDA: 0.091 ms/step/member at 96²/B=32 — 11.6× over serial CPU** (1.05
  ms/step); CPU batching saturates (B=32 worse per member than B=1). Confirms PLAN §7's
  batched-GPU direction for restarts × members.
- **RSS**: 512² rollout peaks ~1.57 GiB — consistent with the §7a per-trainer footprint;
  the memory-guard floor stays correct.

---

## D3 — spectral estimator floor and band edges          [status: DONE]

`scripts/diag_fft_d3.py` (FORBIDDEN — opens payload; AnswerKey discarded unread),
`experiments/diag_fft/d3/results.json`. Sample: `turing_labyrinth/sample_0000`, 512²,
L = 185.006, noise-free synthetic. All k*-relative numbers use the re-measured
k*_obs = 0.283878 (`observables.kstar_of` on channel 0), never the stored annotation.

- Resolution: Δk = 2π/L = 0.033962 → k*_obs = 8.36 bins; one bin = 12.0 % of k*.
- Support of the RAPS peak (contiguous, units of k*_obs): ≥50 % of peak [0.84, 1.08]
  (2 bins); ≥10 % [0.84, 1.20] (3 bins); ≥1 % [0.60, 1.55] (8 bins). The spectrum is far
  more peaked than the provisional (0.5, 1.5) band assumed.
- Harmonic band B_harm = [1.5, 3]·k*: 12 bins carrying 1.5e-2 (RAPS sums) / 3.2e-2
  (2-D mode sums) of B_train power. **There is no observation-noise floor** — the sample
  is noise-free; the spectrum decays ~18 orders and reaches FFT round-off (~7e-11 of
  peak, top-decade median) only beyond ~15·k*. B_harm sits ~8e15× above that floor.
  **Answer to the plan's question: B_harm is unambiguously measurable at 512²/p=8; the
  held-out-band gate does not need to lean on B_low.** B_low ((0, 0.5·k*], 3 bins,
  ~1e-3 of B_train) is also measurable.
- Patch control (SPEC §9.7), 4 quadrants (256², quadrant Δk = 2Δk), 6 pairs, channel 0:
  `spectral_distance_2d` mean 6.69e-3 (5.77e-3–8.61e-3); spec_shape-form log-RAPS
  distance on [0.5,1.5]·k* mean 0.389 (0.122–0.621) ≈ 31 % per-bin power variation
  patch-to-patch. **This is the estimation floor: a fit error far below these numbers is
  fitting estimation noise.**
- Channels 1–2 (near-flat, cv ≈ 0.075): a k* IS measurable on both — 0.947/0.944 of
  channel 0's, within one radial bin, at ~35×/~150× smaller peak power. The held-out
  channels carry real spectral structure for the Stage-0 co-gate.

Band-edge decision → `docs/DECISIONS.md` (closure of the D-FFT-9 ledger item).

---

## D5 — baseline zero-call and seed spread               [status: DONE]

`configs/m3_registry.yaml` (N=3, m=3, observed (0,1,2), 2000 Adam steps × 4 restarts —
the current default objective; base.yaml alone is N=2 and a first launch with it was
discarded as wrong-model), `turing_labyrinth/sample_0000`, seeds 0–9, via
`guarded_run.sh`. Runs: `experiments/diag_fft/d5/runs/m3_registry_20260812_092043_seed*`;
analysis: `scripts/diag_fft_d5_analysis.py` → `experiments/diag_fft/d5/analysis.json`.

- **4/10 seeds Turing-unstable**; the 6 failures all collapse to a dispersion maximum
  pinned at the k-grid floor (`kstar_model` = kmin = 0.00668 identically). The 4 Turing
  seeds hit `kstar_fft_rel_err` ≈ 0.014 each.
- **10 distinct sign structures in 10 seeds — zero baseline sign reproducibility.** The
  R1 statistic's baseline value is 0/45 pairwise agreements. THIS is the A0 number the
  spectral arm must beat.
- **No edge is consistently present or consistently absent across seeds** (all 9 edges
  "mixed" at the 80 % consensus bar), so the zero-call calibration's precondition does
  not exist on the baseline. Per the pre-registered rule (D-FFT-9): a finding, not a
  licence — the threshold stays UNCALIBRATED until Stage-0 spectral runs produce stable
  structures to calibrate on.
- Invariant-combination log10 max-pairwise spreads (SPEC §9.4 combos): tightest is
  d_ratio at 0.44 decades (Turing subset, n=4); Dk*²/δ 2.12; α/δ 3.92; KA 8.0, β/δ 9.2,
  KR 10.9 decades. Absolute-rate-adjacent quantities are wholly unpinned — the
  data-starvation premise (SPEC §2), measured directly.

## D4 — null-spread pilot                                 [status: DONE — gate is NOT vacuous]

`scripts/diag_fft_d4.py` (FORBIDDEN — opens payload; AnswerKey discarded unread) →
`experiments/diag_fft/d4/results.json`. Pilot approximation, stated: the alternative
sign structures are the D5 seeds themselves (10 distinct structures fit to the same data
under identical budget by the baseline objective) — frozen-gate B_train refits are M1
machinery. Rollouts at 256² (grid caveat pending D2's fidelity table), D3's band edges.

- All 4 patterned alternatives land k*_sim within ~one radial bin of k*_obs
  (0.273–0.286 vs 0.284) — the fundamental does NOT discriminate mechanism.
- Their held-out-band spec_shape distances span **47 → 149** (B_harm alone 19.6 → 107.8,
  a 5.5× spread); train-band 6.4 → 41.4. Different wirings differ strongly in exactly
  the bands the Stage-0 fit will never see. **The held-out-band gate has discriminative
  content; it is not vacuous.** (The spread dwarfs D3's patch-to-patch estimation floor.)

---

## D6 — invariance checks                                 [status: DONE, both PASS]

`scripts/diag_fft_d6.py` (FORBIDDEN — opens payload; AnswerKey discarded unread),
`experiments/diag_fft/d6/results.json`.

**Check A — torch RAPS ≡ numpy `observables.raps`, bin-for-bin: PASS.** On the observed
frame (512², 363 bins) and 3 synthetic 256² fields: max per-bin |ΔP|/peak ≤ 6.1e-16,
|Δk*| ≤ 1.8e-15; bin edges/centers bit-identical.
**Finding F-D6-1 (carries into `losses/spectral.py`):** `torch.arange` with a float step
is 1 ulp off numpy's `np.arange` on 63/364 edges, which flips knife-edge lattice points
between bins. The torch implementation must build edges as
`torch.arange(ceil(kmax/dk)) * dk` (integer arange × step), which reproduces numpy's
`i*step` exactly. The residual ~1e-16 is FFT-backend difference only.

**Check B — time-rescaling degeneracy (f, D) → (c·f, c·D): PASS** (fixture = the D1
checkpoint). Analytic: σ_c(k) = c·σ(k) to 1.7e-12 with the argmax bin identical, for
c ∈ {0.1, 3, 10} (note: scaling f by c is not exactly expressible inside RNGRN's
softplus parameterisation, so the analytic check builds c·(J − k²D) directly; the D-half
via theta_D += log c matches to 3.2e-12). Dynamic: same IC, c-scaled reaction and D,
dt/c, same step count → max|ΔX|/std ≤ 2.5e-13, k* bin-identical. SPEC §9.4's invariant
combinations (D-ratios, α/δ, β/δ, K's, D·k*²/δ) confirmed invariant to ≤ 2e-16; absolute
rates carry the free factor exactly (measured factor = c to machine precision).
**R2's restriction to time-scale-invariant combinations is the right one.**
