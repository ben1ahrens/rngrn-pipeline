# DIAGNOSTICS — Fourier-space training, milestone M0

Results of diagnostics D1–D6 (`docs/PLAN_fourier_training.md` §3), run 2026-08-12 on
branch `feature/fft-training`. Written for a zero-context reader; every number names the
artefact behind it. Scripts live in `scripts/diag_fft_d*.py`; machine-readable results in
`experiments/diag_fft/d*/results*.json` (tracked). Payload-opening scripts are in
`tests/test_firewall.py::FORBIDDEN` per CLAUDE.md §5.

None of these numbers is a recovery result. M0 licenses exactly one sentence class:
"the machinery works and the knobs are measured" (PLAN §10).

---

## D1 — IFT feasibility (GO/NO-GO)                      [status: RUNNING — see run.log]

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

Measured at 96² (final verdict pending the FD loop):

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
Consequence for M1: the term is usable, but its gradient carries an O(grid) bias along
translations; if its FD check misses tolerance the options are (a) accept with the
measured bound, (b) compute moments from the band-limited spectrum side. Decision
deferred to the D1 verdict; not silently absorbed.

---

## D2 — forward-solve characterisation                   [status: script authored, full sweep pending]

`scripts/diag_fft_d2.py` (`--smoke` / `--full`), `experiments/diag_fft/d2/`.
Full sweep runs after D1 releases the machine (its timings need a quiet box), via
`bash scripts/guarded_run.sh .venv/bin/python -u scripts/diag_fft_d2.py --full`.

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

## D5 — baseline zero-call and seed spread               [status: pending]

## D4 — null-spread pilot                                 [status: pending, after D5]

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
