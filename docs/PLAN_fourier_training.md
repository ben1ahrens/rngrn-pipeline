# PLAN — Fourier-space training: stages, gates, diagnostics, changes

Written 2026-08-11, companion to `docs/SPEC_fourier_training.md` (the SPEC defines
every term, threshold, and boundary referenced here; this document sequences the
work). Audience: a reader with no context beyond the repo. Branch:
`feature/fft-training`. Nothing merges to `main` until the stage gates below pass
AND the owner has validated the results (CLAUDE.md §2).

House rules that bind every stage: no threshold/test weakening; no firewall
bypass; every science decision lands in `docs/DECISIONS.md` with date, evidence,
and what was rejected (§10); every number ships with its control (§8); trainer
invocations go through `scripts/guarded_run.sh` (§7a); process checks run with the
sandbox disabled (§7b).

First implementation step on this branch, before any code: write the DECISIONS.md
entries for the owner-made decisions this plan encodes (forward map, split, gates,
one-bin rule, composition, amplitude split, morphology deferral, QSS-first), so
the decision record does not lag the spec.

---

## 1. The staged progression

The pre-registered split (PREREGISTRATION.md: per canonical set, 2 tuning/burned
samples + 3 held-out; roles are payload attrs) shapes everything: iteration happens
ONLY on tuning-role samples; held-out samples are touched exactly once, at Stage 3,
with a frozen config. The canonical labyrinth set (`turing_labyrinth`, 5 samples,
3×512×512 single final frames, double_inhibitor topology) contains exactly two
measured labyrinths: `sample_0000` (role: tuning) and `sample_0004` (role:
held_out); the other three are measured holes (D-CANON-5 — the set is a mixture,
never treat it as one class).

| Stage | Data | Establishes | Explicitly does NOT establish |
|---|---|---|---|
| 0 | `turing_labyrinth/sample_0000`, within-sample split | The Fourier machinery fits and recovers in the most favourable setting; within-sample identifiability (null spread); calibrations (zero-call, band edges, ignition floor, seed spread) | ANY generalisation or recovery claim — held-out partitions of one homogeneous sample are not independent data |
| 1 | same sample, full spectrum | DoD-style gates pass as a rehearsal (Turing + stationary + one-bin k* + 5-seed reproducibility) on tuning data | Recovery (still the same burned sample) |
| 2 | remaining tuning-role canonical samples (read `role` attrs at execution time: the spots tuning samples, the holes tuning sample) + noise arms | Morphology diversity; robustness to parameter noise and observation noise; the deferred DECISION POINTS: morphology gating, amplitude-term demotion | Recovery (all tuning-role) |
| 3 | held-out canonical samples, one shot, frozen config — including `sample_0004`, the second labyrinth | **The stopping criterion.** This is the only stage that can satisfy it | — |
| 4+ | finite-μ lift milestone; full-set sweeps | Biological robustness of recovered circuits under the lifted dynamics | — |

Stage 3 discipline: config frozen and committed BEFORE the runs; one run set per
held-out sample; results reported whatever they are. A failed Stage 3 does not
un-burn the samples — that is what makes it evidence.

---

## 2. Stage 0 — exact definition

**Sample**: `turing_labyrinth/sample_0000` — tuning role (legitimate to iterate
on), highest contrast in the set (cv 0.81), p = 8 periods per box, L = 185.01,
512², all three channels observed (`observed_idx=(0,1,2)`, m = N — no latent
machinery). Chosen over `sample_0004` because using a held-out sample for
iteration would contaminate the pre-registered split. Known costs, accepted:
coarsest k* resolution in the set (one bin = 12.5% of k*), near-flat channels 1–2
(cv ≈ 0.075).

**Split** (all masks are loss-level config; the gate/`RecoveryInput` is untouched):
- FIT: species 0 only, spectral terms restricted to the training annulus
  `B_train = [b_lo, b_hi]·k*_obs` (provisional 0.5–1.5, fixed by D3 —
  *closed 2026-08-12 at 0.60/1.55*, DECISIONS.md D-FFT-9 closure 1).
  Normalisation pinned to `B_train` of channel 0 (SPEC §7).
- HELD OUT: (a) k-bands `B_low ∪ B_harm` on channel 0; (b) channels 1–2 entirely.
- Existing non-spectral terms run unchanged (they consume only `k*_obs` and
  `frame.mean()` — `frame.mean()` at Stage 0 is computed over channel 0 only, so
  the anchor cannot see held-out channels; noted because it is the kind of leak a
  within-sample split hides).

**Gates** (SPEC §9.5–9.7): held-out-band beat-all-nulls; channel amplitude-ratio
beat-all-nulls; patch consistency reported only. Plus the machinery bar: training
ignites (forward solve patterns) and completes without SteadyStateError storms.

**Null ensemble construction** (shared by both gates): collect the distinct sign
structures across the multistart restarts of the primary fit; add single-edge-flip
variants of the winning structure until ≥6 members; refit each with gate signs
frozen (θ_g clamped to the structure) on `B_train`, same budget, same seeds
protocol. Frozen-gate fitting is new but small machinery (a mask on θ_g updates).

**What would falsify Stage 0, and responses**:
- IFT gradient wrong or unstable (D1 fails) → the forward map is redesigned
  (fallback: truncated BPTT through the final relaxation steps only) — an owner
  conversation, since it was a joint architecture decision.
- Training never ignites on this sample → the linear-theory stage is the problem,
  not the spectral terms; tune existing weights on the tuning sample (legitimate)
  before touching the new machinery.
- Fail-no-spread on the null gate → within-sample identifiability is weak: the
  finding is reported, the gate is redesigned with the owner (e.g. add cross-band
  or channel constraints to the FIT), never lowered.
- Reproducibility clustering absent (10-seed spread continuous, no clusters) →
  R1/R2 calibration impossible; escalate to owner with the measured landscape —
  this would challenge the revised DoD itself.

---

## 3. Diagnostics BEFORE any training code

Each is a small script under `scripts/` (any that opens `payload.h5` goes into
`FORBIDDEN` at birth), run via `guarded_run.sh` where it trains anything, results
recorded in a tracked note + DECISIONS.md where they set a knob.

- **D1 — IFT feasibility (go/no-go for the forward map).** On a small grid
  (64–96²), for a known-Turing parameter set: solve the patterned state; build the
  matrix-free linearisation; verify the translational zero modes; verify
  `∂L/∂u ⊥` zero modes for each spectral term; compare the IFT/adjoint gradient
  against central finite differences on ~10 random θ-directions (agreement to a
  stated tolerance). Failure here stops the plan (see falsifiers).
- **D2 — forward-solve characterisation.** Cost and reliability of
  ETDRK4 pattern formation vs grid (96², 128², 256², 512²) and batch size, on CPU
  and GPU, under the memory guard; time-to-saturation distribution; the ignition
  amplitude floor (what amplitude separates "patterned" from "still relaxing");
  grid-fidelity check for §8 of the SPEC (does a cheaper validation grid reproduce
  512² k* and class calls on this sample). Sets: ignition floor, validation grid,
  the solver-acceleration targets (§7).
- **D3 — spectral estimator floor and band edges.** RAPS and 2-D block statistics
  of the observed frame: patch-to-patch spread (→ the patch-control yardstick),
  support of the spectrum around k* (→ `b_lo, b_hi`), harmonic band content at 2k*
  (is there measurable signal to hold out? if `B_harm` is empty at 512²/p=8, the
  held-out-band gate must lean on `B_low` — decide from measurement, record).
- **D4 — null-spread pilot.** Cheap frozen-gate fits of a few alternative sign
  structures; measure held-out-band spread. If the spread is zero the Stage-0 gate
  is vacuous and gets redesigned BEFORE the machinery is built around it.
- **D5 — zero-call and seed-spread measurement.** ≥10 seeds of the CURRENT
  (baseline) objective on sample_0000: recovered sign structures, magnitude
  distributions of present/absent edges (→ zero-call calibration, SPEC §9.3),
  parameter spread on invariant combinations (→ R2 calibration rule input, and the
  baseline number the spectral arm must beat to claim it improved robustness).
- **D6 — invariance checks.** Verify the time-rescaling degeneracy numerically
  (jointly scaling (f, D) leaves the pattern invariant) so R2's invariant
  combinations are the right ones; verify the torch RAPS matches
  `observables.raps` bin-for-bin.

D5 doubles as the baseline arm's Stage-0 record: it is the "current objective"
row every later comparison reads against.

---

## 4. Milestones and gates

Each milestone gates the next; a failed gate stops the line and the failure is
reported as a finding, not retried into silence.

- **M0 — diagnostics D1–D6 complete.** Gate: D1 passes; every knob D2–D6 was set
  has a DECISIONS.md entry. Falsified by: D1 gradient mismatch (→ owner
  conversation, forward-map redesign).
- **M1 — spectral machinery lands** (torch spectral module, new terms wired
  through the six touchpoints, ignition gating, ported 2-D comparator + drift
  test, firewall classifications updated, `test_losses` enumeration updated,
  batched path refuses new terms loudly until a batched form exists). Gate: full
  suite green (sandbox disabled), including new unit tests (gradient-flow and
  zero-location per new term — the repo's [VALIDATE] convention), firewall audit
  clean. Falsified by: firewall test cannot classify the new modules cleanly (→
  raise with owner before restructuring the test).
- **M2 — Stage 0 passes its gates** (§2). Falsifiers and responses as listed there.
- **M3 — Stage 1: DoD rehearsal on sample_0000.** Full-spectrum fit; gates F1–F3 +
  R1 (with calibrated zero-call) on 5 seeds; R2 reported with its calibrated
  threshold proposal. Falsified by: rollout of the recovered model fails F2/F3 —
  i.e. good spectral fit, wrong dynamics — which is the central scientific risk of
  spectral training; response: the ablation ladder (§5) isolates which term
  family is responsible; morphology/harmonic content of the failure is the
  evidence.
- **M4 — Stage 2: breadth + robustness.** Per-sample training on the remaining
  tuning-role canonical samples; noise arms (§6); the two deferred decision
  points are DECIDED here with the owner: morphology gating (now that rollouts
  exist in numbers), amplitude-term demotion under noise. Gate: M3-level gates
  hold on ≥ the spots tuning samples and the holes tuning sample, clean arms; noise
  arms reported against their clean controls.
- **M5 — Stage 3: the stopping criterion**, one shot, frozen config, held-out
  samples including `sample_0004`. Gate: Criteria 1 and 2 of the SPEC. This is
  the only milestone whose pass ends the project's iteration loop.
- **M6 — finite-μ lift milestone** (post-DoD evidence, per owner decision):
  spatial simulation of the lifted system with FUTURE_WORK.md's preconditions
  (dt-convergence; QSS-field reproduction at μ ≤ 1e-4; exact-linear gate update),
  reporting whether recovered circuits still pattern at literature-band μ.

---

## 5. Ablation ladder

Run at Stage 1 scale on sample_0000 (tuning-legal), each arm 5 seeds, all
reported against A0. The baseline arm is PERMANENT: it stays in the repo as a
config option indefinitely.

- **A0 — baseline**: the current default objective, verbatim (kstar + turing +
  anchor + anticollapse). The config exists today; it is frozen as
  `configs/baseline_linear.yaml` and never edited.
- **A1 — + spec_shape + real_moments** (gain-invariant additions only).
- **A2 — A1 + spec_aniso** (does 2-D structure help training, as it does scoring?).
- **A3 — A2 + spec_amp_mean + spec_amp_fluct** (the full Stage-0 objective).
- **A4 — spectral-only after ignition**: A3 but the k*-anchor weight ramps to 0
  once ignited (tests whether the spectrum subsumes the anchor — deferred design
  question, answered by measurement).
- **A5 — RAPS-only**: A3 with spec_aniso off — the in-loss test of the repo's
  54.3%/83.5% radial-vs-2-D finding.
- **A6 (optional) — resid-on**: the historical real-space pixel arm
  (exp06 measured it harmful; included only if capacity allows, as the true
  "real-space baseline" the ladder's name promises).

---

## 6. Noise-robustness arms (Stage 2)

Two separate axes, never conflated, clean control always alongside:

- **Parameter noise**: Gaussian perturbation of θ during optimisation (per-step
  σ_θ swept over a stated grid; SAM-style variant optional). Config-keyed
  (`train.param_noise`), off by default. Claim tested: does it widen the basin /
  improve 5-seed reproducibility? Read against D5's baseline spread.
- **Observation noise**: additive Gaussian background on the training frame at
  stated SNR levels (grid set from the frame's own contrast, recorded), applied
  to the FIT only — gates still evaluate against the clean frame. Claims tested:
  does training still ignite, pass F1–F3, and reproduce? Known interactions,
  stated in advance: white noise adds a flat power floor (band-limited shape
  terms partially absorb it); it invalidates the "noise-free ⇒ amplitude is clean
  signal" premise of the amplitude terms — this arm is where amplitude-term
  demotion toward the gain-invariant mode is decided.

---

## 7. Solver-acceleration workstream (stiff systems are slow)

Owner requirement. The forward solve is the new cost centre (rollouts measured
0.9–1.7 s/field at eval sizes; training needs one per ignited step per member).
Targets, measured in D2 and pursued as needed, in order of expected yield:

1. **Batched GPU ETDRK4** — batch over restarts/members × the existing
   half-spectrum integrator; the repo's measured direction (batched spectral ops
   favour GPU by up to 162×; the serial path favours CPU).
2. **Grid reduction with proven fidelity** (D2's grid-fidelity check gates any
   downsampled training grid).
3. **Warm-starting**: reuse the previous step's patterned state as IC for the next
   forward solve (θ moves little per Adam step); measure the speedup and verify it
   does not bias the reached branch.
4. **Early stopping tuned on the saturation detector** (existing [TUNE] knobs).
Gate for this workstream: forward-solve throughput sufficient to run the 5-seed
Stage-1 protocol within a session budget on this machine, number recorded when
measured. Memory: every training invocation through `scripts/guarded_run.sh`
(§7a is load-bearing; forward solves add per-member field storage — D2 measures
the RSS delta and the guard floor is revisited if needed, as a recorded decision).

---

## 8. Reporting and paper-grade figures (owner requirement)

- **Run records** (existing rails: run-index row, `train_results.json`,
  checkpoints, `arrays/*.npz` — all tracked): add per-run `w_<new keys>`, spectral
  gate values (train-band and held-out-band errors), null-ensemble results
  (per-member structure + error), ignition step, forward-solve counts/cost,
  rollout F1–F3 outcomes, morphology class (reported), ground-truth sign
  comparison, and the 5-seed cluster summary (sign agreement matrix, invariant-
  combination spreads). Learned parameters and sign structure are already
  reported (`RecoveryResult.params`, `topology`) — surfaced, not rebuilt.
- **Figures module** (new, scoring-side, under `rngrn/plotdata.py` extensions +
  a figures script; outputs under `experiments/figures_report/`, the tracked
  location): loss-term trajectories with ignition marked; predicted vs observed
  RAPS and 2-D spectra with train/held-out bands shaded; null-ensemble error
  distributions with the recovered model marked; 5-seed reproducibility cluster
  plots; optimisation-strategy view (weights/staging vs step); rollout panel
  (target frame | recovered rollout | spectra). These are the owner's paper
  diagnostics; they follow the existing plot-arrays schema conventions
  (ARRAY_SCHEMA_VERSION bump when columns change).

---

## 9. Module change map

New (recovery-side unless noted; every new module classified in
`tests/test_firewall.py` at birth — the completeness test enforces this inside
`src/rngrn/`):

- `src/rngrn/losses/spectral.py` — torch FFT, RAPS, band masks, the four new
  terms + `real_moments`; ignition gating helper. (Invasive: none — new file.)
- `src/rngrn/forward.py` (or `losses/pattern.py`) — the gradient-free forward
  solve wrapper around `eval/numerics` integrators + the IFT/adjoint solve.
  (New file; `eval/numerics.py` and `eval/rollout.py` are already RECOVERY_SIDE,
  so training may legally call them.)
- side-neutral port of the 2-D spectral comparator into `rngrn/utils.py` (+ drift
  test vs `scoring/morphology.spectral_distance_2d`).
- `scripts/diag_fft_*.py` — diagnostics D1–D6 (payload-openers added to
  `FORBIDDEN`).
- `configs/baseline_linear.yaml` (frozen A0), Stage-0 config, noise-arm configs.

Modified (touch-points, expected invasiveness):

- `losses/total.py` — wire new terms into `compute_terms`; batched path refuses
  them loudly until batched forms exist (mirror the resid refusal). (Moderate.)
- `src/rngrn/config.py` + `configs/base.yaml` — new weight keys + a
  `loss.spectral` section (bands, channels, ignition floor). (Small, additive,
  appended at the end of the dataclass per the anti-conflict convention.)
- `recover.py` — ignition detection, forward-solve invocation, spectral term_kw
  threading. (Moderate — the serial loop grows one stage; the batched loop only
  grows a loud refusal initially.)
- `train.py` / `export.py` — run-index columns, gate evaluations, null-ensemble
  driver, 5-seed protocol driver. (Moderate.)
- `plotdata.py` / figures — §8. (Moderate, additive.)
- `tests/` — `test_losses.py:279` enumeration + new-term unit tests;
  `test_firewall.py` classifications + FORBIDDEN additions; new tests for
  spectral module, IFT gradient (against finite differences, small grid), torch-
  vs-numpy RAPS, comparator drift, ignition gating, frozen-gate fitting.
  (Substantial, all additive except the enumeration line.)
- `CLAUDE.md` §7c — "training never simulates" amended in the same change that
  lands the forward solve; `TUNING.md` — new knobs tagged [TUNE]/[VALIDATE]/
  UNCALIBRATED at birth. (Small, mandatory.)

Untouched: `model.py` (dispersion, parameterisation), `observables.py` (numpy
reference measurement side), `data/` (gate, registry, solver), `scoring/`
(except none — the port COPIES arithmetic into utils, scoring keeps its own),
`eval/` internals (consumed, not modified), the pre-registration and its split.

---

## 10. What each stage's pass does and does not license saying

Per CLAUDE.md §8 and `.claude/rules/reporting-numbers.md`. Stage 0 pass: "the
Fourier machinery fits and recovers within one sample; the within-sample gates
have discriminative content" — never "recovery achieves X". Stage 1 pass: "the
DoD gates pass on tuning data". Stage 2: robustness statements, each against its
clean control. Only Stage 3 licenses the stopping-criterion sentence, and it
names `sample_0004` and the other held-out samples explicitly. Any quoted
`kstar_fft_rel_err` from a spectrally-trained run carries the circularity caveat
(SPEC §7) in the same sentence.
