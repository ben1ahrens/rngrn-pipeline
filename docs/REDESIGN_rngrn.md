# REDESIGN_rngrn — a clean-slate design for robust GRN recovery, verified by the dynamical lift

**Status: RATIFIED DESIGN (2026-08-17).** Authored as a proposal, then ratified the
same day: the owner reviewed it interactively and explicitly delegated the §8
owner-decision register (*"you decide on the Owner-decision register, I trust you"*).
The rulings live in `docs/DECISIONS.md` (D-LIFT-1, D-REDESIGN-1) and the
preregistration amendments in `docs/PREREGISTRATION.md` (§3.7 added; §3.4 amendment
note), all dated before any run they judge. UNCALIBRATED numbers remain UNCALIBRATED —
ratification fixed the rules, not the values.
Every number carried over from prior work is traced to a run directory, a document
section, or a source file. Numbers *introduced* here are design choices: each is marked
UNCALIBRATED at its point of use and listed in §8 item 14 with its calibration rule.

Written 2026-08-17 on branch `docs/redesign-rngrn`, from the codebase state of
`chore/fft-refactor` (10cff1b), and revised against a full evidence audit of the same
date (every quantitative claim below was checked against its cited source; the audit's
corrections are incorporated). Owner constraints set at plan review, on record:

1. The model stays a **mechanistic GRN** — every learned parameter is a biological
   quantity. No black-box function approximators anywhere.
2. **Fixing edge signs is not an option.** The sign structure stays learnable; no
   hard-masked topology enumeration.
3. The first campaign targets **one 3-node system**, not the corpus.
4. Training is **GPU-optimised**, and the design must answer whether we can
   **backpropagate through the network** (it can — two ways; §4.2).

---

## 0. Thesis

Linear theory ignites, the simulated spectrum discriminates, the population is the null
ensemble, constraints move from soft loss terms into the model's construction, and the
finite-μ dynamical lift — once it has climbed a validation ladder — becomes the
verification gate.

The recovered object is not a point estimate. It is the modal sign structure among the
scored seeds, plus the measured distributions of time-scale-invariant parameter
combinations across the gate-passers. The redesign's central, testable bet is that
simulation-based spectral losses collapse a sign-structure multiplicity that
dispersion-only training has **no measured means of collapsing**: the current objective
is provably blind to a 2-parameter continuous gauge plus transpose and permutation
(D-C1-GAUGE), and quotienting by that whole group was measured *not* to explain the
multiplicity (0.125 → 0.375 against a 0.75 bar) — the seeds found different networks,
and nothing in a σ(k)-only objective distinguishes them.

## 1. Scope: one 3-node target first

Target: **`turing_labyrinth/sample_0000`** (competitive form, N=3, 512², cv=0.81, p=8,
L=185.01 — the Stage-0/Stage-1 tuning sample of `docs/PLAN_fourier_training.md` §1-2).
Chosen because it is already burned for tuning, so no held-out sample is consumed, and
because the baseline to beat was measured on exactly this sample:

> **D5 baseline** (`experiments/diag_fft/d5/target_reports.jsonl`, git_sha 1b9ecda,
> 2026-08-12, 10 seeds, current default objective): 4/10 seeds Turing-unstable;
> 10/10 distinct recovered sign structures (0/45 pairwise agreements,
> `mean_agreement` 0.499 ≈ the 0.5 expected under independent ±1 signs, given the row's
> 5% zero-call rate); `kstar_fft_rel_err` median 0.977; `topology_consistency` 0.1;
> recovered α and δ inside the plausibility box in 0/10 seeds; recovered d_ratio
> in-box in 10/10 (see §2 on what that asymmetry means).

**Comparability, stated up front.** Three D5 columns are comparable to the redesign arm
and are the ones R2/R4 are read against: `turing_frac`, `n_distinct_structures`, and
`kstar_fft_rel_err`. **Not comparable, and never to be reported as an improvement over
D5:** the α/δ plausibility columns (structural by §3.3), any spectral-target-derived
quantity (single frame vs the N-frame average of §4.6), and `topology_consistency`
unless computed over the same population as D5's (§3.4). Breadth — other samples, the
`nc1` form, noise arms, held-out one-shot — is a later stage of the same design (§7)
and is not on the first campaign's critical path.

## 2. Why a redesign: the measured failure inventory

Each item names its evidence. Together they say the current objective cannot identify a
network, and the machinery built to fix it (the M1 spectral path) has one open
structural defect of its own.

- **The objective is gauge-blind by construction** (`docs/DECISIONS.md` D-C1-GAUGE).
  On the region the optimiser occupies, the default loss is a function of
  σ(k) = max Re eig(J − k²D) alone, hence exactly invariant under transpose, diagonal
  similarity (a 2-parameter continuous gauge at N=3), and node permutation. Quotienting
  the reproducibility statistic by that whole group rescues 0.125 → 0.375 at best,
  against a 0.75 bar — and D-C1-GAUGE records the negative explicitly: the observed
  sign-structure multiplicity is *not* gauge-orbit multiplicity. The seeds found
  different networks.
- **Data starvation is measured, not hypothesised** (`docs/SPEC_fourier_training.md` §2;
  D5 above). ~3N²+3N = 36 parameters at N=3, constrained by essentially two scalars
  (k*_obs and frame.mean()) plus priors. D5's invariant-combination log-spreads run
  from 0.44 decades (d_ratio, Turing subset n=4; 1.70 over all 10 seeds) to 10.9
  decades (KR, Turing subset).
- **The k*-anchor has a degenerate minimum** (`docs/STATE_OF_THE_SCIENCE.md` §2.4,
  exp09). While a model is not yet Turing-unstable, σ peaks at k=0 and the anchor loss
  can be reduced by *flattening* σ — killing D-separation and ‖J‖ — rather than moving
  its peak. Held-out generalisation: 2/6 samples ever succeed; 0/8 seeds on the other
  four; the best-anchored failing sample still fails 0/8.
- **The strict-convergence premise of the M1 gradient fails at the training geometry**
  (`docs/DIAGNOSTICS_fft.md` F-D1-5, NEGATIVE, 2026-08-13). On the data box (L=185.01,
  ~5.2 periods of the fixture pattern) the Newton polish wall-caps at residuals of
  order 1e-5 at every grid in {96,128,192,256}, against the 1e-9 bar. F-D1-5's
  unifying pattern — recorded there **as a hypothesis, stated as such** — is that every
  convergent solve to date used a commensurate box. Because θ varies continuously
  during training, incommensurability is the *generic* condition. Flagged in the
  diagnostics as a PLAN-§2-falsifier-class finding awaiting an owner decision; §4.3 is
  this design's proposed resolution.
- **Hidden-channel recovery has no objective** (`docs/IDENTIFIABILITY_EXPERIMENTS.md`,
  `TUNING.md`). `stationarity_residual` is measured harmful — all nine swept cells
  collapsed the Turing rate to 1/8 seeds vs 5/10 with the term off, at best-median k*
  error 11.8% vs 0.4% (`losses/terms.py` note above DEFAULT_WEIGHTS; STATE_OF_THE_
  SCIENCE §3) — yet is required for m<N; there is no known-good weight.
- **The biological prior is inert where it is needed and unnecessary where it is not.**
  `param_prior` defaults to weight 0 (`losses/terms.py::DEFAULT_WEIGHTS`). At that
  weight D5 measured α 0/10 and δ 0/10 in-box — but d_ratio **10/10 in-box** (and β
  correctly unscored, its box row being UNCITED). So the two rows §3.3 hard-boxes are
  the two that fail, and the row it keeps as a soft prior already passes without one:
  that prior's purpose is to hold the recovered ratio near the literature value 7.5
  against the generator's ~135 — making the viability-vs-recovery tension measurable —
  not to fix a measured failure.
- **The dynamical lift is written but unvalidated, and deferred.** `eval/lifted.py`
  implements the finite-μ system (promoter occupancies GA, GR as explicit fast state,
  dim N+2N² = 21 at N=3, only x diffuses, fixed points μ-independent) and even a
  stiff-safe integrator (§5.2) — but it has never been validated in space or at finite
  μ, and D-FFT-4 deliberately kept it off the verification path. The one measured
  μ-scan (`eval/lifted.py` §"Measured", harvest survivor nc1/mobile3 draw 0 — n=1, the
  only one on record) shows a **re-entrant Turing band**: strictly Turing for μ ≤ 1e-2,
  lost over μ ~ 1e-1..1e1 via a Hopf/oscillatory route, Turing again for μ ≥ 1e2. On
  that single draw the biological band μ ∈ [1.1e-5, 9.2e-3] (central 7.2e-4;
  `eval/lifted.py` MU_BIO_*; Chen et al. 2014 TF residence times, Müller et al. 2012
  protein half-lives) sits entirely inside the first Turing window — weak evidence
  *for* safety at biological μ, not against it. What the scan does establish is that
  the verdict is **non-monotone in μ**, so single-point extrapolation is unsound and
  the band must be scanned. Whether QSS-Turing implies lifted-Turing at biological μ
  across a population is UNMEASURED — it is precisely what §5.3's V4 measures. The
  owner's brief for this redesign makes the lift the verification vehicle — that
  reversal is proposed explicitly in §5 and §8, not assumed.

What the redesign keeps faith with: the data firewall (§5 of `CLAUDE.md`; recovery never
sees ground truth), the preregistration discipline (bars frozen before results), and the
evidence rules (`.claude/rules/reporting-numbers.md`).

## 3. Model and identifiability

### 3.1 The mechanistic core survives; the sign structure stays learned

The gated-promoter parameterization of `model.py` survives as-is: shared binding budget
s = softplus(θ_s), gate g = sigmoid(θ_g), KA = s·g, KR = s·(1−g) (so KA+KR = s exactly),
production α = softplus, degradation δ = softplus, diffusivity D = exp, both regulation
forms (competitive Shea-Ackers and nc1), Hill n=2, autograd-only Jacobian, and the
closed-form cubic dispersion backend (N=3, measured 162× over `torch.linalg.eigvals` on
CUDA — `model.py` docstring). Per owner constraint 2, **no edge masking and no fixed
sign patterns**: discrete sign multiplicity is handled by the objective (§4) and by
ensemble reporting (§3.4), never by restricting the hypothesis class.

### 3.2 Gauge fixed by construction, not by post-hoc canonicalization

D-C1-GAUGE measured that quotienting after the fact does not work. The redesign removes
the gauges where they live:

- **Concentration gauge (diagonal similarity).** Pin the model's homogeneous fixed point
  to the data: for observed channels x*_i ≡ mean of the observed frame channel; for
  hidden channels x*_i ≡ 1 by convention. Implemented by *solving β out of the
  steady-state condition*: β_i = δ_i·x*_i − prod_i(x*), with a hinge penalty enforcing
  β_i ≥ 0 and gate-time rejection of violators. Four effects at once: the 2-parameter
  continuous gauge is gone (all N scales pinned); the amplitude channel becomes a hard
  constraint instead of the never-swept `frame_scale_anchor`; N parameters (β) are
  eliminated (36 → 33 at N=3); and the per-step Newton solve for x* — with its
  convergence fragility and its role in the σ-flattening escape route — leaves the
  training loop (the fixed point is *given*; the solver survives in scaffolding
  checks). The β ≥ 0 hinge is a mechanistic constraint (basal production cannot be
  negative), not a tuning knob.

  **The estimator's measured properties bound what this can claim.** `frame.mean()`
  tracks the true x*_0 at Spearman ρ = 0.95 but with ratio median **0.921**
  (IQR 0.796–1.036, n=127 `three_gene` samples — exp12, `docs/STATE_OF_THE_SCIENCE.md`
  §2.8), which `TUNING.md` records as making it "a scale regulariser, **not a fit**".
  Pinning x* to it therefore imports a systematic ~8% scale error into the model's
  fixed point, measured on a *different dataset family* than the target. The
  estimator's bias on `turing_labyrinth/sample_0000` is **UNMEASURED and must be
  measured before R2** — otherwise the β-hinge feasibility test and every downstream
  rate inherit an uncharacterised offset.
- **Time gauge.** (f, D) → (c·f, c·D) is exactly invariant from a single static frame —
  verified at 1.7e-12 analytically and 2.5e-13 dynamically, with SPEC §9.4's invariant
  combinations unchanged to ≤2e-16 (`docs/DIAGNOSTICS_fft.md` D6 Check B). Fix by
  convention: geomean(δ) ≡ 1; all reported rates are time-scale-invariant combinations
  (the R2 restriction already recorded in `docs/SPEC_fourier_training.md` §1). The only
  data that could pin absolute rates is transient-pair imaging — designed, deferred
  (§4.6).
- **Permutation.** Observed channels carry their data identity; hidden species (later
  m<N arms) ordered by descending D. Trivial once stated.
- **Transpose.** Not a symmetry of the mechanistic model — there is no parameter map
  realising Jᵀ in general — only of σ(k)-based objectives (spec(Mᵀ) = spec(M)). It is
  therefore broken *objective-side*, by loss terms that depend on eigenvectors and
  nonlinear saturation, i.e. the simulated-spectrum terms of §4.4. Any run trained with
  σ-only losses must report the transpose partner as an explicit unresolved 2-fold
  ambiguity.

### 3.3 Biological boxes as construction, priors where the box would lie

- **α and δ: hard box-sigmoid reparameterization.** θ → low + (high−low)·sigmoid(raw)
  with the cited working box (`configs/bio_box.yaml`: α ∈ [0.5, 10], δ ∈ [0.4, 5],
  provenance `docs/GOAL_tica_equivalent.md` §2.2 — the project's own recorded working
  box, and the doc must keep saying so). Plausibility for α, δ becomes 1.0 **by
  construction — therefore not a result, and never to be reported against D5's
  measured 0/10**; R4 reports these columns as `structural`, not as numbers. This
  changes preregistration §3.4's α/δ component from a measured outcome into a
  structural truth — prereg-touching, owner-decision item 3.
- **D-ratio: soft prior, not a hard box.** The box row is [1, 60], centre 7.5
  (Nodal/Lefty), while the generator population's D-ratio median is ~135 — *outside
  the box by recorded owner decision* (`configs/bio_box.yaml` header: priors are
  centred on biologically viable literature values, not on the synthetic data). A hard
  box would make the synthetic target itself unrepresentable. Keep the existing
  `param_prior` log-normal on the d_ratio of the two most mobile species (centre 7.5,
  spread 1.0 [TUNE]) at nonzero weight — promoted from its current weight-0 default,
  with §2's caveat: this row already passes 10/10 at weight 0, so the prior's role is
  to make the viability tension measurable, not to fix a failure.
- **`anticollapse` is retired in the redesign arm** on the structural argument that
  box-confined rates cannot let ‖J‖ collapse to zero. **Note what is and is not
  measured:** `anticollapse=0.5` is the shipped default (`losses/terms.py::
  DEFAULT_WEIGHTS`) and was live in the D5 baseline; the measured-rejected result is
  *strengthening* it to 2.0 (6/6 → 2/6, STATE_OF_THE_SCIENCE §2.4/§3), not removing
  it. Removal is an untested change to the objective the D5 row was measured under —
  owner-decision item 11, with the A0 arm keeping it at 0.5 as the control.

### 3.4 The deliverable is an ensemble

Following the recorded Kho et al. reading (stochasticity as a feature, providing
multiple parameter alternatives — `docs/STATE_OF_THE_SCIENCE.md` §2.7), the recovery
deliverable on the target is two-level:

1. **The claim:** the modal recovered sign structure, scored by `topology_consistency`
   **over the same population D5 used — every seed that produced a scored recovery,
   ignited or not — so the number is read against D5's 0.1.** A gate-passers-only
   variant is reported alongside as a conditional statistic, labelled as such.
   **Neither is a preregistration §3.1 verdict:** §3.1 requires independent seeds and
   the cross-target control, and §4.5's diversity cull makes the Phase-II population
   non-independent by construction — reconciling the population loop with §3.1/R1's
   independent-runs requirement is owner-decision item 13. The scorer
   (`scoring/reproducibility.py`) carries `Status: SCAFFOLD` and says so.
2. **The uncertainty:** the gate-passing seeds reported as distributions (median +
   decade spread) of a named invariant set: the SPEC §9.4 combinations d_ratio, α/δ,
   β/δ (β now derived, §3.2), and D·k*²/δ, plus per-edge promoter occupancy at the
   operating point KA_ij·x*ⱼⁿ/(1 + …) ∈ [0,1] as the bounded companion to KA/KR.
   **KA and KR themselves — the two worst D5 spreads, 8.0 and 10.9 decades — are
   untouched by anything in §3.3 and are still reported in raw decades**; substituting
   occupancy alone would remove exactly the worst numbers from view.

## 4. Training objective and optimization

### 4.1 GPU-first (owner directive)

The training loop is designed to be resident on GPU and batched across the restart
population. What exists and what is new work, stated plainly:

- **Already batched:** the ignition-phase terms (`compute_terms_batched` covers
  `kstar_anchor`, `turing_hinges{,_split}`, `anticollapse`, `frame_scale_anchor`,
  `param_prior`) on `BatchedRNGRN`, with the closed-form cubic dispersion backend
  (162× over eigvals on CUDA; `model.py`).
- **Batched since R3 Phase B** *(amended 2026-09-01, R3 Task 21; this bullet
  originally read "Not batched, and refused by name" — that described the pre-R3 tree)*:
  the spectral terms and the forward solve are batched
  (`losses/total.py::compute_terms_batched`, `forward.BatchedPatternSolver`), and the
  by-name refusal is replaced by input validation (the `resid` term's batched refusal
  survives unchanged — the batched reaction still has no per-pixel form;
  `losses/total.py:326-335`). The mandate in this section — a
  bit-level equivalence test before the batched path trains anything — is satisfied by
  Phase A's equivalence suite. One refusal is deliberately retained (D-R3-5): the
  promoted default `gradient_path='unrolled'` is serial-only (`unrolled_relax` has no
  batched twin), so `batched=True` with a non-zero spectral weight REFUSES; a batched
  spectral run must select `gradient_path='adjoint'` explicitly, with the demotion
  acknowledged. Measured raw material behind the batching: ETDRK4 at 3.25 ms/step at
  512² GPU-native (D-FFT-11 flag-1 closure) and 11.6× over serial CPU at 96²/B=32
  (`docs/DIAGNOSTICS_fft.md` D2). CPU forward solves are disqualified for training
  (938–1374 s/solve at 96², `forward.py` docstring).
- Population sizes are set from measured GPU throughput and RSS (§4.5), under the
  `CLAUDE.md` §7a host-RAM guard, which still governs every launch.
- Serial per-member CPU paths survive as reference implementations; every batched twin
  — the batched spectral/forward path included, since Phase A — carries a bit-level
  equivalence test against its serial reference. *(Amended 2026-09-01: this bullet
  originally deferred the spectral path's test to future work.)*

### 4.2 Backpropagation through the network: two gradient paths

The owner asked whether we can backpropagate through the network. Yes — in two distinct
senses, and the redesign uses both:

- **Implicit (adjoint/IFT) — primary.** The existing D-FFT-10 path in `forward.py` *is*
  backpropagation through the simulated network at its patterned steady state:
  dL/dθ = −λᵀ ∂F/∂θ with one minimal-norm LSMR adjoint solve per backward pass. Exact,
  O(1) memory, verified by D1 (worst FD rel. err. 6.4e-8 over the spectral terms,
  8.1e-8 including `real_moments`, vs tol 1e-4 at 96²). Its premise is strict
  convergence F(u*) = 0 — which F-D1-5 measured as generically unreachable on
  incommensurate boxes. And convergence genuinely bounds nothing loosely: D-FFT-10
  measured that a projected-Krylov adjoint whose *true* residual stalled at
  5.5e-4–5.7e-3 biased gradients by **1e-5–5.5e-2 relative** — up to an order of
  magnitude *above* its own residual, not merely at its order.
- **Unrolled/truncated backprop through the GPU integrator — the structural fallback.**
  Run the ETDRK4 relax as differentiable torch ops and backpropagate through the
  unrolled steps with gradient checkpointing. This requires **no convergence at all**:
  the loss is defined on the relaxed field after finite time, so the F-D1-5 stall
  class cannot touch it. Costs, stated plainly: checkpointed activation memory
  (bounded by segment length × fields, tractable at the n=96 training grid).
  *(AMENDED 2026-09-01 per D-R3-3, R3 Task 12: this bullet originally added "gradients
  through the exponential-growth phase of the instability can explode — mitigated by
  truncated backprop". That explosion did NOT reproduce — no segment length in
  [1, 2048] blew up on either arm at the measured operating point; the patterned
  attractor is contracting. What truncation actually buys there is bounded activation
  memory — 24,576 B/step checkpointed vs 753 kB/step un-checkpointed, 30× — and a
  gradient that stays finite-difference-verifiable; it does not buy protection from an
  observed divergence. The segment length is no longer UNCALIBRATED: 128 steps,
  CALIBRATED inside the saturated-warm-state regime and UNCALIBRATED outside it,
  D-R3-2. Scope: one fixture, one box, two placeholder losses — not a claim that the
  unrolled path cannot explode in general.)* Feasible only because of §4.1.
- **A/B discipline.** Before either path feeds a training run at the operating point, a
  D1-style finite-difference check (10 directions × the active loss terms, tol 1e-4)
  runs on *both* paths at the same θ. During training: adjoint where the Newton polish
  converges to the 1e-9 bar, truncated-unrolled where it stalls. Promotion of the
  unrolled path to primary — if the A/B shows it FD-faithful at training tolerances —
  changes the gradient estimator the results depend on: owner-decision item 8.
  *(AMENDED 2026-09-01: item 8 was ruled by the owner on 2026-08-19 — the unrolled path
  IS promoted to primary/default for every member, and the adjoint path is retained as
  the A/B verification path, not as a fallback. The "primary"/"structural fallback"
  labels above describe the pre-promotion design; D-R3-5 carries the ruling and its
  evidence, including the A0 bit-identity clause.)*

### 4.3 The forward-solve geometry: adaptive commensurate solve box (F-D1-5 resolution)

Adopt F-D1-5 option (a) with the (d) verification clause, taking the commensurability
pattern as the working hypothesis it is recorded as (§2):

- Training solves run on a box **commensurate with the model's current pattern**:
  L_solve = p·2π/k̂(θ) with p = 8 (the nearest integer to D3's measured 8.36 data
  periods, and D1's verified 8-period geometry) and n = 96 (12 px/wavelength, D1's
  verified density *for the gradient*). k̂(θ) is the current dispersion argmax,
  **detached** — solve-box geometry is not differentiated.
- **Stated plainly as a gap — the training-grid spectral fidelity is unmeasured.** D2
  measured band-limited spectral quantities agreeing with 512² at **256², not at 96²**,
  where k* one-bin agreement fails 2/6 (`docs/DIAGNOSTICS_fft.md` D2). Training
  therefore compares a 96² model spectrum to a 512² observed spectrum with the
  model-side estimator's fidelity UNMEASURED at this grid. R3 measures the
  96²-vs-512² RAPS band distance on the fixture before any spectral gradient is
  believed; if it does not clear D3's ~31%/bin estimation floor, n rises to the
  coarsest grid that does.
- **Scale-free comparison:** observed log-RAPS targets are interpolated onto the solve
  box's bin centres expressed in k/k*_obs units, once per re-tile. Interpolation error
  is *expected* far below the ~31%/bin floor; it is **UNMEASURED** and is measured once
  per re-tile geometry at R3, alongside the FD check, before any spectral gradient is
  consumed. Band masks stay defined relative to k*_obs exactly as in
  `docs/SPEC_fourier_training.md` §4.
- **Re-tile hysteresis:** re-tile only when k̂ drifts more than half a solve-box bin
  (|Δk̂|/k̂ > 1/(2p)); on re-tile, rebuild the k-grid, clear warm starts, re-interpolate
  targets.
- **Verification gate before any training consumes these gradients:** the §4.2 FD check
  at the operating point — commensurate box, training grid, training-geometry θ.
  Commensurability is proven neither necessary nor sufficient (the sat512 defect-mode
  stall channel is independent), which is exactly why this is a gate, not an
  assumption.
- **Stall accounting:** a per-run stall-rate counter. If more than ~20% of
  ignited-member solves miss the 1e-9 Newton bar on the commensurate box
  (**UNCALIBRATED** — no baseline exists; calibrate at R3 from the measured stall-rate
  distribution, with the switch fraction chosen against the measured gradient-error
  difference between the two paths, not for convenience), those members switch to the
  truncated-unrolled gradient path (§4.2) rather than loosening the convergence bar —
  F-D1-5 option (b), which touches a pre-registered-class threshold, stays off the
  table unless the owner takes it explicitly.
- **Validation never uses the adaptive box.** All gate evaluations run at 512² on the
  data box: no cheaper grid is licensed (morphology class flips 2/3 seeds at 256² —
  D-FFT-9 closure 3).

### 4.4 The objective

**Ignition terms (always on; guard-rail weight after ignition):**

- `kstar_si` — a **new** scale-invariant k*-anchor replacing `kstar_anchor` in the
  redesign arm (the old key survives verbatim for the baseline arm A0;
  replacement is owner-decision item 11):
  σ̄(k) = σ(k) / (max σ − min σ + ε), loss = lse(temp·σ̄)/temp − σ̄(k*_obs).
  Invariant under σ → c·σ up to the ε regulariser (exactly invariant at ε = 0), so the
  measured flattening escape (§2) is not profitable; only relocating the argmax
  reduces the loss. The ε-degenerate direction (range → 0) is closed by the split
  Turing hinges, which floor the range at twice their margin whenever active.
  **ε and `temp` are UNCALIBRATED**: `temp` starts at `kstar_anchor`'s inherited 60.0
  (`losses/terms.py`, itself never swept), ε at the smallest value keeping the gradient
  finite while the hinges are inactive; both are swept at R2 — and the claim that
  flattening is never profitable is a design argument, verified numerically at R2, not
  a measured result.
- `turing_hinges_split` — form unchanged (disjoint k-support; stability at k → 0,
  instability at k > 0). **Separately: `k_min_frac`'s grid-relative definition is
  proposed to become k*-relative.** `TUNING.md` records that this *changes what the
  term means* and requires re-measurement (the current grid puts the floor at
  0.822·k*_obs against exp05's 0.698·k*_obs for the same nominal setting) —
  owner-decision item 12, with the A0 arm keeping the grid-relative form as control.
- `param_prior` — promoted to nonzero weight, with the §3.3 split (hard α/δ boxes in
  the parameterization; soft D-ratio prior in the loss).
- Retired in the redesign arm: `anticollapse` (§3.3 — owner-decision item 11),
  `frame_scale_anchor` (§3.2), `stationarity_residual` and its latent-fields machinery
  (measured harmful; §2). `morphology_consistency` is already inert (not in
  `compute_terms`; STATE_OF_THE_SCIENCE §4) — removed from the config for clarity, no
  behaviour change.

**Spectral terms (primary once ignited):** the five M1 terms of `losses/spectral.py`,
unchanged in form — `spec_shape` (gain-invariant log-RAPS band shape on
B_train = [0.60, 1.55]·k*_obs, D-FFT-9 closure 1), `spec_aniso` (2-D log-power block),
`spec_amp_mean`, `spec_amp_fluct`, `real_moments` — computed against the §4.3 solve,
per-member ignition-gated with the existing omitted-never-zeroed semantics (per-member
`spectral_skipped`; note this gating exists today only on the serial path — §4.1). One
change of *role*, not form: once a member ignites, the spectral terms are **primary**
and the ignition terms drop to a guard-rail weight (~10% of their ignition weights —
UNCALIBRATED, swept at the first campaign). This amends the SPEC §3 decision that every
existing term stays on unchanged for the whole run — owner-decision item 5. Rationale:
the linear-theory objective as primary carrier is the measured D5 failure; the
mechanism-discriminating information lives in the simulated spectrum — alternative
wirings matching the fundamental k* still separate 47→149 in held-out-band spec_shape
distance (3.2×; 5.5× on B_harm alone, 19.6→107.8 — D4), against the ~31%/bin
estimation floor (D3). One addition contingent on §4.6: per-bin inverse-variance
weights in `spec_shape`, measured from the multi-realization frames. Held-out bands
(B_low ∪ B_harm) and held-out channels are never trained on and never used for
selection — they remain the Stage-0 gates of SPEC §9.

**The transpose note:** only the spectral terms break the transpose degeneracy (§3.2).
A member that never ignites is reported with its transpose partner as an unresolved
ambiguity.

### 4.5 Population strategy: ignite wide, cull to diversity, refine narrow

- **Phase I — ignition.** B members batched (`BatchedRNGRN`), ignition terms only,
  ~1500 Adam steps (lr 0.05, per-member grad clipping — existing machinery).
  **B = 512 is a starting point, UNCALIBRATED** — the largest measured batch is B=32
  at 96² (`docs/DIAGNOSTICS_fft.md` D2); B is set at R2 from a measured
  throughput-and-RSS curve under the §7a guard. On ignition yield: D5 reached Turing
  in 4/10 *seeds* at 4 restarts each — a per-seed, best-of-4 rate under the *old*
  objective — so it bounds neither the per-member ignition probability nor the
  redesign arm's; the per-member rate is one of R2's deliverables. Width beats depth:
  the hit rate saturates by ~4000 steps (`docs/STATE_OF_THE_SCIENCE.md` §3 — measured
  on `three_gene_val/sample_0000`, 16 seeds, a different sample family, and the source
  flags its cross-budget k* comparison as not like-for-like).
- **Cull — diversity-aware top-K.** Rank ignited members by ignition loss; keep at
  most one member per distinct recovered sign structure, best-first. **K = 16 is
  UNCALIBRATED** — set at R2 from the measured distinct-structure count (the cull
  cannot keep more members than there are distinct structures).
  **The culled runners-up are a *candidate pool* for the SPEC §9.5 null ensemble, not
  the null itself.** SPEC §9.5 (BINDING at Stage 0) requires each alternative sign
  structure *refit to B_train with gates frozen to that structure under the identical
  budget*, plus single-edge-flip variants of the winner. Substituting unrefit pre-cull
  members would weaken a binding gate — owner-decision item 10; the design's default
  is to *implement §9.5 as written*, seeding the refits from the candidate pool.
- **Phase II — spectral refinement.** The K survivors train with spectral terms
  primary (§4.4), forward relaxes across members on GPU (the batched implementation is
  R3 work — §4.1), warm-started, Newton polish + adjoint per §4.2/§4.3. Budget ~2000
  steps, lr 0.02 (both UNCALIBRATED, swept). A de-ignited member falls back to
  guard-rail terms — the existing semantics.
- **Phase III — LBFGS polish, ablation arm only.** Existing `lbfgs_steps` knob on the
  winner, ignition gate frozen, abort on de-ignition. **No measurement of
  `lbfgs_steps` exists in this repo** (every recorded run sets it to 0), so nothing
  rests on it. (STATE §3's "continuation had no effect" result concerns warm-starting
  the Newton *steady-state solve*, not LBFGS.)
- **Selection:** winner and gate-passers chosen on fitted-band loss only; held-out
  bands are consulted exactly once, by scoring code, at gate time.

### 4.6 Data protocol (owner decision; small-box extension accepted by the owner 2026-08-17)

**Training frames: few and large.** Adopt **N = 8 independent realization frames** of
the *same* target system (same generator parameters, independent IC noise seeds, each
run to steady state at 512², p = 8 — a generator-side change only;
`scripts/gen_tg3.py` already draws per-sample `sim_seed`s), **split 6 train / 2
held-out over realizations**. Firewall-legal: all frames are observation-side. The
training 6 give a direct per-bin variance estimate for the spectral targets (replacing
the single ~31%/bin patch-quadrant floor), inverse-variance weighting in `spec_shape`,
and a √R-tighter averaged target. The held-out 2 are the **realization-consistency
check**: the fitted model's spectra must match them as well as the trained
realizations — an honest overfitting axis that spatial splits cannot provide (SPEC §1:
a Turing pattern is statistically homogeneous, so within-frame splits carry no
independent evidence; independent IC realizations do). Read honestly, a pass here says
the fit captured the *system's statistics* rather than one realization's defect
layout — it is not evidence of recovery, and does not upgrade R4's licensed sentence.
The 6/2 split and N = 8 are starting choices, UNCALIBRATED; the measured per-bin
variance calibrates both.

**Validation frames: many and small, held out entirely.** Additionally generate the
same target system at **small boxes, p ∈ {2, 3, 4}**, S seeds each (S UNCALIBRATED,
starting at 8), 64²–128² (≥16 px/λ, above the measured 6 px/λ floor). These are
**never trained on**: at p = 2–3 the fitted band holds only ~2–3 RAPS bins (bin width
k*/p) and each realization snaps to a coarse mode lattice — the regime D-CANON-2
measured as artifact-dominated (stripes largely a small-box artifact, vanishing at
p ≥ 11). That box-dominance is exactly what makes them *validation* assets, two ways:

1. **Cross-L transfer on the canonical target** (§3.5a's logic, previously testable
   only on legacy `three_gene_multiL`): simulate the *recovered* model at the same
   small boxes, no refit, and compare physical-k* invariance and the periods-per-box
   slope against the generator's own held-out small-box realizations — the
   mode-quantisation stress test that makes cross-L transfer non-vacuous.
2. **Mode-selection distribution** (reported, never gated): across S seeds at small p
   the generator produces a distribution over which lattice mode wins; the recovered
   model, run identically, should reproduce it. This is sensitive to the dispersion
   peak's shape and nonlinear saturation — but it is a **novel, unvalidated
   estimator**, so it enters report-only and requires its own decision entry before it
   could ever bind.

**Spectral estimator policy (added 2026-08-17, owner-accepted).** The unwindowed
full-frame RAPS (`observables.raps`, sub-bin centroid k*) stays the **primary**
estimator for the synthetic campaign: the generator's fields are periodic by
construction, so the full-frame FFT has zero leakage, and windowing a periodic frame
only convolves the spectrum with the window transform (~2-bin main lobe for Hann),
buying nothing and costing peak resolution. A **Hann–Welch windowed estimator path**
(2-D Hann window, ~50%-overlapping patches, window-power-normalised) is added alongside
it, with three jobs and one strict rule:

1. **Per-bin variance from within a frame**: cropped patches are not periodic, so
   unwindowed patch spectra leak — D3's ~31%/bin floor was measured on plain quadrants;
   windowed overlapping patches give roughly twice the effective independent spectral
   samples per frame area with leakage suppressed, tightening the §4.6
   inverse-variance weights.
2. **Within-frame vs across-realization variance comparison**: windowed-patch variance
   against the 6/2 realization split's across-frame variance. Agreement says
   realization noise is patch noise writ large (fewer generated frames suffice);
   excess across-realization variance is genuine realization-level structure — this
   comparison is the calibration measurement for N (register item 14).
3. **Real-data readiness**: experimental images are cropped, non-periodic, and
   boundary-contaminated — there, full-frame FFT with assumed periodicity is simply
   wrong and windowing is mandatory. Validating the windowed estimator against the
   unwindowed one on synthetic data, where both are computable, is the on-ramp.

**The rule: it is an estimator *pair*, or it is a bug.** Every spectral loss is a
comparison between an observed and a model-side spectrum; any windowed estimator is
applied identically on both sides so the window bias cancels (the discipline D6
enforced for torch-vs-numpy RAPS). Because it changes what k*_obs and every spectral
target means — the sub-bin centroid and the one-bin bar are calibrated on the
unwindowed estimator — the windowed path is **report-only until calibrated against the
unwindowed baseline on synthetic frames** (register item 15); it never silently
replaces the primary.

**Transient-pair data is designed but deferred.** Two *steady-state* frames dt apart
carry zero rate information; the informative protocol is a transient pair (one frame at
~half saturation plus the final frame), whose band-power growth estimates σ(k*) in
absolute units — the only way to break the time gauge (§3.2) and pin absolute rates.
No current gate needs absolute rates (the R2 restriction exists precisely for this), so
this stays a designed follow-up arm, not critical path.

### 4.7 Weight-noise arm (owner-requested 2026-08-17; adopts D-FFT-14 under its own rules)

Parameter noise during training enters the redesign as an **ablation arm**, exactly as
D-FFT-14 pre-registered it — never a silent default, because it changes what every
recorded number means. The two payoffs are evaluated separately, per that entry:
optimization annealing (D5 measured ten seeds in ten distinct sign-structure basins; if
good mechanisms occupy wider basins, noise concentrates seeds into them — an empirical
claim about *this* landscape, not a given) and a mechanism-robustness prior (noise
penalises sharp minima, selecting θ whose Turing instability survives perturbation —
the same axis §3.2's `turing_volume` measures post-hoc). D-FFT-14's placement
constraints bind verbatim: **noise never touches the θ the forward solve sees** (the
warm-start cost is sharply sensitive to per-step θ displacement, and noise near the
ignition margin would toggle the gate — ignition chatter, wasted solves); it is
injected in the gradient or the cheap linear-theory evaluations only, and every solve
runs at the clean current θ. Gate integrity likewise: the arm is judged on the
held-out-band distances and channel co-gates alongside seed agreement, never on
agreement alone — noise that merely smooths the landscape can raise agreement without
raising truthfulness. The noise schedule and magnitude are **UNCALIBRATED** and must be
pre-registered in `docs/DECISIONS.md` before the arm runs, with the D-FFT-14 A/B
protocol (matched no-noise control, identical seeds and budget). In the §4.5 loop the
natural placement is Phase I (linear-theory terms, population-wide, milliseconds); a
Phase II variant would need its own cost measurement first. Register item 16.

### 4.8 Loss registry, training telemetry, and the two notebooks (owner-requested 2026-08-17)

Infrastructure, not science — no thresholds live here — but designed now so the
campaign is observable and reproducible:

- **Loss registry.** Loss terms become registry entries on the existing generic
  `Registry` (`rngrn/registry.py`), one record per term: name, serial callable, batched
  twin (or an explicit refusal, like today's spectral-terms-in-batched refusal),
  default weight, and a calibration tag (`CALIBRATED(<source>)` | `UNCALIBRATED`).
  `DEFAULT_WEIGHTS`, `compute_terms`/`compute_terms_batched` and the config validation
  enumerate the registry instead of hand-maintained key sets, and an
  enumeration-contract test asserts every registered term is classified (batched twin
  or refusal, calibration tag present) — the same completeness-test pattern
  `test_firewall.py` uses for module classification. New terms (`kstar_si`) and retired
  ones (A0-only terms) are then registry states, not scattered edits.
- **Training telemetry.** `TrainingHistory` already records per-step, per-member loss
  parts and raw parameter vectors (`history.py::record_serial/record_batched/
  to_arrays`); it extends to the population loop with: per-member ignition, de-ignition,
  stall, cull and death events (timestamped by step); per-term loss traces through both
  phases; and — because raw θ is gauge-ridden — the **canonical-gauge invariant
  combinations of §3.4 recorded alongside the raw vectors**, so "how parameters are
  being learned" is watchable in coordinates where convergence means something. Arrays
  land in the run directory under `arrays/` (tracked, per the D-PLOT-1 gitignore
  policy) with a `viz` module turning them into the standard figures: per-term loss
  curves, invariant trajectories per member, ignition/cull timelines, and the Phase-II
  spectral-distance trace against the D3 estimation floor.
- **Two notebooks**, in `notebooks/`, both thin drivers over library code (no science
  in notebook cells): `redesign_pipeline.ipynb` runs the full pipeline for the
  campaign — config resolution, the R-milestone stages, each trainer launch wrapped in
  `bash scripts/guarded_run.sh` (§7a of `CLAUDE.md` binds notebook launches too), with
  the run directory as the sole output contract; `redesign_plots.ipynb` consumes a
  finished run directory (tracked indexes + `arrays/`) and regenerates every figure via
  the `viz` module. Firewall note: notebooks sit harness-side, like `train.fit` — they
  may orchestrate both sides but import no answer-key reader into any recovery-side
  module, and they keep `scripts/` off the recovery-side import graph (the §5
  completeness-test blind spot for `scripts/` is exactly where notebook convenience
  imports have bitten before).

### 4.9 Hidden channels (design now, run later)

No latent field variables and no `resid`. The forward solve integrates all N species;
spectral terms are computed on observed channels only; the unobserved species is
constrained implicitly through its effect on the observed channels' saturated spectra,
via the same §4.2 gradients — the hidden-state situation the IFT path handles with no
extra machinery. The predicted latent channel's RAPS/mean/std are *reported* every
evaluation as the identifiability diagnostic. The arm is scored by
`scoring/overparam.py`, whose `DEFAULT_COUPLING_THRESHOLD = 0.05` is **UNCALIBRATED**
(so marked in the code and `TUNING.md`, with its calibration rule — the
`expB_control_matched` coupling distribution — already recorded); R5 executes that
calibration before the arm reports anything. Whether latent identifiability actually
holds is an empirical question the arm answers; the first campaign (§1) observes all
channels.

## 5. Verification: the dynamical lift as the gate

### 5.1 What is being proposed, stated plainly

This section **proposes to reverse D-FFT-4** ("forward validation gates on the QSS
rollout; the finite-μ lift is a later milestone, never the gate") and **to reinstate
the *vehicle* of the withdrawn preregistration §3.6 — but not its criterion.** §3.6
(added and withdrawn 2026-08-03, before any measurement; retained verbatim in
`docs/PREREGISTRATION.md` as a transparency record) required §3.2's `turing_volume`
bars recomputed on the lifted Jacobian *at every μ across the biological band*. The
§3.7 proposed here gates three point verdicts at μ_central and *reports* the
robustness row and the band sweep — both differences are narrowings, and both are in
the register (§8 items 1 and 6). The proposal: a new, additive preregistration §3.7
defining the lifted gate, plus a DECISIONS.md entry (working name D-LIFT-1,
"supersedes D-FFT-4") — ratified by the owner *before* the first gated read-out,
preregistration discipline intact. Nothing in §3.1–§3.5 is weakened; the QSS criteria
remain binding co-gates.

D-FFT-4's rejection rationale — never-built stiff numerics on the critical path let a
numerics artefact read as a recovery failure — is answered structurally: the lift may
not gate anything until it has climbed the validation ladder below.

### 5.2 The integrator already exists; validation does not

`eval/lifted.py::simulate_lifted` already implements Strang splitting with an **exact**
gate substep: at frozen x the gate block is linear in (GA, GR), and the substep
integrates it analytically (closed-form rank-one matrix exponential for the competitive
form; scalar exponential for nc1) — unconditionally stable at any dt/μ, reducing
exactly to the QSS scheme as dt/μ → ∞, with the −1/μ stiffness never touching the
ETDRK4 x-substep. This is the right scheme; what is missing is *validation*, and one
accuracy trap the design states loudly:

> **Exactness of the substep is not accuracy of the coupling.** At
> μ_central = 7.2e-4 with the current dt = 0.2/jac_rate policy, dt/μ ~ 30–300: each
> gate substep relaxes fully to quasi-steady state, silently re-imposing QSS within a
> step. A finite-μ oscillatory (Hopf) route could be smoothed into a stationary-looking
> pattern — the mirror image of the artefact D-FFT-4 feared. Mitigations, both
> mandatory: the linear verdict L1 (exact in μ, no integrator involved) is a co-gate;
> and gated rollouts use dt = min(0.2/jac_rate, μ/2) with a dt-halving convergence
> check, plus at least one anchor run per gated model with dt refined below μ.

Cost, estimated and unmeasured: per step ≈ 2–3× the QSS ETDRK4 (3 diffusing fields +
18 elementwise gate fields, no extra FFTs — an estimate, not a measurement). At the
measured 3.25 ms/step 512² CUDA scaling that is ~11–16 min/field at 1e5 steps and
~2–3 h/field at 1e6 — a projection for a GPU port that does not exist
(`simulate_lifted` is numpy today); the port's own cost measurement replaces this.
The GPU port is therefore a precondition (§7), consistent with §4.1. Independent
cross-check: implement the `integrate_bdf1_newton_krylov` stub
(`src/rngrn/eval/numerics.py`) for one 128² gated field; its current **silent fallback
to ETDRK4 is removed** (made loud) regardless of anything else in this document.

### 5.3 The validation ladder V0–V4 (strictly ordered; each rung licenses the next)

- **V0 — algebraic invariants** (extend existing tests): fixed-point residual of the
  lifted system **at or below the bar the existing tests already hold, 1e-7**
  (`tests/test_lifted.py`); the achievable floor is **UNCALIBRATED** — the only number
  on record is 1.28e-8 over 8 systems × 7 μ, in a parked artefact its own docs mark
  do-not-cite — so V0 *measures* the floor and records it rather than asserting one.
  `rescale_mu` equals a fresh autodiff Jacobian to round-off; gates-at-QSS reproduce
  `model.reaction` exactly. The algebraic and linear claims are already tested at N=3
  on both forms (at tolerance 1e-7); extend to ≥20 generator draws.
- **V1 — linear: eigenvalue continuation, all 21 branches.** `lifted_dispersion` at
  μ ∈ {1e-7, 1e-6, 1e-5} vs QSS dispersion on the same k-grid: N slow branches converge
  to the QSS eigenvalues at empirical order ≈ 1 in μ (slope check across the triplet);
  2N² fast branches ≈ −(1+u)/μ, all Re < 0; k* of the lifted dispersion within one grid
  point of the QSS k*. Branches tracked by eigenvector overlap between adjacent μ, not
  by sorting real parts — branch crossings are exactly what the re-entrant band is made
  of. Absolute error constants: UNCALIBRATED — measured and recorded, not invented.
- **V2 — temporal, 0-D.** The 21-dim well-mixed lifted ODE against a trusted stiff
  reference (scipy Radau, rtol 1e-10): (i) trajectories converge to the QSS 3-dim ODE
  as μ → 0; (ii) the Strang stepper itself matches the reference at
  μ ∈ {1e-5, 7.2e-4, 1e-2, 1e-1, 1} under dt-halving, observed order ≈ 2 where
  dt ≲ μ, sup-norm error at the rollout horizon ≤ 0.1 × pattern_floor
  (= 0.1 × max(1e-3, 0.02·|x*_0|) — derived from the existing patterning floor:
  integration error must sit an order below the smallest amplitude ever called a
  pattern).
- **V3 — spatial.** (a) `simulate_lifted` vs the QSS rollout — today the only spatial
  lifted test is N=2 at 16², μ=1e-6 (`tests/test_lifted.py`); extend to N=3, both
  forms, at μ ∈ {1e-4, 1e-5, 1e-6}, same seed/dt/horizon: patterned flag equal,
  morphology class equal (512² only), k* within one radial bin — `|k*_lift − k*_qss| ≤
  2π/L`, i.e. **12.5% at this target's p=8** (`docs/SPEC_fourier_training.md` §9.1,
  D-FFT-3; the pre-registered 8.3% figure is *half* a bin, derived on the legacy
  `three_gene` sets, and D-FFT-3 records rejecting its import here as sub-resolution
  at p=8) — and field relative L2 difference decreasing with μ (absolute bound
  UNCALIBRATED — the measured curve becomes the calibration). (b) Then at μ_central
  with the §5.2 dt policy and a dt-halving pair — the first run in territory where the
  lift can say something new; report-only until the ladder completes.
- **V4 — re-entrant-band survey.** `mu_critical` (with its re-entrance detection) and
  `robustness_vs_mu` over every Turing-positive system on hand at single-target scope:
  the target's own gate-passing recovered models, plus generator draws already
  harvested. Deliverables: distribution of the first Turing-loss edge vs the biological
  band; fraction re-entrant; and the decision-relevant number,
  **P(lifted-Turing across [1.1e-5, 9.2e-3] | QSS-Turing)**. The one measured draw
  (§2) lost Turing only at μ ~ 1e-1 — an order above MU_BIO_HI; if that separation is
  generic, the whole biological band sits inside the first Turing window and band-wide
  gating becomes cheap. If a substantial fraction is lift-dead *inside* the band, that
  is the headline scientific result of the survey, and it triggers §5.5. V4 is a
  measurement, not a gate: no pass threshold, by design.

Estimated, unmeasured: V0–V2 minutes (eigenscans amortised by `rescale_mu`; 0-D ODEs);
V3 hours (a few 512² fields at 2–3× QSS cost); V4 hours (batched 21×21 eigenproblems).

### 5.4 The gate (proposed preregistration §3.7)

Evaluated per gate-passing candidate at 512² on the data box:

- **L1 — lifted linear verdict at μ_central = 7.2e-4:** strict Turing on the full
  21-branch lifted Jacobian AND not oscillatory (the `stationary` classification —
  built deliberately to avoid the trace test that a −1/μ diagonal defeats). The Hopf
  exclusion is non-negotiable: a wave instability is not the claimed stationary
  pattern.
- **L2 — lifted rollout patterns at μ_central:** amplitude above the existing
  pattern_floor, `stopped_reason == "horizon"` (a step-budget truncation is not
  evidence), under the §5.2 dt policy with a passing dt-halving check.
- **L3 — wavelength:** k* of the lifted rollout within one radial bin of k*_obs —
  `|k*_lift − k*_obs| ≤ 2π/L`, **12.5% at this target's p=8** (SPEC §9.1 / D-FFT-3,
  the binding one-bin bar; see V3 for why the 8.3% figure is not imported).
- **Reported, never gated:** the L1 verdict at 9 log-spaced μ across the band plus
  both endpoints (near-free via `rescale_mu`); `mu_critical` with the re-entrance
  flag; the drawn-μ robustness row (`robustness_vs_mu` — the criterion half of
  withdrawn §3.6, reported not gated; §5.1); morphology class of the lifted field vs
  target (the existing classifier — stripes measured 57.1% recall in-bank and 33.3%
  on held-out n=3, recorded as too weak to gate, an open decision; the target here is
  a labyrinth, 94.4% held-out, and the classifier is report-only regardless); and the
  QSS-vs-lifted field difference.

**Gate at μ_central, report across the band.** μ is a three-decade *uncertainty band*
on a literature-derived timescale ratio, not a swept parameter: gating band-wide would
turn one uncertain quantity into a conjunction of claims biology does not license, and
the measured re-entrance means an edge point can flip on the least-certain digit.
Band-wide gating is the materially stronger claim the owner may still choose —
owner-decision item 6, revisited after V4. **QSS F1–F3 remain co-gates** (continuity
with the D5 baseline; the lift must agree with QSS as μ → 0 anyway, so they cost
nothing and preserve comparability).

**Null/control structure:** L1–L3 run identically on the SPEC §9.5 null ensemble —
alternative sign structures refit to B_train with gates frozen to their structure
under the identical Phase-II budget, plus single-edge-flip variants of the winner,
seeded from the §4.5 candidate pool (owner-decision item 10 governs any deviation
from §9.5 as written). The candidate must pass; the null pass-fraction is *reported* —
and a high null pass-rate is itself the finding that the lifted gate does not
discriminate mechanism. Numerics controls per gated model: the dt-halving check
(mandatory), 5-seed IC replication of the lifted rollout, and the one
`integrate_bdf1_newton_krylov` cross-check field.

### 5.5 Training–verification coupling

A periodic **non-differentiable** lifted-linear audit inside training: one lifted
Jacobian at μ=1, `rescale_mu` to μ_central, eigenscan of the 21×21 across a coarse
k-grid — cheap enough to run every M steps (cost measured at R1, not asserted). Used
for **checkpoint selection** (prefer lifted-Turing checkpoints; reject
QSS-Turing-but-lift-dead candidates before expensive verification), not as a loss
term: nonsymmetric-eigendecomposition gradients are ill-conditioned exactly at the
eigenvalue crossings that define this system's μ-structure, and a differentiable
penalty there is fragile machinery bolted onto the degenerate points that matter.
Escalation rule, tied to V4: if P(lifted-Turing at biological μ | QSS-Turing) is high
(the working notion is ≈ ≥0.95 over the surveyed population — UNCALIBRATED, set from
V4 itself), the audit stays diagnostic; if materially lower, it becomes a hard
model-selection filter; a differentiable proxy (e.g. a few power-iteration steps on
the lifted J at selected k) is last resort and is an objective change —
owner-decision class, not taken here.

## 6. What survives, what is replaced

| Component | Fate |
| --- | --- |
| `model.py` gated-promoter core (both forms, KA/KR budget-gate, autograd Jacobian, cubic backend, `BatchedRNGRN`) | **Survives unchanged** (§3.1) |
| β = softplus(θ_β) as a free parameter | **Replaced**: solved out via fixed-point pinning; β ≥ 0 hinge (§3.2) |
| α, δ softplus; D = exp | **Replaced**: box-sigmoid for α, δ; D via D_max × in-box ratios with soft D-ratio prior (§3.3) |
| `kstar_anchor` | **Replaced** by `kstar_si` (new key; old key kept for baseline arm A0) — owner-decision item 11 (§4.4) |
| `turing_hinges_split` | Survives in form; `k_min_frac` proposed k*-relative — owner-decision item 12 (§4.4) |
| `frame_scale_anchor`, `anticollapse`, `stationarity_residual` (+ latent fields) | **Retired** in the redesign arm (anticollapse retirement untested — owner-decision item 11); baseline arm A0 keeps all at current defaults as the control |
| `morphology_consistency` | Already inert (not in `compute_terms`); removed from config for clarity, no behaviour change |
| `param_prior` + `configs/bio_box.yaml` | **Promoted** to nonzero weight (D-ratio soft prior) (§3.3) |
| `losses/spectral.py` (five terms, bands, ignition, RAPS) | **Survives in form; requires a new batched-across-members implementation** (§4.1) plus target interpolation onto solve-box bins + per-bin inverse-variance weights (§4.3, §4.6) |
| `forward.py` (PatternSolver, LSMR minimal-norm adjoint + tripwire, translation correction, warm modes) | **Survives in form, promoted to central; the batched form is new work** (§4.1); gains re-tileable L with hysteresis (§4.3) |
| ETDRK4 integrators (`eval/numerics.py`, `etdrk4_torch.py`) | Survive; gain a differentiable/checkpointed unrolled mode (§4.2) |
| `recover.py` multistart | **Replaced** by the Phase I/cull/Phase II population loop (§4.5); seeding discipline and per-member clipping survive |
| QSS rollout (`eval/rollout.py`) | Survives — **demoted from sole gate to μ→0 control** and F1–F3 co-gate (§5.4) |
| `eval/lifted.py` (dispersion, verdicts, `rescale_mu`, `mu_critical`, `robustness_vs_mu`, `simulate_lifted`) | **Promoted from parked to gate — after the V0–V4 ladder**; GPU port required (§5.2–§5.3) |
| `integrate_bdf1_newton_krylov` silent ETDRK4 fallback | **Removed/made loud** unconditionally (§5.2) |
| Morphology scorer, robustness cloud, leak controls, L-transfer harness, firewall tests | Survive unchanged; L-transfer stays QSS (lifted L-transfer is future work) |
| Data generator | Gains multi-realization emission (N = 8 at 512²/p=8, 6/2 realization split) and held-out small-box sets (p ∈ {2,3,4} × S seeds) (owner decision, §4.6) |
| `rngrn/registry.py` (generic `Registry`) | Survives; gains the **loss-term registry** (name, serial fn, batched twin or explicit refusal, default weight, calibration tag) with an enumeration-contract test (§4.8) |
| `history.py` (`TrainingHistory`) | Survives; extended with population events (ignition/de-ignition/stall/cull/death), per-term traces through both phases, and canonical-gauge invariant trajectories (§4.8) |
| Visualization / notebooks | **New**: `viz` module + `notebooks/redesign_pipeline.ipynb` (guarded full-pipeline driver) + `notebooks/redesign_plots.ipynb` (figures from a finished run dir) (§4.8) |
| Weight noise (D-FFT-14) | **Adopted as an ablation arm** at R4, under D-FFT-14's placement and A/B rules; never a default (§4.7, register item 16) |

Firewall note: every new training-side component (solve-box logic, unrolled gradients,
lifted audit) is recovery-side and touches only observation-derived quantities; the
lifted gate evaluation joins `validate.score_recovery` on the scoring side. New modules
land on the `RECOVERY_SIDE`/`SCORING_SIDE` lists of `tests/test_firewall.py` at birth —
the completeness test enforces this inside `src/rngrn/`.

## 7. Milestone sketch (single target; each milestone gates the next)

- **R0 — ratification. COMPLETE 2026-08-17** (owner delegation; §8 preamble). Prereg
  §3.7 and the §3.4 amendment note are in `docs/PREREGISTRATION.md`; D-LIFT-1 and
  D-REDESIGN-1 are in `docs/DECISIONS.md`.
- **R1 — lift validation ladder** V0–V4 (§5.3), including the GPU port of
  `simulate_lifted` and the loud `bdf1` fallback fix. Deliverable: the ladder report
  and the V4 survey number.
- **R2 — model reconstruction + observability.** Fixed-point pinning (β solved out —
  including the frame-mean bias measurement on the target required by §3.2),
  box-sigmoid reparameterization, `kstar_si`, retired terms; baseline arm A0 preserved
  verbatim. The §4.8 infrastructure lands here too: the loss registry with its
  enumeration-contract test, the `TrainingHistory` population/event/invariant
  extension, the `viz` module, and both notebooks (the pipeline notebook driving R2's
  own run is its first test). Deliverables: a Phase-I-only population run on the
  target read against the D5 row on the §1 comparable columns, the measured per-member
  ignition rate and throughput/RSS curve that set B and K (§4.5), and the run's
  figures regenerated end-to-end from `redesign_plots.ipynb`.
- **R3 — gradient machinery.** Batched spectral/forward implementation with its
  bit-level equivalence test (§4.1); adaptive commensurate solve box; FD A/B on both
  gradient paths at the operating point plus the 96²-vs-512² RAPS fidelity and
  target-interpolation measurements (§4.2–§4.3). Hard gate: FD-faithfulness at
  tol 1e-4. Deliverable: the A/B report, the fidelity measurement, and the measured
  stall-rate distribution (which calibrates the §4.3 switch fraction).
- **R4 — the campaign.** Full Phase I → cull → Phase II on the target with the
  N-frame training targets (§4.6); guard-rail and spectral weights swept (all UNCALIBRATED at birth);
  gate-passers through the §5.4 lifted gate with the §9.5 null ensemble; the §4.7
  weight-noise arm runs here as a D-FFT-14 A/B against its matched no-noise control,
  after its schedule and magnitude are pre-registered. Deliverable:
  the redesign's row on the §1 comparable columns — Turing fraction, distinct sign
  structures, k* error — plus the population-consistent `topology_consistency` (§3.4),
  the lifted-gate pass/null-pass fractions, the held-out realization-consistency check
  (§4.6), and — report-only — the small-box cross-L and mode-selection comparisons
  (§4.6).
  **What R4 cannot say:** per `docs/PLAN_fourier_training.md` §1, work on this burned
  tuning sample supports **no recovery and no generalisation claim**. R4's licensed
  sentence class is "the redesign's objective changes the measured baseline statistics
  on the tuning sample, in this direction, by this much, against the D5 row and the
  null ensemble". A recovery claim requires R5.
- **R5 — breadth (out of first-campaign scope).** Remaining canonical samples, nc1
  form, noise arms, held-out one-shot under the frozen config; the m<N arm (§4.9,
  including the recorded coupling-threshold calibration); transient-pair protocol if
  ratified.

## 8. Owner-decision register

**RATIFIED 2026-08-17 under explicit owner delegation.** The owner reviewed this
proposal interactively and delegated the register: *"you decide on the Owner-decision
register, I trust you."* The rulings are recorded in `docs/DECISIONS.md` (D-LIFT-1 for
item 1, which supersedes D-FFT-4; D-REDESIGN-1 for the register as a whole, including
the per-item rulings on 6, 8 and 13), and the preregistration amendments (§3.7 added;
§3.4 amendment note) are dated 2026-08-17 — before any run they judge. R0 of §7 is
therefore complete. Items marked UNCALIBRATED stay UNCALIBRATED: ratification fixes the
*rules*, not the numbers.

1. **The lift becomes the verification gate** — proposes reversing D-FFT-4; revives
   the *vehicle* of withdrawn §3.6 (not its criterion — §5.1) as a new additive §3.7
   (prereg amendment). §5.1.
2. **F-D1-5 resolution**: adopt the adaptive commensurate solve box (option a) with
   the FD-verification clause (d); option (b) — a looser convergence bar — stays off
   the table; stalled members switch to the unrolled gradient path instead. §4.3.
3. **Hard bio-box reparameterization for α/δ** (changes prereg §3.4's α/δ component
   from measured outcome to structural truth — prereg-touching); **soft D-ratio prior**
   retained per the recorded bio_box decision. §3.3.
4. **Data protocol** (small-box extension accepted by the owner 2026-08-17, pending
   the same R0 ratification as the rest): N = 8 realization frames of the target
   system at 512²/p=8, split 6 train / 2 held-out over realizations; plus held-out
   small-box sets (p ∈ {2, 3, 4}, S seeds each) used only for cross-L transfer and the
   report-only mode-selection distribution; transient-pair arm designed, deferred.
   §4.6.
5. **Linear terms demoted to guard-rail weight after ignition** — amends the SPEC §3
   "unchanged for the whole run" decision; baseline arm A0 preserved. §4.4.
6. **Gate at μ_central vs band-wide** — **RULED: gate at μ_central = 7.2e-4, report
   across the band.** Rejected: band-wide gating (a conjunction of claims over a
   three-decade uncertainty band, with measured re-entrance able to flip an edge
   point on the least-certain digit). Revisited after V4 only if the survey shows the
   biological band generically inside the first Turing window. §5.4.
7. **β solved out / fixed-point pinning** — supersedes `frame_scale_anchor`
   (record-decision entry; not a prereg change), conditional on the §3.2 frame-mean
   bias measurement on the target. §3.2.
8. **Gradient estimator policy** — **RULED: adjoint primary, truncated-unrolled as
   stall fallback**, with the promotion rule pre-specified now so no post-hoc choice
   exists: the unrolled path may become primary only if R3's FD A/B shows it
   FD-faithful at tol 1e-4 on both converged and stalled members AND its measured
   cost does not exceed the adjoint path's. §4.2.
9. **Reserved** (folded into item 14; kept so earlier cross-references stay stable).
10. **The SPEC §9.5 null ensemble** — the design's default implements §9.5 as written
    (frozen-gate refits + single-edge flips, identical budget, seeded from the §4.5
    candidate pool); any substitution of unrefit pre-cull members weakens a BINDING
    Stage-0 gate and happens only by explicit owner decision. §4.5, §5.4.
11. **Objective recomposition beyond item 5**: `kstar_anchor` → `kstar_si`, and the
    retirement of `anticollapse` (untested — D5 ran with it at 0.5; the
    measured-rejected result is strengthening, not removal). A0 keeps both at current
    defaults as the control. §3.3, §4.4.
12. **`k_min_frac` redefined k*-relative** — `TUNING.md` records this changes what the
    term means and requires re-measurement; A0 keeps the grid-relative form. §4.4.
13. **Reconciling the population loop with §3.1/R1's independent-seeds requirement** —
    **RULED: the unit of independence is the full pipeline replicate.** §3.1/R1
    statistics are computed across the winners of K_rep independent
    Phase I → cull → Phase II replicates run from independent master seeds
    (K_rep = 5, matching SPEC R1's five independent runs); members inside one
    replicate's population are never counted as independent seeds. Rejected: treating
    culled members as seeds — the cull selects them *for* sign-structure distinctness,
    which biases exactly the statistic §3.1 measures. §3.4, §4.5.
14. **Numbers introduced by this design, all UNCALIBRATED, with their calibration
    rules**: B = 512 and K = 16 (R2 throughput/RSS and distinct-structure counts,
    §4.5); the ~20% stall-switch fraction (R3 stall-rate distribution vs measured
    gradient-error difference, §4.3); N = 8 frames with the 6/2 realization split, and
    S = 8 small-box seeds per p (the measured per-bin and per-mode variance, §4.6);
    the ~10% guard-rail ratio and Phase-II budget/lr (first-campaign sweeps,
    §4.4–§4.5); `kstar_si`'s ε and temp (R2 sweeps; temp inherits the never-swept
    60.0, §4.4); the truncated-backprop segment length (R3 gradient-error curve,
    §4.2); the V-ladder tolerances not already derivable (V0 floor, V1 error
    constants, V3 field-difference bound — measured by the ladder itself, §5.3); the
    dt policy min(0.2/jac_rate, μ/2) and the 9-point μ report grid (§5.2, §5.4); the
    ≥20-draw V0 extension and 5-seed IC replication counts (§5.3–§5.4); and the ≥0.95
    working notion in §5.5 (set from V4). None becomes binding without a measurement
    behind it.
15. **Spectral estimator policy** — unwindowed full-frame RAPS stays primary for the
    synthetic campaign; the Hann–Welch windowed path is added under the
    both-sides-identical rule and is **report-only until calibrated against the
    unwindowed baseline on synthetic frames** (its three jobs and the rule: §4.6).
    Rejected: swapping the primary estimator — the sub-bin centroid k* and the
    one-bin bar are calibrated on the unwindowed estimator, and a silent swap would
    change what every spectral number means.
16. **Weight-noise arm** (owner-requested 2026-08-17) — **RULED: adopted as an
    ablation arm under D-FFT-14's pre-registered rules verbatim** (gradient/
    linear-theory injection only, solves at clean θ; matched no-noise control;
    judged on held-out bands alongside seed agreement, never agreement alone).
    Schedule and magnitude UNCALIBRATED — pre-registered in `docs/DECISIONS.md`
    before the arm runs. Rejected: making noise a default (changes what every
    recorded number means — D-FFT-14's own prohibition). §4.7; D-REDESIGN-2. The
    loss registry, telemetry and notebooks of §4.8 are engineering, carry no
    thresholds, and need no register item.
