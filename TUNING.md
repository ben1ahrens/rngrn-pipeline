# TUNING.md — what Claude Code tunes, implements, and validates on CUDA

This template **builds, wires together, and dry-runs on CPU**. It does NOT claim to
*recover* anything yet — the milestone science (does recovery find the right regime?)
is the next stage, on the local CUDA machine. This file is the explicit list of the
knobs, stubs, and unproven pieces to work on, grouped by pipeline stage, with where
each lives and what "done" looks like.

Legend: **[TUNE]** = a numeric/choice knob to search · **[IMPL]** = a stub to implement
· **[VALIDATE]** = repackaged-but-unproven science to confirm with a test.

---

## Stage 1 — data generation & storage

- **[TUNE] generator horizon & timestep** — `configs/*.yaml: data.T_max, data.dt`, used by
  `data/solver.py::simulate_to_attractor`. The template caps `dt` to a stability-safe
  value (`0.2/max|eig J|`) and sub-steps to reach `T_max`; the reference systems converge,
  but confirm the frame is a *saturated attractor* (not still transient) at your resolution.
  Done: the generated frame's k* is stable under a longer horizon.
- **[TUNE] resolution** — `data.resolution` (template 128). Higher resolution sharpens k*
  but costs solver time; a GPU/implicit generator lets you push it.
- **[IMPL] a faster / implicit generator** — `data/solver.py` is explicit IMEX with a python
  reaction. For high resolution or stiff systems, port the reaction to torch/GPU or use an
  implicit step. This is answer-key side, so correctness (converged pattern) matters, speed is
  a convenience.
- **[IMPL] in-pipeline 3-node reference systems** — to *generate* new N=3 systems (rather than
  read the external HDF5), register a 3-species `ReferenceSystem` in `data/rd_models.py`
  (the `RD_SYSTEMS` registry). The 6 topologies × 2 kinetics from the 3-gene dataset generator
  are the natural set. Until then, N=3 enters only via `data.gate.from_3gene_hdf5`.

## Stage 2 — the objective (losses/)

- **[TUNE] loss weights** — `loss.weights = {kstar, turing, resid, anticollapse, morphology}`
  (`config.py::LossConfig`, starting values from design-doc §5). These are the primary tuning
  axis. k* and turing lead; resid is normalised to O(1); anticollapse floors ‖J‖.
- **[TUNE] spectral term weights — all 0.0 (OFF), UNCALIBRATED at birth** —
  `loss.weights = {spec_shape, spec_aniso, spec_amp_mean, spec_amp_fluct, real_moments}`
  (added M1, 2026-08-12; docs/DECISIONS.md D-FFT-11). Swept at Stage 0 per
  `docs/SPEC_fourier_training.md` §5; settled values go to the D-FFT-9 ledger. Any nonzero
  weight ignites the forward pattern solve (`src/rngrn/forward.py`) on detected Turing
  instability — cost at training grids is UNMEASURED (~3–9 s/solve at 64², an unrecorded
  test timing), so weight
  sweeps need `scripts/guarded_run.sh` and a measured budget first. Serial path only;
  the batched path refuses loudly.
- **[TUNE] `loss.spectral_ignition_margin` (1e-3) — UNCALIBRATED**, mirrors the
  `turing_hinges_split` margin default (`losses/terms.py:186`). The band edges
  `loss.spectral_b_lo/b_hi = 0.60/1.55` are MEASURED (D-FFT-9 closure 1), not free
  knobs — re-measure per sample class (Stage 2) rather than sweeping them.
- **[IMPL] k\* tolerance band `tau` — INERT. Reads nothing; sweeping it is a no-op.**
  `loss.tau` (0.12) is threaded `train.py:211` → `recover.py:224` → `total.py` →
  `terms.kstar_anchor` (`losses/terms.py:291`, and `kstar_anchor_batched` at `:732`) and is
  **never referenced in either function body** — those bodies use only `sig`, `temp`,
  `kgrid` and `kstar_obs`. It is exported to the run index (`export.py:47`), so a sweep over
  `loss.tau` will produce differently-labelled rows carrying bit-identical losses and
  bit-identical recovered models. Do not report such a sweep as "k\* tolerance has no
  effect"; it measures nothing. `docs/DATA_INTO_MODEL.md:439` already recorded this.
  *Intended* behaviour, if implemented: absorb the measured +5%–+13% selected-vs-linear
  wavelength bias — widen if recovery fights the bias, tighten to sharpen k\* selectivity.
  Either implement the band or delete the parameter from the whole chain.
  **[VALIDATE] flagged 2026-07-29 (unit 8, kstar-honesty):** one
  FFT bin is measured at 16.7% of k* (`validate.py::_leak_instrumentation`'s
  `kstar_fft_bin_width`), so `tau=0.12` is only 0.72 of one bin — BELOW the FFT
  estimator's own resolution. Since scoring now heads on `kstar_fft_rel_err` (the
  2026-07-29 headline swap, `docs/STATE_OF_THE_SCIENCE.md` line 499), a tolerance tighter
  than the measurement's own quantisation may be discriminating noise, not signal. Not
  re-tuned here — flagged for whoever owns the loss-weighting sweep.
- **[TUNE] anti-collapse floor `jac_floor`** — `loss.jac_floor` (1.0). Too low → the f≡0
  trivial minimum leaks back in; too high → distorts the recovered J magnitude.
- **[IMPL] a differentiable morphology term** — `loss.weights.morphology` is currently INERT:
  `losses/terms.py::morphology_consistency` is a non-differentiable numpy diagnostic and is not
  in `losses/total.py::compute_terms`. To use the ~0.1 weight, add a differentiable morphology
  term (e.g. spectral anisotropy of the lifted field) and wire it into `compute_terms`.
- **[VALIDATE] every loss term** — `terms.py` is repackaged from the SCAFFOLD objective. Confirm
  per-term: gradient flows to θ, and the term's zero sits where the design doc says
  (`tests/test_science.py` has the dispersion/Jacobian anchors; add per-term zero-location tests).
  Done for the three promoted terms in `tests/test_losses.py`; the rest are still open.

### Promoted from the experiments (unit 1)

- **[TUNE] frame-scale anchor weight** — `loss.weights.anchor` (2.0),
  `terms.py::frame_scale_anchor`. Ported from `scripts/exp05_pixel_minibatch.py` where 2.0 took
  convergence from 2/40 to 38/40 seeds. **NEVER SWEPT** — 2.0 is the only value ever run. It is
  also the term with the least principled magnitude in the objective, because its target
  `frame.mean()` is a *biased* estimator of x\*₀ (exp12: Spearman ρ = +0.950, ratio median 0.921),
  so it is a scale regulariser, not a fit. Sweep it before trusting any anchor-sensitive result.
- **[TUNE] hinge k-floor** — `loss.hinge_k_min_frac` (0.1), i.e. the instability hinge maximises
  over `k ≥ kgrid[int(0.1·K)]`. Inherited unchanged from exp02/exp05; never swept. Too small and
  the split stops separating the two conditions; too large and it cannot select a low-k pattern.
- **[VALIDATE] the k-floor is grid-relative, not k\*-relative** — because `k_min_frac` indexes the
  grid, the floor moves with the grid's span. Measured on `three_gene_val/sample_0000`
  (k\*_obs = 0.4320): `recover._kgrid_for` puts `k_min` at **0.822·k\*_obs**, the exp05 grid at
  **0.698·k\*_obs**. So the promoted library hinge excludes a wider band below k\* than the
  experiment that validated it. Defining the floor relative to `kstar_obs` instead would change
  what the term means, so it was NOT changed here — decide it deliberately, then re-measure.
- **[TUNE] data-first staging fractions** — `loss.staging_off_frac` / `loss.staging_ramp_frac`
  (0.25 / 0.25), `weighting.py::staging_factor`. Exactly the exp05 schedule; never swept. Note
  the schedule is a fraction of `train.adam_steps`, so changing the step budget silently moves
  the ramp.
- **[VALIDATE] `loss.detach_xstar`** — the library differentiates the dispersion terms through
  x\* (`steady_state_diff`); exp05 passes `xs.detach()` to them. Default is `False` = library
  behaviour, so nothing silently changed, but **the 36.8 % measurement was made with `True`**.
  Which one matters has NOT been isolated in an A/B. Run one before quoting a library-vs-exp05
  reproduction either way.
- **[NOTE] `k≈0` grid point** — `turing_hinges_split` treats `kgrid[0]` as the uniform mode, but
  `recover._kgrid_for` starts the grid at `kstar_obs/50`, not 0 (the experiments used exactly 0).
  The gap is O(D·k²) at that k; unmeasured.
- **[NOTE] `loss.weights.resid = 0.0` is SETTLED, not untuned** — exp06 swept pixel batch
  {64,128,512} × weight {1,3,10}, 8 seeds each; all nine cells collapsed to 1/8 Turing seeds,
  best median k\* error 11.8 % against 0.4 % with it off. The term is kept for future arms.
  Evaluating it costs **45 % of a forward+backward step** (measured, 96×96 N=3, 40 reps:
  9.39 ms with, 5.15 ms without), so `compute_terms(compute_resid=False)` omits it — and
  `recover.py` selects that only when the strategy's weights are static *and* its base
  `resid` weight is 0. Re-enabling the residual re-enables the cost automatically.
- **[IMPL] hidden-channel (m &lt; N) recovery has no objective at `resid = 0`** — the latent
  fields enter the objective through `stationarity_residual` and **nothing else**, so at weight 0
  their gradient is exactly `0.0` (measured, N=3 m=2). `recover()` now REFUSES such runs rather
  than returning the random init as a "recovered" latent field. Since exp06 also measured the
  residual as harmful to Turing recovery, Experiment A currently has **no known-good objective**:
  it needs a term that sees the latent fields and is not the full-RHS residual. This is an open
  problem, not a misconfiguration.

## Stage 3 — optimisation (inner + outer)

- **[TUNE] inner schedule** — `train.{n_restarts, adam_steps, adam_lr, lbfgs_steps, grad_clip}`
  (`config.py::TrainConfig`). Restarts matter because the inverse problem is multi-modal.
- **[IMPL] GradNorm weighting** — `losses/weighting.py::GradNormWeighting` **raises
  `NotImplementedError` in `__init__`** (`weighting.py:69`) and defines no `combine`. *(Corrected
  2026-08-04: it does NOT silently run with fixed weights — that was the old behaviour, and
  raising is the deliberate replacement.)* Implementing it means writing `__init__` **and** a new
  `combine`: the gradient-magnitude balancing update (backprop each term, measure ‖∂θ(w_k L_k)‖,
  nudge weights toward equal magnitude every `update_every` steps).
- **[IMPL] NTK weighting** — `losses/weighting.py::NTKWeighting`, same shape: raises in `__init__`
  (`weighting.py:87`), no `combine`. Implement the NTK-trace / residual-decay-rate estimator and
  set weights inversely (handles the dissimilar per-term convergence rates, incl.
  structural-parameter sensitivity — the inverse-problem case).
- **[NOTE] `ratio` weighting IS implemented** — `losses/weighting.py:94`, selectable as
  `loss.strategy: ratio`. No other doc mentions it.
- **[TUNE] seed replicates** — `optim/sweep.py::run_sweep(seeds=...)`. Rank configs by a seed
  aggregate, never one init. `sweep_example.yaml` shows the axis format.

## Stage 4 — validation / milestones (validate.py, train.fit)

- **[TUNE/UNCALIBRATED] Milestone 1 (N=m=2)** — `configs/milestone1_{gm,schnak}.yaml`. Get
  recovery to land in-regime and match k* within ~15% on both reference frames. This is where
  you tune §2–3 knobs.
  **The `~15%` is UNCALIBRATED and currently unreachable (D-EVID-17, 2026-08-11).** It dates
  from the initial template commit and nothing has ever measured it. Worse, neither reference
  frame can be generated at the shipped defaults: `simulate_to_attractor` at
  `resolution=128, T_max=4000, dt=0.1` raises `FloatingPointError: solver diverged` at step
  **133** for `gierer_meinhardt` and step **29** for `schnakenberg`. Fix the solver (or lower
  `dt`) before treating this milestone as a bar — and calibrate the tolerance once frames
  actually exist.
- **[TUNE/IMPL] Milestone 2 (N=2, m=1)** — `configs/milestone2_gm_partial.yaml`. Latent-field
  inference is wired (`recover.py`, the `m<N` branch: latent fields init from the smoothed observed
  mean, co-optimised). Tune the latent init/regularisation and measure identifiability degradation
  vs M1 (`validate.py::score_recovery` reports the metrics; the benchmark harness aggregates them).
- **[VALIDATE] scoring** — `validate.py::score_recovery` grades k* / regime / J-sign only
  (parameter values deliberately not scored). Confirm the sign-structure comparison is meaningful
  for your systems.

## Stage 5 — post-training rollout & analysis (eval/)

- **[TUNE] integrator choice** — `solver.integrator ∈ {etdrk4, etdrk4_rfft, imex_split,
  bdf1_newton_krylov}`. ETDRK4 is the stiff-safe default; `etdrk4_rfft` is the same scheme on
  the half spectrum (1.8–1.9× faster, agreeing to ≤6.7e-16 after 100 steps) and is what the
  morphology rollout uses. `imex_split` is the reference that COLLAPSES under stiff
  recovered diffusion (keep it only to reproduce the pitfall).
- **[TUNE] rollout horizon** — `solver.horizon_growth_times` (40× the growth timescale 1/σ_max)
  and `solver.n_grid`. The dt is derived from the fastest reaction rate (diffusion is not a CFL
  limit); confirm the saturated pattern is timestep-independent.
- **[IMPL] fully-coupled ETDRK4** — `eval/numerics.py::integrate_etdrk4` uses the diagonal
  linear operator −D_i k² per mode (correct for diagonal diffusion). For the fully-coupled
  (J − k²D) matrix-exponential variant (tighter for strongly cross-coupled J), extend the
  contour-integral φ-functions to the matrix case.
- **[IMPL] BDF1 Newton–Krylov** — `eval/numerics.py::integrate_bdf1_newton_krylov` currently
  delegates to ETDRK4. Implement the implicit-Euler + `scipy.optimize.newton_krylov` solve so you
  have the independent stiff cross-check the validation plan calls for.
- **[VALIDATE] the stiff-collapse claim** — `tests/test_science.py::test_etdrk4_finite_under_stiff_diffusion`
  checks ETDRK4 stays finite; add the paired assertion that `imex_split` collapses on the SAME
  stiff parameters (the concrete evidence for the pitfall).

### Morphology rollout (unit 7)

- **[TUNE] saturation stopping rule** — `solver.morphology_saturation_tol` (0.01) and
  `solver.morphology_saturation_window` (5), consumed by `eval/rollout.py::_saturated`.
  UNCALIBRATED, and measured NOT to fire on the Turing fixture in `tests/test_rollout.py`:
  the amplitude goes flat by ~step 200 of 609 but k\* keeps creeping as the labyrinth
  coarsens, and the rule requires both signals. Calibrate it on real recovered models
  before relying on it, or delete it.
- **[TUNE] collapse margin** — `eval/rollout.py::simulate(collapse_margin=0.1)`. The
  collapse rule stops a linearly stable model once the amplitude has been an order of
  magnitude below the `patterned` threshold for a full window. This is the stop that
  actually pays (a near-marginal recovered model went 15000 steps / ~28 s → 1000 steps /
  1.7 s on `three_gene_val/sample_0000`). The margin has not been scanned.
- **[TUNE] morphology step budget** — `solver.morphology_max_steps` (15000). At the
  measured `etdrk4_rfft` step costs this bounds one field at 13 / 26 / 42 s on a
  64 / 96 / 128 grid, so a 128×128 target would need it lowered to stay under 30 s.
- **[VALIDATE] unpatterned ⇒ morphology mismatch?** — OPEN METRIC DECISION, not taken.
  When the recovered model produces no pattern, `train.py::_morphology_rollout` withholds
  the field and `morphology_scored` stays `"target_only"`, with `rollout_patterned=False`
  recording the fact. The alternative — recording `morphology_match=False`, since a flat
  field certainly does not reproduce the target's pattern — changes what the headline
  morphology number means across a whole benchmark and needs an owner's decision.
- **[VALIDATE] resolution independence** — measured on one synthetic Turing fixture only:
  |Δk\*|/k\* ≤ 1.7e-2 and morphology features ≤ 9.4e-2 between 64 / 96 / 128 grids at
  fixed L. Timestep independence is three orders tighter (≤ 1.1e-5). Re-measure on real
  recovered models before treating the resolution number as a bound.

## Stage 6 — benchmarking (optim/benchmark.py)

- **[TUNE] identifiability metric** — `benchmark.py` reports `kstar_identifiability_std` (spread
  of recovered k* across in-regime seeds). Decide whether to add param-spread / robustness-volume
  columns as additional identifiability headlines.
- **[TUNE] index backend** — `tracking.index_backend` (`jsonl` default | `sqlite`). JSONL is
  zero-setup for the template; switch to `sqlite` (`index.py`) once you have hundreds of runs and
  want SQL filtering/aggregation. `benchmark.build_table(backend=...)` and the CLI `--index-backend`
  flag both honor it. Same row schema either way — the switch is lossless.
- **[NOTE] dataset registry** — datasets you download/generate elsewhere go in the store via
  `rngrn register-data --dataset-id <id> --payload <file.h5>` and load by name (`source: registry`,
  `dataset_id: <id>`), through the same firewall gate. `data/registry.py` + `gate.from_registry`.
- **[TUNE] spare-species inertness threshold** — `scoring/overparam.py::DEFAULT_COUPLING_THRESHOLD`
  (0.05, relative). UNCALIBRATED starting point. Calibrate from the distribution of
  `max_abs_coupling_*` in the `expB_control_matched` arm, where no spare species exists.
- **[VALIDATE] Experiment A identifiability read** — with N=3 and m=2 the permutation search is
  degenerate (one hidden index), so permutation alignment is a no-op there; the honest read is
  `latent_field_quality` (use |r|, since sign/scale is a gauge freedom). Validate that the
  latent correlation separates good from bad recoveries before trusting it as a metric.
- **[TUNE] hidden-channel choice** — `model.observed_idx` in `expA_hidden_channel.yaml`. Rotate
  which gene is hidden ([0,1] / [0,2] / [1,2]); identifiability may depend on which node in the
  topology is unobserved (a hub vs a leaf).
- **[VALIDATE] acceptance test** — once a config reliably converges, freeze it as an acceptance
  test with tolerances on k* + Turing conditions (the summary in `tests/` explains the "only once
  it passes" rule).

## Cross-cutting

- **[TUNE] float precision / device** — the library sets `torch.set_default_dtype(float64)` for
  tight autodiff on small N. On GPU with large N you may drop to float32 for the rollout; keep the
  Jacobian/dispersion in float64 if stability suffers.
- **[TUNE] model init scales** — `model.py::RNGRN.__init__` raw-parameter init (the `*0.5`, `-1.0`
  offsets). These set where recovery starts; if many restarts fail the steady-state solve
  (`SteadyStateError`), widen or recentre them.
- **[NOTE] OpenMP affinity** — `tests/conftest.py` sets `KMP_AFFINITY=disabled`,
  `OMP_NUM_THREADS=1` to dodge an `OMP: Error #179` seen in sandboxed containers on `import torch`.
  Harmless on a normal workstation/GPU box; remove if not needed.

## L-generalisation (unit 12 — `recover.py::recover(nondim=...)`, `scoring/lgen.py`)

- **[TUNE] `model.nondim`** — `False` (DEFAULT, the pre-existing dimensional path) recovers in
  physical coordinates; `True` recovers on the unit box `x_hat = x/L`, so the learned object is
  `(J, D/L**2)` and the optimisation never reads L at all. Both paths report PHYSICAL
  `kstar_model` and `params["D"]`, so scoring is unchanged. UNMEASURED on real data: which path
  recovers better at a fixed step budget is not known — and see the confound below before trying
  to answer it.
- **[VALIDATE] the nondim D init is off by ~200x** — `model.py` inits `D = exp(theta_D)` at median
  1.00 (measured over 2000 seeds). On three_gene_val (19 samples, L = 40..139) the generator's D
  has median 28.8 and `D/L**2` median 4.97e-3, so the dimensional init starts ~29x too SMALL and
  the unit-box init ~202x too LARGE — opposite directions. Any dimensional-vs-nondim comparison at
  a fixed budget is confounded by this. The L-free repair is to set the D init from the frame's
  own `k*_obs` (`D ~ |J| / k*_obs**2`, firewall-clean), which would change BOTH paths and so
  invalidate every recorded number — it belongs with the priors/init work, not here.
- **[FIXED 2026-07-29, unit B4] `_kgrid_for`'s absolute 2.0 floor was an L-generalisation bug in
  the DIMENSIONAL path.** *(This entry said "NOT fixed here" until 2026-08-04; it had been fixed
  for a week.)* The old `kmax = max(2.0, 8 * kstar_obs)` is in rad/length, so once
  `kstar_obs < 0.25` the k-grid stopped tracking the observed wavenumber and became a fixed
  absolute window — on this data `L > 150.8`, i.e. **11 of the 287 registered samples (3.8%)**,
  pinned to the wrong band. The nondim path could not hit it (`k_hat*` is 37.7 for every
  three_gene sample).
  **Now** (`recover.py:129-139`): `kmin = kstar_obs/50 + 1e-3`,
  `kmax = max(span * kstar_obs, 2 * kmin)` — scale-free, no absolute constant. See
  `CLAUDE.md` §7c point 2.
  **Comparability:** numbers recorded on the dimensional path *before* the fix for those 11
  samples are not comparable to numbers after it. The 3.8 % measurement is retained above as the
  record of what it affected.
- **[TUNE] the cross-L agreement statistic** — `scoring/lgen.py::modal_sign_agreement` is entrywise
  modal agreement and the control is within-L across-seed agreement. Both are choices, both are
  documented in that module, and NO pass threshold is defined. Settle them against real
  `three_gene_multiL` runs, not in the abstract.
- **[NOTE] cross-L grouping needs `system_id`** — the gate reads an optional per-sample `system_id`
  attribute into `AnswerKey.system_id` (scoring side only). Datasets without it cannot be grouped
  across L; `lgen_consistency` excludes them rather than scoring them as failures.
