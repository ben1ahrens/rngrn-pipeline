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
- **[TUNE] k\* tolerance band `tau`** — `loss.tau` (0.12). Absorbs the measured +5%–+13%
  selected-vs-linear-wavelength bias. Widen if recovery is fighting the bias, tighten to
  sharpen k* selectivity.
- **[TUNE] anti-collapse floor `jac_floor`** — `loss.jac_floor` (1.0). Too low → the f≡0
  trivial minimum leaks back in; too high → distorts the recovered J magnitude.
- **[IMPL] a differentiable morphology term** — `loss.weights.morphology` is currently INERT:
  `losses/terms.py::morphology_consistency` is a non-differentiable numpy diagnostic and is not
  in `losses/total.py::compute_terms`. To use the ~0.1 weight, add a differentiable morphology
  term (e.g. spectral anisotropy of the lifted field) and wire it into `compute_terms`.
- **[VALIDATE] every loss term** — `terms.py` is repackaged from the SCAFFOLD objective. Confirm
  per-term: gradient flows to θ, and the term's zero sits where the design doc says
  (`tests/test_science.py` has the dispersion/Jacobian anchors; add per-term zero-location tests).

## Stage 3 — optimisation (inner + outer)

- **[TUNE] inner schedule** — `train.{n_restarts, adam_steps, adam_lr, lbfgs_steps, grad_clip}`
  (`config.py::TrainConfig`). Restarts matter because the inverse problem is multi-modal.
- **[IMPL] GradNorm weighting** — `losses/weighting.py::GradNormWeighting.combine` is a stub that
  runs with fixed weights. Implement the gradient-magnitude balancing update (backprop each term,
  measure ‖∂θ(w_k L_k)‖, nudge weights toward equal magnitude every `update_every` steps).
- **[IMPL] NTK weighting** — `losses/weighting.py::NTKWeighting.combine` stub. Implement the
  NTK-trace / residual-decay-rate estimator and set weights inversely (handles the dissimilar
  per-term convergence rates, incl. structural-parameter sensitivity — the inverse-problem case).
- **[TUNE] seed replicates** — `optim/sweep.py::run_sweep(seeds=...)`. Rank configs by a seed
  aggregate, never one init. `sweep_example.yaml` shows the axis format.

## Stage 4 — validation / milestones (validate.py, train.fit)

- **[TUNE] Milestone 1 (N=m=2)** — `configs/milestone1_{gm,schnak}.yaml`. Get recovery to land
  in-regime and match k* within ~15% on both reference frames. This is where you tune §2–3 knobs.
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
