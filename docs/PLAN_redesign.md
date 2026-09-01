# RNGRN Redesign Implementation Plan (R1 + R2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking. Dispatch briefs per
> `.claude/rules/orchestration.md` — including the decision-point return block and the
> sandbox clauses — and route models per `CLAUDE.md` §11.

**Goal:** Implement milestones R1 (lift validation ladder) and R2 (model reconstruction +
observability) of the ratified redesign, on the single target `turing_labyrinth/sample_0000`.

**Architecture:** R1 validates the finite-μ lifted system (V0–V4 ladder, GPU port of the
lifted integrator) so the §5.4 gate may later bind; R2 rebuilds the model side
(fixed-point pinning, box reparameterization, `kstar_si`), adds the observability layer
(loss registry, telemetry, viz, notebooks), and produces the first D5-comparable
Phase-I population measurement.

**Tech Stack:** Python ≥3.10, torch 2.13 (CUDA, float64 default), numpy/scipy, pytest;
per-worktree `.venv`; runs under `experiments/` via `bash scripts/guarded_run.sh`.

**Spec:** `docs/REDESIGN_rngrn.md` (ratified 2026-08-17; register in §8) — read it before
any task. Decision records: `docs/DECISIONS.md` D-LIFT-1, D-REDESIGN-1, D-REDESIGN-2.

## Plan-of-plans (scope ruling)

This document plans **R1 and R2 only**. R3–R5 are measurement-gated by the spec and get
their own plan documents when their inputs exist:

- `PLAN_redesign_R3.md` — **written and executed** (2026-08; 22 tasks in 5 phases, on
  the `docs/redesign-rngrn` branch; results collated in `docs/DIAGNOSTICS_r3.md`).
  Was unlocked, as planned here, by R2's throughput/RSS curve and ignition-rate
  measurement and the spec §4.1 batched-implementation surface. *(Pointer updated
  2026-09-01, R3 Task 21.)*
- `PLAN_redesign_R4.md` — unlocked by R3's FD A/B report, stall-rate distribution, and
  96²-vs-512² RAPS fidelity measurement; includes the D-FFT-14 weight-noise arm, whose
  schedule/magnitude must be pre-registered in `docs/DECISIONS.md` before that plan runs.
- `PLAN_redesign_R5.md` — unlocked by R4's campaign report and owner review.

## Global Constraints

- **Never commit to `main`.** One unit → one worktree at `../worktrees/<slug>` → one
  `feature/`-prefixed branch (create per `.claude/skills/new-worktree`, including the
  per-worktree `.venv` from `/home/benja/.claude-science/conda/envs/python/bin/python3.11`
  and `bash scripts/link_payloads.sh`).
- **Run `pytest` with the sandbox DISABLED** — a sandboxed run reports ~15 fake
  `PermissionError`s (payload.h5 read-deny). Full-suite runs take ~5 min.
- **Any trainer/sweep/experiment launch goes through `bash scripts/guarded_run.sh`**
  (`.claude/hooks/guard_trainer.py` refuses anything else).
- **Every new module under `src/rngrn/losses/` or `src/rngrn/eval/` must be classified**
  in `tests/test_firewall.py` (`RECOVERY_SIDE`/`SCORING_SIDE`) — the completeness test
  fails otherwise. **Any new `scripts/` module that opens `payload.h5` or reads an
  `AnswerKey` must be appended to `FORBIDDEN`** (the completeness test is blind to
  `scripts/`).
- **A0 is untouchable:** the baseline objective (`kstar_anchor`, `turing_hinges_split`,
  `frame_scale_anchor` w=2.0, `anticollapse` w=0.5, softplus/exp parameterization) must
  keep bit-identical behaviour after every task. New behaviour is opt-in via config.
- **UNCALIBRATED discipline:** every new threshold carries the code-site comment naming
  its calibrator (spec §8 item 14) — no silent numbers.
- **Numbers leaving a run directory pass `.claude/rules/reporting-numbers.md`.**
- **Validation grid is 512² on the data box** — no cheaper grid is licensed for any
  morphology or gate claim (D-FFT-9 closure 3).
- torch default dtype is float64 (`torch.set_default_dtype` is global in this package).

---

# Milestone R1 — the lift validation ladder (worktree `../worktrees/lift-ladder`, branch `feature/lift-ladder`)

R1's output is a diagnostics campaign, in the house pattern of `docs/DIAGNOSTICS_fft.md`:
library code in `src/rngrn/eval/`, thin drivers in `scripts/`, run artifacts under
`experiments/lift_ladder/<rung>/`, findings in a new `docs/DIAGNOSTICS_lift.md`, and
measured-tolerance closures recorded in `docs/DECISIONS.md`. No rung licenses a recovery
claim; the ladder licenses the *gate*.

### Task 1: Make the `bdf1_newton_krylov` fallback loud

**Files:**
- Modify: `src/rngrn/eval/numerics.py` (function `integrate_bdf1_newton_krylov`)
- Test: `tests/test_numerics_contract.py` (create)

**Interfaces:**
- Produces: `integrate_bdf1_newton_krylov(...)` now raises `NotImplementedError` instead
  of silently returning `integrate_etdrk4(...)`. The `INTEGRATORS` registry entry stays
  (callers must see the loud failure, not a KeyError).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_numerics_contract.py
import pytest
from rngrn.eval import numerics

def test_bdf1_stub_raises_rather_than_silently_delegating():
    # D-REDESIGN spec §5.2: a stub that silently returns another integrator's
    # result is the D-EVID-2 defect class. It must raise until implemented.
    with pytest.raises(NotImplementedError, match="bdf1_newton_krylov"):
        numerics.INTEGRATORS["bdf1_newton_krylov"](
            None, None, None, n=16, L=10.0, dt=0.1, steps=1
        )
```

- [ ] **Step 2: Run it — must FAIL** (the stub currently returns an ETDRK4 result or
  errors differently): `pytest tests/test_numerics_contract.py -v` (sandbox disabled).
- [ ] **Step 3: Implement** — replace the fallback body:

```python
def integrate_bdf1_newton_krylov(*args, **kwargs):
    """STUB — deliberately loud. docs/REDESIGN_rngrn.md §5.2 removed the silent
    ETDRK4 fallback (a result labelled bdf1 that was not bdf1). Implemented at the
    gate milestone (R4 plan) as the 128² cross-check integrator."""
    raise NotImplementedError(
        "bdf1_newton_krylov is not implemented; it previously fell back to ETDRK4 "
        "silently. Use 'etdrk4'/'etdrk4_rfft', or implement the BDF1 scheme."
    )
```

Keep the exact existing signature when replacing the body; adapt the test call to that
signature if it differs from the sketch above.
- [ ] **Step 4: Run the new test (PASS) and grep the repo for callers**
  (`grep -rn "bdf1" src/ tests/ scripts/ notebooks/`) — if any caller depended on the
  silent fallback, that is a decision point: report it, do not patch around it.
- [ ] **Step 5: Full suite (sandbox disabled), then commit**
  `git commit -m "fix: bdf1_newton_krylov stub raises instead of silently running ETDRK4"`

### Task 2: V0 — algebraic invariants at N=3, both forms, ≥20 draws; measure the residual floor

**Files:**
- Create: `src/rngrn/eval/ladder.py` (library home for all ladder rungs)
- Modify: `tests/test_firewall.py` (add `"eval/ladder.py"` to `RECOVERY_SIDE`)
- Test: `tests/test_lift_ladder.py` (create)

**Interfaces:**
- Produces: `ladder.v0_invariants(models: list, mus: list[float]) -> dict` returning
  `{"max_fixed_point_residual": float, "max_rescale_mu_err": float,
  "max_gate_qss_err": float, "n_models": int}`. Model draws come from
  `ladder.draw_models(n: int, form: str, seed: int) -> list[RNGRN]` — random inits via
  the existing `RNGRN(N=3, form=form, init="low_basal", seed=...)` constructor,
  filtered to those with a converged steady state (`losses.terms.steady_state`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lift_ladder.py
import pytest
from rngrn.eval import ladder

@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_v0_invariants_hold_at_the_existing_bar(form):
    models = ladder.draw_models(n=20, form=form, seed=101)
    out = ladder.v0_invariants(models, mus=[1e-6, 1e-4, 1e-2, 1.0, 1e2])
    assert out["n_models"] >= 20
    # Bar = the bar the existing tests already hold (tests/test_lifted.py: 1e-7).
    # The achievable floor is UNCALIBRATED and is MEASURED by the v0 script, not
    # asserted here — docs/REDESIGN_rngrn.md §5.3 V0.
    assert out["max_fixed_point_residual"] < 1e-7
    assert out["max_rescale_mu_err"] < 1e-9
    assert out["max_gate_qss_err"] < 1e-9
```

- [ ] **Step 2: Run — FAIL** (`ladder` does not exist).
- [ ] **Step 3: Implement `ladder.py`** — reuse, do not reimplement:
  `lifted.fixed_point_residual`, `lifted.rescale_mu` vs `lifted.lifted_jacobian`,
  `lifted.gates_qss`/`production_from_gates` vs `model.reaction`. `draw_models` loops
  seeds, keeps models whose `steady_state` converges, until `n` collected (raise after
  10n attempts — fail loud). Module docstring states PROVENANCE: ladder scaffolding,
  validated by `tests/test_lift_ladder.py`.
- [ ] **Step 4: Run the test file (PASS), then the firewall test**
  (`pytest tests/test_firewall.py -v`) — it must fail before the `RECOVERY_SIDE` edit
  and pass after; that failure is the completeness test working.
- [ ] **Step 5: Commit** `git commit -m "feat: lift-ladder V0 — algebraic invariants over ≥20 draws, both forms"`

### Task 3: V1 — eigenvalue continuation of all 21 branches, tracked by eigenvector overlap

**Files:**
- Modify: `src/rngrn/eval/ladder.py`
- Test: `tests/test_lift_ladder.py`

**Interfaces:**
- Produces: `ladder.v1_continuation(model, kgrid: np.ndarray,
  mus=(1e-7, 1e-6, 1e-5)) -> dict` with keys `{"slow_branch_order": float,
  "kstar_grid_offset": int, "fast_branches_stable": bool, "max_slow_err": dict}`.
  Branch pairing between adjacent μ values uses eigenvector overlap
  (`|v_i(μ_a)^H v_j(μ_b)|` maximum-overlap assignment), never Re-sorting.

- [ ] **Step 1: Write the failing test**

```python
def test_v1_slow_branches_converge_to_qss_at_first_order():
    m = ladder.draw_models(n=1, form="competitive", seed=7)[0]
    kgrid = ladder.default_kgrid(m)          # reuse eval.analysis scan band
    out = ladder.v1_continuation(m, kgrid)
    assert out["fast_branches_stable"] is True
    assert abs(out["slow_branch_order"] - 1.0) < 0.35   # empirical order ≈ 1
    assert out["kstar_grid_offset"] <= 1                 # within one grid point
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — per k: assemble the lifted Jacobian via
  `lifted.rescale_mu` from one μ=1 Jacobian; `numpy.linalg.eig`; pair branches across
  the μ triplet by eigenvector overlap; slow-branch error `e(μ) = |λ_slow(μ) − λ_qss|`,
  order = mean log-slope of e across the triplet; fast branches must satisfy
  `Re λ < 0` and scale ~1/μ. `default_kgrid` wraps the existing analysis scan band.
- [ ] **Step 4: Run (PASS).**
- [ ] **Step 5: Commit** `git commit -m "feat: lift-ladder V1 — 21-branch eigenvalue continuation with overlap tracking"`

### Task 4: V2 — 0-D lifted ODE vs Radau reference; Strang stepper order and error

**Files:**
- Modify: `src/rngrn/eval/ladder.py`
- Test: `tests/test_lift_ladder.py`

**Interfaces:**
- Produces: `ladder.v2_temporal(model, mu: float, T: float, dts: list[float]) -> dict`
  with `{"strang_order": float, "sup_err_at_horizon": float, "radau_ok": bool}`;
  0-D lifted RHS built from `lifted`'s gate ODEs with the diffusion term dropped;
  reference `scipy.integrate.solve_ivp(method="Radau", rtol=1e-10, atol=1e-12)`.

- [ ] **Step 1: Failing test**

```python
@pytest.mark.parametrize("mu", [1e-5, 7.2e-4, 1e-2])
def test_v2_strang_matches_radau_and_shows_order_two(mu):
    m = ladder.draw_models(n=1, form="competitive", seed=11)[0]
    out = ladder.v2_temporal(m, mu=mu, T=5.0, dts=[mu/2, mu/4, mu/8])
    assert out["radau_ok"]
    assert abs(out["strang_order"] - 2.0) < 0.5          # dt ≲ mu regime
    # ≤ 0.1 × pattern_floor with x*-free 0-D floor: use the absolute arm, 1e-4
    assert out["sup_err_at_horizon"] < 1e-4
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — the 0-D Strang step reuses the *same* exact gate substep
  as `simulate_lifted` (call the existing helper; do not re-derive the rank-one
  exponential) with the x-substep as plain RK4 on the 0-D reaction (no FFT in 0-D).
  Order = log2(err(dt)/err(dt/2)) averaged over the halving pairs.
- [ ] **Step 4: Run (PASS).**
- [ ] **Step 5: Commit** `git commit -m "feat: lift-ladder V2 — 0-D Strang vs Radau, measured order"`

### Task 5: GPU port of the lifted integrator

**Files:**
- Create: `src/rngrn/eval/lifted_torch.py`
- Modify: `tests/test_firewall.py` (add `"eval/lifted_torch.py"` to `RECOVERY_SIDE`)
- Test: `tests/test_lifted_torch.py` (create)

**Interfaces:**
- Consumes: `etdrk4_torch` (existing GPU ETDRK4 x-substep), `lifted.py`'s gate algebra.
- Produces: `lifted_torch.simulate_lifted_torch(model, L, n, mu, dt, steps, seed,
  device="cuda") -> dict` with the same return contract as `lifted.simulate_lifted`
  (fields, `stopped_reason`, amplitude), float64 throughout.

- [ ] **Step 1: Failing equivalence test (CPU-vs-numpy first — runs anywhere)**

```python
# tests/test_lifted_torch.py
import numpy as np
from rngrn.eval import lifted, lifted_torch
from rngrn.eval.ladder import draw_models

def test_torch_lifted_matches_numpy_on_cpu_small_grid():
    m = draw_models(n=1, form="competitive", seed=23)[0]
    kw = dict(L=20.0, n=32, mu=1e-3, dt=5e-4, steps=400, seed=5)
    a = lifted.simulate_lifted(m, **kw)
    b = lifted_torch.simulate_lifted_torch(m, device="cpu", **kw)
    np.testing.assert_allclose(b["field"], a["field"], rtol=1e-10, atol=1e-12)
```

Adapt argument names to `simulate_lifted`'s actual signature — read it first; the
equivalence contract (same dt, same seed, same noise draw) is the point.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — gate substep is elementwise closed-form (exp in torch);
  x-substep delegates to the existing `etdrk4_torch` machinery; the IC noise draw must
  reproduce the numpy path bit-for-bit (draw with numpy, transfer) so the equivalence
  test is exact rather than statistical.
- [ ] **Step 4: Run (PASS). Then, on the GPU machine only, measure and record cost**
  (ms/step at 512², μ=7.2e-4) into `experiments/lift_ladder/gpu_port/results/cost.json` —
  this number replaces spec §5.2's estimate; do not leave the estimate as the citation.
- [ ] **Step 5: Full suite; commit** `git commit -m "feat: torch lifted integrator, bit-equal to numpy; measured GPU cost"`

### Task 6: V3 spatial + V4 survey drivers, run the campaign

**Files:**
- Modify: `src/rngrn/eval/ladder.py` (`v3_spatial`, `v4_survey`)
- Create: `scripts/lift_ladder.py` (thin CLI driver: `v0|v1|v2|v3|v4 --out experiments/lift_ladder/<rung>`)
- Test: `tests/test_lift_ladder.py` (unit-level only; the 512² campaign is a run, not a test)

**Interfaces:**
- Produces: `ladder.v3_spatial(model, mus, n, L, seed) -> dict`
  (`patterned_agree: bool, morphology_agree: bool, kstar_within_one_bin: bool,
  l2_diff_by_mu: dict`) comparing `simulate_lifted`(/`_torch`) against
  `rollout.simulate` at matched seed/dt/horizon; `ladder.v4_survey(models, mu_band,
  n_mu: int) -> dict` (`p_lifted_given_qss: float, frac_reentrant: float,
  edges: list`) wrapping `lifted.mu_critical` + `lifted.robustness_vs_mu`.
- The script writes one JSON per rung under `experiments/lift_ladder/<rung>/results/`
  plus the arrays the viz needs under `arrays/`.

- [ ] **Step 1: Failing unit tests** for both functions at 32²/N=3 (patterned flags and
  dict shape only — pattern the test on the Task 5 fixture; assert keys exist and
  `l2_diff_by_mu` is monotone non-increasing as μ decreases on the fixture).
- [ ] **Step 2: Run — FAIL. Step 3: Implement. Step 4: Run (PASS).**
- [ ] **Step 5: Run the campaign** (guarded, unsandboxed):
  `bash scripts/guarded_run.sh .venv/bin/python scripts/lift_ladder.py v0 --out experiments/lift_ladder/v0` …through v4.
  V3(a) at μ ∈ {1e-4, 1e-5, 1e-6}, both forms; V3(b) anchor at μ_central with
  dt = min(0.2/jac_rate, μ/2) on 128² (torch) plus 512² at QSS-dt; V4 over every
  Turing-positive draw on hand (`draw_models` harvest + the d5 four).
- [ ] **Step 6: Commit results** (tracked: `results/*.json`, `arrays/*.npz`)
  `git commit -m "data: lift-ladder V0–V4 campaign results"`

### Task 7: DIAGNOSTICS_lift.md + ladder-tolerance closures

**Files:**
- Create: `docs/DIAGNOSTICS_lift.md` (mirror `docs/DIAGNOSTICS_fft.md`'s structure:
  per-rung method, measured numbers with run paths, PASS/FAIL/finding)
- Modify: `docs/DECISIONS.md` (append `D-LIFT-2 — ladder tolerances closed by
  measurement`, recording the measured V0 floor, V1 error constants, V3 field-difference
  curve — closing spec §8 item 14's ladder entries), `docs/REDESIGN_rngrn.md` §5.3
  (UNCALIBRATED marks → measured values with citations)
- **Decision point (return to orchestrator, do not rule):** whether the ladder as
  measured licenses the §5.4 gate — that verdict updates D-LIFT-1's precondition and
  belongs to the orchestrating session, with the numbers in front of it.

- [ ] **Step 1: Write the diagnostics doc from the run JSONs** (every number cites its
  `experiments/lift_ladder/...` path; reporting-numbers rules apply).
- [ ] **Step 2: Append D-LIFT-2; update the spec's §5.3 marks.**
- [ ] **Step 3: Run `evidence-auditor` agent over the new doc; fix findings.**
- [ ] **Step 4: Full suite; commit; push branch; write `docs/HANDOFF_lift_ladder.md`**
  (zero-context reader: what was measured, what passed, what the gate verdict awaits).

---

# Milestone R2 — model reconstruction + observability (worktree `../worktrees/redesign-model`, branch `feature/redesign-model`)

Tasks 8–10 are the observability layer (registry, telemetry, viz); 11–14 the model
changes; 15 the notebooks; 16 the measurement run. Order matters: the registry lands
before new terms so `kstar_si` is born registered.

### Task 8: The loss registry

**Files:**
- Create: `src/rngrn/losses/term_registry.py`
- Modify: `src/rngrn/losses/terms.py` (`DEFAULT_WEIGHTS` derived from the registry),
  `src/rngrn/losses/total.py` (enumerate registry, keep behaviour identical),
  `tests/test_firewall.py` (add `"losses/term_registry.py"` to `RECOVERY_SIDE`)
- Test: `tests/test_term_registry.py` (create)

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class LossTerm:
    name: str
    fn: Callable                      # serial callable, existing signature
    batched_fn: Callable | None       # None requires refusal_reason
    refusal_reason: str | None
    default_weight: float
    calibration: str                  # "CALIBRATED(<source>)" or "UNCALIBRATED"

LOSS_TERMS: Registry[LossTerm]        # rngrn.registry.Registry instance
def default_weights() -> dict[str, float]   # replaces the DEFAULT_WEIGHTS literal
```

- [ ] **Step 1: Failing tests**

```python
# tests/test_term_registry.py
from rngrn.losses import term_registry, terms

def test_every_registered_term_is_fully_classified():
    for key in term_registry.LOSS_TERMS.keys():
        t = term_registry.LOSS_TERMS.get(key)
        assert (t.batched_fn is not None) != (t.refusal_reason is not None), key
        assert t.calibration.startswith(("CALIBRATED(", "UNCALIBRATED")), key

def test_default_weights_are_bit_identical_to_the_legacy_dict():
    # A0 protection: the registry refactor may not change a single default.
    assert term_registry.default_weights() == dict(terms.DEFAULT_WEIGHTS)
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement** — register the existing 12 terms with
  their current defaults and honest calibration tags (`resid`:
  `CALIBRATED(exp06 — measured harmful, off)`; spectral five: `UNCALIBRATED`; etc.);
  spectral terms get `refusal_reason` quoting the existing `total.py` refusal. Rewire
  `DEFAULT_WEIGHTS = default_weights()` at import so the legacy name stays valid.
- [ ] **Step 4: Run new tests + `tests/test_losses.py` (PASS, no behaviour change).**
- [ ] **Step 5: Full suite; commit** `git commit -m "refactor: loss terms move onto a registry; defaults bit-identical"`

### Task 9: TrainingHistory events + invariant trajectories

**Files:**
- Modify: `src/rngrn/history.py`
- Test: `tests/test_history_events.py` (create)

**Interfaces:**
- Produces: `TrainingHistory.record_event(step: int, member: int, kind: str)` with
  `kind ∈ {"ignition","deignition","stall","cull","death"}` (extend the existing
  `record_death` to delegate); `TrainingHistory.record_invariants(step, member,
  invariants: dict[str, float])` for the spec §3.4 set (d_ratio, alpha/delta,
  beta/delta, D·k*²/delta, per-edge occupancy); `to_arrays()` gains `"events"` and
  `"invariants"` keys. Existing array keys and shapes are unchanged (A0 protection).

- [ ] **Step 1: Failing test** — record two events + one invariants row on a
  2-member history; assert `to_arrays()["events"]` round-trips (step, member, kind)
  and legacy keys are untouched.
- [ ] **Step 2–4: FAIL → implement → PASS** (also run `tests/test_history*.py`).
- [ ] **Step 5: Commit** `git commit -m "feat: TrainingHistory population events and invariant trajectories"`

### Task 10: The viz module

**Files:**
- Create: `src/rngrn/viz.py` (top-level, matplotlib-only, reads `to_arrays()` dicts or
  a run directory — imports no data/, no scoring, no answer key)
- Test: `tests/test_viz.py` (create; smoke tests on synthetic arrays, `Agg` backend)

**Interfaces:**
- Produces: `viz.loss_curves(arrays, out_png)`, `viz.invariant_trajectories(arrays,
  out_png)`, `viz.event_timeline(arrays, out_png)`, `viz.spectral_trace(arrays,
  floor: float, out_png)` — each returns the written path; all four take the Task 9
  `to_arrays()` dict.

- [ ] **Step 1: Failing smoke test** (build a minimal fake arrays dict, call all four,
  assert the PNGs exist and are nonempty). **Step 2–4: FAIL → implement → PASS.**
- [ ] **Step 5: Commit** `git commit -m "feat: viz module — loss, invariants, events, spectral-trace figures"`

### Task 11: Frame-mean bias measurement on the target (spec §3.2 precondition)

**Files:**
- Create: `scripts/measure_frame_bias.py`
- Modify: `tests/test_firewall.py` — **append `"measure_frame_bias"` to `FORBIDDEN`**
  (it reads the AnswerKey; scripts are outside the completeness test's view)
- Output: `experiments/redesign_r2/frame_bias/results/bias.json` (tracked)

**Interfaces:**
- Produces: JSON `{"dataset": "turing_labyrinth", "sample": "sample_0000",
  "ratio_mean_over_xstar": [c0, c1, c2], "kstar_obs": float}` — per-channel
  `frame.mean()/x_star_i` from the registry loader + AnswerKey. Scoring-side by
  construction; it must never be imported by recovery code (hence FORBIDDEN).

- [ ] **Step 1: Write the script** (thin: load via `data.registry`, print + write JSON).
- [ ] **Step 2: Firewall test must pass with the FORBIDDEN addition.**
- [ ] **Step 3: Run it (unsandboxed), commit the JSON.**
  **Decision point (return, do not rule):** if the measured bias is far from the
  n=127 legacy figure (median ratio 0.921), whether pinning to raw frame means or to
  bias-corrected means goes back to the orchestrator with the number.
- [ ] **Step 4: Commit** `git commit -m "data: frame-mean vs x* bias measured on the target (spec §3.2)"`

### Task 12: Fixed-point pinning (β solved out)

**Files:**
- Modify: `src/rngrn/model.py`
- Test: `tests/test_pinned_model.py` (create)

**Interfaces:**
- Produces: `RNGRN(..., pin_xstar: Sequence[float] | None = None)` (appended at the end
  of the constructor signature, comment naming this plan). When set: `beta` becomes a
  derived property `beta_i = delta_i * xstar_i - prod_i(xstar)` (autograd-visible,
  no `theta_beta` parameter registered), and `model.beta_hinge() -> Tensor` returns
  `softplus(-beta).sum()` for the loss. When `None`: bit-identical legacy behaviour.

- [ ] **Step 1: Failing tests**

```python
# tests/test_pinned_model.py
import torch
from rngrn.model import RNGRN

def test_pinned_model_has_exact_fixed_point_and_no_free_beta():
    xs = [0.7, 1.3, 1.0]
    m = RNGRN(N=3, form="competitive", seed=3, pin_xstar=xs)
    x = torch.tensor(xs, dtype=torch.float64)
    assert torch.allclose(m.reaction(x), torch.zeros(3), atol=1e-12)
    assert not any(n == "theta_beta" for n, _ in m.named_parameters())

def test_unpinned_model_is_bit_identical_to_legacy():
    a = RNGRN(N=3, form="competitive", seed=3)
    b = RNGRN(N=3, form="competitive", seed=3, pin_xstar=None)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb)
```

- [ ] **Step 2–4: FAIL → implement → PASS** (plus `tests/test_model.py`,
  `tests/test_science.py` — the dispersion/Jacobian contracts must be untouched).
- [ ] **Step 5: Commit** `git commit -m "feat: pin_xstar — fixed point pinned to data, beta derived with hinge"`

### Task 13: Box-sigmoid reparameterization for α and δ

**Files:**
- Modify: `src/rngrn/model.py` (constructor arg `param_boxes: dict | None = None`,
  appended after `pin_xstar`; when given, `alpha`/`delta` map through
  `low + (high-low)*sigmoid(raw)` instead of softplus; boxes loaded by the caller from
  `configs/bio_box.yaml` — the model takes numbers, not file paths)
- Test: `tests/test_box_reparam.py` (create)

- [ ] **Step 1: Failing tests** — (a) with
  `param_boxes={"alpha": (0.5, 10.0), "delta": (0.4, 5.0)}`, every element of
  `m.alpha`/`m.delta` lies strictly inside its box for 50 random seeds; (b) with
  `param_boxes=None`, parameters and outputs are bit-identical to legacy (same pattern
  as Task 12's test); (c) gradients flow (`.alpha.sum().backward()` populates
  `theta_alpha.grad`).
- [ ] **Step 2–4: FAIL → implement → PASS.**
- [ ] **Step 5: Commit** `git commit -m "feat: box-sigmoid reparameterization for alpha/delta (opt-in)"`

### Task 14: `kstar_si` — the scale-invariant anchor, born registered

**Files:**
- Modify: `src/rngrn/losses/terms.py` (add `kstar_anchor_si` + `kstar_anchor_si_batched`),
  `src/rngrn/losses/term_registry.py` (register, weight 0.0, `UNCALIBRATED`)
- Test: `tests/test_kstar_si.py` (create)

**Interfaces:**
- Produces: `kstar_anchor_si(sig: Tensor, kgrid: Tensor, kstar_obs: float,
  temp: float = 60.0, eps: float = 1e-12) -> Tensor` — signature mirrors the existing
  `kstar_anchor` (read it first and match, adding `eps`); σ̄ = σ/(max σ − min σ + eps);
  loss = logsumexp(temp·σ̄)/temp − σ̄(k*_obs). Batched twin over leading dim B.

- [ ] **Step 1: Failing property tests**

```python
# tests/test_kstar_si.py
import torch
from rngrn.losses.terms import kstar_anchor, kstar_anchor_si

def test_kstar_si_is_invariant_under_sigma_rescaling():
    k = torch.linspace(0.05, 4.0, 400, dtype=torch.float64)
    sig = -((k - 1.5) ** 2) + 0.3
    a = kstar_anchor_si(sig, k, kstar_obs=1.0)
    b = kstar_anchor_si(sig * 17.0, k, kstar_obs=1.0)
    assert torch.allclose(a, b, rtol=1e-9)

def test_flattening_does_not_reduce_kstar_si_but_reduces_the_legacy_anchor():
    k = torch.linspace(0.05, 4.0, 400, dtype=torch.float64)
    sig = -((k - 1.5) ** 2) + 0.3
    legacy_gap = kstar_anchor(sig, k, kstar_obs=1.0) - kstar_anchor(sig * 0.1, k, kstar_obs=1.0)
    si_gap = kstar_anchor_si(sig, k, kstar_obs=1.0) - kstar_anchor_si(sig * 0.1, k, kstar_obs=1.0)
    assert legacy_gap > 0          # the measured degenerate direction exists
    assert abs(si_gap) < 1e-9      # and kstar_si closes it
```

Adapt the call signature to `kstar_anchor`'s actual one after reading it; the two
properties are the contract.
- [ ] **Step 2–4: FAIL → implement (serial + batched, batched matching serial to
  1e-12 on a B=3 stack) → PASS.** UNCALIBRATED comments at the `temp`/`eps` sites name
  the R2 sweep as calibrator (spec §8 item 14).
- [ ] **Step 5: Commit** `git commit -m "feat: kstar_si — scale-invariant anchor; flattening no longer profitable"`

### Task 15: The two notebooks

**Files:**
- Create: `notebooks/redesign_pipeline.ipynb`, `notebooks/redesign_plots.ipynb`
- Test: `tests/test_notebooks_are_thin.py` (create — asserts, by parsing notebook JSON,
  that no cell defines a function longer than ~10 lines and no cell imports
  `data.gate`/`AnswerKey`; the "thin drivers" contract of spec §4.8, made checkable)

- [ ] **Step 1: Write `redesign_pipeline.ipynb`** — cells: (1) config + paths;
  (2) provenance printout (git SHA, venv path check from CLAUDE.md §2); (3) Phase-I
  launch cell that shells out `bash scripts/guarded_run.sh .venv/bin/python
  scripts/r2_ignition_run.py --out experiments/redesign_r2/phase1` (never a bare
  trainer call); (4) results summary cell reading the run's `results/*.json`.
- [ ] **Step 2: Write `redesign_plots.ipynb`** — one parameter cell (`RUN_DIR`), then
  one cell per `viz` figure, loading `arrays/` and the tracked indexes only.
- [ ] **Step 3: Write and pass the thinness test; full suite; commit**
  `git commit -m "feat: pipeline + plots notebooks (thin drivers, guarded launches)"`

### Task 16: Phase-I population run — the first D5-comparable numbers

**Files:**
- Create: `scripts/r2_ignition_run.py` (driver: BatchedRNGRN population, ignition terms
  only — `kstar_si` + `turing_hinges_split` + `param_prior` + `beta_hinge`, pinned/boxed
  model; records TrainingHistory with events + invariants; sweeps B ∈ {64, 128, 256, 512}
  for the throughput/RSS curve; A0 control arm with the legacy objective, same seeds)
- Output: `experiments/redesign_r2/phase1/` (tracked indexes, results, arrays)

- [ ] **Step 1: Write the driver** (reuse `recover.py`'s batched-restart machinery for
  the optimizer loop; per-member seeding via the existing `_restart_seed` discipline).
- [ ] **Step 2: Launch** via `bash scripts/guarded_run.sh` (§7a; unsandboxed; watch
  RSS — the §7a guard's `MemAvailable` floor governs).
- [ ] **Step 3: Report** per `.claude/rules/reporting-numbers.md`: per-member ignition
  rate, distinct-sign-structure count, and the B/K throughput-RSS curve — read against
  the D5 row's comparable columns only (spec §1), A0 arm alongside.
  **Decision point (return, do not rule):** the measured B and K values for R3, and
  any surprise in the comparable columns.
- [ ] **Step 4: `evidence-auditor` over the report; full suite; commit; push; write
  `docs/HANDOFF_redesign_r2.md`.**

---

## Self-review (done at authoring)

- **Spec coverage:** R1 tasks 1–7 cover §5.2–§5.3 (loud fallback, V0–V4, GPU port,
  diagnostics + closures); R2 tasks 8–16 cover §4.8 (registry, telemetry, viz,
  notebooks), §3.2 (pinning + bias measurement), §3.3 (boxes), §4.4 (`kstar_si`), and
  §7-R2's deliverables. Deliberately *not* covered here, per the plan-of-plans: §4.1
  batched spectral, §4.2–§4.3 gradient machinery (R3); §4.7 weight noise, §5.4 gate
  runs, §4.6 generator changes (R4 — the generator emission lands with the campaign
  that consumes it); §4.9 hidden channels (R5).
- **Placeholder scan:** the only deferred content is behind explicitly named unlock
  measurements (plan-of-plans), not TBDs inside tasks. Signature sketches that say
  "read the existing signature first and match" are deliberate: the executor verifies
  against source rather than trusting a plan transcription (house rule §8).
- **Type consistency:** `ladder.draw_models` feeds Tasks 2–6; `to_arrays()` dict feeds
  Tasks 9→10→15; `pin_xstar`/`param_boxes` (Tasks 12–13) feed Task 16's driver;
  `kstar_anchor_si` (Task 14) is the name Task 16 uses. Registry `LossTerm` fields are
  used only in Tasks 8 and 14.
