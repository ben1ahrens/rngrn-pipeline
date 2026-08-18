# HANDOFF — `feature/gpu-optim` (GPU-first pipeline work, INTERRUPTED)

**DO NOT ASSUME COMPLETE.** The session that built this branch was interrupted by the owner
mid-execution on 2026-08-17 and left no plan, ledger, stash, or report — the five commits are
the only record of intent. Written 2026-08-18 by the redesign controller after a verification
pass ordered by the owner ("verify now, integrate at R3").

## What this branch is

Five commits (`10cff1b..2b11010`, all 2026-08-17) that GPU-optimise the training pipeline —
including the item the ratified `docs/REDESIGN_rngrn.md` §4.1 names as R3's critical path:
**batching the spectral terms and the forward solve** (both were previously refused by name in
`compute_terms_batched`).

| Commit | What | Completeness (reviewed) |
|---|---|---|
| `346bbf2` | desync losses hot path; one Jacobian per step; hoist constants | **Complete** — verified arithmetic-neutral |
| `e7d01bd` | `auto` dispersion backend (cubic at N=3); liveness cadence; LBFGS errors recorded | Mostly complete — mechanism verified; missing test + 2 decision entries |
| `375cb4a` | docs: D-PERF-3 (the backend default) | Complete |
| `8f93dfe` | forward-solve desync; cached grid constants; device-resident spectral geometry | **Cut short on evidence** — 3 undocumented training-time semantic changes |
| `2b11010` | batched `PatternSolver` — spectral training unlocked on the batched path | **Clearly cut short** — ~1,000 lines, zero tests, suite left red |

Lineage: sits on `10cff1b` ("declutter to the fft-training minimum"), the same base the
redesign docs branch grew from. It does **not** contain R2's T8–T15 loss-term registry
refactor; the two lines collide in `losses/terms.py`, `losses/total.py`, `recover.py`
(26 collision points enumerated in the review, §10).

## What is VERIFIED (2026-08-18)

- **Suite @ `2b11010`** (unsandboxed, worktree venv): **2 failed / 605 passed / 24 skipped**
  in 7:18. Both failures are `tests/test_ignition_gating.py::test_{compute_terms,total_loss}_batched_refuses_spectral`
  — they pin the old spectral-refusal contract that `2b11010` deliberately removed, and were
  never updated. Truncation signature, not rot: nothing pre-existing broke.
- **Firewall (§5): CLEAN.** No forbidden name in the diff; every heavily-touched module is
  already on `RECOVERY_SIDE`; no new `scripts/` module.
- **`auto` dispersion backend: correct and fail-loud** (N=2→eig, N=3→cubic, N=4→eig,
  explicit cubic@N≠3 raises; resolved value written to checkpoints).
- **Batched spectral terms probed at ≤ 4.3e-14 relative vs serial** (reviewer measurement,
  CPU float64 — a probe, not a committed test).
- **`_damped_newton` vectorised line search: bit-identical** over 40 seeded models
  (reviewer measurement; costs ~12 % on CPU, undocumented).

## What is NOT verified — the reason the review verdict is "unsafe to integrate as-is"

Full findings: `docs/REVIEW_gpu_optim_delta.md` (3 Critical, 6 Important, 10 Minor).
The three load-bearing items:

1. **C1** — the deleted refusal left the suite red (the 2 failures above) **and** left
   `test_recover_raises_on_batched_with_a_spectral_weight` passing *vacuously* (it now trips
   the LBFGS guard whose message also says "batched"). Repairing only the red tests leaves a
   false green behind.
2. **C2** — the entire batched forward solve (~1,000 lines: `BatchedPatternSolve/Solver`,
   `relax_to_pattern_torch_batched`, `make_spatial_F_batched`, …) has **zero tests**. The
   load-bearing gap is `BatchedPatternSolve.backward`'s adjoint gradient scatter — a wrong
   scatter would not crash, it would train every member on a blend of the others' gradients.
3. **C3** — `observables.raps_torch` was promoted to the relax saturation detector on two new
   call paths while its own docstring says "not PINNED by a test — add one before any further
   caller relies on it", with a measured 14 % one-bin divergence in exactly the noisy regime
   the detector runs in.

Also: **8 semantic changes, only 1 recorded in `docs/DECISIONS.md`** (D-PERF-3). The missing
entries — LSMR stop cadence, ETDRK4 per-call blow-up check, relax detector swap, bdf1 raise,
liveness cadence, LBFGS error recording, the refusal deletion — are the most systematic gap
after the missing tests. And `forward.py:31`'s cost-split numbers cite "the unit's report",
which does not exist; treat them as UNMEASURED.

## Where to start

1. Read `docs/REVIEW_gpu_optim_delta.md` end to end — especially §9 (possibly-incomplete list:
   where the interruption probably cut) and §10 (the R2 collision points).
2. Minimum to lift the verdict from "unsafe" to "needs fixes": repair C1 (both red tests AND
   the vacuous third), add the `BatchedPatternSolve.backward` gradient-equivalence test, pin
   `raps_torch` (C3).
3. Integration is planned for **R3** (`PLAN_redesign_R3.md`, unlocked by R2's T16 measurement):
   this branch is the base *implementation source*, integrated onto the redesign line with the
   REDESIGN-mandated per-twin equivalence tests — not merged wholesale.
4. The owner's standing directive: the whole pipeline is GPU-first by default
   (REDESIGN §4.1). This branch is the furthest anyone has gotten toward that.
