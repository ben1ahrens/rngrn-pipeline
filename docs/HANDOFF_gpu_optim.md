# HANDOFF — `feature/gpu-optim-repair` (GPU-first pipeline work, Phase A of R3)

Rewritten 2026-08-19 at Phase A close-out, for a zero-context reader. Supersedes the prior
version of this file, which described the PRE-repair state (`2b11010`, 2026-08-18) and is no
longer accurate — that state is preserved in git history, not restated here.

## Verdict

**APPROVED FOR INTEGRATION** as Phase A of `PLAN_redesign_R3` (2026-08-19). This moves the
branch verdict from the prior review's "unsafe to integrate as-is" (`docs/REVIEW_gpu_optim_delta.md`,
2026-08-18, 3 Critical / 6 Important / 10 Minor) to approved, conditional on the items still
open below. The three Critical findings (C1, C2, C3) are repaired and pinned by tests; every
one of the 8 undocumented semantic changes the prior review found now has a `docs/DECISIONS.md`
entry. What remains open is recorded honestly rather than closed silently — see "Still open"
below.

## What this branch is

Sits on `10cff1b` ("declutter to the fft-training minimum"). GPU-optimises the training
pipeline, including the item `docs/REDESIGN_rngrn.md` §4.1 names as R3's critical path:
batching the spectral terms and the forward solve. The original five commits
(`10cff1b..2b11010`, all 2026-08-17) were interrupted mid-session with no plan or ledger; this
branch (`feature/gpu-optim-repair`) is the repair pass ordered by the 2026-08-18 review.

## Critical findings — repaired and pinned

- **C1 (deleted batched-spectral refusal left the suite red and one test vacuous).** Repaired
  in `72b471f` and `48105ad`: batched-spectral input validation restored, the refusal test
  de-vacuated. Pinned by `tests/test_ignition_gating.py`.
- **C2 (`BatchedPatternSolve.backward`'s adjoint gradient scatter had zero test coverage).**
  Repaired/tested in `e8e6927` and `7942fe8`: gradient-equivalence tests against finite
  differences and against the serial path, isolating the gradient scatter. Pinned by
  `tests/test_batched_forward_solve.py`.
- **C3 (`observables.raps_torch` promoted to the relax saturation detector against its own
  docstring, no pinning test).** Repaired in `df45033`: the detector swap is recorded
  (D-OBS-1) and a real defect found in the process — `_raps_torch_bins` binned with `floor`
  instead of `raps`'s own `np.digitize`, which diverged up to 30.2% on k* at the detector's
  operating point; fixed to `digitize`, closing the divergence to round-off (~1.6e-15).
  Pinned by `tests/test_raps_torch_parity.py` (16 tests).

Further equivalence coverage for the batched spectral terms and forward-solve surface was
added in `f939cf7` (`tests/test_batched_spectral_terms.py`, more of
`tests/test_batched_forward_solve.py`, `tests/test_determinism.py`).

## Semantic changes — all 8 now recorded

The prior review found 8 semantic changes with only 1 `docs/DECISIONS.md` entry
(D-PERF-3, the `dispersion_backend='auto'` default). All 8 now have entries:

- **D-PERF-3** — `dispersion_backend` defaults to `'auto'` (pre-existing, from the original
  branch).
- **D-PERF-4** — the batched-spectral refusal stays deleted; explicit entry-point validation
  replaces what it used to provide implicitly.
- **D-PERF-5** — `_lsmr_torch`'s three departures from scipy `lsmr` (stop-check cadence,
  branchless Givens rotation, Krylov-breakdown handling) stay as landed.
- **D-PERF-6** — the ETDRK4 blow-up check moved from per-step to per-call; the numpy/torch
  return-array parity break on a blow-up is now recorded in
  `relax_to_pattern_torch`'s docstring.
- **D-PERF-7** — the liveness-sync cadence (25 steps) stays; its prior "harmless" claim is
  corrected, not the cadence.
- **D-PERF-8** — `lbfgs_error` recording replaces a bare `except: pass`; the serial-only
  asymmetry against `_batched_restarts` is recorded, not closed.
- **D-PERF-9** — `integrate_bdf1_newton_krylov`'s stub now raises instead of silently falling
  back to ETDRK4.
- **D-OBS-1** — the relax saturation detector's k* estimator swap to `kstar_of_torch`, plus
  the `digitize` binning fix (see C3 above).

## Suite

**660 passed / 24 skipped / 0 failed**, unsandboxed, this worktree's own `.venv`. See the
integration commit for the exact wall-clock duration recorded at verification time.
`payload.h5` reads require the sandbox disabled — a sandboxed run reports spurious
`PermissionError`s that are not code defects (`CLAUDE.md` §3).

## Still open — not closed by this phase

- **M7 — cost split is UNMEASURED.** The batched spectral path's per-step device→host sync
  cost relative to the serial path has not been measured. → R3 Task 17.
- **M8 — `_half_coeffs_batched` has no size guard.** Its contour intermediate is ~230 MB
  complex128 at b=32/n=96, per batched solve, per Adam step, with no cap and no
  `MemAvailable` interaction (the binding resource per `CLAUDE.md` §7a). Integration must
  size this before any sweep runs it at scale.
- **I4 — the frozen config records `dispersion_backend: "auto"`, not the resolved backend**
  (`eig` or `cubic`), so a run's config alone does not say which arithmetic path executed.
  → R3 Task 8 decision point.
- **M9 — `active=alive if spec_cfg is not None else None`** (`recover.py` ~:254, ~:305) is a
  conditional that cannot change the value passed (`alive` is always defined); it exists only
  to keep the non-spectral batched row schema unchanged. Recorded as review M9, left as a
  comment rather than removed — unresolved by choice, not by oversight.

## Where to start

1. Read `docs/REVIEW_gpu_optim_delta.md` for the full original findings (3 Critical / 6
   Important / 10 Minor) and `docs/DECISIONS.md` (D-PERF-4..9, D-OBS-1) for how each was
   resolved or explicitly left open.
2. This branch does **not** contain R2's T8–T15 loss-term registry refactor; the two lines
   collide in `losses/terms.py`, `losses/total.py`, `recover.py` (enumerated in the review,
   §10). Integration onto the redesign line still needs to resolve those collision points.
3. The owner's standing directive: the whole pipeline is GPU-first by default (REDESIGN §4.1).
   This branch is Phase A of getting there under `PLAN_redesign_R3`.
