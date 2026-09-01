# HANDOFF — the R3 redesign line (`feature/r3-integration`)

Written 2026-09-01 at the Task 22 close, for a **zero-context reader**. Every claim below
names its artifact; read the artifact, not this file, before repeating a number
(`.claude/rules/reporting-numbers.md`).

## What this branch is

The R3 phase of the redesign: the gradient machinery that lets spectral (Fourier-target)
training run at population scale. Planned in `docs/PLAN_redesign_R3.md` (22 tasks, on the
`docs/redesign-rngrn` branch until merged), executed 2026-08-12 → 2026-09-01, ledgered in
`worktrees/redesign-doc/.superpowers/sdd/PLAN_redesign_R3/progress.md` (untracked; on-disk
in that worktree). Base: main @ 48441e4.

## What changed, by phase

- **Phase A–B (infrastructure + batched forward).** `src/rngrn/forward.py`: the serial
  IFT/adjoint pattern solve (`PatternSolver`/`PatternSolve`, D1-verified) plus the BATCHED
  twin (`BatchedPatternSolver`/`BatchedPatternSolve`) — batched ETDRK4 relax
  (`_half_coeffs_batched`), per-member Newton polish, per-member LSMR adjoint.
  `src/rngrn/etdrk4_torch.py` (torch integrator, D2-verified against `eval/numerics.py`),
  `src/rngrn/solve_box.py` (Task 11: adaptive commensurate solve box), batched spectral
  loss terms (`losses/spectral.py`, `losses/term_registry.py`), batched steady-state
  Newton (`losses/terms.py`). The gpu-optim-repair branch was integrated by MANUAL
  TRANSPLANT (Task 9) — do not also git-merge `feature/gpu-optim-repair`.
- **Phase C (gradient paths).** `src/rngrn/unrolled.py` (Task 12): the truncated
  differentiable ETDRK4 segment, `SEGMENT_STEPS_DEFAULT = 128` (D-R3-2, calibrated
  in-regime only; rationale corrected by D-R3-3). `recover.py`: `gradient_path`
  selection, the stall-switch solver (`_spectral_solve_with_stall_switch`), Task 13's
  adjoint-primary switch. **This is the first branch on which training simulates** —
  only inside the unrolled segment; the dispersion path (`model.py::dispersion`,
  `recover._kgrid_for`) is untouched (Task 22 numerics review).
- **Phase D (measurement + promotion).** Task 14 FD A/B (`scripts/r3_fd_ab.py`,
  D-R3-1/D-R3-5), Task 15 paired-IC RAPS fidelity (`scripts/r3_raps_fidelity.py`,
  D-R3-4/D-R3-6), Task 16 stall survey (`scripts/r3_stall_survey.py`, D-R3-7), Task 17
  B/K cost curve (`scripts/r3_batch_curve.py`, D-R3-8). **The item-8 PROMOTION**
  (owner-ruled, D-R3-5): `train.gradient_path` defaults to `"unrolled"` for every member;
  the adjoint is the retained A/B-verification path. A0 is IMMUNE by measurement, not
  config: its five spectral weights are 0.0 so no solver is built —
  `tests/test_gradient_path.py` pins bit-identity under both estimators plus a tripwire
  that fails if any weight is raised.
- **Phase E (docs + audits).** `docs/DIAGNOSTICS_r3.md` (the T14–T17 collation),
  D-R3-9 (register-item-14 status roll-up), D-R3-10 (correcting note), this file.

## The refusal surface (all loud, all tested — `tests/test_gradient_path.py`)

- `stall_switch=True` + `gradient_path="unrolled"` → ValueError (nothing to fall back to).
- `stall_switch=True` + `batched=True` → ValueError (no batched unrolled twin).
- `gradient_path="unrolled"` + `batched=True` + non-zero spectral weight → ValueError:
  **a batched spectral run must select `gradient_path="adjoint"` deliberately** until the
  batched unrolled twin exists (open, un-commissioned R4 gap).
- Adjoint residual above `_ADJOINT_RESIDUAL_TRIPWIRE=1e-8` → RuntimeError in the bridge
  backward, serial and batched (D-FFT-10 class; now test-pinned, Task 22).

## What is VERIFIED

- **Suite:** 810 passed / 24 skipped at the Task 21 merge (13ec8d6), + the Task 22
  fix-round tests (see that commit's message for the exact reconciled count), all
  unsandboxed against this worktree's venv.
- **Task 22 whole-branch audits (2026-09-01, all four reports in the session ledger):**
  merge-damage-hunter — CLEAN (no duplicate keys/shadowed names; the three historically
  lost values and every promotion value verified by line history); firewall-auditor —
  CLEAN, 23/23 (main's red `eval/lifted.py` classification test is FIXED here);
  evidence-auditor — no Critical/High, its Mediums fixed in the Task 22 fix round;
  numerics-reviewer — no Critical/High: batched contour coefficients match the serial
  reference to 1e-12, unrolled gradients FD-verified (1e-9..3e-7, reviewer's own check),
  LSMR verified against scipy, no silently trusted unconverged adjoint. Its Medium (the
  `THETA_NAMES` pinned-model collision in the IFT bridges) is FIXED: the bridges now
  iterate the model's registered thetas (`forward._registered_theta_params`), test-pinned
  on serial and batched pinned models.
- **Equivalence/parity inventory:** `tests/test_batched_forward_solve.py`,
  `tests/test_batched_spectral_terms.py` (member-by-member vs serial, FD on the batched
  backward), `tests/test_unrolled_grad.py` (checkpointed ≡ plain to 1e-14),
  `tests/test_etdrk4_torch.py`, `tests/test_stall_accounting.py` (estimator routing).
- **Data check (Task 22 step 6):** no real (non-symlink) `payload.h5` in any R3 worktree;
  nothing to harvest.

## What is NOT established

- **B and K are NOT ruled.** D-R3-8 is PROPOSED: evidence returned to the owner
  (wall-clock, not memory, bounds B on this host; largest measured B=64 at 1 thread,
  2298 s per Adam step; superlinear per-member cost). B=512/K=16 stay UNCALIBRATED
  (register item 14; D-R3-9 has the per-number status).
- **The batched unrolled twin does not exist.** Batched + spectral requires explicit
  `gradient_path="adjoint"`; building the twin is the standing R4 direction, not
  commissioned.
- **Adjoint error in the MILD-miss regime (Newton residual 1e-7..1e-5)** is a scoped R4
  gap (Task 16 found genuine stalls there, milder than Task 14's forced probe).
- **reps=1 noise:** every Task 17 cell is a single measurement; the curve's trend is
  load-bearing, its fine structure is not (D-R3-8 states this; the B=8→16 doubling
  measured 3.68× vs the ~2.2–2.6× band).
- **The estimator run-index columns are ABSENT unless a switch-aware spectral solve was
  possible** (045658a ruling). Edge: an adjoint-primary spectral run without
  `stall_switch` records no `gradient_path` column — the frozen config is the source of
  truth for what was configured (accepted-as-implemented, T17-era fix round).
- **CPU backend note:** a stall-switch on/off A/B on CPU is not bit-comparable (numpy vs
  torch relax can select translationally different patterns; values unaffected) — see
  `recover._spectral_solve_with_stall_switch` and `forward.PatternSolver._relax`.
- Dry-run/plumbing checks aside, **no recovery-quality claim is made anywhere on this
  branch** — it built and verified machinery; tuned-recovery results are R4's to earn.

## Where to start

1. `docs/DIAGNOSTICS_r3.md` — the measured findings, with caveats attached.
2. `docs/DECISIONS.md` D-R3-1 … D-R3-10 — every ruling, its evidence, what was rejected.
3. The SDD ledger: `worktrees/redesign-doc/.superpowers/sdd/PLAN_redesign_R3/progress.md`
   (+ `task-1{4,5,6,7}-report.md` beside it) — the full order of events, reviews, and
   owner directives.
4. `configs/base.yaml` `loss.weights` — the A0 pin (five spectral zeros) and its comment.

**Do not merge to main without the owner's validation** (Task 22 step 10; the owner
rulings roll-up carries the open items).
