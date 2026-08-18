# RNGRN Redesign Implementation Plan — R3 (gradient machinery)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Dispatch briefs per `.claude/rules/orchestration.md` —
> including the decision-point return block and the sandbox clauses — and route models per
> `CLAUDE.md` §11. This plan is the R3 successor to `docs/PLAN_redesign.md` (R1 + R2); read that
> file's Global Constraints too, they still bind.

**Goal:** Implement milestone R3 of the ratified redesign — the batched spectral terms and the
batched forward solve (§4.1), the two gradient paths and their FD A/B (§4.2), and the adaptive
commensurate solve box (§4.3) — on the single target `turing_labyrinth/sample_0000`. Plus the
ledgered deferrals that R1/R2 handed forward.

**Architecture:** R3 does **not** write the batched forward solve from scratch. It already
exists, on `feature/gpu-optim` @ `bba3e28`, written by an interrupted session and reviewed as
**unsafe to integrate as-is**. R3 therefore runs in five phases: **(A)** repair that branch to
"needs fixes → green" on its own line; **(B)** integrate it *onto* the redesign line
(`feature/redesign-model`) hunk-by-hunk against a 27-point collision map, never as a wholesale
merge; **(C)** build the §4.2–§4.3 gradient machinery on top; **(D)** measure what R4 is gated
on and clear R1/R2's ledgered deferrals; **(E)** docs, auditors, rulings roll-up. No merging to
`main` — the owner validates first.

**Tech Stack:** Python ≥3.10, torch 2.13 (CUDA, float64 default), numpy/scipy, pytest;
per-worktree `.venv`; runs under `experiments/` via `bash scripts/guarded_run.sh`.

**Spec:** `docs/REDESIGN_rngrn.md` (ratified 2026-08-17). R3-binding sections: **§4.1** (what is
batched, what was refused, and the equivalence-test mandate), **§4.2** (the two gradient paths,
A/B discipline, promotion rule = register item 8), **§4.3** (the commensurate solve box,
re-tile hysteresis, stall accounting, the FD gate), **§7** (R3's deliverables), **§8** items 2,
8, 14 (the UNCALIBRATED numbers assigned to R3: the ~20 % stall-switch fraction, the
truncated-backprop segment length). Decision records: `docs/DECISIONS.md` (D-FFT-10, D-FFT-9,
D-PERF-3, D-LIFT-1/2, D-REDESIGN-1..5).

**Base-implementation source:** `feature/gpu-optim` @ `bba3e28`. **Read all three of these
before Task 1, in order:** `docs/HANDOFF_gpu_optim.md` (on that branch — headline is DO NOT
ASSUME COMPLETE), `docs/REVIEW_gpu_optim_delta.md` (same branch; 3 Critical / 6 Important /
10 Minor; **§5** the no-test symbol table, **§6** the semantic-change table, **§9** the
possibly-incomplete list, **§10** the 26 collision points), and the SDD ledger entries tagged
`GPU-OPTIM` in `.superpowers/sdd/PLAN_redesign/progress.md`.

---

## Plan-of-plans (scope ruling)

This document plans **R3 only**. It supersedes `docs/PLAN_redesign.md`'s one-line R3 pointer.

- `PLAN_redesign_R4.md` — unlocked by **this plan's** Task 14 (FD A/B report), Task 15
  (96²-vs-512² RAPS fidelity + target-interpolation error), Task 16 (stall-rate distribution)
  and Task 17 (the B/K curve *with the forward solve in the loop*). Includes the D-FFT-14
  weight-noise arm, whose schedule/magnitude must be pre-registered in `docs/DECISIONS.md`
  before that plan runs, and the §4.6 generator changes.
- `PLAN_redesign_R5.md` — unlocked by R4's campaign report and owner review.

Deliberately **not** in R3: the §4.5 cull and Phase II campaign (R4), §4.6 multi-realization
generation (R4), §4.7 weight noise (R4), §4.9 hidden channels (R5), the §5.4 gate *runs* (R4 —
though R3 Task 20 licenses L2, which the gate needs).

---

## Global Constraints

Everything in `docs/PLAN_redesign.md` §Global Constraints still binds verbatim. Repeated here
because a zero-context controller must not have to open two files to start, plus the six that
are new to R3.

- **Never commit to `main`.** One phase → one worktree at `../worktrees/<slug>` → one
  `feature/`-prefixed branch (create per `.claude/skills/new-worktree`, including the
  per-worktree `.venv` from `/home/benja/.claude-science/conda/envs/python/bin/python3.11` and
  `bash scripts/link_payloads.sh`). Verify with
  `.venv/bin/python -c "import rngrn,os;print(os.path.dirname(rngrn.__file__))"` — the path must
  be **inside the worktree** or every test result is meaningless.
- **Run `pytest` with the sandbox DISABLED** — a sandboxed run reports ~15 fake
  `PermissionError`s (payload.h5 read-deny). Full-suite runs take ~6–8 min at this size.
  `ps`/`pgrep`/`lsof` are also blind in the sandbox (§7b) — disable it for any liveness check.
- **Any trainer/sweep/experiment launch goes through `bash scripts/guarded_run.sh`**
  (`.claude/hooks/guard_trainer.py` refuses anything else). **Host RAM is the binding resource**
  (§7a); one trainer pool at a time across *all* worktrees.
- **NEW — a paper sprint owns the GPU until ~morning** (owner deadline declared 2026-08-18
  ~22:00). Tasks 14–20 launch guarded GPU work. Do not launch any of them while the paper
  sprint's runs are live: check `lsof` on `.trainer.lock` **unsandboxed** first, and treat an
  empty sandboxed `ps` as no information at all. If the GPU is busy, the correct action is to
  run the CPU-only tasks and leave the GPU tasks queued — **not** to launch alongside.
- **Every new module under `src/rngrn/losses/` or `src/rngrn/eval/` must be classified** in
  `tests/test_firewall.py` (`RECOVERY_SIDE`/`SCORING_SIDE`) — the completeness test fails
  otherwise. **Any new `scripts/` module that opens `payload.h5` or reads an `AnswerKey` must be
  appended to `FORBIDDEN`** (the completeness test is blind to `scripts/`). Read the list in the
  test file, never the one in `CLAUDE.md` §5 — that one is **stale** (it says 12 names; the file
  has 27), and Task 21 fixes it.
- **A0 is untouchable:** the baseline objective (`kstar_anchor`, `turing_hinges_split`,
  `frame_scale_anchor` w=2.0, `anticollapse` w=0.5, softplus/exp parameterization) must keep
  bit-identical behaviour after every task. New behaviour is opt-in via config. **NEW — R3
  inherits a live conflict with this constraint; see Task 8's decision point.**
- **NEW — integration is per-hunk, never `git merge`.** `feature/gpu-optim` sits on `10cff1b`
  and does **not** contain R2's T8–T15 registry refactor. The review's §10 enumerates 26
  collision points across `losses/total.py`, `losses/terms.py`, `losses/spectral.py`,
  `recover.py`, `model.py`, `config.py`. §11's warning binds with full force: git resolves at
  line granularity, and "keep both sides" is **wrong** inside a function signature or a dict
  literal, where it silently yields duplicate definitions of which only the last takes effect —
  the exact failure that reverted `resid` and dropped `model_seed` in phase A. **Every one of
  the 26 points (plus the 27th, Task 9) must have a named owning task and a recorded
  disposition.** `merge-damage-hunter` runs after integration (Task 10) and again at the end
  (Task 22).
- **NEW — adopting an undocumented semantic change requires documenting it.** The review's §6
  finds 8 semantic changes with 1 `DECISIONS.md` entry. The 7 missing entries (LSMR stop
  cadence, ETDRK4 per-call blow-up check, relax-detector swap, `bdf1` raise, liveness cadence,
  `lbfgs_error` recording, refusal deletion) are written **by the task that adopts each change**
  — they are step items, not a docs backlog. Use `.claude/skills/record-decision`.
- **NEW — `git stash` is repo-global.** Two concurrent units stashing will apply each other's
  changes. Do not use it.
- **UNCALIBRATED discipline:** every new threshold carries the code-site comment naming its
  calibrator (spec §8 item 14) — no silent numbers.
- **Numbers leaving a run directory pass `.claude/rules/reporting-numbers.md`.** A number with
  no run directory under `experiments/` is not reportable.
- **Validation grid is 512² on the data box** — no cheaper grid is licensed for any morphology
  or gate claim (D-FFT-9 closure 3). The 96² adaptive box of §4.3 is a *training* grid and is
  never a validation grid.
- torch default dtype is float64 (`torch.set_default_dtype` is global in this package).
- **Decision points return to the controller.** Paste `.claude/rules/orchestration.md`'s block
  verbatim into every dispatch. Subagents cannot ask the owner; a precisely diagnosed defect a
  unit could not fix is a better outcome than a green test that hides it.

### What T16 measured, and what it does *not* parameterize

R2's Task 16 (Phase-I ignition run) is **in flight as this plan is written**; its report
(`.superpowers/sdd/PLAN_redesign/task-16-report.md`) does not exist yet. Two consequences, both
load-bearing for R3's batch sizing:

1. **Read `task-16-report.md` at execution time** and prefer its numbers over anything below.
   If it still does not exist, read the tracked run JSONs directly:
   `experiments/redesign_r2/phase1/phase1_{a0,r2}_B{64,128,256,512}/results/run.json` on
   `feature/redesign-model` (keys `B`, `s_per_step`, `member_steps_per_s`, `peak_rss_mb`,
   `cuda_max_alloc_mb`, `mem_available_mb_after`, `n_turing`, `n_distinct_structures_all`).
2. **PROVISIONAL, NOT VETTED — do not cite these numbers anywhere.** Read off the on-disk
   run JSONs while T16 was mid-sweep, before any review; the ledger's standing rule is that an
   unreviewed campaign's numbers are not citable. They are here only to size the plan:

   | arm | B | s/step | member-steps/s | peak RSS MB | CUDA alloc MB | Turing | distinct structures |
   |---|---|---|---|---|---|---|---|
   | a0 | 64 | 1.0657 | 60.1 | 1988.9 | 91.7 | 3/64 | 44 |
   | r2 | 64 | 0.0295 | 2168.1 | 1915.2 | 91.4 | 0/64 | 3 |
   | r2 | 128 | 0.0343 | 3736.6 | 1989.4 | 112.8 | 0/128 | 3 |

   Three things this shape says, which Task 17 must confirm or refute against the final report:
   - **Host RAM is not the binding resource for Phase I.** B 64→128 moved peak RSS 1915→1989 MB
     and CUDA allocation 91→113 MB. B = 512 is plausibly reachable on the ignition-only path.
     **B = 512 stays UNCALIBRATED until measured** (spec §8 item 14).
   - **The ignition curve does NOT parameterize Phase II, and R3 must not reuse it.** The ~36×
     step-time gap between the arms is the pinned model skipping the steady-state Newton solve.
     R3 puts a *batched forward solve* (relax + Newton + LSMR adjoint, per member, at n=96) into
     the step — a per-step cost of a completely different order. **Task 17 measures R3's own
     B curve with the forward solve in the loop**; the T16 curve bounds nothing about it.
   - **The r2 arm did not ignite (0/64, 0/128) against a0's 3/64.** That is D-R2-1, Task 18.

---

# Phase A — repair the base implementation

**Worktree:** `../worktrees/gpu-optim-repair` · **Branch:** `feature/gpu-optim-repair` ·
**BASE:** `feature/gpu-optim` @ `bba3e28`.

Phase A never touches the redesign line. Its output is a `feature/gpu-optim-repair` whose suite
is green and whose review verdict has moved from *unsafe to integrate* to *approved* — so that
Phase B integrates **reviewed** code, not the interrupted session's word for it. The suite at
`bba3e28` is **2 failed / 605 passed / 24 skipped** (controller-verified, unsandboxed); those
two failures are Task 1's starting point, not a regression to diagnose.

Tasks 1–3 are the three owner-mandated entry tasks, in order.

### Task 1: Repair C1 — restore the batched-spectral contract, including the vacuously-green test and the deleted input validation

**Files:**
- Modify: `src/rngrn/losses/total.py` (`compute_terms_batched` — restore input validation at the
  entry point), `tests/test_ignition_gating.py` (three tests: `:158`, `:165`, `:194`)
- Modify: `docs/DECISIONS.md` (the refusal-deletion entry — semantic change #11)

**Interfaces:**
- Produces: `compute_terms_batched` raises loudly when its `spectral` argument carries a solver
  lacking `solve_subset` (i.e. a serial `PatternSolver` handed to the batched assembler), and
  when `model` is not a `BatchedRNGRN`. Consumes: nothing new.

**Why this first:** the review's C1 says the dangerous item is not the two red tests — it is
`test_recover_raises_on_batched_with_a_spectral_weight` (`:194`), which **passes vacuously**. It
asserts `pytest.raises(ValueError, match="batched")` without passing `lbfgs_steps`, so it now
trips `recover()`'s *LBFGS* guard, whose message happens to contain the word "batched". A red
test gets fixed; a false green does not. Repairing only the two failures leaves the batched/
spectral combination believed-covered and uncovered.

- [ ] **Step 1: Reproduce the RED state.** `pytest tests/test_ignition_gating.py -v` (sandbox
  disabled). Expect `test_compute_terms_batched_refuses_spectral` and
  `test_total_loss_batched_refuses_spectral` failing with
  `AttributeError: 'NoneType' object has no attribute 'B'`. Record the exact output — it is the
  RED evidence for this task.
- [ ] **Step 2: Prove the third test is vacuous before touching it.** Temporarily pass
  `lbfgs_steps=0` in a scratch copy of `:194` and confirm it then **fails** (no `ValueError` is
  raised at all). Record that transcript. Do not commit the scratch copy.
- [ ] **Step 3: Write the replacement tests (they must FAIL first).** Three of them:
  - `test_compute_terms_batched_accepts_a_batched_spectral_context` — the five spectral keys
    appear per member with the exact-0-loss / NaN-record split of the omitted-never-zeroed
    semantics (§4.4). Use the existing `_StubSpectralSolver` pattern in that file, extended with
    a `solve_subset`.
  - `test_compute_terms_batched_refuses_a_serial_solver` — hand it a solver object exposing
    `.solve()` but not `.solve_subset()`; assert a `ValueError`/`TypeError` naming
    `solve_subset`, raised **at the entry point**, before any steady-state solve or Jacobian.
  - `test_recover_raises_on_batched_with_a_spectral_weight` — rewritten to pass
    `lbfgs_steps=0` explicitly, so the LBFGS guard cannot be what fires, and to assert against
    whatever the *current* contract is. **Decision point (return, do not rule):** if the current
    contract is that the combination is now *legal*, this test's replacement is an end-to-end
    `batched=True, lbfgs_steps=0` spectral run against its serial twin — which is Task 2/4
    territory. Report which shape you found and stop that thread.
- [ ] **Step 4: Implement.** The review's §8 names the two-line fix:
  `hasattr(spectral.solver, "solve_subset")` at the top of `compute_terms_batched`, plus the
  `model` type check that the deleted refusal used to provide implicitly. Fail loud (§4) — raise
  with the offending type in the message, never fall back to the serial path.
- [ ] **Step 5: Write the `DECISIONS.md` entry** for semantic change #11 (the refusal deletion):
  date, what was deleted, why the structural reasons in spec §4.1 no longer hold (a batched
  `PatternSolver` now exists), what replaced the validation, and what was rejected (leaving the
  refusal in place, which would make the batched spectral path unreachable). Per
  `.claude/skills/record-decision`. Suggested key: `D-PERF-4`; check `DECISIONS.md` for the next
  free number rather than trusting this one.
- [ ] **Step 6: Full suite unsandboxed; commit.**
  `git commit -m "fix: restore batched-spectral input validation; de-vacuate the refusal test (C1)"`

**Model hint:** sonnet. Clear spec, three named tests, a two-line fix.

### Task 2: `BatchedPatternSolve.backward` gradient equivalence — the load-bearing gap (C2)

**Files:**
- Create: `tests/test_batched_forward_solve.py`
- Modify: `src/rngrn/forward.py` **only if a defect is found** (a fix is a decision point, see
  below)

**Interfaces:**
- Consumes: `forward.BatchedPatternSolve` (`forward.py:1141`, `backward` at `:1148`),
  `forward.BatchedPatternSolver` (`:1196`), `forward.make_spatial_F_batched` (`:930`), and their
  serial twins `PatternSolve` / `PatternSolver` / `make_spatial_F`.
- Produces: three test functions (below). No production interface.

**Why this is the load-bearing item:** the serial twin is pinned by
`tests/test_forward_solve.py` against finite differences (`best <= 5e-3`) and by an
adjoint-vs-scipy check (`rel < 1e-6`). The batched form loops `solve_adjoint` per member and
then does **one** stacked `torch.autograd.grad` through `make_spatial_F_batched` with
`grad_outputs=-lam_stack`. Whether that final scatter lands on exactly row `members[j]` of each
`(B, …)` parameter tensor and is **zero elsewhere** is asserted in a docstring and checked by
nothing. A transposed or broadcast-collapsed scatter would not crash — it would train every
member on a blend of every other member's gradient and look like slow convergence.

- [ ] **Step 1: Write the three failing tests.** All at small scale (n = 32 or 48, B = 3, N = 3,
  `competitive`, CPU float64) so they run in CI without a GPU. Read the actual signatures first
  and match them; the contracts below are the point, not the sketch.

```python
# tests/test_batched_forward_solve.py
# 1. vs finite differences — the D1 tolerance class, tol 1e-4 per spec 4.2's A/B discipline
def test_batched_backward_matches_finite_differences():
    # 10 random directions in theta, central differences on the scalar loss;
    # worst relative error over directions must clear the serial path's own bar.

# 2. vs the serial per-member path — the equivalence mandate of spec 4.1
def test_batched_backward_matches_serial_pattern_solve_member_by_member():
    # Build B=3 members; solve each with the serial PatternSolver and with
    # BatchedPatternSolver.solve_subset; assert grads agree to <= 1e-10 relative.

# 3. gradient-scatter isolation — the failure that would not crash
def test_batched_backward_scatters_to_exactly_one_member_row():
    # Backprop a loss that depends on member j ONLY. Assert every other member's
    # row in every (B, ...) parameter .grad is EXACTLY zero (not merely small).
```

- [ ] **Step 2: Run — the tests must FAIL for the right reason first** (the file does not exist;
  then, once written, they either pass or expose a defect). Record which.
- [ ] **Step 3: If all three pass** — the code was right and is now pinned. Commit and stop.
- [ ] **Step 4: If any fails — STOP. Decision point (return, do not rule).** A defect in the
  adjoint scatter is a correctness finding about ~1,000 lines of unreviewed gradient machinery.
  Report: which test, the measured discrepancy, the minimal reproduction, your diagnosis, and
  your recommended fix. Do **not** patch and continue — an R3 that silently repairs its own
  gradient path is exactly what the review warns against.
- [ ] **Step 5: Also pin the two runner-up items in the same file** (both from review §5's
  table, both cheap now the fixture exists): `relax_to_pattern_torch_batched`'s
  snapshot-on-finish semantics (a member's returned field equals what the serial
  `relax_to_pattern_torch` would have returned, and its per-member `blew_up`/`no_saturation`
  verdicts match), and `BatchedPatternSolver.solve_subset`'s reason vocabulary parity
  (`"ok"`/`"solve_failed"`/`"not_patterned"`) plus warm-state keying and clearing.
- [ ] **Step 6: Full suite unsandboxed; commit.**
  `git commit -m "test: BatchedPatternSolve.backward vs FD and vs serial; gradient-scatter isolation (C2)"`

**Model hint:** opus. Adjoint gradients, an unreviewed implementation, and a live possibility of
finding a real defect.

### Task 3: Pin `observables.raps_torch` (C3), then close its two self-declared TODOs

**Files:**
- Create: `tests/test_raps_torch_parity.py`
- Modify: `src/rngrn/observables.py` (docstring TODOs at `:135-136`, `:138-140`) — and the
  `floor`→`digitize` binning **only** under Step 4's ruling
- Modify: `docs/DECISIONS.md` (relax-detector swap — semantic change #6)

**Interfaces:**
- Consumes: `observables.raps` (numpy, the calibrated primary estimator),
  `observables.raps_torch`, `observables.kstar_of_torch`.
- Produces: a parity test pinning `raps_torch` against `raps`, on **both** regimes.

**Why:** `raps_torch` was promoted to the relax saturation detector on two new call paths
(`forward.py::relax_to_pattern_torch` and `forward.py:1023 _kstar_of_torch_batched`) in the same
delta whose docstring says *"not PINNED by a test — add one before any further caller relies on
it"*. The measured divergence is **one bin differing by 14 % and k\* by 0.24 %** on a 16² noise
field. The detector's flatness ratio runs at `flat_tol=1e-4` — four orders below 0.24 % — and it
runs **from the first chunk**, when the field is x\* plus 1e-2 noise, i.e. exactly the
ill-conditioned regime where the divergence was measured. Register item 15 makes the unwindowed
RAPS the *primary* estimator and forbids silent swaps, because the sub-bin centroid k\* and the
one-bin bar are calibrated on it.

- [ ] **Step 1: Write the failing parity test.** Two regimes, explicitly separated:
  patterned field (the docstring claims 3e-16 — assert it), and **noise field at the detector's
  own operating point** (x\* + 1e-2 noise, the grid sizes the detector actually sees). Assert
  per-bin agreement and `kstar_of` vs `kstar_of_torch` agreement, with the tolerance stated as
  a measured number, not a guess.
- [ ] **Step 2: Run — record what it measures.** The 14 %/0.24 % figures come from a docstring,
  not a run; this test is the first measurement with provenance.
- [ ] **Step 3: Assess the detector's exposure.** Does a 0.24 % k\* difference at `flat_tol=1e-4`
  actually move the stop chunk? Demonstrate it: run `relax_to_pattern_torch` with each estimator
  on the same field/seed and compare the returned chunk index and u\*. This is the measurement
  that decides Step 4.
- [ ] **Step 4: Decision point (return, do not rule).** Two options, both legitimate, and the
  spec makes this an estimator question (register item 15):
  **(a)** switch `raps_torch` to `digitize` binning so it matches `raps` exactly — closing the
  divergence but changing values for a shared observable; **(b)** keep `floor` and record an
  explicit `DECISIONS.md` entry accepting the divergence, bounded by Step 3's measurement.
  Report both with the numbers. **Do not choose.**
- [ ] **Step 5: Implement the ruling; write the `DECISIONS.md` entry** for semantic change #6
  (the relax-detector swap `obs.kstar_of` → `obs.kstar_of_torch`) either way — the swap itself
  is undocumented regardless of which binning wins. Remove the now-closed TODOs from the
  docstring; leave nothing claiming more than the code does (§4).
- [ ] **Step 6: Full suite unsandboxed; commit.**
  `git commit -m "test+fix: pin raps_torch against raps; record the relax-detector swap (C3)"`

**Model hint:** opus. It carries an estimator decision point and touches a register item.

### Task 4: Equivalence tests for the rest of the batched surface

**Files:**
- Modify: `tests/test_batched_forward_solve.py` (Task 2's file)
- Create: `tests/test_batched_spectral_terms.py`
- Modify: `tests/test_determinism.py` (the `'auto'` resolution gap)

**Interfaces:** consumes only; produces tests. Every symbol below is from review §5's table and
has **zero** references under `tests/` today.

- [ ] **Step 1: `losses/spectral.py` batched twins** — `raps_torch_batched`,
  `spectral_block_torch_batched`, `spec_shape_batched`, `spec_aniso_batched`,
  `spec_amp_mean_batched`, `spec_amp_fluct_batched`, `real_moments_batched`,
  `spectral_terms_batched`, `is_ignited_batched`. Member-by-member against the serial twins on a
  `(4, 3, 64, 64)` stack, CPU float64. The reviewer measured **≤ 4.3e-14 relative, worst term
  (`spec_amp_fluct`), the other four exactly 0.0** — that probe is not a test and does not
  survive into CI. Set the tolerance from your own measurement, at that order.
- [ ] **Step 2: `losses/total.py:188 _apply_spectral_batched`** — the per-member
  ignite-or-omit contract: exact-0 loss placeholder vs NaN record, `spectral_skipped` reason
  strings, `spec_computed`/`spec_ignited`, and the `abandoned` vs `not_ignited` distinction.
  This is the semantics spec §4.4 calls "omitted-never-zeroed" and §4.1 notes exists today only
  on the serial path.
- [ ] **Step 3: the remaining `forward.py` symbols** — `batched_reaction_fields` (`:906`),
  `make_spatial_F_batched` (`:930`), `_member_F` (`:950`), `_batched_reaction_builder` (`:967`,
  including its loud `form != "competitive"` refusal), `_half_coeffs_batched` (`:998`, per-member
  `dt`/`D`), `_kstar_of_torch_batched` (`:1023`, the mask-vs-slice centroid argued equal "to
  floating-point associativity").
- [ ] **Step 4: `dispersion_backend='auto'` resolution.** `tests/test_determinism.py:79` covers
  only the explicit-`cubic`-at-wrong-N rejection. Add: `N=2 → eig`, `N=3 → cubic`, `N=4 → eig`,
  `'auto'` never survives construction (`.dispersion_backend` always reads concrete), and
  `BatchedRNGRN.from_seeds` resolves identically.
- [ ] **Step 5: Full suite unsandboxed; commit.**
  `git commit -m "test: equivalence tests for the batched spectral twins and forward-solve surface"`

**Model hint:** sonnet. Mechanical, against a clear per-symbol table.

### Task 5: The four remaining `DECISIONS.md` entries and the docstring-truth fixes

**Files:**
- Modify: `docs/DECISIONS.md` (four entries), `src/rngrn/forward.py` (I5, I6),
  `src/rngrn/etdrk4_torch.py` (I2's parity note), `src/rngrn/recover.py` (M3, M4, M9),
  `src/rngrn/losses/spectral.py` (M6), `src/rngrn/eval/numerics.py` (I3)

**Interfaces:** documentation and comment truth only. **No behaviour change in this task** — if
a fix would change behaviour, it is a decision point.

- [ ] **Step 1: Write the four entries** (Task 1 wrote #11, Task 3 wrote #6). Each with date,
  evidence, and what was rejected:
  - **LSMR stop cadence** (review I1, table #4) — `_LSMR_STOP_CHECK_EVERY = 25`, the branchless
    `_sym_ortho_t`, and the changed exact-Krylov-breakdown branch. This changes the stopping
    semantics of a solve whose verified behaviour is recorded in **D-FFT-10**. State the
    argument that the refinement loop's *true residual* still decides convergence, and state
    that `tests/test_forward_solve.py` was written against the *faithful* port and was not
    re-examined.
  - **ETDRK4 blow-up check per-step → per-call** (review I2, table #5). The boolean equivalence
    argument is sound (nan is absorbing; the reviewer checked the reaction closure for a path
    back to finite and found none). The entry must record the part the docstring omits:
    `eval/numerics.integrate_etdrk4_rfft` (numpy) returns **the field at the first non-finite
    step** while the torch port now returns **the field after all `nsteps`** — so on a blow-up
    the two backends return different arrays, weakening `relax_to_pattern_torch`'s standing
    "same trajectory up to FFT round-off" claim.
  - **Liveness sync cadence** (review M3, table #8) — the early break can fire up to 24 steps
    late; during those steps `total` is an exact zero *with a graph*, so gradients are zero but
    **Adam's momentum still moves the parameters**. The inline comment's "harmless" is stronger
    than the truth. Also note `verbose` at such a step computes `float(loss_vec[alive].mean())`
    over an empty selection → NaN.
  - **`lbfgs_error` recording** (review M4, table #10) — `except: pass` → a recorded
    `restart_log` key. Note the asymmetry: serial rows only, never `_batched_restarts` rows, so
    the two paths emit different column sets into the run index (§4 wants flat scalars, which a
    `str | None` satisfies; the asymmetry is the issue).
- [ ] **Step 2: `bdf1_newton_krylov` — check before writing an entry.** R1's Task 1
  (`docs/PLAN_redesign.md`) made this stub raise, on `feature/lift-ladder`, independently of
  gpu-optim's `numerics.py:205` doing the same thing. **This is a 27th collision point, absent
  from the review's §10 list.** Determine whether R1's version already carries a `DECISIONS.md`
  entry; if it does, cite it rather than writing a second. Also do what neither branch did:
  `grep -rn "bdf1" configs/ notebooks/ experiments/ scripts/` for anything selecting
  `integrator: bdf1_newton_krylov` — anything that does now hard-crashes, and every existing run
  recorded under that label actually used ETDRK4. **That relabelling is a number whose meaning
  changed and §10.4 requires it announced loudly** — put it in the entry and in the phase report.
- [ ] **Step 3: Docstring truth fixes** (§4: no docstring claims more than its code does):
  - **I5** — `PatternSolver`'s docstring says the torch LSMR cadence is "applied on BOTH
    devices". False: `newton_polish` and `solve_adjoint` select `_minnorm_solve_t`/`precon_t`
    only under `on_device = u.device.type != "cpu"`; the CPU path routes through scipy and never
    enters `_lsmr_torch`. The numpy-`eigvals`-for-`jac_rate` half *is* both-devices and is
    correctly described. Fix the sentence — a reader establishing whether CPU results are still
    the D1 reference must get "yes".
  - **I6** — `forward.py:31-34` quotes a cost split ("32²/B=3: relax 30 %, Newton 40 %, adjoint
    30 %; 64²/B=3: 10/75/15") citing "the unit's report", which does not exist, and draws a
    design conclusion from it. Mark **UNMEASURED** with a pointer to Task 17, or delete. Do not
    leave a number with no run (`reporting-numbers.md` step 1). The `_half_coeffs_batched`
    memory figures (58 MB at b=8/n=96, 230 MB at b=32/n=96) are arithmetic from shapes and stay.
  - **M6** — `spectral_block_torch`'s summary line says "normalise by total power" while its
    body, its DEVIATION paragraph, and `spectral_block_torch_batched`'s docstring all normalise
    by the *central block's own* power. A wording defect, but the batched twin now states the
    opposite of the serial summary in the same file.
  - **M5** — `recover()`'s docstring says `'cubic' (exact for N<=3 ONLY)` while `model.py:149`
    raises for `N != 3`.
  - **M1/M2** — record the CPU cost the docstrings are silent about: the vectorised
    `_damped_newton` line search is bit-identical (max |Δx\*| = 0.0 over 40 seeded models, 0
    flag disagreements) but costs **~12 % on the CPU reference path** (0.1322 s vs 0.1181 s per
    40 solves, `OMP_NUM_THREADS=1`); `steady_state_batched`'s 5-halving cadence carries the same
    trade.
- [ ] **Step 4: Run `evidence-auditor` over the changed docs and docstrings; fix findings.**
- [ ] **Step 5: Full suite unsandboxed; commit.**
  `git commit -m "docs: DECISIONS entries for the four remaining gpu-optim semantic changes; docstring truth fixes"`

**Model hint:** sonnet for the writing, but the `bdf1` collision discovery in Step 2 returns to
the controller if a caller is found.

### Task 6: Phase A close — review, green suite, handoff

- [ ] **Step 1: Full suite unsandboxed** on `feature/gpu-optim-repair`. It must be green with
  **zero** failures; record the count and duration in the phase report. The starting point was
  2 failed / 605 passed / 24 skipped.
- [ ] **Step 2: Run `firewall-auditor`.** Firewall was CLEAN at `bba3e28`; Phase A added tests
  only, but `test_raps_torch_parity.py` and `test_batched_forward_solve.py` are new files and
  any new `scripts/` module (there should be none) would owe a `FORBIDDEN` entry.
- [ ] **Step 3: `superpowers:requesting-code-review` over `bba3e28..HEAD`.** The bar: the
  review's own "minimum to reach Needs fixes rather than Unsafe" is C1 + C2 + C3; Phase A aimed
  past that at *approved*. Address Critical and Important findings; report Minor.
- [ ] **Step 4: Push the branch** (hooked push; on green-suite + SIGPIPE, verify `git ls-remote`,
  deliberate `--no-verify` retry, record the occurrence — this has happened 7 times).
- [ ] **Step 5: Update `docs/HANDOFF_gpu_optim.md`** — the branch is no longer
  "unsafe to integrate as-is"; say what changed, what is now pinned by a test, and what is still
  only a docstring. Commit and push.

---

# Phase B — integrate onto the redesign line

**Worktree:** `../worktrees/r3-integration` · **Branch:** `feature/r3-integration` ·
**BASE:** `feature/redesign-model` at **T16's reviewed head** (read `git log` — T16 was in flight
when this plan was written; do not guess a SHA).

**This phase does not run `git merge feature/gpu-optim-repair`.** It transplants hunks. The two
lines share `10cff1b` as an ancestor and diverge over exactly the three files R2 refactored onto
a loss-term registry. Work with `git diff` and `git cherry-pick -n` per commit, resolving by
**reading both sides and writing the union deliberately**, never by accepting a merge result.

### Task 7: Build the collision ledger before touching a line

**Files:**
- Create: `docs/INTEGRATION_r3_collisions.md` (tracked)

**Interfaces:**
- Produces: a table with one row per collision point — id, file, symbol, gpu-optim's change,
  the redesign line's change, the chosen resolution, the task that owns it, and a
  verified-after-integration checkbox.

- [ ] **Step 1: Transcribe review §10's 26 points** into the table, verbatim ids 1–26, plus:
  **27 —** `eval/numerics.py::integrate_bdf1_newton_krylov`, changed by *both* lines
  independently (R1 Task 1 and gpu-optim `375cb4a`-era); **28 —** `LossTerm.batched_fn` /
  `refusal_reason` for the five spectral terms, which R2's registry recorded as *refused* and
  gpu-optim now *implements* (the registry's enumeration-contract test asserts exactly one of
  the two is set, so this flips by construction, not by choice).
- [ ] **Step 2: For each row, read BOTH sides in source** and write the intended union in prose
  before any edit. Flag the four shapes §11 names as silently destructive:
  (a) **four signature-level merges in one file** — `compute_terms`/`total_loss` gain
  `obs_scale`, `kstar_idx`; `compute_terms_batched`/`total_loss_batched` gain `obs_scale`,
  `kstar_idx`, `active` (points 3, 4) — this is the precise shape of the phase-A duplicate-
  keyword failure; (b) **dict-literal edits** — `build_frame_targets`'s new `band_idx` key
  (point 15), `recover.py`'s `param_prior_kw` reshaping `box_path`→`box` (point 19) — a
  duplicate key silently keeps the last; (c) **deletions**, which merge badly against any edit
  to the same block and can be silently resurrected — the batched-spectral refusal (point 8) and
  the `use_spectral and batched` refusal (point 21), both of which Task 1 *replaced* rather than
  restored; (d) **field defaults** — `ModelConfig.dispersion_backend` (point 26), the class that
  silently reverted `resid` in phase A.
- [ ] **Step 3: Commit the ledger** before integrating.
  `git commit -m "docs: R3 integration collision ledger — 28 points with named resolutions"`

**Model hint:** opus. This is the judgement that determines whether Phase B damages the line.

### Task 8: Integrate the sound half — hoist/desync, `auto` backend, liveness, `lbfgs_error`

**Files:** `src/rngrn/losses/terms.py`, `src/rngrn/losses/total.py`, `src/rngrn/recover.py`,
`src/rngrn/model.py`, `src/rngrn/config.py`, `src/rngrn/eval/analysis.py`,
`src/rngrn/etdrk4_torch.py`, `src/rngrn/io.py`, `src/rngrn/eval/lifted.py`,
`src/rngrn/eval/rollout.py`, `src/rngrn/eval/numerics.py`

**Scope:** gpu-optim's `346bbf2` + `e7d01bd` + `375cb4a` — the half the review calls
"substantially sound … approve with minor fixes". Collision-ledger points **9–13, 19, 20, 24,
25, 26, 27**, plus the neutral items 1, 12, 13, 14, 15 of the semantic table.

- [ ] **Step 1: Transplant the arithmetic-neutral hunks** — one Jacobian per step threaded as
  `J=` into the dispersion-side terms (table #1), the vectorised `_damped_newton` (#2),
  `steady_state_batched`'s restructured inner loop (#3), constant hoisting (#12), the stacked
  `eigvals` loops (#13), `_model_JD` device placement (#14), `io.load_checkpoint(map_location)`
  (#15). **The registry is the resolution vehicle for points 9 and 10:** rather than threading
  `J=`/`idx=` through six term signatures by hand against a registry that standardised them,
  express the hoist through `LossTerm`'s call contract. If that is not possible without changing
  the registry's contract, that is a decision point — report it.
- [ ] **Step 2: Re-run R2's A0 pin tests after every hunk**, not once at the end. The registry's
  `test_default_weights_are_bit_identical_to_the_legacy_dict` is **tautological since T14** (a
  deferred minor in the ledger — both sides derive from the registry), so it will **not** catch a
  default flip. Do not rely on it. Add an explicit non-tautological pin if one is cheap.
- [ ] **Step 3: `dispersion_backend='auto'` — DECISION POINT (return, do not rule). Do not
  adopt it without a ruling.** The conflict, stated fully:
  - `docs/PLAN_redesign.md` Global Constraints say **"A0 is untouchable: the baseline objective
    … must keep bit-identical behaviour after every task."**
  - D-PERF-3 changes `ModelConfig.dispersion_backend`'s default from `eig` to `auto`, which
    resolves to `cubic` at N=3 — i.e. **every A0 run's dispersion backend changes**, and
    D-PERF-3 itself states the runs are "not bit-comparable" (`_sigma_max_cubic` validated at
    sigma_max MAE 9.2e-13 over 127 answer-key Jacobians). Spec §1 makes A0's comparability with
    the D5 row the whole point of the arm.
  - Complicating fact, measured: **T16's a0 arm already ran at `backend: cubic`** (an explicit
    driver choice — `feature/redesign-model`'s `model.py` still defaults to `eig`). So the
    comparability question is live on the redesign line *independently* of gpu-optim.
  - There is also a provenance regression D-PERF-3 does not mention: `frozen_config.yaml` records
    `"auto"` rather than the backend that ran, while `reporting-numbers.md` step 4 says to read
    the frozen config, not re-derive from it. **Writing the resolved value into the frozen
    config closes this and should be done whichever way the ruling goes.**
  Report all four points with the numbers and stop that thread.
- [ ] **Step 4: Adopt the liveness cadence and `lbfgs_error`** (points 24, 25) — and square up
  the asymmetries the review flagged as an unfinished pass: `lbfgs_error` on batched rows too
  (M4), and `active=alive if spec_cfg is not None else None` (M9) — the conditional cannot change
  a value and makes the batched non-spectral path take a different branch for no stated reason.
  If squaring either up would change a recorded column set, say so in the report.
- [ ] **Step 5: Full suite unsandboxed; commit.**
  `git commit -m "integrate(gpu-optim): hoist/desync half onto the loss registry; liveness + lbfgs_error"`

**Model hint:** opus. Registry-vs-hoist reconciliation plus a live A0 conflict.

### Task 9: Integrate the batched half — forward-solve desync and the batched `PatternSolver`

**Files:** `src/rngrn/forward.py`, `src/rngrn/losses/spectral.py`, `src/rngrn/losses/total.py`,
`src/rngrn/losses/term_registry.py`, `src/rngrn/recover.py`, plus every test Phase A wrote

**Scope:** gpu-optim's `8f93dfe` + `2b11010` + all of Phase A. Collision-ledger points **1–8,
14–18, 21–23, 28**.

- [ ] **Step 1: Transplant `forward.py` and `losses/spectral.py` wholesale** — R2 did not touch
  either file, so these are additive and low-risk. Bring Phase A's tests with them; they must be
  green in the integrated tree before any `total.py`/`recover.py` work starts.
- [ ] **Step 2: `losses/total.py` — the highest-density surface** (points 1–8). Rewrite
  `compute_terms`/`compute_terms_batched`'s term-assembly through the registry rather than
  merging two hand-written blocks; insert `_apply_spectral_batched` (point 5); wire
  `total_loss_batched`'s `L_<key>` NaN-masking loop to `SPECTRAL_TERM_KEYS` and
  `parts["spec_computed"]` (point 6); add `parts_member`'s object/string dtype branch (point 7);
  keep Task 1's replacement validation, **not** the old refusal (point 8).
- [ ] **Step 3: Point 28 — flip the registry entries.** The five spectral terms move from
  `refusal_reason=<the total.py refusal text>` to a real `batched_fn`. The enumeration-contract
  test (`test_every_registered_term_is_fully_classified`) asserts exactly one of the two is set,
  so it will fail until this is done — that failure is the completeness test working. Update
  each term's `calibration` tag honestly (the spectral five are `UNCALIBRATED`).
- [ ] **Step 4: `recover.py`** (points 19, 21, 22, 23) — `_batched_restarts` gains `spec_cfg`,
  `spec_targets`; `build_frame_targets(device=dev)`; the deleted `use_spectral and batched`
  refusal stays deleted with Task 1's validation in its place.
- [ ] **Step 5: Full suite unsandboxed.** Expect the registry contract test and any A0 pin to be
  the informative failures; a failure in `tests/test_losses.py` or `tests/test_ignition_gating.py`
  means the transplant changed serial behaviour and is a **stop**, not a fix-forward.
- [ ] **Step 6: Commit.**
  `git commit -m "integrate(gpu-optim): batched forward solve + batched spectral terms onto the registry"`

**Model hint:** opus.

### Task 10: Integration damage sweep — the class the suite cannot see

- [ ] **Step 1: Run the `merge-damage-hunter` agent** over `BASE..HEAD`. Its named quarry:
  duplicate keys in a dict literal, duplicate keyword arguments, a name defined twice at
  module/class scope where the second silently wins, `**`-expansion building a run-index row.
- [ ] **Step 2: Verify every value a parent branch set is still set.** Check `resid`, the anchor
  weight, and `model_seed` **specifically** — all three were silently lost in the phase-A merge.
  Then check R2's own additions: `pin_xstar`, `param_boxes`, `kstar_anchor_si`'s registry entry
  and weight, `LossConfig.weights`' key set.
- [ ] **Step 3: Close the collision ledger.** Every one of the 28 rows gets its
  verified-after-integration box ticked, with the evidence (a test name, a grep, a diff hunk).
  A row you cannot verify is a finding, not a formality.
- [ ] **Step 4: Run `firewall-auditor`.** Review §7 found the delta firewall-CLEAN at
  `bba3e28`, but integration adds imports on both sides. `forward.py`, `etdrk4_torch.py` and
  `losses/spectral.py` are already on `RECOVERY_SIDE`; confirm nothing new under `losses/` or
  `eval/` is unclassified, and that no new `scripts/` module reads `payload.h5`.
- [ ] **Step 5: `superpowers:requesting-code-review` over the whole integration.** Full suite
  unsandboxed; record the count. Push (hooked; SIGPIPE protocol).
- [ ] **Step 6: Commit the closed ledger.**
  `git commit -m "docs: integration collision ledger closed — 28/28 verified"`

**Model hint:** opus for the review; the agents are fixed-model.

---

# Phase C — the gradient machinery (§4.2–§4.3)

Same worktree, `../worktrees/r3-integration`, sequential after Phase B. This is the phase spec
§7 actually names as R3.

### Task 11: The adaptive commensurate solve box, with re-tile hysteresis

**Files:**
- Create: `src/rngrn/solve_box.py` (top-level, recovery-side)
- Modify: `tests/test_firewall.py` (classify the new module — decide `RECOVERY_SIDE`; note the
  completeness test globs `losses/` and `eval/` only, so a top-level module may need the list
  extended; **touching `test_firewall.py` is a hard-stop return** per orchestration — get the
  ruling, then edit)
- Modify: `src/rngrn/forward.py` (accept a re-tileable `L`/`n` and clear warm starts on re-tile)
- Test: `tests/test_solve_box.py` (create)

**Interfaces (spec §4.3, which is binding — read it, do not work from this summary):**
- Produces: `solve_box.geometry(kstar_hat: float, p: int = 8, n: int = 96) -> dict` with
  `{"L_solve": p * 2*pi / kstar_hat, "n": n, "p": p}`; `solve_box.needs_retile(kstar_hat_now,
  kstar_hat_at_tile, p) -> bool` implementing the hysteresis `|Δk̂|/k̂ > 1/(2p)` (half a
  solve-box bin); `solve_box.interpolate_targets(log_raps_obs, k_obs, kstar_obs, box) -> Tensor`
  putting the observed log-RAPS targets onto the solve box's bin centres **expressed in
  k/k\*_obs units**, once per re-tile.
- `k̂(θ)` is the current dispersion argmax and is **detached** — solve-box geometry is not
  differentiated. Assert that in a test.
- Band masks stay defined relative to `k*_obs` exactly as in `docs/SPEC_fourier_training.md` §4.
  **`SPEC_fourier_training.md` is a hard-stop file — do not edit it.**

- [ ] **Step 1: Failing tests** — (a) `L_solve` gives exactly `p` periods of `k̂` in the box;
  (b) hysteresis fires at `|Δk̂|/k̂` just above `1/(2p)` and not just below; (c) `k̂` carries no
  gradient into `L_solve` (`.requires_grad is False` on the geometry, and a `backward()` through
  a loss computed on the box does not populate a grad path via `L`); (d) re-tiling clears warm
  starts and rebuilds the k-grid.
- [ ] **Step 2–4: FAIL → implement → PASS.** `p = 8` (nearest integer to D3's measured 8.36 data
  periods, and D1's verified 8-period geometry) and `n = 96` (12 px/wavelength, D1's verified
  density *for the gradient*) are **spec-fixed starting values**, with `n` explicitly subject to
  Task 15's measurement. Comment both at their code sites naming their calibrator.
- [ ] **Step 5: Measure the interpolation error** and write it into the task report. Spec §4.3:
  "Interpolation error is *expected* far below the ~31 %/bin floor; it is **UNMEASURED** and is
  measured once per re-tile geometry at R3." This step is that measurement. It runs on CPU and
  does not need the GPU.
- [ ] **Step 6: Full suite; commit.**
  `git commit -m "feat: adaptive commensurate solve box with re-tile hysteresis (spec 4.3)"`

**Model hint:** opus. Numerics with a firewall touch and a spec-binding geometry.

### Task 12: The truncated-unrolled gradient path

**Files:**
- Modify: `src/rngrn/etdrk4_torch.py` (a differentiable/checkpointed unrolled mode — spec §6's
  survives-and-gains row)
- Create: `src/rngrn/unrolled.py` (the truncated-backprop driver), or add to `forward.py` if the
  integrated shape makes that natural — read the code and choose
- Modify: `tests/test_firewall.py` (classification — **hard-stop return first**)
- Test: `tests/test_unrolled_grad.py` (create)

**Interfaces (spec §4.2):**
- Produces: an unrolled-relax gradient path — ETDRK4 run as differentiable torch ops with
  **gradient checkpointing**, the warm-started state **detached**, and only the final saturated
  segment differentiated. **Segment length is UNCALIBRATED**, set from the measured
  gradient-error-vs-length curve (spec §8 item 14, assigned to R3) — that curve is this task's
  deliverable, and it is what Task 14 A/Bs against the adjoint.
- Requires **no convergence at all**: the loss is defined on the relaxed field after finite time,
  so the F-D1-5 stall class cannot touch it. That is the entire point of the path.

- [ ] **Step 1: Failing tests** — (a) gradients flow through the unrolled segment to `theta`;
  (b) the detached warm-start contributes **exactly zero** gradient; (c) checkpointed and
  non-checkpointed unrolled gradients agree to float64 round-off on a short segment;
  (d) activation memory scales with segment length, not total steps (assert on a torch memory
  counter or on a synthetic proxy — read what the codebase already uses).
- [ ] **Step 2–4: FAIL → implement → PASS.**
- [ ] **Step 5: Measure the gradient-error-vs-segment-length curve** at the n=96 training grid:
  unrolled-vs-FD relative error against segment length, at several lengths, on the commensurate
  box of Task 11. Record which lengths explode — spec §4.2 warns gradients through the
  exponential-growth phase of the instability can blow up, which is *why* the path is truncated.
  Write the curve to `experiments/redesign_r3/unrolled_segment/results/curve.json` (tracked).
- [ ] **Step 6: Decision point (return, do not rule)** — the segment length to adopt. It is an
  UNCALIBRATED number in register item 14; report the curve and your recommendation.
- [ ] **Step 7: Full suite; commit.**
  `git commit -m "feat: truncated-unrolled gradient path with checkpointing; measured segment-length curve"`

**Model hint:** opus.

### Task 13: Stall accounting and the two-path switch

**Files:**
- Modify: `src/rngrn/recover.py` (per-run stall-rate counter; per-member gradient-path
  selection), `src/rngrn/history.py` (a `"stall"` event already exists in R2's Task 9 event
  vocabulary — reuse it, do not add a second)
- Test: `tests/test_stall_accounting.py` (create)

**Interfaces (spec §4.3):**
- Produces: a per-run counter of ignited-member solves missing the **1e-9 Newton bar** on the
  commensurate box, and a per-member switch to the truncated-unrolled path for those members.
- **The ~20 % switch fraction is UNCALIBRATED** — spec §4.3 says explicitly: "no baseline
  exists; calibrate at R3 from the measured stall-rate distribution, with the switch fraction
  chosen against the measured gradient-error difference between the two paths, not for
  convenience." Task 16 supplies the distribution; this task supplies the machinery and the
  counter.
- **F-D1-5 option (b) — loosening the convergence bar — stays off the table.** It touches a
  pre-registered-class threshold and is owner-only. If any step tempts you toward it, that is a
  hard stop.

- [ ] **Step 1: Failing tests** — the counter increments on a member that misses 1e-9 and not on
  one that meets it; a stalled member's gradient comes from the unrolled path and a converged
  member's from the adjoint; the `"stall"` event lands in `TrainingHistory` with the right
  member and step; the counter survives into the run index as a **flat scalar** (§4 — rows are
  built by `dict.update()`, never `**`-expansion; run identity wins on collision).
- [ ] **Step 2–4: FAIL → implement → PASS.** Ship the switch fraction as a config knob whose
  default is the spec's ~20 % **with an `UNCALIBRATED` comment naming Task 16 as its
  calibrator**.
- [ ] **Step 5: Full suite; commit.**
  `git commit -m "feat: stall accounting and per-member gradient-path switch (spec 4.3, UNCALIBRATED fraction)"`

**Model hint:** sonnet. Clear spec, clear tests, no open design question.

---

# Phase D — the measurements R4 is gated on, and the ledgered deferrals

Tasks 14–17 are **GPU work under the guard**. Read the Global Constraint about the paper sprint
before launching any of them. Tasks 18 and 19 are CPU-only and can run in their own worktrees
**concurrently** with anything above — Task 18 in particular is independent of the whole
integration and should be dispatched early if there is a free agent.

### Task 14: The FD A/B on both gradient paths — R3's hard gate

**Files:**
- Create: `scripts/r3_fd_ab.py` (thin driver; **if it opens `payload.h5` or reads an `AnswerKey`
  it must be appended to `FORBIDDEN`** — it should not need to, the observed frame is enough,
  but check and report)
- Output: `experiments/redesign_r3/fd_ab/results/*.json` (tracked), plus the report

**Interfaces (spec §4.2, the A/B discipline — this is a gate, not a measurement):**
- **10 directions × the active loss terms, tol 1e-4**, run on **both** gradient paths at the
  **same θ**, at the operating point: commensurate box (Task 11), training grid n=96,
  training-geometry θ.
- Reference points that exist: D1 measured worst FD relative error **6.4e-8** over the spectral
  terms and **8.1e-8** including `real_moments`, vs tol 1e-4, at 96². Those were on the *serial*
  adjoint at a non-commensurate box; this run is the commensurate-box, batched-path version.
- **The gate:** spec §7 — "Hard gate: FD-faithfulness at tol 1e-4." A path that misses it does
  not feed a training run.
- **Register item 8's promotion rule is pre-specified and must not be re-decided here:** the
  unrolled path may become primary **only if** the FD A/B shows it FD-faithful at tol 1e-4 on
  **both converged and stalled members** AND its measured cost does not exceed the adjoint
  path's. Report against that rule; the promotion itself is an owner-decision item.

- [ ] **Step 1: Write the driver.** Reuse `tests/test_forward_solve.py`'s existing FD harness
  rather than writing a third one — read it first.
- [ ] **Step 2: Launch guarded, unsandboxed.**
  `bash scripts/guarded_run.sh .venv/bin/python scripts/r3_fd_ab.py --out experiments/redesign_r3/fd_ab`
- [ ] **Step 3: Report** per `.claude/rules/reporting-numbers.md`: worst relative error per path
  per term, on converged and stalled members separately, plus measured per-step cost of each
  path. Name the run directory for every number.
- [ ] **Step 4: Decision point (return, do not rule)** — the gate verdict, and whether register
  item 8's promotion condition is met. Both belong to the orchestrating session with the numbers
  in front of it.
- [ ] **Step 5: Commit results.** `git commit -m "data: R3 FD A/B on both gradient paths at the operating point"`

**Model hint:** opus.

### Task 15: 96²-vs-512² RAPS band fidelity — the measurement that may move `n`

**Files:**
- Create: `scripts/r3_raps_fidelity.py` (reads the observed frame — check whether it needs
  `payload.h5`; **if yes, append to `FORBIDDEN`**)
- Output: `experiments/redesign_r3/raps_fidelity/results/*.json` (tracked)

**Why (spec §4.3, stated there as a gap):** training compares a **96² model spectrum** to a
**512² observed spectrum** with the model-side estimator's fidelity **UNMEASURED at that grid**.
D2 measured band-limited spectral quantities agreeing with 512² at **256², not at 96²**, where
k\* one-bin agreement fails **2/6** (`docs/DIAGNOSTICS_fft.md` D2).

- [ ] **Step 1: Measure the 96²-vs-512² RAPS band distance on the fixture**, on
  `B_train = [0.60, 1.55]·k*_obs` (D-FFT-9 closure 1), using the **unwindowed** primary
  estimator (register item 15 — the Hann–Welch path is report-only until calibrated).
- [ ] **Step 2: Compare against D3's ~31 %/bin estimation floor.** Spec §4.3: "if it does not
  clear D3's ~31 %/bin estimation floor, **n rises to the coarsest grid that does**."
- [ ] **Step 3: If `n` must rise — decision point (return, do not rule).** Raising `n` changes
  the cost of every Phase II step and therefore Task 17's B curve; it is not a local choice.
  Report the coarsest grid that clears the floor, and its measured per-step cost.
- [ ] **Step 4: Fold Task 11 Step 5's interpolation-error number into the same report** — spec
  §4.3 pairs them ("measured once per re-tile geometry at R3, alongside the FD check").
- [ ] **Step 5: Commit results.** `git commit -m "data: 96²-vs-512² RAPS band fidelity and target-interpolation error"`

**Model hint:** sonnet with the spec section pasted in; the decision point returns.

### Task 16: The stall-rate distribution

**Files:**
- Modify: `scripts/r3_fd_ab.py` or create `scripts/r3_stall_survey.py`
- Output: `experiments/redesign_r3/stall_rate/results/*.json` (tracked)

- [ ] **Step 1: Run a short population on the commensurate box** and record, per member per
  step, whether the Newton polish met the 1e-9 bar. Guarded, unsandboxed.
- [ ] **Step 2: Report the distribution**, not just the mean — spec §4.3 asks for the
  *distribution* because the switch fraction is chosen against it.
- [ ] **Step 3: Decision point (return, do not rule)** — the calibrated switch fraction,
  chosen against the measured gradient-error difference between the two paths (Task 14's
  numbers), never for convenience. This closes one of register item 14's UNCALIBRATED entries.
- [ ] **Step 4: Commit results + the `DECISIONS.md` entry once the ruling lands.**

**Model hint:** sonnet.

### Task 17: The B/K curve with the forward solve in the loop

**Files:**
- Create: `scripts/r3_batch_curve.py`
- Output: `experiments/redesign_r3/batch_curve/results/*.json` (tracked)

**Why this is a new measurement and not a re-read of T16:** see Global Constraints. T16's curve
is the **ignition-only** path, where the pinned model skips the steady-state Newton solve
entirely. R3's Phase-II step contains a batched relax + per-member Newton + LSMR adjoint at
n = 96. `_half_coeffs_batched` alone allocates ~58 MB at b=8/n=96 and **~230 MB at b=32/n=96**
(arithmetic from shapes, checkable on its face), per batched solve, per Adam step, with **no cap
and no `MemAvailable` interaction** (review M8) — on a machine whose binding resource is host RAM
(§7a, `MemTotal` 17.58 GiB, guard floor 8192 MB).

- [ ] **Step 1: Read T16's final numbers** from `task-16-report.md` (or the run JSONs) and put
  them in the report as the ignition-path row, clearly labelled as **not** parameterizing this
  one.
- [ ] **Step 2: Sweep B** on the Phase-II step with the forward solve in the loop, recording
  `s_per_step`, `member_steps_per_s`, `peak_rss_mb`, `cuda_max_alloc_mb`,
  `mem_available_mb_after` — the same key set T16 used, so the two curves are readable side by
  side. **A rung that cannot fit is a measurement, not a failure** — record it and stop the
  sweep there.
- [ ] **Step 3: Report** the largest B that fits under the §7a guard, and K's implied ceiling
  (the cull cannot keep more members than there are distinct structures — T16 measured the
  distinct-structure count).
- [ ] **Step 4: Decision point (return, do not rule)** — the B and K values R4 will run at.
  **B = 512 and K = 16 stay UNCALIBRATED until this ruling** (register item 14).
- [ ] **Step 5: Commit results.** `git commit -m "data: R3 batch-size curve with the batched forward solve in the loop"`

**Model hint:** sonnet; the guard and the key set are prescribed.

### Task 18: Diagnose D-R2-1 — why the pinned+boxed ignition objective does not ignite

**Worktree:** its own, `../worktrees/r2-ignition-diag`, branch `feature/r2-ignition-diag`, BASE
`feature/redesign-model` at T16's head. **Independent of Phases A–C — dispatch it early, in
parallel.**

**Files:**
- Create: `docs/DIAGNOSTICS_r2_ignition.md`
- Output: `experiments/redesign_r2/ignition_diag/results/*.json` (tracked)

**Scope: diagnosis and a decision-point return. NOT an open-ended fix.** Changing the objective
is R4's business and is register-item class; this task's deliverable is a cause with evidence
behind it and a ranked list of candidate fixes.

**What T16 measured (PROVISIONAL until its report is reviewed — say so in the doc):** at B = 64,
the **r2 arm reached 0/64 Turing against a0's 3/64**, and at B = 128 it reached 0/128. Two
mechanisms are already on the table from T16's interim finding:
- **population decoupling** — off-diagonal/diagonal coupling median **0.00268**, i.e. the
  members are converging to near-uncoupled systems, which cannot pattern by construction;
- **box-sigmoid recentring** — the boxes move the *initialisation* by **α 7.58× and δ 3.88×**
  relative to legacy, so the r2 arm does not start where the a0 arm starts.

- [ ] **Step 1: Reproduce** the 0/64 result from the committed config, and confirm both measured
  numbers from the run's own arrays. If they do not reproduce, that is the finding — stop and
  report.
- [ ] **Step 2: Separate the two mechanisms by ablation.** The obvious 2×2: {pinned, unpinned} ×
  {boxed, unboxed}, same seeds, same steps, ignition terms only. Which cell recovers a0's
  ignition rate isolates the cause. Add a third axis only if the 2×2 is ambiguous.
- [ ] **Step 3: Test the decoupling hypothesis directly.** Is 0.00268 where the members *start*,
  or where they *go*? Plot the coupling-median trajectory from `TrainingHistory` (R2 Task 9's
  invariants are recorded for exactly this). A monotone decay is an objective-gradient story; a
  flat line at the init value is a parameterization story.
- [ ] **Step 4: Check `kstar_si` specifically.** Its `temp` (inherited 60.0, never swept) and
  `eps` are **UNCALIBRATED** and spec §4.4 assigns their sweep to R2 — which did not happen.
  A flat/degenerate `kstar_si` gradient at these boxes' scale is a live candidate. Measure the
  term's gradient magnitude against `turing_hinges_split`'s at the r2 init; if `kstar_si` is
  orders down, that is the answer.
- [ ] **Step 5: Write `docs/DIAGNOSTICS_r2_ignition.md`** — every number citing its run path,
  what is *not* known stated as plainly as what is, and a ranked list of candidate fixes with the
  evidence for each.
- [ ] **Step 6: Run `evidence-auditor` over the doc; fix findings.**
- [ ] **Step 7: Decision point (return, do not rule)** — the diagnosis and the recommended fix.
  Any objective change is register-item class and goes to the owner via the controller. **Note
  loudly:** this does not block the paper — the paper's claims rest on the legacy D5/FFT line
  and the lift ladder, not on the r2 arm.
- [ ] **Step 8: Commit; push.** `git commit -m "docs: D-R2-1 diagnosis — why the pinned+boxed ignition objective decouples"`

**Model hint:** opus. Open-ended diagnosis on a live negative result.

### Task 19: Fix F-L10 — `l2_diff_dev_by_mu` returns a finite number 288 orders out of range

**Worktree:** `../worktrees/lift-ladder` (the R1 tree, branch `feature/lift-ladder` @ `52ef7dd`)
or a fresh branch off it. CPU-only; can run any time.

**Files:**
- Modify: `src/rngrn/eval/ladder.py` (`_rel_l2_dev`'s denominator guard)
- Modify: `docs/DIAGNOSTICS_lift.md` (F-L10's "Reported, not fixed" line)
- Test: `tests/test_lift_ladder.py`

**The defect (F-L10, `docs/DIAGNOSTICS_lift.md:453`):** `harvest/nc1__immobile/1`'s 512² anchor
run has a QSS control that decayed to exactly zero amplitude, so `_rel_l2_dev`'s
deviation-from-channel-means denominator sits at its **1e-300 guard** and the function emits
**9.4e+288**. The raw `l2_diff_by_mu` on the same row is **1.75e-15** — correct, both fields are
zero. `CLAUDE.md` §8: a value that arrived this way is a **defect, not a datum**.

- [ ] **Step 1: Failing test** — a zero-amplitude control field must produce `NaN` (documented
  as deliberate at the site that produces it) or a raise, never a finite number. Assert the raw
  `l2_diff_by_mu` is unaffected on the same input.
- [ ] **Step 2–4: FAIL → implement → PASS.** Fail loud (§4). If you choose NaN over a raise, the
  site must carry the comment saying why the NaN is meaningful, per §4's rule for deliberate NaN.
- [ ] **Step 5: Re-emit the affected row** if cheap, or state in `DIAGNOSTICS_lift.md` that the
  committed JSON predates the fix and which key to read instead. Do **not** silently leave a doc
  saying "reported, not fixed" once it is fixed.
- [ ] **Step 6: Full suite; commit; push.**
  `git commit -m "fix: _rel_l2_dev raises/NaNs on a zero-amplitude control instead of emitting 9.4e+288 (F-L10)"`

**Model hint:** sonnet. Precisely diagnosed, single-site.

### Task 20: The L2 gate-licensing run, and the V3(b) μ_gate re-run

**Worktree:** `../worktrees/lift-ladder`. **GATED ON MACHINE AVAILABILITY — run this LAST.**
It is the single most expensive item in the plan and it must not collide with the paper sprint's
GPU work. Check `lsof` on `.trainer.lock` **unsandboxed** before launching; if anything is live,
queue it and report that it is queued.

**Files:**
- Modify: `src/rngrn/eval/lifted.py` (`simulate_lifted`'s `max_steps` default 200 000 — the
  required step counts are 3.5e5–5.2e6; **it is a knob, not a bar**, but raising it is a change
  to a default and the report must say so)
- Modify: `scripts/lift_ladder.py` (per-rung args already exist — read them)
- Output: `experiments/lift_ladder/v3_gate/results/*.json` and `arrays/*.npz` (tracked)
- Modify: `docs/DIAGNOSTICS_lift.md` §7 (the L2 verdict), `docs/REDESIGN_rngrn.md` §5.3 (F-L13's
  recorded deviation) — **the REDESIGN edit is a controller doc edit on `docs/redesign-rngrn`,
  returned as exact text, not made by the unit.**

**Two runs, one task.** Both close deviations `docs/DIAGNOSTICS_lift.md` §7.2 records.

**(a) L2's own operating point.** §5.4's L2 will be evaluated at **512², μ_gate = 1e-3,
dt = μ/2 = 5e-4, horizon 40/|σ_max|, with the dt-halving check** — and that configuration has
never been run. The two existing V3(b) legs cover complementary halves and **their union is not
it**: the dt-halving pair runs at the policy dt but on a 1.44-time-unit transient at 128² and at
μ_central = 7.2e-4; the 512² leg reaches the attractor and agrees 8/8 but at μ_central and at the
QSS dt where **dt/μ = 26.4–252.8** — exactly the coupling trap §5.2 names, the regime in which
the lifted run is *expected* to reproduce the QSS run, so 8/8 there is weak evidence for L2.
Cost arithmetic from the 8 anchor systems' measured σ_max at dt = 5e-4 and the slower measured
CUDA cost (6.378 ms/step at 512²): cheapest **3.54e5 steps ≈ 0.63 h**, median **5.49e5 ≈ 0.97 h**,
dearest **5.20e6 ≈ 9.21 h**; a halving *pair* is ~3× one leg, so one cheap gate-representative
field is **~2 h**, the median ~3 h, all 8 at both dts ~57 h. **Run at least one
`morphology_claimable` system** — `competitive__mobile3/1` and `nc1__immobile/0` are the cheap
ones that also pattern.

**(b) The V3(b) anchor re-run at μ_gate = 1e-3** (controller ruling, 2026-08-18). The original
anchor ran at μ_central = 7.2e-4; the deviation is recorded as **F-L13** in `REDESIGN` §5.3.
Same machinery, same driver: a **128² dt-halving transient pair** plus **one 512² field at the
QSS dt**, both at μ_gate. This closes the deviation on record.

- [ ] **Step 1: Check the machine is free** (`lsof` on `.trainer.lock`, unsandboxed; `free -h`).
  If not, stop and report queued. Do not launch alongside.
- [ ] **Step 2: Raise `max_steps`** and confirm the run reports `stopped_reason == "horizon"` —
  §5.4's L2 already demands it. A run that stops for any other reason does not license anything.
- [ ] **Step 3: Launch (b) first** — it is the cheaper of the two and closes a recorded
  deviation. Guarded, unsandboxed, detached.
- [ ] **Step 4: Launch (a)** on one cheap `morphology_claimable` system, both dts.
- [ ] **Step 5: Report** per `reporting-numbers.md`. Say **"Turing-unstable"** or **"patterns"**,
  never one as a proxy for the other.
- [ ] **Step 6: Decision point (return, do not rule)** — whether L2 is now licensed. D-LIFT-1
  makes the gate conditional on the ladder; the verdict updates D-LIFT-1's precondition and
  belongs to the orchestrating session. **Weakening a pre-registered pass condition is
  owner-only** and is not on the table here — the run either clears the bar or it does not.
- [ ] **Step 7: Commit results; push.** `git commit -m "data: L2 gate-licensing run at 512²/μ_gate + V3(b) anchor re-run at μ_gate (F-L13)"`

**Model hint:** opus for the verdict framing; the run itself is mechanical.

---

# Phase E — docs, auditors, and the rulings roll-up

### Task 21: The docs pass

**Files:**
- Modify: `CLAUDE.md` §5 (the FORBIDDEN-count drift), `docs/PLAN_redesign.md` (plan-of-plans
  R3 pointer → "planned in `PLAN_redesign_R3.md`"), `docs/REDESIGN_rngrn.md` §4.1 (descriptive
  drift), `docs/TUNING.md` (new knobs)
- Create: `docs/DIAGNOSTICS_r3.md`

- [ ] **Step 1: Fix `CLAUDE.md` §5's FORBIDDEN count.** It says **12** names;
  `tests/test_firewall.py` has **27**. Re-read the test file, not this plan, for the true list
  and count, and re-state the paragraph so it points at the file as authority rather than
  restating a list that drifts. The source wins and the doc gets fixed (§8) — **in this change,
  not later.**
- [ ] **Step 2: Correct `REDESIGN` §4.1's descriptive drift.** It says the spectral terms and
  forward solve are "**Not batched, and refused by name**". After Phase B they are batched and
  the refusal is replaced by input validation. **This is a controller doc edit on
  `docs/redesign-rngrn`** (REDESIGN is not a hard-stop file, but it is the ratified spec — the
  unit returns exact text, the controller applies it, per the T5 §5.2 / T6 §5.3 precedent). The
  §4.1 *mandate* — an equivalence test before the batched path trains anything — is unchanged
  and is now satisfied by Phase A.
- [ ] **Step 3: Write `docs/DIAGNOSTICS_r3.md`** in the house pattern of `DIAGNOSTICS_fft.md` /
  `DIAGNOSTICS_lift.md`: per-measurement method, measured numbers with run paths, PASS/FAIL/
  finding, and an explicit "what this does NOT establish" section.
- [ ] **Step 4: Roll up every UNCALIBRATED number R3 closed** into `docs/DECISIONS.md`, and
  leave the ones it did not close explicitly still marked. Register item 14's R3 entries: the
  ~20 % stall-switch fraction (Task 16), the truncated-backprop segment length (Task 12), B and
  K (Task 17).
- [ ] **Step 5: `evidence-auditor` over everything written in Phase E; fix findings.**
- [ ] **Step 6: Commit; push.**

### Task 22: Final whole-branch reviews and the rulings roll-up

Per `.claude/rules/pre-merge-checklist.md`, which binds whatever produced the branch.

- [ ] **Step 1: `merge-damage-hunter`** over the whole of `feature/r3-integration`.
- [ ] **Step 2: `firewall-auditor`** — the branch touched `model.py`, `recover.py`, `losses/`,
  `eval/`, `forward.py`, `observables.py`. Confirm every new module under `losses/`/`eval/` is
  classified, every new top-level recovery-side module is on a list, and every new `scripts/`
  module that opens `payload.h5` is in `FORBIDDEN`. Confirm no criterion used to *judge*
  recovery is derived from ground truth — `kstar_obs` from an FFT of the observed frame is
  legal; anything from the generating parameters is not.
- [ ] **Step 3: `evidence-auditor`** — the branch added thresholds, metrics and claims. Every
  new threshold is calibrated against a control or explicitly `UNCALIBRATED` in **both** the
  code and `docs/DECISIONS.md`.
- [ ] **Step 4: `numerics-reviewer`** — the branch touched the dispersion backend, the damped
  Newton, the k-grid, and added an unrolled integrator path. It knows that **training never
  simulates** and that the Laplacian is analytic; the §4.2 unrolled path is the first thing in
  this repo that makes training simulate, on one gradient path only, and that is worth a
  numerics reviewer's eyes specifically.
- [ ] **Step 5: Address every Critical and High; fix Medium where possible; report Low.**
- [ ] **Step 6: Data check** — `find data/datasets -name payload.h5 -type f` in every R3
  worktree; any real (non-symlink) payload harvested to the canonical store in the main checkout
  and its checksum verified against the tracked `manifest.json`, per §6a.
- [ ] **Step 7: Full suite unsandboxed; record the count.** Push (hooked; SIGPIPE protocol:
  verify `git ls-remote`, deliberate `--no-verify` retry, record the occurrence).
- [ ] **Step 8: Write `docs/HANDOFF_r3.md`** for a **zero-context reader**: what changed, what
  is verified, what is **not**, and where to start.
- [ ] **Step 9: Roll the rulings up to the owner** — every decision point returned during R3,
  the verdict given, and the evidence behind it. Flag separately anything that is owner-only:
  register item 8's promotion (if Task 14's numbers meet it), the A0/`auto`-backend
  comparability question (Task 8), and any objective change Task 18 recommends.
- [ ] **Step 10: NO MERGING.** The owner validates first (§2, §3). Passing tests alone is not
  the bar, and `superpowers:finishing-a-development-branch`'s offer to merge locally off a green
  suite **does not apply here**.

---

## Self-review (done at authoring)

- **Spec coverage.** §4.1 (batched spectral + forward, with the mandated equivalence tests) →
  Phase A Tasks 1–4 and Phase B Tasks 9–10. §4.2 (two gradient paths, A/B discipline, register
  item 8's pre-specified promotion rule) → Tasks 12, 14. §4.3 (commensurate box, hysteresis,
  target interpolation, stall accounting, the FD gate, the 96²-vs-512² fidelity measurement) →
  Tasks 11, 13, 15, 16. §7's R3 deliverables — the A/B report, the fidelity measurement, the
  stall-rate distribution — are Tasks 14, 15, 16 respectively. §8 item 14's R3-assigned
  UNCALIBRATED numbers (stall-switch fraction, segment length, B and K) are Tasks 16, 12, 17.
  Deliberately *not* covered, per the plan-of-plans: the §4.5 cull and Phase II campaign, §4.6
  generation, §4.7 weight noise, §5.4 gate runs (R4); §4.9 hidden channels (R5).
- **Owner-ratified constraints, each with an owning task.** The three gpu-optim entry tasks open
  the plan (Tasks 1, 2, 3, in the mandated order). Integration is per-hunk with per-twin
  equivalence tests (Phase B), all 26 review collision points plus 2 discovered here get an owner
  in Task 7's ledger and a verification in Task 10. The 7 missing `DECISIONS.md` entries are
  written by the tasks that adopt each change (Task 1: refusal deletion; Task 3: relax-detector
  swap; Task 5: LSMR cadence, ETDRK4 blow-up, liveness cadence, `lbfgs_error`, `bdf1` raise).
  `CLAUDE.md` §5's FORBIDDEN count is Task 21 Step 1. D-R2-1 diagnosis is Task 18, scoped as
  diagnosis + decision-point return. F-L10 is Task 19; the L2 licensing run plus the V3(b)
  μ_gate re-run is Task 20, gated on machine availability and ordered last.
- **Ordering for one overnight session.** Entry tasks and integration first (1–10); gradient
  machinery next (11–13, CPU-testable); GPU measurements after (14–17); the two independent CPU
  deferrals (18, 19) can be dispatched in parallel from the start; the expensive L2 run (20)
  last, gated. Phase E closes.
- **Placeholder scan.** Two things are deliberately read-at-execution rather than baked in:
  T16's final B/K numbers (its report did not exist at authoring — the provisional table is
  flagged NOT VETTED and the plan says to prefer the report), and `feature/redesign-model`'s
  T16 head SHA. Everything else names a file, a line, or a measured number with its source.
- **Type consistency.** Task 11's `solve_box.geometry` dict feeds Tasks 12–16; Task 12's segment
  length and Task 16's switch fraction both feed Task 13's config knobs; Task 14's per-path cost
  feeds Task 16's switch-fraction calibration and register item 8's promotion test; Task 15's
  possible `n` rise feeds Task 17's B curve, which is why 17 is ordered after 15.
- **Known conflict carried forward, not resolved.** `docs/PLAN_redesign.md`'s "A0 is
  untouchable — bit-identical behaviour after every task" versus D-PERF-3's `dispersion_backend`
  default flip, which its own entry says makes runs "not bit-comparable". Task 8 Step 3 returns
  it as a decision point with the fourth complicating fact (T16's a0 arm already ran at `cubic`)
  rather than picking a side inside a unit.
