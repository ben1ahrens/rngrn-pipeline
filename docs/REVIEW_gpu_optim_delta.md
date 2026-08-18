# Review — `feature/gpu-optim`, `10cff1b..2b11010` (5 commits)

Reviewer: read-only pass over the full diff (4051 lines) plus targeted verification against the
source in `/home/benja/projects/personal/rngrn/worktrees/gpu-optim`. No files in that worktree
were edited. Two measurements were taken in scratch scripts under `/tmp` against the worktree's
own `.venv` (recorded inline below); the authoritative suite run is the controller's.

Lenses: `CLAUDE.md` §5 (firewall), §7/§7c (compute reality; training never simulates), §8
(evidence discipline), §10 (decisions must land in `docs/DECISIONS.md`), §11 (merge damage),
plus `.claude/rules/reporting-numbers.md`.

---

## 1. Verdict summary

**Unsafe to integrate as-is** — as the base implementation for R3.

The delta splits cleanly in two, and the two halves deserve different verdicts:

* **`346bbf2` + `e7d01bd` + `375cb4a` (the "hoist and desync the serial/batched loss" half)** —
  substantially sound. The changes are arithmetic-neutral or documented, the reasoning in the
  comments holds up against the source, and the one default flip (`dispersion_backend='auto'`)
  has an honest `docs/DECISIONS.md` entry that states its own limits. **Approve with minor fixes.**
* **`8f93dfe` + `2b11010` (the forward-solve desync and the batched `PatternSolver`)** —
  ~1,150 lines of new and rewritten gradient machinery with **zero test coverage**, **two live
  test failures**, **three undocumented training-time semantic changes**, and a module docstring
  quoting performance numbers with no run behind them. **Unsafe to integrate as-is.**

The controller's suite run (unsandboxed, worktree venv, 2 failed / 605 passed / 24 skipped,
7:18) is direct evidence that `2b11010` was never run green. That is not a stylistic complaint:
it means nothing in the batched forward-solve path has been validated by anything, at any point.

The *encouraging* half of the picture, stated as plainly as the problems: I probed the batched
spectral term functions directly (a (4, 3, 64, 64) stack, two channels, CPU float64) against
their serial twins member by member, and they agree to **≤ 4.3e-14 relative, worst term**
(`spec_amp_fluct`; the other four were exactly 0.0). The batched *terms* look right. What is
untested is the layer beneath them — the batched relax, the per-member Newton, and above all
`BatchedPatternSolve.backward`'s adjoint gradient, which is precisely where a silent bias would
live and where D-FFT-10's tripwire exists because it has bitten before.

**Counts: 3 Critical, 6 Important, 10 Minor.**

---

## 2. Critical

### C1 — The delta deletes a contract that three tests still pin; the suite is red as committed

`src/rngrn/losses/total.py:281-291` (refusal block deleted), `src/rngrn/recover.py` (the
`use_spectral and batched` refusal deleted), against `tests/test_ignition_gating.py:158`,
`:165`, `:194`.

Confirmed against the diff: `2b11010` removes the "`compute_terms_batched` cannot compute the
spectral terms" `ValueError` from `compute_terms_batched`, and removes the matching
`use_spectral and batched` refusal from `recover()`. `git diff --stat 10cff1b..2b11010 -- tests/`
is **empty** — no test file was touched anywhere in the range.

Consequences, all three of them:

1. `tests/test_ignition_gating.py:158 test_compute_terms_batched_refuses_spectral` — **FAILS**.
2. `tests/test_ignition_gating.py:165 test_total_loss_batched_refuses_spectral` — **FAILS**.
   Both now die on `AttributeError: 'NoneType' object has no attribute 'B'` because, with the
   refusal gone, the placeholder `model=None` flows into `T.steady_state_batched(model)`.
3. `tests/test_ignition_gating.py:194 test_recover_raises_on_batched_with_a_spectral_weight` —
   **PASSES, VACUOUSLY**. It asserts `pytest.raises(ValueError, match="batched")` and does not
   pass `lbfgs_steps`, so it now trips `recover()`'s *LBFGS* guard ("batched=True requires
   lbfgs_steps=0"), whose message happens to contain the word "batched". The test is green and
   no longer tests what its name says. This is the more dangerous of the three, because a red
   test gets fixed and a vacuously green one does not.

**Failure scenario:** the branch cannot merge; and the one test that was supposed to guard the
batched/spectral combination will keep passing after the other two are fixed, so whoever repairs
this will believe the combination is covered when it is not.

### C2 — The entire batched forward solve, including its IFT adjoint, has no test of any kind

`src/rngrn/forward.py:906` (`batched_reaction_fields`), `:930` (`make_spatial_F_batched`),
`:998` (`_half_coeffs_batched`), `:1023` (`_kstar_of_torch_batched`), `:1054`
(`relax_to_pattern_torch_batched`), `:1141` (`BatchedPatternSolve`), `:1196`
(`BatchedPatternSolver`), plus `losses/spectral.py`'s nine `*_batched` functions and
`losses/total.py:188` (`_apply_spectral_batched`).

A repo-wide grep finds zero references to any of these symbols under `tests/`. Not an
equivalence test, not a smoke test, not an import. The only tests that touch batched-spectral
wiring (`test_ignition_gating.py`, `test_losses.py`) deliberately substitute a hand-written
`_StubSpectralSolver`, and `tests/test_batched.py`'s own docstring scopes itself to "reaction,
Jacobian, steady state, dispersion and every loss term" — it never imports `forward.py`.

The redesign mandates an equivalence test for every batched twin. Here is what that means
concretely, ranked by what would actually catch a defect:

* **`BatchedPatternSolve.backward` (`forward.py:1148`) is the load-bearing gap.** The serial
  twin is pinned by `tests/test_forward_solve.py` against finite differences (`best <= 5e-3`)
  and by an adjoint-vs-scipy check (`rel < 1e-6`). The batched form loops `solve_adjoint` per
  member and then does **one** stacked `torch.autograd.grad` through `make_spatial_F_batched`
  with `grad_outputs=-lam_stack`. That final scatter — whether the gradient lands on exactly
  row `members[j]` of each `(B, ...)` parameter tensor and is zero elsewhere — is asserted in
  the docstring and checked by nothing. A transposed or broadcast-collapsed scatter would not
  crash; it would train every member on a blend of every other member's gradient and look like
  slow convergence.
* `relax_to_pattern_torch_batched`'s snapshot-on-finish semantics (a member's field is the one
  it held when *its* detector fired, not the stack's final state) — nothing checks that a
  member's snapshot equals what the serial `relax_to_pattern_torch` would have returned.
* `_kstar_of_torch_batched`'s mask-vs-slice centroid, argued equal "to floating-point
  associativity" — untested.
* The `*_batched` spectral terms — **I measured these** (see §1): ≤ 4.3e-14 relative against
  serial. That measurement is not a test and does not survive into CI.

**Failure scenario:** an R3 phase built on this inherits a gradient path whose correctness rests
entirely on the docstrings of the session that wrote it — a session that, per the coordinator,
was interrupted and left no report.

### C3 — `observables.raps_torch` is promoted to a training-loop decision variable against its own docstring's explicit instruction

`src/rngrn/observables.py:108-157` (`raps_torch`), `:158` (`kstar_of_torch`);
callers added in this range: `forward.py::relax_to_pattern_torch` (serial GPU relax detector)
and `forward.py:1023 _kstar_of_torch_batched` (batched relax detector).

The function's own docstring, written in this same delta, says:

> "PARITY PORT: faithful by construction and now MEASURED against `raps` on patterned and noise
> fields (above), but still **not PINNED by a test — add one before any further caller relies on
> it**." (`observables.py:138-140`)

and, on the `floor` vs `digitize` binning divergence:

> "switching would change values for a shared observable, so it is **left as a flagged
> follow-up, not taken here**." (`observables.py:135-136`)

Two new callers were then added in the same delta, and no test was added. The measured
divergence is not hypothetical: the docstring records **one bin differing by 14 % and k\* by
0.24 %** on a 16² noise field. The consumer is the saturation detector's flatness ratio at
`flat_tol=1e-4` — four orders of magnitude below 0.24 %. The docstring argues the *patterned*
regime agrees to 3e-16, which is fair, but the detector runs from the *first* chunk, when the
field is x\* plus 1e-2 noise and is exactly the ill-conditioned regime where the divergence was
measured.

**Failure scenario:** the relax stops one chunk earlier or later than the numpy path, returning a
different u\*, feeding a different spectral loss, on a code path that has no test and whose
own author flagged it as not-ready. Nothing raises; the run just produces a different number
from the serial reference and no one can tell why.

---

## 3. Important

### I1 — `_lsmr_torch`'s deviations from scipy are real numerics changes with no `DECISIONS.md` entry

`src/rngrn/forward.py:180` (`_LSMR_STOP_CHECK_EVERY = 25`), `:183` (`_lsmr_torch`), `:147`
(`_sym_ortho_t`).

Three distinct departures from the D-FFT-10-verified minimal-norm solve:

1. The stopping test runs every 25 iterations instead of every iteration, so the solve can
   overshoot scipy's stop by up to 24 iterations and its iterate is not the one at the crossing.
2. `_sym_ortho` (Python floats, scipy-verbatim) is **deleted** and replaced by a branchless
   0-d-tensor `_sym_ortho_t` with guarded denominators.
3. The exact-Krylov-breakdown branch changes algorithm: scipy *skips* the `v` update entirely
   when `beta == 0`; the new code always runs it with `u / where(beta==0, 1, beta)`, leaving an
   unnormalised vector rather than terminating.

All three are documented honestly and at length in the docstring, and the argument that the
refinement loop's *true residual* still decides convergence is sound. But this is a science
decision under §10 — a change to the stopping semantics of a solve whose verified behaviour is
recorded in a numbered decision (D-FFT-10) — and `docs/DECISIONS.md` gained exactly one entry in
this range (D-PERF-3, about the dispersion backend). There is no D-PERF entry for the LSMR
change. The docstring names `tests/test_forward_solve.py` as pinning the agreement; that test
was written against the *faithful* port and was not re-examined.

**Failure scenario:** a future reader diffing the torch LSMR against scipy finds a mismatch,
looks in `DECISIONS.md` for the ruling that authorised it, finds nothing, and either reverts a
deliberate change or spends a session re-deriving why it is safe.

### I2 — The ETDRK4 blow-up check moved from per-step to per-call: undocumented in `DECISIONS.md`, untested, and it breaks numpy/torch return parity

`src/rngrn/etdrk4_torch.py:124` (`blew = not bool(torch.isfinite(v).all())`).

The boolean-equivalence argument is correct — every step operation is linear or an FFT over the
whole field, so a non-finite value cannot be erased once it appears, and `isfinite(v).all()`
after the loop yields the same flag. I checked the reaction closure for a path back to finite
(`clamp(X,0)**n`, `1 + einsum`, division) and found none: inf/inf → nan, and nan is absorbing.

What the docstring states as "what is lost" is understated in one respect it does not mention:
`eval/numerics.integrate_etdrk4_rfft` (numpy) still returns **the field at the first non-finite
step**, while the torch port now returns **the field after all `nsteps`**. So on a blow-up the
two backends return different arrays, which quietly weakens `relax_to_pattern_torch`'s standing
claim that the two backends "relax the same trajectory up to FFT-backend round-off".

`tests/test_etdrk4_torch.py` pins numpy equivalence at `delta <= 1e-12` (CPU) / `1e-9` (CUDA) —
on *non-blowing-up* trajectories, so it neither catches nor covers this. It was not updated. No
`DECISIONS.md` entry.

### I3 — `integrate_bdf1_newton_krylov` now raises where it silently fell back: correct, breaking, and unrecorded

`src/rngrn/eval/numerics.py:205`.

The change is right on the merits and the reasoning ("a run that actually executed ETDRK4 under
that label would be provenance that claims an integrator which never ran") is exactly §8. But it
is a **breaking behaviour change** landed inside a commit labelled `perf:`, with no
`DECISIONS.md` entry and no check that no tracked config, notebook, or `experiments/` frozen
config selects `integrator: bdf1_newton_krylov`. Anything that does now hard-crashes.

It also silently re-labels prior runs: every existing run recorded with
`integrator='bdf1_newton_krylov'` used ETDRK4, and this delta is the first place that is stated.
Under §10.4 that is a number whose meaning changed and it should be announced.

### I4 — D-PERF-3 flips a default for every N=3 run, with the suite unrun and self-declared not independently validated

`src/rngrn/model.py:147` (resolution), `src/rngrn/config.py` (`ModelConfig.dispersion_backend`),
`src/rngrn/recover.py` (`recover(dispersion_backend="auto")`), `docs/DECISIONS.md` (D-PERF-3).

The mechanism is correct — I verified it directly: `N=2 → eig`, `N=3 → cubic`, `N=4 → eig`,
explicit `cubic` at `N=4` raises `ValueError`, and `BatchedRNGRN.from_seeds` resolves to `cubic`
at N=3. `'auto'` never survives construction, so `.dispersion_backend` always reads concrete.
`io.save_checkpoint` writes the resolved value and `io.load_checkpoint` defaults old checkpoints
to `"eig"`, so no stored artefact changes meaning. **Fails loud correctly (review dimension 4:
satisfied).**

The problem is evidentiary, and D-PERF-3 states it against itself, which is to its credit:

> "the ~1e-12 agreement figure is the previously measured one; no fresh cubic-vs-eig full-run
> comparison was made in this session, and **the test suite had not yet been re-run** when this
> entry was written (owner deferred testing to the end of the branch)."

Two things it does not say:

* **Provenance regression.** `config.ModelConfig.dispersion_backend` now defaults to the string
  `"auto"`, so `frozen_config.yaml` records `"auto"` rather than the backend that ran. It is
  recoverable (read N from the same file), but `reporting-numbers.md` step 4 says to read the
  frozen config, not to re-derive from it. Writing the resolved value into the frozen config
  would close this.
* `_sigma_max_cubic`'s validation is "sigma_max MAE 9.2e-13" over 127 answer-key Jacobians —
  which is why D-PERF-3 correctly says runs are "not bit-comparable". Given this branch is
  proposed as the R3 base, that non-comparability propagates into every R3 number by default.

### I5 — `PatternSolver`'s docstring claims a deviation that does not happen on CPU

`src/rngrn/forward.py`, `PatternSolver` class docstring:

> "on CPU the algebra is the D1 port's, with two round-off level departures from it … **applied
> on BOTH devices**: |eig(J)|_max is evaluated by numpy rather than torch, and **the torch LSMR
> stops on a cadence (`_lsmr_torch`)**."

The second half is false. `newton_polish` (`forward.py:466`) selects `_minnorm_solve_t` /
`precon_t` only under `on_device = u.device.type != "cpu"`, and `solve_adjoint` branches the
same way; the CPU path routes through scipy's `_minnorm_solve` and never enters `_lsmr_torch`.
The first half (numpy `eigvals` for `jac_rate`) *is* both-devices and is correctly described.

§8 forbids a docstring claiming more than the code does. Here it over-claims a *deviation*
rather than a capability, which is the harmless direction — but it will mislead exactly the
reader trying to establish whether CPU results are still the D1 reference. They are.

### I6 — Performance numbers in the module docstring have no run behind them

`src/rngrn/forward.py:31-34`:

> "Measured split of a batched solve at 32²/B=3: relax 30 %, Newton 40 %, adjoint 30 %; at
> 64²/B=3 it is 10 / 75 / 15 — so the per-member loop, not the relax, is what a further
> optimisation has to attack."

and, immediately above, "see the unit's report and the docstrings below".

`.claude/rules/reporting-numbers.md` step 1: "Name the run directory under `experiments/`. …
A number with no run is not reportable." Neither figure names one, and per the coordinator no
report or SDD ledger survived that session, so "the unit's report" is a dangling reference.

The same paragraph draws a *design conclusion* from those numbers ("the per-member loop … is
what a further optimisation has to attack"), which is exactly the kind of steer R3 would inherit.
Either the run gets committed under `experiments/` or the sentence gets marked UNMEASURED.

Note by contrast that `_half_coeffs_batched`'s memory figures (58 MB at b=8/n=96, 230 MB at
b=32/n=96) are arithmetic from shapes and are checkable on their face — those are fine.

---

## 4. Minor

* **M1 — vectorised `_damped_newton` line search costs ~12 % on the CPU reference path.**
  `losses/terms.py:36-70`. I measured it: over 40 seeded N=3 models, the new form and a
  faithful reimplementation of the old loop produce **max |Δx\*| = 0.0** with **0
  convergence-flag disagreements** — the selection rule is preserved exactly, including the
  `0.5**30` fall-through. Cost, 20 repeats of 40 solves, `OMP_NUM_THREADS=1`: **new 0.1322 s vs
  old 0.1181 s per 40 solves**. The change is a GPU-sync win that is a small CPU loss on the
  default, reference path; the docstring does not mention the CPU cost.
* **M2 — `steady_state_batched`'s 5-halving cadence** (`losses/terms.py:605`) is
  value-preserving (an accepted member's `lam` is frozen by the `torch.where`, an inactive
  member's `x_new` is masked), but costs up to 4 extra batched reaction evaluations per Newton
  iteration on CPU. Same trade as M1, same silence about it.
* **M3 — the liveness cadence lets Adam keep stepping an all-dead batch.**
  `recover.py:236,258`. The early break can fire up to 24 steps late; during those steps
  `total` is an exact zero *with a graph*, so gradients are zero but Adam's momentum still
  moves the parameters. No reported number depends on it (`final_alive` is all-False, so every
  member logs `steady_state_failed`), but the inline comment's "harmless — those extra steps
  optimise an all-dead, all-masked batch and produce nothing" is slightly stronger than the
  truth: they do move parameters. Also, `verbose` at such a step computes
  `float(loss_vec[alive].mean())` over an empty selection → NaN.
* **M4 — `lbfgs_error` is added to serial `restart_log` rows only** (`recover.py:667,687`),
  never to the batched rows built in `_batched_restarts`. The two paths now emit different
  column sets into the run index. §4 wants flat scalars, which a `str | None` satisfies; the
  asymmetry is the issue.
* **M5 — `recover()`'s docstring says `'cubic' (exact for N<=3 ONLY)`** while
  `model.py:149` raises for `N != 3`. A caller reading the docstring would expect N=2 to accept
  `'cubic'`; it raises.
* **M6 — `spectral_block_torch`'s summary line contradicts its own body.** It says "normalise
  by total power" while the body (and its DEVIATION paragraph, and the new
  `spectral_block_torch_batched` docstring) normalise by the *central block's* own power. I
  verified the batched twin matches the serial body, so this is a wording defect, not an
  arithmetic one — but the batched twin now states the opposite of the serial summary in the
  same file.
* **M7 — the batched spectral path introduces new per-step device→host syncs** in a commit whose
  thesis is sync elimination: each `*_batched` term does `.detach().cpu().numpy()` on its (b,)
  value (`losses/spectral.py`, five terms plus per-channel parts), and
  `total_loss_batched` pulls every term again. The serial path paid the same syncs via
  `float(...)`, so this is parity rather than a regression — worth a look when the cost split
  in I6 is actually measured.
* **M8 — `_half_coeffs_batched`'s 230 MB contour intermediate has no guard.**
  `forward.py:998`. The docstring names the mitigation ("chunking the contour") and declines it.
  At b=32/n=96 that is a 230 MB complex128 allocation per batched solve, per Adam step, on a
  machine whose binding resource is RAM (§7a). No cap, no check, no `MemAvailable` interaction.
* **M9 — `active` is wired only on the spectral branch:**
  `active=alive if spec_cfg is not None else None` (`recover.py:254`, `:297`). It cannot change
  a value either way, so the conditional buys nothing and makes the batched non-spectral path
  take a different branch for no stated reason.
* **M10 — `spec_shape` still does `targets["band_idx"].to(power.device)` every step**
  (`losses/spectral.py`), which the new `build_frame_targets(device=)` plumbing was supposed to
  remove. It is a no-op transfer once the target is placed, but the per-step Python call remains.

---

## 5. Batched code paths with NO equivalence test

Every symbol below is new in this range and has zero references under `tests/`. Grouped by what
a completed test would have to assert.

**`src/rngrn/forward.py` — the whole batched forward solve:**

| Symbol | Line | What an equivalence test must assert |
|---|---|---|
| `batched_reaction_fields` | 906 | per-member equality with serial `reaction_fields`; gradient lands on row `idx[j]` only |
| `make_spatial_F_batched` | 930 | `F_b(u)[j] == make_spatial_F(member j)(u[j])`; differentiability through theta |
| `_member_F` | 950 | the single-member view equals the serial closure |
| `_batched_reaction_builder` | 967 | equals `etdrk4_torch._torch_reaction_builder` per member; nc1 refusal |
| `_half_coeffs_batched` | 998 | equals `torch_half_coeffs` per member for per-member `dt`/`D` |
| `_kstar_of_torch_batched` | 1023 | equals `observables.kstar_of_torch` per member (and see C3 — the serial twin is itself unpinned) |
| `relax_to_pattern_torch_batched` | 1054 | a member's snapshot equals the serial `relax_to_pattern_torch` field; per-member `blew_up`/`no_saturation` verdicts |
| `BatchedPatternSolve.backward` | 1148 | **the load-bearing one** — batched IFT gradient vs finite differences, and vs the serial `PatternSolve` per member; gradient scatter isolation across members |
| `BatchedPatternSolver.solve_subset` | 1291 | reason vocabulary parity ("ok"/"solve_failed"/"not_patterned"); warm-state keying and clearing; two-pass warm/fresh fall-through |

**`src/rngrn/losses/spectral.py` — batched twins** (`raps_torch_batched`,
`spectral_block_torch_batched`, `spec_shape_batched`, `spec_aniso_batched`,
`spec_amp_mean_batched`, `spec_amp_fluct_batched`, `real_moments_batched`,
`spectral_terms_batched`, `is_ignited_batched`). I measured these at ≤ 4.3e-14 relative vs
serial on CPU float64 — they are very likely correct, and the test is a half-hour of work that
was simply not done.

**`src/rngrn/losses/total.py:188 `_apply_spectral_batched`** — the per-member ignite-or-omit
contract: exact-0 loss placeholder vs NaN record, `spectral_skipped` reason strings,
`spec_computed`/`spec_ignited`, the `abandoned` vs `not_ignited` distinction.

**`dispersion_backend='auto'`** — partially covered. `tests/test_determinism.py:79` tests only
the explicit-`cubic`-at-wrong-N rejection. Nothing asserts what `'auto'` resolves to, on either
`RNGRN` or `BatchedRNGRN.from_seeds`. (I verified both by hand; see I4.)

---

## 6. Semantic-change table

| # | Change | Neutral or semantic | Documented where | `DECISIONS.md`? |
|---|---|---|---|---|
| 1 | One Jacobian per step for the 3 dispersion-side terms (`total.py:131`, `:321`; `J=` kwarg on 7 term fns) | **Neutral** — same tensor, three consumers; autograd accumulates the identical sum | inline comment + each term's `J=` docstring | not needed |
| 2 | `_damped_newton` vectorised 30-candidate line search (`terms.py:36`) | **Neutral** — verified bit-identical over 40 seeds, 0 flag disagreements | docstring (states the ulp caveat) | not needed |
| 3 | `steady_state_batched`: `active.any()` moved after the solve; singular-J folded unconditionally; 5-halving cadence (`terms.py:605`) | **Neutral** — frozen-`lam` and masked-`x` arguments both hold | inline comments | not needed |
| 4 | `_lsmr_torch` cadenced stop + branchless `_sym_ortho_t` + guarded Krylov breakdown (`forward.py:180,183,147`) | **SEMANTIC** — not iteration-for-iteration scipy; different degenerate branch | docstring + `_minnorm_solve_t` docstring | **MISSING** |
| 5 | ETDRK4 blow-up check per-step → per-call (`etdrk4_torch.py:124`) | **SEMANTIC** — different returned field on blow-up; loses the step index; breaks numpy/torch return parity | module + function docstring | **MISSING** |
| 6 | Relax saturation detector `obs.kstar_of` → `obs.kstar_of_torch` (`forward.py`, `:1023`) | **SEMANTIC** — `floor` vs `digitize` binning; can stop a chunk earlier/later | `relax_to_pattern_torch` + `observables.raps_torch` docstrings | **MISSING** |
| 7 | `dispersion_backend` default `eig` → `auto` (`model.py:147`, `config.py`, `recover.py`) | **SEMANTIC** — N=3 runs change backend; explicitly not bit-comparable | **D-PERF-3** + model/recover docstrings | **present** ✔ |
| 8 | Liveness sync cadence, 25 steps (`recover.py:236`) | **Semantic-adjacent** — `died_at` values exact; break up to 24 steps late; Adam momentum moves an all-dead batch | inline comment (claims "harmless"; see M3) | **MISSING** |
| 9 | `bdf1_newton_krylov` silent fallback → `NotImplementedError` (`numerics.py:205`) | **SEMANTIC / breaking** | function + module docstring | **MISSING** |
| 10 | LBFGS `except: pass` → recorded `lbfgs_error` (`recover.py:667`) | **SEMANTIC** — failures now surface; new `restart_log` key | inline comment | **MISSING** |
| 11 | Batched-spectral refusal **deleted** in `compute_terms_batched` / `total_loss_batched` / `recover` | **SEMANTIC / contract change** | docstrings ("deleted, not weakened") | **MISSING** — and contradicted by 3 live tests (C1) |
| 12 | Constant hoisting: `param_prior(box=)`, `obs_scale`, `kstar_idx`, `build_frame_targets(device=)`, `_RAPS_BINS`/`_RAPS_TORCH_BINS` caches, cached grid geometry in both solvers | **Neutral** | docstrings throughout | not needed |
| 13 | `turing_ok` / `rollout.simulate` / `lifted.simulate_lifted` eigvals loop → stacked `eigvals(M)` | **Neutral** | undocumented (trivially so) | not needed |
| 14 | `analysis.linear_stability` / `_model_JD` place x\* on the model's device | **Neutral on CPU, a FIX on CUDA** | undocumented | not needed |
| 15 | `io.load_checkpoint(map_location="cpu")` | **Neutral / fix** — CUDA-saved checkpoints load on CPU-only hosts | docstring | not needed |

**Summary: 8 semantic changes, 1 documented in `DECISIONS.md`.** §10 requires an entry with
date, evidence, and what was rejected for each. That gap is the single most systematic problem
in this delta after the missing tests.

---

## 7. Firewall (§5) — CLEAN

Re-read from **this worktree's** `tests/test_firewall.py`, not from `CLAUDE.md` (whose §5 list
is stale — it says 12 forbidden names; the file now has 27).

* `RECOVERY_SIDE` already includes `forward.py`, `etdrk4_torch.py` and `losses/spectral.py`
  (added 2026-08-12), so every module this delta touches heavily is inside the static AST audit.
  `test_every_loss_and_eval_module_is_classified` covers the new code by construction.
* New imports introduced in the range, all intra-package and side-neutral:
  `forward.py` ← `_phi_contour_torch`, `_spectral_k2_half`, `_reaction_raw`;
  `losses/total.py` ← `numpy`, `is_ignited_batched`, `spectral_terms_batched`;
  `recover.py` ← `forward.BatchedPatternSolver`, `spectral.SpectralContext`,
  `terms._load_box_bounds`. **No forbidden name appears anywhere in the diff.**
* **No new `scripts/` module** was added, so the known `scripts/`-glob blind spot is not
  triggered and `FORBIDDEN` needs no addition.
* Observed-data reads: `float(frame.mean())` (hoisted into `recover()`, still carrying its
  FIREWALL comment at both `compute_terms` sites) and `kstar_obs` from an FFT of the observed
  frame. Both were already legal recovery quantities; hoisting changes where they are computed,
  not what is read.
* `build_frame_targets(device=dev)` moves observed-frame-derived *targets* onto the training
  device. Targets, not truth — clean.
* No criterion used to judge recovery is derived from generating parameters anywhere in the
  delta.

---

## 8. Device / dtype discipline and fail-loud

**Good, with one gap.**

* **No silent CPU fallback anywhere.** Both `PatternSolver.__init__` and
  `BatchedPatternSolver.__init__` **raise** on a device mismatch with the model.
  `_batched_reaction_builder` refuses `form != "competitive"` loudly. `_lsmr_torch` raises on a
  non-finite `||x||` rather than returning a NaN solution. The adjoint residual tripwire is
  applied per member and names the offender.
* `torch.set_default_dtype(torch.float64)` is preserved (`losses/terms.py:31`).
* Host round-trips are deliberate, minimal and documented: `jac_rate` via numpy LAPACK (both
  solvers, correct per §7 — no batched cuSOLVER kernel for small non-symmetric eig);
  `_orthonormalise_modes` one stacked pull; `_lsmr_torch` one stacked pull per 25 iterations;
  the batched relax detector 3 scalars per 500-step chunk. I found no *undocumented* `.cpu()` in
  a hot path.
* Cache keys include `device` **and** `dtype` in both `_RAPS_BINS` and `_RAPS_TORCH_BINS`, so
  there is no cross-device tensor leakage. Both dicts are unbounded, which the docstrings
  address (a run touches one or two geometries).
* `torch` was dropped from `losses/weighting.py` imports — correctly, it was orphaned.

**The gap — input validation on the batched spectral entry point.** The deleted refusal
(C1) was the only validation `compute_terms_batched` performed on its `spectral` argument, and
nothing replaced it. Two consequences:

1. The controller's `AttributeError: 'NoneType' object has no attribute 'B'` is the test's own
   `model=None` placeholder, not a production defect — but it demonstrates that the argument now
   flows unchecked into the body.
2. The real exposure: `PatternSolver` exposes `.solve()` and `BatchedPatternSolver` exposes
   `.solve_subset()`, and neither has the other's method. Handing a serial solver to the batched
   assembler — a newly plausible mistake now that both combinations are legal — surfaces as
   `AttributeError: 'PatternSolver' object has no attribute 'solve_subset'` from inside
   `_apply_spectral_batched`, mid-step, after the batch has already paid for a steady-state
   solve and three Jacobians. §4 wants that refused loudly at the entry point. A two-line
   `hasattr(spectral.solver, "solve_subset")` check in `compute_terms_batched` restores what
   the deletion removed.

---

## 9. Possibly incomplete (truncation, not decision)

Per the coordinator: the session was cut mid-execution and left no plan, ledger or stash. Read
this list as "where the cut probably landed", with what the finished form was most likely meant
to be. Severity is unchanged by truncation — the fix changes, the verdict does not.

1. **The two failing tests (C1).** `tests/test_ignition_gating.py:158`, `:165`.
   *Completed form:* replaced, not deleted — `test_compute_terms_batched_accepts_a_batched_spectral_context`
   asserting the five keys appear per member with the exact-0 / NaN split, plus a retained
   refusal test asserting that a solver **without** `solve_subset` is rejected loudly (§8 gap).
2. **`test_recover_raises_on_batched_with_a_spectral_weight` (`:194`), now vacuously green.**
   *Completed form:* deleted and replaced by an end-to-end `batched=True, lbfgs_steps=0` spectral
   run against its serial twin. As it stands it is worse than a failing test — it is a false
   green that the C1 repair will not touch.
3. **Zero test files in a 1,773-insertion range.** The equivalence tests the redesign mandates
   (§5 above) are the obvious next commit and simply do not exist.
4. **`observables.raps_torch`'s two self-declared TODOs** (`observables.py:136`, `:139`) — "add a
   test before any further caller relies on it" and "switching to `digitize` … left as a flagged
   follow-up". Two callers were added in the same delta and neither TODO was closed.
   *Completed form:* the parity test, then the `digitize` switch (or an explicit decision entry
   accepting the divergence).
5. **Seven `DECISIONS.md` entries' worth of semantic change, one entry written.** D-PERF-3 is
   dated 2026-08-17 and is the *third* commit of five; the two later commits carry the three
   biggest undocumented semantic changes (#4, #5, #6 in the table). The pattern — decisions
   recorded up to `375cb4a`, none after — is itself a fair marker of where the session stopped.
   *Completed form:* D-PERF-4/5/6 for the LSMR cadence, the ETDRK4 blow-up check, and the relax
   detector swap; plus an entry for the bdf1 raise and the deleted batched-spectral refusal.
6. **`forward.py:31`'s cost split cites "the unit's report"** which does not exist, and names no
   run directory. *Completed form:* the measurement committed under `experiments/`, or the
   sentence marked UNMEASURED.
7. **`BatchedPatternSolver.warm_mode="newton"` is unreachable from the library** —
   `recover.py:254` always constructs with `warm_mode="relax"`. It mirrors the serial default,
   so this looks like faithful-port scaffolding rather than a defect, but it is untested dead
   code on the library path.
8. **`relax_to_pattern_torch_batched` passes `dt` to `integrate_etdrk4_rfft_torch`, which never
   reads it** (`forward.py:1101`, flagged in its own comment as "signature parity only"). Minor,
   but the kind of thing a finishing pass removes.
9. **`_half_coeffs_batched`'s named mitigation not implemented** (M8) — "chunking the contour
   would trade it back for kernel launches and **is not done here**" reads as deferred work.
10. **`active` wired on one branch only** (M9) and **`lbfgs_error` on one path only** (M4) — both
    are the asymmetries a finishing pass squares up.
11. **No `docs/HANDOFF_*.md`** for a substantial branch (§2). Given the coordinator's note that
    no record survived, this is the single most valuable missing artefact.

---

## 10. Collision points with R2's loss-term registry refactor

Listed, not judged. R2 independently refactored `losses/terms.py`, `losses/total.py` and
`recover.py` into a loss-term registry; every hunk below sits in the blast radius. §11's warning
applies with full force: git resolves at line granularity, and "keep both sides" is **wrong**
inside a function signature or a dict literal, where it silently yields duplicate definitions of
which only the last takes effect.

**`losses/total.py` — highest-density collision surface:**

1. `compute_terms:131-146` — the entire term-assembly block rewritten: hoisted
   `J = model.jacobian(x_disp, create_graph=True)` plus `J=`/`idx=` threaded into four term
   calls. A registry replaces exactly this block. **The single most conflict-prone hunk.**
2. `compute_terms_batched:321-336` — the same rewrite, mirrored.
3. `compute_terms` / `total_loss` signatures — `obs_scale`, `kstar_idx` appended.
4. `compute_terms_batched` / `total_loss_batched` signatures — `obs_scale`, `kstar_idx`,
   `active` appended. **Four signature-level merges in one file** is the precise shape of the
   phase-A duplicate-keyword failure.
5. `_apply_spectral_batched:188-253` — a new 66-line function inserted immediately above
   `compute_terms_batched`.
6. `total_loss_batched`'s `L_<key>` NaN-masking loop — newly couples the parts-writing loop to
   `SPECTRAL_TERM_KEYS` and `parts["spec_computed"]`.
7. `parts_member` — new dtype-`kind` branch for object/string arrays.
8. The deleted `spectral is not None` refusal in `compute_terms_batched` — a **deletion**, which
   merges badly against any edit to the same block and can be silently resurrected.

**`losses/terms.py`:**

9. `J=None` added to six term functions (`turing_hinges`, `turing_hinges_split`, `anticollapse`,
   `kstar_anchor`, `turing_hinges_batched`, `turing_hinges_split_batched`,
   `anticollapse_batched`, `kstar_anchor_batched`) — a registry standardising term signatures to
   `(model, ctx) -> (val, parts)` collides with every one.
10. `idx=None` on `_sigma_at`, `kstar_anchor`, `kstar_anchor_batched`.
11. `box=` promoted over `box_path=` in `param_prior` (and the matching call-site change in
    `recover.py`).
12. `_damped_newton:36-70` — body fully rewritten (`_LINE_SEARCH_HALVINGS` module constant added
    above it).
13. `steady_state_batched._run:590-610` — inner loop restructured (break relocated, cadence
    added, singular-J handling inlined).

**`losses/spectral.py`:**

14. `_RAPS_BINS` cache + `_raps_bins` inserted above `raps_torch`; `raps_torch` body rewritten.
15. `build_frame_targets` — new `device=` parameter and a **new `band_idx` key in the returned
    targets dict**. Any R2 change to that dict literal collides at the dict level (§4/§11: a
    duplicate key silently keeps the last).
16. `spec_shape` — band selection rewritten from boolean mask to `index_select`.
17. A new ~150-line "BATCHED twins" section between `is_ignited` and `SpectralContext`.
18. `SpectralContext`'s docstring fully rewritten (the duck-typed interface is now
    assembler-dependent).

**`recover.py`:**

19. `param_prior_kw` construction moved out of the `term_kw` literal and reshaped
    (`box_path` → `box`) — a dict-literal edit, §11's named failure mode.
20. `term_kw` gains `obs_scale` and `kstar_idx` by post-hoc assignment.
21. The `use_spectral and batched` refusal block **deleted**.
22. `build_frame_targets(...)` call gains `device=dev`.
23. `_batched_restarts` signature gains `spec_cfg`, `spec_targets`; its call site gains both.
24. `_batched_restarts`'s liveness block rewritten (`died_at_step`, `LIVENESS_SYNC_EVERY`); two
    `total_loss_batched` call sites gain `active=`.
25. Serial loop: `lbfgs_error` variable and a new `restart_log` key.

**`model.py` / `config.py`:** 26. the `dispersion_backend` default string in three places
(`RNGRN.__init__`, `BatchedRNGRN.from_seeds`, `ModelConfig`) plus the resolution block at
`model.py:147` — low-conflict, but if R2 also touched `ModelConfig` this is a
**field-default** merge, the class that silently reverted `resid` in phase A.

---

## 11. Commit-by-commit completeness

| Commit | Assessment |
|---|---|
| `346bbf2` perf: desync losses hot path; one Jacobian per step; hoist per-step constants | **Complete.** All changes arithmetic-neutral (I verified #1–#3 in the table, and measured #2 bit-identical). Docstrings updated at every site. No decision entry needed. |
| `e7d01bd` perf+fix: 'auto' backend; liveness cadence; LBFGS errors recorded | **Mostly complete.** Mechanism verified correct and fail-loud. Missing: a test for `'auto'` resolution; `DECISIONS.md` entries for the liveness cadence and the LBFGS-error change (only the backend got one, in the next commit). |
| `375cb4a` docs: D-PERF-3 | **Complete**, and unusually good practice — it states its own non-validation and announces non-comparability. |
| `8f93dfe` perf: desync the forward-solve path; cache grid constants; device-resident spectral geometry | **Cut short on the evidence side.** The code is careful and the reasoning holds, but it lands three training-time semantic changes (LSMR cadence, ETDRK4 blow-up check, relax detector swap) with docstrings only — no `DECISIONS.md`, no test updates, and it puts two new callers on a port its own docstring says not to rely on yet (C3). |
| `2b11010` feat: batched PatternSolver | **Clearly cut short.** ~1,000 new lines, zero tests, two pre-existing tests left failing and a third left vacuously green, a deleted contract with no replacement validation, and a module docstring citing a cost split and a report that do not exist. |

**Minimum to reach "Needs fixes" rather than "Unsafe":** repair C1 (fix the two failures *and*
the vacuous third test), add the `BatchedPatternSolve.backward` gradient equivalence test (C2's
load-bearing item), and pin `observables.raps_torch` (C3). The remaining Importants are
documentation debt that can be paid alongside.
