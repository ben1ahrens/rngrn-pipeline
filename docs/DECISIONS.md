# DECISIONS.md — the science decision register

**Status: information, not instruction.** This is an assembly, not new analysis. Every
entry below is traceable to something already in the repo (a merged PR body, a
docstring, a `[TUNE]`/`[IMPL]`/`[VALIDATE]` marker, or another doc) — nothing here was
invented or re-decided while writing this file. Where a source itself records an open
tension, that tension is preserved, not resolved.

This project is headed for a paper. The point of this register is that every number
in a future write-up can be traced back to the decision that shaped it, who made that
decision, and what evidence it was checked against.

**Format per entry:** ID · date · status (DECIDED / OPEN / SUPERSEDED) · the decision
in one sentence · who decided it · the evidence · what was rejected and why · where it
lives in code.

**Status legend**
- **DECIDED** — a choice has been made and is live in the code/config as described.
- **OPEN** — laid out, not yet chosen; picking it would bias what later numbers mean.
- **SUPERSEDED** — a later decision reversed this one; kept for the record because
  numbers measured under the old decision are not comparable to numbers measured
  after it.

---

## Part 1 — Evidence-integrity defects (read this before trusting any pre-2026-07-29 number)

These are not science decisions; they are defects in how evidence was recorded. They
are listed first, prominently, because they change how earlier numbers should be read.

### D-EVID-1 — `cli.py -o` silently kept only the last override

**Date found:** 2026-07-29 (unit 10, PR 10). **Fixed:** 2026-07-29 (merge `e662a15`).
**Status:** SUPERSEDED (defect; now fixed).

`sp.add_argument("-o", "--override", nargs="*", ...)` without `action="extend"` meant
repeated `-o key1=val1 -o key2=val2` flags on the CLI **replaced** `args.override` on
each occurrence instead of accumulating — so only the last `-o` took effect. Any prior
run invoked with multiple `-o` flags silently ran with the config's *default* values
for every override before the last one, not the budget the command line actually
asked for.

**Found by:** unit 10, while trying to reproduce a stage-2 e2e run
(`docs/CODE_REALITY.md` §"NOT fixed" list on PR 10). Confirmed by comparing
restart/step progress logs against the actually-requested override values.

**Fix:** `src/rngrn/cli.py:149` — `nargs="*", action="extend"`, with a comment
recording why it is load-bearing.

**Consequence for earlier results:** any run whose invocation used more than one `-o`
flag (rather than one `-o` with space-separated `key=val key=val`) needs its actual
effective config checked against `frozen_config.yaml`/`train_results.json`, not
assumed from the command line that launched it.

### D-EVID-2 — the pre-push hook printed "skipping tests" and exited 0 for all 13 phase-A units

**Date found / fixed:** 2026-07-29 (merge `e662a15`, "Integrate the 13-unit wave:
repair merge damage and close three wiring gaps"). **Status:** SUPERSEDED (defect;
now fixed).

The pre-push hook (`.githooks/pre-push`), meant to be the authoritative local test
gate, silently took the "no python interpreter found — skipping tests" branch and
exited 0 — every one of the 13 phase-A units pushed with a **green pre-push that ran
no tests at all**. PR 11's own body independently reports the same symptom
(`git push printed "pre-push: no python interpreter found; skipping tests."`),
corroborating this from the unit-11 side.

**Fix:** `.githooks/pre-push:31` now `echo "pre-push: no python interpreter found —
REFUSING to push untested." >&2` and fails the push instead of skipping silently. The
fix's own comment (`.githooks/pre-push:19`) records the defect it replaces almost
verbatim.

**Consequence for earlier results:** none of the 13 phase-A PRs' "tests pass" claims
were enforced by the push gate — each PR's reported pytest counts were produced by
manually running `pytest -q` inside the session (visible in each PR body above), so
the counts themselves are still evidence, but nothing mechanically prevented an
untested push from having landed. `main` @ `4509632` is the first commit with the real
gate; anything before it relied entirely on the acting agent's self-report.

**Where it lives:** `.githooks/pre-push`, `docs/CODE_REALITY.md` §8 ("The
authoritative test run is local").

### D-EVID-3 — `rngrn evaluate` re-ran every archived model at the WRONG domain size

**Date found / fixed:** 2026-07-30 (cross-domain-size evaluation unit). **Status:**
SUPERSEDED (defect; now fixed).

`cli.py::cmd_evaluate` called `simulate(model, L=cfg.data.L, ...)`. `cfg.data.L` is a
**generator** parameter that file-backed configs deliberately do not set — `config.py`
`DataConfig.L` says so in its own comment ("Do not add `L:` to a file-backed config
expecting it to apply"), because each registry sample carries its own `L`. So for every
registry config `cmd_evaluate` used the `base.yaml` default **100.0** while the model had
been recovered on the sample's real domain size. Re-running an archived model therefore
reproduced nothing: k\*, the pattern and the morphology were all computed on a box of the
wrong size.

**Measured, on the one archived run in this worktree carrying both files**
(`m3_registry_20260729_221743`, `three_gene_qvar/sample_0000`):
`config/frozen_config.yaml` → `data.L = 100.0`; `checkpoints/model.pt` → `L =
78.01357861389891`. A **28.2 % error in the length scale**, silent.

**Fix:** `cmd_evaluate` now defaults `L` to the **checkpoint's stored `L`**
(`io.save_checkpoint` has recorded it since unit 12) and warns at `RuntimeWarning` when
`cfg.data.L` disagrees. It also routes the model through
`eval.lgen_eval.physical_model_from_checkpoint` first, so a non-dimensional checkpoint's
`theta_D` is converted before anything is integrated (D16).

**Consequence for earlier results:** no number in the run index came from `cmd_evaluate` —
the `rollout_*` columns in `runs.jsonl` are written by `train._morphology_rollout`, which
has always used the target sample's own `ri.L` and is unaffected. The damage is limited to
any `rngrn evaluate` output quoted by hand from a terminal; such a figure is not
reproducible and must be re-measured.

**Where it lives:** `src/rngrn/cli.py` (`cmd_evaluate`).

### D-EVID-4 — `train.seed` does not change what recovery does: every "seed replicate" is the SAME draw

**Date found:** 2026-07-30 (unit P1, while validating the new `seed` / `model_seed` run-index
columns). **Status:** **OPEN — FOUND, NOT FIXED.** Reported rather than repaired because
changing the seed plumbing would change what every previously recorded "seed replicate"
number means, and that is the owner's call, not the implementing agent's.

**What was found.** `train.fit` calls
`R.recover(..., seed=cfg.train.seed, model_seed=cfg.model.seed)`. Inside `recover`,
`model_seed = seed if model_seed is None else model_seed` — and `cfg.model.seed` is `0`, never
None, so **`model_seed` is 0 on every run regardless of `train.seed`**. Restart inits come
from `_restart_seed(model_seed, r)` and a per-restart `torch.Generator`, never from the global
RNG that `seed_everything(cfg.train.seed)` seeds. `train.seed` therefore reaches only the
post-training rollout's noise field (`eval.rollout.simulate(seed=...)`), not the optimisation.

**The measurement.** A 12-run sweep over `train.adam_steps` ∈ {60, 120, 240} ×
`model.form` ∈ {competitive, nc1} × `train.seed` ∈ {0, 1} on
`three_gene_val/sample_0000`: the six pairs differing **only** in `train.seed` have
**bit-identical** loss (e.g. competitive/60: 0.223712 and 0.223712; nc1/240: 0.073764 and
0.073764) and identical `kstar_fft_rel_err`. The defect was invisible before unit P1 because
neither `seed` nor `model_seed` was on the run row.

**What it plausibly affects, stated as a question and not as a conclusion.**
`optim/target_report.py` builds its K per-seed jobs with `apply_overrides(..., ["train.seed=<seed>"])`
and nothing else, so on the reading above a K-seed target report runs K identical recoveries
and its cross-seed reproducibility statistics would be measuring nothing. **This has not been
verified end to end here** — it follows from the code path and from the sweep evidence above,
but no target report was re-run to confirm it, and any existing reproducibility number should
be re-checked against `model_seed` before it is quoted.

**Where it lives:** `src/rngrn/train.py` (the `model_seed=cfg.model.seed` argument);
`src/rngrn/recover.py` (`_restart_seed`, the `model_seed` default);
`src/rngrn/optim/target_report.py` (`run_target_report`'s per-seed overrides). The new
`seed` / `model_seed` columns on the run row are what make it visible.

### D-EVID-7 — the k\* leak control was measured against the SECONDARY reference, so the headline column shipped with no control

**Date found / fixed:** 2026-08-04 (multi-agent branch review). **Status:** SUPERSEDED
(defect; now fixed). **Decided by:** the implementing agent under delegated authority (§10).

`validate.score_recovery` reports two k\* errors against two different references:
`kstar_fft_rel_err` (**headline**, vs `answer_key.kstar_fft`) and `kstar_rel_err`
(secondary, vs `answer_key.kstar`). `_leak_instrumentation` was called as
`_leak_instrumentation(L, out["kstar_true"])` — the **linear** reference only — so the
image-blind control `trivial_kstar_err` and the resolution floor `kstar_fft_bin_width` were
both normalised by `kstar`, and **neither controlled the column the claim is gated on**.

**Why it matters.** The legacy generators set `L = clip(6*2*pi/k*, 18, 220)`, so the
image-blind predictor `k_trivial = 6*2*pi/L` is exact against `kstar` (~1e-16) while
`kstar_fft` differs from `kstar` by a measured median of **8.4%** (`data/gate.py:59`). A row
reading `kstar_fft_rel_err = 0.046` beside `trivial_kstar_err = 0.000` therefore looked like
a 14-order-of-magnitude win over a baseline that had **never been measured on that column**.

**Evidence — this is present in the tracked ledger, not hypothetical.**
`experiments/stage0_bioviab/runs.jsonl`, `stage0_prior_off_0004`: `trivial_kstar_err =
0.000` (periods-per-box exactly 6) next to `kstar_fft_rel_err = 0.0458`. The honest control
for that row is ~4–8%, i.e. the same order as the result.

**Fix:** `_leak_instrumentation(L, kstar_true, kstar_fft_true)` now emits four columns
paired by reference — `trivial_kstar_err` / `kstar_fft_bin_width` (vs `kstar`) and
**`trivial_kstar_fft_err` / `kstar_fft_bin_width_rel_fft`** (vs `kstar_fft`). Each is NaN
when its own reference is absent, never a fallback to the other. `trivial_kstar_err` keeps
its exact former meaning, so rows recorded before this change stay comparable. The pairing
is propagated to `optim/benchmark.py` (`COLUMNS`, `DEGRADATION_COLUMNS`, both table
builders) and `optim/target_report.py` (`_pattern_block`, the per-seed rows). Five tests in
`tests/test_gate_contract.py` pin it, including that the new column does **not** fall back
to the linear reference.

**What was rejected and why:** silently renormalising the existing `trivial_kstar_err` onto
`kstar_fft`. Rejected because it would change the meaning of a column already present on
every tracked row, making old and new rows non-comparable without saying so — exactly the
failure §10.4 forbids. Adding a second column keeps both readable.

**Consequence for earlier results:** every k\* claim read against `trivial_kstar_err`
overstates the margin over the image-blind baseline wherever the leak relation happens to
hold. Re-read them against `trivial_kstar_fft_err`.

**Corrected 2026-08-04 — an earlier draft of this paragraph said non-legacy datasets are
"unaffected in kind", and that was wrong in a way that contradicted this entry's own
evidence.** `scripts/gen_tg3.py` draws periods-per-box from 3..14 rather than fixing it at
6, so the leak relation does not hold *by construction* there — but a free draw can still
land on 6, and it did: the evidence row cited above, `stage0_prior_off_0004`, is
`three_gene_qvar` and carries `trivial_kstar_err` of exactly 0.000. All 13 tracked rows are
`three_gene_qvar` / `three_gene_multiL`, and **every one of them was mis-paired**. The
mitigation on the new generators is statistical, not categorical, and must not be stated as
categorical.

**Where it lives:** `src/rngrn/validate.py::_leak_instrumentation`;
`src/rngrn/optim/benchmark.py`; `src/rngrn/optim/target_report.py`.

### D-EVID-8 — the pre-registered 8.3 % k\* bar is HALF AN FFT BIN, which is not a constant across datasets

**Date:** 2026-08-04. **Status:** OPEN — the resolution below is the owner's, not an
agent's, because it bears on a pre-registered pass condition (§10).

**Not circular, which was the first thing checked.** `PREREGISTRATION.md` §3.3 derives
8.3 % as *half an FFT bin*: one bin is 16.7 % of k\*, so ±8.33 % is the estimator's own
resolution floor. That derivation is independent of the measured median
`|kstar_fft/kstar − 1| = 0.084`. The near-coincidence of 0.083 and 0.084 is **not**
circularity — both are ~half a bin, because half-a-bin *is* what FFT quantisation error
looks like.

**But the consequence is real:** the bar sits exactly at the noise floor, so "passing" means
"indistinguishable from the FFT's own quantisation". That is the same fact as D-EVID-7 — it
is *why* an image-blind predictor can pass it.

**And the floor is dataset-dependent.** One bin as a fraction of k\* is `1/(periods per
box)`. The legacy generators fix that at 6 (hence 16.7 %). `scripts/gen_tg3.py` draws it
freely from 3..14. Measured over the 13 tracked run rows, `kstar_fft_bin_width` ranges
**0.100 → 0.333**, i.e. half-bin floors of **5.0 % → 16.7 %**. On **7 of 13 rows the
pre-registered 8.3 % bar is BELOW the estimator's own resolution** — it demands precision
the measurement cannot deliver. *(Corrected 2026-08-04: this said "6 of 13". The measured
floors are 5.00, 5.00, 6.25, 6.25, 7.14, 7.14, 8.33, 8.33, 12.50, 12.50, 16.67, 16.67,
16.67 %; seven exceed 8.3 %, counting the two 8.33 % rows, which sit at the bar to within
rounding. Re-measure rather than trusting the tally.)*

**What this does not do:** it does not weaken or amend §3.3. Only the owner may do that.

**Options, for the owner:** (a) keep 8.3 % as a fixed bar and restrict §3.3 claims to
periods-per-box ≈ 6 samples; (b) make the bar per-sample at
`kstar_fft_bin_width_rel_fft / 2`, which is what the prereg's own calibration argument
implies; (c) keep both and report them side by side. Note §1 already excludes legacy data
from k\* claims, so the honest answer may be that nothing intended for publication changes.

**Where it lives:** `docs/PREREGISTRATION.md` §3.3; the new
`kstar_fft_bin_width_rel_fft` column (`src/rngrn/validate.py`).

### D-EVID-9 — the firewall audit was a hand-copied name allowlist, so new modules were unaudited by default

**Date found / fixed:** 2026-08-04. **Status:** SUPERSEDED (defect; now fixed).
**Decided by:** the implementing agent under delegated authority.

`RECOVERY_SIDE` was re-typed verbatim in **four** test files (`test_firewall.py:19`,
`test_plot_arrays.py:319`, `test_permutation_scoring.py`, `test_morphology_scoring.py`) —
the same eleven module names each time, deliberately mirrored "so the audits stay
independent". The cost of that independence was completeness: a module on **no** list is
audited by **nobody**, and the suite stays green.

Two such modules existed. `eval/lgen_eval.py` (564 lines, added this branch) was on none of
them. `history.py` was on none of them **and** did
`from .scoring.plausibility import d_ratio_of` while running inside the Adam loop
(`recover.py:245`, `:492`) — recovery-side code reaching into the scoring package, whose
siblings `overparam.py` and `permutation.py` do read the answer key.

**Not a truth leak.** `d_ratio_of` is a pure function of the model's own `D`. The defect is
that nothing would have caught it if it had not been.

**Fix (three parts):**
1. `test_every_loss_and_eval_module_is_classified` — every module under `losses/` and
   `eval/`, plus `history.py`, must be declared `RECOVERY_SIDE` **or** `SCORING_SIDE`.
   Membership of neither fails. This forces the classification at the moment a module is
   added. Verified to fire by adding a throwaway module and watching it fail.
2. `test_recovery_side_does_not_import_the_scoring_package` — the package-level rule
   `CLAUDE.md` §5 states, enforced in the firewall test itself rather than only in three
   scorers' own files. `history.py` added to `RECOVERY_SIDE`.
3. `d_ratio_of` / `_to_numpy` **moved** to the side-neutral `rngrn/utils.py`, re-exported
   from `scoring/plausibility.py` so every existing caller is unchanged.
   `FORBIDDEN` gained `data.gate` and `data.registry` — `gate.from_registry` returns the
   full `(RecoveryInput, AnswerKey)` pair, so importing it was a two-line route to ground
   truth that passed every previous version of the audit. The names must stay **dotted**:
   a bare `registry` false-positives on `rngrn/registry.py`, the component registry that
   `model.py` and `losses/weighting.py` legitimately import.

**What was rejected and why:** copying `d_ratio_of` into `history.py` (the obvious local
fix). Rejected because the definition is a recorded science decision (D2, largest /
second-largest, chosen so a near-immobile node never enters the ratio) and two copies would
drift. Also rejected: sharing one `RECOVERY_SIDE` constant across all four test files —
the mirroring is deliberate and keeps the audits independent; completeness is what was
missing, not deduplication.

**Consequence for earlier results:** none. No truth reached recovery; this closes the route,
it does not correct a number.

**Where it lives:** `tests/test_firewall.py`; `src/rngrn/utils.py`;
`src/rngrn/scoring/plausibility.py`; `src/rngrn/history.py`.

### D-EVID-10 — ledger sweep: which recorded rows the 2026-08-04 review affects

**Date:** 2026-08-04. **Status:** DECIDED (a measurement, recorded so the next fix does not
have to redo it). **Decided by:** the implementing agent.

Before changing anything, all **13 tracked run rows** (`experiments/audit_check`,
`experiments/lgen_transfer`, `experiments/stage0_bioviab`) were swept for the columns the
review's six headline defects touch. Result:

| defect | affects recorded rows? | evidence |
|---|---|---|
| M1 leak control (D-EVID-7) | **YES** | `trivial_kstar_err` on all 13; the 6-periods case is live on `stage0_prior_off_0004` |
| M2 `turing_ok` uses `tr(J)<0` | **YES** | `recovered_turing` / `recovered_sig_max` on all 13 |
| M3 sign-rtol sweep is a no-op | no | every row has `repro_sign_zero_rtol = 0.05`; nothing was ever run at 0.02 / 0.10 |
| M4 cross-seed grouping dead | no *reported* number | max group size is 1 on every index, so no aggregate has been published yet |
| M5 `morphology_match` species mismatch | **no** | `observed_idx = [0,1,2]` on all 13, so `observed_idx[0] == 0` and the comparison was correct |
| M6 nondim `D` misread | **no** | `nondim = False` on all 13; no committed config sets it |

**Consequence:** M5 and M6 can be fixed with **zero** comparability impact, and M3/M4 have
produced no wrong published number yet. Only M1 and M2 touch recorded values, and M2 is the
larger of the two because it would move `recovered_turing` itself.

**Where it lives:** `experiments/*/runs.jsonl` (tracked).

### D-EVID-11 — `turing_ok` tested `tr(J) < 0`, so uniformly UNSTABLE systems were scored Turing-unstable

**Date found / fixed:** 2026-08-04 (multi-agent branch review, M2). **Status:** SUPERSEDED
(defect; now fixed). **Decided by:** the implementing agent under delegated authority (§10).

**The defect.** `eval/analysis.turing_ok` evaluated uniform stability as `tr(J) < 0`. The
trace is the **sum** of the eigenvalues, so it can be negative while one eigenvalue is
positive — i.e. while the homogeneous state is unstable. Compounding it, the default scan
grid began at `k = 1e-3`, where `sigma(k)` is still essentially `sigma(0)`, so the *same*
uniform instability also satisfied the "structurally unstable" half. Both conditions could
therefore be met by a system that merely blows up uniformly, with `kstar` reported as the
grid floor.

**Reproduced exactly:** `J = diag(0.5, -1, -1)`, `D = (1, 10, 20)`. `tr(J) = -1.5 < 0`;
`max Re eig(J) = +0.5 > 0`. Old `turing_ok` → `ok=True`, `kstar=0.001`, `sig_max=0.5`.

**The correct criterion already existed in the same file.** `_perturb_cloud` has always
used `np.linalg.eigvals(Jp).real.max() < 0` and taken the structured max over `sig[:, 1:]`,
excluding `k=0`. So `recovered_turing` and the `turing_volume_*` columns — both on every run
row — were answering different questions. `scripts/exp11_robustness_baseline.py:23` recorded
the discrepancy in prose and it was never fixed in the library.

**Fix:** `turing_ok` now computes `sigma(0) = max Re eig(J)` exactly (independent of the
caller's grid), requires `sigma(0) < 0`, and takes the structured max over `k > 0` only. It
raises if the supplied grid has no `k > 0`. The superseded verdict is returned alongside as
`turing_loose` / `stable_uniform_loose`, and surfaced on the run row as
`recovered_turing_loose`, so rows recorded either side of the change stay readable against
each other — the convention `_perturb_cloud` already used with `frac_loose`. A new
`recovered_sig0` column exposes the uniform mode directly.

**Consequence for earlier results — MEASURED, and smaller than feared.** All 12 tracked run
records that store a Jacobian were re-scored under both criteria: **0 of 12 change verdict.**
No published recovery number moves. The correction bites at **initialisation**, not on
recoveries, because a trained model that reaches Turing does so genuinely.

**Consequence for the low-basal init — this is the real casualty.** Re-measured at N=3 over
400 seeds:

| init | converged | any positive J diagonal | STRICT Turing | loose (superseded) |
|---|---|---|---|---|
| `default` | 400 | 0/400 = 0.000 | **0/400 = 0.000** | 0/400 |
| `low_basal` | 398 | 114/398 = **0.286** | **0/398 = 0.000** | 206/398 = 0.518 |

The recorded "0 % → 82 % Turing-unstable at init" (D9, `STATE_OF_THE_SCIENCE.md` §10,
`model.py`, `exp03_turing_first.py`) is **withdrawn**. Note the loose figure is 51.8 %, not
82 %, even before the criterion change: D9's 80.8 % was `206/255 converged`, and unit B3's
steady-state multistart since raised convergence to 398/400 while the loose-Turing numerator
stayed at **exactly 206**. So the number moved twice, for two independent reasons.

**What survives:** low_basal produces the positive Jacobian diagonal Turing requires
(28.6 % vs 0 %). Necessary, not sufficient. `tests/test_science.py::
test_low_basal_init_gains_the_positive_diagonal_but_NOT_turing_reachability` pins both the
effect and the absence of the claimed one, and fails loudly if the loose criterion returns.

**What was rejected and why:** (a) renaming `recovered_turing` to `recovered_turing_loose`
and introducing a new strict column — rejected because the plain name should carry the
correct verdict, and because the ledger sweep showed no recorded row changes, so there is
nothing to protect by renaming; (b) leaving `turing_ok` loose and adding a separate strict
column — rejected because two live definitions of "Turing" in one library is what caused
this; (c) changing the default k-grid — deliberately **not** done, so `kstar` values stay
comparable. That grid (`linspace(1e-3, 50, 4000)`) is an absolute band in rad/length and is
now marked `[UNCALIBRATED]` at `analysis.py::_TURING_KGRID`: it is the same class of defect
unit B4 removed from `recover._kgrid_for`, currently wide enough for the registered L range
(18..220 → k\* 0.17..2.09) but not scale-free and untested at its boundary.

**Where it lives:** `src/rngrn/eval/analysis.py::turing_ok`, `::_TURING_KGRID`;
`src/rngrn/validate.py` (`recovered_turing`, `recovered_turing_loose`, `recovered_sig0`);
`tests/test_science.py`.

### D-EVID-12 — the pre-registered sign-threshold sweep was a no-op, and reported the wrong threshold's answer under the requested threshold's label

**Date found / fixed:** 2026-08-04 (multi-agent branch review, M3). **Status:** SUPERSEDED
(defect; now fixed). **Decided by:** the implementing agent under delegated authority (§10).

**The defect, in one line:** the threshold was applied once at score time and the
information needed to apply a different one was thrown away at that moment, so every later
"re-threshold" silently returned the score-time answer.

`validate.score_recovery` calls `REPRO.per_run_fields(...)` without a `sign_zero_rtol`, so J
is collapsed to {-1, 0, +1} at `DEFAULT_SIGN_ZERO_RTOL = 0.05` and only the collapsed
`repro_sign_vector` was stored. Every downstream consumer re-applied `sign_structure` to
that already-collapsed vector — and `sign_structure` is a **no-op** on such a matrix for any
`rtol < 1`, because the entries are 0 or ±1 and the scale is 1, so `|arr| > rtol * scale`
never changes an outcome. The module docstring even relied on this idempotence as a feature.

So `aggregate_target_report(..., sign_zero_rtol=0.02)` computed at **0.05** and wrote
`reproducibility_sign_zero_rtol: 0.02`.

**Why it matters.** `docs/PREREGISTRATION.md` §3.1 commits, in writing, to reporting
`topology_consistency` at **0.02 / 0.05 / 0.10** "so the conclusion's sensitivity to that
choice is visible rather than hidden". The shipped path returned **three identical numbers
under three different labels** — the sensitivity was not merely hidden, it was affirmatively
misreported. Both `DEFAULT_SIGN_ZERO_RTOL = 0.05` and the modal-fraction definition were
flagged UNCALIBRATED by unit 3, which is precisely why the sweep was promised.

**Reproduced.** Two seeds differing only in one edge at 7 % of max|J|:

| sign_zero_rtol | from the raw Jacobians | through the shipped path |
|---|---|---|
| 0.02 | 0.500 | 0.500 |
| 0.05 | 0.500 | 0.500 |
| 0.10 | **1.000** | 0.500 ← labelled 0.10 |

**Fix, three parts:**
1. `per_run_fields` now also stores **`repro_J_vector`** — the RAW Jacobian, row-major JSON,
   N*N floats (9 at N=3), a flat scalar for run-index purposes. This is the actual repair:
   the collapse is no longer lossy.
2. `_sign_matrix_from_metric(metric, sign_zero_rtol)` applies the **requested** threshold to
   that raw J. When a row carries no `repro_J_vector` (recorded before this change) it may
   still be used at **exactly** its recorded rtol, and otherwise **raises** with a message
   naming both thresholds. Answering the wrong question quietly is now impossible.
3. The report emits the three pre-registered cells from **one** run —
   `topology_consistency_rtol_0p02` / `_0p05` / `_0p10` — because the threshold is applied
   post hoc, so re-running `target-report` per threshold would burn K trainings to recompute
   a number that only needs the stored Jacobians. A cell that the rows cannot answer is
   **NaN**, never a repeated number. Defaults are seeded into the block's `base` dict so
   every early return (`insufficient_seeds`, `error`) emits the same schema.

**Consequence for earlier results:** none — no sweep was ever run. `docs/DECISIONS.md`
D-EVID-10 measured that all 13 tracked run rows carry `repro_sign_zero_rtol = 0.05`, i.e.
nothing was ever computed at 0.02 or 0.10, so no published number is wrong. What was missing
is a pre-registered deliverable, not a correction. **Rows recorded before this change cannot
be swept retroactively** — they carry no raw J — so the §3.1 sweep requires re-running the
targets it will be reported on.

**What was rejected and why:** (a) the interim guard alone (raise when
`sign_zero_rtol != metric["repro_sign_zero_rtol"]`) — it makes the lie impossible but leaves
the pre-registered sweep impossible too, and §3.1 is binding; (b) re-deriving J from the
tracked checkpoints at aggregation time — the checkpoint holds parameters, not the Jacobian
at the recovered steady state, so it would mean re-solving Newton per seed inside a
reporting function; (c) storing the raw J only in `arrays/*.npz` — the aggregation reads the
run index, and reaching into per-run array files from the reporting path couples them
needlessly.

**Still open, and NOT decided here:** `optim/benchmark.py::_row_to_sign_matrix` has the same
lossy shape. It feeds `reproducibility_table`, which the same review confirmed has **zero
callers repo-wide and no CLI flag**. Whether that table is wired up or deleted is
housekeeping for the owner; fixing dead code was not worth the churn, but it must not be
wired up as-is.

**Where it lives:** `src/rngrn/scoring/reproducibility.py::per_run_fields`;
`src/rngrn/optim/target_report.py` (`_sign_matrix_from_metric`, `_sensitivity_cells`,
`PREREGISTERED_SIGN_ZERO_RTOLS`); `tests/test_reproducibility_scoring.py`;
`tests/test_target_report.py`.

### D-EVID-13 — every cross-seed aggregation was dead: `build_table` grouped on `config_id`, which hashes the seed

**Date found / fixed:** 2026-08-04 (multi-agent branch review, M4). **Status:** SUPERSEDED
(defect; now fixed). **Decided by:** the implementing agent under delegated authority (§10).

`optim/benchmark.build_table` promises "one row per (config × target), **averaged over
seeds**". Its `_group_key` included `row["config_id"]`, and `Config.config_id()` is a SHA of
`asdict(self)` — the **whole** config, `train.seed` included. Both `optim/sweep.py:63` and
`optim/target_report.py:493` set `train.seed` per seed via `apply_overrides`. So every
K-seed replicate landed in its **own group of one**.

Consequences, on every row of that table:
- `n_seeds` was always **1**;
- `kstar_identifiability_std` — defined as the spread of recovered k\* *across* seeds, and
  computed only `if len(kstars_ok) > 1` — was always **NaN**.

`docs/HANDOFF_identifiability.md:176` and `docs/IDENTIFIABILITY_EXPERIMENTS.md:107` both
instruct the reader to weigh that column "as seriously as the means". It could never be
computed. `degradation_table` keys differently (`arm`, dataset, `n_true`, `n_model`) and was
unaffected, which is why the defect was invisible in the one table anyone had run.

**Reproduced:** `train.seed=0 → a36b4723c040`, `seed=1 → f94d58ade69e`, `seed=2 →
23cad61cdcc3`.

**Fix.** New `Config.arm_id()` — the same hash with the seed fields (`train.seed`,
`model.seed`) held at a fixed sentinel. `model.seed` is included because it defaults to
`None` meaning "derive from `train.seed`", so leaving it in would reintroduce the seed
dependence whenever it is set explicitly. Verified on the real code path: three configs
built exactly as `sweep.py`/`target_report.py` build them share `arm_id = 7688d6ecb934`
while their `config_id`s all differ, and changing `data.sample_key` still separates them.

`arm_id` is now written on every run row (`train.py`, both the meta and the index row) and
`_group_key` keys on it. The table reports `arm_id` in place of `config_id` — a row of that
table is an *arm* aggregated over seeds, and `config_id` is per-run by construction — plus
**`seeds`**, the sorted list actually aggregated, because `n_seeds` alone cannot be audited:
a group of 3 built from seeds `[0,0,0]` would otherwise be indistinguishable from one built
from `[0,1,2]`.

**`sample_key` is now in the group key explicitly.** It never was — it reached the key only
via `config_id`. Simply dropping `config_id` and keying on the coarse tuple would have
**pooled different targets into one mean**, a worse defect than the one being fixed. It is
also covered by `arm_id` (it is part of the config), so the two agree; both are present
because the failure mode is severe and silent.

**Legacy rows keep splitting per seed.** A row written before `arm_id` existed carries no
seed-independent identity, so `_group_key` falls back to `config_id` and the row reports
`arm_id = None`. Deliberate: guessing a grouping risks pooling genuinely different configs,
and an honest ungrouped row beats a plausible wrong mean. Confirmed against the tracked
ledger — all 10 `stage0_bioviab` rows come back as `arm_id=None, n_seeds=1`.

**Consequence for earlier results:** none, and this was already measured. D-EVID-10 found
max group size 1 on every tracked index, so **no cross-seed aggregate has ever been
published** — there is nothing to retract. What was missing is the capability. Note the
corollary: **`kstar_identifiability_std` and the §3.1 sweep both require targets re-run
after today**, since existing rows carry neither `arm_id` nor `repro_J_vector` (D-EVID-12).

**What was rejected and why:** (a) keying on an enumerated tuple of row fields
(`source, dataset, sample_key, N, m, form, strategy, adam_steps, …`) — retroactively
groupable, but two configs differing in any field not enumerated (a loss weight, a staging
fraction) would pool silently, trading a visible defect for an invisible one; (b) reading
each run's `config/frozen_config.yaml` at aggregation time to recompute an exact arm
identity — retroactive and exact, but couples the aggregator to the run-directory layout and
breaks for pruned directories; (c) dropping config identity from the key entirely — pools
different step budgets and loss weights into one mean, breaking the function's own contract.

**Still open:** `reproducibility_table` shares `_group_key` and so inherits the fix, but it
remains uncalled repo-wide with no CLI flag (see D-EVID-12's closing note). Wiring it up or
deleting it is housekeeping for the owner.

**Where it lives:** `src/rngrn/config.py::Config.arm_id`;
`src/rngrn/optim/benchmark.py` (`_group_key`, `COLUMNS`, `build_table`);
`src/rngrn/train.py`; `tests/test_benchmark_grouping.py`.

### D-EVID-14 — three physics consumers read the DIMENSIONLESS `model.D` on the non-dimensional path

**Date found / fixed:** 2026-08-04 (multi-agent branch review, M6). **Status:** SUPERSEDED
(defect; now fixed). **Decided by:** the implementing agent under delegated authority (§10).

`recover(nondim=True)` optimises on the unit box, where the learned parameter is
`D_hat = D / L**2`. `recover()` returns the model **unchanged** (`recover.py`'s "back to
physical units" block converts the *reported* values into `RecoveryResult.D_phys` and
leaves `model.theta_D` alone — correctly, since rewriting it would corrupt the objective's
own parameterisation). So `result.model.D` is dimensionless on that path while
`result.D_phys` is physical.

Three consumers read `result.model.D` and then applied it at the **physical** length scale:

| site | consequence |
|---|---|
| `validate.py` → `robustness_volumes` | a **wrong number**: the unstable band moves outside `_CLOUD_KGRID`'s absolute 4.0 cap, so the cloud reports ~0 |
| `train.py` → `_morphology_rollout` → `simulate` | diffusion starved by L\*\*2, so `patterned=False`, `rollout_status='unpatterned'`, morphology never scored |
| `scoring/overparam.py` | `D_observed_max` / `D_extra_max` reported 1/L\*\*2 off |

`validate.py` already did this correctly for `turing_ok` (it reads `result.D_phys` with a
`model.D` fallback), two lines above the `robustness_volumes` call that did not — so the
correct pattern was present and simply not applied.

**Reproduced** on the branch's own Turing fixture (`tests/test_rollout.py::turing_model`,
L = 60, D = [0.467, 3.338, 0.299]):

| `turing_volume_` | 1 % | 4.8 % | 10 % | 20 % |
|---|---|---|---|---|
| physical D | 1.000 | 0.995 | **0.835** | 0.595 |
| dimensionless D | 0.000 | 0.005 | **0.010** | 0.015 |

A maximally robust circuit reported as maximally fragile, with no error raised.

**Fix:** `analysis._model_JD`, `robustness_cloud`, `robustness_volumes` and
`rollout.simulate` all take an explicit `D=`, defaulting to `model.D` (correct on the
dimensional path, where `D_phys == model.D` by construction). `validate.score_recovery`
passes the `D_rec` it already computes; `_morphology_rollout` passes `result.D_phys`;
`overparam_report` reads `result.D_phys` with the same fallback.

**Consequence for earlier results: none.** All 13 tracked run rows carry `nondim=False`
(D-EVID-10), and no committed config or script sets `nondim: true`. Nothing recorded is
affected — this is a latent defect fixed before it could produce a number. The natural
trigger is `PREREGISTRATION.md` §3.5b, which is precisely the work that would have run it.

**What was rejected and why:** (a) rewriting `model.theta_D` to the physical value inside
`recover()` so `model.D` is always physical — rejected because the model *is* the object the
non-dimensional objective was optimised in; mutating it would make the returned model
inconsistent with the loss that produced it and silently change what a reloaded checkpoint
means; (b) a module-level "always convert" helper that inspects `result.nondim` — rejected
because it would still be something a caller must remember to call, i.e. the same failure
mode one level up. An explicit `D=` parameter puts the units question in the signature,
where a reviewer sees it.

**Also marked here:** `_CLOUD_KGRID`'s absolute 4.0 cap is now flagged `[UNCALIBRATED]`.
It is the same class of defect unit B4 removed from `recover._kgrid_for` (an absolute band
in rad/length rather than a scale-free one), it is adequate for the registered L range, and
it is the reason this defect showed as ~0 rather than as a merely shifted k\*.

**Where it lives:** `src/rngrn/eval/analysis.py` (`_model_JD`, `robustness_cloud`,
`robustness_volumes`, `_CLOUD_KGRID`); `src/rngrn/eval/rollout.py::simulate`;
`src/rngrn/validate.py`; `src/rngrn/train.py`; `src/rngrn/scoring/overparam.py`;
`tests/test_nondim_units.py`.

### D-EVID-15 — what the adversarial re-review of D-EVID-7..14 found, including a regression those fixes introduced

**Date:** 2026-08-04. **Status:** DECIDED (a correction round). **Decided by:** the
implementing agent under delegated authority, after three independent reviews — two Claude
subagents and an independent Codex 0.144.4 pass. All three re-derived the six diagnoses;
**all six were confirmed sound.** The objections were to *completeness*, and one was a
regression the fixes themselves caused.

**1. D-EVID-13 BROKE `reproducibility_table` — and the commit message said the opposite.**
Widening `_group_key` to an 8-tuple updated `build_table` but left
`reproducibility_table` unpacking seven, so it raised
`ValueError: too many values to unpack` on **every** call. Both the commit message and this
register claimed it "shares `_group_key` and so inherits the fix". It did not; it was broken
by it, and nothing caught that because the function has zero callers and had zero tests.
Fixed (unpack, `arm_id`/`sample_key` in `REPRODUCIBILITY_COLUMNS`), plus
`test_reproducibility_table_still_runs` so **dead code cannot silently become broken code**.
Its D-EVID-12 limitation — it re-thresholds an already-collapsed sign vector — is now stated
in its docstring with an explicit "do not wire this up before porting the raw-J path".

**2. D-EVID-14 missed a fourth consumer: `rngrn analyze`.** `cmd_analyze` loaded a raw
checkpoint and called `linear_stability` and `robustness_cloud` without
`physical_model_from_checkpoint`, which its sibling `cmd_evaluate` calls twelve lines above
with a comment saying not doing so "would integrate the wrong diffusivity silently". Measured
on the Turing fixture: at L=220 a genuinely Turing-unstable circuit prints `turing: false`
with `sig_max = −0.033`. Fixed; `linear_stability` also gained an explicit `D=`.

**3. D-EVID-7's own DECISIONS entry contradicted its own evidence.** It said the new
generators are "unaffected in kind". The cited evidence row, `stage0_prior_off_0004`, **is**
`three_gene_qvar` with `trivial_kstar_err = 0.000` — a free draw from 3..14 can still land
on 6. The mitigation is statistical, not categorical. Corrected above.

**4. A published conclusion built on the defect was never revisited.**
`docs/BIO_VIABILITY.md` §4.4 concluded that `sample_0004`'s k\* "carries no information
about the model either way" because `trivial_kstar_err = 0.000`. Recomputed against the
honest reference: the control is **0.0754**, so prior-OFF (0.046) **does** beat it — the
strongest k\* number in that table — while prior-ON (0.116) **loses** to it, a cost the old
framing hid. Corrected, with both directions stated.

**5. Counts.** "6 of 13 rows" where the resolution floor exceeds the 8.3 % bar is **7 of
13** (floors: 5.00, 5.00, 6.25, 6.25, 7.14, 7.14, 8.33, 8.33, 12.50, 12.50, 16.67, 16.67,
16.67 %). The separate claim of "12 tracked run rows that store a Jacobian" was challenged
and is **correct** — 13 result files, 12 carrying `recovered.J`.

**6. `arm_id` was neutralising `model.seed` unconditionally, pooling two different
experiments.** `ModelConfig.seed`'s own note says an int "holds the model init FIXED while
train.seed varies" — a *design* variable, and the shape of D-EVID-4. Now neutralised **only
when None** (the derived case), so a pinned-init arm and a free-init arm no longer share a
`kstar_identifiability_std`.

**7. `_sensitivity_cells` violated fail-loud.** A bare `except Exception: nan` made a
truncated `repro_J_vector`, a shape mismatch and a scorer bug render identically to "this
row predates the column". Narrowed to `ValueError`, and every cell now carries a
`..._status` recording the reason rather than discarding it.

**8. Smaller repairs.** `arm_id` added to `export.RUN_ID_COLS` (it was melting into an
observation row, so the tidy frame could not be grouped by arm — the one thing it exists
for); `REPORT_ID_COLS` corrected from the never-emitted `n_seeds` to `n_seeds_requested`;
`build_table` now emits `n_unique_seeds` because duplicate rows for one seed are weighted in
every mean; `turing_criterion` recorded on each row so a ledger spanning 2026-08-04 can
separate the two `recovered_turing` definitions, with `index.py` documenting it; a
`SIDE_NEUTRAL` firewall class added so `utils.py` — imported by both sides after D-EVID-9
moved `d_ratio_of` there — is audited rather than being an unaudited hop; and the stale
`pytest.skip` in `test_gate_contract.py` (whose message "train.fit does not yet thread
RecoveryInput.L" had been false for some time) replaced by hard assertions, since it was the
only end-to-end guard that the leak controls populate at all.

**What this round did NOT fix — see the review backlog.** Three reviewers surfaced further
findings that are real but larger than a correction round, including several where a
*pre-registered pass condition does not compute what it says*. Two are owner-only under §10:
`PREREGISTRATION.md` §3.3 still mandates pairing the headline with `trivial_kstar_err`
(D-EVID-7 established that is the wrong control), and §3.1's `topology_consistency`
denominator is the surviving-seed count rather than K. Neither may be amended by an agent.

**Where it lives:** `src/rngrn/optim/benchmark.py`, `src/rngrn/cli.py`,
`src/rngrn/eval/analysis.py`, `src/rngrn/config.py`, `src/rngrn/export.py`,
`src/rngrn/index.py`, `src/rngrn/validate.py`, `src/rngrn/optim/target_report.py`,
`docs/BIO_VIABILITY.md`, `tests/test_firewall.py`, `tests/test_benchmark_grouping.py`,
`tests/test_gate_contract.py`, `tests/test_target_report.py`.

### D-EVID-16 — branch consolidation: `model.init` was a silent no-op, and the ledger sweep is redone at 154 rows

**Date:** 2026-08-04. **Status:** DECIDED (a consolidation + one defect fix).
**Decided by:** the implementing agent under delegated authority, after a read-only survey of
all 12 worktrees by three explore agents.

**The survey.** Of 32 local branches, **26 were fully merged** into `feature/turing-training`
— the July wave landed correctly. Five carried unmerged work: `docs/agent-conventions` (2
commits, 100 % superseded — dropped), `docs/hooks-config` (3; one genuinely missing —
cherry-picked), `feature/rngrn-c-mu` (2, deliberately PARKED and unvalidated — left
unmerged), `feature/rngrn-c-tune-nc1` (10) and `feature/rngrn-c-tune-comp` (21).

**The defect: `model.init` was a silent no-op with asserting provenance.** `train.py::fit()`
never passed `cfg.model.init` to `recover()`. It referenced it in exactly one place — writing
`model_init` onto the run-index row — so `-o model.init=low_basal` ran the **default** init
while `frozen_config.yaml` and the `model_init` column both asserted `low_basal`. Same class
as D-EVID-4 (`train.seed`) and D-EVID-5 (`param_prior`): the knob does nothing and the
evidence record says it did something. D9 had *documented* this as a known limitation since
2026-07-29 without fixing it; unit C1 fixed it, and the fix reached this branch only now.

**Consequence for recorded results: NONE — all four `low_basal` rows are genuine.** Of 154
consolidated run rows, **4 carry `model_init: low_basal`**, all `three_gene_qvar/sample_0000`,
seed 0, in `experiments/runs.jsonl`: two at `git_sha 570f3c8`, two at `git_sha 1a2363b`.
Everything else is `model_init: default`, where the no-op was harmless by construction.

> **CORRECTED before publication of this entry.** A first draft of this paragraph claimed the
> `570f3c8` pair predated the fix and so "used `default`" while the record said `low_basal`.
> **That was wrong**, and the error is worth recording because the instrument was wrong, not
> just the answer: I tested `git merge-base --is-ancestor 326822c 570f3c8`, which fails —
> but it fails because **`570f3c8` is a REBASE TWIN of the fix commit**, not its predecessor.
> Ancestry is the wrong test for a rewritten hash. Three checks settle it:
> `git show 570f3c8:src/rngrn/train.py` carries `init=cfg.model.init` at line 232;
> `git diff 570f3c8 326822c -- src/rngrn/train.py` is **empty**; and the two pairs' losses
> agree **bit-for-bit** at `1.7522335969650809` and `0.8104004347760894`. Sixteen significant
> figures cannot coincide across two different inits.
>
> **The real finding is the opposite one.** Those four rows are **two pairs of deterministic
> re-runs** — the same computation recorded twice. They must be **de-duplicated, not
> discarded**: `build_table("experiments")` reports them as `n_seeds=2, n_unique_seeds=1,
> seeds=[0,0]`, i.e. one run counted twice in every mean. The `n_unique_seeds` column added
> hours earlier under D-EVID-15 is what surfaced it, which is the column doing exactly the
> job it was added for.

`570f3c8` is on no branch — an orphaned pre-rewrite commit — but the run rows still carry it
verbatim, and that honesty is what made the pairing recoverable at all. A run row's `git_sha`
must never be normalised to "the equivalent commit on a branch".

**Ledger sweep redone.** D-EVID-10 audited 13 tracked run rows. Consolidation takes that to
**154 rows / 130 result files** across 12 experiment roots. Re-swept:

| defect | consolidated impact | evidence |
|---|---|---|
| **D-EVID-11** strict `turing_ok` | **0 flips in 129 re-scored runs** | every `recovered.J`/`D_phys` re-scored under σ(0)<0 ∧ max σ(k>0)>0 |
| **D-EVID-14** dimensionless `D` | **none** | `nondim = False` on all 154 |
| **M5** morphology species mismatch | **none** | `observed_idx = [0,1,2]` on all 154 |
| **D-EVID-12** rtol sweep | **no published sweep** | `repro_sign_zero_rtol = 0.05` on all 154 |

So D-EVID-11's "0 of 12" now rests on **0 of 129** — a 10× stronger base, and the strongest
statement available that the `turing_ok` correction moved no recorded number.

**What the merge did NOT resolve, carried forward:** both new tuning docs
(`C1_COMPETITIVE_TUNING.md`, `C2_NC1_TUNING.md`) read `kstar_fft_rel_err` against
`trivial_kstar_err`, which D-EVID-7 established is the wrong control, and both cite the
"0 % → 82 % Turing-unstable at init" figure D-EVID-11 withdrew. Their *numbers* survive; some
of their *readings* invert. Correcting them is doc work on evidence that is now in-tree.

**Where it lives:** `src/rngrn/train.py`; `tests/test_smoke.py`; `docs/DECISIONS.md` D9;
`experiments/**` (12 roots).

---

## Part 2 — Decisions

### D1 — D-ratio prior centred at 7.5 (literature Nodal/Lefty), not ~100 (generator median)

**Date:** 2026-07-29 (unit 5 session). **Status:** DECIDED, with a deliberately
unresolved tension recorded alongside it. **Decided by:** the user, per
`configs/bio_box.yaml`'s header comment ("USER DECISION ON RECORD (see the unit 5
task brief)") — **reversing an earlier in-session choice of ~100** (matching the
generator's own draws).

**The decision:** the biological-plausibility box's D-ratio prior centre is **7.5**,
the measured Nodal/Lefty differential-diffusion ratio in live zebrafish (canonical
literature value, `docs/STATE_OF_THE_SCIENCE.md` §11), not ~100–135, which is where
the `three_gene` generator's own sampling distribution sits.

**Evidence:**
- `three_gene` generator D-ratio: min 16.6 / **median 134.9** / max 249.3 over 127
  samples (`docs/STATE_OF_THE_SCIENCE.md` §7, table). Sampling range quoted elsewhere
  as `10**U(0.9, 2.4)` ≈ 7.9–251.
- Literature anchor: Nodal diffuses ~7.5× slower than Lefty (zebrafish, cited in
  `docs/STATE_OF_THE_SCIENCE.md` §11) — a **~15× gap** from the generator median.
- The tension is explicitly *not* treated as a contradiction: the same literature
  reports that in extended networks (more nodes) differential diffusivity is not
  required at all for patterning, and this project's own data corroborates a version
  of that — `docs/ROBUSTNESS_MEASUREMENT.md` §4.4 measures that immobilising the
  slowest diffuser (making D-ratio effectively infinite in the classical sense) keeps
  **127/127** systems strictly Turing and *improves* median robustness (1.000 vs
  0.960 baseline).

**What was rejected and why:** centring on ~100 (the generator's own median) was the
earlier in-session choice, reversed because it would calibrate a "biological
plausibility" prior against the project's own synthetic sampling box rather than
against literature — precisely the kind of firewall-adjacent circularity
`docs/STATE_OF_THE_SCIENCE.md` §1 warns about for any new prior: "Centring a D-ratio
prior on the measured median true D-ratio... routes `AnswerKey.D` into recovery
through judgement rather than through an import."

**Deliberately left as an unresolved tension, not settled here:** the ~15× gap
between 7.5 and the generator's ~135 is recorded as intentional and unresolved in
`docs/STATE_OF_THE_SCIENCE.md` §11 ("Open design choice: centre the prior on ~7.5
... or ~100 ... a wide spread spanning both keeps the prior informative but honest").
The *bio_box* prior settles on 7.5; the wider open question of whether training itself
should ever be pointed at ~100 is not closed by that choice.

**Where it lives:** `configs/bio_box.yaml` (`d_ratio.centre: 7.5`, `spread: 1.0`
natural-log units, bounds `[1.0, 60.0]`); `src/rngrn/scoring/plausibility.py`;
`src/rngrn/losses/terms.py::param_prior` (`loss.dratio_centre` default `7.5`,
`loss.dratio_spread`); `docs/STATE_OF_THE_SCIENCE.md` §11.

### D2 — `d_ratio` is defined as the ratio of the two MOST-MOBILE species, excluding an immobile node

**Date:** 2026-07-29 (PR 6, unit 5). **Status:** DECIDED (a design choice, flagged as
`[IMPL]`, not independently validated against alternatives). **Decided by:** the
implementing agent under delegated authority, reasoned from measured robustness data.

**The decision:** both `scoring/plausibility.py::d_ratio_of` and
`losses/terms.py::param_prior`'s D-ratio use `largest / second-largest` diffusivity
(the two most-mobile species), never global `max/min`. For N≥3 this excludes the
single smallest D from the ratio by construction.

**Evidence:** `docs/ROBUSTNESS_MEASUREMENT.md` §4.4 — immobilising the slowest
diffuser keeps **127/127** `three_gene` systems strictly Turing (vs 81/127 for the
middle diffuser, 38/127 for the fastest). A near-immobile third node is therefore the
*mechanism*, not a defect, and a max/min ratio definition would penalise exactly the
node Tica et al.'s design intentionally makes near-zero.

**What was rejected and why:** plain global `max/min` over all N species — rejected
because it would drive the D-ratio to near-infinity (or force a floor/clamp) whenever
a near-immobile modulatory node is present, penalising a configuration the project's
own robustness baseline shows is *beneficial*, not aberrant.

**Not independently validated:** the module docstring and the task itself flag this as
`[IMPL]`, tested for the specific behaviour claimed
(`test_d_ratio_of_ignores_a_near_immobile_node`,
`test_param_prior_does_not_penalise_an_immobile_node`) but not compared empirically
against the max/min alternative on real recoveries.

**Where it lives:** `src/rngrn/scoring/plausibility.py::d_ratio_of`;
`src/rngrn/losses/terms.py::param_prior`.

### D3 — `bio_box` D-ratio upper bound of 60.0 is a reasoned proxy, not a measured bound

**Date:** 2026-07-29 (PR 6). **Status:** DECIDED, explicitly flagged as unmeasured.
**Decided by:** the implementing agent under delegated authority.

**The decision:** the plausibility box's D-ratio range is `[1.0, 60.0]`. The lower
bound (1.0) is Turing's own 1952 requirement that the inhibitor diffuse faster than
the activator. The upper bound (60.0) is **not** a measured D-ratio bound from any
source — it is derived from the same `STATE_OF_THE_SCIENCE.md` §11 paragraph's cited
*absolute* diffusivity spread across measurement regimes (~60 µm²/s FCS-local to
~1–2 µm²/s FRAP-global, roughly two orders of magnitude), reused as a ceiling on
physiologically plausible differential diffusion.

**What was rejected and why:** no direct literature citation for a D-ratio ceiling
was found; rather than leave the field uncited (as `beta` is, see below) or invent an
independent number, the two-orders-of-magnitude absolute-diffusivity spread was
repurposed as a proxy ceiling and labelled as such in the source comment.

**Where it lives:** `configs/bio_box.yaml` (`d_ratio.high: 60.0`, with the `source:`
field stating the reasoning above verbatim).

### D4 — `beta` (basal production) left UNCITED in the plausibility box

**Date:** 2026-07-29 (PR 6). **Status:** DECIDED not to invent a bound. **Decided by:**
the implementing agent, following the project's evidence-discipline rule.

**The decision:** `configs/bio_box.yaml`'s `beta` row has `low: null, high: null,
source: UNCITED`. An UNCITED row is reported (its in/out-of-box verdict is computed)
but never scored in/out-of-box and never hinged in the prior loss.

**Evidence rejected as insufficiently grounded:** `docs/STATE_OF_THE_SCIENCE.md` §10
records a `beta ~ 1e-4..1e-2` range, but that range came from an **init-distribution
search** (the low-basal init sweep, see D8 below) explicitly marked "a modelling
assumption rather than a neutral prior" and "deliberately not adopted" for training —
repurposing it as a biological-plausibility claim would misrepresent its provenance.

**Where it lives:** `configs/bio_box.yaml` (`beta:` block).

### D5 — `k_star_fft` (image-derived) is the headline validation target, not analytic `k_star`

**Date:** decided 2026-07-29, reversing the 2026-07-26 decision. **Status:** DECIDED
(current), SUPERSEDES the 2026-07-26 choice. **Decided by:** the user.

**The decision:** validation and every scoring table now head on `kstar_fft_rel_err`
(FFT-measured k\* from the image) as HEADLINE, with `kstar_rel_err` (analytic k\* from
the true J, D) as SECONDARY. This reverses the 2026-07-26 decision recorded at
`docs/STATE_OF_THE_SCIENCE.md` line 499, which had made analytic `k_star` the headline
and `k_star_fft` the secondary diagnostic.

**Evidence for the reversal:** analytic `k_star` is `argmax_k σ(k)` from the *true* J
and D — a property of the generating equations no real experiment can observe.
`k_star_fft` is a property of the image, which is what an actual inverse problem has
access to. For a pipeline whose goal involves experimentally obtained patterns, the
observable target is the defensible one (`docs/STATE_OF_THE_SCIENCE.md` §8.1).

**Circularity checked and found real but limited:** `docs/STATE_OF_THE_SCIENCE.md`
§8.2 measured `k_star_fft` (the stored attr) against `kstar_obs` (the live loss
anchor, `observables.kstar_of`) on 128 samples across 4 splits: median relative
difference 1.56%–3.67% by split, max up to 39.3%, **exact matches 0/128**, within 1%
22/128, correlation r=0.998. So validating against `k_star_fft` is not literally
circular (it is an independently computed FFT estimate) but §8.3 notes both
estimators share the same bias structure, so a model that reproduces the bias is
still rewarded to some degree.

**What was rejected and why:** keeping analytic `k_star` as headline was rejected
because it grades agreement with a latent quantity the model was never given access
to, rather than agreement with something measurable.

**Consequence for earlier numbers:** every k\* number recorded before 2026-07-29 was
reported against the analytic target and is **not directly comparable** to numbers
reported after the reversal (`docs/CODE_REALITY.md` §4, `docs/STATE_OF_THE_SCIENCE.md`
§8.1).

**Where it lives:** `src/rngrn/validate.py` (module docstring + `score_recovery`
inline comment); `src/rngrn/optim/benchmark.py` (`COLUMNS`, `DEGRADATION_COLUMNS`,
both docstrings, FFT column ordered first); `src/rngrn/data/gate.py` (`AnswerKey`,
`from_registry` docstrings). PR 2.

### D6 — the domain-size leak: vary L with random periods-per-box, decorrelating L from k\*

**Date:** 2026-07-29 (unit 11, "the L-generalisation task", executed as unit 12/PR 12,
`three_gene_ldata`). **Status:** DECIDED — settles open decision #2 in
`docs/CODE_REALITY.md` §11. **Decided by:** the user's requirement ("the model must
generalise across domain sizes") forecloses option (a); settled by unit 11/12.

**The decision:** every prior generator set `L = clip(6·2π/k*, 18, 220)`, holding
exactly 6 wavelengths in every box, so `L` was an algebraic function of `k*_true`. The
new policy is `L = p · (2π/k*)` with `p` a free integer drawn per-system from
`U{3..14}`; candidates admitting no feasible `p` in `[18, 220]` are rejected rather
than clipped (clipping recreates the same collapse at the clip bound). Two new
datasets were generated under this policy: `three_gene_qvar` (34 systems, one L each)
and `three_gene_multiL` (23 systems × 4 domain sizes each, same kinetics/`k_star`/
init seed per group, differing only in L).

**Evidence — the leak, measured:** on the original 127 `three_gene_{train,val,test}`
samples, the image-blind predictor `k_hat = 6·2π/L` scores **0.0% median relative
error** (100% of samples within 1%) — periods-per-box is exactly 6.000 for every one.
On `three_gene_qvar` (34 systems, `p ~ U{3..14}`) the same blind predictor scores
**45.5% median error**. On `three_gene_multiL` (92 samples, `p` in {4,7,10,13}) it
scores **45.0%**. An **oracle** best-fixed-`p` (chosen after seeing the answers, the
single strongest possible fixed-period predictor) still costs 28.6% on `qvar` and
29.2% on `multiL` — no fixed periods-per-box explains the new data.

**What was rejected and why** (all three options were measured and put to the user,
per `docs/STATE_OF_THE_SCIENCE.md` §7.1):
- **(a) fixed L** — matches the user's originally literal statement ("I wanted all of
  my patterns to have the same domain size") but makes L carry zero information and
  forecloses any domain-size-generalisation claim, and caps k\* to a 5.3× span at
  grid=96 versus the generators' 0.08–3.0 screening band. Rejected once the user
  clarified the model must generalise across domain sizes — a requirement fixed-L
  makes untestable.
- **(c) fixed L with grid raised to 192/256** — widens the band but at 4–7× simulation
  cost; not chosen (not depended on to close the leak).
- Chose **(b)**, vary L with periods-per-box randomised.

**Old data not deleted:** the original 127 `three_gene_{train,val,test}` samples
remain the comparison baseline (`p=6` exactly, by construction) precisely because they
now demonstrate what the leak looked like.

**A related permanent limitation recorded alongside this decision:** the staging
generator seeded each screen with `abs(hash(topo))`. Python salts string hashing per
process, so the recorded seed **never reproduced the screen** — confirmed directly
(two runs of the smoke mode at the same nominal seed screened a different number of
candidates). Fixed going forward with a SHA-256-derived seed (`scripts/gen_tg3.py`),
and both new datasets were regenerated after the fix — but **the existing 127
`three_gene` samples cannot be regenerated from their recorded seed.** This is a
permanent limitation of the legacy data, not something this decision (or any later
one) can repair; it does not affect the *validity* of those 127 samples, only the
ability to re-derive them byte-for-byte from the recorded provenance.

**Also fixed alongside this decision:** the generator (`data/staging/tg3/generator.py`)
lived only in a gitignored `data/staging/` tree and was never in version control.
Tracked now as `scripts/gen_tg3.py`.

**What is still OPEN, not settled by this decision:** whether `three_gene_multiL`
should be split by `system_id` so no system straddles a train/val/test split — both
new datasets ship with `splits: {}` (unsplit), left to the consumer. See D13 below.

**Where it lives:** `scripts/gen_tg3.py`; `docs/DATASETS_L.md`; `data/datasets/
three_gene_qvar/`, `data/datasets/three_gene_multiL/`; `docs/CODE_REALITY.md` §11
item 2 (struck through / marked SETTLED).

### D7 — `loss.weights.resid` defaulted to 0.0 — settled OFF

**Date:** decided in exp06 (pre-phase-A), promoted into the library defaults
2026-07-29 (PR 11, unit 1). **Status:** DECIDED. **Decided by:** exp06's measured
sweep; promoted into the library by the unit-1 agent under delegated authority
(a mechanical promotion of an already-decided value, not a new decision).

**The decision:** the PDE-residual loss term (`resid`) defaults to weight **0.0** in
`losses/total.py::compute_terms` / `LossConfig`. The term is kept in the code (not
removed) for future use, but is off by default and — as an optimisation — is *omitted
from computation entirely*, not computed and multiplied by zero, when the active
strategy's weights are static and its base weight is 0 (`compute_terms
(compute_resid=False)`, measured at 45% of a forward+backward step: 9.39ms with vs
5.15ms without, 96×96, N=3, 40 reps). `parts["resid_skipped"] = True` is set so no
downstream reader can mistake "not computed" for "computed as zero".

**Evidence:** exp06 swept pixel batch size {64, 128, 512} × residual weight {1, 3, 10}
— **9 cells × 8 seeds** — and **all nine cells collapsed to 1/8 Turing seeds**, with
best median k\* error 11.8% against 0.4% with the residual off.

**What was rejected and why:** any nonzero residual weight in the {1, 3, 10} × {64,
128, 512} grid tried — rejected because every cell measurably collapsed Turing
recovery relative to the residual-off control.

**Known consequence, recorded as an open problem (not a bug):** hidden-channel
(m < N) recovery has **no known-good objective** at `resid = 0`, because the latent
fields enter the objective through `stationarity_residual` and nothing else — their
gradient is exactly 0.0 at that weight (measured, N=3, m=2). `recover()` now refuses
such runs outright rather than silently returning the random init as a "recovered"
latent field, per PR 11.

**Where it lives:** `src/rngrn/config.py::LossConfig` (`weights.resid` default 0.0);
`src/rngrn/losses/total.py::compute_terms`; `TUNING.md` Stage 2, "`loss.weights.resid
= 0.0` is SETTLED, not untuned".

### D8 — split hinges + frame-scale anchor promoted into the library, taking library Turing fraction from 0% to 36.8%

**Date:** 2026-07-29 (PR 11, unit 1). **Status:** DECIDED. **Decided by:** the
implementing agent under delegated authority, promoting an already-validated
experiment-script objective (exp05) into the library — a mechanical promotion of
already-tested arithmetic, verified bit-identical before landing.

**The decision:** three terms proven in `scripts/exp05_pixel_minibatch.py::fit`
replace the library `losses/total.py`'s prior objective (which had never itself
produced a measured Turing recovery):
- `terms.turing_hinges_split` — the two Turing conditions evaluated on disjoint
  k-support (`i_min = max(1, int(0.1·len(kgrid)))`) — replaces the single-direction
  hinge, now kept only as the documented control arm.
- `terms.frame_scale_anchor` — `mean((log(frame.mean()) − log(x*))²)`, firewall-legal
  since `frame.mean()` is an image observable.
- `weighting.staging_factor` / `DataFirstStaging` — the exp05 schedule (off for the
  first 25% of steps, ramped to 1.0 over the next 25%), as a composable wrapper.

**Evidence:**
- **Bit-identical promotion, not reimplementation:** `turing_hinges_split` vs
  `exp02::turing_hinges_split` vs `exp05::split_hinges`, and `frame_scale_anchor` vs
  the exp02/exp05 expression, over 20 random-init seeds — **max |difference| =
  0.000e+00**.
- **The library-vs-experiment gap this closes:** the pre-promotion library objective
  scored **0% Turing** on the reference used at that time (`docs/CODE_REALITY.md`
  §1); `scripts/exp05_pixel_minibatch.py`, running the same terms outside the
  library, reaches **36.8%** — reconfirmed post-promotion at **38/40 converged, 14
  Turing, turing_frac 0.3684** over 40 seeds × 400 steps (per the phase-B unit brief's
  "what phase A established").
- End-to-end regression test: at the same config/seeds/budget, the pre-promotion
  objective fails all 4 restarts' steady-state solve; the promoted objective
  completes and recovers Turing (`recovered_turing = True`, kstar_rel_err 0.0506).

**What was rejected and why:** two design choices were identified but **deliberately
left unmade**, because settling either would bias what "36.8%" means and is a science
decision, not a mechanical one:
- **`detach_xstar`** — exp05 detaches x\* from the dispersion-term gradient; the
  library differentiates through it via `steady_state_diff`. Default kept `False`
  (library behaviour), but the 36.8% measurement was made with `True` — the two
  settings have never been A/B'd.
- **The hinge k-floor being grid-relative rather than k\*-relative** — measured on
  `sample_0000` (k\*_obs = 0.4320): the promoted library grid puts `k_min` at
  **0.822×k\*_obs** vs exp05's **0.698×k\*_obs**. Defining the floor relative to
  `kstar_obs` would change what the term means; not changed here.

**A defect found and fixed during this promotion (not itself the decision, but load-
bearing for it):** `recover()`'s final post-training scoring pass was previously
unguarded against `SteadyStateError` — one restart landing on a bad steady state
aborted the *entire* recovery and discarded every other restart's result. Fixed to log
`failed_at="final_eval"` and skip that restart only. This fired on restart 0 of the
verifying e2e run; without the fix, that run would have produced nothing.

**Where it lives:** `src/rngrn/losses/terms.py` (`turing_hinges_split`,
`frame_scale_anchor`); `src/rngrn/losses/weighting.py` (`staging_factor`,
`DataFirstStaging`); `src/rngrn/config.py::LossConfig` (`split_hinges`,
`hinge_k_min_frac`, `staging_keys`, `staging_off_frac`, `staging_ramp_frac`,
`detach_xstar`); `docs/CODE_REALITY.md` §11 item 6 (settles the open decision);
`TUNING.md` "Promoted from the experiments (unit 1)".

### D9 — low-basal init: implemented, available via `model.init="low_basal"`, but DEFAULT OFF

**Date:** 2026-07-29 (PR 9, unit 9). **Status:** DECIDED — default stays off; the
knob exists but is not wired into the CLI/`train.fit()` path. **Decided by:** per the
task brief's instruction to leave the default off; the implementing agent measured
the training-time consequence and reported it rather than reversing the default.

**The decision:** `RNGRN(init="default"|"low_basal")` — a firewall-safe alternate raw-
parameter init (`beta` 1e-4..1e-2, `s` 1e-2..10^-0.3, `alpha` 10^0.3..10^1.5, `delta`
0.1..10^0.3, gate logit ~N(0, 2.5), D-ratio 10^0.9..10^2.4), ported unmodified from
`scripts/exp03_turing_first.py::low_basal_init`. `ModelConfig.init` defaults to
`"default"` everywhere.

> **FIXED 2026-08-04 by the branch consolidation — see D-EVID-16.** This paragraph used to
> end: *"`train.py`'s `cfg.model.init` is **not** threaded into `recover()` (out of the
> unit's file scope), so today the field round-trips into `frozen_config.yaml` but has no
> effect via the CLI path."* That is an accurate description of a **silent no-op whose
> provenance asserts the opposite** — the same defect class as D-EVID-4 (`train.seed`) and
> D-EVID-5 (`param_prior`) — and it was recorded here as a known limitation rather than
> fixed. Unit C1 fixed it (`init=cfg.model.init` on the `R.recover(...)` call, guarded by
> `tests/test_smoke.py::test_model_init_is_threaded_from_config_into_recover`), and that fix
> reached this branch only via the consolidation merge. `-o model.init=low_basal` now does
> what it says.

**Evidence — at init, low_basal is dramatically more Turing-unstable:** 400 seeds per
setting (Newton steady state + Jacobian-sign check only, no fit) —
**0/400 (0%)** Turing-unstable-at-init for `default` vs **206/255 converged (80.8%)**
for `low_basal`, consistent with the ~82% figure already documented in
`docs/STATE_OF_THE_SCIENCE.md` §10 for this same beta upper bound.

> ### ⚠ THE EVIDENCE ABOVE IS WITHDRAWN — re-measured 2026-08-04, see D-EVID-11
>
> That 80.8% was produced by `turing_ok`'s **loose** `tr(J) < 0` criterion, which a
> uniformly UNSTABLE system can satisfy. Under the corrected strict criterion the
> low-basal init is **0/398 (0.0%)** Turing-unstable at init.
>
> The number moved for **two independent reasons**, and both matter:
> 1. **The denominator changed.** 255 of 400 inits converged then; **398 of 400** converge
>    now (unit B3's steady-state multistart). The numerator is **206 in both
>    measurements** — the extra 143 converged inits contributed no loose-Turing draws. So
>    the same data now reads 206/398 = **51.8%**, not 80.8%, before any criterion change.
> 2. **The criterion changed.** All 206 were uniformly unstable with k\* pinned exactly to
>    the grid floor. Strict fraction: **0.000**.
>
> **What survives:** low_basal buys the *positive Jacobian diagonal* that Turing requires —
> 114/398 (28.6%) vs 0/400 for `default`. That separation is real, is the honest motivation
> for the init, and is what `tests/test_science.py::
> test_low_basal_init_gains_the_positive_diagonal_but_NOT_turing_reachability` now pins.
> A positive diagonal is necessary for Turing instability; it is not sufficient.
>
> **Effect on this decision:** D9 stays **OPEN**, but the case for adopting low_basal is
> materially weaker than recorded — it was never "0% → 82% Turing-reachable". Combined with
> the 40/40 training failure below, there is now no measured benefit to set against a
> measured cost.

**Evidence — but it fails under actual training, which is why the default stays off:**
direct `recover(ri, init="low_basal", adam_steps=200, lbfgs_steps=0)` on
`three_gene_val/sample_0000` **failed all 40/40 restarts** (8 restarts × 5 seed
offsets) to converge to a valid steady state during training, where the matched
`init="default"` call (same restart/step budget) succeeded (loss=0.517). The raw
init-only measurement above shows low_basal's Newton solve *itself* converges less
often even before any objective is added (255/400 vs 400/400) — the training-time
failure compounds that.

**What was rejected and why:** flipping the default to `low_basal` was rejected/not
attempted, because the training-time measurement above shows it changes *which*
solutions recovery finds and currently fails outright on the one real dataset tested
— exactly the "biases what results mean" criterion that puts a decision outside
mechanical scope, and `docs/STATE_OF_THE_SCIENCE.md` §10 already recorded it as
"deliberately not adopted" prior to this unit.

**Where it lives:** `src/rngrn/model.py` (`RNGRN.__init__`, `_low_basal_raw_params`);
`src/rngrn/config.py::ModelConfig.init`; `src/rngrn/recover.py` (`init` kwarg, unwired
from `train.py`); `docs/CODE_REALITY.md` §11 item 5; `docs/STATE_OF_THE_SCIENCE.md`
§10.

### D10 — `DEFAULT_SIGN_ZERO_RTOL = 0.05` and `topology_consistency := modal_fraction`, both flagged UNCALIBRATED

**Date:** 2026-07-29 (PR 7, unit 3). **Status:** OPEN / UNCALIBRATED — implemented as
the working definition, explicitly not validated against the alternative.
**Decided by:** the implementing agent under delegated authority, as a metric-
definition choice flagged rather than made silently.

**The decisions (two, bundled in the same scorer):**
1. **Near-zero entry dead zone:** a recovered Jacobian entry is treated as
   structurally zero if it is below 5% of that matrix's own max |J| —
   `DEFAULT_SIGN_ZERO_RTOL = 0.05`. Same magnitude as (and for the same reason as)
   `scoring.overparam.DEFAULT_COUPLING_THRESHOLD` (also 0.05, also uncalibrated):
   J's overall scale is not pinned by the objective, so only a per-matrix relative
   cut is meaningful. Deliberately coarser than the 1e-9 "zero to floating-point
   round-off" cut used elsewhere — the two thresholds answer different questions.
2. **`topology_consistency` is defined as the modal fraction** (the fraction of seeds
   producing the single most common exact sign structure), not mean pairwise
   agreement. Both are computed and returned (`mean_agreement` is also in the
   report), but only `modal_fraction` is the headline scalar.

**Evidence for the choice, as reasoned (not measured against real data):**
`modal_fraction` is the strictest, most literal reading of the user's stated primary
success metric — "does the model consistently learn the same topology across seeds,
for one target". `mean_agreement` rewards partial pairwise closeness even when no two
seeds are identical, which is a softer question than the one asked.

**What was rejected and why:** `mean_agreement` as the headline — rejected as
answering a different (softer) question than the user's literal framing, though it is
still computed and reported alongside the headline so it is not lost.

**Explicitly UNCALIBRATED — not settled by this entry:** neither the 0.05 rtol nor
the modal-vs-mean choice has been validated against real cross-seed recovery data;
PR 7 could not obtain a second successful CLI recovery on the same target within a
reasonable seed budget (CLI hit `SteadyStateError` on ~30/31 seeds tried), so the
cross-run aggregation is verified only on synthetic matrices with known sign
patterns, not on live data.

**Where it lives:** `src/rngrn/scoring/reproducibility.py` (module docstring, "THE
topology_consistency DESIGN CHOICE" section); `src/rngrn/validate.py::score_recovery`.

### D11 — `gradnorm`/`ntk` loss-weighting strategies now RAISE rather than silently running fixed weights

**Date:** 2026-07-29 (PR 5, unit 13). **Status:** DECIDED. **Decided by:** the
implementing agent, correcting a fail-loud violation per the project's standing rule.

**The decision:** `GradNormWeighting.combine` and `NTKWeighting.combine`
(`losses/weighting.py`) were stubs that silently ran with fixed weights whenever
selected — a run configured for gradient-magnitude balancing or NTK-based weighting
would appear to execute normally while doing neither. Both now `raise
NotImplementedError` at construction. A working `ratio` strategy (Matas-Gil & Endres'
actual published technique — `weight_k = loss_data / loss_k`, recomputed every
`update_every` steps) was implemented and adopted in their place as the strategy this
project actually ships.

**Evidence this was a real defect, not a hypothetical:** a grep-verifiable code state
(the PR body records finding the stubs by reading `losses/weighting.py` directly) —
any prior config setting `loss.strategy: gradnorm` or `loss.strategy: ntk` would have
silently produced fixed-weight results while claiming otherwise.

**What was rejected and why:** leaving the stubs silently running fixed weights was
rejected outright as a fail-loud violation (project convention #2). Implementing
GradNorm/NTK properly was out of scope for this unit; raising is the correct interim
state until someone implements them for real.

**Where it lives:** `src/rngrn/losses/weighting.py` (`GradNormWeighting.combine`,
`NTKWeighting.combine`, both raising; `ratio` strategy new); `src/rngrn/config.py`
(`LossConfig.ratio_update_every`, default 50); `tests/test_weighting.py`.

### D12 — Domain-size design settled by decorrelating L from k\* (see D6)

Cross-reference: this is the same decision as D6 above (`docs/CODE_REALITY.md` open
decision #2). Recorded once, at D6, to avoid a duplicate entry with different
evidence excerpts drifting apart.

### D13 — Both new datasets left unsplit; whether `three_gene_multiL` should split by `system_id` — OPEN

**Date raised:** 2026-07-29 (PR 12, unit 11). **Status:** OPEN. **Decided by:** no one
yet — explicitly left to the consumer.

**The decision not yet made:** `three_gene_qvar` and `three_gene_multiL` both ship
with `splits: {}` (unsplit), matching the existing registered sets' convention. For
`three_gene_multiL` specifically — 23 systems × 4 domain sizes each, sharing identical
kinetics/`k_star`/init seed within a group — whether a train/val/test split must keep
every system's 4 replicates on the same side (so no system straddles a split) is
unresolved.

**Why it matters:** if a system's replicates are allowed to straddle a split, a model
could see one L of a system in training and be tested on another L of the *same*
system — which would inflate any cross-L generalisation claim, since the model would
have partial information about that exact system's kinetics rather than facing a
genuinely unseen one.

**What is NOT decided here:** this register does not pick an answer. Flagged as an
explicit `OPEN` entry per the task's instruction not to close a decision that has not
actually been made.

**Where it lives:** `data/datasets/three_gene_multiL/manifest.json` (`splits: {}`);
`docs/DATASETS_L.md`.

### D14 — Generator seeded from `abs(hash(topo))`: fixed going forward, permanent limitation for legacy data

Cross-reference: recorded in full under D6 above (same PR, same decision context).
Restated here as its own line because the brief calls it out separately: the existing
127 `three_gene` samples **cannot be regenerated from their recorded seed**, because
Python salts string hashing per process. This is a **permanent limitation of the
legacy data** — not something any later decision can retroactively fix — recorded so
a future reader does not attempt to re-derive those 127 samples byte-for-byte from
their nominal seed and conclude the pipeline is non-deterministic in general; the
non-determinism was specific to `abs(hash(str))` and is fixed in
`scripts/gen_tg3.py` (SHA-256-derived seeding) for every dataset generated after it.

### D15 — cross-L rollouts hold **dx** fixed (n scales with L), with a MEASURED floor of 6 pixels per wavelength

**Date:** 2026-07-30. **Status:** DECIDED. **Decided by:** the implementing agent, as an
explicitly delegated technical choice.

**The decision:** `eval/lgen_eval.evaluate_across_L` simulates the recovered model at each
domain size on a grid `n(L) = nearest even int to n_grid * L / L_train`, i.e. it holds the
grid spacing `dx = L/n` fixed rather than the grid size. Every row additionally records
`pixels_per_wavelength`, and a row that produces a pattern below **6.0** px/wavelength
**raises** instead of reporting the number.

**Why this way round:** at fixed `n`, a 4× box has a quarter of the pixels per wavelength,
so the k\* estimate degrades with L — and the degradation would land in `kstar_phys_cv`,
the headline invariance statistic, looking exactly like physics. Holding `dx` fixed makes
`pixels_per_wavelength` constant across L to within the even-rounding, so the comparison
across L is a comparison of the *pattern* and not of the discretisation.

**The floor is measured, not asserted.** Scanning `n` at fixed L on the Turing fixture of
`tests/test_rollout.py` (L = 60, q ≈ 6.2, `etdrk4_rfft`, seed 0), k\* referenced to the
n = 192 run:

| n | 16 | 24 | 32 | 48 | 64 | 96 | 128 | 192 |
|---|---|---|---|---|---|---|---|---|
| px/wavelength | 2.76 | 3.94 | 5.12 | 7.64 | 10.10 | 14.98 | 20.32 | 30.79 |
| \|Δk\*\|/k\* | 6.9 % | 2.4 % | 0.3 % | 0.8 % | 1.6 % | 2.8 % | 1.0 % | — |
| morphology class | spots | laby | laby | laby | laby | laby | laby | laby |

k\* survives to 3.94 px/wavelength within ~3 %; at 2.76 it moves 6.9 % **and** the class
call flips. 6.0 sits above that breakdown with margin and coincides with the coarsest
resolution the generators ever produced (`three_gene_qvar`/`multiL` draw up to p = 14
periods on a 96 grid → 96/14 = 6.86 px/wavelength), so the floor refuses exactly the
resolutions the data never contained. The ±3 % grid-to-grid spread in that table is also
the **noise floor of `kstar_phys_cv`**: no CV below it is meaningful.

**What was rejected and why:** (a) holding `n` fixed and merely reporting
`pixels_per_wavelength` — allowed by the brief, rejected because it puts a resolution
artefact into the headline statistic and relies on the reader to subtract it; (b) *clamping*
`n` at an `n_max` — rejected as a silent change to the one quantity the rule exists to hold
fixed, so `grid_for_L` raises at both ends instead; (c) tuning the floor down to the
measured 4 px/wavelength — rejected as over-fitting one fixture.

**A second resolution limit that this rule does NOT fix, recorded so it is not mistaken for
one:** `observables.raps` bins on the fundamental `dk = 2π/L`, so k\* is quantised at a
*relative* width of `1/q`. A small box holds few periods and its k\* is coarsely measured
however fine the grid (at q = 3, the bin is 31 % wide). Every row therefore records
`kstar_bin_width` and `kstar_bin_width_rel`, and `kstar_phys_cv` must be read next to the
largest of them. No q threshold was invented on top of the px/wavelength floor.

**Where it lives:** `src/rngrn/eval/lgen_eval.py` (`grid_for_L`, `PPW_FLOOR`,
`PPW_FLOOR_PROVENANCE`); `tests/test_lgen_eval.py`.

### D16 — the cross-L statistics: population CV of k\*, an origin-fixed q-vs-L slope, and NO pass threshold

**Date:** 2026-07-30. **Status:** DECIDED (the statistics); the **thresholds remain OPEN by
design**. **Decided by:** the implementing agent for the definitions; the thresholds are
deliberately left to `docs/PREREGISTRATION.md` and the caller.

**The decision, and the theory it tests.** k\* is the argmax of `σ(k) = eig(J − k²D)`;
neither J nor D contains the domain size, so for one recovered network simulated at several
L, Turing theory predicts an invariant **physical** k\*, a periods-per-box count
`q(L) = L·k*/2π` **linear in L**, and a preserved morphology class. The failure mode being
excluded is the opposite: q pinned to the box while k\* moves as 1/L, which is a model that
memorised a box. `summarise_across_L` reports:

- **`kstar_phys_cv`** — *population* (ddof = 0) SD of `kstar_phys` over patterned rows,
  divided by the mean. Population, not sample, SD so the number does not inflate at the 3–4
  L values this is ever used over.
- **`periods_slope_rel_err`** — q fitted against L **through the origin**
  (`slope = Σ L·q / Σ L²`), compared to the theory slope `mean(k*)/2π`. Because
  `q_i ≡ L_i k_i/2π` identically, the fit is the L²-weighted mean of `k_i/2π` and the
  prediction is the unweighted one: they agree exactly under invariance and diverge when k\*
  trends with L. It is therefore a differently-weighted **second view** of the same
  invariance, not an independent measurement, and both slopes are reported so that is
  visible.
- **`morphology_class_preserved`** — fraction of patterned non-reference rows matching the
  class at L_train, always alongside `n_class_compared` (rows that did not pattern have no
  class and are excluded rather than scored as mismatches; excluding them flatters the
  fraction).

**Separation, measured — the metric can fail.** Positive control (the Turing fixture at
L/L_train = 0.5, 1, 2, 4): `kstar_phys_cv` **0.021**, `periods_slope_rel_err` **0.017**.
Negative control (a synthetic frame carrying a fixed 6.5 periods per box at every L, pushed
through the same row and summary arithmetic): **0.715** and **0.624** — 33× and 37× larger.

**Morphology is explicitly the weakest of the three.** On the positive control the two
*intensity* features are invariant (skew 0.43–0.51, kurtosis −0.74 to −0.81 across an 8×
span of L) while the two *angular* features drift monotonically with q (ang_conc
0.190 → 0.074), because they are computed on a ring of radius q in pixel-frequency units —
a property of the measurement, not of the pattern. The class consequently reads
labyrinth → spots and `morphology_class_preserved` = 1/3, on borderline margins (0.17–0.62),
against a centroid bank fitted at q ≈ 6. And it fails the other way too: the box-pinned
negative control preserves its class **perfectly (1.0)** while k\* moves as 1/L. Class
agreement alone is evidence of nothing; the raw `morph_vector` is recorded on every row so a
reader can see which features moved.

**NO PASS/FAIL LIVES IN THE MODULE.** `summarise_across_L` returns a `verdict_components`
dict of raw quantities and no boolean — `tests/test_lgen_eval.py` asserts that, so a
threshold cannot be smuggled in later without the test noticing. This follows
`scoring/lgen.py`'s decision (3) ("NO PASS THRESHOLD is defined") rather than re-opening it.

**The nondim trap, closed.** `recover()` reports `params["D"]` physically on both paths, but
the **checkpoint** keeps `theta_D` in the units the objective ran in — `D/L_train²` when
`nondim=True` (`recover.py` lines 518–528 convert the *reported* params only). Simulating a
reloaded nondim checkpoint without converting integrates the wrong diffusivity and yields a
k\* wrong by a factor of L_train, silently. `physical_model_from_checkpoint` rewrites
`theta_D` on a deep copy, verifies `model.D` afterwards, and **raises** on a checkpoint
lacking the `nondim`/`L` extras: the two paths are indistinguishable after the fact, so an
older checkpoint is refused rather than assumed dimensional.

**Where it lives:** `src/rngrn/eval/lgen_eval.py` (`evaluate_across_L`,
`summarise_across_L`, `physical_model_from_checkpoint`); `src/rngrn/cli.py`
(`--eval-L` / `--l-factors`); `tests/test_lgen_eval.py`.

---

### D-PLOT-1 — the plottable ARRAYS are persisted per run, at float32 for fields and float64 for k axes, and are TRACKED in git

**Date:** 2026-07-30. **Status:** DECIDED. **Decided by:** the implementing agent, under the
owner's requirement "all data need to be stored in a way that I can make plots on this stage
for my research".

**The problem.** A run recorded scalars (`runs.jsonl`, `target_reports.jsonl`), the recovered
parameters and Jacobian as text, and a checkpoint. That audits a number and re-runs a model.
It cannot draw a figure: **you cannot plot a pattern, a dispersion relation or a spectrum
from a scalar**, and every array was discarded when the process exited. The canonical Turing
figure — recovered σ(k) overlaid on the answer key's σ(k) — was impossible to draw at all.

**The decision.** `train.fit` writes one compressed `<run_dir>/arrays/plot_arrays.npz`
(`solver.save_plot_arrays`, default **ON**) carrying `target_frame`, `model_field` (only when
the rollout patterned), `dispersion_k` + `dispersion_sigma` + `dispersion_sigma_true` on ONE
shared k grid, `raps_k` + `raps_target` + `raps_model`, the two 4-D morphology vectors with
their `morph_feature_order`, and a self-describing JSON `meta` (run_id, git sha, dataset_id,
sample_key, form, L, and every k\*).

**dtype, split deliberately.** *Fields and spectra* → **float32**: they are plotted, never
differentiated, and float32's ~7 significant digits are far beyond what a colour map or a
log-y axis resolves; it halves the file. *k axes and σ(k)* → **float64**: these are read as
numbers (k\* is their argmax, and σ near onset is ~1e−3 against a J of order 1), so they keep
the precision they were computed in.

**MEASURED size, and why it is tracked.** Per-run npz, real runs on `three_gene_multiL`
(96×96, N=3): **110 KB** for a short serial run with no history and no model field;
**118 KB** batched, 8 restarts, 60 steps; and **645 KB** for the full phase-C shape — the
tracked `m3_registry_20260730_013119` (64 restarts, 400 steps, `history_every=10`, both fields
present, `plot_arrays_bytes = 644886`). Of its 806 KB uncompressed content the training
trajectory is **570 KB** and the two fields **221 KB**; float32 mantissas barely compress, so
this is not an estimate that will improve. Projected for a phase-C wave of ~96 runs (8 seeds ×
6 targets × 2 forms): **≈62 MB**, plus **0.82 MB per cross-L evaluation** (5 L × 3 × 128 × 128
float32, measured on the tracked `m3_registry_20260730_005701` evaluation).

That is tens of MB, not GB, and no single run is multi-MB — so `.gitignore` now **tracks**
`experiments/**/arrays/*.npz` (it previously ignored `experiments/**/*.npz` wholesale, which
cost nothing only because nothing wrote one). Two levers exist if a wave needs to be cheaper,
both explicit rather than silent: `solver.save_plot_arrays=false` (no npz at all) and
`train.history_every` (the trajectory is 70% of the file; stride 20 halves it). The
per-run byte count goes on the run row as `plot_arrays_bytes`, so the cost stays auditable.

**Figures are NOT tracked** (`experiments/**/figures/`, `*.png` stay ignored): they are
redrawn from the arrays in seconds by `scripts/make_figures.py`. The arrays are the artefact.

**The one guard that matters, carried over.** A rollout that did not pattern writes **no**
`model_field` key, and `meta.rollout_status` says why. The project already recorded one false
`morphology_match` from classifying decayed float noise (see `train._morphology_rollout`);
saving that noise as a field would re-open the same hole from the plotting side.
`tests/test_plot_arrays.py::test_unpatterned_rollout_gets_no_model_field` pins it.

**Firewall.** `plotdata.py` reads the answer key's (J, D) — passed in as plain numpy by
`train.fit` — so it is SCORING side. It imports nothing truth-side, and
`tests/test_plot_arrays.py::test_recovery_side_does_not_import_the_scoring_side_writers`
audits statically that no recovery-side module imports it, mirroring `tests/test_firewall.py`.

**Where it lives:** `src/rngrn/plotdata.py`; `src/rngrn/train.py` (`_save_run_arrays`);
`src/rngrn/config.py` (`solver.save_plot_arrays`); `.gitignore`;
`scripts/make_figures.py`; `tests/test_plot_arrays.py`.

### D-PLOT-2 — the training trajectory is recorded at stride 10, for ALL members, at float32

**Date:** 2026-07-30. **Status:** DECIDED, with the stride flagged as a SIZE tradeoff rather
than a scientific one. **Decided by:** the implementing agent; the default stride of 10 was
the owner's suggestion and is adopted with its cost measured.

**The problem.** `recover()` kept one summary row per RESTART and nothing per step. A run
that reached its loss by step 100 and one that crawled there by step 1900 left identical
records, the weights actually in force were never written down (and `DataFirstStaging` ramps
`turing` from 0 across the first 50% of the budget, so a loss curve read without them is
misleading), and the D trajectory — whether recovery reaches biologically plausible
diffusivities is the live question behind D1/D3 — was invisible.

**The decision.** `history.TrainingHistory` records, per recorded step and per **member**:
every loss term (`L_*`) and the total, every weight in force (`w_*`), the **constrained
physical** parameters KA/KR/α/δ/β/D (36 columns at N=3 — *not* raw θ, which is a link-function
artefact), the derived D-ratio, the per-step diagnostics (`kstar_model`, `sig_max`,
`sig_max_pos`, `rel_err`), and `ss_converged`. Both the serial and the batched path are
instrumented, since phase C runs batched on CUDA.

**Stride 10, with the cost measured.** One record per step × 64 restarts × 36 parameters is
~921k floats per 400-step run (**≈3.7 MB** at float32) — not affordable across a 96-run wave.
Stride 10 measures **≈0.51 MB** per run (S=41, B=64, Q=17, P=36, float32, measured by
`savez_compressed`), and still puts 21 recorded points across the staging ramp at
`adam_steps=400`. **Step 0, the last training step, and the post-training evaluation step are
recorded unconditionally whatever the stride**, so no curve's endpoints are interpolated.
`train.history_every=0` disables the recorder entirely.

**ALL members are kept, thinned — never member 0 relabelled as "the run".** On the batched
path B members are optimised jointly and the reported result is the **best** member, whose
index is only known after the last step; recording one lane during training would either be
the wrong lane or require guessing. `meta.history_best_member` records which lane won, so a
figure can plot the winner without the trace having pre-committed to it. The offered
alternative (full trace for the winner plus min/median/max across the rest) was **not**
taken: it costs a second pass to identify the winner and would have discarded the
restart-to-restart spread, which is itself the answer to "is this recovery reproducible".

**A member that dies is an EVENT, not a gap.** A restart whose steady state diverges is
abandoned; its lane becomes NaN from that step on, and `hist_death_step` records the step it
died at. Lanes are never reindexed, so the seed-to-lane mapping stays auditable.

**float32, and one measured performance note.** The trajectory is plotted, not
differentiated. On CUDA the first draft read parameters per (member, parameter), which is
6·B synchronising device→host copies per recorded step (15 744 for a 400-step, 64-restart run
at stride 10); `history._param_block` reads each parameter once for the whole batch instead —
6 copies per recorded step. The loss/diagnostic entries of a batched `parts` dict are already
numpy (`losses/terms._np`), so they cost no extra sync.

**Where it lives:** `src/rngrn/history.py`; `src/rngrn/recover.py` (both loops);
`src/rngrn/config.py` (`train.history_every`); `src/rngrn/export.py` (`tidy_history`);
`tests/test_plot_arrays.py`.

### D-PLOT-3 — the cross-L table is persisted through the run-index machinery; exports are TIDY/long

**Date:** 2026-07-30. **Status:** DECIDED. **Decided by:** the implementing agent.

**The problem, twice over.** (1) `eval.lgen_eval.evaluate_across_L` returns a rich per-L table
and `cli.cmd_evaluate` **printed it to stdout and stored nothing** — so the measurement in
`docs/LGEN_TRANSFER_FIRST_RESULT.md` could not be re-plotted without re-running a 9-minute
recovery. (2) The indexes are one row per run with ~120 columns, which is a record, not a
plotting frame: drawing "outcome vs hyperparameter" from a wide row requires knowing which
columns are identifiers and reshaping first.

**The decisions.**

- **Cross-L results go through the SAME machinery** `optim/target_report.py` uses —
  `index.open_index(runs_root, <table>, backend)` — as **flat scalar** rows built with
  `dict.update()`: one row per (run_id, L) in table `lgen_eval`, one summary row per
  evaluation in `lgen_summary`. Both carry `run_id` and `git_sha` so a row joins back to the
  run that produced the model and the code that evaluated it. Two tables rather than one
  mixed table with a `row_kind` discriminator, because a tidy export of a mixed table would
  have to filter before it could be plotted. The per-L **fields** go to
  `<run_dir>/arrays/lgen_fields.npz` (`evaluate_across_L(keep_fields=True)`), one key per L
  because under `grid_rule='constant_dx'` the grids differ between L and cannot be stacked.
  Verified by re-running the tracked `m3_registry_20260730_005701` evaluation: every number in
  `docs/LGEN_TRANSFER_FIRST_RESULT.md`'s transfer table reproduced exactly
  (`kstar_phys_cv` 0.047636, `periods_slope_rel_err` 0.014345,
  `morphology_class_preserved` 1.0, `min_pixels_per_wavelength` 11.055).
- **`rngrn export` writes LONG/TIDY csv** — `runs_tidy.csv`, `target_reports_tidy.csv`,
  `lgen_tidy.csv`, `history_tidy.csv` — as `<identifier columns…>, variable, value,
  value_num`: **one observation per row**. `value` keeps the verbatim text (so a categorical
  observation like `morphology_pred` survives) and `value_num` the numeric reading, empty when
  there is none; booleans are numeric 0/1 *and* text. The **hyperparameters stay as identifier
  columns** (`export.RUN_ID_COLS`), which is exactly what makes an outcome-vs-hyperparameter
  plot a group-by rather than a reshape; a sweep already writes one run row per cell
  (`optim/sweep.py` calls `fit()` per point), so a sweep axis is a column like any other.
  Stdlib `csv` only — a tidy export must not be the reason a plotting environment needs
  pandas. `tests/test_plot_arrays.py` asserts the tidiness directly: every record has exactly
  one variable/value pair and (identifiers, variable) is unique.
- **Missing hyperparameter columns were added to the run row**: `seed` and `model_seed` (their
  absence made a seed-replicate sweep — this project's standard design — impossible to
  disaggregate from the index alone), `adam_lr`, `lbfgs_steps`, `grad_clip`, `tau`,
  `jac_floor`, `dratio_centre`, `dratio_spread`, `ratio_update_every`, the remaining loss
  weights, `n_grid`, `history_every`, `git_sha`, `plot_arrays_bytes`.
- **`history_tidy.csv` defaults to the winning member only** (`--history-members best`), which
  is what a learning curve "for this run" means and keeps the file at ~S·(Q+P) rows instead of
  ~112k for one 400-step, 64-restart run. `--history-members all` exports every recorded
  restart. Nothing is dropped from the npz either way — the arrays always hold every member;
  the flag chooses what the CSV projects, and it is documented in `--help`.

**Where it lives:** `src/rngrn/plotdata.py` (`lgen_rows`, `save_lgen_fields`);
`src/rngrn/cli.py` (`_persist_lgen`, `cmd_export`); `src/rngrn/export.py`;
`src/rngrn/eval/lgen_eval.py` (`keep_fields`); `tests/test_plot_arrays.py`.

---

## Part 2b — Stage 0: the biological-viability precondition (unit c-bioviab, 2026-08-03)

Decisions taken while measuring whether `configs/bio_box.yaml` is Turing-EMPTY — the
feasibility precondition behind `docs/PREREGISTRATION.md` §3.4. Results in
`docs/BIO_VIABILITY.md`; the D-EVID-5 entry below belongs with Part 1's evidence-integrity
defects and is placed here only to keep the unit's entries together.

---

### D-EVID-5 — `loss.weights.param_prior` was a NO-OP: the biological prior was never in the objective

**Date found:** 2026-08-03 (unit c-bioviab). **Fixed:** 2026-08-03, same unit.
**Status:** SUPERSEDED (defect; now fixed).

`losses/terms.py::param_prior` existed, was tested (`tests/test_plausibility.py`), had a
weight in `config.py` (`loss.weights.param_prior`, default 0.0), had its own knobs
(`loss.dratio_centre`, `loss.dratio_spread`, `loss.bio_box_path`) and had a column on
every run-index row (`w_param_prior`, `dratio_centre`, `dratio_spread`, written by
`train.py`) — **and no caller on the path recovery runs.** `losses/total.py`'s
`compute_terms` / `compute_terms_batched` — the assemblers `recover.py` actually calls —
never referenced it. Its only caller was `losses/terms.py::composite_loss`, which the
module's own docstring labels "the standalone reference form" and which `recover.py` does
not use. `train.py` also never passed `cfg.loss.dratio_centre` / `dratio_spread` /
`bio_box_path` to `recover()`.

Consequence: a run configured with `loss.weights.param_prior = 1.0` would have trained
with **no** biological prior, recorded `w_param_prior: 1.0` on its index row, and said
nothing. Any future claim of the form "recovery under the biological prior does X" made
before this fix would have been measuring the prior-OFF arm. No such run exists in the
index yet (`w_param_prior` is 0.0 on every recorded row), so no published number is
affected — this was caught before it produced evidence, not after.

A second, smaller defect found in the same read and **deliberately left alone**:
`composite_loss` computes `loss` twice, and its second assignment drops the
`w['anchor']*L_s` term that the first included. It is a standalone reference form with no
caller in `src/`, so fixing it is out of this unit's scope; recorded here so it is not
rediscovered as new.

**Fix:** `compute_terms` / `compute_terms_batched` take `param_prior_kw`; `None` (the
default) omits the key from `term_vals` entirely, so `strategy.combine`'s
`sum(weight * term)` never sees it and every previously recorded number is reproduced
bit-for-bit (`tests/test_plausibility.py::test_param_prior_kw_none_leaves_the_objective_byte_identical`).
`recover.py` builds the kwargs from the strategy's base weight, exactly as it already
decides `compute_resid`, and **raises** if the prior is asked for with an adaptive
(non-static) strategy, which would re-weight the prior by its own magnitude. A batched
twin `terms.param_prior_batched` was added (the serial one reads only `model.D/.alpha/
.delta`, all of which carry the leading `B`) and is pinned equal to the serial term member
by member. `configs/base.yaml`'s `weights` dict gained a `param_prior: 0.0` row — value
unchanged, but without it `apply_overrides` fails loud on an unknown key and
`-o loss.weights.param_prior=...` was unreachable from the CLI.

**Where it lives:** `src/rngrn/losses/total.py`, `src/rngrn/losses/terms.py`
(`param_prior_batched`), `src/rngrn/recover.py`, `src/rngrn/train.py`,
`configs/base.yaml`, `tests/test_plausibility.py` (4 new tests).

---

### D-EVID-6 — the pre-push hook tested a DIFFERENT worktree's source

**Date found:** 2026-08-03 (unit c-bioviab). **Fixed:** 2026-08-03, same unit.
**Status:** SUPERSEDED (defect; now fixed).

`.githooks/pre-push` falls back to `../turing-training/.venv/bin/python` when the current
worktree has no `.venv` of its own — which is every worktree. That venv was created by an
**editable install**, which writes `turing-training`'s absolute `src` path into a `.pth`
file. So from any other worktree the hook ran `pytest` against
`../turing-training/src/rngrn`, **not against the source being pushed.**

**MEASURED, not inferred.** The four tests added in this unit for the `param_prior` wiring
FAILED under the hook and PASS under `PYTHONPATH=$PWD/src` — because the hook was exercising
a tree that does not contain the change. Directly:

```
hook would import: .../worktrees/turing-training/src/rngrn
with PYTHONPATH  : .../worktrees/c-bioviab/src/rngrn
```

The failure mode is worse than a false red: a worktree whose change BREAKS something that
`turing-training/src` does not contain would have been green-lit. This is the same class of
defect as D-EVID-2 — a hook that reads as protection while protecting nothing — and it was
active for every worktree push since the sibling-venv fallback was added.

**Fix:** the hook now exports `PYTHONPATH="$repo_root/src"` before running pytest
(PYTHONPATH takes precedence over `.pth` entries) and PRINTS the resolved `rngrn` path, so
which tree was tested is visible in the push output instead of having to be inferred.

**Where it lives:** `.githooks/pre-push`.

---

### D-BIO-1 — `beta` for the box-viability sampler is the GENERATOR's own basal draw, and is never scored

**Date:** 2026-08-03. **Status:** DECIDED (scope: the Stage-0 sampler only).

`configs/bio_box.yaml` marks `beta` `source: UNCITED`, and `plausibility.py` never scores
an UNCITED row. But a value is still needed to *build* a model, so one had to be picked.
Picked: `beta ~ 10**U(-2, -0.3)` (0.01–0.5), which is `scripts/gen_tg3.py:157`'s own basal
draw `b = 10**rng.uniform(-2, -0.3, 3)` — i.e. calibrated against the data-generating
process this pipeline actually fits, not invented. **Rejected:** inventing a bound and
putting it in `bio_box.yaml`; and `docs/STATE_OF_THE_SCIENCE.md` §10's beta ~1e-4–1e-2
range, which that document explicitly marks as an init-distribution search result "not a
neutral prior" and "deliberately not adopted".

Sensitivity is measurable rather than assumed: `--beta-mode {generator,low,high}` reruns
the whole part-1 measurement at fixed beta = 1e-3 and beta ≈ 0.5.

**Where it lives:** `scripts/stage0_bio_viability.py::BETA_MODES`; `docs/BIO_VIABILITY.md`.

---

### D-BIO-2 — the D vector is sampled under TWO paired schemes, because the near-immobile node is an INTENDED escape valve

**Date:** 2026-08-03. **Status:** DECIDED (scope: the Stage-0 sampler only).

`scoring/plausibility.py::d_ratio_of` takes the ratio of the two MOST MOBILE species
deliberately, so a near-immobile third node never enters the scored `d_ratio` — the
mechanism `docs/ROBUSTNESS_MEASUREMENT.md` §4.4 measures as keeping 127/127 generator
systems strictly Turing when the slowest diffuser is immobilised. A sampler that ignored
that would answer a different question from the one the box asks. So D is drawn under two
schemes, both satisfying the `d_ratio` row by construction and reported separately:

* `mobile3` — all three species genuinely mobile: `D_lo ∈ 10**U(-1,0) × D_mid`.
* `immobile` — a near-immobile third node: `D_lo ∈ 10**U(-4,-2) × D_mid`.

Both draw from the SAME rng stream in the same order under a common seed, so they differ
in the slowest D and in nothing else: a PAIRED comparison. The absolute D scale is fixed
at `D_max = 1` because it is a pure length rescale (`k* ∝ 1/√D`) that part 3 absorbs into
its choice of L.

**[UNCALIBRATED]** the two decade windows (−1..0 and −4..−2) are a reasoned separation of
"mobile" from "near-immobile", not a measured boundary. What they support is the
*comparison* between the arms, not an absolute rate.

**Where it lives:** `scripts/stage0_bio_viability.py::D_SCHEMES`, `draw_batch`.

---

### D-BIO-3 — the Turing verdict is the STRICT test, on a LOG k-grid to 10^3.5

**Date:** 2026-08-03. **Status:** DECIDED.

Stability is `max Re eig(J) < 0`, not `tr(J) < 0`. `docs/ROBUSTNESS_MEASUREMENT.md` §3
measured the trace test overcounting by up to 70 % of draws, and the robustness fix already
moved to strict for that reason; `eval/analysis.py::turing_ok` still uses the trace test, so
its verdict is reported here only as `turing_loose`, never led with. **Measured in this
unit:** on 80 000 in-box draws the loose criterion accepts 1215 and the strict criterion 19
— the trace test overcounts by a factor of ~64 on this population, because every one of its
extra "successes" is a UNIFORMLY unstable state (`max Re eig(J) > 0`) whose dispersion peaks
at k → 0. That is not a Turing pattern; it is a homogeneous instability.

The k-grid is `[0] ++ logspace(-3, 3.5, 500)`, not `eval/analysis.py`'s
`linspace(1e-3, 50, 4000)`: with a near-immobile node (D ~ 1e-6 × D_max) the branch that
node dominates only turns over at `k ~ √(|J|/D_lo) ~ 10^3`, so a grid stopping at k = 50
cannot see whether it grows there. k = 0 is prepended and EXCLUDED from the instability
search so the uniform mode can never be reported as the growing one.

A draw whose steady state does not converge is a FAILED draw: it stays in the denominator
with `converged=False` and is never silently dropped.

**Where it lives:** `scripts/stage0_bio_viability.py::KGRID`, `turing_verdict`,
`turing_verdict_batch`.

---

### D-BIO-4 — the box-constrained sampler is BATCHED, and pinned against the serial reference

**Date:** 2026-08-03. **Status:** DECIDED.

The serial path costs ~55 ms per draw (model construction + `losses/terms.steady_state`),
so 80 000 draws would be 75 min. Batched through `BatchedRNGRN` + `steady_state_batched` +
batched numpy `eigvals` it is ~130 draws/s. Because a fast path that quietly disagrees with
the reference would invalidate every rate in this unit, part 1 re-runs 32 draws through the
serial `RNGRN` + `steady_state` + serial verdict and compares. **Measured:** 32/32 verdicts
agree; worst `|Δ max Re eig J|` = 4.27e-06, which comes from the two solvers legitimately
stopping at different residuals (the serial relaxation fallback accepts `|f| < 1e-4`, the
batched Newton reaches ~1e-10 on the same root). A verdict disagreement, or a deviation
above 1e-4, is fatal and aborts the run.

**A note on the artefact's size, since it was weighed rather than ignored.**
`part1_box_draws_generator.npz` is **25 MB** — 80 000 draws x 39 float64 parameters. That is
the single largest tracked artefact in the repo. It is kept whole because §1's rates are the
headline claim of this unit and D-PLOT-1's standing rule is that the arrays behind a claim
are versioned with it; 25 MB also sits inside the 10-62 MB per phase-C wave that rule already
accepts. Two cheaper options were considered and rejected: storing the parameters at float32
(halves it, but then the stored draws are not the draws that were evaluated), and storing
parameters only for the 19 survivors (the rest are exactly regenerable from the recorded
seed, but every consumer would then need an index-remapping layer for a saving of 23 MB).
Recorded so the choice is visible; if the repo later needs the space, regenerating from
`seed=0, n=20000, beta_mode=generator` reproduces the file exactly.

**Where it lives:** `scripts/stage0_bio_viability.py::part1` (the equivalence block).

---

### D-BIO-5 — the mu sweep varies the GEOMETRIC MEAN of delta, and separately one species at a time

**Date:** 2026-08-03. **Status:** DECIDED.

`delta` is a vector but the box row bounds every species, so there is no single "the delta".
Two parameterisations, both reported:

* **uniform** — hold the base draw's per-species delta *pattern* and rescale it so its
  geometric mean equals the swept value `d`. Individual `delta_i` can then sit outside
  [0.4, 5] even when `d` is inside it, so `all_delta_in_box` is recorded per point.
* **per_species** — move ONE `delta_i` and hold the others at the base draw.

Range: 41 log-spaced points over [0.04, 50], i.e. the cited box row [0.4, 5.0] plus one
decade either side, so the boundary is located relative to the box edges rather than
asserted to be outside them.

**Useful fact, worth stating explicitly:** the generator draws `mu = 10**U(-0.4, 0.7)`
= 0.398…5.01 (`scripts/gen_tg3.py:159`) and the box row is [0.4, 5.0]. These are the SAME
range to two significant figures, so unlike the D-ratio axis (generator ~8–250 vs box
[1, 60]) the degradation axis is one where the box does NOT fight the data.

**Where it lives:** `scripts/stage0_bio_viability.py::SWEEP_D`, `_sweep_one`.

---

### D-BIO-6 — the rollout domain is L = 6 × 2π/k* on a 64×64 grid

**Date:** 2026-08-03. **Status:** DECIDED.

Six full wavelengths of the fastest-growing linear mode across the box. Six is the
generator's OWN original choice — `scripts/gen_tg3.py:66` records "The original generator
hard-coded 6 periods per box" — and sits inside its replacement acceptance window
`PERIODS_CHOICES = {3..14}`, so these figures use the same domain size the training data was
built at. 64×64 is the owner's "small grid" and costs 0.54 s a field (`eval/rollout.py`,
measured, `etdrk4_rfft`, `OMP_NUM_THREADS=1`, N=3, 609 steps).

`stopped_reason` is recorded and printed on every panel: a run that ended on `step_budget`
was TRUNCATED and its field is not a statement about the attractor, so it is labelled as
such rather than shown as a pattern. `sig_max` is the SIGNED dispersion maximum.

**Where it lives:** `scripts/stage0_bio_viability.py::PERIODS`, `GRID_N`, `part3`.

---

### D-BIO-8 — part 3 shows the channel with the LARGEST spatial std, and records BOTH pattern verdicts

**Date:** 2026-08-03. **Status:** DECIDED.

`eval/rollout.py::simulate` defines `amplitude` and `patterned` on **channel 0 only**. That
is right for a real dataset, where channel 0 is the observed channel. It is wrong for the
Stage-0 sampler, and by the sampler's own doing: `draw_batch` shuffles the species order
(nothing distinguishes the species except which one drew the small D), so channel 0 is
arbitrary. **MEASURED:** a competitive/`mobile3` survivor has channel-0 std **2.0e-4** and
channel-1 std **0.76** — the library calls it unpatterned while a large labyrinth sits one
channel over. Reading channel 0 gave 4/12 patterned; reading any channel gives **9/12**.

Decided: each panel shows the channel with the largest spatial std and NAMES it in the
title, with all three channel stds listed beside it. Both verdicts are recorded and neither
overrides the other — `patterned` stays the library's channel-0 number, untouched, and
`patterned_any` is the new any-channel one. **Rejected:** changing `eval/rollout.py`'s
definition, which is correct for its actual callers; and quietly reporting the any-channel
number under the library's name.

**Where it lives:** `scripts/stage0_bio_viability.py::part3`,
`scripts/stage0_figures.py::fig_pattern_gallery`.

---

### D-BIO-9 — the sweep grid always contains the base draw's own delta, and the harvest is round-robined across D-schemes

**Date:** 2026-08-03. **Status:** DECIDED. Both are corrections to defects found by
dry-running the pipeline before the real data landed.

1. **The base point is inserted into the sweep grid.** The 41-point log grid does not
   generally contain the base draw's own `delta`, so a system harvested BECAUSE it is
   strictly Turing could report `0/41` strictly Turing — a curve missing the one point it is
   known to pass through. `_sweep_one` now adds the base value and marks it (`is_base`,
   drawn as an open diamond).
2. **Survivors are round-robined across the D-schemes.** `survivors(mobile3)[:n] +
   survivors(immobile)[:n]` truncated to `n` yields `n` mobile3 and ZERO immobile whenever
   mobile3 has `n` of its own — which is exactly what happened on the first part-2 run, and
   would have made every curve and panel one scheme's anecdote, the precise thing the two
   schemes exist to prevent. `balanced_picks` alternates and falls through when one pool
   empties. The re-run gives 3 `mobile3` / 3 `immobile` per form.

**Where it lives:** `scripts/stage0_bio_viability.py::_sweep_one`, `balanced_picks`.

---

### D-BIO-7 — the binding parameters (KA/KR) are the FREE directions and are sampled over four decades

**Date:** 2026-08-03. **Status:** DECIDED, marked **[UNCALIBRATED]**.

`configs/bio_box.yaml` has no row for the binding budget `s` or the gate `g` at all, so
`plausibility_score` is blind to them: they are the directions in which the Turing region is
free to live. The sampler draws `s ~ 10**U(-2, 2)` per entry and the gate logit
`~ N(0, 2.5)` (the wide, sub-saturated split of `model.py::_low_basal_raw_params`). The
four-decade window is chosen to bracket the generator's own Hill thresholds — `gen_tg3.py:160`
draws `K = 10**U(-1, 0.6)` with `n ∈ {2,3,4}`, so the comparable binding scale `1/K^n` spans
roughly 1e-2…1e2 — but the mapping between a Hill threshold and this model's `KA` is not an
identity, so this is a bracketing argument, not a calibration. Marked UNCALIBRATED rather
than presented as derived.

This matters for how the acceptance rate is read: it is the acceptance rate **under a stated
prior**, and a wide prior on an unscored direction lowers it. Figure `f1` panel (d) reports
where the survivors actually sit in `s`, which is the part of this that is a measurement.

**Where it lives:** `scripts/stage0_bio_viability.py::S_LOG_RANGE`, `GATE_LOGIT_SD`.

---

## Part 3 — Reconciling `docs/CODE_REALITY.md` §11 ("open decisions currently blocking progress")

`docs/CODE_REALITY.md` §11 lists six open decisions as of the phase-A merge. Reconciled
against Part 2 above:

1. **"Which reading of 'more robust'"** (`GOAL_tica_equivalent.md` §2.1, four
   incompatible readings A–D). **Still OPEN.** No entry in this register settles it;
   `GOAL_tica_equivalent.md` §2.1 itself states "The project has not chosen between
   them" and "Under the project's autonomy rule this is a science decision, not a
   mechanical one." Nothing found in the 13 merged PRs picks one.

2. **"Domain-size design for regenerated data"** — **SETTLED.** By unit 11/12
   (D6 above): option (b), vary L with random periods-per-box. Evidence: the
   image-blind predictor going from 0.0% median error on the old 127 samples to
   45.5%/45.0% on the two new datasets. `docs/CODE_REALITY.md` §11 item 2 already
   marks this struck through as settled; this register's D6 is the fuller record.

3. **"The morphology pass condition"** — stripes classifies at 33.3% on held-out data
   (3 samples), and the options (balanced accuracy excluding stripes / continuous
   `morphology_distance` / generate more stripes first) are unchanged. **Still OPEN.**
   Nothing in the 13 merged PRs picks a pass condition; PR 13 (unit 7) made the
   morphology *rollout* affordable and wired it into every run
   (`morphology_scored: "compared"` now reachable), which makes the metric
   *measurable* but does not decide what counts as a pass. PR 13 separately raised its
   own OPEN METRIC DECISION (whether an unpatterned recovery should record
   `morphology_match=False` or stay `"target_only"`) — related but not identical to
   this item; also unresolved.

4. **"D-ratio prior centre — ~7.5 vs ~100"** — **DECIDED for the bio_box prior; the
   underlying tension is explicitly left open.** See D1 above: the user decided 7.5
   for `configs/bio_box.yaml`, reversing an earlier in-session ~100 choice. But
   `docs/STATE_OF_THE_SCIENCE.md` §11 still frames this as an "open design choice"
   at the document level, because 7.5 is a decision about the *plausibility box*
   specifically, not a resolution of whether training itself should ever target ~100.
   Recorded as decided-with-a-caveat rather than fully closed, per the task's
   instruction not to overclose a tension the source material calls deliberate.

5. **"Whether to adopt the low-basal init"** — **Still largely OPEN, but now with
   training-time evidence against adopting it as the default.** See D9: PR 9 measured
   that low_basal fails 40/40 restarts under actual training on the one dataset
   tested, versus `default` succeeding under the same budget. The *default* stays
   `"default"` and this register treats that as effectively answered by the evidence
   (do not flip the default) even though no one has stated "we will never adopt this."
   The knob exists, is unwired from the CLI path, and adopting it would require first
   explaining or fixing the training-time failure mode.

6. **"Whether `split_hinges` and the frame-scale anchor are promoted into the
   library"** — **SETTLED.** By unit 1/PR 11 (D8 above): promoted, bit-identical to
   the validated originals (max diff 0.000e+00 over 20 seeds), library Turing fraction
   confirmed at 38/40 converged, 14 Turing, turing_frac 0.3684 over 40 seeds × 400
   steps.

**Summary:** of the six, **items 2 and 6 are settled**; **item 4 is decided for its
narrow scope (the bio_box prior) but its wider tension is deliberately still open**;
**items 1, 3, and 5 remain open** — 5 now has training-time evidence pointing away
from adopting the alternative, but no one has made that a formal decision.

---

## Sources consulted

- `gh pr view` for PRs 1–13 (`ben1ahrens/rngrn-pipeline`, all merged 2026-07-29).
- `docs/CODE_REALITY.md` (§4, §11, and the evidence-integrity items cross-checked
  against `.githooks/pre-push` and `src/rngrn/cli.py` directly).
- `docs/STATE_OF_THE_SCIENCE.md` §1, §7, §8, §10, §11.
- `docs/GOAL_tica_equivalent.md` §2.1, §2.2.
- `docs/ROBUSTNESS_MEASUREMENT.md` §4.
- `TUNING.md` (Stage 2 and the "Promoted from the experiments" / "Morphology rollout"
  sections).
- `configs/bio_box.yaml` (per-parameter `source:` fields, read directly).
- `src/rngrn/losses/weighting.py`, `src/rngrn/cli.py`, `.githooks/pre-push` (read
  directly to confirm the two evidence-integrity fixes are actually in the code, not
  only claimed in a PR body).

---

### D-PERF-1 — an outer CONVERGENCE CHECK on the loss would be fast and WRONG

**Question asked (owner, 2026-08-03):** "would the code run faster if we had a convergence
check?" `recover.py:226` and `:474` run `for step in range(adam_steps)` with no early exit,
so the full budget is always spent. The inner steady-state Newton already has convergence
checks and an early exit; the outer Adam loop has none.

**Measured, from 18 committed runs' `hist_scalars` (no new compute):**

| quantity | median | p90 | max |
|---|---|---|---|
| step reaching 99 % of total loss drop | **30** | 180 | 399 |
| step reaching 99.9 % of loss drop | **35** | 190 | 400 |
| last step any restart flips Turing state | **160** | — | **270** |

**The decision: do NOT add a loss-based convergence check.** The loss is done by step ~30,
but restarts are still flipping into and out of the Turing regime until step 160–270. A
loss-based early stop would fire at ~35, look like a 10x speedup, and silently change the
Turing rate — the criterion the project actually scores. It is a concrete instance of the
standing rule that the lowest-loss configuration is not the best recovery.

**A check on Turing-state stability instead would be sound but modest**: ~100 quiet steps
after the last flip puts the stop at 260–370 of 400, i.e. **25–35 %**, not a transformation.

**And it does not touch the real cost.** The ~30x per-target spread occurs at IDENTICAL step
counts, so it is per-step cost in the steady-state solve, not step count. The correlation
runs the wrong way for an early stop: legacy 174/512 restarts @ 370 s, qvar/sample_0003
31/512 @ 1024 s, qvar/sample_0000 2/512 @ 1275 s — monotone in both. The expensive targets
are the non-patterning ones, where the optimiser thrashes and a convergence check fires
LEAST. It would accelerate the cheap runs and leave the expensive ones untouched.

**Where the real win is, recorded but NOT taken:** 510 of 512 restarts never reach the
Turing regime and burn the full budget. Culling provably-dead restarts (sig_max flat and
negative for N steps) would cut cost on exactly the expensive targets, and the hook already
exists — `hist_death_step` is in the history arrays and is non-finite for all 64 members,
i.e. nothing ever dies. NOT taken here because it changes WHICH restart wins: a science
change needing its own measurement, and it would break comparability with every cell
measured so far.

**Cheaper unexplored alternative:** `train.adam_steps` has never been swept. Given 99.9 % of
the loss drop lands by step 35 and the last Turing flip by 270, a 400-step budget may
already be generous, and measuring 200/300/400 is a config-only answer to "can this run
faster" that requires no code change and no comparability argument.

---

### D-C1-GAUGE — the J-degeneracy of the objective, and what it does and does not explain
*Decided by unit C1 (competitive tuning), 2026-08-03, under the threshold-setting authority
recorded in `PREREGISTRATION.md` §0. Evidence: `scripts/c1_gauge.py`,
`docs/C1_COMPETITIVE_TUNING.md` §8, computed on 24 already-committed runs.*

**The finding.** On the region the optimiser occupies (‖J‖_F > 1, true for all 24 runs, where
the `anticollapse` hinge is exactly flat), the trained objective is a function of
σ(k) = max Re eig(J − k²D) alone. Since D is diagonal, that makes it **exactly** invariant
under (i) transpose, (ii) diagonal similarity J → SJS⁻¹ (a 2-parameter continuous gauge at
N=3), and (iii) node permutation. Verified numerically at max |Δσ| ≈ 1e-15 on real recovered
(J, D). A dimension count (σ(k) fixes 7 independent functions of J's 9 entries) says these
are the whole of the blindness, not a subset.

**What was decided, and it is a negative.** This is **not** adopted as the explanation of the
criterion 3.1 failure, because quotienting by the full group does not rescue the statistic:
raw → +perm → +transpose → +balance is 0.125 → 0.375 → 0.375 → 0.375 on the legacy control
and 0.125 → 0.125 → 0.125 → 0.375 on `baseline`, against a bar of 0.75. The tempting
conclusion — "the seeds found the same network in different gauges" — is measured to be
false and is recorded as false so it is not re-proposed.

**What is adopted, as an instrument limitation rather than a threshold change.**
`topology_consistency`'s hard cut at rtol·max|J| lands on the **median** entry magnitude on
`baseline` (q50 = 0.0512 against rtol 0.05), with 33 % of entries within a factor 3 of it,
and the statistic wanders non-monotonically over 0.125–0.500 as rtol is swept 0.005→0.5.
`PREREGISTRATION.md` §3.1 already flagged the tolerance UNCALIBRATED and required
0.02/0.05/0.10 to be reported; this quantifies it. **No bar moves and the raw modal fraction
at rtol 0.05 remains the number read against 0.75** — the criterion fails at every tolerance
in the sweep, so nothing here converts a failure into a pass. The consequence is for the
paper's wording and for any *future* instrument, not for this pre-registration.

**Left explicitly OPEN.** Whether a gauge-invariant, magnitude-weighted reproducibility
statistic should replace the entrywise one. It must not be decided now: inventing a metric
after seeing that the current one fails is exactly what `PREREGISTRATION.md` §5 forbids. If
it is ever adopted it goes in a new dated section that says plainly it was added after
seeing results.

### D-C1-DIAG — `max diag(J) > 0` adopted as a secondary readout, not as a criterion
*Same unit and date. Evidence: `docs/C1_COMPETITIVE_TUNING.md` §8.5.*

A positive Jacobian diagonal entry separates patterning perfectly across all 24 committed
runs (10 Turing / 10 positive, 14 non-Turing / 14 negative, no off-diagonal cell). This is a
**necessary condition being confirmed, not a discovery** — diffusion-driven instability with
diagonal D requires it — and is recorded that way. It is adopted only as a cheap continuous
proxy for rate progress, beside the pooled per-restart `sig_max_pos` rate, because
`turing_frac` is a floored count over K = 8 and cannot rank two failing configurations. It
replaces no pre-registered criterion.

### D-C1-TURINGW — `loss.weights.turing` promoted to the lead rate axis
*Same unit and date.*

The axis is not in C1's original eight and had never been tried by this unit. C2 measured
`-o loss.weights.turing=8.0` moving `nc1` on qvar `sample_0000` from `turing_frac` 0.0625 to
0.750 at K = 8. Promoted to first position in `scripts/c1_queue4.sh` and proven LIVE on both
the serial and the batched path before any cell using it is believed, per the standing rule
that five silent no-ops have already been found in this codebase. Reordering a queue changes
no threshold and drops no cell.

### D-TDPLOT-1 — "patterned" and "periodic" are scored as two separate verdicts, and the periodicity threshold is calibrated on the corpus
*2026-08-10. Unit: training-data visual inspection (`feature/training-data-plots`).*
*Evidence: `notebooks/training_data_simulations.ipynb` §2/§2b, run over all 413 registered*
*samples; code in `scripts/td_figures.py::patterning_verdict`; tests in*
*`tests/test_td_figures.py`.*

**The decision.** The inspection notebook reports two independent booleans per sample rather
than one:

- `has_contrast` — `cv >= 0.05`, the spatial coefficient of variation of species 0. This is
  **not** a new threshold: it is the generator's own accept/reject rule
  (`scripts/gen_tg3.py:240-241` discards a simulation with `cv0 < 0.05` as "collapsed to
  homogeneous"). Reusing the identical number means the verdict carries the meaning it had
  at generation time.
- `periodic` — the RAPS peak sits at bin index `>= PEAK_BIN_MIN = 3`.

**`has_contrast` is NEARLY A TAUTOLOGY on this corpus and is recorded as such.** Every
generator here applies the same `cv < 0.05` reject rule, so "413/413 clear `cv >= 0.05`,
corpus minimum 0.0633" is what the filter does, not evidence about the data. An earlier
draft of this branch reported it under "Established"; that was wrong and is corrected. What
the check *does* establish is payload integrity — the recomputed cv matches the stored
`cv0` attribute to <= 2.2e-07 across all 413 samples. The informative screen is `periodic`.

**Naming collision, avoided deliberately.** The key is `has_contrast`, **not** `patterned`,
because `src/rngrn/eval/rollout.py:272` already owns `patterned` for a different quantity —
rollout amplitude against `max(1e-3, 0.02*|x*_0|)` — and *that* one is pre-registered
(`docs/PREREGISTRATION.md` §3.5a). The two are never comparable, and no number from this
notebook may be read against a rollout `patterned` rate.
`tests/test_td_figures.py::test_the_contrast_key_does_not_collide_with_the_prereg_patterned_name`
holds the separation.

**Why two and not one.** `cv` asks "is there spatial contrast?" and never "is the contrast
**periodic**?" A field made of a handful of isolated blobs clears `cv` comfortably while its
power spectrum decays monotonically from the lowest resolvable bin — there is no interior
peak, hence no characteristic wavelength, hence no Turing pattern. Merging the two into one
boolean would have hidden exactly the case the corpus turned out to contain.

**How the threshold was set.** The RAPS peak-bin distribution over all 413 registered
samples, measured 2026-08-10 — the FULL distribution, summing to 413:

```
bin:    1   2    3    4    5    6    7    8    9   10   11   12   13   14
count:  1   0   14   44   78  155   49   12   15   14    8    9   11    3
                                                  min=1  1st pct=3  median=6  max=14
```

> **Correction, same day.** The first version of this entry printed the table capped at
> bin 9 with the residue mislabelled `9+`. That silently dropped the 45 samples at bins
> 10–14 and summed to 368, not 413. The cap came from the notebook's own generating
> expression (`range(pb.min(), min(pb.max(), 9) + 1)`), now removed and replaced by an
> assertion that the histogram accounts for every sample. The gap-at-bin-2 conclusion is
> unaffected — everything omitted lay further *above* the threshold — but a table in this
> register that drops 11 % of the corpus is a defect regardless.

Exactly one sample sits at bin 1; **bin 2 is empty**; every other sample is at bin 3 or
above. `PEAK_BIN_MIN = 3` therefore falls inside a genuine gap in the data rather than
cutting through a populated region. Had the distribution been continuous across bins 1–3 the
screen would have been reported as unusable instead of adopted.

**"Calibrated" would be too strong a word, so the limits are recorded here rather than left
implicit** (CLAUDE.md §8):

- The threshold is set **descriptively**, from the corpus distribution. On real data its
  decision boundary has been exercised **n = 1** times: one sample is separated, 412 sit on
  the other side. That makes it a corpus-specific outlier detector, not a transferable
  periodicity criterion.
- **`PEAK_BIN_MIN = 2` gives an identical partition** on this corpus, and `4` would flag 15
  samples. The threshold is insensitive downward and brittle upward.
- `peak_bin >= 1` is **structural, not a property of the data**: `observables.raps` forces
  `power[0] = 0`, so bin 0 can never win. "min = 1" is a floor of the estimator.
- **The known-answer controls are synthetic and live in the tests, not in the corpus.**
  `tests/test_td_figures.py` passes a synthetic periodic field (must clear the screen) and a
  synthetic isolated-blob field (must fail it) through this exact code path. No *real*
  sample of independently-known periodicity was available to read it against.

**What it found.** `three_gene_qvar/sample_0032` in a domain of L = 177.8 — ~95 % of pixels
within 1 % of the field's *dynamic range* above its minimum (90.4 % within 1 % of the
minimum *value*; the earlier wording of this sentence conflated the two), species 1 and 2
flat to `cv = 0.002`, and a
monotonically decaying spectrum. The generator labelled it `morphology = 'spots'` because
`gen_tg3.classify` assigns `spots` on `area_fraction < 0.34` alone, with **no lower bound on
area fraction or component count** — a condition an almost-empty field satisfies trivially.
It is 1/413 (0.24 %) of the corpus and 1/34 (2.9 %) of `three_gene_qvar`.

**What was rejected.** (a) A "fraction of pixels pinned at the minimum > 0.6" screen: it
flags 8 samples, of which only one is the real defect. Six are legitimate dense spot
patterns — e.g. `three_gene_qvar/sample_0023`, 83 connected components with a clean spectral
peak at bin 9. The seventh is `three_gene_classical_val/sample_0007`, the marginal 4-spot
case discussed in `docs/HANDOFF_training_data_plots.md` §5: it clears the periodicity screen
at bin 4 but is *sparse*, not dense, so it is neither a defect nor a dense-spot exemplar.
The eighth is `sample_0032` itself. So the screen over-flags 7:1 and was rejected.
(b) Folding periodicity into the cv verdict — that would silently redefine a
number the generator already owns. (c) Tightening `gen_tg3.classify`'s `spots` rule: that
would change what every existing dataset's stored `morphology` attribute means, and is a
generator change, not an inspection change.

**Explicitly NOT decided here.** Whether `sample_0032` should be excluded from training. It
is reported, not acted on — dropping a sample changes what every `three_gene_qvar` number
means and is the owner's call, not this unit's.

**Scope.** This decision governs an inspection/reporting notebook only. It introduces no
recovery criterion and touches nothing on the recovery side of the firewall.

### D-CANON-2 — `stripes` is not a stable pattern class in this generator; it is largely a small-box artefact
*2026-08-10. Unit: canonical datasets (`feature/canonical-datasets`).*
*Evidence: `scripts/canon_select.py::compute_stability` over all 57 re-simulatable systems;*
*probe simulations recorded in `data/canonical_stability_cache.json`.*
*Status: **DECIDED** (the measurement); the remedy is **OPEN**.*

**This changes how existing `stripes` results should be read, so it is announced rather than
filed quietly** (CLAUDE.md §10).

**The measurement.** A system's morphology label was re-tested by re-simulating it at a
different periods-per-box and asking whether the class survives. For `three_gene_multiL` the
probe is free — those systems were already simulated at p ∈ {4,7,10,13}. For
`three_gene_qvar` it costs one extra 96×96 run per system.

- **9 of 23 multiL systems (39 %) change class when *only* the box size changes.** Every
  single flip is `labyrinth ↔ stripes`. No other pair ever flips.
- **All 5 gated `qvar` stripes candidates flip to `labyrinth`.** Zero stripes survive gates
  plus stability anywhere in the re-simulatable corpus.
- **Instability is graded by class**, over the full probed pool of 47 gated systems:

  | class | stable | flips | flip rate |
  |---|---|---|---|
  | `spots` | 27 | 1 | **3.6 %** |
  | `labyrinth` | 10 | 4 | **29 %** |
  | `stripes` | 0 | 5 | **100 %** |

  The single `spots` flip is `three_gene_qvar:24`. (An earlier draft of this entry said
  "spots never flips" — that was wrong, read off a truncated log, and is corrected here.
  The graded gradient is the stronger result anyway: it is exactly what a box-size artefact
  predicts, since `spots` is the class furthest from the anisotropy cut and `stripes` sits
  on it.)

**The mechanism, and why it is not a bug.** Anisotropy `A` is a nematic order parameter over
the dominant Fourier ring. A small box admits few orientations, so the pattern is forced onto
a near-single axis, `A` rises above 0.55, and `classify` says `stripes`. Give the same system
room and the pattern relaxes into a labyrinth. The corpus bears this out:

```
periods/box:   3-4    5-6    7-8   9-10  11-14
n:              31      5     31     26     33
median A:    0.122  0.394  0.087  0.072  0.048
% stripes:     26%    20%    16%     8%     0%
```

`corr(periods_per_box, anisotropy) = -0.312`. Stripes samples sit at **median p = 4.5**
(range 3–10) against **9.5** (range 3–14) for everything else, and **no sample at p ≥ 11 is
ever labelled `stripes`**.

**Consequences.**
1. `stripes` is **not** shipped as a canonical dataset. The canonical range p = 16..32 lies
   entirely above the regime where the class has ever occurred, so a "stripes" set generated
   there would ship the artefact, not the pattern.
2. The stored `morphology` attribute on the 49 `stripes` samples in the corpus is **partly a
   statement about their box size**. Any result stratified on `stripes` inherits that.
3. This is a second, independent reason for the already-documented weakness of the class:
   49/413 of the corpus, only 7 among the 127 samples every scorer calibration rests on, and
   33.3 % held-out accuracy. It was never a robust class, and now there is a mechanism.

**What was rejected.** (a) Lowering a gate to reach 5 stripes — the gates are what make the
strata mean anything. (b) Shipping stripes at low p anyway — it would sit at a 6–12 % k\*
precision floor against 1.6–3.1 % for the other sets, non-comparable, and would still be the
artefact. (c) Redefining `classify` to make stripes L-invariant — that would change the
meaning of the `morphology` attribute on all 413 existing samples, which D-TDPLOT-1 already
considered and rejected as a generator change rather than an inspection change.

**Left OPEN, deliberately.** Whether genuine, box-independent stripes are reachable at all in
this generator. There is a principled lever: spots-vs-stripes selection is governed by the
quadratic coefficient of the amplitude equation, and Hill kinetics are generically asymmetric
— which is the likely reason `spots` dominates at 54 % of the corpus. Screening for
near-symmetric systems is the obvious experiment and has not been run. **Nothing here claims
that real Turing stripes are unreachable**, only that *this* generator's `stripes` label, in
*this* corpus, does not survive a change of box.

---

### D-CANON-1 — two canonical datasets at 512×512, selected by margin and label stability
*2026-08-10. Same unit. Evidence: `data/canonical_selection.json`, `scripts/canon_select.py`,*
*`scripts/canon_generate.py`, tests in `tests/test_canon_{select,generate}.py`.*

**The decision.** Ship `turing_spots` and `turing_labyrinth`, 5 distinct 3-gene systems each,
at 512×512, generated once and reused for all simulated-data experiments. `stripes` is
excluded per D-CANON-2; `holes` is not shipped as its own class because only 3 gated, stable
hole systems exist, below the 5 required.

> **Corrected 2026-08-10 by D-CANON-5.** This paragraph originally said `holes` was
> "structurally unreachable… the observed channel is positively skewed by construction".
> That is **wrong**. Hole patterns are common — 7 of 57 systems produce them, several with
> negative skew, and 3 of the 5 shipped `turing_labyrinth` samples are hole patterns. What is
> unreachable is the *label*: `classify` needs `phi > 0.66` to say `holes`, which a connected
> bright matrix cannot reach because much of the matrix falls below the `z > 0.4` line. The
> corpus containing zero samples *labelled* `holes` is a fact about the classifier, not about
> the physics, and this entry read it the wrong way round.

**The eligible pool is 57 systems, not 413.** Re-simulating at a new resolution needs the
generating kinetics *and* the simulation seed. Only `three_gene_qvar` (34 systems) and
`three_gene_multiL` (23) carry both. The 127 legacy `three_gene_*` samples carry neither —
their generator lived in a gitignored tree — and the 160 `*_classical_*` samples carry
kinetics but no seed and are not 3-gene GRNs.

**Selection rule**, deterministic and seeded: admission gates (`peak_bin ≥ 3`, `cv ≥ 0.30`,
positive class margin) → rank by distance from the class boundary → require the label to
survive a probe at a different box size → prefer already-burned systems for tuning slots.

`peak_bin ≥ 3` is **load-bearing, not cosmetic**: ranking by margin alone puts
`three_gene_qvar/sample_0032` *first* among spots, because its area fraction of 0.032 gives
it the largest possible distance below the 0.34 cut — and that sample is the one confirmed
non-Turing frame in the corpus (D-TDPLOT-1). Without the gate, the worst sample available
would have become a canonical exemplar.

**Why 512 and not 1024.** Recovery is nearly pixel-count-free — every loss term lives on
N×N objects, not on the image — so a large frame is cheap to *fit*. The cost is the post-hoc
ETDRK4 morphology rollout, whose step count is reaction-rate limited and therefore scales
purely with pixel count: ~45 s typical / ~17 min worst at 512, against ~3 min / ~70+ min at
1024 with `eval/numerics._phi_contour` peaking at 4–6 GB, on a host with five recorded OOM
kills. `eval/lgen_eval.grid_for_L` also refuses grids above 512 by design.

**Why periods-per-box is drawn, never fixed.** With a single p, `k* = p·2π/L` inverts exactly
and the domain size becomes the label again — the leak that made `kstar_rel_err` a gate
rather than evidence on the legacy data (D6). Each dataset draws 5 *distinct* p from
{16..32}, seeded with SHA-256 so the draw is process-independent.

**Measured properties of the shipped sets.** px/wavelength 17.1–30.1, all far above the 6.0
floor of D15, giving a k\* half-bin precision of 1.6–3.1 % against 8.3 % on the legacy data —
the first time the pre-registered `kstar_phys_cv ≤ 0.10` bar sits meaningfully above the
estimator's own noise floor rather than on it. Domain sizes run 245.8–794.0.

**A 96²-era bound was relaxed, deliberately and narrowly.** `simulate_and_classify` asserted
`18 ≤ L ≤ 220`; all ten canonical samples exceed it. Those bounds encode *resolution* at a
96×96 grid, and `L` enters the physics only as a unit (CLAUDE.md §7c) — the real constraint
is pixels-per-wavelength, which `canon_generate` now enforces directly at
`PPW_MIN/PPW_MAX = 16/32`. The bound is now a parameter defaulting to the old values, so
every existing caller is bit-for-bit unaffected.

**Stored, and not stored.** Final frame at full resolution, all three channels, plus a cv
time-trace for the saturation gate. **No trajectory** — nothing under `src/rngrn/` reads it,
so at 512² it would cost 6× the storage for data no consumer touches. All three channels are
kept even though a real photograph gives one, so `m=1` vs `m=3` stays a controlled comparison
on identical data rather than two different datasets.

**Split declared before generation**, per `docs/PREREGISTRATION.md`: 2 tuning / 3 held-out per
class, with tuning slots seeded from systems recovery experiments have already burned and
held-out slots restricted to systems never run.

**NOT established here.** No recovery result; nothing is fitted. "Patterned" is not
"Turing-unstable" — σ(k) is evaluated nowhere in this unit. The `m<N` objective remains an
open problem (exp06 measured the residual as harmful, 9/9 cells collapsed), and `raps` still
assumes a periodic tile with no windowing, so a cropped real image will bias k\* silently.
The saturation tolerance (1 % over the last 20 % of the run) is a convergence tolerance, not
a calibrated threshold, and is enforced as a fail-loud gate rather than used to judge
anything.

### D-CANON-3 — periods-per-box is a geometric ladder over {8..40}, checked against a leak bar
*2026-08-10. Same unit. Evidence: `scripts/canon_select.py::draw_periods` /*
*`oracle_leak_error`; tests in `tests/test_canon_select.py`; the measurement below.*

**The decision.** Each canonical dataset's five periods-per-box are laid out as a geometric
ladder across {8..40} with a seeded sub-rung offset, and the result is **checked** against
`LEAK_MIN_ORACLE_ERR = 0.25` and rejected if it fails, rather than assumed adequate.

**Why this needed a decision at all.** The obvious reading of "vary p so L does not encode
k\*" is *draw distinct integers*. That is not sufficient, and the shortfall was measured on
generated data rather than reasoned about in advance. A first attempt drew five distinct p
i.i.d. from {16..32} and produced `{17, 22, 23, 24, 28}` for `turing_labyrinth`. An oracle
blind predictor `k = q·2π/L` — one fixed integer q chosen *after* seeing the answers — fits
that to **4.5 %** median error. The legacy leak scores 0.0 % and `three_gene_qvar` 45.5 %, so
4.5 % sits far closer to the defect this project exists to have fixed than to the fix.

The cause: the predictor's relative error is exactly `|q − p| / p`, so protection comes from
spread in **log** space. Measured over 4000 random 5-sample draws:

| p range | spread | median oracle error | draws below 15 % | px/wavelength at 512 | k\* floor |
|---|---|---|---|---|---|
| 16–32 | 2.0× | 9.1 % | **95 %** | 16.0–32.0 | 1.6–3.1 % |
| 8–40 | 5.0× | 15.0 % | 48 % | 12.8–64.0 | 1.2–6.2 % |
| 3–14 (legacy qvar) | 4.7× | 20.0 % | 27 % | 36.6–170.7 | 3.6–16.7 % |

{8..40} gives a wider relative spread than the legacy `qvar` range while keeping every sample
better than the legacy data on **both** resolution and k\* precision. Replacing the i.i.d.
draw with a geometric ladder then lifts the realised figures well above the random median:

| dataset | periods | spread | oracle error |
|---|---|---|---|
| `turing_spots` | 8, 11, 16, 24, 36 | 4.5× | **37.5 %** |
| `turing_labyrinth` | 8, 10, 15, 23, 35 | 4.4× | **33.3 %** |

**The limit, recorded because it does not go away.** At n=5 no range decouples strongly — an
oracle single q can always sit near the middle of five values, and even the legacy {3..14}
range only reaches 20 % at n=5 against 45.5 % at n=34. **Corpus-level `kstar_rel_err`
medians are therefore not meaningful on a five-sample dataset at any period range.** These
sets support per-sample k\* claims; they do not support a corpus median.

**What was rejected: fixing L outright.** A constant L, chosen without reference to any
system's k\*, would make the leak *structurally absent* rather than small, and would make the
periods emergent from the physics. It was implemented and measured — L = 300 puts all 37
gated candidates at 9.1–30.7 emergent periods and 16.7–56.0 px/wavelength — and then
reverted on owner instruction. The reasoning: `λ = 2π/k*` is set by the network's Jacobian
and diffusivities either way, so the *periodicity of the pattern* is always the physics.
Choosing `L = p·λ` only sets how much of it is in view — a field-of-view choice of the kind a
microscope makes. Commit `948281d` and its revert `ac11847` hold the implementation if the
question is reopened.

---

### D-CANON-4 — the canonical sets become the training data source
*2026-08-10. Owner instruction. Status: **DECIDED**.*

**The decision.** From 2026-08-10, `turing_spots` and `turing_labyrinth` are the training
data source for simulated-data work. `docs/PREREGISTRATION.md` §1 is amended to match.

**What this does not do.** `three_gene_qvar` is not deprecated and none of its numbers are
withdrawn. It is the provenance of every canonical system — each canonical sample is a
re-simulation of a qvar or multiL system at 512² — and existing results against it stand. It
stops being where *new* headline claims are drawn from. `three_gene_multiL` keeps its
cross-L role under §3.5a. The legacy and classical families are unaffected: still barred from
k\* claims, still dormant respectively.

**Two consequences that must be stated rather than discovered.**

1. **The primary evidence base is now 10 samples, 6 of them held out**, against 26 held-out
   in the `three_gene_qvar` split. This follows from the owner's requirement of the smallest
   number of datasets, one per pattern type, and is not a defect. But it means **a per-sample
   result is the unit of evidence**, and a median over five samples is not a corpus
   statistic. Combined with D-CANON-3, `kstar_rel_err` medians must not be quoted from these
   sets at all.
2. **No config points here yet.** `configs/m3_registry.yaml`, `nc1_m3_registry.yaml`,
   `expA_control_full.yaml` and `expA_hidden_channel.yaml` still name `three_gene_val`;
   `expB_*.yaml` still name `two_gene_classical_val`. Repointing them is a separate change,
   deliberately not made while generating the data, and it will change what those configs
   measure — so it is announced here rather than done quietly.

### D-CANON-5 — canonical morphology is MEASURED from the field; `turing_labyrinth` is a mixed class
*2026-08-10. Same unit. Evidence: `scripts/phase_topology.py`, `tests/test_phase_topology.py`,*
*`scripts/canon_annotate.py`; the cross-tabulation below over all 57 re-simulatable systems.*

**Two decisions, one measurement.**

**(a) The canonical classes are defined by measured phase topology, not by the generator's
stored label.** `phase_topology.measure` splits the field at its Otsu threshold and asks two
questions: which phase fragments into domains, and are those domains round or worm-like.
Round bright islands in a connected dark matrix → `spots`; round dark voids in a connected
bright matrix → `holes`; neither → `labyrinth`.

**The corpus is NOT relabelled.** All 413 registered samples keep their stored `morphology`
attribute, so no existing number changes meaning. The measured value is written *alongside*
it, on the canonical payloads only, as `morphology_measured` plus the statistics behind it.

**Why.** The generator's rule assigns morphology from the area fraction above a fixed
contrast threshold. Cross-tabulated against measured topology over the 57 distinct
re-simulatable systems:

| stored label | → spots | → holes | → labyrinth |
|---|---|---|---|
| `spots` (29) | **28** | 0 | 1 |
| `labyrinth` (17) | 3 | **7** | **7** |
| `stripes` (11) | 2 | 1 | 8 |

The `spots` label is reliable — 28 of 29. `labyrinth` is not: only 41 % of it is a labyrinth,
and most of the rest is a **hole** pattern, a genuinely distinct morphology.

**A correction to D-CANON-1, which was wrong.** That entry said `holes` is "structurally
unreachable, because species 0 is the self-activator in all six topologies, so the observed
channel is positively skewed by construction". **The patterns are not unreachable — 7 of 57
systems produce them, several with negative skew.** What is unreachable is the *label*:
detecting holes requires `phi > 0.66`, i.e. two thirds of all pixels more than 0.4 SD above
the mean, but a connected bright matrix has its own spread and much of it falls below that
line. The test can essentially never fire regardless of the field. The corpus contains zero
samples *labelled* `holes` — which is a fact about the classifier, not about the physics, and
D-CANON-1 read it the wrong way round.

**(b) `turing_labyrinth` ships as generated, and it is a MIXED class.** Measured composition:

| sample | stored | measured | bright domains / circularity | dark domains / circularity |
|---|---|---|---|---|
| `sample_0000` | labyrinth | **labyrinth** | 35 / 0.77 | 3 / 0.29 |
| `sample_0001` | labyrinth | **holes** | 1 / 0.02 | 64 / 1.22 |
| `sample_0002` | labyrinth | **holes** | 1 / 0.01 | 95 / 0.94 |
| `sample_0003` | labyrinth | **holes** | 1 / 0.00 | 297 / 1.29 |
| `sample_0004` | labyrinth | **labyrinth** | 41 / 0.74 | 189 / 0.42 |

So the dataset is **3 hole patterns + 2 labyrinths**, and its name is to that extent
misleading. Owner's call, taken with the measurement in hand: keep the data as generated and
document the mix rather than spend another generation run. Every sample carries its measured
morphology, and the figures print `stored → MEASURED` wherever the two disagree, so the mix
is visible at the point of use rather than buried here.

`turing_spots` is unaffected: 5 of 5 measured `spots`, in agreement with the stored label.

**What was rejected.** (a) Re-selecting `turing_labyrinth` as 5 true labyrinths — exactly 5
qualify, so it was feasible but with zero slack, and the owner chose not to spend the run.
(b) A `turing_holes` third class — only 3 gated, stable hole systems exist, below the 5
required. (c) Changing `gen_tg3.classify` — that would alter the meaning of the `morphology`
attribute on all 413 existing samples, which D-TDPLOT-1 already considered and rejected.

**A bug this surfaced, recorded because the class of error recurs.** The first version of the
speckle floor discarded domains smaller than a fixed fraction of the FRAME. At 512² with 36
periods a spot is ~7 px across (area ~38 px) while the floor was 52 px, so every domain was
deleted and a clean spot lattice measured as `labyrinth` with **zero** domains. The floor is
now a fraction of the pattern's own wavelength squared, read from the field's spectrum.
`tests/test_phase_topology.py` pins the regression at the exact resolution that failed.

**Not established.** The circularity cut of 0.55 is read off the separation measured on the
canonical frames (round domains 0.94–1.57, worms 0.29–0.77) and is **not calibrated against a
control**. It is used for labelling only, never as a pass condition, and no threshold in
`docs/PREREGISTRATION.md` depends on it.

---

### D-EVID-17 — the `~15%` k\* tolerance is UNCALIBRATED *and* unreachable: both Milestone-1 reference frames diverge on their own shipped defaults

**Date:** 2026-08-11. **Status:** DECIDED (marked **[UNCALIBRATED]**; no calibration attempted).
**Decided by:** the implementing agent under §10 delegated authority, during a repo-wide docs
audit. This is a bookkeeping correction, not a threshold change — `TUNING.md` is not
`docs/PREREGISTRATION.md`, no pass condition is touched, and nothing is weakened.

**The gap.** `TUNING.md:131-132` said: *"[TUNE] Milestone 1 (N=m=2) — `configs/milestone1_{gm,schnak}.yaml`.
Get recovery to land in-regime and match k\* within ~15% on both reference frames."* That `~15%`
dates from the initial template commit and has never been measured against anything. It carried
only a `[TUNE]` tag, and no entry in this file addressed it — so it failed CLAUDE.md §8's
"thresholds are calibrated, never inherited" and §10's requirement that an uncalibratable number
be marked UNCALIBRATED in **both** the code/doc and here.

**Measured, and worse than uncalibrated — the frames cannot be generated at all.** Running the
two shipped systems through `data/solver.py::simulate_to_attractor` at its own defaults
(`resolution=128, T_max=4000, dt=0.1`), with the classes' shipped parameters:

| config | system | result |
|---|---|---|
| `configs/milestone1_gm.yaml` | `GiererMeinhardt(a=0.5, b=1.0, c=6.1, K=0.003, n=2, Du=1.0, Dv=100.0, L=100.0)` | `FloatingPointError: solver diverged at step 133; reduce dt` |
| `configs/milestone1_schnak.yaml` | `Schnakenberg(a=0.1, b=0.9, gamma=100.0, Du=1.0, Dv=40.0, L=100.0)` | `FloatingPointError: solver diverged at step 29; reduce dt` |

Both overflow in the reaction term before the diffusion step (`rd_models.py:80` for GM,
`rd_models.py:105` for Schnakenberg). This is **live code, not dead code** — both systems are
registered in `data/rd_models.py` and the `data.source="reference"` dispatch in `train.py` is
fully wired — so it fails loudly rather than silently, per §4. Reproduced twice independently
(an audit subagent, then re-run directly) before this entry was written.

**Root cause — `solver.py` has no diffusion CFL bound.** `data/solver.py:55-59` picks the
timestep from the *reaction* Jacobian alone:

```python
rate      = float(np.max(np.abs(np.linalg.eigvals(J0))))
dt_stable = 0.2 / (rate + 1e-9)
dt_eff    = min(dt, dt_stable)
```

The diffusion half is explicit (`_laplacian_fft` each step) and is never consulted. Explicit
diffusion needs roughly `dt < dx^2 / (2*D_max)`; at the shipped `L=100, resolution=128` →
`dx ≈ 0.78`, so with `Dv=100` that bound is `~3e-3` while the reaction cap leaves `dt = 0.1`,
about **30x too large**. GM (`Dv=100`) therefore blows up sooner in D-terms than
Schnakenberg (`Dv=40`) survives longer in step count only because its reaction cap is much
tighter (`gamma=100`). The comment above that code says the cap "keeps the reference
generator stable without a config edit" — it does not, for any diffusion-dominated system.

**This also means the README quickstart is broken**, since its first command is
`rngrn generate-data --config configs/milestone1_schnak.yaml`. README now says so inline.

---

**FIXED 2026-08-11, and a SECOND defect found underneath it.** Owner instruction: fix the
solver, with the steer that "dt should be around 0.001" — which the measurement above
confirms exactly (the explicit bound is 6.2e-4 / 1.5e-3).

**The fix.** Making the old fully-explicit scheme CFL-safe would have cost 2.6–6.5 M steps
per frame (~1.4–3.6 h at the measured 2 ms/step). Instead `simulate_to_attractor` now solves
the diffusion half **exactly** in Fourier space (Lie-split IMEX, `exp(-D k^2 dt)`), which
removes the diffusion stability constraint outright rather than paying it. `stable_dt()` still
computes and enforces the explicit bound for any caller integrating explicitly, and that bound
is what `tests/test_reference_solver.py` pins. A convergence check was added so the function
stops when the field stops moving instead of always burning the horizon; its `atol=1e-6` is
calibrated, not inherited — the measured change-rate floor is 4.2e-7 (gierer_meinhardt, which
never reaches 1e-8) and 4.4e-13 (schnakenberg).

**Result: both frames now generate.** gierer_meinhardt ~32 s and schnakenberg ~16 min per
128x128 frame, both finite, both patterned (contrast 0.90 / 1.43).

**THE SECOND DEFECT — the output is stable but NOT dt-converged. Status: UNCALIBRATED, OPEN.**
Refining `dt -> dt/4` moves gierer_meinhardt's contrast ~35 % (0.891 -> 0.579) and shifts k\*
by a whole FFT bin (0.5368 -> 0.4785). **Read against its control**, as §8 requires: 6 seeds at
fixed `dt` spread only **2.6 %** with k\* pinned to a single bin. The dt dependence is
therefore ~13x the seed variation and is a real integration error, not attractor selection.

Two candidate fixes were tried and **both rejected on measurement**:
- *Symmetric (Strang) splitting instead of Lie.* No measurable improvement — 34 % -> 35 % —
  while costing an extra FFT pair per step (gierer_meinhardt 31.7 s -> 39.1 s at 128²). Owner
  instruction, given the null result: drop back to Lie. **The shipped scheme is Lie.**
- *RK2 midpoint reaction substep,* to stop the Euler substep capping the order at O(dt).
  **Worse:** at the shipped dt it destroyed pattern formation entirely, contrast -> 0.000.

**The shipped timestep is the outlier, not the refinements.** Across both schemes and both
refinement levels the contrast clusters at **0.45–0.58** (Lie dt/4 0.577; RK2 dt/4 0.453,
dt/16 0.474), while the shipped dt alone gives **~0.88**. Whatever the correct answer is, it is
much closer to the refined cluster than to what this generator currently produces.

So the dominant error is **not** splitting order, and is not yet diagnosed. Consequence, stated
plainly: frames from this path are fine for plumbing and for morphology, but **no k\* or
contrast value measured from them may be quoted as a reference number** until this is closed.
That is also why the `~15%` tolerance stays UNCALIBRATED rather than being calibrated now — the
frames it would be calibrated against are themselves not converged.

**A separate resolution finding, recorded so it is not rediscovered.** At the shipped
`L=100, resolution=128`, schnakenberg's pattern has wavelength ~1.8 against `dx = 0.78` — only
**2.3 pixels per wavelength**, barely above the Nyquist limit of 2. That is a data-quality
ceiling set by the config, not by the solver; gierer_meinhardt is comfortable at ~15 px/wavelength.

**NOT ESTABLISHED.** What causes the dt dependence; which dt (if any) gives the correct
attractor; and therefore what a calibrated Milestone-1 k\* tolerance would be. All open.

**So the tolerance is unreachable, not merely unmeasured.** A number cannot be calibrated
against a frame that cannot be produced. `TUNING.md` now tags the knob `[UNCALIBRATED]` and
names the divergence, so nobody tunes against it believing it is a real bar.

**What was rejected.** (a) *Deleting the Milestone-1 line.* The reference-frame path is a real,
wired capability and the milestone is still a sensible target; deleting it would hide a defect
rather than record it. (b) *Picking a defensible tolerance now (e.g. the 10% used elsewhere).*
That would be inventing a threshold with no measurement behind it — precisely the failure §8
names. (c) *Fixing the solver by reducing `dt`.* Tempting and probably correct, but it is a
numerics change with no owner request and no test behind it; the divergence is now recorded so
it can be fixed deliberately. See `docs/FUTURE_WORK.md`.

**Not established.** Whether the reference frames are recoverable at a smaller `dt`, and hence
what a calibrated k\* tolerance for Milestone 1 would actually be. Both remain open.

### D-FFT-1 — training gains a forward map: implicit differentiation at the patterned steady state

**Date:** 2026-08-11 (Fourier-training design session, `feature/fft-training`). **Status:** DECIDED
**Decided by:** the owner, choosing among options laid out by the implementing agent.

**The decision:** training acquires a model-side predicted pattern: the QSS
reaction–diffusion system is integrated to its patterned stationary state `u*(θ)` under
`no_grad` (existing `eval/numerics.integrate_etdrk4_rfft`), and spectral losses of `u*` are
differentiated through the stationarity condition `D∇²u* + f(u*;θ) = 0` by the implicit
function theorem — one adjoint linear solve, no backprop through the integrator. This makes
`CLAUDE.md` §7c ("training never simulates") false; the plan schedules its amendment in the
change that lands the solve.

**Evidence:** the current objective consumes exactly two frame statistics (`kstar_obs`,
`recover.py:417`; `frame.mean()`, `losses/total.py:79`) against ~3N²+3N parameters — data
starvation, with the measured failure mode to match (C1: 7/8 seeds never reach the Turing
regime, `docs/C1_COMPETITIVE_TUNING.md`). Linear theory cannot predict the saturated
spectrum, so any spectral data term needs a nonlinear forward map. No `torch.fft` exists
anywhere in the training path (grepped this session).

**What was rejected and why:** (a) *BPTT through the rollout* — backprop through thousands
of stiff PDE steps; most expensive and fragile, and structurally identical gradients to IFT
at the fixed point. (b) *Analytic-only enrichment* (dispersion-shape terms, no simulation) —
cannot constrain saturated amplitudes or morphology, so it cannot close the gap the stopping
criterion demands; retained as ablation arm A0/A1 territory instead.

**Not independently validated:** nothing is implemented. Diagnostic D1
(`docs/PLAN_fourier_training.md` §3) — IFT gradient vs finite differences, translational
zero-mode orthogonality — is the go/no-go before any training code is accepted.

**Where it lives:** `docs/SPEC_fourier_training.md` §3; `docs/PLAN_fourier_training.md` §3 (D1).

### D-FFT-2 — the stopping criterion is revised: wavelength is primary, ground-truth topology match is reported-only, reproducibility is the binding recovery gate

**Date:** 2026-08-11 (design session). **Status:** DECIDED
**Decided by:** the owner, explicitly superseding the session kickoff's own criterion 1.

**The decision:** recovery no longer requires the learned GRN to match the generator's
topology. Binding instead: (R1) over 5 independent runs, all 5 recovered sign structures
identical under the identity species mapping; (R2) parameter similarity on
time-scale-invariant combinations, reported-only until calibrated per D-FFT-9. Forward
validation (Criterion 2) gates on: strict Turing instability, stationary non-homogeneous
rollout, and wavelength within one radial bin (D-FFT-3). Ground-truth sign comparison stays
in every report as a diagnostic.

**Evidence:** the identifiability concern is structural — many (J, D) share a k\*, and the
owner's stated goal is a GRN that *produces the pattern*, reproducibly, not the generator's
circuit per se. R2's restriction to time-scale-invariant combinations follows from a
verified degeneracy: a single stationary frame is invariant under jointly scaling (f, D), so
absolute rates are pinned only by priors and would fail a similarity gate uninformatively.

**What was rejected and why:** the originally proposed ground-truth bar (9/9 signed entries
under identity) — rejected by the owner as the wrong claim; also the softer 8/9 and
off-diagonal-only variants, mooted with it. Permutation-aligned scoring as the headline —
rejected because channels are observed and ordered (m = N), so species are not anonymous.

**Where it lives:** `docs/SPEC_fourier_training.md` §1, §9.3–9.4.

### D-FFT-3 — the wavelength tolerance is one radial bin: |k\*_sim − k\*_obs| ≤ 2π/L

**Date:** 2026-08-11 (design session). **Status:** DECIDED
**Decided by:** the owner, from options with the resolution argument laid out.

**The decision:** forward-validation F3 and the training-side check use
`|k\*_sim − k\*_obs| ≤ 2π/L`, both sides measured by the same estimator
(`observables.raps`) on the same grid and L. Per-sample: rel-err ≤ 1/p — 12.5 % at p=8
(`turing_labyrinth/sample_0000`), 2.9 % at p=35. Always reported with
`kstar_fft_bin_width` and the L-only trivial-predictor control (D-EVID-7/8 discipline).

**Evidence:** the RAPS resolution *is* Δk = 2π/L; the pre-registered 8.3 % was derived on
the three_gene sets and D-EVID-8 records it is not dataset-independent — at p=8 it demands
sub-bin precision the measurement does not carry. One bin clears the trivial predictor
(33–37 % error on canonical sets, `docs/HANDOFF_canonical_datasets.md`) by ~3×, so the
number is meaningful against its control.

**What was rejected and why:** (a) *importing the 8.3 %* — an inherited threshold, exactly
what §8 forbids, and sub-resolution at p=8. (b) *half a bin* — leans on sub-bin centroid
precision characterised but not certified at this p. (c) *a fixed percentage for all
samples* — either loose at high p or sub-resolution at low p, the failure D-EVID-8
documents.

**Where it lives:** `docs/SPEC_fourier_training.md` §9.1.

### D-FFT-4 — forward validation gates on the QSS rollout; the finite-μ lift is a later milestone, never the gate

**Date:** 2026-08-11 (design session). **Status:** SUPERSEDED by D-LIFT-1 (2026-08-17;
the lift becomes the gate *after* a validation ladder — numbers gated under this
decision's QSS-only regime remain comparable only to other QSS-gated numbers)
**Decided by:** the owner (who had asked for "simulated under the dynamical lift"), after
the numerics risk was laid out.

**The decision:** the DoD's forward validation runs the existing QSS rollout
(`eval/rollout.simulate`, spectral ETDRK4). Spatial simulation of the μ-lifted system is
milestone M6, post-DoD, admissible only after `docs/FUTURE_WORK.md`'s preconditions:
demonstrated dt-convergence and reproduction of the QSS field at μ ≤ 1e-4 (the lift's gates
relax at 1/μ; a stiff integrator that damps the instability manufactures "patterns die at
finite μ" for free).

**Evidence:** `docs/FUTURE_WORK.md` — the lifted system has never been simulated in space;
`eval/lifted.py` is written, not validated. Putting never-built stiff numerics on the DoD
critical path would let a numerics artefact read as a recovery failure.

**What was rejected and why:** (a) *lift as the gate* — strongest biological claim, but see
above. (b) *dropping the lift entirely* — abandons the owner's stated interest; kept as
evidence-stage instead.

**Where it lives:** `docs/SPEC_fourier_training.md` §1; `docs/PLAN_fourier_training.md` §4 (M6).

### D-FFT-5 — Stage 0 is `turing_labyrinth/sample_0000` with a k-band + channel holdout, gated beat-all against an alternative-sign-structure null

**Date:** 2026-08-11 (design session). **Status:** DECIDED
**Decided by:** the owner, through the "GRN-mechanism lens" he set: a split must certify
mechanism, not curve-fitting.

**The decision:** Stage 0 fits species 0 only, spectral terms restricted to the training
annulus `B_train` around k\*_obs; held out are the k-bands `B_low ∪ B_harm` and channels
1–2 entirely. Gate 1: the recovered model's held-out-band prediction error beats **every**
member of a ≥6-member null ensemble — alternative sign structures from the multistart
restarts plus single-edge flips of the winner, each refit to `B_train` with gates frozen,
identical budget. Gate 2: predicted channel-1/2 amplitude ratios beat every null member's.
Patch (quadrant) consistency is reported, never gated. Sample: `sample_0000` (tuning role).
A Stage-0 pass licenses only "the machinery fits and the gates discriminate" — never a
recovery claim.

**Evidence:** the canonical set stores single final frames (time-snapshot splits
impossible, D-CANON discussion) and `sample_0000`'s channels 1–2 are near-flat (cv ≈ 0.075
vs 0.81) yet noise-free, so channel prediction tests the eigenvector structure of
`J − k²D` with clean signal; spectral bands are coupled by the kinetics' nonlinearity, so
held-out-band prediction is forced by mechanism, which a 3N²+3N-parameter model cannot
fake by memorisation. `sample_0004` (the other measured labyrinth) is **held-out role** —
using it for iteration burns 1 of 6 held-out samples corpus-wide
(`docs/PREREGISTRATION.md` split).

**What was rejected and why:** (a) *sample_0004 first* — contaminates the pre-registered
split. (b) *spatial patches as the gate* — near-trivial under statistical homogeneity;
demoted to reported control. (c) *channel prediction as the sole primary gate* — tiny
dynamic range, no calibrable null on its own. (d) *beat-the-null-median* — with a ≤10-member
ensemble, "beats most wrong mechanisms" is the weaker story; beat-all keeps every outcome
informative, including fail-no-spread, which IS the identifiability measurement.

**Where it lives:** `docs/SPEC_fourier_training.md` §9.5–9.7; `docs/PLAN_fourier_training.md` §2.

### D-FFT-6 — composition: every existing term stays on throughout; spectral terms ignite on detected patterning, omitted-not-zeroed

**Date:** 2026-08-11 (design session). **Status:** DECIDED
**Decided by:** the owner, ratifying the implementing agent's recommendation verbatim.

**The decision:** the existing objective (kstar, turing, anchor, anticollapse) runs
unchanged for the whole optimisation — it is the ignition system that reaches the Turing
regime and the recovery net when a model drifts out of it. The forward solve runs only when
the model is currently Turing-unstable; spectral terms contribute only when the solve
actually patterned (amplitude above a floor, D-FFT-9); otherwise they are **omitted** with a
`spectral_skipped` flag (the `resid_skipped` precedent), never computed-and-zeroed. The
current default objective is frozen as a permanent baseline config
(`configs/baseline_linear.yaml`, ablation arm A0).

**Evidence:** a random init is almost never Turing-unstable (0/398 measured for the
low_basal family, `model.py:158-166`), so a spectral loss is undefined exactly when the
model most needs guidance; the linear-theory terms are what fixed 0 % → 36.8 % Turing
convergence (D8, `docs/STATE_OF_THE_SCIENCE.md` §2.1).

**What was rejected and why:** (a) *joint-from-step-0* — identical up to wasted forward
solves on homogeneous states and muddier logs. (b) *spectral replaces the k\*-anchor* —
removes the only data term that works pre-pattern and destroys the cleanest ablation
(status quo vs status quo + spectral); "spectral-only after ignition" survives as ablation
arm A4, a measurement rather than a commitment. (c) *schedule-based ramp
(`DataFirstStaging`) for the spectral terms* — ignition is checkable, so a schedule that
fires before the model patterns just burns solves.

**Where it lives:** `docs/SPEC_fourier_training.md` §3, §5; `docs/PLAN_fourier_training.md` §5 (A0, A4).

### D-FFT-7 — spectral shape terms are gain-invariant; amplitude enters only through separate, opt-in terms

**Date:** 2026-08-11 (design session). **Status:** DECIDED
**Decided by:** the owner.

**The decision:** `spec_shape` and `spec_aniso` are normalised by training-band power only
(gain-invariant, and normalisation over held-out bands would be leakage); amplitude enters
solely through `spec_amp_mean` and `spec_amp_fluct` — separate term keys, ON for Stage 0,
individually ablatable, wholesale switch-off-able. This consciously breaks, as a *mode*,
the repo's scored-spectral-quantities-are-amplitude-blind convention
(`docs/STATE_OF_THE_SCIENCE.md` §2.8) while preserving it as the available configuration.

**Evidence:** Stage-0 frames are synthetic, noise-free, absolute concentrations — amplitude
is legal, clean signal, and the channel co-gate (D-FFT-5) is only meaningful if the loss
sees amplitude somewhere; the amplitude-blind convention exists for unknown fluorescence
gain on future real data, which flipping two weights restores.

**What was rejected and why:** (a) *fully amplitude-blind* — guts the channel co-gate and
drops the "amplitudes inform concentration fields" premise. (b) *one amplitude-aware
spectral distance* — shape and amplitude errors become inseparable in logs and ablations,
and restoring gain-invariance later means redesigning the term, not flipping a weight.

**Where it lives:** `docs/SPEC_fourier_training.md` §4, §5. The Stage-2 observation-noise
arm is the scheduled decision point for demoting the amplitude terms
(`docs/PLAN_fourier_training.md` §6).

### D-FFT-8 — morphology is reported-only in forward validation, for now

**Date:** 2026-08-11 (design session). **Status:** DECIDED (the deferral); the gating
question itself is OPEN, scheduled at Stage 2.
**Decided by:** the owner: *"start with morphology being reported only, and focus on
recovering Turing patterns."*

**The decision:** the DoD's forward-validation gate is F1 (strict Turing) + F2 (stationary,
non-homogeneous) + F3 (one-bin wavelength). The rollout's morphology class — judged by
`scripts/phase_topology.measure`, the canonical sets' own annotator, at matched grid and L —
plus `morphology_distance` and the 2-D spectral distance are recorded in every report but
gate nothing. All comparisons are rotation-invariant; pattern orientation never matters.
Whether morphology becomes binding is decided at Stage 2 (M4), when rollouts exist in
numbers.

**Evidence:** the two available classifiers disagree in provenance (the scoring-side bank
was trained on the superseded three_gene sets; `phase_topology` labelled the canonical
data and caught its 3 mislabelled holes samples, D-CANON-5), and its 0.55 circularity cut
is UNCALIBRATED — a binding gate today would hang on an uncalibrated judge.

**What was rejected and why:** (a) *gating on class match now* — premature per the owner's
priority ordering, though noted as non-trivial (sample_0000's class margin, 0.035, is the
smallest in the dataset). (b) *gating on class + calibrated distance* — adds a calibration
burden Stage 0 does not need.

**Where it lives:** `docs/SPEC_fourier_training.md` §1, §8; `docs/PLAN_fourier_training.md` §4 (M4).

### D-FFT-9 — calibration ledger for the Fourier-training knobs

**Date:** 2026-08-11 (design session). **Status:** OPEN — each item below is UNCALIBRATED
until its named measurement closes it; none may gate before then.
**Decided by:** the implementing agent under delegated authority (the *rules* below are the
decision; the numbers are deliberately not chosen yet).

**The decision:** every new knob is born UNCALIBRATED with a pre-registered calibration
rule, so the number that eventually closes it cannot be tuned-to-pass:

- **Zero-call threshold** (edge-absent rule; inherited 5 %-of-max is uncalibrated):
  from Stage-0 ≥10-seed runs, place the cut at the separation point of the magnitude
  distributions of consistently-present vs consistently-absent edges. Non-separation is a
  finding (sign structure unstable), not a licence to pick a cut. Closes R1 (D-FFT-2).
- **Parameter-similarity threshold (R2):** measured within-cluster log-space spread of the
  Stage-0 seed set on time-scale-invariant combinations, plus margin. R2 reports until then.
- **Band edges `b_lo, b_hi`** (provisional 0.5/1.5 × k\*_obs): set from the measured
  spectral support of `sample_0000` (diagnostic D3), including whether `B_harm` carries any
  measurable signal at p=8 — if not, the held-out-band gate leans on `B_low` and says so.
- **Ignition amplitude floor:** from diagnostic D2's separation between "patterned" and
  "still relaxing" forward solves.
- **New term weights** (`spec_shape`, `spec_aniso`, `spec_amp_*`, `real_moments`): swept at
  Stage 0 on the tuning-role sample; settled values recorded here.
- **Channel co-gate threshold:** beat-all against the D-FFT-5 null ensemble (a rule, not a
  number — inherits the null's spread).

**What was rejected and why:** binding provisional numbers today (e.g. log-spread ≤ 0.2 for
R2) — an UNCALIBRATED-but-binding gate passes or fails for reasons nobody can defend, the
failure §8 exists to prevent. The two live precedents (the ~15 % k\* tolerance, D-EVID-17;
`coupling_threshold = 0.05`) show how inherited numbers calcify.

**Where it lives:** `docs/SPEC_fourier_training.md` §9.3–9.8; `docs/PLAN_fourier_training.md`
§3 (D2, D3, D5). Each closure appends its number and evidence to this entry.

#### D-FFT-9 closure 1 — band edges: b_lo = 0.60, b_hi = 1.55 (× k*_obs)

**Date:** 2026-08-12. **Status:** CLOSED (this item; the ledger stays open for the rest).
**Decided by:** the implementing agent under §10 delegated authority, from diagnostic D3.

**The decision:** `B_train = [0.60, 1.55]·k*_obs` — the measured ≥1 %-of-peak contiguous
support of the RAPS spectrum of `turing_labyrinth/sample_0000` channel 0 (8 radial bins at
512²/p=8). `B_low = (0, 0.60)·k*_obs` excluding DC; `B_harm = [1.55, 3.0]·k*_obs`.

**Evidence** (`experiments/diag_fft/d3/results.json`; script `scripts/diag_fft_d3.py`):
support at ≥50 %/≥10 %/≥1 % of peak = [0.84, 1.08] (2 bins) / [0.84, 1.20] (3 bins) /
[0.60, 1.55] (8 bins). B_harm above these edges carries 1.5–3.2 % of B_train power across
12 bins, ~15 orders above the float floor (the sample is noise-free; there is no
observation-noise floor), so the held-out-band gate does NOT need to lean on B_low —
answering the question this ledger item pre-registered. The 1 % edges essentially
reproduce the provisional (0.5, 1.5), which is therefore adopted as measured rather than
inherited.

**What was rejected and why:** the ≥10 %-of-peak edges [0.84, 1.20] — only 3 bins, i.e.
2 free shape ratios, which re-starves the very data channel this design exists to widen
(SPEC §2); and keeping the provisional 0.5/1.5 unmeasured — same numbers, but a
placeholder is not a measurement. Caveat recorded: these edges are measured on ONE
tuning-role sample; Stage 2 re-measures per sample class before any cross-sample claim.

**Where it lives:** `loss.spectral` config when M1 lands; `docs/DIAGNOSTICS_fft.md` §D3.

### D-FFT-10 — the IFT adjoint is the MINIMAL-NORM solution (LSMR + refinement); a translation-projected GMRES adjoint is rejected, with measurement

**Date:** 2026-08-12. **Status:** DECIDED (binds M1's forward/adjoint module).
**Decided by:** the implementing agent under §10 delegated authority, from diagnostic D1.

**The decision:** the adjoint system Aᵀλ = ∂L/∂u at the patterned state is solved for the
minimal-norm least-squares solution — right-preconditioned LSMR (M = (γ + D k²)⁻¹) with
iterative refinement on the TRUE residual until ≤1e-10 — never by a Krylov method whose
space is projected off the translation modes. Newton polish of the pattern solve carries a
2×2 translation-subspace correction (G_ij = ⟨t̂_i, A t̂_j⟩) per step.

**Evidence** (`scripts/diag_fft_d1.py`; probe artefacts summarised in
`docs/DIAGNOSTICS_fft.md` F-D1-2/F-D1-3): grid pinning makes the translations near-null,
not null (‖At‖/‖Av‖ 3.6e-4 at 96²). The projected-GMRES adjoint reported residual 1e-13
while its true residual stalled at 5.5e-4–5.7e-3, biasing gradients by 1e-5–5.5e-2
relative — flat in ε (systematic), with a tangent-mode cross-check localising the error to
the solve. With the minimal-norm scheme, true residuals reach ≤3.6e-12 and all five loss
terms pass FD verification at ≤2.8e-5 (tolerance 1e-4). *(Correction 2026-08-12, evidence
audit: those two figures are the 64² PROBE's; the committed 96² acceptance record —
`experiments/diag_fft/d1/results.json` — measured residuals ≤6.1e-12 and worst FD rel err
6.4e-8. Both regimes pass; the probe numbers were conflated with the record here.)* The theory: for translation-
invariant losses (∂L/∂u ⊥ t at ≤1e-18) the pinning response cancels in dL/dθ, leaving
exactly the minimal-norm adjoint as the correct object.

**What was rejected and why:** (a) projected GMRES — measurably biased, above; (b) loosening
the D1 tolerance to accept the biased gradients — the bias was a defect, not a floor, and
the non-negotiable forbids tolerance changes to make things pass; (c) explicit
deflation/eigen-computation of the near-null defect modes — unnecessary once the
minimal-norm formulation removed the projection error, and far more machinery.

**Where it lives:** `scripts/diag_fft_d1.py::solve_adjoint` (the reference
implementation); to be ported into the M1 forward/adjoint module with a regression test
against finite differences (small grid).

#### D-FFT-9 closure 2 — ignition amplitude floor: KEEP the existing rollout pattern floor, now measured

**Date:** 2026-08-12. **Status:** CLOSED (this item).
**Decided by:** the implementing agent under §10, from diagnostic D2.

**The decision:** the ignition detector uses the repo's existing
`pattern_floor = max(1e-3, 0.02·|x*_0|)` (eval/rollout.py) unchanged.

**Evidence** (`experiments/diag_fft/d2/results_full.json`): on the known-Turing fixture,
saturated Turing amplitudes span 0.170–0.172 while the non-Turing contrast model decays to
≤6.7e-17 — a 2.6e15 separation. The existing floor (9.7e-3 for this fixture) sits inside
that gap with ≥17× margin to the patterned side and ~14 orders to the decayed side.

**What was rejected and why:** the geometric-mean proposal (3.4e-9) — needlessly deep into
the decayed range; a floor that low would call float-noise "patterned" under any future
noise arm. Inventing any new number when the existing, already-tested rule is measured to
sit mid-gap would be change without evidence.

**Where it lives:** the ignition gate reuses `pattern_floor` when M1 lands; noted in
`docs/DIAGNOSTICS_fft.md` §D2.

#### D-FFT-9 closure 3 — validation grid: 512² only; NO cheaper grid is licensed

**Date:** 2026-08-12. **Status:** CLOSED (this item).
**Decided by:** the implementing agent under §10, from diagnostic D2's grid-fidelity table.

**The decision:** forward validation (F1–F3 + morphology reporting) runs on the target
frame's grid, 512². No downsampled validation grid is adopted.

**Evidence** (`experiments/diag_fft/d2/results_full.json` grid_fidelity, 3 seeds × 4
grids, same L): morphology class agrees with 512² in only 1/3 seeds at 256² and 4/6 at
96²/128² (spots-vs-labyrinth flips); k* one-bin agreement fails 2/6 at ≤128². SPEC §8
pre-registered exactly this check as the licence condition for a cheaper grid; it fails.
Cost accepted: ~39 s per 512² rollout (measured, 606 steps).

**What was rejected and why:** validating at 256² (the tempting 6× saving) — it
misclassifies morphology on this very fixture. Band-limited SPECTRAL quantities (k*,
band powers) did agree at 256², which is why D4's pilot rollouts at 256² remain valid for
their band-distance purpose — but no class call below 512² may be reported.

**Where it lives:** SPEC §8 step 1 (the licence stays unexercised); `docs/DIAGNOSTICS_fft.md` §D2.

#### D-FFT-9 ledger note — zero-call threshold REMAINS OPEN; the baseline cannot calibrate it

**Date:** 2026-08-12. **Status:** OPEN (evidence recorded, no number chosen — deliberately).
**Decided by:** the implementing agent under §10, from diagnostic D5.

The pre-registered rule places the cut at the separation point of consistently-present vs
consistently-absent edge magnitudes. On the 10-seed baseline
(`experiments/diag_fft/d5/analysis.json`) that precondition does not exist: 10 seeds
produce 10 distinct sign structures and every one of the 9 edges is "mixed" at an 80 %
consensus bar — there are no consistent edges to build the two distributions from. Per the
rule's own text this is a finding (the baseline's sign structure is unstable), not a
licence to pick a cut. The calibration re-runs on the Stage-0 SPECTRAL seed set, where
stable structures are the success condition; the inherited 5 %-of-max rule stays marked
UNCALIBRATED in the meantime. The same run set fixes the R2 baseline inputs: log10
max-pairwise spreads d_ratio 0.44 (Turing subset) to KR 10.9 decades.

### D-FFT-11 — M1 spectral machinery lands INERT; the aniso normaliser follows the comparator, not the D1 prototype

**Date:** 2026-08-12. **Status:** DECIDED (implementation record for milestone M1).
**Decided by:** the implementing agents under §10 delegated authority (units U1–U4 of the
approved M1 plan).

**What landed:** `src/rngrn/forward.py` (the D-FFT-10 forward/adjoint machinery, ported
from `scripts/diag_fft_d1.py` with constants verbatim, wrapped in a
`torch.autograd.Function` whose backward runs exactly ONE minimal-norm adjoint solve on
the accumulated cotangent), `src/rngrn/losses/spectral.py` (torch RAPS with
integer-arange×dk bin edges per F-D6-1, band masks at the D-FFT-9 closure-1 edges
0.60/1.55, the five terms, ignition helpers), the side-neutral `utils.spectral_block`
port (drift-tested against `scoring.morphology._spectral_block`), and wiring through
`losses/total.py`, `config.py`, `configs/base.yaml`, `recover.py`, `history.py`,
`plotdata.py` (ARRAY_SCHEMA_VERSION 1→2), `train.py`/`export.py`. Suite: 615 passed /
1 skipped, unsandboxed.

**Decisions bundled here:**

1. **All five weights land at 0.0 — UNCALIBRATED at birth** (SPEC §5's rule). The
   default path is bit-identical to pre-M1 behaviour; the existing term-key enumeration
   assertion (`tests/test_losses.py:279`) passes byte-unmodified and pins that branch.
   Ignition gating derives from a nonzero spectral weight (the `resid`/`param_prior`
   precedent) — no separate enable flag was added.
2. **`spec_aniso` normalises the central block by the BLOCK's own power, following
   `scoring.morphology._spectral_block` — NOT by whole-spectrum power as the D1
   prototype (`scripts/diag_fft_d1.py::spectral_block_torch`) does.** The prototype's
   normaliser was never parity-checked against the comparator it claims to prototype;
   ported verbatim it fails parity at 0.3–0.6 % relative (reproduced at audit
   2026-08-12: max relative deviation 2.8e-3–5.3e-3 over 5 seeds at 96²/nblk 24). SPEC §5
   mandates the *ported comparator arithmetic*, so the comparator wins. Parity is now exact (1e-12), bound
   three ways: torch vs morphology, `utils` numpy port vs morphology (drift test), torch
   vs utils. D1's gradient verification is unaffected — it verified the IFT chain, which
   holds under either normaliser, and the suite's FD regression re-verified the ported
   chain at rel err 1.9e-10.
3. **The suite carries an IFT-vs-FD regression TRIPWIRE, tolerance 5e-3**
   (`tests/test_forward_solve.py`: 64², tracked fixture checkpoint, 1 direction × 2 term
   forms, ~42 s). This is a port-breakage detector (a wrong adjoint, sign error, or
   dropped correction produces O(1) errors), NOT a re-acceptance and NOT a weakening:
   D1's acceptance (tol 1e-4, measured 6.4e-8 at 96² × 10 directions) stands and is
   re-runnable via `scripts/diag_fft_d1.py`. Measured at port time: 1.9e-10.
4. **`ignition_margin` defaults to 1e-3**, mirroring `turing_hinges_split`'s `margin`
   default (`losses/terms.py:186`) — UNCALIBRATED, tagged in `SpectralConfig` and
   TUNING.md. Distinct from the real-space `pattern_floor` (D-FFT-9 closure 2), which
   gates patterned-ness of the completed solve, not ignition.
5. **History stability across intermittent ignition is by NaN back-fill**: skipped steps
   write `L_spec_* = NaN` (honest "not computed") plus numeric `spec_ignited` 0/1, so
   `TrainingHistory`'s frozen column set never changes mid-run; `term_vals` itself omits
   skipped keys (omitted-never-zeroed, SPEC §3). ARRAY_SCHEMA_VERSION bumped 1→2 for the
   changed column set.

**What was rejected and why:** (a) zeroing skipped spectral terms instead of omitting
them — a log could then not distinguish "no pattern yet" from "spectral loss = 0"
(SPEC §3 forbids it); (b) keeping the D1 prototype's whole-spectrum normaliser — fails
parity with the comparator the SPEC names (point 2); (c) a separate `spectral_enabled`
config flag — redundant with the weight-derived gating precedent and one more knob to
drift.

**Owner flags carried forward (not blockers, weight-0 defaults shield all three):** the
forward-solve cost at training grids is UNMEASURED (per-solve ~3–9 s at 64², an
unrecorded test timing; a relax step-budget cap is an owner call before calibration);
`real_moments`' gradient carries the F-D1-1 finite-grid translation leakage into the
combined adjoint cotangent once its weight is nonzero; and (numerics review) the
warm-started Newton chain is BRANCH CONTINUATION with no dynamical-stability check — over
a long drifting θ trajectory it can track a patterned F=0 branch that has become
dynamically unstable, and residual + amplitude floor both still pass. D1's warm-Newton
evidence covers ±ε FD pairs about a fixed base, not a long chain; a periodic fresh
re-relax cadence or a linear-stability spot check on u* is an owner decision before any
calibration run.

**Post-review hardening (same day, from the three audits — all refusals/guards, no
behaviour change on any default or spectral-identity path):** `train.fit` now threads the
five `spectral_*` knobs into `recover()` (they existed at both ends but were never
passed — the unit-C1 silent-NO-OP class, caught by the firewall audit before any run
existed); `recover()` refuses a non-identity `observed_idx` on the fitted spectral
channels (the terms compare `u_star[c]` to `frame[c]` directly — routing
`u_star[observed_idx[c]]` is deferred until a non-identity spectral run exists) and
refuses adaptive-weight strategies for spectral runs (ignition between weight refreshes
would contribute 0 then jump — the param_prior rule); `spec_shape`/`spec_amp_mean`/
`spec_amp_fluct` fail loud on zero/non-finite model-side power, mean, or std;
`raps_torch` refuses non-square fields; twelve pre-existing payload-opening `scripts/`
modules (`canon_stripes_evidence`, `exp02`–`exp12`) were added to
`tests/test_firewall.py::FORBIDDEN` (drift closure, not new code); `forward.py` imports
`THETA_NAMES` from `model.py` instead of redefining it. Final suite after hardening:
618 passed / 1 skipped, unsandboxed (the +3 over the pre-review 615 are the 512² RAPS
parity case and the two new refusal tests).

#### D-FFT-11 owner-flags closure — all three signed off by the owner

**Date:** 2026-08-13. **Status:** CLOSED (owner sign-off, in session, on the consolidated
evidence summary of the 2026-08-12 measurements).

- **Flag 1 (forward-solve cost) — CLOSED, measured.** CPU disqualified for training
  solves (fresh 938–1374 s at 96², warm-Newton 5030 s, 64 ms/step at 512²;
  `experiments/diag_fft/cost/`). GPU-native adopted (D-FFT-13): 3.25 ms/step at 512²
  fp64, CUDA chain FD-verified 4.75e-8. The 512² Newton stall (sat512: all handoff
  tolerances stall ~1.3e-4; F-D1-4) makes the TRAINING-solve grid a design choice: train
  at the finest grid where ‖F‖/‖u‖ ≤ 1e-9 is reachable, 512² stays validation-only
  (rollout detector). The coarse-grid convergence + FD measurement decides the number.
- **Flag 2 (real_moments F-D1-1 leakage) — kept as a STANDING SWEEP CONSTRAINT:**
  `w_real_moments` stays 0 in Stage-0 sweeps unless the sweep design explicitly accounts
  for the measured leakage scale (~1e-6 at 96², shrinking with grid).
- **Flag 3 (warm-Newton branch continuation) — CLOSED by redesign.** Training warm
  starts are ETDRK4 re-relax + Newton (D-FFT-13): the flow cannot converge to a
  dynamically unstable branch, so the root-tracking failure mode is designed out;
  residual stable-but-different-branch risk is caught by the fresh 512² validation
  gates; solver stalls degrade to skips (omitted-not-zeroed), a throughput cost, not a
  correctness leak. The fresh-vs-warm agreement check joins Stage-0 validation instead
  of standing alone.

### D-FFT-12 — three M2 sweep-design additions harvested from the legacy feasibility notebook (rules pre-registered; every number deliberately unchosen)

**Date:** 2026-08-12 (comparison session: `RNGRN_Diffusion_notebook_vf.ipynb` — the
pre-pipeline toy in the project root — read against this branch's SPEC/PLAN/METHODS).
**Status:** OPEN — proposed for the M2 calibration design, for owner review alongside the
M2 validation of M0+M1. Nothing here is live in code, and no item may gate anything. The
three D-FFT-11 owner flags remain the M2 blockers; this entry adds none.
**Decided by:** drafted by the implementing agent under §10 delegated authority.

**Provenance and its limits:** the legacy notebook "solved" the inverse problem only with
ground truth in the objective — the true Jacobian, homogeneous steady state, dispersion
curve and D as loss anchors (λ = 5/5/15+25/50), sign topology initialised from the true
Jacobian, θ_D initialised at the exact truth — i.e. item for item what the firewall now
forbids; and its data was the full trajectory plus exact du/dt, not one frame. Its results
are therefore PRECEDENT, never evidence, on this data: every number below is born
UNCALIBRATED and each idea enters only as a sweep axis or reported metric with the current
behaviour as its control arm. All three items are firewall-neutral (model-side quantities
only) and none touches a pre-registered pass condition in `docs/PREREGISTRATION.md`.

**Item 1 — spectral-weight ramp-in after ignition (sweep axis).** Today ignition is a
step: the first ignited step contributes the five spectral terms at full calibrated weight
(D-FFT-11, omitted-never-zeroed). Proposed axis `spectral_ramp_steps`: after ignition the
spectral weights scale by min(1, n/ramp), n counted per contiguous ignited stretch.
Pre-spec detail to fix before the sweep: whether n resets or resumes on re-ignition —
reset is proposed, since a de-ignited model has left the patterned branch. ramp = 0
(today's step) is the control. Precedent: the notebook warmed its five physics terms over
the first three epochs (`ramp = min(1, (epoch+1)/3)`) so the data term shaped the basin
first. Calibration rule: judged by the same Stage-0 criteria as the weight sweep itself
(held-out-band error, seed reproducibility) — never by smoother-looking loss curves.

**Item 2 — robustness-under-jitter as a REPORTED-ONLY basin-sharpness metric.** After a
recovery converges, draw M log-normal perturbations of the positive-domain parameters,
θ → θ·exp(s·N(0,1)) in log space, and report two fractions: still strictly Turing-unstable
(`turing_ok`), and dispersion-argmax k* within one radial bin (2π/L) of the unperturbed
model's. Model-side dispersion evaluations only — firewall-legal, no rollout, cheap.
Purpose: contextualise R1/R2 — 5/5 sign agreement on a flat basin and 5/5 on a knife edge
are different findings, and the D-FFT-9 R2 calibration needs that context recorded next to
the clustering it measures. Precedent: the notebook's 1000-draw sweep at s = 0.1 (57.7 %
remained Turing-unstable there); those numbers do not transfer. s is UNCALIBRATED — report
a small ladder of s values rather than one, and no threshold on the output exists or may
be created without a control. Never gates.

**Item 3 — per-parameter-group learning rate for θ_D (sweep axis).** Proposed axis: an LR
multiplier on the θ_D param group, with multiplier 1.0 (today's single LR) as the control.
Precedent: the notebook trained θ_D at 100× lower LR than every other group, on the stated
ground that D-ratios of order 100 are otherwise unlearnable. Consistent with — not
evidenced by — D5's invariant-combination spreads (KR up to 10.9 decades,
`experiments/diag_fft/d5/analysis.json`) pointing at conditioning as well as data
starvation. An optimizer detail, firewall-neutral.

**What was rejected and why:** (a) adopting any notebook number as a default (the 3-epoch
ramp, s = 0.1, the 100× LR split) — inherited numbers calcify (the two D-FFT-9
precedents); each idea enters as an axis against the current behaviour as control, or not
at all. (b) Porting the notebook's other machinery here: its soft-argmax k* duplicates the
existing power-weighted-centroid and lse-gap machinery; its Sobol study linearises a
DIFFERENT (non-competitive) model than the one it trained — a defect, so nothing of it is
ported; its dispersion-curve MSE against the true σ(k) is firewall-illegal by
construction. (c) Making item 2 a gate — no control exists behind any threshold on it.
(d) Folding the notebook's gate-hardening projector into this entry — it is deliberately
deferred to the frozen-gate null-refit machinery M2 already requires (SPEC's
beat-all-nulls gates), where it will be specified with that machinery, not as a sweep axis.

**Not independently validated:** nothing here has run on this data. The notebook's numbers
were measured on 2-species GM/Schnakenberg targets at 15×15 with truth anchors, and are
cited only as the reason these axes are worth sweeping at all.

**Where it lives:** nowhere in code yet — this entry pre-registers the rules. On adoption:
items 1 and 3 join the M2 sweep grid (`guarded_run.sh` sweeps on
`turing_labyrinth/sample_0000`), item 2 lands as a side-neutral model-side reporter with
reported-only columns; settled values append as D-FFT-12 closures, per the D-FFT-9
pattern.

### D-FFT-13 — training warm starts are ETDRK4 re-relax + Newton (warm_mode="relax"); Newton-only stays the FD-instrumentation contract; the forward solve gains a CUDA path with a torch-LSMR minimal-norm solver

**Date:** 2026-08-12. **Status:** DECIDED (§10, from the cost measurements and the owner's
GPU direction).

**The decision, three parts:** (1) TRAINING warm solves (recover.py) run
warm_mode="relax": a short ETDRK4 re-relax from the previous u* (budget
warm_max_chunks=40, UNCALIBRATED) then Newton polish, falling back to fresh on failure.
D1's Newton-ONLY warm rule was an FD-instrumentation constraint — phase pinning for
finite differences (F-D1-1); training's losses are translation-invariant, so warm-relax
phase drift is harmless there. (2) `PatternSolver`'s own default stays
warm_mode="newton" — the D1/FD contract, pinned byte-unmodified by the existing tests.
(3) The solver derives its device from the model; on CUDA the relax runs the D2-verified
torch integrator (`rngrn/etdrk4_torch.py`) and Newton/adjoint run a faithful torch port
of scipy's LSMR (conlim=1e8 stop included) inside the unchanged minimal-norm +
true-residual refinement wrapper — D-FFT-10 semantics constant-for-constant.

**Evidence:** warm-Newton at Adam-scale θ displacement 5030 s vs fresh 938 s at 96² CPU
(`experiments/diag_fft/cost/run_attempt1.log`); CUDA integrator 3.25 ms/step at 512²
fp64 (`experiments/diag_fft/gpu_probe/results.json`); torch-vs-scipy LSMR parity 7.8e-15
and bit-matching stall residuals near-singular (tests); CUDA-path FD regression 4.75e-8
vs the 5e-3 tripwire (tests, printed); CUDA 512² fresh solve attempt: the relax exhausts
its 200k-step budget without satisfying the flat_tol=1e-4 saturation detector (683.9 s,
solve_failed) — the 512² relax criterion is UNCALIBRATED and is the blocking knob for
512²-native training (measurement in flight; see the sat512 sweep when it lands).

**What was rejected and why:** (a) warm_mode="relax" as the PatternSolver-wide default —
it would silently change the FD instrumentation contract the suite pins; (b) any
projection/deflation in the torch solver (D-FFT-10 forbids); (c) dropping scipy's conlim
stop from the port — part of the verified stopping behaviour; (d) loosening flat_tol to
make the 512² one-shot pass — that is a calibration measurement, not an implementation
call.

**Where it lives:** `src/rngrn/forward.py`, `src/rngrn/etdrk4_torch.py`, `recover.py`
(warm_mode threading), `losses/spectral.py` (device joins), `tests/test_etdrk4_torch.py`,
`tests/test_forward_solve.py`.

### D-FFT-14 — PROPOSED ablation arm: parameter-noise ("weight noise") during training as an annealer and robustness prior

**Date:** 2026-08-12. **Status:** ADOPTED into the redesign as an ablation arm, 2026-08-17
(D-REDESIGN-2; `docs/REDESIGN_rngrn.md` §4.7, R4) — under this entry's rules verbatim, with
schedule and magnitude still UNCALIBRATED and to be pre-registered here before the arm
runs. Originally PROPOSED by the owner (not designed into any stage, not run; recorded so
the idea survives the session). Adoption went through the register, not a silent default,
because it changes what every recorded number means.

**The proposal:** inject annealed noise into the raw parameters θ (or the gradient,
SGLD-style) during training. Two distinct payoffs, to be evaluated separately:
(1) *optimization annealing* — D5 measured ten seeds falling into ten distinct
sign-structure basins; if good mechanisms occupy wider basins, noise concentrates seeds
into them, which is what the R1 gate (5/5 identical sign structures) needs — an empirical
claim about THIS landscape, not a given; (2) *mechanism-robustness prior* — noise
penalises sharp minima, i.e. selects θ whose Turing instability and pattern survive
parameter perturbation; biologically motivated (network-atlas work ranks topologies by
Turing-viable parameter volume), and the hinge margins are already a cheap cousin. A
SAM-flavoured variant on the LINEAR-THEORY terms only (dispersion evaluated at perturbed
θ) costs milliseconds and no extra forward solves.

**Placement constraints (from the 2026-08-12 cost measurements):** noise must NOT touch
the θ the forward solve sees — warm-start cost is sharply sensitive to per-step θ
displacement (warm-Newton 5030 s pathology), and noise near the ignition margin would
toggle the Turing gate (ignition chatter, wasted solves). Inject in the gradient or the
cheap linear-theory evaluations; the solve runs at the clean current θ.

**Gate integrity:** noise that merely smooths the landscape could raise seed agreement
(R1) without raising truthfulness — the held-out-band gates and channel co-gates are the
defence (D4 measured them non-vacuous), and this arm must be judged on those alongside
R1, never on R1 alone.

**Not to be conflated with:** the PLAN's existing "noise arms", which are OBSERVATION
noise on the data (estimator robustness) — a different axis. Related but distinct:
D-FFT-12 item 2 measures basin sharpness post-hoc as a reported-only metric; this
arm injects noise DURING training.

**A/B protocol when run:** the 10-seed Stage-0 protocol, arm vs matched no-noise control;
metrics: R1 agreement count, Turing fraction, held-out-band distances vs the D4 null
spread; noise schedule and magnitude pre-registered in this ledger before the run.

**Where it lives:** the files above; tests `test_forward_solve.py`,
`test_spectral_terms.py`, `test_spectral_utils.py`, `test_ignition_gating.py`, plus the
enumeration-contract test in `test_losses.py`. CLAUDE.md §7c amended in the same change
(training MAY now simulate — ignition-gated, off by default).


---

### D-PERF-2 — agent model routing is pinned in frontmatter, not left to inheritance; `repo-mapper` stays sonnet

**Date:** 2026-08-13. **Status:** DECIDED (§10; owner-set routing, recorded here because it
changes what a review costs and, for `repo-mapper`, reverses an existing pin).

**The decision.** `CLAUDE.md` §11 now carries a role → model table and
`.claude/rules/orchestration.md` its dispatch form. Four of the five agents in
`.claude/agents/` pinned **no** `model:`, so they inherited the session model. With the session
model moving to the planning tier (`fable`), every firewall, evidence and numerics audit would
have silently run on the planner. Pinned instead: `firewall-auditor`, `evidence-auditor`,
`numerics-reviewer` → `opus`; `merge-damage-hunter` → `sonnet`; `repo-mapper` **stays**
`sonnet`.

**Evidence.** Inheritance was verified against the frontmatter (only `repo-mapper` pinned a
model). The tier assignments follow the owner's routing: a firewall breach does not crash — it
produces a *better-looking result* — so those three take the top tier, while
`merge-damage-hunter`'s own brief says "find it mechanically, not by eye" and
`repo-mapper`'s says "You answer where is X and what calls it. Nothing else… Never edit."
`CLAUDE.local.md` records the owner preference that mechanical Explore/mapping subagents run on
a cheap model.

**What was rejected.** Leaving the four unpinned and relying on the dispatcher to pass `model`
per call — rejected because it fails silently and exactly on the highest-risk reviews.

**`repo-mapper` was briefly dropped to `haiku` and reverted the same day, by owner decision.**
The case for `haiku` was `CLAUDE.local.md`'s "mechanical Explore/mapping subagents run on a
cheap model". The case against, which won: `superpowers:subagent-driven-development` warns the
cheapest tier routinely takes 2-3x the turns on multi-step work and costs more overall, and
`repo-mapper` does multi-step graph queries against `graphify-out/` with every hit verified
against source — the multi-step case that warning names. Its existing `sonnet` pin was
therefore deliberate, and is now recorded as such rather than resting on nobody having
questioned it. **UNCALIBRATED:** no turn-count or cost measurement was taken on either side;
the decision rests on the SDD guidance and the agent's own task shape, not on a measurement
here. The `CLAUDE.local.md` line is narrowed to mean single-shot mechanical Explore, not this
agent.

**Also superseded:** `CLAUDE.local.md`'s "judgement agents keep the default". Written under an
Opus session, "the default" meant the strong model; under a planning-tier session it would mean
the planner. §11's table wins.

**Where it lives:** `CLAUDE.md` §11, `.claude/rules/orchestration.md`, and the `model:` line in
each of the five files under `.claude/agents/`.

### D-LIFT-1 — the finite-μ dynamical lift becomes the verification gate, after a validation ladder; supersedes D-FFT-4

**Date:** 2026-08-17 (redesign session, branch `docs/redesign-rngrn`). **Status:** DECIDED
**Decided by:** the implementing agent under explicit owner delegation, given in the same
session: the owner's redesign brief named verification "by simulating the learned parameters
with the dynamical lift", and on the register the owner said *"you decide on the
Owner-decision register, I trust you."*

**The decision:** forward verification of a recovered model gates on the lifted system at
μ_central = 7.2e-4 — L1 `eval/lifted.py::turing_verdict_lifted` strict-Turing AND stationary
(Hopf excluded); L2 `simulate_lifted` patterns (existing `pattern_floor`, horizon-stop, dt
policy min(0.2/jac_rate, μ/2) with a dt-halving check); L3 k* of the lifted rollout within
one radial bin `|k*_lift − k*_obs| ≤ 2π/L` (12.5% at the target's p=8, per D-FFT-3) — with
the L1 verdict, `mu_critical`, and `robustness_vs_mu` *reported* across the biological band
[1.1e-5, 9.2e-3]. QSS F1–F3 remain co-gates. **The gate binds nothing until the V0–V4
validation ladder of `docs/REDESIGN_rngrn.md` §5.3 passes** — that ladder is the structural
answer to D-FFT-4's numerics-risk rationale, which was correct and is not being overridden,
only satisfied. Preregistration §3.7 (added 2026-08-17, additive) is the binding form.

**Evidence:** the lift's fixed points are μ-independent and its machinery exists
(`eval/lifted.py`: `rescale_mu`, `lifted_dispersion`, `turing_verdict_lifted`,
`simulate_lifted` with the exact gate substep); the one measured μ-scan (`eval/lifted.py`
"Measured" note, n=1) shows re-entrance with the biological band inside the first Turing
window; validation status and the coupling trap (dt ≫ μ silently re-imposes QSS) are stated
in `docs/REDESIGN_rngrn.md` §5.2–§5.3. No gated number exists yet under either regime that
this changes.

**What was rejected and why:** (a) keeping the QSS rollout as the sole gate (D-FFT-4) — the
owner's stated verification target is the lift, and QSS-Turing does not imply lifted-Turing
at a given μ (measured re-entrance); (b) gating band-wide across μ — a conjunction over a
three-decade uncertainty band whose edge points can flip on the least-certain digit of a
literature ratio (revisited after V4); (c) gating before the ladder — exactly the artefact
risk D-FFT-4 named.

**Not independently validated:** the lift itself — that is what V0–V4 exists to do; the
ladder's own tolerances are UNCALIBRATED where marked in §5.3.

**Where it lives:** `docs/REDESIGN_rngrn.md` §5; `docs/PREREGISTRATION.md` §3.7;
`src/rngrn/eval/lifted.py` (unchanged as of this entry — implementation is milestone R1).

### D-REDESIGN-1 — the REDESIGN_rngrn owner-decision register is ratified; per-item rulings recorded

**Date:** 2026-08-17 (redesign session, branch `docs/redesign-rngrn`). **Status:** DECIDED
**Decided by:** the implementing agent under explicit owner delegation (*"you decide on the
Owner-decision register, I trust you"*), after interactive owner review that itself set
three of the items (single 3-node target scope, learnable signs, the §4.6 data protocol
including the small-box extension and the estimator policy).

**The decision:** all 15 items of `docs/REDESIGN_rngrn.md` §8 are ratified as written
there. Rulings made under the delegation, with rejected alternatives: **item 6** — gate at
μ_central, band reported (rejected: band-wide; see D-LIFT-1); **item 8** — adjoint-primary
gradient policy with the unrolled-path promotion rule pre-specified (FD-faithful at 1e-4 on
converged AND stalled members, cost not exceeding the adjoint's; rejected: leaving promotion
to a post-hoc choice after results exist); **item 13** — the unit of independence for
preregistration §3.1/R1 is the full pipeline replicate, K_rep = 5 independent master seeds,
population members never counted as seeds (rejected: counting culled members as seeds — the
cull selects for sign-structure distinctness, biasing exactly the statistic §3.1 measures);
**item 3** — hard bio-boxes for α/δ with the §3.4 amendment note making those rows
`structural`, soft prior for the D-ratio (rejected: a hard D-ratio box — the generator
population's median ~135 sits outside [1, 60] by the recorded bio_box decision, and a hard
box would make the synthetic target unrepresentable); **item 15** — the Hann–Welch windowed
spectral estimator enters report-only under the both-sides-identical rule (rejected:
swapping the primary estimator, which would silently change what k*_obs and every spectral
target means).

**Evidence:** each item's grounding is cited at its section of `docs/REDESIGN_rngrn.md`
(a document revised against a full evidence audit the same day; the audit verified every
carried-over number against its source). The register's UNCALIBRATED numbers (item 14)
remain UNCALIBRATED with their calibration rules named — ratification fixes rules, not
values.

**What was rejected and why:** treating the delegation as blanket authority to weaken
pre-registered conditions — it is not read that way. §3.7 is additive; the §3.4 amendment
narrows two rows to `structural` while the criterion's verdict continues to read on the
still-measured rows (d_ratio; β stays UNCITED/unscored); nothing in §3.1–§3.5 is loosened,
and any future change that would weaken a bar still goes to the owner by name.

**Where it lives:** `docs/REDESIGN_rngrn.md` §8; `docs/PREREGISTRATION.md` §3.4 amendment
note and §3.7.

### D-REDESIGN-2 — weight-noise adopted as a redesign ablation arm under D-FFT-14's rules; loss registry, training telemetry and pipeline/plots notebooks added as infrastructure

**Date:** 2026-08-17 (redesign session, branch `docs/redesign-rngrn`). **Status:** DECIDED
**Decided by:** the implementing agent under the same owner delegation as D-REDESIGN-1;
the weight-noise request, the loss registry, the telemetry/visualisation requirement and
the two notebooks were owner-requested in this session.

**The decision:** (1) parameter noise during training enters the redesign as an
**ablation arm at R4**, governed by D-FFT-14 verbatim — injection in the gradient or the
linear-theory evaluations only, never the θ the forward solve sees; matched no-noise
control at identical seeds and budget; judged on held-out-band distances and channel
co-gates alongside seed agreement, never agreement alone; noise schedule and magnitude
UNCALIBRATED until pre-registered here before the run. (2) Loss terms move onto a
registry built on `rngrn/registry.py::Registry` — per term: serial callable, batched twin
or explicit refusal, default weight, calibration tag — with an enumeration-contract test;
`DEFAULT_WEIGHTS` and `compute_terms{,_batched}` enumerate it. (3) `TrainingHistory`
extends to the population loop (ignition/de-ignition/stall/cull/death events, per-term
traces, canonical-gauge invariant trajectories alongside raw θ), with a `viz` module and
two thin-driver notebooks: `notebooks/redesign_pipeline.ipynb` (full pipeline; trainer
launches wrapped in `scripts/guarded_run.sh`) and `notebooks/redesign_plots.ipynb`
(figures from a finished run directory). Items (2)–(3) are engineering: they carry no
thresholds and change no number's meaning.

**Evidence:** D-FFT-14's recorded cost measurements (warm-Newton 5030 s pathology;
ignition chatter) fix the injection placement; D5's ten-distinct-basins measurement is
the annealing motivation; `history.py` already records per-step per-member parts and
parameter vectors, so the extension is incremental; `rngrn/registry.py` already provides
the registry primitive.

**What was rejected and why:** weight noise as a default (D-FFT-14's own prohibition —
it changes what every recorded number means); recording only raw θ trajectories (gauge-
ridden; convergence is only visible in the §3.4 invariant coordinates); science logic in
notebook cells (notebooks are thin drivers over library code, or their results are not
reproducible from the run directory).

**Not independently validated:** nothing here is run yet; the notebooks and telemetry are
R2 deliverables, the noise arm an R4 deliverable.

**Where it lives:** `docs/REDESIGN_rngrn.md` §4.7–§4.8, §6, §7 (R2, R4), §8 item 16.

### D-R2-1 — the box-sigmoid reparameterization silently RE-CENTRES the default init: α 7.6× and δ 3.9× higher than legacy

**Date found:** 2026-08-18 (Task 16, branch `feature/redesign-model`).
**Status:** OPEN — measured, deliberately NOT fixed here (see "what was rejected").

**The defect.** `RNGRN.__init__` draws `theta_alpha ~ N(0, 0.5)` and `theta_delta ~ N(0, 0.3)`
— raw scales chosen for the SOFTPLUS map, where `softplus(0) = 0.693`. Task 13's
`param_boxes` replaces that map with `low + (high−low)·sigmoid(raw)`, where a raw near 0
lands at the box **midpoint**. The raw init distribution was not adjusted, so switching the
box on moves the physical starting point without any caller asking it to.

**Measured**, over the exact 512 member seeds the Phase-I run uses
(`recover._restart_seed(0, r)`, r = 0..511; N=3, form=competitive, init='default',
box = `configs/bio_box.yaml` α∈[0.5,10], δ∈[0.4,5]; 4608 α entries and 1536 δ entries):

| parameter | legacy (softplus) median [5th, 95th] | boxed median [5th, 95th] | ratio |
|---|---|---|---|
| α | 0.692 [0.365, 1.194] | 5.246 [3.407, 7.121] | **7.58×** |
| δ | 0.698 [0.468, 0.982] | 2.710 [2.121, 3.277] | **3.88×** |

**Why it matters for Turing, mechanistically.** The Jacobian diagonal is
`J_ii = ∂prod_i/∂x_i − δ_i`, and Turing instability requires at least one positive diagonal
(`docs/STATE_OF_THE_SCIENCE.md` §10: 0/200 default-init diagonals are positive, 88/88 true
systems' are). The box **floors δ at 0.4 and starts it at 2.71**, so self-activation must
overcome a decay rate ~3.9× larger than under the legacy init before the optimiser has moved
anything. Measured Jacobian diagonals at the pinned x* for three boxed members are −2.90 /
−3.01 / −1.99, −2.18 / −2.33 / −1.65 and −2.68 / −1.34 / −2.92 — uniformly negative and of
the order of the re-centred δ.

**This is the same argument Task 13 already accepted for a different init.** `RNGRN.__init__`
refuses `param_boxes` with `init='low_basal'` on the grounds that "the same raws would land
at a DIFFERENT alpha/delta value under a box, silently defeating the low-basal Jacobian
diagonal prior it exists to provide". The identical reasoning applies to `init='default'`,
where it was not caught because the default init carries no *stated* prior to defeat — only
an unstated one, namely the scales `TUNING.md` records as "they set where recovery starts".

**What was rejected and why.** Fixing it inside Task 16 was rejected: the Phase-I run's job
is to measure the code as reviewed, and re-centring the init mid-task would have made the
measured ignition rate a number about an unreviewed init rather than about the ratified
design. The two candidate repairs, both untested, for whoever takes this:
(a) shift `theta_alpha`/`theta_delta` by `logit((legacy_median − low)/(high − low))` so the
boxed init starts where the legacy init did — cheap, and keeps the box's guarantee; or
(b) declare the box midpoint the intended starting point and re-derive the raw scales from
the box — defensible, but it changes what every boxed number means and needs its own
control arm. Choosing between them is a science decision with a live pre-registration
consequence (§3.3 makes α/δ plausibility `structural`), so it is RETURNED, not ruled.

**Where it lives:** `src/rngrn/model.py::_box_sigmoid` and `RNGRN.__init__` (the raw draws);
`docs/REDESIGN_rngrn.md` §3.3; the measurement is reproducible from the table above with no
run directory required (it is a property of the init, not of a run).

### D-R2-2 — Phase-I run configuration: the four ignition weights, the arms, and the device/backend

**Date:** 2026-08-18 (Task 16). **Status:** DECIDED (every threshold UNCALIBRATED, marked).
**Decided by:** the implementing agent under CLAUDE.md §10.

**The decision, and what each part rests on.**

1. **Ignition weights `kstar_si=1.0, turing=1.0, param_prior=1.0, beta_hinge=1.0`. ALL FOUR
   UNCALIBRATED.** `kstar_si` inherits `kstar`'s shipped 1.0 verbatim so the §4.4 swap is a
   substitution and not simultaneously a re-weighting; `turing` is unchanged from the shipped
   default and from D5's frozen config; `param_prior` is the §3.3/§4.4 "promoted to nonzero
   weight" with no calibrated value in existence, so 1.0 is the neutral promotion; `beta_hinge`
   is born with this task and has no prior art at all. Rejected: tuning any of them before the
   first measurement existed, which would have made the first Phase-I number a tuned one.
2. **A0 control = D5's own weight dict, quoted** from
   `experiments/diag_fft/d5/target_reports.jsonl`'s `frozen_config`, including its inert
   `morphology: 0.1` — so the control is the objective the baseline row was measured under
   rather than a re-derivation of it. Rejected: using `terms.DEFAULT_WEIGHTS` (which carries
   `morphology: 0.0`), because then the control would differ from D5 by an edit nobody asked
   for, however inert.
3. **Budget 1500 Adam steps, lr 0.05, grad-clip 10.0, no LBFGS, both arms**, per
   `docs/REDESIGN_rngrn.md` §4.5. This is NOT D5's budget (2000 Adam + 50 LBFGS, serial), so
   A0 is the matched control and D5 is context, never the control. Rejected: matching D5's
   budget instead, which would have unmatched the two arms actually being compared.
4. **`dispersion_backend='cubic'` on CUDA.** The `eig` backend has no batched cuSOLVER kernel
   for small non-symmetric matrices (~700 µs per matrix, flat in B — measured 2500× worse
   than cubic at B=8, `model.BatchedRNGRN` docstring), so `eig` on CUDA is refused by the
   driver rather than merely discouraged. `cubic` is exact for N=3 and was validated against
   `eig` on 127 real Jacobians (σ_max MAE 9.2e-13, k* MAE 0, 0/127 verdict flips), so this
   changes the arithmetic path but not the answer. D5 ran `eig`; the difference is recorded.
5. **Ignition EVENTS use the training k-grid; the reported Turing FRACTION uses
   `eval.analysis.turing_ok`.** The event flag is `sig0 < 0 and sig_max_pos > 1e-3` read off
   the quantities `turing_hinges_split_batched` already computes on `recover._kgrid_for`'s
   k*_obs-anchored grid; the final verdict is the strict D-EVID-11 criterion on `turing_ok`'s
   own `linspace(1e-3, 50, 4000)`. The two can disagree near the boundary, so only the second
   is ever reported as a Turing fraction. Rejected: reporting the cheap in-loop flag as the
   headline, which would not be comparable to D5's `turing_frac`.
6. **`n_distinct_structures` / `topology_consistency` use `scoring.reproducibility` at
   `sign_zero_rtol = 0.05`** — the same function and the same threshold that produced D5's
   10/10 and 0.1. Rejected: `recover._topology`'s KA/KR-gate signs, which are a different
   quantity and would not be comparable to the D5 row.
7. **History (the `arrays/plot_arrays.npz` trajectory) is recorded for the LARGEST B only, at
   stride 100.** An npz per sweep rung would add tens of MB to a tracked tree for a curve
   whose scientific content is one scalar per rung. Rejected: recording every rung.

**Where it lives:** `scripts/r2_ignition_run.py` (the constants carry these reasons at their
definition); `experiments/redesign_r2/phase1/` (the runs); `docs/HANDOFF_redesign_r2.md`.

### D-R2-3 — the R2 ignition objective does not ignite: the pinned, β-derived model makes a DECOUPLED network free, and `param_prior` at 1.0 overwhelms the k\* anchor

**Date:** 2026-08-19 (Task 16, the first Phase-I population run).
**Status:** OPEN — two causes measured and attributed; neither fixed here (Task 16's job was
to measure the design as ratified, not to redesign the objective mid-measurement).

**The measurement.** Phase-I, `turing_labyrinth/sample_0000`, CUDA, cubic backend, 1500 Adam
steps at lr 0.05, member seeds `recover._restart_seed(0, r)` shared across arms. Run dirs
under `experiments/redesign_r2/phase1/` and `.../phase1_ablation/`.

| arm | B | Turing (strict, `eval.analysis.turing_ok`) | distinct sign structures | median `kstar_fft_rel_err` |
|---|---|---|---|---|
| r2 (pin + box + prior) | 64/128/256/512 | **0** at every B | 3 at every B | 0.9769 |
| a0 (legacy objective, control) | 64 | 3 (4.7 %) | 44 | 0.9769 |
| a0 | 128 | 11 (8.6 %) | 62 | 0.9769 |
| r2_nobox (pin, softplus α/δ) | 64 | 0 | 3 | 0.9769 |
| r2_noprior (pin + box, prior 0) | 64 | 0 | 5 | **0.0332** |

**Cause 1 — `param_prior` at the promoted weight 1.0 wins against `kstar_si`.** Removing it
moves the median k\* error from 0.977 to **0.0332 (29×)**. The B=512 invariant trace shows
`d_ratio` driven to **exactly 7.500** (5th and 95th percentile both 7.5000 over 512 members)
while `L_kstar_si` moves 0.0809 → 0.0791 (**2 %**) across the whole run. `docs/REDESIGN_rngrn.md`
§3.3 promoted the prior on the stated grounds that "the prior's role is to make the viability
tension measurable, not to fix a failure" — the measurement is that at weight 1.0 it does not
measure the tension, it resolves it in the prior's favour and takes the wavenumber with it.
The generator's own D-ratio median is ~135 (`configs/bio_box.yaml` header) against the prior's
centre 7.5, so the two objectives are pulling in opposite directions by design; what was not
anticipated is the margin.

**Cause 2 — the decoupled network is an exactly feasible, unpenalised optimum.** Ignition fails
in all three r2 variants, so neither the box nor the prior causes it. Measured at B=512:
median `max|off-diag J| / max|diag J|` = **0.0027** for r2 against **0.582** for a0; the binding
budget `s = KA+KR` collapses 0.314 → **0.0133**; promoter occupancy collapses 0.0244 → **5.7e-05**;
and β/δ converges to **0.4193 / 0.2674 / 0.3504** against the pin x\* = (0.41938, 0.75467,
0.35046) — a 0.03 % match on species 0 and 2. That is the analytic signature: as `prod(x*) → 0`
the derived β is exactly `δ_i·x*_i`, which satisfies the pinned fixed point exactly, clears the
β ≥ 0 hinge comfortably, and costs nothing in any other term. **No term in the R2 ignition
objective requires non-zero cross-regulation**, and §3.2/§3.3 retired the two terms
(`frame_scale_anchor`, `anticollapse`) that incidentally discouraged the trivial network.

**`anticollapse` would not have caught it, computed rather than assumed.** It floors ‖J‖_F at
`jac_floor = 1.0`; the r2 arm's ‖J‖_F is **6.66**, so `softplus(1.0 − 6.66) ≈ 0.003` — inactive.
§3.3's structural argument for retiring it ("box-confined rates cannot let ‖J‖ collapse to
zero") is correct about ‖J‖ and *irrelevant to this failure*: what collapses is the
off-diagonal, through the binding budget `s`, which §3.3 explicitly leaves unboxed ("D and s/g
are NOT boxed in this design"). This does not reopen owner-decision item 11 in `anticollapse`'s
favour — it says the term that is missing bounds COUPLING, not NORM, and no such term has ever
existed in this repo.

**What was rejected and why.** (a) Adding a coupling term inside Task 16 — rejected: the run's
purpose was to measure the ratified objective, and inventing a term mid-run would have produced
a number about an unreviewed objective. (b) Extending the budget past 1500 steps to chase
ignition — rejected: `sig_max_pos` rises −1.149 → −0.0329 and does approach the +1e-3 margin,
but a longer run cannot fix an objective whose optimum is the trivial network, and §4.5's
budget-saturation citation is on a different sample family and flagged there as not
like-for-like. (c) Reporting r2's "3 distinct structures vs D5's 10" as a reproducibility
improvement — rejected as actively misleading: it is a collapse onto the trivial solution, and
its sign structures are diagonal-only.

**Not independently validated:** the three candidate repairs named in
`docs/HANDOFF_redesign_r2.md` §7 (an off-diagonal ‖J‖ floor, an occupancy floor, boxing `s`)
are untested; so is any claim about what the redesign arm would do with them.

**Where it lives:** `experiments/redesign_r2/phase1/`, `experiments/redesign_r2/phase1_ablation/`;
`docs/HANDOFF_redesign_r2.md` §4c; `docs/REDESIGN_rngrn.md` §3.2, §3.3, §4.4, §4.5.

---

### D-PERF-3 — `dispersion_backend` gains `'auto'`, which resolves to the closed-form cubic at N = 3 (MECHANICS adopted on this line; the DEFAULT stays `'eig'` by ruling — see the R3 amendment at the end of this entry, and D-PERF-10)

**Date:** 2026-08-17 (GPU-optimisation branch `feature/gpu-optim`, orchestrating session).
**Status:** DECIDED
**Decided by:** the orchestrating session under delegated authority (§10).

**The decision:** `RNGRN.__init__`, `BatchedRNGRN.from_seeds`, `config.ModelConfig.dispersion_backend`
and `recover()` all default to `'auto'`, resolved at construction to `'cubic'` when `N == 3`
and `'eig'` otherwise. The resolved concrete value is what `.dispersion_backend` reads —
`'auto'` never survives construction. Explicit `'eig'`/`'cubic'` are respected unchanged, and
explicit `'cubic'` still raises for N ≠ 3.

**Evidence:** measured in this repo and recorded in `model.py`'s docstrings and `CLAUDE.md` §7:
batched cubic dispersion 0.97 ms vs `torch.linalg.eigvals` 156.6 ms on 127 matrices (162×);
on CUDA the eig backend costs ~816 ms per restart-step flat in B (~2500× the cubic at B=8),
because small non-symmetric eigendecomposition has no batched cuSOLVER kernel. The cubic form
is exact for N ≤ 3 by construction; per-step agreement with eig was previously measured at
~1e-12. The dispersion relation is evaluated twice per optimiser step, so this is the dominant
kernel cost of every N = 3 run on either device.

**What was rejected and why:** (a) keeping `'eig'` as the default and only raising loudly on
the eig+CUDA+N=3 combination — that guards the GPU path but leaves every CPU run misrouted
away from a measured 162× win for no benefit; (b) changing the default to a bare `'cubic'` —
breaks every N ≠ 3 construction site and every reference-system test at N = 2.

**Announced loudly:** runs recorded after this change are **not bit-comparable** to runs
recorded before it for any N = 3 config that did not pin the backend explicitly — cubic and
eig agree to ~1e-12 per step but diverge over a full optimisation. Any cross-branch comparison
of recovered parameters must pin `dispersion_backend` explicitly on both sides.

**Not independently validated:** the ~1e-12 agreement figure is the previously measured one
(`model.py`); no fresh cubic-vs-eig full-run comparison was made in this session, and the test
suite had not yet been re-run when this entry was written (owner deferred testing to the end
of the branch).

**Where it lives:** `src/rngrn/model.py::RNGRN.__init__` (resolution), `::BatchedRNGRN.from_seeds`,
`src/rngrn/config.py::ModelConfig`, `src/rngrn/recover.py::recover`.

**R3 INTEGRATION AMENDMENT — 2026-08-19, `feature/r3-integration` Task 8.** The paragraphs
above are transcribed verbatim from `feature/gpu-optim-repair`, because they are the record of
the decision as taken there. **On this line the decision landed only in part, by controller
ruling** (`docs/INTEGRATION_r3_collisions.md` row 26 / §2.8), and the difference is
load-bearing:

- **ADOPTED — the mechanics.** `'auto'` is accepted by `RNGRN.__init__`'s three-way assert and
  resolved at construction by `model.resolve_dispersion_backend` (`model.py:42`, called at
  `model.py:251`). `self.dispersion_backend` is never the string `'auto'`. Explicit
  `'eig'`/`'cubic'` are respected; explicit `'cubic'` at N ≠ 3 still raises.
- **NOT ADOPTED — the default flip.** `RNGRN.__init__`, `BatchedRNGRN.from_seeds`,
  `ModelConfig.dispersion_backend` and `recover()` all still default to `'eig'`.
  `docs/PLAN_redesign.md`'s Global Constraints require the A0 baseline objective to keep
  bit-identical behaviour after every task, and this entry's own "Announced loudly" paragraph
  says an `'auto'` default makes N = 3 runs not bit-comparable. The two cannot both hold, so
  the flip is parked as a decision point returned to the controller, not taken here.
  **RULED 2026-08-19: the flip is REJECTED at integration — see D-PERF-10**, which records the
  four facts and the rejected alternatives.
  **Consequence: nothing in the paragraph above describing `'auto'` as "the default" is true on
  this line.** It is an opt-in.
- **ADOPTED and NOT in the original — the frozen-config fix.** `train.fit` resolves
  `cfg.model.dispersion_backend` through the same helper *before* writing
  `config/frozen_config.yaml` (`train.py:205-212`). Without it a run configured `'auto'` froze
  the literal string `'auto'`, so the file could not say which backend produced the number —
  against `.claude/rules/reporting-numbers.md` step 4, which says to read the frozen config
  rather than re-derive from it. This was authorised whichever way the flip ruling went.

**Pinned by:** `tests/test_dispersion_cubic.py::test_auto_resolves_at_construction_and_never_survives_it`
(the mechanics), `::test_the_default_backend_is_eig_everywhere` (the ruling — all four
defaults, plus the resolved effect at N = 3, so a flip must edit the test in the same commit),
and `::test_frozen_config_records_the_resolved_backend_not_the_request` (the provenance fix).

**Not independently validated (this line):** no cubic-vs-eig full-run comparison was made
here either — the 162x and ~1e-12 figures above are inherited, not re-measured. The
frozen-config test reproduces `train.fit`'s resolve-then-write sequence rather than running
`fit` (which needs a dataset); the ORDERING inside `fit` is pinned by a source-order
assertion, which is weaker than a behavioural test and is labelled as such at the site.

---

### D-PERF-4 — the batched-spectral refusal stays deleted; explicit entry-point validation replaces what it used to provide implicitly

**Date:** 2026-08-19 (R3 repair task 1, `feature/gpu-optim-repair`, on top of
`feature/gpu-optim`). **Status:** DECIDED
**Decided by:** the implementing agent under delegated authority (§10).

**The decision:** `losses/total.py::compute_terms_batched` no longer refuses a non-None
`spectral` argument unconditionally, and that refusal is NOT restored. In its place, two
explicit checks run at the top of the function, before any steady-state solve or Jacobian:
(1) `model` must be a `model.BatchedRNGRN`, else `ValueError` naming the offending type; (2)
when `spectral is not None`, `spectral.solver` must expose `solve_subset` (the
`forward.BatchedPatternSolver` contract), else `ValueError` naming the offending type and
`solve_subset` by name. `total_loss_batched` inherits both checks unchanged, since it only
ever calls through `compute_terms_batched`.

**Why the refusal's original reason no longer holds:** the deleted text (`10cff1b`) said
"`forward.PatternSolver` owns per-restart warm-start state and cannot be shared across a
batched member axis" — true when written, because no batched forward solve existed.
`feature/gpu-optim`'s `2b11010` added `forward.BatchedPatternSolver`, which owns
per-MEMBER warm state keyed by global member index and solves only the subset of members
ignited at a given step (`_apply_spectral_batched`, `losses/total.py:188`). This is exactly
the batched forward solve `docs/REDESIGN_rngrn.md` §4.1 names as R3's critical path. The
structural obstruction the refusal existed to enforce is gone; keeping the refusal would make
that critical-path combination permanently unreachable through this API, which is the
opposite of what R3 needs.

**Evidence:** `docs/REVIEW_gpu_optim_delta.md` C1 diagnosed the gap: `10cff1b..2b11010`
deleted the refusal but added no replacement, leaving `compute_terms_batched`/
`total_loss_batched` red (`AttributeError: 'NoneType' object has no attribute 'B'` on a
`model=None` placeholder — reproduced here, RED, before this fix: 2 failed) and leaving
`tests/test_ignition_gating.py::test_recover_raises_on_batched_with_a_spectral_weight`
**passing vacuously** — it asserts `pytest.raises(ValueError, match="batched")` without
passing `lbfgs_steps`, so it trips `recover()`'s unrelated LBFGS guard
(`"batched=True requires lbfgs_steps=0"`), whose message happens to contain "batched". Passing
`lbfgs_steps=0` explicitly (this task's Step 2, scratch, not committed) makes that same call
raise **nothing at all** — `recover(batched=True, spectral weight != 0, lbfgs_steps=0)`
completes cleanly, which is independent confirmation (from `recover.py`'s own code path, not
just its docstring) that the combination is legal today. `recover.py:396-401`'s own docstring
already states this in-place ("This combination USED TO RAISE … and the refusal is now
DELETED because the solve gained a member axis, not relaxed because the reason stopped
mattering") and `_batched_restarts` (`recover.py:238-248`) wires a real
`forward.BatchedPatternSolver` into `SpectralContext` whenever `spec_cfg is not None`. The
two-line entry-point check was verified against a well-formed batched `SpectralContext`
(`_StubBatchedSpectralSolver` exposing `solve_subset`) producing the per-member
exact-0-loss/NaN-record split `_apply_spectral_batched` documents
(`test_compute_terms_batched_accepts_a_batched_spectral_context`), and against a serial-shaped
solver (`.solve()`, no `.solve_subset()`) refused before any solve is attempted
(`test_compute_terms_batched_refuses_a_serial_solver`, `n_calls == 0`).

**What was rejected and why:** (a) restoring the deleted refusal verbatim — makes the
newly-legal, REDESIGN-critical-path combination unreachable through `compute_terms_batched`,
which is a functional regression relative to `2b11010`, not a repair; (b) leaving no
validation at all (the state as committed) — a `PatternSolver` (serial) handed to the batched
assembler by mistake surfaces as `AttributeError: 'PatternSolver' object has no attribute
'solve_subset'` from inside `_apply_spectral_batched`, **after** the batch has already paid
for a batched steady-state solve and three batched Jacobians, rather than being refused
before either runs.

**Not independently validated:** this task (R3 task 1) repairs only the entry-point
validation and the test gap it exposed (C1). `BatchedPatternSolve.backward`'s adjoint
gradient — the load-bearing untested item the review calls C2 — is out of scope here and
remains unverified; the accepts-test above exercises `compute_terms_batched`'s per-member
wiring with a hand-written stub solver, not the real `forward.BatchedPatternSolver`.

**RULED — controller ruling 2026-08-19, same task:** the legality shape found above is
CONFIRMED as the current contract: `batched=True` combined with a non-zero spectral weight is
**legal**, not refused.
`tests/test_ignition_gating.py::test_recover_raises_on_batched_with_a_spectral_weight` (`:194`)
is renamed to `test_recover_accepts_batched_with_a_spectral_weight` and rewritten to PIN that
contract non-vacuously: `R.recover(ri, strategy=_spectral_on_strategy(), batched=True,
lbfgs_steps=0, adam_steps=0, n_restarts=1)` is asserted to complete WITHOUT raising
(`lbfgs_steps=0` explicit so the unrelated LBFGS guard cannot be what fires or what doesn't;
`adam_steps=0` keeps it a fast validation-layer check, ~3.2 s measured). It asserts only that
the combination is accepted at the validation layer — nothing about the recovered result's
numerical correctness.

**Still assigned onward — end-to-end equivalence is Task 4's job, not this task's:** an
end-to-end `batched=True, lbfgs_steps=0` spectral `recover()` run checked NUMERICALLY against
its serial twin (the batched-vs-serial equivalence `docs/REVIEW_gpu_optim_delta.md` C1/C2
still want) remains unwritten. `:194`'s docstring names Task 4 explicitly as the completing
test. `BatchedPatternSolve.backward`'s untested adjoint gradient (C2) is still out of scope for
this task and remains unverified — see "Not independently validated" above.

**Where it lives:** `src/rngrn/losses/total.py::compute_terms_batched`;
`tests/test_ignition_gating.py::test_compute_terms_batched_accepts_a_batched_spectral_context`,
`::test_compute_terms_batched_refuses_a_serial_solver`,
`::test_compute_terms_batched_refuses_a_non_batched_model`,
`::test_recover_accepts_batched_with_a_spectral_weight`.

---

### D-OBS-1 — the relax saturation detector's k\* estimator is `observables.kstar_of_torch`, and that function now bins with `raps`'s own `np.digitize`, not `floor`

**Date:** 2026-08-19 (R3 task 3, branch `feature/gpu-optim-repair`, repairing
`docs/REVIEW_gpu_optim_delta.md` C3). **Status:** DECIDED
**Decided by:** the controller, on measurements taken by the implementing agent under
delegated authority.

One entry, two coupled facts: the estimator **swap** that `feature/gpu-optim` made silently
(semantic change #6), and the **binning fix** that makes the swap legitimate.

**The decision (part 1 — the swap, now recorded).** `forward.relax_to_pattern_torch`'s
saturation detector evaluates k\* with `observables.kstar_of_torch` instead of pulling the
channel-0 frame back to the host and calling `observables.kstar_of`; `forward._kstar_of_torch_batched`
is its batched twin and shares the same cached binning. The motive is transfer, not
arithmetic: the numpy detector costs up to 400 × 2 MB of device-to-host copy plus a host FFT
per solve at 512², where the torch one moves two scalars per chunk. This swap was made in
`feature/gpu-optim` with no register entry; that omission is what this decision closes. The
serial CPU path (`relax_to_pattern`, chosen by `PatternSolver._relax` when the device is CPU)
still calls `kstar_of` and is unchanged.

**The decision (part 2 — the binning).** `observables._raps_torch_bins` assigns bins with
`np.clip(np.digitize(|k|, kbins) - 1, 0, nbins - 1)` — literally `raps`'s expression. It was
`np.clip(np.floor(|k| / dk), 0, nbins - 1)`. `observables.raps` (the unwindowed RAPS,
register item 15's **primary** k\* estimator) is untouched, as is every numpy consumer of it
(`recover`, `eval/rollout`, `eval/lifted`, `relax_to_pattern`, `morphology`).

**Evidence.** All measured 2026-08-19 in this worktree's `.venv`, CPU float64, one thread,
sandbox disabled; the tracked checkpoint
`experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/checkpoints/model.pt` on a box
of 4 periods of its fastest-growing linear mode (L = 142.74286494132343). Reproduced by
`tests/test_raps_torch_parity.py`.

- **`floor` is not equal to `digitize` here, and not by a little.** Against `raps`, the
  `floor` binning gave, worst per configuration: saturated 64² pattern **11.9 % on a bin,
  0.36 % on k\***; the detector's own operating point (x\* + 1e-2 noise, n ∈ {16, 32, 64, 96}
  × seeds {0,1,2}) **22.5 % on a bin, 30.2 % on k\***.
- **Mechanism.** A lattice radius that is an exact integer multiple of `dk = 2π/L` sits
  exactly ON a bin edge, where `|k|/dk` can evaluate 1 ulp low and `floor` drops the point a
  bin. At this L those radii are the **on-axis** modes — m = 5 (n=16), 5,10 (n=32), 5,10,20
  (n=64), 5,10,20,40,43 (n=96), i.e. 4/8/12/20 of the n² points. The box is 4 periods, so
  m = 5 is adjacent to the pattern's dominant mode: the misbinned modes carry the most power,
  which is why a 1-ulp defect moves a bin by double-digit percent.
- **No bound on the divergence generalises.** Which radii collide turns on the last bits of
  the float L: truncating L to 142.7429 makes n = 16 and n = 32 collision-free (the two then
  agree to 5.08e-16) while giving n = 64 a *different* collision set (m = 29, 35).
- **The fix closes it to round-off.** With `digitize`: **1.59e-15** worst bin on the 64²
  pattern (k\* exactly 0.0) and **8.36e-16 / 3.72e-16** worst over the 12 operating-point
  configurations.
- **The round-off floor itself is measured, not assumed** (scratch-script measurement, not
  committed under experiments/). On 13 (n, L) geometries where the
  two binnings provably assign every lattice point identically, over 78 field/geometry pairs
  (noise and multi-mode cosine, n = 16…128): worst per-bin **4.80e-14**, worst k\*
  **4.50e-16**. That residue is FFT backend plus summation order (`np.bincount` vs
  `scatter_add_`). `tests/test_raps_torch_parity.py` sets its bars at 1e-12 / 1e-14, ~20×
  above the floor.
- **Detector exposure, measured before the fix** (scratch-script measurement, not committed
  under experiments/). One trajectory of the torch integrator,
  **both** estimators evaluated on every chunk, the flatness detector then replayed over each
  series (this isolates the estimator; a numpy-vs-torch relax would confound it with FFT
  backend). 16 trajectories = seeds 0–7 × {32², 64²}, `chunk=500`, production
  `flat_tol=1e-4`: the stop chunk was **identical in 16/16** and the returned u\* differed by
  **exactly 0.0**. The detector tests a *relative* spread, and the divergence was close to a
  constant offset per trajectory, which cancels out of that ratio.

**Consequence for existing numbers — stated plainly in both directions.**

- **No recorded result changes meaning.** The detector's decisions are demonstrably unchanged
  at the production tolerance (16/16 identical stop chunks, u\* difference exactly 0.0), the
  primary `raps` is untouched, and the GPU relax path exists only on this unmerged, unvalidated
  branch — nothing on `main` has ever run it.
- **ANNOUNCED LOUDLY (§10.4): `observables.raps_torch` / `kstar_of_torch` /
  `forward._kstar_of_torch_batched` return DIFFERENT VALUES from this commit forward** — by up
  to 30.2 % on k\* and 22.5 % on a bin at the relax detector's operating point. Any k\* read
  off the torch path before this commit is not comparable to one read after it. `raps` /
  `kstar_of` values are unchanged.
- **The case rests on the estimator contract and a 1.6× margin, not on a demonstrated
  failure.** No stop-chunk split was observed at `flat_tol=1e-4`. What was observed: the
  detector's flatness ratio differs between the two estimators by up to **45 %** at the firing
  window (n=64 seed 7: 7.83e-06 vs 1.42e-05), and sweeping `flat_tol` over 41 values in
  [1e-6, 1e-2] the two return **different stop chunks for 8/8 seeds at 64²** and 1/8 at 32² —
  at n=64 seed 1 they split at `flat_tol` = 6.3e-5 and 7.9e-5, within a factor **1.6** of the
  production value. The observed exposure is bounded; the possible exposure is not.

**What was rejected and why.** Keeping `floor` and recording the divergence as accepted
(option (b), explicitly considered and measured before being rejected). Register item 15 makes
the unwindowed `raps` the **primary** estimator and forbids silent swaps, precisely because the
sub-bin centroid k\* and the one-bin bar are calibrated on it. A detector that reads k\* from a
port diverging 30.2 % from the primary at its own operating point is a **second estimator**,
not a port, and formalising it as one would need owner sign-off it does not have. The
measurements say `floor` is not an alternative convention but a **defective implementation of
the documented half-open binning** `[m·dk, (m+1)·dk)`; the cost of the fix is zero (the binning
is built host-side in numpy and cached per (n, L, device, dtype), so `np.digitize` was already
available — the original "torch has no `digitize`" justification died when the binning moved to
the host); and under (b) the only honest test is one pinning a 30 % divergence, which is a
worse artefact than the defect it documents.

**Two prior claims corrected — neither was reproducible.** `raps_torch`'s docstring stated
"on a saturated 64² pattern `kstar_of_torch` and `kstar_of` agree to 3e-16" and "on a
pure-noise 16² field one bin differed by 14 % and k\* by 0.24 %", and
`docs/REVIEW_gpu_optim_delta.md` C3 quotes the latter as the measured divergence. Measured here
at the fixture geometry: **3.58e-03** (not 3e-16) and up to **22.5 % / 30.2 %** (not
14 % / 0.24 %). The 3e-16 figure is what a *collision-free* L produces, i.e. it was luck of the
float. Both figures are removed from the docstring; `tests/test_raps_torch_parity.py` is the
first measurement of this parity with provenance, and it pins the measured round-off floor
instead.

**Not independently validated:** every number above is CPU float64 at n ≤ 96 on one
checkpoint. No CUDA measurement was taken (the torch relax path's reason for existing is GPU
transfer, and no GPU run was made here), and the detector-exposure sweep covers 8 seeds × 2
grid sizes on a single model, not the config space.

**Where it lives:** `src/rngrn/observables.py::_raps_torch_bins` (the `digitize` line and its
docstring), `::raps_torch` (docstring);
`src/rngrn/forward.py::relax_to_pattern_torch` (docstring), `::_kstar_of_torch_batched`;
`tests/test_raps_torch_parity.py`.

---

### D-PERF-5 — `_lsmr_torch`'s three departures from scipy `lsmr` stay; this is the missing entry for semantic-change-table #4

**Date:** 2026-08-19 (R3 task 5, `feature/gpu-optim-repair`, repairing
`docs/REVIEW_gpu_optim_delta.md` I1 / table #4). **Status:** DECIDED
**Decided by:** the implementing agent under delegated authority (§10), writing the entry the
review found missing; the underlying code change itself was not authored by this task.

**The decision:** three departures from a scipy-`lsmr`-faithful port of the D-FFT-10 adjoint
solve stay as landed, undocumented until now:

1. `forward.py:186` (`_LSMR_STOP_CHECK_EVERY = 25`) — the stopping test runs every 25
   iterations instead of every iteration, so the solve can overshoot scipy's stop point by up
   to 24 iterations; the returned iterate is not the one at the crossing scipy would report.
2. `forward.py:153` (`_sym_ortho_t`) — `_sym_ortho` (Python floats, scipy-verbatim) is
   deleted and replaced by a branchless 0-d-tensor version with guarded denominators.
3. `forward.py:189` (`_lsmr_torch`) — the exact-Krylov-breakdown branch changes algorithm:
   scipy *skips* the `v` update entirely when `beta == 0`; this code always runs it with
   `u / where(beta==0, 1, beta)`, leaving an unnormalised vector rather than terminating.

**Why this is accepted rather than reverted:** the refinement loop that follows LSMR decides
convergence on the solve's *true residual*, not on LSMR's own internal stopping signal — so a
cadenced, up-to-24-iterations-late stop changes which iterate refinement starts FROM, not what
gets accepted as the final answer. All three departures are already documented at length,
correctly, in `_lsmr_torch`'s own docstring and `_minnorm_solve_t`'s docstring — what was
missing was this register entry, not the code-level explanation.

**Evidence:** `docs/REVIEW_gpu_optim_delta.md` I1 (§3, table #4) is the source of the above:
it independently re-derived all three departures from the diff, confirmed the docstrings
state them honestly, and confirmed the true-residual argument is sound. It also states the
gap this entry closes: `docs/DECISIONS.md` gained exactly one entry in the `feature/gpu-optim`
range (D-PERF-3, about the dispersion backend) — none for this change — and that
`tests/test_forward_solve.py`, named in the docstring as pinning agreement, "was written
against the *faithful* port and was not re-examined" against the cadenced/branchless version.
No fresh scipy-vs-torch LSMR comparison was run in this task — it is documentation-only, no
behaviour change (per this task's own scope).

**What was rejected and why:** (a) reverting to an iteration-for-iteration scipy-faithful stop
check and restoring `_sym_ortho`'s Python-float branching — this would reintroduce a host sync
on every LSMR iteration, which is exactly the cost this branch's whole thesis (§7a: batching
and desyncing the training-time hot path) exists to eliminate; (b) leaving the change
undocumented — the status quo, which is what produced I1 in the first place and which
`CLAUDE.md` §10 forbids for a science decision that changes the stopping semantics of a solve
D-FFT-10 verified.

**Not independently validated:** this task made no new measurement. `tests/test_forward_solve.py`
was not re-examined against the cadenced/branchless/guarded-breakdown version specifically —
the review's finding that it "was written against the faithful port" stands unresolved. A
future task re-deriving or re-verifying scipy-vs-torch LSMR agreement under the current code
should start there.

**Where it lives:** `src/rngrn/forward.py:186` (`_LSMR_STOP_CHECK_EVERY`), `:153`
(`_sym_ortho_t`), `:189` (`_lsmr_torch`); D-FFT-10 (the decision this modifies the stopping
semantics of); `tests/test_forward_solve.py` (unexamined against this version).

---

### D-PERF-6 — the ETDRK4 blow-up check moved from per-step to per-call; the numpy/torch return-array parity break on a blow-up is now recorded

**Date:** 2026-08-19 (R3 task 5, `feature/gpu-optim-repair`, repairing
`docs/REVIEW_gpu_optim_delta.md` I2 / table #5). **Status:** DECIDED
**Decided by:** the implementing agent under delegated authority (§10), writing the entry the
review found missing; the underlying code change itself was not authored by this task.

**The decision:** `etdrk4_torch.py:133` (`integrate_etdrk4_rfft_torch`) evaluates
`isfinite(v).all()` ONCE per call, after the full step loop, rather than once per step as
`eval/numerics.integrate_etdrk4_rfft` (the numpy original) does. This stays as landed.

**Why this is accepted:** the boolean-equivalence argument is sound. Every operation in an
ETDRK4 step is linear or an FFT over the whole field, so a non-finite value cannot be erased
once it appears — `isfinite(v).all()` after the loop yields the identical flag a per-step
check would have produced. The reaction closure (`clamp(X,0)**n`, `1 + einsum`, division) was
checked for a path back to finite and none exists: `inf/inf -> nan`, and `nan` is absorbing.
On CUDA a per-step `isfinite` check is a blocking device sync whose cost does not shrink as
the step's own FLOPs shrink, so it dominates at small training geometries — eliminating it is
squarely within this branch's thesis.

**What the prior docstring omitted, now recorded:** `eval/numerics.integrate_etdrk4_rfft`
(numpy) returns **the field at the first non-finite step**; the torch port now returns **the
field after all `nsteps`**. So on a blow-up the two backends return numerically DIFFERENT
arrays, not merely a different step index — this is a genuine deviation from
`relax_to_pattern_torch`'s docstring claim that the two backends "relax the same trajectory up
to FFT-backend round-off" (`forward.py:628`), not just lost diagnostic granularity about which
step blew up.

**Evidence:** `docs/REVIEW_gpu_optim_delta.md` I2 (§3, table #5). The boolean-equivalence check
and the reaction-closure absorbing-nan argument are the reviewer's, stated there; this entry
restates them as the record `CLAUDE.md` §10 requires and adds the parity-break framing
verbatim from the review. `tests/test_etdrk4_torch.py` pins numpy equivalence at `delta <=
1e-12` (CPU) / `1e-9` (CUDA) on non-blowing-up trajectories only — it neither catches nor
covers the blow-up parity break, and per I2 it was not updated when the change landed.

**What was rejected and why:** (a) reverting to a per-step `isfinite` check — reintroduces the
per-step blocking CUDA sync this change exists to remove; (b) leaving the parity break
unrecorded — the state as landed, which silently weakens a standing cross-backend claim
(`relax_to_pattern_torch`'s docstring) without that claim's own text reflecting the exception.

**Not independently validated:** no test currently exercises a blow-up trajectory against both
the numpy and torch backends to confirm the returned arrays actually diverge as described (the
argument is analytic — nan absorption plus differing return timing — not measured here). This
task added the parity note to `integrate_etdrk4_rfft_torch`'s docstring but made no new
measurement and changed no behaviour.

**Where it lives:** `src/rngrn/etdrk4_torch.py:133` (`blew = ...`), its updated docstring;
`src/rngrn/eval/numerics.py::integrate_etdrk4_rfft` (the differing numpy contract);
`src/rngrn/forward.py::relax_to_pattern_torch` (docstring, `:627-631` — this task appends
the blow-up exception to the "same trajectory up to FFT-backend round-off" claim, which
was previously unconditional); `tests/test_etdrk4_torch.py` (does not cover the blow-up
case).

---

### D-PERF-7 — the liveness-sync cadence (25 steps) stays; its "harmless" claim is corrected, not the cadence

**Date:** 2026-08-19 (R3 task 5, `feature/gpu-optim-repair`, repairing
`docs/REVIEW_gpu_optim_delta.md` M3 / table #8). **Status:** DECIDED
**Decided by:** the implementing agent under delegated authority (§10), writing the entry the
review found missing; the underlying code change itself was not authored by this task.

**The decision:** `recover.py:236` (`LIVENESS_SYNC_EVERY = 25`) inside `_batched_restarts`
stays: the host-side "is everyone dead" bookkeeping and early break are checked every 25 Adam
steps (and on the final step), not every step. This can let the early break fire up to 24
steps late once every member of a batch has died.

**What the prior inline comment overclaimed, now corrected:** the comment at the sync point
called the delayed break "harmless -- those extra steps optimise an all-dead, all-masked batch
and produce nothing." That is stronger than the truth. During those extra steps `total`
(`torch.where(alive, loss_vec, zeros).sum()`) is an exact-zero tensor **with a graph** once
every member is dead, so the fresh gradient contribution from `total.backward()` is zero -- but
Adam's momentum and second-moment state, accumulated from steps BEFORE the batch died, keep
decaying and being applied to the parameters regardless of the current gradient being zero.
The parameters DO move during the lag. What is true, and is what the comment should have said,
is narrower: no REPORTED number depends on it, because `final_alive` is all-False for such a
batch and every member logs `steady_state_failed` downstream, whatever its parameters drifted
to during the lag. Separately, `verbose` printing at such a step computes
`float(loss_vec[alive].mean())` over an empty boolean selection, which silently evaluates to
NaN rather than raising.

**Evidence:** `docs/REVIEW_gpu_optim_delta.md` M3 (§4, table #8) is the source of both the
mechanism (Adam momentum vs. zero gradient) and the verbose-NaN observation; this entry
restates them as the record `CLAUDE.md` §10 requires. No new run was made to measure how
often, or by how much, parameters actually drift during the lag -- the review's own claim is
that no reported number depends on the answer, and this task's scope is documentation only.

**What was rejected and why:** (a) shortening `LIVENESS_SYNC_EVERY` to catch a fully-dead
batch sooner -- would reintroduce the per-step host sync this cadence was introduced to
eliminate (the same D2H sync it replaced two per-step syncs with), for a case the review
confirms affects no reported number; (b) silencing or fixing the verbose NaN print -- a
behaviour change, out of scope for a documentation-only task; recorded here instead as a known,
harmless (cosmetic-only) side effect for a future task to pick up if it matters.

**Not independently validated:** the magnitude of parameter drift during a lag, and how often
a lag actually occurs in practice, are both unmeasured. The claim that "no reported number
depends on it" rests on `final_alive`/`steady_state_failed` excluding such members from every
downstream consumer, which was verified by reading the code, not by a run.

**Where it lives:** `src/rngrn/recover.py:236` (`LIVENESS_SYNC_EVERY`), `:258-274` (the
`is_sync_step` block and its corrected inline comment), `:298-300` (the verbose NaN-mean
print, unchanged).

> **R3 INTEGRATION AMENDMENT — 2026-08-19, `feature/r3-integration` Task 10, integration
> review I3.** This entry's cadence rationale implies a benefit the code does not deliver:
> "eliminate... the same D2H sync it replaced two per-step syncs with" reads as if the
> batched step becomes async-pipelineable once those two syncs are gone. It does not --
> the batched step is unconditionally sync-bound regardless of this cadence, through six
> OTHER, pre-existing D2H syncs on the same step: `losses/terms.py:817-818`
> (`_np(sig0)`/`_np(sig_max)`/`_np(sig_max_pos)` inside `turing_hinges_split_batched`),
> `losses/total.py:358` (`ss_converged=conv.detach().cpu().numpy()`), and
> `losses/total.py:396,400` (`parts["total"]` and each `term_vals` entry's
> `.detach().cpu().numpy()` in `total_loss_batched`). Removing the two `.any()` syncs this
> cadence targeted is therefore a minor tidy, not an unlock -- and even that tidy's own
> benefit (fewer syncs at the SAME cadence point) is UNMEASURED; no run was made to time it
> before or after. The behavioural cost this entry documents -- up to
> `LIVENESS_SYNC_EVERY - 1` steps of decaying Adam momentum applied after a batch is fully
> dead, with no reported number depending on it -- stands unchanged; this amendment
> corrects only the implied performance framing, not the decision to keep the cadence.

---

### D-PERF-8 — `lbfgs_error` recording replaces a bare `except: pass`; the serial-only asymmetry against `_batched_restarts` is recorded, not closed

**Date:** 2026-08-19 (R3 task 5, `feature/gpu-optim-repair`, repairing
`docs/REVIEW_gpu_optim_delta.md` M4 / table #10). **Status:** DECIDED
**Decided by:** the implementing agent under delegated authority (§10), writing the entry the
review found missing; the underlying code change itself was not authored by this task.

**The decision:** in the serial restart loop (`recover.py:679-699`), an LBFGS-polish failure
that used to be swallowed by a bare `except: pass` is now caught, formatted as
`lbfgs_error = f"{type(e).__name__}: {e}"`, and appended as an `lbfgs_error` key on that
restart's `restart_log` row (`recover.py:728-731`). This stays as landed.

**Why this is accepted:** CLAUDE.md §4 requires failing loud; the deleted `except: pass` was
exactly the failure mode the house style forbids -- the code comment at the site already notes
this is "where the FIRST async CUDA error of a GPU run would previously have been swallowed."
The LBFGS polish stays optional (a failed polish does not abort the restart; Adam's parameters
are kept), so recording rather than raising is the correct severity: informational, not
fatal.

**The asymmetry, now recorded:** `lbfgs_error` is added to SERIAL `restart_log` rows only. The
batched path's row construction (`_batched_restarts`, `recover.py:322-325`) never runs an
LBFGS polish per member and never emits the key. `recover(batched=False)` and
`recover(batched=True)` therefore write different column sets into the run index for what is
nominally the same log structure. `CLAUDE.md` §4's flat-scalar rule is satisfied either way (a
`str | None` is a flat scalar) -- the asymmetry is the missing column, not the type.

**Evidence:** `docs/REVIEW_gpu_optim_delta.md` M4 (§4, table #10) is the source of the
diagnosis, including the exact line references and the flat-scalar clarification. A short
inline comment was added at `recover.py:322` (this task) noting the missing key at the site
where a reader would otherwise expect it.

**What was rejected and why:** (a) reverting to `except: pass` -- rejected outright, it is the
defect being fixed and doing so would violate CLAUDE.md §4 again; (b) adding a matching
`lbfgs_error=None` key to every `_batched_restarts` row in this task -- rejected as a
behaviour change: `_batched_restarts` has no per-member LBFGS polish step to report on, so
adding the key would either be a dead constant or would require adding the polish itself,
neither of which is a documentation fix and both are out of this task's no-behaviour-change
scope.

**Not independently validated:** no test asserts on the presence or absence of `lbfgs_error`
across the two `recover()` paths; nothing currently pins the asymmetry this entry describes,
so a future change could close or widen it silently again.

**Where it lives:** `src/rngrn/recover.py:699` (`lbfgs_error` assignment), `:728-731`
(`restart_log.append(..., lbfgs_error=lbfgs_error)`); `src/rngrn/recover.py::_batched_restarts`
`:320-325` (the row missing the field, with the new inline comment).

---

### D-PERF-9 — `bdf1_newton_krylov` stub now raises instead of silently falling back to ETDRK4; independently duplicated on two branches (integration collision point 27)

**Date:** 2026-08-19 (R3 task 5, `feature/gpu-optim-repair`, repairing
`docs/REVIEW_gpu_optim_delta.md` I3 / table #9). **Status:** DECIDED
**Decided by:** the controller, on evidence gathered by the implementing agent under
delegated authority (§10).

**The decision:** `src/rngrn/eval/numerics.py:194-209`
(`integrate_bdf1_newton_krylov`) raises `NotImplementedError` naming the reason, rather
than silently running ETDRK4 under the `bdf1_newton_krylov` label as it previously did.
This stays as landed.

**Integration collision point 27** (absent from `docs/REVIEW_gpu_optim_delta.md` §10's
list of 26): the SAME change was made independently, on two branches neither aware of the
other:

- `feature/gpu-optim`'s `numerics.py:194-209` (this branch, inherited unmodified via
  `feature/gpu-optim-repair`);
- R1's Task 1 stub on `feature/lift-ladder`'s own `src/rngrn/eval/numerics.py::
  integrate_bdf1_newton_krylov`, whose docstring instead cites
  `docs/REDESIGN_rngrn.md` §5.2 ("removed the silent ETDRK4 fallback... implemented at the
  gate milestone (R4 plan) as the 128^2 cross-check integrator") and points forward to R4,
  not to any DECISIONS entry. Checked directly for this entry:
  `git -C worktrees/lift-ladder show feature/lift-ladder:docs/DECISIONS.md` (3528 lines at
  `2f50fff`) has no `bdf1` entry — grepped case-insensitively, zero matches.

Both raises share the identical reasoning, independently arrived at: "a run that actually
executed ETDRK4 under that label would be provenance that claims an integrator which never
ran" (`numerics.py:200-203`, and R1's docstring makes the same point in its own words) — but
the two stubs are semantically identical, not textually identical: different docstrings,
different `NotImplementedError` messages. Whichever branch merges first, the other's hunk at
the same lines is this 27th collision point for Phase B's merge ledger to watch, alongside
the 26 in `docs/REVIEW_gpu_optim_delta.md` §10 — it will merge as a CONFLICT, not a silent
duplicate-keeps-both, and Phase B's ledger should expect that. Cite this entry there.

**No relabelling occurred — stated explicitly, this is the finding.**
`docs/REVIEW_gpu_optim_delta.md` I3 raised the possibility that the raise "silently
re-labels prior runs: every existing run recorded with `integrator='bdf1_newton_krylov'`
used ETDRK4," and flagged under §10.4 that a changed number's meaning must be announced.
This task checked rather than repeated that claim: `grep -rln "bdf1_newton_krylov"
configs/ experiments/ notebooks/ scripts/` in this worktree returns exactly one file,
`configs/base.yaml:52`, and that line is the enum-documenting comment
(`# 'etdrk4' | 'imex_split' | 'bdf1_newton_krylov'`), not a selection — no tracked config,
no `frozen_config.yaml`, no notebook, and no experiment anywhere in this repository ever
set `integrator: bdf1_newton_krylov`. **Therefore no run was ever recorded under that
label, no number's meaning has changed, and no loud announcement is owed** — the absence
itself, not a relabelling, is what this entry records.

**What was rejected and why:** leaving the silent-ETDRK4-fallback stub in place — a run
could otherwise complete, log `integrator='bdf1_newton_krylov'`, and report results that
were never produced by that integrator, which is exactly the false-provenance failure mode
CLAUDE.md §8 exists to prevent. Both branches independently rejected the silent fallback
for this same reason, without coordinating.

**Not independently validated:** no BDF1 Newton-Krylov integrator has been implemented on
either branch; the stub raises in both and is not scheduled before R4 per R1's own
docstring. This entry records only that the raise is correct and currently unreachable
from any tracked config, not that a working BDF1 path exists anywhere.

**Where it lives:** `src/rngrn/eval/numerics.py:194-209`

**R3 INTEGRATION NOTE — 2026-08-19, `feature/r3-integration` Task 8.** D-PERF-7, D-PERF-8 and
D-PERF-9 above are transcribed verbatim from `feature/gpu-optim-repair`. Two corrections apply
to them on this line:

1. **Line citations.** The `Where it lives` line numbers were written against
   `feature/gpu-optim-repair`'s `recover.py`, which carries the batched-spectral hunks this
   task did **not** transplant (they are Task 9's). On this line the same code sits at:
   `recover.py:227` (`LIVENESS_SYNC_EVERY`), `:238-262` (the `is_sync_step` block),
   `:279-280` (the verbose NaN-mean print), `:651` (`lbfgs_error = None`), `:671-674` (the
   `except Exception as e` recorder), `:703` (`lbfgs_error=lbfgs_error` on the row),
   `:302-305` (the batched row's inline note that it has no such key),
   `eval/numerics.py:194-209` (the `bdf1_newton_krylov` raise, unchanged).
2. **D-PERF-7's cadence landed WITHOUT the `active=` argument, deliberately.** On
   `feature/gpu-optim-repair` the same rewrite also passes
   `active=alive if spec_cfg is not None else None` into both `total_loss_batched` calls.
   `spec_cfg` reaches `_batched_restarts` only through collision-ledger row 23, and
   `total_loss_batched` gains its `active=` parameter only through row 4 — **both owned by
   Task 9**. Neither exists here yet, so passing the mask was not possible and was not faked.
   The liveness cadence itself is complete and behaves exactly as D-PERF-7 describes. Review
   finding **M9** (whether the `spec_cfg is not None` conditional should be a bare
   `active=alive`) is therefore untouched by this task and stays open for Task 9, which owns
   both halves of it.

---

### D-PERF-10 — at integration the `dispersion_backend` DEFAULT stays `'eig'`; D-PERF-3's default flip is REJECTED, its mechanics are kept

**Date:** 2026-08-19 (R3 Phase B Task 8, `feature/r3-integration` @ `087de3d`; collision ledger
row 26 / §2.8, plan Task 8 Step 3). **Status:** DECIDED
**Decided by:** the controller, on the four facts returned by the implementing agent under
delegated authority (§10). The agent implemented the mechanics and returned the flip rather
than taking it; this entry records the ruling that closed that decision point.

*(Id note: `D-PERF-4`, `-5` and `-6` are not free — they exist on `feature/gpu-optim-repair`
and describe code that Task 9 transplants, so they are reserved rather than skipped. `-10` is
the next free key in this series. This entry rules on `D-PERF-3`, which it should be read with.)*

**The decision:** on the integrated line, `dispersion_backend` defaults to `'eig'` in all four
places — `RNGRN.__init__`, `BatchedRNGRN.from_seeds`, `config.ModelConfig` and `recover()`.
**D-PERF-3's flip of those four defaults to `'auto'` is REJECTED at integration.** What IS
adopted is D-PERF-3's *mechanics*, which are separable from its default: `'auto'` is a legal,
opt-in value, accepted by `RNGRN.__init__`'s three-way `assert` and resolved at construction by
`model.resolve_dispersion_backend` to `'cubic'` at N == 3 and `'eig'` otherwise. The string
`'auto'` never survives construction, and `.dispersion_backend` always reads a concrete backend.
Explicit `'cubic'` at N ≠ 3 still raises. **Nothing about which backend an existing N = 3 run
uses changes**, so the BACKEND-CHOICE half of A0 bit-identity holds by construction rather
than by argument.

**Scope of that claim — read D-PERF-11 before relying on it.** It is about the backend and
nothing else. The same integration's one-Jacobian-per-step hoist leaves the OBJECTIVE
bit-identical but perturbs the GRADIENT at the last bit, so an A0 *trajectory* re-run after
this task is not guaranteed to reproduce a pre-integration A0 run bit-for-bit. D-PERF-11
announces that separately; it is not covered by the sentence above.

**Evidence — the four facts the decision point returned.**

1. **The binding constraint.** `docs/PLAN_redesign.md` Global Constraints: *"A0 is untouchable:
   the baseline objective … must keep bit-identical behaviour after every task."* A flipped
   default changes the backend of every N = 3 caller that omits the argument, which includes the
   A0 baseline path.
2. **D-PERF-3's own concession.** That entry's "Announced loudly" paragraph states runs recorded
   after the flip are **not bit-comparable** to runs before it for any N = 3 config that did not
   pin the backend. The cubic form is exact for N = 3 and agrees with `eigvals` to
   σ_max MAE 9.2e-13 / k\* MAE 0 / 0 of 127 Turing-verdict flips on real answer-key Jacobians,
   and ~1e-12 per step — but D-PERF-3 says plainly that this diverges over a full optimisation.
   The two documents therefore point opposite ways, and one had to yield; the constraint won.
3. **T16's a0 arm already ran `cubic`, by explicit driver choice — and this is an OPEN FACT, not
   a resolved one.** Every `experiments/redesign_r2/*/*/config/frozen_config.yaml` records
   `dispersion_backend: cubic`, passed explicitly, while `feature/redesign-model`'s `model.py`
   still defaults to `'eig'`. **So whether T16's a0 arm is comparable to the D5 row is a live
   question on the redesign line independently of gpu-optim, and this ruling does not answer it.**
   It only ensures that the integration does not make it worse by changing a default underneath
   it. **This question is carried to the owner roll-up (Task 22); it is not settled here.**
4. **A provenance regression D-PERF-3 does not mention, now closed.** `train.fit` wrote
   `config/frozen_config.yaml` from the *requested* config, so a run configured `'auto'` froze
   the literal string `'auto'` — a file that cannot say which backend produced the number, against
   `.claude/rules/reporting-numbers.md` step 4 ("read the frozen config, do not re-derive from
   it"). `fit` now resolves through `model.resolve_dispersion_backend` **before** the write, so
   the frozen config records the backend that ran. This was authorised whichever way the flip
   ruling went, and landed independently of it.

**What was rejected and why.**

- **The default flip itself** (D-PERF-3 as written). Rejected because it trades a measured A0
  bit-identity constraint for no measurable gain *on the runs that exist*: fact 3 shows every
  run that matters already pins the backend explicitly, so the flip changes behaviour only for
  callers that omit the argument — of which the A0 baseline is the one that must not change.
- **Silently keeping `'auto'` in frozen configs** — i.e. adopting the mechanics and leaving the
  provenance regression in place. Rejected outright: it would make a run directory unable to
  answer which algorithm produced its numbers, which is the false-provenance failure mode
  CLAUDE.md §8 exists to prevent. This is why the fix landed regardless of the flip ruling.
- **Deleting the `'auto'` mechanics along with the default** (integrate neither half). Rejected:
  the resolution block and the three-way assert are inert until a caller opts in, cost nothing,
  and are the vehicle by which a future GPU sweep gets the measured 162× cubic win with one
  string — the separability finding is precisely what made a partial adoption safe.
- **Adding `lbfgs_error=None` to `_batched_restarts`'s rows** to square up review finding M4.
  Rejected **by ruling, on the same grounds D-PERF-8 already gave**: the batched path runs no
  per-member LBFGS polish, so an always-`None` column would conflate "no polish exists" with
  "the polish succeeded", and it would change the recorded column set of every batched run for
  a field that can never carry information. **The serial-only asymmetry stays, recorded at both
  sites, not closed.**

**Not independently validated:** no cubic-vs-eig full-run comparison was made on this line —
the 162× and ~1e-12 figures in D-PERF-3 are inherited, not re-measured here. The
frozen-config test reproduces `train.fit`'s resolve-then-write sequence rather than running
`fit` (which needs a dataset); the *ordering* inside `fit` is pinned by a source-order
assertion, which is weaker than a behavioural test and is labelled as such at the site. And,
restating fact 3 because it is the caveat most likely to be forgotten: **this entry does not
establish that T16's a0 arm is comparable to the D5 row.** It establishes only that the
integration did not change any backend that was in use.

**Where it lives:** `src/rngrn/model.py:42` (`resolve_dispersion_backend`, the one definition of
the rule), `:185` (the three-way assert), `:251` (resolution in `RNGRN.__init__`);
`src/rngrn/config.py::ModelConfig.dispersion_backend`; `src/rngrn/recover.py::recover`;
`src/rngrn/train.py:205-212` (resolve before freeze). Pinned by
`tests/test_dispersion_cubic.py::test_the_default_backend_is_eig_everywhere` (the ruling — all
four defaults plus the resolved effect at N = 3, so a flip cannot land without editing the
test), `::test_auto_resolves_at_construction_and_never_survives_it` (the mechanics) and
`::test_frozen_config_records_the_resolved_backend_not_the_request` (the provenance fix).
Read with **D-PERF-3** (amended in place) and **D-PERF-8** (the M4 asymmetry).

---

### D-PERF-11 — ANNOUNCED LOUDLY: the one-Jacobian hoist leaves the objective bit-identical but the gradient 1-ulp different; A0 TRAJECTORY reproduction against pre-integration runs is NOT guaranteed

**Date:** 2026-08-19 (R3 Phase B Task 8, `feature/r3-integration`; raised as Important finding
I2 by the Task 8 review, which independently reproduced the measurement). **Status:** DECIDED
**Decided by:** the controller, on the review's finding; recorded by the implementing agent.

**Why this entry is separate and loudly titled:** CLAUDE.md §8 and §10.4 require a change in
what an existing number *means* to be announced, not buried. This one is easy to bury, because
every test is green and the objective is exactly unchanged — the affected quantity is
reproducibility of a *trajectory*, which nothing in the suite pins.

**The decision:** the hoist stays. `losses/total.py::compute_terms` and
`::compute_terms_batched` evaluate **one** autograd Jacobian per step at `x_disp` with
`create_graph=True` and share it across `kstar_anchor`, the Turing hinges and `anticollapse`,
replacing three identical Jacobian builds of which two were pure waste. The consequence below
is accepted and announced rather than avoided.

**Evidence — measured on the integrated tree, twice, independently.**

- **The objective is bit-identical.** Hoisted vs. per-term Jacobians, N = 3, full forward:
  **|ΔLoss| = 0.0 exactly** on every seed tried (6 seeds, implementer; 8 seeds, reviewer).
- **The gradient is identical only to ~1 ulp.** **max |Δgrad| ≤ 2.2e-16** over the same
  models, **non-zero on 8 of 8 seeds** in the reviewer's independent reproduction. The cause
  is arithmetic-order, not semantics: one Jacobian feeding three consumers accumulates its
  backward contributions in a different order than three separate Jacobians do. Float
  addition is not associative, so the sum differs in the last bit.
- **Therefore:** `docs/REVIEW_gpu_optim_delta.md` semantic-table #1's wording, *"Neutral —
  same tensor, three consumers; autograd accumulates the identical sum"*, **overstates it.**
  The sum is identical to float64 round-off, not exactly identical. Recorded here because a
  reader diffing gradients across this merge needs the bar to be 1e-16, not exact equality.

**What this means for A0 — the announcement.** A 1-ulp gradient difference at step 0 is
amplified by the optimiser over thousands of steps. So:

> **An A0 re-run after this task is NOT guaranteed to reproduce a pre-integration A0 run
> bit-for-bit. A future A0 rerun that fails bit-reproduction against an old run is EXPECTED,
> not a bug, and must not be diagnosed as merge damage.**

What *is* preserved, and what a comparison should be read against: the objective function
itself is unchanged to the last bit, so the two runs optimise **the same function** and their
converged results are comparable as science. It is the step-by-step path, and therefore the
exact final parameters of a fixed-seed run, that may differ. This is the **same argument class
D-PERF-10 used to reject the backend flip** — with the crucial difference that the backend
flip changed *which function is optimised* (cubic vs eig dispersion) for no measurable gain,
whereas this changes only summation order and buys the removal of two redundant autodiff
Jacobians per step on the hot path. That asymmetry is the whole reason one was rejected and
this one accepted.

**What was rejected and why.** (a) **Reverting the hoist to protect bit-reproduction of old
A0 trajectories** — rejected: it would keep two provably redundant Jacobian builds per step on
the hot path forever, to protect a property (identical float summation order) that no
pre-registered condition asks for and that any future optimisation would break again. (b)
**Leaving it unannounced because the suite is green** — rejected outright: the suite cannot
see this, which is exactly why §10.4 exists; an unannounced non-comparability is worse than a
missing number. (c) **Claiming, per semantic-table #1, that the gradient is "the identical
sum"** — rejected as false against measurement; corrected above.

**Not independently validated:** no A0 trajectory was actually re-run and diffed against a
pre-integration A0 run — the 1-ulp figure is a single-step forward/backward measurement, and
the amplification-over-training claim is the standard chaotic-optimiser argument, **not
measured here**. Nobody has quantified how far a fixed-seed A0 run drifts by step 1500. If
that number matters to a paper claim, it needs a run, and this entry does not supply one.

**Where it lives:** `src/rngrn/losses/total.py:126-140` (serial hoist), `:239-253` (batched
hoist); `src/rngrn/losses/terms.py` (`J=` on every dispersion-side term, pinned by
`tests/test_term_registry.py::test_dispersion_side_terms_all_accept_the_hoisted_jacobian`).
Read with **D-PERF-10**, whose backend-choice bit-identity claim this entry scopes.

---

### D-PERF-12 — the five spectral terms are registered as BATCHABLE, and the registry's refusal claim is now guarded by a test rather than by a grep

**Date:** 2026-08-19 (R3 Phase B task 9, `feature/r3-integration`, integrating
`feature/gpu-optim-repair`@`862e6f2` onto `feature/redesign-model`). **Status:** DECIDED

**Context.** R2's Task 8 built `losses/term_registry.py`, which pairs every loss term with
its batched twin **or** the reason batching is refused, and asserts exactly one of the two is
set. At the time it was written, batching the five spectral terms genuinely was refused —
`forward.PatternSolver` owned per-restart warm state with no member axis — so all five were
registered `batched_fn=None, refusal_reason=_SPECTRAL_REFUSAL`. gpu-optim then implemented
`spec_shape_batched` … `real_moments_batched` and `forward.BatchedPatternSolver`. D-PERF-4
records the deletion of the matching refusal in `losses/total.py`.

**The decision:** the five spectral keys are re-registered with their real
`getattr(S, f"{key}_batched")` callables and `refusal_reason=None`; `_SPECTRAL_REFUSAL` is
deleted; `term_registry.py`'s module docstring, which said the spectral five "also refuse", is
rewritten. Exactly two terms refuse a batched form now: `resid` and `morphology`.

**Why this needed a decision at all, and it is the important part.** *Nothing in the suite
forced it.* The XOR test passed either way, because the five registrations hardcoded
`None, _SPECTRAL_REFUSAL` in a loop that transplanting `spectral.py` does not touch;
`batched_fn` had **zero runtime readers**; and `refusal_reason`'s only spectral reader was the
`total.py` raise that D-PERF-4 deleted. So the alternative outcome was a **false declaration
under a fully green suite** — the registry asserting that five terms refuse a batched form
sitting in the same package. That failure class is invisible to git and to a diff review, which
is why `docs/INTEGRATION_r3_collisions.md` row 28 is tagged CONTRACT and why the verification
below is part of the decision rather than incidental to it.

**Evidence.**

- The flip's forcing mechanism was **built**, not assumed:
  `tests/test_term_registry.py::test_a_refusing_term_has_no_batched_implementation_sitting_next_to_it`
  asserts that a term declaring a `refusal_reason` has no `<fn>_batched` in its serial
  callable's own module. Written and run **before** the flip, with `spectral.py` already
  transplanted, it failed exactly as the ledger predicted (`real_moments declares
  refusal_reason, but rngrn.losses.spectral.real_moments_batched exists`). After the flip:
  7 passed. It is a standing guard on every future entry, not a one-off.
- Runtime enumeration after the flip: the terms declaring a `refusal_reason` are exactly
  `morphology` and `resid`; `grep -n "_SPECTRAL_REFUSAL" src/` returns zero;
  `grep -rn "cannot compute the spectral terms" src/` returns zero (the retired refusal text
  did not resurrect anywhere).
- Equivalence of the batched twins to their serial forms is **not** established by this entry.
  It is `tests/test_batched_spectral_terms.py` and `tests/test_batched_forward_solve.py`
  (Phase A, transplanted with the code) that measure it; this entry only stops the registry
  lying about whether they exist.

**Calibration — stated plainly.** All five stay `default_weight=0.0` and
`calibration="UNCALIBRATED"`. Nothing has measured a spectral weight on this data, and having
a batched twin is not evidence about a weight. The flip changes **availability**, not any
number: with every spectral weight at 0 the terms are not consulted at all, so no existing
result becomes non-comparable through this change.

**What was rejected.** (a) **Leaving the five registered as refusing** — rejected: it is a
false statement about the code, and the ledger's own analysis shows nothing would ever have
caught it. (b) **Relying on `grep -n "batched_fn=None"` as the verification**, as row 28
originally specified — rejected as **vacuous** and amended in the ledger: `_register` is called
positionally, so that string never appears in the file and the grep returned zero before the
flip too. (c) **Marking the five `CALIBRATED` because the batched path was measured** —
rejected: what was measured is agreement between two implementations, not a weight.

**Also landed here, and separable from the above — the `active=` guard is removed (review M9).**
`recover._batched_restarts` passed `active=alive if spec_cfg is not None else None` to
`total_loss_batched` at both call sites. Both now pass `active=alive` unconditionally. The
guard was a no-op: `active` is read only inside `losses.total._apply_spectral_batched`, which
runs only under `if spectral is not None`, so on the non-spectral path the mask is never
looked at. Phase A's comment at the site ("the conditional cannot change the value passed") was
internally contradictory — the conditional plainly changes what is passed; what is true is that
the outcome is unaffected. **Measured, not argued:**
`tests/test_batched.py::test_active_mask_is_inert_without_a_spectral_context` asserts the
`parts` column set, the loss vector and the convergence mask are identical with and without a
mask, and a mutation probe (injecting one extra `parts` key under `if active is not None`)
turns it red. The property that guard existed to protect — a stable frozen column set for
`history.TrainingHistory` on the non-spectral batched path — is therefore held up by a test
instead of by a dead branch.

**Not independently validated:** no batched spectral *recovery* has been run to convergence
and compared against a serial one. `recover(batched=True)` with a spectral weight is pinned at
the validation layer only (`tests/test_ignition_gating.py`), and the equivalence work is at the
assembler level from identical preset warm starts, for the reason
`tests/test_batched_forward_solve.py`'s module docstring gives (a fresh relax differs by FFT
backend between the two paths and would confound the comparison). No science claim rests on the
batched spectral path yet.

**Where it lives:** `src/rngrn/losses/term_registry.py` (module docstring, the
`SPECTRAL_TERM_KEYS` loop, `_SPECTRAL_REFUSAL` deleted); `src/rngrn/recover.py`
(`_batched_restarts`, both `total_loss_batched` call sites);
`tests/test_term_registry.py::test_a_refusing_term_has_no_batched_implementation_sitting_next_to_it`;
`tests/test_batched.py::test_active_mask_is_inert_without_a_spectral_context`;
`docs/INTEGRATION_r3_collisions.md` rows 28 and 24 (both carry a T9 amendment). Read with
**D-PERF-4**, which records the deletion of the corresponding `total.py`/`recover.py` refusals.


---

### D-R3-1 — the unrolled path's finite-difference check holds dt, the solve box and the warm state FIXED

**Date:** 2026-08-19 (R3 Task 12, `feature/r3-integration`). **Status:** DECIDED

**Context.** `docs/REDESIGN_rngrn.md` §4.2 gives the redesign two gradient paths, and §8 item
14 leaves the truncated-unrolled one's segment length uncalibrated pending a measured
gradient-error-vs-length curve. Measuring that curve requires deciding *what function* the
finite differences are differencing. D1's own FD protocol (`scripts/diag_fft_d1.py` check 3)
re-solves the pattern at each perturbed theta, because there the differentiated object is the
steady state u*(theta) and the timestep is a solver detail with no effect on the answer. For
the unrolled path that is no longer true: the differentiated object is the finite-time relax
map Phi_S(X0; theta), and `dt`, the box `(n, L)` and `X0` are arguments to it.

**The decision:** the FD check for the unrolled path perturbs **theta only**, holding `dt`,
the solve box and the warm state `X0` fixed at their base-theta values.

**Why.** All three are DETACHED in the analytic path, and deliberately: `dt = 0.2/|eig(J)|_max`
and the box `L_solve = p*2*pi/k-hat(theta)` are geometry, and §4.3 requires solve-box geometry
not to be differentiated; `X0` is detached because §4.2's truncation *is* that detachment. A
detached quantity contributes exactly zero derivative, so an FD that let it move would
difference a *different* function and disagree for a reason that is not an error — it would
report the geometry's sensitivity as gradient bias and read as a defect in the path.

**Evidence.** Under this protocol the measured unrolled-vs-FD relative error on a saturated
warm state is **6.5e-11 to 6.9e-9** across segment lengths 1 to 2048 — five orders of magnitude
below D1's 1e-4 acceptance, at 6 random directions in the full 36-dimensional theta space with
the eps sweep {1e-3, 1e-4, 1e-5, 1e-6} and best-eps per direction. Run:
`experiments/redesign_r3/unrolled_segment/results/curve.json` (method block records
`dt_is_held_fixed_across_fd: true`).

**A DEVIATION FROM THE SPEC'S PROTOCOL, flagged rather than buried.** `REDESIGN_rngrn.md`
§4.2's A/B discipline specifies the FD check at **10 directions** x the active loss terms; this
curve ran at **6**. That is acceptable *for a calibration curve* — its job is the shape of
error-vs-segment-length across 12 lengths x 2 arms x 2 functionals, where direction count trades
against ladder coverage, and the measured spread is five orders of magnitude below tolerance so
four more directions would not move the recommendation. It is **not** acceptable as the gate:
`docs/PLAN_redesign_R3.md` Task 14's FD-faithfulness A/B against the adjoint runs at the
spec's full **10 directions**, at the operating point, on both paths at the same theta. Do not
cite this curve as having discharged that gate.

**Calibration.** Nothing is calibrated by this entry; it fixes a measurement protocol. The
1e-4 tolerance is D1's, inherited unchanged, not re-derived here.

**What was rejected.** (a) **Re-deriving dt per FD point** from the perturbed theta's
|eig(J)|_max — rejected: it differences a function the analytic path does not compute, and the
disagreement would be uninterpretable. (b) **Re-tiling the box per FD point** — rejected for
the same reason, and additionally because k-hat(theta) is detached by §4.3, so the analytic
gradient contains no box-sensitivity term to compare against. (c) **Re-relaxing X0 per FD
point** — rejected: that is the *adjoint* path's question (D1 already answers it), not this
one's; the truncated path is defined on a fixed warm state by construction.

**Where it lives:** `scripts/r3_unrolled_segment.py` (module docstring, `perturbed_value`);
`src/rngrn/unrolled.py` ("WHAT IS DIFFERENTIATED AND WHAT IS NOT");
`experiments/redesign_r3/unrolled_segment/results/curve.json`. Read with **D-FFT-10** (the
adjoint path's own FD verification) and **D-R3-2**.

---

### D-R3-2 — the truncated-unrolled segment length is 128 steps, SCOPED to a saturated warm state

**Date:** 2026-08-19 (R3 Task 12, `feature/r3-integration`; controller ruling on the returned
decision point). **Status:** DECIDED — CALIBRATED inside the stated regime, UNCALIBRATED
outside it

**Context.** `docs/REDESIGN_rngrn.md` §8 register item 14: the truncated-unrolled path's
segment length is UNCALIBRATED, to be set at R3 from the measured gradient-error-vs-length
curve. §4.2's design is "detach the warm-started state and differentiate only the final
**saturated** segment".

**The decision:** `unrolled.SEGMENT_STEPS_DEFAULT = 128`, and it is the default of
`unrolled_relax(..., segment_steps=...)`. The number is adopted **only** for invocation from a
saturated warm state; three conditions travel with it and are written at the constant itself:

* **(a) Calibrated on ONE fixture, ONE box, ONE seed.** The D1/D2 known-Turing checkpoint at
  k-hat = 0.17607 on the p=8, n=96 commensurate solve box, 6 FD directions, two placeholder
  loss functionals (`amp` = mean(u_0^2) and a log band-power proxy over [0.60, 1.55]*k-hat).
* **(b) UNCALIBRATED beyond that.** Transferability to another fixture, box, k-hat or
  objective is not established. `docs/PLAN_redesign_R3.md` Task 14's A/B against the adjoint —
  over BOTH converged and stalled members, and against the real objective once Task 13 wires
  it — is the next calibrator and may move the number.
* **(c) The path must not be invoked from a non-saturated state.** This is a caller contract,
  not a runtime check: saturation is a property of the trajectory that produced the warm
  state, which `unrolled_relax` cannot see, and inventing a single-field proxy for it would be
  an uncalibrated threshold. The caller that ran the relax knows whether its detector fired.

**Evidence** (`experiments/redesign_r3/unrolled_segment/results/curve.json`; CPU, 1 thread,
810 s of measurement; `cos`/`gap` are cosine and relative-norm distance to the S=2048 point of
the same arm, a measured reference and not a proven limit):

- **Saturated arm at S=128:** cosine **0.9999937** (`amp`) / **0.9999967** (band power),
  relative norm gap **0.394%** / **0.287%**; FD-faithful at **9.2e-10** / **3.9e-10** worst over
  6 directions, five orders below D1's 1e-4.
- **The curve is flat above ~128 and steep below ~32.** Norm gap: 21% / 31% at S=16, 4.97% /
  5.17% at 32, 1.15% / 1.97% at 64, 0.394% / 0.287% at 128, 0.067% / 0.175% at 256. Doubling
  128 -> 256 buys 0.33 / 0.11 percentage points for 1.8x the cost; 128 -> 1024 buys 0.39 /
  0.28 for 7.4x.
- **Growth arm, which is what condition (c) rests on.** From a warm state at 20.4% of the
  saturated amplitude, `amp` behaves (3.61% gap at S=128) but the band-power gradient does
  **not converge at all**: cosine wanders 0.936-0.978 and the norm gap sits at **20-41%** from
  S=32 out to S=1024, closing only once the segment is itself long enough to reach saturation.
  At S=128 specifically: **35.6%** norm gap, cosine 0.936. The spectral-flavoured gradient —
  precisely the one Tasks 13/14 will use — is the one that cares.

**Calibration — stated plainly.** 128 is CALIBRATED against the curve above *within* the
saturated regime and UNCALIBRATED outside it. The two loss functionals it was calibrated
against are **placeholders**, not `losses/spectral`'s RAPS shape term; the band-power proxy
exercises the FFT and the band and must not be reported as the trained objective.
`tests/test_unrolled_grad.py::test_the_adopted_segment_length_is_the_calibrated_default` pins
the constant and the signature default together, verified red by mutation.

**What was rejected.** (a) **Shipping no default at all** (the unit's own initial position) —
rejected by the ruling: it makes every future call site re-derive a number that has now been
measured, and an unstated default is not more honest than a stated-and-scoped one. (b) **A
silent global default of 128** — rejected: it would be a real number applied outside the regime
it was measured in, which is exactly the growth-arm failure. (c) **S=256 or larger for safety
margin** — rejected: cost is linear in S and the marginal accuracy is 0.1-0.3 percentage
points, while the FD check's own verifiability *degrades* at long segments (D-R3-3). (d) **A
runtime saturation guard inside `unrolled_relax`** — rejected: it would need a threshold on a
single field that nothing has calibrated, i.e. a guess dressed as a check.

**Where it lives:** `src/rngrn/unrolled.py` (`SEGMENT_STEPS_DEFAULT` and its condition block,
module docstring, `unrolled_relax` docstring);
`tests/test_unrolled_grad.py::test_the_adopted_segment_length_is_the_calibrated_default`;
`scripts/r3_unrolled_segment.py`;
`experiments/redesign_r3/unrolled_segment/results/curve.json`. Read with **D-R3-1** (the
protocol behind the numbers) and **D-R3-3** (why truncation is justified here at all).

---

### D-R3-3 — ANNOUNCED: §4.2's exponential-blow-up rationale for truncation did NOT reproduce; the operative rationale here is memory and FD-verifiability

**Date:** 2026-08-19 (R3 Task 12, `feature/r3-integration`). **Status:** DECIDED —
spec-expectation correction, announced loudly

**Context.** `docs/REDESIGN_rngrn.md` §4.2 states the cost of the unrolled path as: "gradients
through the exponential-growth phase of the instability can explode — mitigated by
**truncated** backprop". That sentence is the stated *reason* the path is truncated, and
`docs/PLAN_redesign_R3.md` Task 12 expected the measurement to exhibit blow-ups.

**The finding:** it did not. **No segment length in [1, 2048] blew up on either arm** — no
non-finite forward field and no non-finite gradient component, over 24 measured points x 2 loss
functionals, each backed by 48 finite-difference forwards. `forward_blew_up` is `false` in
every row of the committed curve. This holds from a growth-phase warm state as well as from a
saturated one.

**Why, mechanically.** The patterned attractor is **contracting**, so the amplification picked
up crossing the exponential-growth phase is bounded rather than compounding. The signature is
visible in the data: the growth arm's FD relative error rises to a **peak of 1.06e-5 at
S=1024** and then *falls back* to 1.9e-6 at S=2048, because by 2048 steps the trajectory has
saturated from either start and the memory of the growth phase has been contracted away.

**So what truncation actually buys at this operating point** — and this is the correction:

1. **Bounded activation memory.** MEASURED (n=32, N=3, float64, saved-tensor hooks): 24,576
   B/step retained at `checkpoint_every=1` — exactly one field — against 753 kB/step
   un-checkpointed (**30x**), and **0 B/step** for the un-differentiated warm-up at any length.
2. **A gradient that is still finite-difference-VERIFIABLE.** What degrades at long segments is
   the check, by five orders of magnitude on the growth arm (6e-11 -> 1.06e-5), while staying
   under D1's 1e-4 throughout.
3. It does **not** buy protection from an observed divergence, because there was none.

**Honest caveat, and it is load-bearing.** As S grows, the loss's higher derivatives grow with
the same transient amplification, so the central-difference *truncation* error grows even for
an exactly correct analytic gradient. The rising FD column is therefore **not** proof that the
analytic gradient is degrading — it is proof that this check can no longer certify it. Both
readings fit the data and this measurement cannot separate them; a Taylor-remainder or
complex-step check could, and neither was in scope.

**Scope of the finding.** One fixture, one commensurate box (p=8, n=96), one k-hat, one seed,
two placeholder losses, CPU float64. It is **not** a claim that the unrolled path cannot
explode in general — a stiffer model, a larger box, or a genuinely unstable dt would be
different questions. It **is** a claim that on the measured operating point the spec's stated
mechanism was absent, so a reader must not cite §4.2's explosion as evidence *from this
repository*.

**Calibration.** Nothing threshold-like is set here. The blow-up detector is
`unrolled_relax`'s own one-`isfinite`-per-call check, which raises rather than returning a NaN
field, and which was verified to fire on a real ETDRK4 instability (dt 1000x the stability
limit: finite at 8 steps, non-finite by 32) rather than only on an injected NaN.

**What was rejected.** (a) **Reporting the curve as confirming §4.2** — rejected: it does not,
and CLAUDE.md §8 makes a silently non-comparable rationale worse than a missing one.
(b) **Extending the ladder until something exploded** — rejected as a fishing expedition: the
S=2048 recovery shows the mechanism is contraction, not a not-yet-reached threshold, and a
blow-up manufactured at an unusable dt would say nothing about the operating point.
(c) **Editing §4.2 unilaterally** — not taken: `docs/REDESIGN_rngrn.md` is the binding spec and
the amendment is the controller's to make; this entry is the announcement it would rest on.

**Where it lives:** `experiments/redesign_r3/unrolled_segment/results/curve.json`;
`scripts/r3_unrolled_segment.py`; `src/rngrn/unrolled.py` (the "TRUNCATED" bullet, rewritten to
state this rather than repeat §4.2's expectation);
`tests/test_unrolled_grad.py::test_a_blown_up_segment_raises_rather_than_returning_a_nan_field`.
Read with **D-R3-2**, whose segment length rests on the convergence and cost argument rather
than on a stability one, and **D-PERF-6** (the per-call blow-up check this path reuses).

### D-R3-4 — Task 15's grid-fidelity verdict comes from a PAIRED-IC design; the unpaired sweep is retained as pattern-selection-variance evidence, and its headline is SUPERSEDED as a fidelity claim

**Date:** 2026-08-19. **Task:** R3 Task 15, fix round 1 (`feature/r3-raps-fidelity` @ the
commit carrying this entry). **Status:** ANNOUNCED — a previously reported number changed
meaning (reporting-numbers rule: stated loudly, here and in the task report).

**The decision.** The 96²-vs-512² RAPS band-fidelity comparison is measured with both grids
integrated from the SAME physical initial condition: the 512² IC is reconstructed
bit-identically from its seed (`eval.rollout.simulate`'s rng path has a single consumer, so
the draw is reproducible), spectrally decimated to each coarser grid (`decimate_field` on the
IC, per species), and each coarse arm is a fresh ETDRK4 solve from that decimated IC under
identical settings. Distances use the byte-identical estimator and band as the unpaired
sweep, so the two designs differ ONLY in IC pairing.

**Why.** The original (unpaired) sweep drew fresh white noise per grid per seed —
`seed s @ 96²` and `seed s @ 512²` share nothing physical — so its 34–45 %/bin distances
measure nonlinear pattern selection from unrelated ICs, not the coarse grid's estimator
fidelity. Task review I1 identified the confound; the re-review verified the paired design
on five checks, including 24/24 IC-reconstruction matches at |Δk*| = 0.0 exactly against
seed-distinct references.

**What changed meaning.** Unpaired: "no sub-512 grid clears D3's ~31 %/bin floor on
worst-seed." Paired: **96² clears the floor on every seed of every fixture** (per-fixture
mean/worst %/bin: 13.37/16.78, 4.04/9.38, 0.05/0.08), monotone in n. The unpaired number is
NOT a fidelity result and must not be cited as one; it is retained
(`results/raps_fidelity.json`) as evidence of pattern-selection variance across ICs, which
is real and feeds the IC-averaging design question (Task 17). Anything written against the
old headline (spec §4.3 fidelity paragraph, Task 17's dispatch, Task 21/22 roll-ups) must be
re-read against `results/raps_fidelity_paired.json`.

**Rejected.** (a) Re-using the committed 512² reference *arrays* — impossible, only scalars
were serialised; replaced by re-running all 24 references and PROVING identity (raises above
1e-9; observed max |Δk*| = 0.0, identical step counts). (b) Repairing the unpaired design by
raising seed counts — averaging over unrelated ICs estimates the variance floor, not grid
fidelity, at any seed count. (c) Deleting the unpaired result — it answers a different
question the project still needs.

**Calibration notes.** `NEAR_DUPLICATE_REL_TOL = 0.01` (fixture enumeration) is UNCALIBRATED
and gates only an arm-matching disclosure. One caveat travels with the binding fixture:
every `tune_comp_seed3` arm stops at its planned 606-step horizon
(`stopped_reason: "horizon"`), not detected saturation; the residual coarse-IC induction
bias is conservative in direction (inflates the coarse arm's distance), so it cannot
manufacture the "clears" verdict. The n-question itself remains an OWNER decision; this
entry records the measurement design, not a ruling on n.

### D-R3-5 — RULED: the truncated-unrolled path is PROMOTED to the primary gradient estimator for every member; the adjoint path is retained as the A/B verification path

**Date:** 2026-08-19. **Task:** R3 register-item-8 promotion unit (`feature/r3-integration`).
**Status:** OWNER-RULED. `docs/REDESIGN_rngrn.md` §8 item 8 pre-specified the promotion rule
before the runs that judge it; Task 14 measured the two quantities the rule reads; the owner
ruled "promote the unrolled path" on 2026-08-19. The rule was NOT re-decided at ruling time.

**The decision.** `train.gradient_path` defaults to `"unrolled"`. Every ignited-member spectral
solve takes `unrolled.unrolled_relax`'s truncated-unrolled gradient — converged members
included, which is precisely what changed. `gradient_path="adjoint"` restores
`forward.PatternSolve`'s implicit-function-theorem backward and is RETAINED as the A/B
verification path: the thing the promotion was measured against, and the thing a
re-measurement compares to. It is not a fallback for convenience.

**Evidence.** `experiments/redesign_r3/fd_ab/results/fd_ab.json` (Task 14, 10 FD directions x
the 5 active spectral terms, tol 1e-4, at the §4.3 operating point: commensurate box p=8,
n=96, k-hat=0.17607, L_solve=285.486). Both pre-specified clauses met:

| arm | adjoint worst rel. err. | unrolled worst rel. err. |
|---|---|---|
| converged (Newton residual 3.45e-13) | 1.70e-06 — PASSES | **1.92e-08 — PASSES** |
| stalled (Newton residual 1.79e-03) | **1.93 — FAILS** | **1.44e-08 — PASSES** |

- *Clause 1, FD-faithful at tol 1e-4 on BOTH converged and stalled members:* met, with four
  orders of margin on the converged arm and eight on the stalled one. The adjoint path fails
  the stalled arm by four orders — its premise F(u*) = 0 is false there by construction.
- *Clause 2, measured cost does not exceed the adjoint path's:* met. **2.95x cheaper** per
  member-step at B=1 — 1.543 s vs 4.562 s (`cost[]`, threads pinned to 1, relax hoisted out of
  both timings, 3 reps after a warm-up). The B=8 adjoint figure in the same artefact (104.3
  s/member) is contaminated in principle — `solve_subset` re-warms every member from one shared
  converged field — and is NOT load-bearing for this ruling. The clean B=1 comparison is.

**Retro-support (not part of the pre-specified rule, recorded because it explains the size of
what was being lost).** T16's population survey measured a **25.7% pooled stall rate**
(9/35 solved member-steps) off the known-Turing checkpoint, per-member median 16.7% and p90
72%, rising with distance from the checkpoint (D-R3-7). Under adjoint-primary every unswitched
stall was an O(1) gradient error. Read as a *caveat* on the rate itself: T16's genuine stalls
sat at residuals 1.2e-7 to 1.4e-5, one to two orders closer to the 1e-9 bar than Task 14's
forced probe at 1.79e-03, and whether the adjoint path's O(1) error also holds at that milder
miss is UNMEASURED.

**Design point resolved here — VALUE/GRADIENT CONSISTENCY.** The spec underdetermined what an
unrolled-primary step uses as its forward loss VALUE. **Ruled: the segment endpoint**, i.e. the
field the unrolled path returns is the one the loss is evaluated on AND the one the gradient
differentiates — one computation, as `scripts/r3_fd_ab.py`'s `UnrolledPath` already did.
Rejected: reporting the Newton-polished u* as the value while differentiating the segment,
which would hand the optimiser a gradient that is not the gradient of the loss it reads. The
swap is measurably benign where it can be checked: on the converged arm the two paths' base
values agree to ~3e-9 relative across all five terms (`arms.converged.paths.*.base_values`).
On a stalled member they do not agree, and there the polished value is the one premised on
F(u*) = 0, which is false.

**Riders — every one of these travels with the ruling.**
1. **The Newton polish is STILL RUN under the promoted default, and its result is discarded.**
   Only its residual is read, to classify the solve as a stall. The promotion therefore saves
   the LSMR adjoint backward — NOT the relax (the unrolled path needs a saturated warm state)
   and NOT the polish (the stall counter needs its residual). The retained polish's cost on top
   of the 1.543 s is **UNMEASURED**, so the 2.95x is the per-member-step estimator cost, not a
   whole-solve speedup. Dropping the polish would silence the stall rate that is this ruling's
   own retro-support; it stays until something measures that trade.
2. **Memory: +433 MB/process at B=1 and +516 MB at B=8 (`rss_step_increment_mb`), UNMEASURED
   beyond B=8.** Read against `CLAUDE.md` §7a, where host RAM is the binding resource and a
   trainer already sits at 1.47-1.68 GiB RSS: a half-GiB per-process increment is roughly a
   third of a trainer, and the guard's 8192 MB `MemAvailable` floor was sized for the old
   footprint. Nothing here re-sizes it; a pool launched under the promoted path must be
   watched.
3. **D-R3-1's S=128 conditionality now sits on the DEFAULT path.** `SEGMENT_STEPS_DEFAULT` was
   calibrated on ONE fixture, ONE box, ONE seed, and is explicitly UNCALIBRATED beyond that
   point (D-R3-2 condition (b)). Promotion does not calibrate it; it raises the stakes on it.
4. **D-R3-2's caller contract now guards the default path for EVERY member.** It is satisfied
   the way D-R3-2 says it should be — by construction, not by a runtime check:
   `forward.relax_to_pattern_torch` RAISES unless its own detector confirms saturation, so
   `recover._spectral_solve_with_stall_switch` cannot hand the unrolled path an unsaturated
   field. **No runtime saturation guard was added; D-R3-2 rejected that as "a guess dressed as
   a check" and this ruling does not reopen it.**
5. **`stall_switch_fraction` is DIAGNOSTIC-ONLY.** T16 retired it as a threshold (D-R3-7); it
   never gated anything, and the promotion answered its question by moving every member rather
   than by picking a rate. The field is KEPT and still recorded. The stall counters
   (`n_ignited_solves`, `n_stalled_solves`) stay live as instrumentation on the promoted path,
   and the run-index row now carries `gradient_path` alongside them.
6. **`gradient_path="unrolled"` REFUSES `batched=True` with a non-zero spectral weight.**
   `unrolled_relax` is serial-model-only — no `BatchedRNGRN` twin exists. It raises rather than
   silently solving through the other estimator, which would make a run non-comparable without
   saying so. A batched spectral run must set `gradient_path="adjoint"` deliberately and report
   that with its numbers. **This is a behaviour change for batched spectral runs**, of which
   there are none in `configs/` today (see the A0 clause).
7. **`stall_switch=True` with the promoted default RAISES** rather than meaning nothing: the
   switch is the fallback away from an adjoint primary, so a caller who set it expected one.

**THE A0 CLAUSE — announced loudly, because a silently non-comparable A0 would be the worst
outcome of this unit.** `docs/PLAN_redesign.md`'s Global Constraints make A0 untouchable and
require new behaviour to be opt-in; a flipped DEFAULT is opt-out, so the two collide unless A0
provably never reaches the seam. **It never does, and this was MEASURED, not argued.** A0's
objective carries no spectral weight — all five of `spec_shape`/`spec_aniso`/`spec_amp_mean`/
`spec_amp_fluct`/`real_moments` are 0.0 in `configs/base.yaml`, which every shipped config
composes — so `recover()` computes `use_spectral=False`, builds NO forward solver at all,
`losses.total` never calls `_apply_spectral`, and `train.gradient_path` is never read.
`tests/test_gradient_path.py::test_a0_recovery_is_bit_identical_under_both_gradient_paths` runs
the A0 objective twice, identical in every argument but the estimator, and asserts bit
identity of loss, `xstar`, `kstar_model` and all five parameter arrays.

**So: nothing needed pinning, and nothing was pinned to a stale estimator.** Pinning
`gradient_path: adjoint` into `configs/base.yaml` was CONSIDERED and REJECTED: base.yaml is
composed by every config, so that pin would silently hold every future redesign arm on the
estimator this ruling retired — the larger of the two risks, and the exact failure the
promotion exists to prevent. What was done instead: an explicit A0-protection comment at
`configs/base.yaml`'s `loss.weights` naming this entry, plus
`test_the_a0_baseline_objective_carries_no_spectral_weight`, which FAILS the moment any of the
five is raised. **Raising one of them is not a tweak to A0 — it is a different arm, on
whichever estimator `train.gradient_path` then selects, and it must be reported as such.**

**What was rejected.** *Keeping adjoint-primary with the Task 13 stall switch.* It forgoes a
measured ~3x cost saving and keeps the assumption-carrying path — the one whose premise
F(u*) = 0 fails on 25.7% of solves and whose gradient is then O(1) wrong — as the default, for
no measured benefit on either arm. The adjoint path measured *worse* than the unrolled path on
the converged arm too (1.70e-06 vs 1.92e-08), so there is not even a fidelity argument left for
it as a default.

**Amendment, 2026-08-19 (same unit).** `scripts/r3_fd_ab.py`'s forced-stall construction screens
its displaced field on a 5% channel-0 amplitude window, now named `SATURATION_AMPLITUDE_TOL` and
marked **UNCALIBRATED** at the site: it is an amplitude PROXY for D-R3-2's caller contract — a
proxy D-R3-2 itself explicitly declined to define — and must not be cited as evidence that the
contract was met.

**Where it lives:** `src/rngrn/recover.py` (`GRADIENT_PATHS`, `uses_switch_solver`,
`_spectral_solve_with_stall_switch`, `recover`), `src/rngrn/config.py`
(`TrainConfig.gradient_path`), `src/rngrn/train.py` (`_stall_columns`, the `fit()` call site),
`configs/base.yaml` (the A0-protection note), `tests/test_gradient_path.py`,
`tests/test_stall_accounting.py`. Evidence:
`experiments/redesign_r3/fd_ab/results/fd_ab.json`. Read with **D-R3-1** (the FD protocol),
**D-R3-2** (the segment length and its caller contract), **D-R3-3** (why truncation is there at
all) and **D-R3-7** (the stall rate).

### D-R3-6 — RULED: n stays 96 for Phase II

**Date:** 2026-08-19. **Task:** R3, arising from Task 15 (`feature/r3-raps-fidelity`), recorded
by the register-item-8 promotion unit. **Status:** CONTROLLER-DECIDED under the owner's explicit
delegation, per `CLAUDE.md` §10 ("You do not need to ask me for all science").

**The decision.** The training solve box keeps **n = 96** (`solve_box.N_DEFAULT`) for Phase II.
No change.

**Evidence — Task 15's PAIRED measurement**
(`experiments/redesign_r3/raps_fidelity/results/raps_fidelity_paired.json`; the design and why
the earlier unpaired headline does NOT bear on this are D-R3-4, which must be read with this
entry). 3 fixtures x 8 seeds x grids {96, 128, 256, 512}, every grid rolled out from a
band-limited copy of the SAME physical 512² initial condition.

1. **96² clears D3's ~31 %/bin estimation floor on every seed of every fixture.** Worst
   per-bin distance by fixture: 16.78 % (`tune_comp_seed3`), 9.38 % (`d5_seed3`), 0.08 %
   (`d5_seed5`); `clears_d3_floor_all_seeds: true` for all three. **Worst-case margin 1.85x**
   (31 / 16.78). The residual coarse-IC induction bias is conservative in DIRECTION — it
   inflates the coarse arm's distance — so it cannot manufacture the "clears" verdict
   (D-R3-4).
2. **n cannot buy band resolution.** `n_band_bins_by_grid` is **7 bins at 96, 128, 256 AND
   512** — the band `[0.6, 1.55] k/k*` is resolved identically at every grid, because the bin
   count is set by the block size `p`, not by n. So raising n cannot improve the one thing a
   spectral objective reads off the band.
3. **512² costs 58.3-69.1x per rollout on the PAIRED sweep** (`seconds_mean` at 512 vs 96 in
   `raps_fidelity_paired.json`: 69.1x `tune_comp_seed3`, 61.6x `d5_seed3`, 58.3x `d5_seed5`).
   **Two artefact-supported cost ranges exist for this quantity and they are not
   interchangeable.** Task 15's reviewed report quotes **59.6-77.6x** — the same ratio measured
   on the **UNPAIRED** sweep (`raps_fidelity.json`). Both come from committed JSON and neither
   is informal; they differ because the two designs integrate different initial conditions.
   **This entry's basis is the paired sweep throughout**, D-R3-4 having superseded the unpaired
   one for exactly this comparison, so 58.3-69.1x is the range to cite alongside this ruling
   and 59.6-77.6x is the range to cite when discussing the unpaired sweep on its own terms.
   The rejection below does not turn on which is used.

**What was rejected.**
- **n = 128 or 256.** Both show real paired gains (worst per-bin at `tune_comp_seed3`: 16.78 %
  -> 10.21 % -> 0.98 %). Rejected anyway: the bar is D3's floor and 96² already clears it with
  1.85x margin, while 128/256 cost 2-8x per rollout for no BAR-RELEVANT gain. A cost that buys
  a better number nobody is judged on is not a reason.
- **n = 512.** Trivially clears (it is the reference), and reprices the entire training loop
  by ~58-69x paired (~60-78x unpaired). Not defensible against a bar already met on either
  measurement.

**The open lever, routed to Task 17 — this is what the evidence actually reframes.** The
quantity that does NOT shrink with n is nonlinear PATTERN-SELECTION variance across initial
conditions: 34-45 %/bin in the unpaired sweep (`results/raps_fidelity.json`, retained and
reframed by D-R3-4), which is *above* D3's 31 % floor. Raising n cannot touch it — it is a
property of the dynamics, not the grid. **Averaging multiple ICs per training step is
therefore the candidate lever that raising n is not**, and Task 17 owns whether it is worth
its cost. Recording that here so the n-question is not silently re-opened as a proxy for it.

**Scope / what is NOT established.** Three fixtures, all N=3, all p=8, one band, one estimator,
`ic_noise=0.01`. The verdict is for the Phase II training solve box only. **Validation is
untouched: it stays 512² on the data box** — no cheaper grid is licensed for any morphology or
gate claim (D-FFT-9 closure 3, `docs/REDESIGN_rngrn.md` §4.3). D-R3-7's stall-rate distribution
was measured at n=96 and is a TRANSFER CAVEAT against this ruling: the Newton residual, and so
the stalled/converged classification, is a function of the grid, and that distribution does not
automatically carry over if this ruling ever changes.

**Where it lives:** `experiments/redesign_r3/raps_fidelity/results/raps_fidelity_paired.json`,
`scripts/r3_raps_fidelity.py`, `src/rngrn/solve_box.py` (`N_DEFAULT`). Read with **D-R3-4**
(the paired design and the superseded unpaired headline) and **D-R3-7** (the n=96 transfer
caveat).

### D-R3-7 — the population stall-rate distribution for register item 14 (stall_switch_fraction)

**Date:** 2026-08-19, fix round 2026-08-19 (R3 Task 16, `feature/r3-integration`).
**Status:** COMMITTED with the register-item-8 promotion unit (D-R3-5/6/7 together). The
`stall_switch_fraction` question this measurement fed was answered by the promotion, which
moved every member to the unrolled path rather than picking a rate — see D-R3-5 rider 5. This
entry records the measurement.

**Context.** `docs/REDESIGN_rngrn.md` §4.3 / spec §8 register item 14 leaves
`stall_switch_fraction=0.20` UNCALIBRATED. Task 13 wired the accounting; Task 14's FD A/B gate
measured that an unswitched stall is a genuinely wrong gradient (adjoint O(1) wrong at a forced
residual of 5e-4 to 6e-3, worst 1.93, vs the unrolled path's ~1e-8 on both arms) but found no
genuine stall in its own budget-limited (1 trial/scale) hunt. This task ran a population survey
(6 perturbation scales x 2 members x 6 steps = 72 member-steps worst case, 48 attempted, 35
solved) directly against the commensurate box (Task 11's `solve_box.geometry`/`needs_retile`)
and Task 13's stall decision logic (a residual-capturing copy of
`_spectral_solve_with_stall_switch`, equivalence-checked against the original).

**The finding.** Pooled stall fraction 25.7% (9/35), but the per-member distribution is wide
(median 16.7%, p90 72%, range 0-100%, n=8 members with >=1 solved step) and rises sharply with
distance from the known-Turing checkpoint (0% at scale 0.1, 60% at scale 2.0, small-n at the
extremes). The measured genuine-stall residuals (1.2e-7 to 1.4e-5) are 1-2 orders of magnitude
closer to the 1e-9 bar than Task 14's forced-stall probe (5e-4 to 6e-3) — whether the adjoint
path's O(1) gradient error also holds at that milder miss is UNMEASURED.

**Two divergences from production's own counters, disclosed rather than silently absorbed
(review findings I1/I2), because the 25.7% is not a bit-for-bit reproduction of what a real
run's `n_stalled_solves`/`n_ignited_solves` would show:**
1. The ignition gate used is `sig.max() > 1e-3` on this script's own fixed k-grid, not
   `losses.spectral.is_ignited`'s real `sig_max_pos` gate (whose k-floor would have de-ignited
   this entire population — all `khat` here fall in [0.101, 0.191], below where `sig_max_pos`
   starts looking). This survey's gate is strictly MORE PERMISSIVE than production's.
2. A step whose solve fails for a non-stall reason is excluded from both the numerator and
   denominator here; production counts it in the denominator as a non-stall
   (`recover._account_for_stall`), unconditional on the solve's outcome.
   Zero non-"ok" solves occurred in this run, so 25.7% is unaffected by this divergence this
   time, but the two denominators are not the same thing in general.

**What was NOT decided here**: whether `stall_switch_fraction` should be set to a specific
number, treated as a post-hoc diagnostic only (re-reading `_spectral_solve_with_stall_switch`'s
own docstring, routing is UNCONDITIONAL on whether the given solve stalled — the fraction is
recorded, not read as a gate, in the code as it stands today), or left for calibration against
a real training run once R4 wires the commensurate box into `recover()`. Four options are laid
out in `task-16-report.md`'s Step 3 section for the controller.

**Scope of the finding.** One checkpoint (N=3, `m3_registry_20260803_190250_seed3`), theta
perturbations of one known-Turing point (not a from-scratch training trajectory), an Adam loop
driven only by the 5 spectral terms (not the full A0 objective). 4 of 12 members left the
Turing class before any solve was attempted and are correctly excluded from the denominator —
VERIFIED post-hoc by a cheap re-scan (`init_scan`, no training steps) of all 12 members'
starting dispersion, which reproduces `n_init_ignited=8` exactly (review finding I3c).
**Measured at n=96** (`solve_box.N_DEFAULT`) — the Newton residual, and therefore the
stalled/converged classification itself, is a function of the grid, not of theta alone. The
controller has since ruled n=96 for Phase II (D-R3-6); this distribution is a
TRANSFER CAVEAT against that ruling — it does not automatically carry over if the ruled grid
ever changes (review finding I4).

**Where it lives:** `experiments/redesign_r3/stall_rate/results/stall_rate.json` (fix round 1
added `init_scan` — the init-only re-scan verifying the 4/12 split — to the committed
artifact; the full per-member per-step `all_members` trace is persisted by the FIXED code for
future runs only and is NOT in the committed artifact, which predates the fix);
`scripts/r3_stall_survey.py`. Read with **D-R3-1** (the FD protocol) and **D-R3-3** (the R3
unrolled-path finding this task's residual comparison extends).
### D-R3-8 — PROPOSED: the R3 Phase-II B/K curve — wall-clock, not memory, bounds B (register item 14)

<!-- Task 17, feature/r3-bk-curve. Numbered D-R3-8 because D-R3-5..7 were taken on the
rolling branch while this task was in flight; this block appends cleanly under the
anti-conflict rule. -->

**Date:** 2026-08-19/20 (measured), 2026-09-01 (harvested). R3 Task 17,
`feature/r3-bk-curve`. **Status:** PROPOSED — evidence returned to the controller; the
B/K ruling itself (B=512/K=16, register item 14) is NOT made here, and both stay
UNCALIBRATED until it is.

**Context.** The redesign register carries B=512/K=16 as uncalibrated placeholders. The
only prior curve (R2 Task 16) is ignition-only — pinned x*, no steady-state solve, flat
RSS — and its own report declines to set K from it. This task measured the missing
object: per-Adam-step cost and memory of the R3 Phase-II step WITH the batched forward
solve in the loop; primary arm unrolled (per D-R3-5), one fresh step per (arm, B) cell,
1 thread, spawn-isolated, §7a-guarded (`scripts/r3_batch_curve.py`).

**The finding.** Unrolled B=1..64 measured; B=128 timed out at the script's 3600 s/cell
wall-clock valve; the §7a memory guard never fired (`refused` empty).
1. **The binding constraint on B is wall-clock, not memory**: s/step grew 7.67 (B=1) →
   2298 (B=64, 38.3 min), ≈2.2–2.6× per doubling (one doubling, B=8→16, measured 3.68×);
   peak cell RSS reached only 3257 MB and MemAvailable never went below ~13 GB against
   the 8192 MB floor. The RSS-budget arithmetic projects a memory ceiling near B≈516 —
   moot on this host, wall-clock excludes B≥128 first.
2. **Per-member cost is superlinear**: member-steps/s falls 0.1305 → 0.0278 (4.7×)
   across B=1→64, so raising B buys much less than proportional throughput; what it buys
   at full price is IC-diversity per Adam step, which D-R3-4's
   pattern-selection-variance finding suggests is where B's statistical value lives
   (framed, not ruled).
3. **Adjoint cross-check** (B∈{1,8}): unrolled beat adjoint 4.9×/4.5× per member-step —
   independently consistent with D-R3-5's ~5×; adjoint peak RSS lower (717–1009 MB).
4. B=512 unrolled at this host/convention projects to several hours per single Adam
   step (projection, not measurement).

**What was NOT decided here:** the B/K ruling (returned to the controller); K's
distinct-structure ceiling (cited from R2 Task 16 only — this task ran no population to
convergence and adds no count); the IC-averaging-vs-throughput reading (framed per
D-R3-4, not ruled); the B=32/64 RSS jump above the linear `_half_coeffs_batched` model
(open, did not affect any finding). Cells are single-sample (reps=1, by design —
fresh-relax steps at ~70 s/solve), so individual doubling ratios are noisy against R2's
measured ≥1.65× repeat spread; the trend and orders of magnitude are the load-bearing
content, and rejected alternatives (reps=3 warm-started, per `r3_fd_ab.py`) are
documented in the script and report.

**Where it lives:** `experiments/redesign_r3/batch_curve/results/batch_curve.json` (one
clean invocation, total 7831.8 s); `scripts/r3_batch_curve.py`;
`.superpowers/sdd/PLAN_redesign_R3/task-17-report.md`. Read with **D-R3-4** (IC
variance), **D-R3-5** (unrolled promotion), and R2's `task-16-report.md` (the
ignition-only curve this one is explicitly NOT a re-read of).
