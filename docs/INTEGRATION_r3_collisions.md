# INTEGRATION_r3_collisions.md — the R3 Phase-B collision ledger

Built by **Task 7 of `docs/PLAN_redesign_R3.md`, before a single line of Phase B was edited.**
Its purpose is the one CLAUDE.md §11 names: git resolves at line granularity and cannot tell
that two additions are independent, so "keep both sides" is right for an additive block and
**silently wrong** inside a function signature or a dict literal. This file writes down the
intended union *first*, so that when the merge produces something else the difference is
visible rather than invisible.

Nothing here is a merge. No `src/` file was edited by Task 7.

---

## 0. The two sides, and how they were read

| | ref | what it is |
|---|---|---|
| **BASE** | `10cff1b` (`chore/fft-refactor`) | the true merge base of the two lines, computed with `git merge-base` |
| **redesign line** | `db40995` = `feature/redesign-model` (= `feature/r3-integration` at BASE) | R2's T8–T15 loss-term registry refactor + T16's `BatchedRNGRN` pin/box support, plus the reviewed Phase-I record |
| **gpu-optim line** | `dbd46fe` = `feature/gpu-optim-repair` | the gpu-optim delta **as repaired by R3 Phase A**; Phase A's verdict is APPROVED for integration. A doc-only close commit may land after `dbd46fe`; `src/` is frozen at it. |

Every row below was read on **both** sides in source (`git show <ref>:<path>`), not inferred
from the review. Points 1–26 are transcribed **verbatim** from
`docs/REVIEW_gpu_optim_delta.md` §10 on the gpu-optim-repair branch (they are quoted in the
"gpu-optim's change" column and preserved id-for-id); points 27–28 come from this plan's Task 7
brief; points 29–35 were added by Task 7 from the Phase-A review findings and from reading the
two trees.

### 0a. The measured conflict surface is much narrower than §10 implies

`comm -12` over the two sides' changed-file lists:

```
$ comm -12 <(git diff --name-only 10cff1b..db40995 | sort) \
           <(git diff --name-only 10cff1b..dbd46fe | sort)
docs/DECISIONS.md
src/rngrn/losses/terms.py
src/rngrn/losses/total.py
src/rngrn/model.py
```

**Exactly four files are touched on both sides.** Everything else in §10 —
`losses/spectral.py` (points 14–18), `recover.py` (points 19–25), `config.py` (part of 26),
`observables.py`, `forward.py`, `eval/*`, `tests/test_ignition_gating.py` — is **one-sided**:
git will apply it without a textual conflict. That is *not* the same as safe. A one-sided hunk
can still contradict a contract the other side introduced (point 28 is exactly that: no
conflicting text, a false registry declaration). The ledger therefore marks each row with a
**Shape** that says what can go wrong, independently of whether git will complain.

### 0b. Shape legend — the four silently-destructive classes (Task 7 brief Step 2)

| tag | shape | why it is silent |
|---|---|---|
| **SIG** | signature-level merge | a duplicate keyword argument is a `SyntaxError` if git interleaves badly, but a *reordered* or *dropped* one is not — the phase-A duplicate-keyword failure |
| **DICT** | dict-literal edit | a duplicate key silently keeps the **last** |
| **DEL** | a deletion on one side | merges badly against any edit to the same block and can be **silently resurrected** |
| **DEF** | field / argument default | the class that silently reverted `resid` in phase A |
| **CONTRACT** | no textual conflict, but one side's declaration becomes false | invisible to git *and* to a diff review |
| — | additive / mechanical | keep both sides; the ordinary case |

---

## 1. The ledger

Owner column follows `docs/PLAN_redesign_R3.md` task scopes and the controller's standing
rulings. **One owner per row, exactly** (Ruling R3-PF-1). Task 10 Step 3 ticks the *Verified*
box on every row with named evidence.

| # | File | Symbol | gpu-optim's change | redesign line's change | Chosen resolution | Shape | Owner | Verified |
|---|---|---|---|---|---|---|---|---|
| 1 | `losses/total.py` | `compute_terms:131-146` | term-assembly block rewritten: hoisted `J = model.jacobian(x_disp, create_graph=True)` plus `J=`/`idx=` threaded into four term calls | **none in the block itself.** R2's whole `total.py` delta is the new import `from .term_registry import LOSS_TERMS` (**`total.py:16`**) plus the two refusal messages (row 8) | Rebuild the block **through the registry**, not by merging two hand-written blocks. The `:16` import is what makes that possible and **must survive the merge** — row 8 removes the only two lines that currently use it, so a careless resolution can drop it as newly-unused (§2.1) | DICT | **T9** | ☐ |
| 2 | `losses/total.py` | `compute_terms_batched:321-336` | the same rewrite, mirrored | **none** | mirror of row 1, same vehicle | DICT | **T9** | ☐ |
| 3 | `losses/total.py` | `compute_terms` / `total_loss` signatures | `obs_scale`, `kstar_idx` appended | **none** | append both, at the end, defaulting `None`; keep the None-means-compute-it-here semantics verbatim (§2.2) | **SIG** | **T9** | ☐ |
| 4 | `losses/total.py` | `compute_terms_batched` / `total_loss_batched` signatures | `obs_scale`, `kstar_idx`, `active` appended | **none** | as row 3 plus `active=None`. **Four signature merges in one file** — the phase-A shape (§2.2) | **SIG** | **T9** | ☐ |
| 5 | `losses/total.py` | `_apply_spectral_batched:188-253` | new 66-line function inserted immediately above `compute_terms_batched` | **none** | insert verbatim; purely additive | additive | **T9** | ☐ |
| 6 | `losses/total.py` | `total_loss_batched`'s `L_<key>` **NaN-masking** loop | newly couples the parts-writing loop to `SPECTRAL_TERM_KEYS` and `parts["spec_computed"]`: a spectral term not computed for a member has its *loss* placeholder at an exact 0 but its **record** masked to NaN, so a reader can tell "not computed" from "zero loss" (CLAUDE.md §4) | **none** | adopt verbatim, NaN masking included — dropping the mask is a silent evidence defect, not a behaviour change. Interacts with `history.py` — row 32 | additive | **T9** | ☐ |
| 7 | `losses/total.py` | `parts_member` | new dtype-`kind` branch for object/string arrays | **none** | adopt verbatim | additive | **T9** | ☐ |
| 8 | `losses/total.py` | the `spectral is not None` refusal in `compute_terms_batched` | **deletion** — *and Phase A Task 1 changed its shape*: it is no longer a bare deletion but a **19-line replacement validation block at the head of `compute_terms_batched`** (`isinstance(model, BatchedRNGRN)` + `hasattr(spectral.solver, "solve_subset")`) | **R2 rewrote the very same two raises** to single-source their text from `LOSS_TERMS.get(...).refusal_reason` | **Keep the replacement validation; never restore the old refusal.** R2's single-sourcing survives only for `resid`; the spectral refusal text is retired with the refusal (§2.3) | **DEL** | **T9** | ☐ |
| 9 | `losses/terms.py` | `J=None` on `turing_hinges`, `turing_hinges_split`, `anticollapse`, `kstar_anchor`, `turing_hinges_batched`, `turing_hinges_split_batched`, `anticollapse_batched`, `kstar_anchor_batched` | new `J=None` on **eight** term functions. *(§10's prose says "six"; the list it then gives has eight entries. Eight is correct — verified in source. This is a deliberate correction of the review, not a transcription slip.)* | R2 **standardised these onto the registry** (`LossTerm.fn` / `.batched_fn`) and added `kstar_anchor_si` / `kstar_anchor_si_batched` as a ninth and tenth | **Express the hoist through the registry's call contract, not by hand** (plan T8 Step 1). If that is impossible without changing `LossTerm`'s contract, that is a decision point to **return** (§2.4) | **SIG** | **T8** | ☐ |
| 10 | `losses/terms.py` | `idx=None` on `_sigma_at`, `kstar_anchor`, `kstar_anchor_batched` | precomputed bracketing index | R2's `kstar_anchor_si` / `_si_batched` **inline their own `searchsorted`** and take no `idx` | Same registry vehicle as row 9; **and thread `idx=` into the two `_si` twins too**, or they silently keep the per-step round trip the hoist exists to remove (§2.4) | **SIG** | **T8** | ☐ |
| 11 | `losses/terms.py` | `param_prior` | `box=` promoted over `box_path=` | **none** | adopt; the matching call-site change is row 19, owned by T8 | **SIG** | **T8** | ☐ |
| 12 | `losses/terms.py` | `_damped_newton:36-70` | body fully rewritten (vectorised 30-candidate line search; `_LINE_SEARCH_HALVINGS` module constant added above it) | **none** | adopt wholesale. Verify it against R2's **pinned/boxed** models (§2.5) | additive | **T8** | ☐ |
| 13 | `losses/terms.py` | `steady_state_batched._run:590-610` | inner loop restructured (break relocated, 5-halving cadence, singular-J handling inlined) | **none** | adopt wholesale; same pinned/boxed verification as row 12 | additive | **T8** | ☐ |
| 14 | `losses/spectral.py` | `_RAPS_BINS` cache + `_raps_bins`; `raps_torch` body rewritten | new cache above `raps_torch` | **none (file untouched by R2)** | transplant wholesale (T9 Step 1) | additive | **T9** | ☐ |
| 15 | `losses/spectral.py` | `build_frame_targets` | new `device=` parameter and a **new `band_idx` key in the returned targets dict** | **none** | transplant wholesale. R2 makes no edit to that dict literal, so the §11 duplicate-key hazard does **not** fire here — record that as the reason, not as luck | **DICT** | **T9** | ☐ |
| 16 | `losses/spectral.py` | `spec_shape` | band selection rewritten from boolean mask to `index_select` | **none** | transplant wholesale | additive | **T9** | ☐ |
| 17 | `losses/spectral.py` | the "BATCHED twins" section | new ~150 lines between `is_ignited` and `SpectralContext` | **none** | transplant wholesale. These are the callables row 28 puts into the registry | additive | **T9** | ☐ |
| 18 | `losses/spectral.py` | `SpectralContext` docstring | fully rewritten (the duck-typed interface is now assembler-dependent) | **none** | transplant wholesale | additive | **T9** | ☐ |
| 19 | `recover.py` | `param_prior_kw` | construction moved out of the `term_kw` literal and reshaped (`box_path` → `box`), loading `_load_box_bounds(bio_box_path)` once | **none** | adopt. **Owned by T8** (Ruling R3-PF-1); T9's Step-4 mention of point 19 reduces to *verification* (§2.6) | **DICT** | **T8** | ☐ |
| 20 | `recover.py` | `term_kw` | gains `obs_scale` and `kstar_idx` by post-hoc assignment | **none** | adopt as post-hoc assignment, **not** folded back into the literal — that is what keeps it merge-safe (§2.6) | **DICT** | **T8** | ☐ |
| 21 | `recover.py` | the `use_spectral and batched` refusal block | **deleted** — and, like row 8, *replaced* rather than merely removed: the RAISES list in `recover`'s docstring is rewritten and the two remaining refusals (`split_hinges=False`, adaptive strategy) stay | **none** | keep it deleted; keep the rewritten docstring. **Do not restore** (§2.3) | **DEL** | **T9** | ☐ |
| 22 | `recover.py` | `build_frame_targets(...)` call | gains `device=dev` | **none** | adopt | additive | **T9** | ☐ |
| 23 | `recover.py` | `_batched_restarts` | signature gains `spec_cfg`, `spec_targets`; its call site gains both | **none** | adopt; append at the end of the signature | **SIG** | **T9** | ☐ |
| 24 | `recover.py` | `_batched_restarts` liveness block | rewritten (`died_at_step`, `LIVENESS_SYNC_EVERY=25`); two `total_loss_batched` call sites gain `active=` | **none** | adopt. Note the two `active=` sites are guarded `active=alive if spec_cfg is not None else None` — preserve that guard exactly (§2.7) | **SIG** | **T8** | ☐ |
| 25 | `recover.py` | serial loop | `lbfgs_error` variable and a new `restart_log` key | **none** | adopt. The batched path deliberately does **not** emit the key (D-PERF-8) — that asymmetry is recorded, not closed | **DICT** | **T8** | ☐ |
| 26 | `model.py`, `config.py` | `dispersion_backend` default in `RNGRN.__init__`, `BatchedRNGRN.from_seeds`, `ModelConfig`, `recover()`; plus the resolution block at `model.py:147` and the three-way `assert` | default `"eig"` → `"auto"`, resolved to `cubic` at N=3 in `__init__` | **R2 edits the same three sites**: `RNGRN.__init__` gains `pin_xstar`/`param_boxes` **and a large validation block immediately after the `dispersion_backend` assert**; `from_seeds` gains both; `member(b)` re-passes them | **RULED by the controller: adopt the MECHANICS, keep the DEFAULT at `"eig"`.** The three-way assert and the `if dispersion_backend == "auto"` resolution block land; all four defaults (`RNGRN.__init__`, `BatchedRNGRN.from_seeds`, `ModelConfig`, `recover`) stay `"eig"`. **The flip itself is parked at T8 Step 3 — T8 must NOT re-escalate it.** The SIG half is unblocked; §2.8 gives the exact line-level union | **DEF** + **SIG** | **T8** | ☐ |
| 27 | `eval/numerics.py` | `integrate_bdf1_newton_krylov` | stub now **raises** instead of silently falling back to ETDRK4 (D-PERF-9) | **none on this line.** Task 7 verified: the other independent change lives on **`feature/lift-ladder`** (`413bc8a`, `091303b`), which `feature/redesign-model` does **not** contain | **Straight transplant here; the real conflict is deferred to the lift-ladder merge.** The two texts *are* different (§2.9) — a scoping correction to T8, not a ruling | additive **here**, **DEL/CONTRACT later** | **T8** | ☐ |
| 28 | `losses/term_registry.py` | the five `SPECTRAL_TERM_KEYS` entries (`:103-104`) | gpu-optim **implements** batched spectral terms (`spec_shape_batched`, … `real_moments_batched`) and `spectral_terms_batched` | R2 registered all five with `batched_fn=None`, `refusal_reason=_SPECTRAL_REFUSAL` | Flip to `batched_fn=getattr(S, f"{key}_batched")`, `refusal_reason=None`; delete `_SPECTRAL_REFUSAL`; fix the module docstring. **NOTHING FORCES THIS — corrected.** The XOR test does **not** fail (`:103-104` hardcodes `None, _SPECTRAL_REFUSAL`, so the XOR still holds), `batched_fn` has **zero runtime readers**, and row 8 deletes `total.py:209`, `refusal_reason`'s only spectral consumer. A T9 that lands rows 8 + 14–18 and skips the flip leaves a **false registry declaration under a fully green suite**. Verification is mandatory, not incidental (§2.10) | **CONTRACT** | **T9** | ☐ |
| 29 | `losses/total.py:18` | `from ..model import BatchedRNGRN` | Phase A added a **module-level** import — the loss layer's first dependency on `model.py` | R2's registry: `term_registry` imports `..registry`, `.spectral`, `.terms`; `terms.py` imports `.term_registry` at line 842 (a deliberate bottom-of-module cycle) | **Adopt. Task 7 pre-verified there is no cycle** — importing `rngrn.model` pulls in zero `rngrn.losses.*` modules (§2.11). That is a pre-merge check on the *unmerged* trees; the box below stays **unticked** because this column means verified-**after**-integration, and T9/T10 must re-run the one-liner in the merged tree | CONTRACT | **T9** | ☐ |
| 30 | `observables.py` | `_raps_torch_bins`, `raps_torch`, `kstar_of_torch` (D-OBS-1: `np.digitize` binning, not `floor`) | ~117 new lines | **file untouched by R2 since BASE** | Transplant wholesale at **T9 Step 1** (standing Ruling R3-PF-2). Task 7 VERIFIED it is purely additive: `raps` and `kstar_of` — the functions `recover.py` actually calls for `kstar_obs` — are **byte-unchanged**, so **no R2 number becomes non-comparable through this hunk** (§2.12) | additive | **T9** | ☐ |
| 31 | `tests/test_ignition_gating.py` | section (d); two renamed tests | rewritten by Phase A (+203 lines) | **file untouched by R2** — confirmed by the `comm` in §0a | Adopt wholesale. **No conflict surface exists**; the risk the brief anticipated does not materialise | additive | **T9** | ☐ |
| 32 | `history.py` | `_scalars_from` / `_names` frozen-column contract | `_apply_spectral_batched` writes three new per-member `parts` keys: `spectral_skipped` (object array of **strings**), `spec_computed` (bool), `spec_ignited` (float) | R2 rewrote `history.py` (+62): `EVENT_KINDS`, `record_event`, `record_invariants` | **Compatible — Task 7 VERIFIED by reading the source.** `_scalars_from` admits only `DIAG_KEYS`, `L_*`, `ss_converged` and `weights_used`; `spectral_skipped`/`spec_computed` match none of those and are never passed to `float()`. `spec_ignited` **is** in `DIAG_KEYS` and gpu-optim writes it as a float array — the intended shape. Re-verify at T10 (§2.13) | CONTRACT | **T9** | ☐ |
| 33 | `docs/DECISIONS.md` | — | appends `D-PERF-3` … `D-PERF-9`, `D-OBS-1` | appends `D-LIFT-1`, `D-REDESIGN-1/2`, `D-R2-1/2/3` | Keep both sides; the id namespaces are **disjoint** (verified). A pure append conflict — resolve by concatenation, chronological | additive | **T8** (first to touch it) | ☐ |
| 34 | `tests/test_firewall.py` | `RECOVERY_SIDE`, `FORBIDDEN` | untouched by gpu-optim | R2 added `losses/term_registry.py` to `RECOVERY_SIDE`, and `measure_frame_bias` + `r2_ignition_run` to `FORBIDDEN` | **HARD STOP.** No Phase-B task may edit this file without an orchestrator ruling. No textual conflict exists; the open question is whether the transplanted `forward.py`/`spectral.py`/`observables.py` surface needs re-classification — **that is T10 Step 4's `firewall-auditor` run, not an edit** (§2.14) | CONTRACT | **T10** | ☐ |
| 35 | `tests/test_term_registry.py` | `test_default_weights_are_bit_identical_to_the_legacy_dict` | — | R2 made `terms.DEFAULT_WEIGHTS = _term_registry.default_weights()` (`terms.py:843`), so both sides of this assertion derive from the same source | **DEFERRED MINOR, recorded so it is not mistaken for protection.** The A0 pin is **tautological** and cannot catch a default flip. T8 Step 2 is already told not to rely on it; add a non-tautological pin **if cheap**, do not gold-plate (§2.15) | CONTRACT | **T8** | ☐ |

---

## 2. Intended union, in prose — written before any edit exists

### 2.1 Rows 1–2 — the term-assembly block (`total.py`)

The union is **not** "gpu-optim's block with R2's registry bolted on". gpu-optim hoists one
`J = model.jacobian(x_disp, create_graph=True)` per step and threads `J=`/`idx=` into four
hand-written term calls. R2 exists precisely to stop those four calls being hand-written. The
intended union is: `compute_terms` and `compute_terms_batched` iterate the registry, obtaining
each term's callable from `LossTerm.fn` / `LossTerm.batched_fn`, and pass the hoisted `J` and
`idx` through the registry's call contract. The `term_vals = dict(kstar=…, turing=…,
anticollapse=…, anchor=…)` literal is then **built, not written** — which removes the DICT
hazard rather than surviving it.

If that cannot be done without changing `LossTerm`'s field set, **stop and return it**: it is
a contract change to a structure T14 already extended once (`kstar_si`), and it is not Task 9's
to rule on alone.

The arithmetic that must survive unchanged, whatever the vehicle: **one** Jacobian per step,
evaluated at the same `x_disp` with `create_graph=True`, shared by `kstar_anchor`,
`turing_hinges*` and `anticollapse` — gpu-optim's own note is that these were three *identical*
autograd Jacobian calls of which two were pure waste. `frame_scale_anchor` does not take `J`
and must not acquire one.

### 2.2 Rows 3–4 — four signature merges in one file

This is the exact shape of the phase-A duplicate-keyword failure, and it is four occurrences
inside one file. The rule for all four:

- `compute_terms(…, spectral=None, obs_scale=None, kstar_idx=None)`
- `total_loss(…, spectral=None, obs_scale=None, kstar_idx=None)`
- `compute_terms_batched(…, spectral=None, obs_scale=None, kstar_idx=None, active=None)`
- `total_loss_batched(…, spectral=None, obs_scale=None, kstar_idx=None, active=None)`

New parameters go **at the end**, in that order, each defaulting to `None`, and each keeps its
"None reproduces the previous per-step computation exactly" semantics. `active` exists **only**
on the batched pair. After the merge, `grep -c "obs_scale" src/rngrn/losses/total.py` and a
read of each `def` line is the check — a signature that gained the parameter twice, or lost it,
is the failure mode, and neither raises.

### 2.3 Rows 8 and 21 — the two deletions, both of which Phase A *replaced*

Both refusals said the same thing: the forward solve has no batched form. That was true when
written and is no longer. Neither is *weakened* — both are replaced by something narrower and
more specific:

- **Row 8** (`compute_terms_batched`): Phase A Task 1 replaced the deletion with a **19-line
  validation block at the head of the function**, before any steady-state solve or Jacobian —
  `isinstance(model, BatchedRNGRN)`, and `hasattr(spectral.solver, "solve_subset")` to catch
  the now-plausible mistake of handing the *serial* `forward.PatternSolver` to the batched
  assembler. **Keep the replacement validation. Never restore the old refusal.**
- **Row 21** (`recover`): the `use_spectral and batched` raise is gone; the remaining refusals
  (`split_hinges=False`, adaptive strategy, non-identity `observed_idx`) stay, and the
  docstring's RAISES list is rewritten to match. Keep the rewrite.

A resurrected refusal is the specific failure to watch for, because it does not look like
damage — it looks like the code that was always there. After the merge:
`grep -n "cannot compute the spectral terms" src/rngrn/` must return **zero** hits in `src/`.

The R2-side casualty is stated plainly: R2's `total.py` change existed to single-source those
two raise texts from the registry. **`resid`'s single-sourcing survives; the spectral one does
not**, because the string it single-sourced is being retired along with the refusal. Row 28
finishes the job by deleting `_SPECTRAL_REFUSAL` itself, so nothing is left half-wired.

### 2.4 Rows 9–10 — `J=` / `idx=` against a registry that standardised the signatures

`docs/PLAN_redesign_R3.md` T8 Step 1 is explicit: **the registry is the resolution vehicle.**
Threading `J=`/`idx=` through six-to-eight term signatures by hand, against a registry whose
whole purpose was to standardise them, reintroduces the divergence R2 removed.

Two things Task 8 must not miss:

1. R2 added **two** functions gpu-optim never saw — `kstar_anchor_si` and
   `kstar_anchor_si_batched` (T14, born registered, not yet wired into
   `compute_terms`/`total_loss`). Both call `model.dispersion(xstar, kgrid, J=None)` and
   `kstar_anchor_si_batched` inlines its own `searchsorted`. If the hoist is threaded through
   the registry, these get it for free; if it is threaded by hand, they **silently keep the
   per-step round trip**, and the divergence is invisible because neither is wired in yet.
2. `_sigma_at(sig, kgrid, k, idx=None)` is the shared helper. `kstar_anchor_si` calls it;
   `kstar_anchor_si_batched` does not (it duplicates the interpolation inline). Union: give
   `_sigma_at` the `idx=` parameter once, and let both `_si` twins use it.

### 2.5 Rows 12–13 — the two rewritten Newton solvers, against R2's pinned/boxed models

`_damped_newton` and `steady_state_batched._run` are one-sided rewrites, so git will apply them
cleanly. The risk is semantic and is entirely R2's: T12 makes `beta` a **derived** quantity
(`delta * x* - prod(x*)`) with no `theta_beta` parameter, and T13 maps `alpha`/`delta` through
`low + (high-low)*sigmoid(raw)` instead of softplus.

Task 7 traced the shapes and found no breakage: the vectorised line search evaluates
`model.reaction(xn)` at `xn` of shape `(30, N)`, and a pinned `RNGRN`'s `beta` is `(N,)` while
`prod` broadcasts to `(30, N)` — `beta + prod - delta * x` is well-formed. The batched path is
unchanged in shape. **But this is a shape argument, not a measurement.** Task 8 must run
`tests/test_pinned_model.py`, `tests/test_box_reparam.py` and
`tests/test_batched_pinned_boxed.py` after each of rows 12 and 13, not once at the end.

Note also what gpu-optim itself states about row 12: the vectorised form is **not** guaranteed
bit-identical to the last ulp (a candidate exactly on the acceptance boundary could be decided
either way), and it is a **measured ~12% CPU slowdown** in exchange for a GPU-sync win. Row 13
carries an unmeasured CPU cost of the same kind. Neither is a defect; both are things a reader
of a post-integration number needs told.

### 2.6 Rows 19–20 — `recover.py`'s two dict-literal reshapes

**Row 19 is owned by Task 8 (Ruling R3-PF-1). Task 9's Step-4 mention of point 19 is
verification only.** The union:

```python
param_prior_kw = None
if use_param_prior:
    from .losses.terms import _load_box_bounds
    param_prior_kw = dict(dratio_centre=…, dratio_spread=…, box=_load_box_bounds(bio_box_path))
term_kw = dict(split_hinges=…, hinge_k_min_frac=…, detach_xstar=…, compute_resid=…,
               param_prior_kw=param_prior_kw)
term_kw["obs_scale"]  = float(frame.mean())
term_kw["kstar_idx"]  = int(torch.searchsorted(...).clamp(1, len(kgrid) - 1))
```

`box_path` must be **gone**, not present alongside `box` — a `term_kw` carrying both is exactly
the "duplicate key silently keeps the last" failure, except worse: the two keys have different
names, so nothing collides and `param_prior` quietly re-reads YAML every step while a `box`
argument sits unused. Row 20's post-hoc assignment form is deliberate and must be preserved:
assignments after the literal cannot produce a duplicate key.

### 2.7 Row 24 — the liveness rewrite

Adopt as written, including `LIVENESS_SYNC_EVERY = 25` and D-PERF-7's **corrected** caveat
(the extra steps are not "harmless": Adam's momentum keeps being applied; what is true is only
that no *reported* number depends on it). The detail that merges badly is the guard on both
`total_loss_batched` call sites:

```python
active=alive if spec_cfg is not None else None
```

Not a bare `active=alive`. Passing a live mask on the non-spectral path would change nothing
computationally but would make the argument's "only skips the forward solve" contract false.

### 2.8 Row 26 — `dispersion_backend`: RULED (mechanics yes, default no)

**CONTROLLER RULING, recorded here so Task 8 does not re-escalate it:** adopt the
**mechanics** — the three-way `assert` and the `if dispersion_backend == "auto"` resolution
block — and **keep the default at `"eig"`** in all four places (`RNGRN.__init__`,
`BatchedRNGRN.from_seeds`, `ModelConfig`, `recover`). The **flip** is parked at T8 Step 3.
Task 8 implements the ruling; it does not re-open it.

The two binding documents that pointed opposite ways, for the record:

- `docs/PLAN_redesign.md` Global Constraints: *"A0 is untouchable: the baseline objective …
  must keep bit-identical behaviour after every task."*
- D-PERF-3 flips `ModelConfig.dispersion_backend` from `"eig"` to `"auto"`, which resolves to
  `"cubic"` at N=3 — i.e. **every A0 run's backend changes** — and D-PERF-3 itself states the
  runs are not bit-comparable.

Two findings from Task 7 that informed the ruling and should stay attached to it:

1. **R2's own Phase-I runs are unaffected either way.** Every
   `experiments/redesign_r2/*/*/config/frozen_config.yaml` records `dispersion_backend: cubic`,
   passed explicitly — which is what `auto` resolves to at N=3. The flip changes behaviour only
   for callers that **omit** the argument. So this was an A0-comparability question, not an
   R2-comparability question.
2. **The mechanics are separable from the default** — which is precisely what the ruling takes
   advantage of. The three-way assert and the resolution block land without a code fork; only
   the four default strings stay behind.

#### The SIG half — the actual line-level union

**Corrected.** An earlier draft of this section placed both sides' new blocks *before*
`self.N = int(N)`. That is wrong on both refs, and following it literally would move code that
neither side moved. The real layout, read from source:

| | `dbd46fe` (gpu-optim) | `db40995` (redesign) |
|---|---|---|
| `def __init__` | `:124-126` | `:160-164` (+`pin_xstar`, `param_boxes`) |
| `assert dispersion_backend in …` | `:129` — widened to `("eig","cubic","auto")` | `:167` — still `("eig","cubic")` |
| `kstar_obs` validation | `:131-133` | `:169-171` |
| `pin_xstar`/`param_boxes` validation | — | **`:172-209`** (7 raises) |
| `self.N` / `form` / `n_hill` / `init` | `:134-137` | `:210-213` |
| `self.pin_xstar` / `self.param_boxes` | — | **`:214-216`** |
| the backend comment block | **`:138-146`** — rewritten (6 new "auto" lines + the 3 shared "Rejected at CONSTRUCTION" lines) | `:217-222` (3 "eig"/"cubic" lines + the same 3 shared lines) |
| `if dispersion_backend == "auto": …` | **`:147-148`** | — |
| `if … "cubic" and int(N) != 3: raise` | `:149-152` | `:223-226` |
| `self.dispersion_backend = …` | `:153` | `:227` |

**The conflict site is the adjacency at `db40995:214-216` ↔ `dbd46fe:138-146`.** R2's two new
`self.pin_xstar` / `self.param_boxes` assignments sit *immediately above* the comment block
gpu-optim rewrote, with no unchanged line between them. Git sees one contiguous changed region
and will conflict there. Everything else in the table is either one-sided (R2's `:172-209` and
`:214-216`) or unchanged on both sides (`:223-226`, `:227`).

Both sides' new blocks are **after** `self.N = int(N)`, not before. The union, in the order the
merged file should read:

```
def __init__(…, kstar_obs, pin_xstar=None, param_boxes=None)   # SIG merge: R2's two params;
                                                               #   dispersion_backend default
                                                               #   stays "eig" per the ruling
assert form …
assert dispersion_backend in ("eig", "cubic", "auto")          # gpu-optim  (mechanics: adopted)
assert init …
kstar_obs validation                                           # BASE, unchanged
pin_xstar / param_boxes validation  (7 raises)                 # R2, one-sided
self.N / self.form / self.n_hill / self.init                   # BASE, unchanged — DO NOT MOVE
self.pin_xstar / self.param_boxes                              # R2  ─┐ CONFLICT SITE:
gpu-optim's rewritten backend comment block                    #      ─┘ adjacent, both changed
if dispersion_backend == "auto": …resolve…                     # gpu-optim  (mechanics: adopted)
if dispersion_backend == "cubic" and int(N) != 3: raise        # BASE, unchanged
self.dispersion_backend = dispersion_backend                   # BASE, unchanged
```

Resolution at the conflict site is "keep both sides", in that order — R2's two assignments,
then gpu-optim's comment block. It is one of the few places where "keep both" is right, because
the two hunks are genuinely independent additions that merely abut.

`RNGRN.__init__` and `BatchedRNGRN.from_seeds` each take **both** sides' new parameters
(`pin_xstar`, `param_boxes` from R2; the changed `dispersion_backend` default from gpu-optim) —
two signature merges in one file, so read both `def` lines after merging. `member(b)` already
re-passes `dispersion_backend=self.dispersion_backend`, which under `auto`-resolution is always
the *concrete* value; that composes correctly and needs no change.

### 2.9 Row 27 — a scoping correction, not a conflict

The Task 7 brief describes point 27 as changed by both lines independently. **Against this
merge's two refs, it is not.** `git show db40995:src/rngrn/eval/numerics.py` is byte-identical
to BASE at `integrate_bdf1_newton_krylov`; the independent change lives on `feature/lift-ladder`
(`413bc8a`, `091303b`), which `feature/redesign-model` does not contain.

D-PERF-9's warning still holds — the two stubs *are* textually different (verified by diff:
different docstrings, different raise messages, and the lift-ladder version restored the
original signature in a follow-up commit). **The merge conflict is real but deferred to the
lift-ladder merge, not this one.** Task 8 should transplant the gpu-optim version as a
straight one-sided hunk and note in its report that the conflict is owed forward.

### 2.10 Row 28 — the registry flip, and why NOTHING forces it

R2 declared the five spectral terms un-batchable because, at T8, they were. gpu-optim
implements `spec_shape_batched`, `spec_aniso_batched`, `spec_amp_mean_batched`,
`spec_amp_fluct_batched`, `real_moments_batched` and `spectral_terms_batched`. The union:

```python
for _key in S.SPECTRAL_TERM_KEYS:
    _register(_key, getattr(S, _key), getattr(S, f"{_key}_batched"), None,
              default_weight=0.0, calibration="UNCALIBRATED")
```

Verified: all five batched twins follow the `<key>_batched` naming, so the `getattr` is uniform.

#### The safety net this row originally named DOES NOT EXIST — corrected

An earlier draft claimed `test_every_registered_term_is_fully_classified` would fail until the
flip lands, "the completeness test working". **That is false, and believing it is the most
dangerous thing this ledger could have said.** Traced in source:

- `term_registry.py:103-104` registers the five keys **by hand**, with `batched_fn=None,
  refusal_reason=_SPECTRAL_REFUSAL` written literally. Transplanting `spectral.py` does not
  touch those two lines, so the XOR at `tests/test_term_registry.py:14` **still holds** and the
  test **still passes**.
- `batched_fn` has **zero runtime readers** — `grep -rn "batched_fn" src/ tests/` returns only
  the dataclass field, the `_register` helper, and that one assertion. Nothing dispatches on it.
- `refusal_reason` has exactly **two** runtime readers, both in `total.py`: `:206` (`resid`,
  which survives) and `:209` (spectral, which **row 8 deletes**).

So a Task 9 that lands rows 8 and 14–18 and forgets row 28 leaves the registry asserting that
five terms refuse a batched form that is sitting in the same package, **under a fully green
suite** — the exact CONTRACT failure this row is tagged for, with nothing to catch it.

#### The real forcing mechanism, and the mandatory verification

The only trace the flip leaves is that `_SPECTRAL_REFUSAL` becomes dead: row 8 removes
`total.py:209`, its last reader. **T9's verification is therefore:**

```
grep -n "_SPECTRAL_REFUSAL" src/rngrn/losses/term_registry.py     # must return ZERO after the flip
grep -n "batched_fn=None" src/rngrn/losses/term_registry.py       # only `resid` and `morphology`
```

And, because a grep is not a test, **add the runtime assertion — it is cheap and real.** Extend
`test_every_registered_term_is_fully_classified` with: for every term that declares a
`refusal_reason`, a batched implementation must not exist under the conventional name.

```python
import sys
mod = sys.modules[t.fn.__module__]
assert not hasattr(mod, t.fn.__name__ + "_batched"), (
    f"{key} declares refusal_reason but {t.fn.__name__}_batched exists")
```

Task 7 ran this probe against the current tree: it passes for all seven refusing terms today
(`morphology_consistency_batched`, `stationarity_residual_batched` and the five
`spec_*_batched` all absent), and it flips to a **failure for exactly the five spectral keys**
the moment `spectral.py` transplants. That is the completeness test the row originally, and
wrongly, assumed already existed.

Three further consequences to carry through in the same change:

- **Delete `_SPECTRAL_REFUSAL`.** With row 8 retiring `total.py`'s use of it, leaving the
  constant behind is a dangling half-truth.
- **Fix `term_registry.py`'s module docstring**, which currently says the five spectral terms
  "also refuse". That sentence becomes false the moment the flip lands, and a docstring that
  claims more than the code does is a §4 violation.
- Note the arity asymmetry, and that it is **pre-existing, not introduced**: the spectral
  `fn`/`batched_fn` are `(u_star, targets, cfg[, members])`, not the `(model, xstar, kgrid, …)`
  of every other entry. R2's registry already stored the serial spectral callables in that
  shape, so this is consistent with what is there.

Watch item folded in here (T10 Step 2 already names it): `kstar_si` is a **thirteenth** registry
key with default weight 0.0, and `test_registry_has_exactly_the_twelve_legacy_keys` is
tautological (see §2.15), so it enforces nothing. Confirm `LossConfig.weights`' key set still
covers what `strategy.combine` reads after the flip.

### 2.11 Row 29 — the import-cycle question, answered

Phase A put `from ..model import BatchedRNGRN` at `losses/total.py:18` — module level, and the
loss layer's first dependency on `model.py`. The concern is that R2's registry made `losses/`
import-order-sensitive: `terms.py` imports `.term_registry` at **line 842**, deliberately at the
bottom, because `term_registry` imports `terms`.

**Task 7 verified there is no cycle**, by trace and by execution:

```
model.py      → .registry (only)
registry.py   → typing (only)
term_registry → ..registry, .spectral, .terms
terms.py      → ..observables, .term_registry (line 842, bottom)
observables.py→ numpy (only)
```

and, run unsandboxed against this worktree's own `.venv`:

```
$ .venv/bin/python -c "import sys, rngrn.model; print([m for m in sys.modules if m.startswith('rngrn.losses')])"
[]
```

`rngrn.model` pulls in **zero** `rngrn.losses.*` modules, so `total.py → model.py` cannot close
a loop. Re-run that one-liner after the merge; it is the whole check.

### 2.12 Row 30 — `observables.py`, and why it does not break R2's numbers

Standing Ruling **R3-PF-2** assigns this transplant to Task 9 Step 1. D-OBS-1 changes
`raps_torch`'s binning from `floor(|k|/dk)` to `raps`'s own `np.digitize(...) - 1`, with a
measured effect of up to 22.5% on a bin and 30.2% on k* at the detector's operating point.
That sounds like a comparability break, and it is worth being precise that it is not one here:

- `raps` and `kstar_of` — the **numpy** functions — are byte-unchanged from BASE.
- `recover.py` computes `kstar_obs = obs.kstar_of(frame[0]…)`, i.e. the numpy path.
- `raps_torch` / `kstar_of_torch` / `_raps_torch_bins` are **new symbols**, wired only into
  `forward.relax_to_pattern_torch`'s saturation detector and `forward._kstar_of_torch_batched`.

So no R2 number moves through this hunk. What *does* move is any figure produced by the torch
relax detector — a surface R2 never exercised. Report it that way rather than as a blanket
non-comparability announcement.

### 2.13 Row 32 — the frozen-column contract meets three new `parts` keys

`history.TrainingHistory._names` freezes the scalar column set on the **first** recorded step
and raises if it ever changes. `_apply_spectral_batched` adds `spectral_skipped` (a `(B,)`
object array of reason **strings**), `spec_computed` (bool), and `spec_ignited` (float) to
`parts`. Since `_scalars_from` calls `float(v)` on everything it admits, a string column would
be a hard failure.

**It is admitted by none of the four routes.** `_scalars_from` takes exactly: `DIAG_KEYS`, any
key starting `L_`, `ss_converged`, and `weights_used`'s entries. `spectral_skipped` and
`spec_computed` match none. `spec_ignited` **is** in `DIAG_KEYS` — deliberately, with a comment
saying so — and gpu-optim writes it as `ignited.astype(float)`, which is the shape `scalar()`
expects. The column set is also stable across steps, because `_batched_restarts` builds the
`SpectralContext` **before** the loop, so the `L_spec_*` keys exist on every step (NaN-masked
when not computed) rather than appearing partway through.

Verified by reading `history.py:150-230` and `total.py::_apply_spectral_batched`. Re-verify at
T10 by running a batched spectral recovery with a `TrainingHistory` attached.

### 2.14 Row 34 — `tests/test_firewall.py` is a hard stop

R2 added `losses/term_registry.py` to `RECOVERY_SIDE` and `measure_frame_bias` /
`r2_ignition_run` to `FORBIDDEN`. gpu-optim did not touch the file, so there is no textual
conflict and Phase B needs no edit to it.

The open question is classification of the *transplanted* surface. `forward.py`,
`etdrk4_torch.py` and `losses/spectral.py` are already on `RECOVERY_SIDE`. Nothing new appears
under `losses/` or `eval/`, and Phase A added no `scripts/` module that opens `payload.h5` —
so on Task 7's reading, no edit is required. **That reading is the `firewall-auditor`'s to
confirm at T10 Step 4, not a Phase-B task's to act on.** Any Phase-B task that finds itself
wanting to edit this file must return first.

### 2.15 Row 35 — a pin that is no longer a pin

`test_default_weights_are_bit_identical_to_the_legacy_dict` asserts
`term_registry.default_weights() == dict(terms.DEFAULT_WEIGHTS)`. Since T8,
`terms.py:843` reads `DEFAULT_WEIGHTS = _term_registry.default_weights()`, so both sides of
the assertion derive from the same call. The test is **tautological** and cannot catch a
default flip — which is exactly the class of damage (`resid` at 0.3, the lost anchor weight)
that phase A suffered. `test_registry_has_exactly_the_twelve_legacy_keys` is tautological for
the same reason, and its name is now wrong besides (there are thirteen keys).

Recorded as a **deferred minor**, not a Phase-B work item. T8 Step 2 already says not to rely
on it and to add a non-tautological pin only if cheap; §2 of the house style says do not build
what was not asked for.

---

## 3. What Task 7 returned to the orchestrator rather than ruling

**Row 26 is now RULED and is no longer on this list** — the controller's ruling (mechanics
adopted, default stays `"eig"`, flip parked at T8 Step 3) is recorded in the row and in §2.8.
Task 8 implements it and must **not** re-escalate.

1. **Row 27 — a scoping correction.** Point 27 has no redesign-side counterpart in this merge;
   the second change lives on `feature/lift-ladder`. Not a ruling — a correction to Task 8's
   stated scope, and a conflict owed forward to the lift-ladder merge.
2. **Rows 9–10 — a conditional decision point.** If the `J=`/`idx=` hoist cannot be expressed
   through `LossTerm`'s existing call contract, changing that contract is a decision Task 8
   must return rather than take (the plan says so; §2.4 says what specifically to watch).
3. **Row 34 — hard stop acknowledged.** `tests/test_firewall.py` needs no Phase-B edit on
   Task 7's reading, but the finding is reported rather than acted on.

No row required a `docs/DECISIONS.md` entry from Task 7 itself: this file records a merge plan,
not a science decision. Rows 26 and 28 will need entries **when they land**, and that belongs
to the task that lands them.

---

## 4. Revision note — 2026-08-19, after review

This ledger was returned **Needs fixes** on first review. Four corrections were applied; they
are recorded here rather than folded in silently, because two of them reversed a claim.

| | what was wrong | what it is now |
|---|---|---|
| **I1** | Row 28 named `test_every_registered_term_is_fully_classified` as the mechanism that would force the flip. **It would not** — `term_registry.py:103-104` hardcodes the refusal, so the XOR still holds; `batched_fn` has no runtime readers; row 8 deletes `refusal_reason`'s only spectral consumer. The row named a safety net that does not exist | Row 28 and §2.10 rewritten: the real (weak) forcing mechanism stated, an explicit `grep` verification given, and a **cheap runtime assertion proposed and probe-tested** against the current tree |
| **I2** | §2.8's `model.py` ordering diagram placed both sides' new blocks **before** `self.N = int(N)`. Wrong on both refs, and it omitted the actual conflicting hunk | §2.8 redrawn from real line numbers on both refs, with a side-by-side table and the conflict site named: `db40995:214-216` (R2's `self.pin_xstar`/`self.param_boxes`) abutting `dbd46fe:138-146` (gpu-optim's rewritten comment block) |
| **I4** | Row 29's box was pre-ticked ☑, which contradicts the column's verified-**after**-integration meaning and invited T10 to skip it | Unticked. The pre-merge evidence is kept in the row text, labelled as pre-merge |
| **folded** | rows 1, 6, 9, 26 | row 1 names the `total.py:16` `LOSS_TERMS` import (and warns it can be dropped as newly-unused); row 6 restores "NaN-masking"; row 9 marks six→eight as a deliberate correction of §10; row 26 records the controller ruling so T8 does not re-escalate |

A fifth correction (**I3**) applies to Task 7's *report*, not to this file: the report's claim
that "17 of 26 points cannot conflict" was wrong; by the file-level test it is **12** (points
14–18, `spectral.py`, and 19–25, `recover.py`). §0a of this ledger was already correct and is
unchanged.
