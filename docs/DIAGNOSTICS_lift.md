# DIAGNOSTICS — the finite-μ dynamical lift, validation ladder V0–V4

Results of the V0–V4 lift-validation ladder (`docs/REDESIGN_rngrn.md` §5.3), run 2026-08-18
on branch `feature/lift-ladder` (every rung's `provenance.date` is that day). Written for a zero-context reader; every number names the
artefact behind it. The rung code is `src/rngrn/eval/ladder.py` (arithmetic) over
`src/rngrn/eval/lifted.py` / `lifted_torch.py` (the lift itself); the campaign driver is
`scripts/lift_ladder.py`; machine-readable results are in
`experiments/lift_ladder/{v0,v1,v2,v3,v4}/results/*.json` and `arrays/*.npz` (tracked).

**The JSONs are the authority.** Where this document and a run JSON disagree, the JSON wins
and this document gets fixed (CLAUDE.md §8). Every number below was read back out of the JSON
named beside it, not copied from a prior narrative; three claims in the Task 6 campaign
report did not survive that re-read and are corrected here (F-L4, F-L5, F-L13).

**What class of sentence this licenses.** The ladder validates the *instrument*, not a
recovery. It licenses "the lifted system's algebra, linear spectrum, 0-D trajectories and
spatial fields behave as the QSS limit requires, to these measured tolerances, on these
populations". It licenses no statement about recovering a target. D-LIFT-1 makes the §5.4
gate conditional on this ladder; §5.4's own licensing question is analysed at the end
(**§7**), and the verdict belongs to the orchestrating session, not to this document.

**THE GATE POINT IS μ_gate = 1e-3, NOT μ_central = 7.2e-4.** The owner set it on 2026-08-17
(`docs/DECISIONS.md` D-REDESIGN-5, on branch `docs/redesign-rngrn`), superseding the delegated
choice of the literature-central 7.2e-4; `docs/REDESIGN_rngrn.md` §5.2/§5.3/§5.4 and §8 item 6
are written in those terms there. **The copy of `REDESIGN_rngrn.md` on this branch is stale and
still says μ_central** — it predates that edit. Both values appear below because the campaign
used both: V3(a) and V2 cover μ_gate = 1e-3 directly, while V3(b)'s anchor was run at
μ_central = 7.2e-4 (F-L13). 7.2e-4 is labelled "μ_central" throughout and is *not* the gate
point.

**Two vocabularies, kept apart throughout.** V0/V1/V4 are *linear* verdicts — "Turing-unstable"
is a property of a Jacobian. V2/V3 are *dynamical* — "patterns" is a property of an integrated
field. Neither is offered as a proxy for the other anywhere below.

---

## 0. How to read the rungs

**Three populations, and why each rung uses the one it does** (`scripts/lift_ladder.py`
§populations):

| name | what it is | n | used by |
|---|---|---|---|
| `low_basal` | `ladder.draw_models(n, form, seed)` — `RNGRN(init="low_basal")` draws filtered to a Newton-tight steady state (‖f(x\*)‖ ≤ 1e-10). **0 of 40 such N=3 draws are strictly Turing** (re-measured on this branch, `v1.json::draw_models_turing_rate`; D-EVID-11 measured 0/398). Right for algebraic/linear/0-D claims, wrong for any claim about patterning. | 16–40 | V0, V1, V2 |
| `harvest` | the 23 strictly-Turing Stage-0 generator draws in `experiments/figures_report/stage0/arrays/part1_harvest_generator.npz` (`docs/BIO_VIABILITY.md` §1.2) — §5.3 V4's "generator draws already harvested" | 23 | V1, V3, V4 |
| `d5` | the 4 Turing-unstable D5 recovered models, seeds 1/3/5/6 of `experiments/diag_fft/d5/runs/m3_registry_20260812_092043_seed*` (`analysis.json::turing_flags`) — §5.3 V4's "the target's own gate-passing recovered models" | 4 | V1, V4 |

**Box size is set per model at p = 8 periods across the box**, `L = 8·2π/k*_qss`. These
systems' k\* spans 0.29 → 40.7, two orders of magnitude; a fixed L would put some of them
below one wavelength per box and no lifted-vs-QSS comparison survives that. p = 8 is
`turing_labyrinth/sample_0000`'s own geometry, at which SPEC §9.1's binding one-bin tolerance
2π/L is exactly 12.5 % of k\* — so every V3 k\* number is judged at the same relative
precision as the target's.

**Inherited bars, not renegotiated here.** The one-radial-bin k\* tolerance `2π/L`
(`docs/SPEC_fourier_training.md` §9.1, D-FFT-3); V2's sup-norm bar 0.1 × `pattern_floor`
(= 1e-4 on these draws); V0's two-constant floor bound (F, C — controller ruling, ledger
2026-08-17, frozen in `tests/test_lift_ladder.py`). Nothing in this document loosens any of
them.

**THE MORPHOLOGY-AGREEMENT ESTIMATOR IS `observables.classify`, NOT the reference-bank
classifier.** `eval/ladder.py` is RECOVERY_SIDE in `tests/test_firewall.py` and therefore may
not import `rngrn.scoring` at all, so `morphology_agree` (V3) is the coarse four-way
`observables.classify` label on channel 0, compared lift-vs-QSS. `validate.morphology_match`
— the number every prior recovery run reported — uses
`scoring/morphology.py::classify_morphology` against a labelled reference bank. **The two
numbers are not comparable and must never be quoted against each other.** Both fields
compared in V3 are the model's own, so no ground truth is involved either way; the
non-comparability is an estimator difference, not a firewall problem.

---

## V0 — algebraic invariants                            [status: **PASS as a measurement**; the floor is now measured, not asserted]

`ladder.v0_invariants`, 20 Newton-tight draws per form, μ ∈ {1e-6, 1e-4, 1e-2, 1, 1e2}, seed
101 → `experiments/lift_ladder/v0/results/v0.json` (15.1 s, `provenance.git_commit`
`b45f256`).

The claim under test: the lifted fixed point z\* = (x\*, G_qss(x\*)) is a fixed point at
**every** μ — i.e. the lift moves *stability*, not the steady state — and `rescale_mu`
(scaling the gate rows by 1/μ) is exact rather than an approximation.

| | competitive | nc1 |
|---|---|---|
| `max_rescale_mu_err` (relative, vs a fresh autodiff Jacobian) | 2.328e-16 | 2.328e-16 |
| `max_gate_qss_err` (absolute; gates-at-QSS reproduce `model.reaction`) | 1.776e-15 | 0.0 |
| `max_fixed_point_residual` (worst over 20 models × 5 μ) | 2.564e-07 | 1.332e-09 |
| `residual_floor_F` | 6.267e-11 | 2.796e-11 |
| `amplification_C` | 1154.5 | 12.59 |
| `exclusion_rate` (steady_state-converged but not Newton-tight) | 0.0 (0/20) | 0.0476 (1/21) |

`residual_floor_F` and `amplification_C` reproduce the constants frozen in
`tests/test_lift_ladder.py` (`F_FORM`, `C_FORM`) exactly, at the same sample size.

**The 2.56e-7 top-line residual is above §5.3 V0's inherited 1e-7 figure, and that is the
already-ruled behaviour, not a regression.** §5.3 V0 says V0 *measures* the floor rather than
asserting one; the controller ruling (ledger 2026-08-17, round 2) replaced the single 1e-7
bar with a two-mechanism, μ-aware bound that the suite now enforces per form:

```
residual(mu) <= max(1e-9, 10*F_form, 10*C_form*eps/mu)          eps = 2.220446049250313e-16
```

- **F (the flat floor)** = max residual over μ ≥ 1e-2 — the x-block round-off, i.e. how far
  `model.reaction(x*)` itself sits from zero. μ-independent by construction: the gate rows do
  not enter this component.
- **C/μ (the amplified term)** = max of `residual(μ)·μ/eps` over μ ≤ 1e-4 — `eval/lifted.py`'s
  two *independently computed* reconstructions of the gate normalisation, divided by μ. See
  **F-L2**.

**Finding F-L1 — C is 92× larger on `competitive` than on `nc1`, and the ratio identifies the
mechanism.** C = 1154.5 vs 12.59, i.e. an absolute gate-block discrepancy of 2.56e-13 vs
2.80e-15. The competitive form's free-promoter fraction is a *sum* over 2N gates
(`free = 1 − ΣGA − ΣGR` in `lifted_rhs_torch`, against the closed form
`free = 1/(1 + Σ(K_A+K_R)x^n)` in `gates_qss`); the nc1 form's is per-entry
(`1 − GA` vs `1/(1+u)`). Cancellation in the summed form is the larger by exactly the margin
the two C's show. This is a *consistency* measurement between two algebraically identical
expressions, not an error in either — see **F-L2** for why single-sourcing them is the wrong
fix.

**Finding F-L2 — the two reconstructions are NOT redundant and must not be merged.**
`gates_qss`'s closed form is a function of **x**; it is the QSS *solution*. `lifted_rhs_torch`'s
`free = 1 − ΣGA − ΣGR` is a function of the **state z**, and away from the fixed point it is
the only correct one — the gates are dynamical variables there, and substituting the
closed form would silently replace the lifted ODE by a different ODE that re-imposes QSS at
every evaluation. They coincide *only* at the QSS gate state, which is exactly why their
difference is a usable diagnostic. Recommendation and evidence: §7.3.

**Finding F-L3 — `steady_state`'s relaxation fallback admits states three orders looser than
its Newton path, and V0 now excludes them.** `losses/terms.py::steady_state` returns
`converged=True` at ‖f(x)‖ < 1e-4 on its relaxation fallback, versus 1e-10 on
`_damped_newton`. A fallback-path draw carries its own residual — constant across every μ —
straight into the invariant (measured on nc1 seed 113: 1.268e-6 at every μ). `draw_models`
now filters on ‖f(x\*)‖ ≤ 1e-10 and *reports the rate it excluded at* (`exclusion_rate`
above: 0/20 competitive, 1/21 nc1). Any later code that assumes a `converged=True` steady
state is Newton-tight is wrong.

---

## V1 — linear: eigenvalue continuation, all N + 2N² branches   [status: **PASS**, with a measured exception on 2 of 23 harvest systems]

`ladder.v1_continuation`, μ ∈ {1e-7, 1e-6, 1e-5}, 4000-point log k-grid →
`experiments/lift_ladder/v1/results/v1.json` + `arrays/v1.npz` (64.0 s, re-run at
`provenance.git_commit` `0c4a19c`).

Method, in the order it matters: one autodiff lifted Jacobian at μ = 1, then `rescale_mu`
(exact) for every μ — no μ costs a second autodiff pass. At the *smallest* μ the N slow
branches are identified by matching QSS eigenvectors against the x-block of the lifted
eigenvectors; that labelling is propagated to larger μ by full 21-branch overlap assignment
between adjacent μ, **never** by sorting real parts (Re-sorting swaps identities through
exactly the crossings the re-entrant band is made of; measured to differ from the tracked
slow set at 44 % of k on one competitive draw). The per-(k, branch) order is the
least-squares slope of log|λ_slow − λ_qss| against log μ.

**Standard population** — 16 `low_basal` draws (both forms × seeds 7/101/202/303 × 2):

| quantity | measured |
|---|---|
| slow-branch order (separation-masked mean) | 0.9405 – 1.0706 (mean 1.031) |
| slow-branch order (all k, unmasked) | 0.9777 – 1.1485 (mean 1.067) |
| `kstar_grid_offset` | **0 on all 16** |
| `min_fast_mu_product` = min\|Re λ_fast\|·μ (the −(1+u)/μ claim) | 0.9756 – 1.0010 |
| fast branches stable (Re < 0 at every k, every μ) | yes, 16/16 |
| `frac_k_separated` (fraction of k where μ·max(D)·k² < 1) | **0.442 – 1.000 (mean 0.616)** |
| `max_slow_err` (unmasked max over all k), μ = 1e-7 / 1e-6 / 1e-5 | 1.05e-2 / 1.40e-1 / 1.19e4 |

**The k\* check is vacuous on this population and says so.** 15 of the 16 have
`kstar_at_grid_floor = True` — σ(k) is monotone decreasing, so both the QSS and the lifted
k\* pin to the grid floor and `kstar_grid_offset == 0` holds by construction rather than by
branch tracking. Only `low_basal/nc1/202/0` has an interior QSS k\* (0.0385), and the offset
is 0 there too. This is why the rung needs a Turing-positive population as well (controller
ruling at T3).

**Turing-positive populations with an INTERIOR k\*** — the 23 harvest systems and the 4 D5
recovered models (`v1.json::turing_positive`, tagged `population`, never pooled):

| | n | slow-branch order (masked) | order (all k) | `kstar_grid_offset` ≠ 0 | `min_qss_overlap` | `frac_k_separated` | `max_mu_D_k2` | `min_fast_mu_product` | fast stable |
|---|---|---|---|---|---|---|---|---|---|
| harvest | 23 | 0.811 – 1.020 (mean 0.975) | identical to masked | 2/23 (offset = 1) | 0.223 – 0.961 | **1.0 on all 23** | 0.025 on all 23 | 1.000 – 1.008 on 21; **5.4e-6 / 6.5e-6 on 2** (F-L5) | 23/23 |
| d5 | 4 | 1.044 – 1.059 | 1.484 – 2.019 | 0/4 | 0.072 – 0.516 | **0.105 – 0.223** | 20.2 – 90.4 | 0.349 – 0.998 | 4/4 |

All 23 harvest systems and all 4 D5 models have `kstar_at_grid_floor = False` (harvest
k\*_qss 1.451 – 40.710; D5 k\*_qss 0.28857 on all four), so `kstar_grid_offset` is a real
measurement on 27 systems rather than two grid floors agreeing.

**The two Turing-positive populations sit in OPPOSITE separation regimes, and that is the
useful part.** D5's diffusion is large enough that μ·max(D)·k² reaches 20–90 over the grid, so
the two-timescale separation the order-1 claim rests on holds only on 10–22 % of k, and the
masked and unmasked orders differ by a factor of ~1.8 (1.05 vs 1.83). The harvest systems'
product tops out at 0.025 on every one of them, so separation holds over the **entire**
k-grid and the masked and unmasked orders are identical. The order estimate on the larger
population is therefore not resting on a minority of k the way D5's is. The same split shows
in `min_fast_mu_product`: ≈ 1 on harvest (where the fast branches really are the −(1+u)/μ gate
branches) but down to 0.349 on D5, whose diffusive slow branches climb toward the gate rates
at large k. The −(1+u)/μ claim is a claim only where the timescales separate, and the two
populations bracket that.

**Finding F-L4 — the low_basal `frac_k_separated` range is 0.442–1.000, not "~1.0".** The
Task 6 report stated ~1.0 for this population; `v1.json::low_basal[*].frac_k_separated` gives
a mean of 0.616 with a minimum of 0.442. Corrected here. The consequence is small — the
masked and unmasked orders on that population differ by <0.1 (1.031 vs 1.067), so the
aggregation choice does not move the verdict — but the number was wrong and the argument
"harvest is the clean-separation population, D5 is not" survives only because harvest is at
1.0, not because low_basal was.

**Finding F-L5 — branch identification BREAKS on 2 of the 23 harvest systems, and the V1
tolerances there are meaningless. This is new; the Task 6 report did not surface it.** On
`harvest/competitive__mobile3/0` and `harvest/competitive__immobile/1`:

| | mobile3/0 | immobile/1 | typical harvest row |
|---|---|---|---|
| `min_fast_mu_product` = min\|Re λ_fast\|·μ | **5.440e-6** | **6.546e-6** | 1.000 – 1.008 |
| `max_fast_re` | −5.4398 | −6.5455 | ≈ −1.0e5 |
| `max_slow_err` at μ = 1e-6 | **999 994.56** | **999 993.45** | ≤ 1.5e-3 |
| `max_slow_err` at μ = 1e-5 | 99 994.56 | 99 993.45 | ≤ 1.5e-2 |
| `min_qss_overlap` | 0.456 | 0.456 | 0.223 – 0.961 (mean 0.617) |
| `slow_order_max` | 7.00 | 6.59 | ≈ 1.3 – 4.3 |

Read together these are the signature of a **slow/fast label swap at some k**: the slow set
contains an eigenvalue ≈ 1/μ away from its QSS partner (999 994.56 = 1e6 − 5.44 exactly, and
99 994.56 = 1e5 − 5.44 — the same branch at both μ), while the fast set contains a branch
whose rate is O(1) rather than O(1/μ). Every gate eigenvalue of the lifted Jacobian is ≥ 1/μ
in magnitude by construction (the competitive gate block has eigenvalues 1/μ and (1+S)/μ), so
a "fast" branch at |Re λ| = 5.44 is not a gate branch at all.

What this does and does not damage:

- On those two rows `max_slow_err`, `min_fast_mu_product` and the order estimate are reading a
  mis-assigned pair, so **none of them is a statement about the lift there**. `slow_branch_order`
  still comes out at 1.011 / 1.006 — the swap affects a minority of k and the fitted mean
  survives it, which is precisely why the aggregate looks clean and the per-row diagnostics do
  not.
- `frac_k_separated = 1.0` on both rows, so **the separation mask does not detect this failure
  mode.** Its criterion (μ·max(D)·k² < 1) tests whether the *slow* relaxation is slow; it says
  nothing about whether a *gate* branch has come down to meet it.
- It does **not** touch the §5.4 L1 verdict. `turing_verdict_lifted` is a max of Re λ over all
  21 branches at each k — tracking-free by construction. A branch-label swap changes which
  branch is called what; it cannot change the maximum.

Recorded as a defect-adjacent diagnostic, not fixed here (this task is docs-only). The
durable fix would be a second separation test on the *gate* side — e.g. flag any row where
`min_fast_mu_product` < 0.5 — so the two rows self-identify rather than being found by
reading a table.

---

## V2 — temporal, 0-D                                     [status: **PASS**; μ ≥ 1e-1 not measured]

`ladder.v2_temporal` / `v2_qss_limit` / `v2_rhs_mirror_error` on
`draw_models(n=1, form, seed=11)[0]`, dts = [μ/2, μ/4, μ/8], horizon T = 5.0 →
`experiments/lift_ladder/v2/results/v2.json` (313.8 s, `provenance.git_commit` `b45f256`).

Two claims: **(i)** the 21-dim lifted ODE converges to the 3-dim QSS ODE as μ → 0; **(ii)** the
Strang stepper matches a trusted stiff reference (scipy Radau, rtol 1e-10 / atol 1e-12) at
order ≈ 2 where dt ≲ μ, with sup-norm error at the rollout horizon ≤ 0.1 × `pattern_floor`
= **1e-4**.

**(ii) — Strang order and horizon error.** `dt_over_mu_max` = 0.5 at every point, so the order
claim's own dt ≲ μ scope is respected everywhere it is quoted.

| form | μ | Strang order | sup-err at horizon | `radau_ok` | reference margin |
|---|---|---|---|---|---|
| competitive | 1e-5 | 1.9939 | 2.813e-10 | yes | 9.5e3 |
| competitive | 7.2e-4 (μ_central) | 1.9948 | 2.773e-07 | yes | 1.9e4 |
| competitive | 1e-3 (μ_gate) | 1.9946 | 5.349e-07 | yes | 4.0e4 |
| competitive | 1e-2 | 1.9940 | 4.364e-05 | yes | 2.0e6 |
| nc1 | 1e-5 | 1.9953 | 1.296e-13 | **no** | 6.9 |
| nc1 | 7.2e-4 | 1.9952 | 6.768e-12 | yes | 21.0 |
| nc1 | 1e-3 | 1.9948 | 9.373e-12 | yes | 27.7 |
| nc1 | 1e-2 | 1.9954 | 5.511e-12 | yes | 148.9 |

Every sup-norm error at the horizon is below the 1e-4 bar, by 3 to 9 orders. The single
`radau_ok = False` is nc1 at μ = 1e-5, where the Strang errors are ~1e-13 — at float64
round-off, so the reference's own self-error (1.91e-14) cannot sit the required order below
them. **That is the guard working — a reference too coarse to judge its subject fails loudly
— not a failed claim**; the order there is still 1.9953. Accepted as informational at T4
(review Approved, no DECISIONS entry owed).

**(i) — the QSS limit.** `qss_order` = 1.0000 (competitive) / 0.9986 (nc1) over
μ ∈ {1e-6, 1e-5, 1e-4}; `qss_gap_rel` 6.46e-7 → 6.46e-5 (competitive) and 8.78e-10 → 8.72e-8
(nc1), i.e. first order in μ on both forms with the gap tracking μ across two decades.

**`v2_rhs_mirror_error` = 0.0 on both forms** — the numpy Radau RHS is *bit-identical* to
`lifted_rhs_torch`, so the reference and the subject cannot drift apart through two
independent transcriptions of the same equations.

**Finding F-L6 — the realised μ set is {1e-5, 7.2e-4, 1e-3, 1e-2}; §5.3 V2's 1e-1 and 1 were
never run.** The T4 brief's parametrisation narrowed the set (and D-REDESIGN-5 added
μ_gate = 1e-3). **The order-2 claim is therefore not demonstrated at μ ≥ 1e-1.** Those are the
points furthest from the QSS limit and two decades above where the gate operates
(μ_gate = 1e-3), so nothing downstream currently rests on them — but the ladder does not cover
them and this document will not imply that it does.

---

## V3 — spatial                                           [status: **PASS within a stated claim scope**; the gate's own operating point is NOT among the configurations run — see §7.2]

`ladder.v3_spatial` over the 23 harvest systems, μ ∈ {1e-3, 1e-4, 1e-5, 1e-6} (§5.3's
{1e-4, 1e-5, 1e-6} plus μ_gate = 1e-3, D-REDESIGN-5), torch/CUDA backend →
`experiments/lift_ladder/v3/results/v3.json` + `arrays/v3.npz` (24 017 s = 6 h 40 m,
`provenance.git_commit` `fa7380f`).

**The comparison is structural, not arranged.** `lifted.step_policy` *is* `eval/rollout.py::
simulate`'s own growth-rate-aware policy evaluated on the QSS Jacobian, and both sides draw
the initial field from `np.random.default_rng(seed).standard_normal((N,n,n))` around the same
x\*. The two runs therefore take the same dt, the same step count, and start from the
bit-identical field; **the only difference between them is how the gates are handled**. That
is what makes the field difference attributable to the lift rather than to the harness.

**`claim_scope_ok` and what it protects.** It is False when either side failed to pattern.
k\* and morphology read off a decayed field mean nothing (this project has already recorded
one false `morphology_match` from exactly that, `plotdata.py`), so those two flags are only
claims inside the scope. `patterned_agree` stays meaningful outside it and is the only flag
that does. `morphology_claimable` = `claim_scope_ok and n ≥ 512`, per §5.3's 512²-only licence
on the morphology class.

### V3(a) — 23 harvest systems at 128² (`d['rows']`, `d['summary']`)

| | all 23 | claim-scope-ok subset (n = 10) |
|---|---|---|
| `patterned_agree` | **23/23 = 100 %** | — |
| `morphology_agree` (`observables.classify`, see §0) | 19/23 = 82.6 % | **10/10 = 100 %** |
| `kstar_within_one_bin` (\|k\*_lift − k\*_qss\| ≤ 2π/L) | 18/23 = 78.3 % | **10/10 = 100 %** |
| `l2_monotone` | 18/23 = 78.3 % | — |
| `l2_diff_by_mu`, median over rows | 2.7569e-2, **identical at all four μ** | |
| `l2_diff_by_mu`, max over rows | 0.17625, **identical at all four μ** | |

### V3(a) at 512² — the morphology-claim block, 8 systems (`d['full_rows']`, `d['summary_full']`)

Cost-blind subset: the first 2 of each of the 4 (form, D-scheme) cells.

| | all 8 | `morphology_claimable` subset (n = 4) |
|---|---|---|
| `patterned_agree` | **8/8 = 100 %** | — |
| `morphology_agree` | 7/8 = 87.5 % | **4/4 = 100 %** |
| `kstar_within_one_bin` | 7/8 = 87.5 % | **4/4 = 100 %** |
| `l2_monotone` | 6/8 = 75 % | — |
| `l2_diff_by_mu`, median / max over rows | 4.1344e-2 / 0.18789 | |

The four systems that license the morphology claim — `competitive__mobile3/1`,
`nc1__mobile3/{0,1}`, `nc1__immobile/0` — agree on morphology class **and** on one-bin k\*.

**Finding F-L7 — CLAIM SCOPE: 10/23 at 128², 4/8 at 512². The morphology claim rests on
n = 4.** 13 of the 23 QSS *controls* did not cross `pattern_floor` within
`rollout.simulate`'s default growth-rate-aware horizon. **This is a horizon artifact, not a
resolution artifact and not evidence against these systems' Turing instability**: the same 3
systems that were sub-threshold at 128² were still sub-threshold at 512², so refining the
grid did not change the verdict, and all 23 are Stage-0-harvested strictly-Turing draws
(V4 re-confirms every one of them under the strict `qss_verdict`). Nonlinear saturation had
simply not completed within the default horizon at this box and grid. Of the 13:

- **4 are both-sides-decayed to round-off** — `competitive__mobile3/2`, `nc1__mobile3/2`,
  `nc1__immobile/1`, `nc1__immobile/5`, with l2 ∈ [1.28e-15, 1.00e-14]. The two fields are
  numerically equal; `morphology_agree` and `kstar_within_one_bin` are False on exactly these
  4 and on no others, i.e. they are reading noise, exactly as `v3_spatial`'s docstring says
  they would.
- **9 sit below the `patterned` bar but well above round-off** (l2 ∈ [4.63e-3, 1.43e-1] —
  real field structure) and still agree on morphology (9/9) and on k\* (8/9; the exception is
  `nc1__immobile/4`, whose QSS k\* is NaN).

**Finding F-L8 — `l2_monotone` 18/23 is ORDERING NOISE AT THE O(dt) FLOOR, and the failing
rows are quantified rather than left beside "identical to all printed digits".** Both
statements in that sentence are true simultaneously and the tension is only apparent. In
every failing row the value at μ = 1e-3 is the *smallest* of the four and the other three are
bit-identical to each other — the "failure" is a last-digit tick **upward** of 1e-12-scale as
μ drops one decade, then nothing:

| row (128²) | l2 at μ = 1e-3 → 1e-4 | absolute spread over the 4 μ | spread / value |
|---|---|---|---|
| `competitive__mobile3/0` | 0.0062575899259888 → …292542 | 3.27e-12 | 5.22e-10 |
| `competitive__immobile/1` | 0.0299040474156123 → …180721 | 2.46e-12 | 8.23e-11 |
| `competitive__immobile/2` | 0.0238830910352416 → …359792 | 7.38e-13 | 3.09e-11 |
| `competitive__mobile3/2` (decayed) | 1.28443e-15 → 1.28499e-15 | 5.63e-19 | 4.39e-4 |
| `nc1__immobile/5` (decayed) | 9.99815e-15 → 9.99975e-15 | 1.60e-18 | 1.60e-4 |

Against row medians of 2.757e-2 (128²) and 4.134e-2 (512²), the absolute spreads are 9 to 16
orders of magnitude smaller than the quantity being ordered. The same holds for the 2 failing
rows at 512² (`competitive__mobile3/0`, spread 8.10e-13; `competitive__immobile/1`, 8.46e-12).
Two of the five are the both-sides-decayed pairs of F-L7, where the relative tick is 1e-4 of a
1e-15 number. **`l2_monotone = False` on these rows is not a μ-dependence reversal; it is the
sign bit of the 12th–16th significant digit at a floor that does not depend on μ.**

**Finding F-L9 — AT THE QSS dt, V3(a) CANNOT SEE μ-DEPENDENCE AT ALL; the μ claim rests
elsewhere.** The lifted-vs-QSS field difference has two components: an **O(μ)** gate-tracking
lag — the thing the lift is *about*, which vanishes as μ → 0 — and an **O(dt)** term that does
not, because `simulate_lifted` freezes the production over the ETDRK4 step (Strang) while
`rollout.simulate` re-evaluates the QSS reaction at every ETDRK4 stage. That is a difference
of *scheme at fixed dt*, not of physics. With dt set by the QSS growth-rate policy,
`dt/μ` ranges from **26.4 to 2.528e5** across this population (computed row-by-row from
`d['rows'][i]['dt']`, which spans 0.0264 – 0.2528), so the exact gate substep is already at
its μ → 0 limit at *every* μ in the V3 set and the whole residual difference is the scheme
term. Hence four identical numbers per row. The μ-dependence is established instead by:

- **the 32² suite tests** (`tests/test_lift_ladder.py::
  test_v3_l2_difference_floors_at_an_O_dt_scheme_difference`), where the horizon is short
  enough for dt ≲ μ to be affordable. Measured, competitive seed 23, 32², L = 20, T = 0.2:
  at μ = 1e-3 the difference is dt-*independent* (5.01e-7, 5.36e-7, 5.45e-7 at
  dt = 5e-4, 2.5e-4, 1.25e-4 — the O(μ) term dominating), while at μ = 1e-6 it **halves with
  dt** (2.04e-7, 9.70e-8, 4.70e-8 — first order, the O(dt) term all that is left). At fixed
  dt = 5e-4 the μ curve falls 27× end to end (5.46e-6 → 2.04e-7) and then flattens.
- **V3(b)'s dt-halving pair**, below.

The tests assert the two claims that are true as measured — end-to-end fall over the campaign
μ set, and step-by-step non-increase over μ ∈ {1e-2, 1e-3, 1e-4} where the lift term dominates
— and the scoping is documented at the site. This mirrors V2's own precedent, whose order-2
claim is scoped to dt ≲ μ with `dt_over_mu_max` reporting whether the scope was respected.

### V3(b) — the anchor: dt-halving pair at 128², plus 512² at the QSS dt — **run at μ_central = 7.2e-4, not at μ_gate**

**Finding F-L13 — V3(b) was run at μ_central = 7.2e-4; §5.3 V3(b) as it now reads asks for
μ_gate = 1e-3.** `scripts/lift_ladder.py::run_v3` anchors on
`lifted.MU_BIO_CENTRAL`, and the anchor rows in `v3.json` are keyed `'0.00072'`. The driver was
written against the pre-D-REDESIGN-5 text, which said μ_central. The gap is a factor 1.39 in μ
(and 3.6e-4 vs 5.0e-4 in the policy dt). It is not a large factor and the qualitative
conclusion — O(μ)-dominated at dt = μ/2 — is not delicate at that scale, but **no dt-halving
pair exists at the gate point itself**, and this document will not pretend otherwise. V3(a)
*does* cover μ_gate = 1e-3 at both grids, at the QSS dt (which is the regime that cannot see μ
— F-L9). Reported, not fixed: re-running the anchor at 1e-3 is a rerun, not a code change.

dt_policy = μ_central/2 = 3.6e-4 on all 8 anchor systems (the μ constraint dominates
`min(0.2/jac_rate, μ/2)` throughout this subset). **The horizon here is a step budget, not a
growth time, and that is forced by the dt policy itself**: 3.6e-4 is up to 702× finer than the
QSS rollout's dt, while the pattern-formation horizon 40/|σ_max| is 10²–10³ time units, so the
attractor horizon at that dt is ~10⁶–10⁷ steps — past `simulate_lifted`'s own 200 000
`max_steps`, and a `step_budget` run is TRUNCATED, i.e. not a statement about the attractor.
V3(b) is therefore scoped to what its dt can pay for: a common short horizon T = 4000·dt =
1.44 time units, run at dt and dt/2. **It is a dt-convergence measurement on the transient and
is reported as one.**

| system | l2(dt) → l2(dt/2) | ratio |
|---|---|---|
| `competitive__mobile3/0` | 5.5709e-07 → 5.6148e-07 | 0.9922 |
| `competitive__mobile3/1` | 2.6281e-06 → 2.6432e-06 | 0.9943 |
| `competitive__immobile/0` | 1.0681e-06 → 1.0742e-06 | 0.9943 |
| `competitive__immobile/1` | 1.8765e-06 → 1.8933e-06 | 0.9912 |
| `nc1__mobile3/0` | 1.4585e-06 → 1.4835e-06 | 0.9832 |
| `nc1__mobile3/1` | 7.2936e-06 → 7.3615e-06 | 0.9908 |
| `nc1__immobile/0` | 7.2936e-06 → 7.3138e-06 | 0.9972 |
| `nc1__immobile/1` | 6.8679e-07 → 6.8970e-07 | 0.9958 |

**Median ratio 0.99324 (`anchor_summary.dt_halving_ratio_median`) — dt-INDEPENDENT, not the
≈ 0.5 a first-order Strang-scheme floor would give.** Read against F-L9's mechanism: halving
dt barely moves the difference, so at μ = 7.2e-4 with dt = μ/2 the **O(μ) gate-tracking-lag
term dominates the O(dt) scheme term** — the same regime the 32² sweep found at μ = 1e-3, and
the opposite regime from V3(a)'s campaign numbers, where dt is set by the much coarser QSS
policy. This is the measurement that says the lift is doing something the QSS run is not, at a
μ within 40 % of the gate point (F-L13).

**`full_grid_patterned_agree` = 8/8 = 100 %** — the separate 512²-at-QSS-dt run in the same
block agrees on `patterned` for every anchor system. That is the attractor-scale statement at
μ = 7.2e-4; the T = 4000·dt pair above is not one.

**Finding F-L10 — `l2_diff_dev_by_mu` returns 9.4e+288 on one row, and that is a
division-by-almost-zero, not a value. FIXED** (`fix/lift-l10`, task 19). `harvest/nc1__immobile/1`'s
512² anchor run has a QSS control that decayed to exactly zero amplitude, so the
deviation-from-channel-means denominator of `_rel_l2_dev` was at its 1e-300 guard. The raw
`l2_diff_by_mu` on the same row is 1.75e-15 (correct — both fields are zero). CLAUDE.md §8's rule
is that a value which arrived this way is a defect, not a datum: `_rel_l2_dev`
(`src/rngrn/eval/ladder.py`) now returns `NaN` — documented as deliberate at the site — whenever
the control's deviation norm is below the same 1e-300 floor, instead of dividing by it.
`tests/test_lift_ladder.py::test_rel_l2_dev_nans_on_a_zero_amplitude_control_instead_of_9e288`
pins the behaviour on a synthetic uniform control and asserts `_rel_l2` (`l2_diff_by_mu`'s
function) is unaffected. **Do not read `l2_diff_dev_by_mu` on any row whose control did not
pattern.**

The three affected loci in the already-committed `experiments/lift_ladder/v3/results/v3.json`
(all `harvest/nc1__immobile/1`) — `rows[18].l2_diff_dev_by_mu` (4.5266524e+288, the 128²
population), `full_rows[7].l2_diff_dev_by_mu` and `anchor[7].full_grid.l2_diff_dev_by_mu` (both
9.4066017e+288, the 512² population/anchor) — **predate the fix and were not re-emitted**: doing
so needs a fresh 512² `v3_spatial` run (`guarded_run.sh`-gated GPU compute), which is not cheap,
and the raw fields behind those rows were never persisted to `arrays/v3.npz` (only the scalar
summaries were), so the pre-fix numbers cannot be recomputed from the committed artifacts either
— only a rerun produces the corrected value. Any reader of that file must skip
`l2_diff_dev_by_mu` on those three loci (or treat it as `NaN` by construction) and read
`l2_diff_by_mu` instead, which was and remains correct (1.75e-15 / 3.37e-15 — both fields
zero-amplitude in that row's grid).

### CPU/CUDA equivalence at campaign scale

`experiments/lift_ladder/v3/cpu_gpu_check/results/v3.json` — the `backend='numpy'` leg for
the first two harvest systems at matched seed/L/dt/horizon against the production
`backend='torch'` (CUDA) rows:

| system | l2 (CPU) | l2 (CUDA) | rel diff |
|---|---|---|---|
| `mobile3/0` (sub-threshold) | 6.257589930e-03 | 6.257589929e-03 | ~1e-11 |
| `mobile3/1` (**patterns**, `claim_scope_ok`) | 1.035647708e-01 | 1.035647708e-01 | ~9e-14 |

`kstar_qss` is bit-identical between the two legs (the QSS control is numpy either way);
`morphology_agree` and `kstar_within_one_bin` are True on both. The V3(b) anchor pair for
`mobile3/0` matches to ~1e-9 relative (CPU 5.570865622e-07 vs CUDA 5.570865617e-07; dt-halving
ratio 0.99217291042 vs 0.99217291127). This extends
`tests/test_lifted_torch.py`'s 32²-fixture equivalence to campaign scale on a system that
actually patterns. **Caveat: it is 2 systems at 128², not a backend-equivalence sweep of the
campaign.**

---

## V4 — re-entrant-band survey                            [status: **MEASUREMENT, not a gate** — §5.3 V4 sets no pass threshold by design]

`ladder.v4_survey` + `ladder.qss_verdict` over 27 Turing-positive systems (23 harvest + the 4
D5), band [1.1e-5, 9.2e-3] = [MU_BIO_LO, MU_BIO_HI], 9 log-spaced μ including both endpoints,
200-draw lognormal parameter clouds at σ_log = 0.10 →
`experiments/lift_ladder/v4/results/v4.json` + `arrays/v4.npz` (1050.3 s,
`provenance.git_commit` `b45f256`).

**The control is the strict pair, on both sides.** `qss_verdict` applies the *same*
`turing_verdict_lifted` conditions (max Re eig(J) < 0 **and** max_{k>0} σ(k) > tol) to the
reduced N×N Jacobian — deliberately **not** `analysis.turing_ok`'s trace test, which Stage 0
measured overcounting by 64×. Conditioning on a looser denominator would inflate it and
deflate the ratio.

**THE DECISION-RELEVANT NUMBER: P(lifted-Turing across [1.1e-5, 9.2e-3] | QSS-Turing) =
27/27 = 1.00.** The denominator is `n_qss_turing = 27` (every system passes the strict QSS
control). The numerator is the **conjunction** over all 9 μ — the conservative reading of
§5.3's "across the band". The per-μ marginals (`frac_turing_by_mu`) are 1.00 at every one of
the 9 points, so nothing is hiding at a band edge.

- **`frac_reentrant` = 2/27 = 0.074.** `harvest/nc1__mobile3/0` and `harvest/nc1__immobile/0`
  regain the verdict at larger μ after losing it — the feature `mu_critical`'s scan (rather
  than a bisection) exists to detect.
- **First Turing-loss edge vs the biological band: `frac_edge_in_band` = 0/27.** 24 of 27 never
  lose strict Turing anywhere up to μ = 1e4 (`status = turing_at_hi`, `mu_crit = inf`); the
  other 3 lose it at μ_crit = **0.0599, 0.0599, 0.1794**, i.e. **6.5–19× above MU_BIO_HI**, and
  in all three the failing condition is `uniform` (the homogeneous state goes unstable), not
  `band`. `mu_crit_median = inf`.
- **No Hopf anywhere in the band**: `frac_stationary_by_mu` = 1.00 at all 9 μ, on all 27
  systems, and `frac_oscillatory` is 0.0 in every cloud row. The §5.4 L1 non-negotiable Hopf
  exclusion is satisfied by every system at every μ tested.
- **Robustness, read against its own control.** `robustness_vs_mu` carries the μ → 0 QSS volume
  on the *same* 200 draws as the baseline, so along a curve only μ varies. Median strict-Turing
  volume: QSS baseline 0.315, μ = 1.1e-5 0.315, μ = 9.2e-3 0.285, drawn-μ 0.310. **Median ratio
  (band top / QSS baseline) 1.000 over n = 26; worst case 0.7209** (`d5/seed1`, 0.310 vs 0.430).
  Across the band the lift costs nothing to the perturbed-parameter Turing volume at the
  median and at most ~28 % on the worst system.

**Finding F-L11 — the median ratio is over 26 systems, not 27: `harvest/nc1__mobile3/4` has a
0/200 QSS-baseline volume and its 0/0 ratio is DROPPED.** Disclosed here because a dropped row
is invisible in the headline. The observation is worth stating on its own terms: **a system
that is strictly Turing at its nominal parameters has zero perturbed-parameter Turing volume
at σ_log = 0.10** — 0 of 200 draws Turing at the baseline and at every μ (`frac_fail_band`
0.975, `frac_fail_uniform` 0.025). Its nominal point sits in a Turing region small compared to
a 10 % parameter perturbation. Dropping it is the right treatment of a 0/0 ratio, and it does
not move the headline: 18 of the 26 ratios are exactly 1.0000, so entering the dropped row as
a 0 would leave the median at 1.000 (it would lower the mean, from 0.9740 to 0.9379).

**Finding F-L12 — the "200-draw cloud" is 121–200 draws on 10 of 27 systems, and
non-converged draws count as non-Turing.** `n_converged` < 200 on
`competitive__mobile3/{0,4}`, `competitive__immobile/{1,2,3}`, `nc1__mobile3/{2,3}`,
`nc1__immobile/{1,4,5}`, lowest at 121/200 (`nc1__immobile/1`). `frac_turing` divides by
n = 200 regardless, so a draw whose steady state did not converge is counted as *not* Turing.
That is the conservative direction — it can only understate the volume — but it means the
reported volumes are a lower bound on those 10 systems, not an estimate.

**Reading V4 against what §5.3 asked.** §5.3 V4 said: *"if that separation is generic, the
whole biological band sits inside the first Turing window and band-wide gating becomes cheap.
If a substantial fraction is lift-dead inside the band, that is the headline scientific result
of the survey, and it triggers §5.5."* **The separation is generic on this population: n = 27,
zero lift-dead inside the band, zero edges inside the band. §5.5's escalation is not
triggered.** The caveats belong in the same breath: the population is 23 in-box generator draws
plus 4 recovered models from one target — **not a random sample of anything**; and
`p_lifted_given_qss = 1.00` is a point estimate whose **Wilson 95 % lower bound is 0.875 at
n = 27** (two-sided, z = 1.95996; the Task 6 report printed 0.874, which matches no standard z
— recomputed here; the exact Clopper–Pearson one-sided 95 % lower bound is 0.895), i.e. the
data are consistent with up to ~12 % of such systems being lift-dead in the band. V4 is a
*linear* verdict throughout: "Turing-unstable" is not "patterns", which is V3's business.

---

## 5. GPU cost of the lifted integrator

`experiments/lift_ladder/gpu_port/results/cost.json` — `eval/lifted_torch.py::
simulate_lifted_torch` (CUDA) against `eval/lifted.py::simulate_lifted` (numpy CPU), 512²,
N = 3 competitive, μ = 7.2e-4, float64, RTX 5070 Ti Laptop, Strang loop only (one warm-up run
discarded; setup excluded).

**Two measurements exist and BOTH are cited, as a range.** `docs/REDESIGN_rngrn.md` §5.2 does
the same; this mirrors it rather than picking a winner.

| run | CUDA ms/step | numpy CPU ms/step | speed-up |
|---|---|---|---|
| T5 original (in git history at `b45f256`; `cost.json::git_commit` = `f367c43`) | 6.378 | 255.86 | 40.1× |
| T6 re-emit (current file, committed `0c4a19c`; `cost.json::git_commit` = `fa7380f`) | 6.076 | 186.85 | 30.75× |

CUDA moved ~5 %, CPU ~27 % — attributed to host load between the two sessions (same GPU, same
model/grid/μ, and no file under `src/rngrn/eval/lifted*.py` differs between the two trees).
Both are the same order and the qualitative claim is unaffected either way. Cost planning
below uses the slower-CUDA end where it matters.

---

## 6. What the ladder does NOT establish

Stated as plainly as the passes, per CLAUDE.md §8.

1. **No configuration at the §5.4 gate's own operating point was run** — 512²,
   μ_gate = 1e-3, dt = min(0.2/jac_rate, μ/2), pattern-formation horizon, with the dt-halving
   check L2 mandates. **No dt-halving pair exists at μ_gate at all** (F-L13: V3(b) ran at
   μ_central = 7.2e-4). See §7.2; this is the one substantive gap.
2. **Nothing here is a recovery result.** Every population is generator draws or already-
   recovered checkpoints. The ladder says the instrument reads true; it says nothing about
   what it will read.
3. **V1 branch identification is unreliable on 2/23 harvest systems** (F-L5), and the
   separation mask does not detect that failure mode.
4. **V2 is untested at μ ≥ 1e-1** (F-L6), and rests on n = 1 model per form.
5. **The V3 morphology-class claim rests on n = 4** (F-L7), with the coarse
   `observables.classify` estimator, which is not comparable to `validate.morphology_match`.
6. **V3(a) is silent about μ** at the QSS dt (F-L9); the μ-dependence rests on the 32² suite
   tests and on V3(b)'s dt-halving pair.
7. **V4's conditional is n = 27 on a non-random population**, Wilson 95 % LB 0.875, with one
   0/0 row dropped (F-L11) and 10 rows on partially-converged clouds (F-L12).
8. **Backend equivalence at campaign scale is 2 systems at 128²**, not a sweep.
9. **Every V3/V4 number is at N = 3** and at the two forms `competitive` / `nc1`; nothing here
   is evidence about larger N.

---

## 7. Does the ladder license the §5.4 gate?

**Returned for the orchestrating session to rule, with the numbers in front of it.** D-LIFT-1
makes the gate conditional on this ladder; the verdict updates D-LIFT-1's precondition and is
not taken here.

### 7.1 Component by component

| §5.4 component | what it needs the lift to be trusted for | ladder support | verdict |
|---|---|---|---|
| **L1** — lifted linear verdict at μ_gate = 1e-3 (strict Turing AND stationary, 21 branches) | the lifted Jacobian is exactly the QSS one's lift; the μ-scan is exact; the verdict is meaningful across the band | V0 (`rescale_mu` 2.3e-16 rel; gates-at-QSS 1.8e-15; fixed point μ-independent to the measured F/C bound), V1 (order ≈ 1 on 43 systems; fast branches stable everywhere; k\* offset 0 on 41/43), V4 (no Hopf on 27 systems × 9 μ; conditional 27/27) | **LICENSED.** F-L5's branch-swap does not touch L1: `turing_verdict_lifted` is a max over all 21 branches, tracking-free. |
| **L3** — k\* of the lifted rollout within one radial bin | the lift does not move k\* relative to the QSS run | V3 `kstar_within_one_bin` 10/10 in-scope at 128², 4/4 at 512², with μ_gate = 1e-3 in the μ set; the 5 out-of-scope failures are all decayed-field rows | **LICENSED for the lift-side half.** L3 compares against k\*_obs, which is a recovery matter the ladder cannot speak to; what the ladder establishes is that the lift contributes no k\* shift beyond one bin. |
| **L2** — lifted rollout patterns at μ_gate, under the §5.2 dt policy with a passing dt-halving check | the *spatial integrator at the gate's μ, dt and horizon* is converged, so "patterns / does not pattern" is the system's property and not the scheme's | V2 (0-D order 1.9946 **at μ_gate = 1e-3**, horizon error 5.35e-7 / 9.37e-12 vs the 1e-4 bar), V3(a) (`patterned_agree` 23/23 at 128², 8/8 at 512², μ_gate in the set), V3(b) (dt-halving ratio 0.9932 — but at μ_central, at 128², on a 1.44-time-unit transient), 512²-at-QSS-dt (`full_grid_patterned_agree` 8/8 — but at μ_central and dt/μ = 45–253) | **NOT YET LICENSED at its own operating point.** See §7.2. |

### 7.2 The gap, stated precisely

The two V3(b) legs cover complementary halves of what L2 needs, **their union is not it, and
neither leg is at μ_gate**:

- the **dt-halving pair** runs at the policy dt (μ/2) but on a 1.44-time-unit transient at
  128², and at μ_central = 7.2e-4 rather than μ_gate = 1e-3 (F-L13) — it shows
  dt-convergence, not patterning;
- the **512² run** reaches the attractor and agrees on `patterned` 8/8, but at μ_central and at
  the QSS dt, where dt/μ = 45–253 — **exactly the coupling trap §5.2 names** ("each gate
  substep relaxes fully to quasi-steady state, silently re-imposing QSS within a step"; §5.2's
  estimate of "dt/μ ~ 20–200" at μ_gate is measured here as 26.4–252.8). A `patterned_agree`
  of 8/8 obtained in that regime is weak evidence for L2 precisely because it is the regime in
  which the lifted run is *expected* to reproduce the QSS run.

So the single configuration L2 will actually be evaluated in — 512², μ_gate = 1e-3, dt = μ/2 =
5e-4, horizon 40/|σ_max|, with the halving check — has not been run. The reason is a resource
limit, not an oversight: `simulate_lifted`'s `max_steps` default is 200 000 and the required
step counts are 3.5e5 – 5.2e6.

**It is affordable on the GPU, and here is the arithmetic**, from the 8 anchor systems'
measured σ_max, at dt = μ_gate/2 = 5e-4 and the slower of the two measured CUDA costs
(6.378 ms/step at 512²):

| | steps | hours, one leg |
|---|---|---|
| cheapest of the 8 (`competitive__mobile3/1`, σ_max 0.2262) | 3.54e5 | 0.63 |
| median of the 8 | 5.49e5 | 0.97 |
| dearest of the 8 (`nc1__immobile/1`, σ_max 0.0154) | 5.20e6 | 9.21 |

A dt-halving *pair* is ~3× one leg. So one cheap gate-representative field costs ~2 h, the
median ~3 h, the worst ~28 h; all 8 at both dts ~57 h. **Recommendation: run L2's own
configuration — 512², μ_gate, dt = μ/2, full horizon, halving pair — on at least one
`morphology_claimable` system before the gate binds.** `competitive__mobile3/1` and
`nc1__immobile/0` are the cheap systems that also pattern. It requires raising
`simulate_lifted`'s `max_steps` (a knob, not a bar) and confirming
`stopped_reason == "horizon"`, which §5.4 L2 already demands. Without it the gate would bind on
an integrator never shown converged in the regime it will be used in — which is the exact risk
D-FFT-4 named and D-LIFT-1 undertook to answer.

### 7.3 Recommendation

**License L1 and L3 now; license L2 conditionally on the single run in §7.2.** The ladder as
measured is strong where it is strong — V0 and V2 pass by orders of magnitude, V1 passes on 43
systems with one precisely diagnosed exception, V3 agrees on the patterned flag 23/23 and 8/8,
and V4 returns 27/27 with no Hopf anywhere — and its weaknesses are all *scope* weaknesses
(small n, two untested large-μ points, one unrun configuration) rather than contradictions. None of
the measured numbers argues against the lift; the §7.2 gap is a missing measurement, not an
adverse one.

Two further items for the same ruling, neither decided here:

- **§5.5's escalation threshold** ("the working notion is ≈ ≥0.95 ... UNCALIBRATED, set from
  V4 itself") is now settable: the point estimate is 1.00 and clears it; the Wilson 95 % lower
  bound is 0.875 and does not. Which of the two the rule reads on is a decision, and it should
  be made before the number is used, not after.
- **Band-wide gating** (§5.4's owner-decision item 6, "revisited after V4") now has its
  evidence: 0/27 edges inside the band, closest 6.5× above MU_BIO_HI. That is an owner
  decision, not an agent one.

---

## Findings index

| id | rung | one line |
|---|---|---|
| F-L1 | V0 | C is 92× larger on competitive than nc1; the ratio identifies the summed-`free` cancellation as the mechanism |
| F-L2 | V0 | the two gate-normalisation reconstructions are NOT redundant — merging them would change the ODE |
| F-L3 | V0 | `steady_state`'s relaxation fallback is 3 orders looser than its Newton path; V0 excludes and reports |
| F-L4 | V1 | low_basal `frac_k_separated` is 0.442–1.000, not "~1.0" (Task 6 report corrected) |
| F-L5 | V1 | branch identification breaks on 2/23 harvest systems; L1 is unaffected (tracking-free) |
| F-L6 | V2 | μ ∈ {1e-1, 1} were never run; the order-2 claim is not demonstrated there |
| F-L7 | V3 | claim scope is 10/23 at 128² and 4/8 at 512²; a horizon artifact, resolution-independent |
| F-L8 | V3 | `l2_monotone` 18/23 is a last-digit tick at the O(dt) floor, quantified |
| F-L9 | V3 | at the QSS dt, V3(a) cannot see μ-dependence; it rests on the 32² tests and V3(b) |
| F-L10 | V3 | `l2_diff_dev_by_mu` = 9.4e+288 on one decayed row — a defect, not a value. **FIXED** (task 19): now NaNs; the 3 pre-fix rows in the committed `v3.json` are unre-emitted, read `l2_diff_by_mu` there instead |
| F-L11 | V4 | `nc1__mobile3/4` has a 0/200 baseline volume; its 0/0 ratio is dropped from the median |
| F-L12 | V4 | 10 of 27 clouds are 121–199 draws; non-converged draws count as non-Turing |
| F-L13 | V3 | V3(b)'s anchor ran at μ_central = 7.2e-4, not at the owner-set μ_gate = 1e-3; no dt-halving pair exists at the gate point |
