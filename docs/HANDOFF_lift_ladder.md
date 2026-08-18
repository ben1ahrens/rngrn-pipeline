# HANDOFF — the lift-validation ladder (branch `feature/lift-ladder`)

Written for a **zero-context reader**. What was built, what was measured, what passed, what is
explicitly not established, and what the gate verdict is still waiting on.

---

## 1. What this branch is

`docs/REDESIGN_rngrn.md` §5 proposes replacing the current forward-verification gate with one
evaluated on the **finite-μ dynamical lift**: instead of judging a recovered model by the
quasi-steady-state (QSS) rollout, judge it by the full system in which the promoter gates are
dynamical variables relaxing at rate 1/μ, with μ the (TF-binding time)/(protein lifetime)
ratio, biologically ≈ [1.1e-5, 9.2e-3]. **The gate point is μ_gate = 1e-3**, set by the owner
2026-08-17 (`docs/DECISIONS.md` D-REDESIGN-5, on branch `docs/redesign-rngrn`), superseding the
literature-central 7.2e-4. The copy of `REDESIGN_rngrn.md` on *this* branch is stale and still
says μ_central; the docs branch is the authority.

`docs/DECISIONS.md` D-LIFT-1 adopted that gate **and made it bind nothing until a validation
ladder V0–V4 passes**, because the risk being managed is precisely that never-validated stiff
numerics turn an integration artefact into a "recovery failure". **This branch is that
ladder.** It is Tasks 1–7 of `docs/PLAN_redesign.md` milestone R1.

Nothing on this branch is a recovery result. The ladder validates the instrument.

## 2. What is on the branch

| | |
|---|---|
| `src/rngrn/eval/ladder.py` | the rung arithmetic: `draw_models`, `v0_invariants`, `v1_continuation`, `v2_temporal`, `v2_qss_limit`, `v2_rhs_mirror_error`, `one_radial_bin`, `v3_spatial`, `qss_verdict`, `v4_survey` |
| `src/rngrn/eval/lifted_torch.py` | the torch/CUDA port of the lifted Strang integrator (bit-equal to numpy on CPU) |
| `scripts/lift_ladder.py` | the campaign driver — `v0|v1|v2|v3|v4 --out <dir>`. No rung's arithmetic lives here, so the campaign and the unit tests exercise the same code |
| `tests/test_lift_ladder.py`, `tests/test_lifted_torch.py` | the unit tests, including the frozen V0 constants `F_FORM` / `C_FORM` |
| `experiments/lift_ladder/{v0,v1,v2,v3,v4,gpu_port}/` | the campaign results (tracked JSON + npz) |
| `docs/DIAGNOSTICS_lift.md` | **read this first** — per-rung method, every measured number with its run path, thirteen findings, and the gate analysis |
| `docs/DECISIONS.md` D-LIFT-2 | the tolerances the ladder closed by measurement, and what was rejected |

## 3. What was measured, in one table

Every number's run path is in `docs/DIAGNOSTICS_lift.md`; the JSONs are the authority.

| rung | claim | verdict | headline |
|---|---|---|---|
| **V0** algebraic | the lifted fixed point is μ-independent; `rescale_mu` is exact | **PASS** (as a measurement) | `rescale_mu` 2.3e-16 rel; gates-at-QSS 1.8e-15; residual floors F = 6.27e-11 / 2.80e-11, amplification C = 1154.5 / 12.59 (competitive / nc1) |
| **V1** linear | all 21 eigenvalue branches continue to the QSS limit at order ≈ 1 in μ; fast branches ≈ −(1+u)/μ; k\* unmoved | **PASS**, 1 exception | order 0.811–1.071 over 43 systems; `kstar_grid_offset` 0 on 41/43; every fast branch stable; branch identification breaks on 2/23 harvest systems (F-L5) |
| **V2** temporal 0-D | the Strang stepper matches a stiff Radau reference at order 2; the lifted ODE → the QSS ODE as μ → 0 | **PASS** | order 1.9939–1.9954; horizon sup-error 1.3e-13 … 4.4e-5 against the 1e-4 bar; `qss_order` 1.0000 / 0.9986; numpy-vs-torch RHS mirror error **0.0** |
| **V3** spatial | lifted field vs QSS rollout at matched seed/dt/horizon: same patterned flag, same morphology class (512² only), k\* within one radial bin, field difference falling with μ | **PASS in scope** | `patterned_agree` 23/23 at 128² and 8/8 at 512²; in-scope morphology and k\* 10/10 (128²) and 4/4 (512²); V3(b) dt-halving ratio 0.9932 (at μ_central, **not** at μ_gate — see §5) |
| **V4** band survey | how the lifted Turing verdict behaves across the biological μ band | **measurement, no pass bar** | **P(lifted-Turing across the band \| QSS-Turing) = 27/27 = 1.00** (Wilson 95 % LB 0.875); 2/27 re-entrant; 0/27 first-loss edges inside the band (closest 6.5× above it); **no Hopf anywhere**; robustness median ratio 1.000, worst 0.721 |
| **cost** | GPU port | measured | 512² CUDA 6.076–6.378 ms/step vs numpy CPU 186.85–255.86 — **30.8×–40.1×**, two runs, host-load variance |

## 4. The four things a reader must not misread

1. **"Turing-unstable" is not "patterns."** V0/V1/V4 are linear verdicts about a Jacobian.
   V2/V3 are dynamical statements about an integrated field. V4's 27/27 says nothing about
   patterning; V3's `patterned_agree` says nothing about the linear band.
2. **V3's `morphology_agree` uses `observables.classify`, not the reference-bank classifier.**
   `eval/ladder.py` is RECOVERY_SIDE in `tests/test_firewall.py` and may not import
   `rngrn.scoring` at all. **These numbers are not comparable to any `validate.morphology_match`
   number** anywhere in the repo. (Both fields compared are the model's own — the estimator
   differs, no ground truth is involved.)
3. **Claim scope: 10/23 at 128², 4/8 at 512².** 13 of the 23 harvest systems' *QSS controls*
   did not cross `pattern_floor` within the default growth-rate-aware horizon, so k\* and
   morphology read off those rows would be reading a decayed field. This is a **horizon**
   artifact, confirmed resolution-independent (the same systems stayed sub-threshold at 512²),
   and is **not** evidence against their Turing instability — V4 re-confirms all 23 under the
   strict QSS verdict. The morphology claim consequently rests on n = 4.
4. **`l2_monotone` 18/23 is not a μ-dependence reversal.** In every failing row the value at
   μ = 1e-3 is the smallest and the other three are bit-identical; the "failure" is an upward
   tick of 5.6e-19 … 8.5e-12 (relative 3.1e-11 … 4.4e-4) against row medians of ~2.8e-2, i.e.
   the sign bit of the 12th–16th digit at a floor that does not depend on μ. Two of the five
   are rows where both fields decayed to round-off. See F-L8, and F-L9 for why the campaign's
   dt cannot see μ at all.

## 5. What is NOT established

1. **The §5.4 gate's own operating point was never run** — 512², μ_gate = 1e-3,
   dt = min(0.2/jac_rate, μ/2), pattern-formation horizon, with the dt-halving check L2
   mandates. Worse, **V3(b)'s anchor was run at μ_central = 7.2e-4, not at μ_gate**
   (`scripts/lift_ladder.py::run_v3` uses `lifted.MU_BIO_CENTRAL`; the driver predates
   D-REDESIGN-5), so no dt-halving pair exists at the gate point at all. V3(a) and V2 *do*
   cover μ_gate = 1e-3. This is the one substantive gap; §6 below.
2. V2 was never run at μ ≥ 1e-1 (the §5.3 list's 1e-1 and 1), and rests on n = 1 model per form.
3. V1's branch tracking is unreliable on 2 of 23 harvest systems, and the separation mask does
   not detect that failure mode (F-L5). It does **not** affect the L1 verdict, which is a max
   over all branches and therefore tracking-free.
4. V4's conditional is n = 27 on a deliberately non-random population (23 in-box generator
   draws + 4 recovered models from one target); the Wilson 95 % lower bound is 0.875, i.e. the
   data are consistent with up to ~12 % of such systems being lift-dead in the band.
5. CPU/CUDA backend equivalence at campaign scale is 2 systems at 128², not a sweep.
6. Everything is N = 3, forms `competitive` and `nc1`.
7. `l2_diff_dev_by_mu` emits 9.4e+288 on one decayed row — a division-by-almost-zero, not a
   value (F-L10). Do not read it on rows whose control did not pattern.

## 6. What the gate verdict is waiting on

**The verdict is not taken on this branch.** `docs/DIAGNOSTICS_lift.md` §7 lays out the
component-by-component analysis and returns it to the orchestrating session, because it
updates D-LIFT-1's precondition. In summary:

- **L1** (lifted linear verdict at μ_gate = 1e-3) — **licensed** by V0 + V1 + V4.
- **L3** (k\* within one radial bin) — **licensed for the lift-side half**: the lift
  contributes no k\* shift beyond one bin. L3's comparison against k\*_obs is a recovery
  matter the ladder cannot speak to.
- **L2** (lifted rollout patterns at μ_gate under the §5.2 dt policy with a passing
  dt-halving check) — **not yet licensed at its own operating point.** The two V3(b) legs cover
  complementary halves, their union is not L2, and neither is at μ_gate: the dt-halving pair
  runs at the policy dt but on a 1.44-time-unit transient at 128² and at μ_central, and the
  512² attractor run uses the QSS dt where dt/μ = 45–253 — which is exactly the coupling trap
  §5.2 names ("each gate substep relaxes fully to quasi-steady state, silently re-imposing QSS
  within a step").

**What would close it, and what it costs.** One dt-halving pair at 512², **μ_gate = 1e-3**,
dt = μ/2 = 5e-4, horizon 40/|σ_max|, on at least one `morphology_claimable` system, with
`stopped_reason == "horizon"` confirmed. Required step counts are 3.5e5–5.2e6, so
`simulate_lifted`'s `max_steps` default of 200 000 must be raised (a knob, not a bar). At the
measured 6.378 ms/step (512², CUDA) that is 0.63 h per leg for the cheapest of the eight anchor
systems, 0.97 h at the median, 9.21 h for the dearest; a halving pair is ~3× one leg.
`competitive__mobile3/1` and `nc1__immobile/0` are the cheap systems that also pattern.

Two adjacent items also await a ruling: whether §5.5's escalation threshold (working notion
"≈ ≥0.95", UNCALIBRATED, to be set from V4) reads on V4's point estimate 1.00 or on its Wilson
lower bound 0.875; and band-wide gating (§5.4 owner-decision item 6, "revisited after V4"),
which now has its evidence — 0/27 edges inside the band — and is an **owner** decision.

## 7. Where to start

1. `docs/DIAGNOSTICS_lift.md` — the measurements, in full, with run paths. §7 is the gate
   analysis.
2. `docs/DECISIONS.md` D-LIFT-2 — what the ladder closed, and what was rejected on the way.
   D-LIFT-1 above it is the gate decision this ladder conditions.
3. `docs/REDESIGN_rngrn.md` §5 — the design being validated. Its §5.3 still carries three
   UNCALIBRATED marks that D-LIFT-2 closes; the replacement text is returned in
   `task-7-report.md` for the controller to apply (that file lives on a different branch).
4. To re-run any rung: `bash scripts/guarded_run.sh .venv/bin/python -u scripts/lift_ladder.py
   <rung> --out experiments/lift_ladder/<rung>` — **always through `guarded_run.sh`**
   (CLAUDE.md §7a) and always with the sandbox disabled. V0 15 s, V1 64 s, V2 314 s, V4 1050 s,
   **V3 6 h 40 m**.
