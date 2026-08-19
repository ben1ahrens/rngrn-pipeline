# DECISIONS.md excerpts — entries behind the paper-sprint claims

Verbatim copies of the decision entries the pack (and the two supplementary claim docs, PAPER_CLAIM_FORMCOMP.md and PAPER_CLAIM_WNOISE.md) reference, extracted 2026-08-19.
Each branch's full `docs/DECISIONS.md` remains the authoritative record. Sources:

- `feature/lift-ladder` @ 2f50fff — D-CANON-4, D-LIFT-1, D-LIFT-2, D-EVID-18, D-PAPER-1
- `feature/paper-claim3-hidden-channel` @ 60fddfb — D-PAPER-2
- `feature/paper-claim5-noise` @ 4515ea1 — D-CLAIM5-1
- `feature/paper-form-robustness` @ bfb9474 — D-FORMCOMP-1
- `feature/paper-weight-noise` @ 58445df — D-WNOISE-1
- `feature/real-stripes` @ 01b9bbf — D-REAL-1

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


---

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


---

### D-LIFT-2 — the V0–V4 ladder's UNCALIBRATED tolerances are closed by measurement: the V0 floor constants F/C, V1's error constants, V3's field-difference curve

**Date:** 2026-08-18 (redesign R1, branch `feature/lift-ladder`). **Status:** DECIDED
**Decided by:** the implementing agent under CLAUDE.md §10 (science decisions are taken
locally, calibrated and recorded). Nothing here weakens a pre-registered pass condition:
D-FFT-3's one-radial-bin k\* tolerance `2π/L`, V2's 0.1 × `pattern_floor` sup-norm bar and
V0's already-ruled two-constant bound are **inherited unchanged**. This entry replaces
placeholders with measured values; it moves no bar.

**The decision — three UNCALIBRATED marks in `docs/REDESIGN_rngrn.md` §5.3 become measured
constants, each with the run behind it.**

1. **V0's achievable fixed-point-residual floor** was UNCALIBRATED ("the only number on
   record is 1.28e-8 over 8 systems × 7 μ, in a parked artefact its own docs mark
   do-not-cite"). It is now two constants per form, because the residual has two honestly
   distinct mechanisms and a single bar is loose by ~5 orders in whichever regime it is not
   set by: `residual_floor_F` (max residual over μ ≥ 1e-2 — the x-block round-off floor,
   μ-independent) and `amplification_C` (max of `residual(μ)·μ/eps` over μ ≤ 1e-4 — the
   gate-block reconstruction discrepancy, amplified by 1/μ). **Measured, 20 Newton-tight
   draws per form, μ ∈ {1e-6, 1e-4, 1e-2, 1, 1e2}, seed 101:
   competitive F = 6.267e-11, C = 1154.5; nc1 F = 2.796e-11, C = 12.59.** The operative bound
   is `residual(μ) ≤ max(1e-9, 10·F_form, 10·C_form·eps/μ)`, frozen per form in
   `tests/test_lift_ladder.py` with <1 % rounding-up margin.
2. **V1's "absolute error constants: UNCALIBRATED — measured and recorded, not invented"** are
   now recorded, on 43 systems across three populations rather than asserted:
   slow-branch order **0.811 – 1.071** (low_basal 0.941–1.071, harvest 0.811–1.020, D5
   1.044–1.059); `kstar_grid_offset` **0 on 41 of 43**, 1 on 2 harvest systems;
   `min_fast_mu_product` (the −(1+u)/μ claim, ≈ 1 when it holds) **0.976 – 1.001 on low_basal
   (16/16) and 1.000 – 1.008 on 21 of the 23 harvest systems**, 0.349 – 0.998 on the 4 D5
   models (their large D brings slow branches up toward the gate rates — the same regime their
   `frac_k_separated` of 0.105–0.223 reports), and **5.44e-6 / 6.55e-6 on 2 harvest systems**,
   recorded as a branch-identification failure rather than as a lift property
   (DIAGNOSTICS_lift F-L5); every fast branch stable at every k and μ on all 43.
   `max_slow_err` is retained as an unmasked floor measurement and is explicitly **not** a
   tolerance — it must be read with `frac_k_separated` and `max_mu_D_k2`.
3. **V3's field-difference bound** was UNCALIBRATED, with §5.3 saying "the measured curve
   becomes the calibration". The measured curve is now on record, and it says the bound cannot
   be set from V3(a): **`l2_diff_by_mu` is identical to all printed digits across
   μ ∈ {1e-3, 1e-4, 1e-5, 1e-6} on every one of the 23 harvest rows** (median 2.7569e-2, max
   0.17625 at 128²; median 4.1344e-2, max 0.18789 at 512²), because with dt set by the QSS
   growth-rate policy dt/μ ∈ [26.4, 2.528e5] and the O(dt) scheme term is all that remains.
   **The calibration therefore comes from the dt ≲ μ regime instead**: V3(b)'s dt-halving pair
   at μ = 7.2e-4 — the literature-central value, **not** the owner-set gate point
   μ_gate = 1e-3 (D-REDESIGN-5; see DIAGNOSTICS_lift F-L13) — gives a median ratio of
   **0.99324** (dt-independent, i.e. O(μ)-dominated), and the 32² suite sweep gives
   2.04e-7 → 4.70e-8 as dt halves at μ = 1e-6
   against 5.01e-7 → 5.45e-7 (flat) at μ = 1e-3. **The definition adopted for the recorded
   curve is the raw relative field difference `‖X_lift − X_qss‖_F / ‖X_qss‖_F`** (the briefed
   key, the one every campaign JSON carries); the pattern-amplitude-normalised
   `l2_diff_dev_by_mu` is reported beside it and is **not** the definition.

**Evidence:** `experiments/lift_ladder/v0/results/v0.json` (V0, 15.1 s);
`experiments/lift_ladder/v1/results/v1.json` + `arrays/v1.npz` (V1, 64.0 s, 16 low_basal + 23
harvest + 4 D5 rows); `experiments/lift_ladder/v2/results/v2.json` (V2, 313.8 s);
`experiments/lift_ladder/v3/results/v3.json` + `arrays/v3.npz` (V3, 24 017 s) and
`experiments/lift_ladder/v3/cpu_gpu_check/results/v3.json`;
`experiments/lift_ladder/v4/results/v4.json` + `arrays/v4.npz` (V4, 1050.3 s);
`experiments/lift_ladder/gpu_port/results/cost.json`. Method, per-rung tables, the thirteen
findings and the run paths for every number: `docs/DIAGNOSTICS_lift.md`. Rung code
`src/rngrn/eval/ladder.py`; driver `scripts/lift_ladder.py`; the frozen V0 constants
`tests/test_lift_ladder.py::F_FORM`/`C_FORM`.

**What was rejected and why:**

- **A single V0 amplification constant** (ruling round 1) — measured floor-dominated at large
  μ and ~5 orders loose at small μ, which is exactly where V1 lives. Rejected in favour of the
  two-mechanism bound above.
- **Keeping the 1e-7 one-model residual bar as the V0 pass condition** — it fails at
  population level (competitive 2.564e-7 over 20 draws) for a reason that is not a defect, and
  §5.3 V0's own text says V0 *measures* the floor. Superseded, not weakened: the replacement
  is tighter than 1e-7 in both regimes it covers.
- **Admitting `steady_state`'s relaxation-fallback steady states into V0** — a fallback-path
  draw carries a residual up to 1e-4 constant across every μ, so the rung would have measured
  solver quality rather than lift algebra. `draw_models` now filters at ‖f(x\*)‖ ≤ 1e-10 and
  reports the rate it excluded at (0/20 competitive, 1/21 nc1).
- **Calibrating V3's absolute field-difference bound from the V3(a) campaign curve** — at the
  QSS dt that curve is a scheme floor with no μ-dependence in it, so a bound read off it would
  be a bound on `simulate_lifted`-vs-`rollout.simulate` Strang-vs-per-stage differencing, not
  on the lift. Rejected in favour of the dt ≲ μ measurements.
- **`l2_diff_dev_by_mu` as the recorded definition** — it is the more sensitive statistic on
  patterned rows but degenerates on decayed ones (measured 9.4e+288 on one row whose control
  has zero deviation; DIAGNOSTICS_lift F-L10). Reported, never adopted.
- **`scoring/morphology.py::classify_morphology` as V3's morphology estimator** — impossible,
  not merely undesirable: `eval/ladder.py` is RECOVERY_SIDE in `tests/test_firewall.py` and
  may not import the scoring package. V3 uses `observables.classify`, and **the resulting
  `morphology_agree` numbers are not comparable to any `validate.morphology_match` number.**

**Not independently validated:** the §5.4 gate's own operating point — 512²,
**μ_gate = 1e-3**, dt = min(0.2/jac_rate, μ/2), pattern-formation horizon, with the mandated
dt-halving check — was **not** run by this ladder (step counts 3.5e5–5.2e6 exceed
`simulate_lifted`'s 200 000 `max_steps` default; measured cost 0.63–9.21 h per leg at 512²
CUDA, median 0.97 h). V3(b) covers the policy dt on a 1.44-time-unit transient at 128² **and at
μ_central = 7.2e-4 rather than at μ_gate** (F-L13 — no dt-halving pair exists at the gate point
at all), and the 512² attractor run uses the QSS dt where dt/μ = 45–253. Whether the ladder
as measured licenses the gate is returned to the orchestrating session with this analysis (`docs/DIAGNOSTICS_lift.md` §7), and updates
D-LIFT-1's precondition when ruled. Also not validated: V2 at μ ≥ 1e-1 (never run); V1's
branch tracking on 2 of 23 harvest systems (F-L5); the V3 morphology claim beyond n = 4;
V4's conditional beyond n = 27 on a non-random population (Wilson 95 % LB 0.875).

**Where it lives:** `docs/DIAGNOSTICS_lift.md`; `docs/REDESIGN_rngrn.md` §5.3 (the three
UNCALIBRATED marks this entry closes); `src/rngrn/eval/ladder.py` (`v0_invariants`,
`v1_continuation`, `v2_temporal`, `v3_spatial`, `v4_survey` docstrings);
`tests/test_lift_ladder.py` (`F_FORM`, `C_FORM`); `docs/HANDOFF_lift_ladder.md`.


---

### D-EVID-18 — PREREG §3.3's morphology-distance threshold, computed: pooled 75th percentile of within-class distances in `three_gene_train` = 2.1072

**Date:** 2026-08-18. **Status:** DECIDED (threshold), adjudication is a reported result.
**Decided by:** measurement, per CLAUDE.md §10 (no owner escalation needed — a calibrated
number against real data, not a new pass condition).

**The decision:** §3.3's third sub-condition — `morphology_distance` ≤ "the 75th
percentile of within-class distances in `three_gene_train`" — had never been computed.
Computed directly from the 88-sample `three_gene_train` payload (42 labyrinth / 42 spots /
4 stripes, ground truth from the payload's own `morphology` attribute), using
`scoring.morphology.morphology_distance` at its own default (data-derived) `TRAIN_SCALE`.
The wording pools grammatically to ONE number over ONE set (not per-class), so the
**pooled** figure over all 1,728 within-class pairs from all three classes is adopted as
the threshold: **p75 = 2.1072**. Per-class values are reported as context, not as the
adjudicating number: labyrinth 1.8409 (n_pairs=861), spots 2.4159 (n_pairs=861), stripes
6.1343 (n_pairs=6 — weakly determined, flagged as such).

**Adjudication (reported result, not a new decision):** the c2_P campaign
(`experiments/c2_P_t8k8_consol/runs/`, 16 runs) scores **7/16** against this threshold on
the distance sub-condition, and **7/16** on the full §3.3 conjunction (match + distance +
`kstar_fft_rel_err` ≤ 8.3% + `trivial_kstar_err` reported) — identical counts, i.e. no run
that clears the distance bar fails on the other three clauses. The 7 passes are exactly the
`..._190648_seed*` batch (distances 0.27–0.59); every `..._192351_seed*` run fails the
distance clause despite `morphology_match = True` (distances 2.74–3.80, 1.3–1.8× the
threshold); one run (`..._190648_seed3`) has no model-side field to compare
(`morphology_scored = "target_only"`) and fails by construction.

**What was rejected and why:** a per-class threshold as the adjudicating number — the
prereg sentence names one distribution, not three, and the stripes class's own p75 rests
on only 6 pairs from 4 samples, too weak to adjudicate against alone (consistent with the
"stripes weakly characterised" limitation already in `scoring/morphology.py`'s docstring).

**Where it lives:** `experiments/figures_paper/prereg_33/` (`compute_prereg_33.py`,
`results.json`, `README.md`, `distance_distribution.png`); `docs/PREREGISTRATION.md` §3.3.


---

### D-PAPER-1 — Paper claim 2's "robust" is judged as lift-invariance of the Turing regime, not PREREG §3.2's absolute volume bars (owner reinterpretation, disclosed)

**Date:** 2026-08-19 (owner ruling given live 2026-08-18 ~23:00; recorded here next
session). **Status:** OWNER DECISION. **Decided by:** the owner — this is an interpretive
choice about the word "robust" in a paper claim, which is owner territory, not a §10
unilateral call.

**The decision:** for the paper's claim 2 ("learned GRN topology + parameters robust to
perturbation under dynamical lift"), robustness is defined as *remaining in the Turing
regime after parameter perturbation, under the finite-μ lift across the biological band* —
i.e. lift-invariance of the perturbation-cloud Turing volume and surviving k*. It is NOT
judged by applying PREREG §3.2's absolute bars (median `turing_volume_10pct` ≥ 0.90,
median `turing_volume_4p8pct` ≥ 0.95) to the lifted systems. `docs/PREREGISTRATION.md` is
NOT edited — it is binding and dated ahead of the runs it judges.

**Measured values that force the disclosure** (the pack must show original bar,
reinterpretation, and measured values side by side — `experiments/figures_paper/robustness/`,
`experiments/lift_ladder/v4/results/v4.json`):
- Per-model perturbation-cloud Turing volume, QSS → top-of-band (μ = 9.2e-3): seed1
  0.430 → 0.310, seed3 0.365 → 0.335, seed5 0.355 → 0.335, seed6 0.315 → 0.285. Relative
  drops 5.6–27.9%; **absolute volumes 0.285–0.430, well below §3.2's 0.90 bar**.
- Aggregate lift-invariance: P(lifted-Turing across the band | QSS-Turing) = 27/27 =
  1.000, Wilson 95% LB 0.8754 at n=27.
- §3.2's bars as written are still passed where they were designed to apply — the
  recovered models' unlifted local volumes (claim 4: 16/16, medians 1.000/1.000). That
  use is unchanged by this decision.

**Consistency note:** the standing prereg already points this way — §3.7 (D-LIFT-1) makes
the robustness-vs-μ row "Reported, never gated", and §3.6 (which would have applied §3.2's
bars to the lifted Jacobian) was added and withdrawn the same day (2026-08-03) before any
measurement existed against it. So the reinterpretation does not contradict a binding pass
condition; it is nevertheless recorded as an owner decision because the paper word
"robust" now means the invariance claim, not the absolute-volume claim.

**What was rejected and why:** editing PREREGISTRATION.md to fit the measured values
(binding document — weakening it is exactly what it exists to prevent); and using
"robust" in the paper without disclosing the original bar (a silently non-comparable
claim is worse than a missing one, CLAUDE.md §8).

**Where it lives:** `docs/PAPER_RESULTS_PACK.md` (claim 2 section, docs/paper-results-pack
branch); `experiments/figures_paper/robustness/README.md` + `caption_data.md`;
`docs/PREREGISTRATION.md` §3.2 (unchanged) and §3.7.


---

### D-PAPER-2 — claim-3 hidden-slow-channel experiment: `resid=0.3` UNCALIBRATED, 32 vs 64
restarts, and the slow-channel/hub confound

**Date:** 2026-08-19. **Status:** DECIDED (design), MEASURED (outcome), one input
UNCALIBRATED.

**The decision.** Paper claim 3 ("recovery of a 3N GRN from partial observations that
patterns robustly") is tested on `three_gene_qvar` / `sample_0001`, seeds 0-7, three arms
under `experiments/claim3_hidden_slow/`: (a) `replica_fullobs` — the exact `c2_P_t8k8_consol`
invocation (CUDA batched, `n_restarts=64`, `resid=0`), run as a code-drift check against the
historical baseline at git `4a61201`; (b) `ctrl_fullobs_resid` — full observation, serial CPU,
`resid=0.3`, `n_restarts=32`; (c) `hidden_slow` — identical to (b) plus `model.m=2`,
`model.observed_idx=[1,2]` (species 0 hidden). (b) and (c) are the claim-3 comparison: same
objective, same restart budget, same seeds, only the observation differs.

**`resid=0.3` is UNCALIBRATED, and that is unavoidable, not an oversight.** `recover.py:376-389`
raises loudly for any `m < N` run with `resid` weight 0 — the latent (hidden) field enters the
objective *only* through the stationarity residual, so at `resid=0` its gradient is exactly
zero and a "recovery" of it would just be reporting the random init. `resid > 0` is therefore
mandatory for arm (c), not a choice. But `recover.py:387-389`'s own comment records that
**exp06 already measured this residual as harmful to Turing recovery — 9/9 swept cells
collapsed** — so hidden-channel recovery "currently has no known-good objective, that is an
open problem, not a config mistake" (source comment, verbatim). `0.3` was picked as a plausible
non-zero value, not calibrated against a control sweep of its own; an honest negative result
(recovery failing because of the residual term, not because of the hidden channel per se) was
always a live possible outcome of this design and is called out here so it isn't misread as a
verdict on hidden-channel identifiability. Arm (b) exists specifically so this confound is
visible: (b) carries the same `resid=0.3` at full observation, so anything (b) itself loses
relative to arm (a) (`resid=0`) is the residual's cost, not the hidden channel's.

**`n_restarts` drops from 64 (arm a) to 32 (arms b/c), and (b)/(c) are restart-matched to each
other, not to (a).** `recover.py:443-446` refuses `batched=True` whenever `m < N` — the batched
reaction takes one state per member, not per-pixel states, so there is no batched residual —
which forces (b) and (c) onto the serial CPU path. A timing probe (`n_restarts=4,
adam_steps=50`, 1 worker, plumbing check only, not a result) measured 20 s wall and
extrapolated to a ~43 min/cell projection at `n_restarts=32`, `--workers=4`, comfortably under
the 2.5 h ceiling, so no further reduction was needed. Measured actual wall times came in well
under the projection: (a) ~18 min (CUDA batched, 64 restarts), (b) ~9 min, (c) ~11 min (both
serial CPU, 32 restarts, 4 workers) — the probe's linear extrapolation was conservative.
Because (a) runs a different restart budget *and* a different `resid`, it is not a
restart-matched or objective-matched comparison to (b)/(c); it exists solely to check that
`main@48441e4`'s training code has not drifted from the code that produced the historical
`c2_P_t8k8_consol` numbers. It has not: arm (a)'s `kstar_fft_rel_err_mean` came back
bit-identical to the `4a61201` baseline (`0.03522518377119767` both), as expected from
identical seeds, identical code, and `train.deterministic=true`.

**Slow-channel identification is unambiguous by both criteria checked.** `sample_0001`'s
ground truth (`payload.h5`, `params_json`): `D = [1.0, 190.293, 133.179]` — species 0 diffuses
~130-190x slower than species 1 or 2 — and `mu = [0.776, 1.321, 2.595]` — species 0 also has
the lowest reaction/degradation rate. Both the diffusion and reaction criteria independently
point at species 0, so hiding it (`observed_idx=[1,2]`) unambiguously hides the slow channel
under either definition.

**The disclosed confound: species 0 is also the topological hub, not just the slow channel.**
`interaction_matrix = [[1,-1,-1],[1,0,0],[1,0,0]]` — species 0 is self-activating and inhibited
by *both* other species, while species 1 and 2 each carry only a self-term. Species 0 is the
only species with any cross-species coupling at all. So this design cannot separate "hidden
slow channel" from "hidden hub" — any recovery degradation attributable to (c) below could be
either effect, or both, and this dataset does not let the two be told apart. That is a property
of the fixed design (§ controller spec), not something this run introduced or could correct.

**Measured outcome, (c) read against its control (b), never against zero (§8):**

| | (b) ctrl_fullobs_resid, full obs, resid=0.3 | (c) hidden_slow, species 0 hidden, resid=0.3 |
|---|---|---|
| n_recovered | 8/8 | 8/8 |
| n_turing (recovered_turing) | 8/8 (turing_frac 1.0) | 7/8 (turing_frac 0.875; seed 6 the exception) |
| kstar_fft_rel_err mean / median | 0.0527 / 0.0503 | 0.0743 / 0.0814 |
| kstar_rel_err mean / median | 0.0287 / 0.0312 | 0.0340 / 0.0330 |
| morphology_n_compared / match_frac | 8 / 0.875 (seed 6 turing-true but morphology-false) | 7 / 1.0 (seed 6 excluded — no rollout comparison for a non-Turing recovery) |

`recovered_turing` (a dispersion claim) and `morphology_match` (a rollout claim) are kept
distinct per `reporting-numbers.md`; they disagree with each other inside (b) itself (seed 6:
Turing-unstable, pattern mismatch), which is exactly the gap `eval/rollout.py` exists to catch,
not a defect in this run. Hidden-channel scoring used `scoring_mode: permutation_aligned`
(`validate.py:447`, via `PERM.observed_subblock_score` against the observed sub-block of the
Jacobian) with `observed_idx=[1,2]`; the permutation search's best alignment was the identity
`(0,1,2)` in the sampled run inspected, i.e. no permutation ambiguity was exploited given the
observed indices are pinned.

**What was rejected.** Calibrating `resid` before running (c) — no time in the paper-sprint
window, and exp06's existing sweep already establishes the qualitative risk; running it
UNCALIBRATED with the risk stated plainly was preferred over inventing an untested value or
blocking on a calibration sweep. Redesigning which channel is hidden, or hiding more than one
— the design is fixed by the controller for this claim, not decided locally. Reducing
`n_restarts` below 32 — the timing probe showed no need.

**Not established.** Whether the ~40-60% relative increase in k\* FFT error and the one lost
Turing recovery in (c) versus (b) is attributable to the hidden channel, the hub confound, the
`resid=0.3` cost (already partly visible as (b) vs (a)'s clean `resid=0` numbers), or some
combination — this single design cannot separate them. No sweep over which species is hidden,
nor over `resid`, was run.

**Where it lives:** `experiments/claim3_hidden_slow/{replica_fullobs,ctrl_fullobs_resid,
hidden_slow}/target_reports.jsonl`; gate at `recover.py:376-389` and `recover.py:443-446`;
`experiments/claim3_hidden_slow/README.md`.


---

### D-CLAIM5-1 — the observation-noise knob, its levels, and the reproducibility substitute

**Date:** 2026-08-19. **Status:** DECIDED (mechanism) / UNCALIBRATED (probe levels), by
design — this is the paper's claim 5 ("robustness of training to noisy training data"), whose
deliverable is a **measured curve**, not a pass/fail bar. No calibrated noise threshold exists
for this project and none is claimed here.

**The mechanism.** `data.obs_noise_sigma` / `data.obs_noise_seed`
(`src/rngrn/config.py::DataConfig`) add gaussian noise to the OBSERVED frame at the firewall
gate (`src/rngrn/data/gate.py::_apply_obs_noise`), strictly after `_observe()` slices out the
answer-key channels. `sigma` is defined **relative to each observed channel's own clean std**
(scale-free, so the same sigma is comparable across samples with different signal amplitude),
not as an absolute noise floor — a fixed absolute sigma would mean something different on a
low-amplitude channel than a high-amplitude one, and this dataset's three channels differ in
scale. `sigma=0` is the identity path (no RNG constructed, bit-identical output); `sigma>0`
with no seed raises `ValueError` (house style, no silent irreproducibility).

**The probe levels — UNCALIBRATED.** `sigma_rel in {0.01, 0.05, 0.2}` were chosen as a small
log-ish spread (1%, 5%, 20% of channel std) spanning "negligible" to "large" perturbation,
purely to see where the curve moves, if at all. They are not calibrated against any control
because there is nothing to calibrate them against — claim 5 is exploratory by the controller's
own framing ("measured curve only, no pass/fail language"). Rejected alternative: picking
levels post-hoc from where the curve broke, which would have been fitting the report to a
result instead of reporting a pre-specified probe.

**The reproducibility substitute — disclosed deviation.** The obvious way to keep noisy runs
byte-reproducible would be to materialise a noisy copy of the dataset as its own registered
payload (own `manifest.json`, own `payload.h5`). This was rejected in favour of noise applied
at load time, seeded and recorded in `frozen_config.yaml`: every noisy frame is regenerable
byte-identically from (tracked dataset checksum, sigma, seed) alone, with **no new
payload-writing script** and therefore no new entry needed in `tests/test_firewall.py`'s
`FORBIDDEN` list (CLAUDE.md §5's `scripts/` gap only applies to scripts that write
`payload.h5`; this path never does). Reproducibility is preserved exactly; the cost is that the
"dataset" a noisy run trained on is not itself a browsable artifact, only a formula plus a seed.

**A consequence, stated because it is deliberate rather than accidental.** Noise is applied to
the observed channels only, which also perturbs `kstar_obs` — the FFT measured off the
(now-noisy) observed frame, and hence the k-grid anchor (`CLAUDE.md` §7c point 2). This is
correct, not a leak: `kstar_obs` is computed from the observed frame at recovery time (never
from ground truth), so perturbing the observed frame is exactly what "noisy training data"
means. A design that kept `kstar_obs` clean while noising only the pixel values would have
tested a narrower and less honest claim.

**What was measured (three_gene_qvar/sample_0001, nc1, c2_P config, 8 seeds/level, cubic
dispersion, batched CUDA; `experiments/claim5_obs_noise/`):**

| sigma_rel | recovered_turing | kstar_fft_rel_err (mean) | kstar_rel_err (mean) | morphology_match | morphology_distance (mean) | plausibility_score (mean) |
|---|---|---|---|---|---|---|
| 0.00 (control) | 8/8 | 0.0352 | 0.0365 | 7/7 scored | 0.3988 | 1.0000 |
| 0.01 | 8/8 | 0.0278 | 0.0365 | 7/8 | 0.4764 | 1.0000 |
| 0.05 | 8/8 | 0.0450 | 0.0261 | 7/7 scored | 0.4041 | 1.0000 |
| 0.20 | 8/8 | 0.0448 | 0.0368 | 8/8 | 0.3762 | 1.0000 |

Every level fully preserves `recovered_turing` (8/8) up to sigma_rel=0.20 on this one target;
`kstar_fft_rel_err` and `morphology_distance` move within roughly the same band as the
seed-to-seed spread already present at sigma=0, not monotonically with sigma. One seed (seed 3)
fails to pattern on rollout (`rollout_status=unpatterned`, `morphology_match=None`) at sigma
0.00 AND 0.05 but patterns at 0.01 and 0.20 — non-monotonic, consistent with restart-seed noise
dominating over the injected observation noise at these levels on this one target, not with a
noise-driven degradation. **This is one target, 8 seeds per level — not a claim of
noise-tolerance in general**, and it should not be read as "noise doesn't matter"; it is a
measured curve on the probe levels chosen, nothing more.

**What was rejected.** (a) *A pass/fail bar on the curve.* Explicitly out of scope per the
controller's framing and CLAUDE.md §8 — no calibrated threshold exists. (b) *Absolute-sigma
noise instead of relative.* Would conflate channel-amplitude differences with noise-robustness
differences on this dataset's non-uniform channel scales. (c) *More than one target.* Time-
boxed by the paper deadline; the design was fixed by the controller to one target x 8 seeds x 4
levels, which is what was run.

**Code:** `src/rngrn/config.py::DataConfig.obs_noise_sigma/obs_noise_seed`,
`src/rngrn/data/gate.py::_apply_obs_noise`, `src/rngrn/train.py::_resolve_recovery_input`.
Tests: `tests/test_obs_noise.py` (8 tests, TDD). Runs:
`experiments/claim5_obs_noise/sigma_{0p00,0p01,0p05,0p20}/`.


---

### D-FORMCOMP-1 — comparing `nc1` vs `competitive` robustness, and the disclosed hyperparameter confound

**Date:** 2026-08-19 (paper form-robustness unit, branch `feature/paper-form-robustness`).
**Status:** DECIDED (design) / the resulting numbers are read as a **disclosed-confound
comparison**, not a clean one — no threshold here claims to isolate the form from the
hyperparameters.

**The decision.** Measure `model.form=competitive` on `three_gene_qvar` `sample_0001` and
`sample_0004`, K = 8 seeds (0…7), using the **exact hyperparameter recipe of the nc1 winning
cell** `c2_P_t8k8_consol` (`docs/C2_NC1_TUNING.md` §2: `loss.weights.turing=8.0`,
`loss.weights.kstar=8.0`, `loss.weights.param_prior=1.0`, 400 Adam steps, 64 restarts,
batched/CUDA/cubic dispersion) — copied verbatim from
`experiments/claim5_obs_noise/run_cells.sh`, changing only `--form nc1` → `--form competitive`
and dropping the noise overrides. Runs land at `experiments/form_compare/comp_{0001,0004}/`
in this worktree.

**Config file used, and why.** The spec asked me to diff `configs/nc1_m3_registry.yaml`
against `configs/m3_registry.yaml` and fall back to `nc1_...yaml -o model.form=competitive`
if they differ by more than the form line. They do differ by more than that line (header
comments, `tracking.run_name`) — but the difference is moot for `target-report`:
`src/rngrn/optim/target_report.py::run_target_report` (lines 471-489) explicitly overrides
`model.form`, `data.dataset_id` and `data.sample_key` onto the loaded config from the CLI
`--form`/`--dataset-id`/`--sample-key` arguments regardless of what the config file itself
says, and both files' bodies are otherwise identical (`_base_`, `data.source`, `model.N/m`,
`train.adam_steps/n_restarts`). So `configs/m3_registry.yaml --form competitive` and
`configs/nc1_m3_registry.yaml -o model.form=competitive` produce byte-identical effective
configs for this command. I used `configs/m3_registry.yaml --form competitive` (its default
`model.form` already matches, so no override is needed) and record this rather than silently
picking one.

**No committed competitive cell was reusable.** `docs/C1_COMPETITIVE_TUNING.md`'s own winning
cell (`turing8`, §9) sweeps `loss.weights.turing=8.0` **alone** — `kstar` stays at its
library default of 1.0, because C1 never swept a `kstar` axis at all (its axis list, §4, has
no `kstar` row) — and it was only ever run on `sample_0000` and `sample_0003` (§9, §10), never
on `sample_0001` or `sample_0004`. There is therefore no committed competitive-form run at
matching hyperparameters *or* matching targets to pair against `c2_P_t8k8_consol`, and new
cells are run rather than comparing across mismatched targets.

**The disclosed confound, named before any number is read.** `kstar=8.0` was chosen for
`nc1` specifically to suppress a mechanism measured on `nc1`: the near-immobile-third-node
route by which `turing=8` alone buys its Turing rate at the cost of an exploding k\* error
(`docs/C2_NC1_TUNING.md` §2, §6 — `kstar=8` took `kfft` from 4.765 to 0.115 on `sample_0000`,
*"at no measured cost in rate"* on `nc1`). Whether `competitive`'s analogous failure mode (if
any) is the same mechanism, and whether `kstar=8` is the right weight to police it on
`competitive`, was never measured — C1 measured only the `turing`-alone axis. So this unit's
`competitive` cells run at a hyperparameter point **chosen for the other form**, not one
`competitive`'s own tuning arrived at. **If `competitive` underperforms `nc1` here, that must
be reported as "underperforms at nc1-tuned hyperparameters", never as "underperforms" bare** —
the honest alternative reading is that `competitive` needs its own `kstar` sweep, which this
unit does not have time to run before the paper deadline. Context, not a gate: Stage-0 measured
the biological-box strictly-Turing acceptance rate at 2.5e-5 (`nc1`) vs 4.5e-4 (`competitive`)
over 80,000 box-constrained draws (`docs/C2_NC1_TUNING.md` header) — an ~18× prior difficulty
gap in the *opposite* direction, i.e. if anything `competitive` starts from an easier box.

**Analysis plan, fixed before the runs.** Per target and pooled across `sample_0001` +
`sample_0004`: `recovered_frac`, `turing_frac`; median/mean/min and full per-seed values of
`turing_volume_{1,4p8,10,20}pct`; `kstar_fft_rel_err` and `kstar_rel_err`
(median/mean, beside `trivial_kstar_fft_err`); `morphology_match_frac`,
`morphology_distance`; `plausibility_score_mean`. Each form's distribution is read against
the population baseline (`experiments/exp11_robustness_baseline.csv`, 127 systems × 400
draws) at all four perturbation levels, noting the 4.8%/10% ceiling effect already seen on
`nc1` (median 1.000) and that 20% is the only level with headroom to discriminate. The
pre-registered §3.2 bars (median `turing_volume_10pct` ≥ 0.90, `turing_volume_4p8pct` ≥
0.95) are reported for each form as-is, since they are pre-registered reference points, not
invented here — no new threshold is introduced. The "2 independent targets, not 3" caveat
(`sample_0001` is the same underlying system as `turing_labyrinth/sample_0000` at a
different grid resolution; only `sample_0004` is a genuinely independent second target — see
`worktrees/paper-pack/docs/PAPER_RESULTS_PACK.md` "three facts", fact 1) is carried into the
comparison verbatim, since it binds this unit's targets identically to `c2_P_t8k8_consol`'s.

**What was rejected and why.** (a) *Reusing `turing8`/`sample_0000`+`sample_0003` as "the
competitive comparison"* — rejected: different targets than `c2_P_t8k8_consol`, so any
delta would confound form with target, not isolate form. (b) *Re-tuning `competitive`'s own
`kstar` weight before comparing* — rejected: out of scope for the paper deadline, and even if
run it would no longer be a same-hyperparameter comparison, which is the point of this unit.
(c) *Treating a competitive shortfall as a pass/fail verdict against §3.2* — rejected;
§3.2 is pre-registered for the whole pipeline, not per-form, and `PREREGISTRATION.md` may not
be reinterpreted locally (CLAUDE.md §10). This unit reports measured numbers beside the bars,
nothing more.

**Where it lives:** `experiments/form_compare/comp_{0001,0004}/`;
`experiments/form_compare/README.md`; `docs/PAPER_CLAIM_FORMCOMP.md`.

**Outcome, recorded after the runs (2026-08-19).** Both cells completed cleanly (16/16
recovered, 16/16 Turing, zero seed errors). Contrary to what the confound above might
suggest, `competitive` at the `nc1`-tuned `t8k8` point does **not** underperform badly: it
clears both §3.2 bars pooled (median `turing_volume_10pct`=0.9800 ≥ 0.90, median
`turing_volume_4p8pct`=1.0000 ≥ 0.95), matching `nc1` at 1%/4.8%/10% closely and trailing
`nc1` by a real but modest margin only at the 20% perturbation level (pooled median 0.8325
vs `nc1`'s 0.9350, both still above the population median of 0.755). This is recorded
because a favourable result is exactly the case where the disclosed confound is easiest to
forget — the finding is "this hyperparameter point transfers reasonably well to
`competitive`", not "`competitive` is confirmed equally robust as a form", since no
`competitive`-own `kstar` sweep was run to check whether a form-appropriate choice would
close, hold, or widen the 20% gap. Full numbers: `docs/PAPER_CLAIM_FORMCOMP.md`,
`experiments/form_compare/numbers.md`.


---

### D-WNOISE-1 — train-time weight noise: per-step lognormal multiplicative noise on the positive physical parameters, smoothed-objective estimator, probe levels UNCALIBRATED

**Date:** 2026-08-19 (paper-experiment wave, Unit A, `feature/paper-weight-noise`).
**Status:** DECIDED (mechanism) / UNCALIBRATED (probe levels), by design — this is the
paper's weight-noise claim ("training is more robust when we inject noise into weights
during training, and consequently the learned GRN patterns more robustly"), whose
deliverable is a **measured curve** against a sigma_w=0 control, not a pass/fail bar
(claim-5 precedent, D-CLAIM5-1). An honest negative is a result.
**Decided by:** the implementing agent under delegated authority, design fixed by the
wave controller.

**The mechanism.** `train.weight_noise_sigma` / `train.weight_noise_seed`
(`src/rngrn/config.py::TrainConfig`, appended at the end of the dataclass) thread through
`train.fit` into `recover()`. When sigma > 0, every Adam step — on BOTH the batched
(`recover._batched_restarts`) and the serial path — evaluates the loss at perturbed
parameters and applies the resulting gradient to the CLEAN parameters (the classic
weight-noise / smoothed-objective estimator: descend `E_eps[L(theta+eps)]` by sampling
one eps per step, constant within the step). Noise is resampled each step from a single
CPU `torch.Generator` seeded by `weight_noise_seed`; sigma > 0 with no seed raises
(house style, same contract as `data.obs_noise_seed`). sigma = 0 is the identity path:
no generator constructed, bit-identical results (tested).

**The noise model, exactly.** For each positive physical parameter family
p in {s, alpha, delta, beta, D}, the perturbed value is `p * exp(sigma * z)` with
z ~ N(0,1) elementwise — lognormal multiplicative on the physical positives, i.e.
additive N(0, sigma^2) in their log space. This is EXACT, not approximate:
`theta_D` is a log (`D = exp(theta_D)`) so it takes `sigma*z` additively, and the
softplus-parameterised families are mapped through the exact numerically-stable softplus
inverse (`recover._softplus_inverse`). Chosen to match the EVALUATION perturbation model:
`eval/analysis._draw_JD_cloud` draws lognormal multiplicative factors on the physical
(J, D). Two stated mismatches with that model, neither hidden: (1) the eval cloud
perturbs the derived linearisation (J, D) while training noise perturbs the model's own
kinetic parameters — J's entries therefore receive correlated, nonlinearly-propagated
noise rather than independent factors; (2) the gate logit `theta_g` is NOT perturbed —
the gate is a bounded (0,1) split of the binding budget s into KA/KR, not a positive
scale, so lognormal multiplicative is undefined for it, and leaving it clean preserves
the perturbation's sign structure exactly as `_draw_JD_cloud` does by construction.
Since KA = s*g and KR = s*(1-g), noising s multiplies KA and KR of the same edge by the
SAME lognormal factor (correlated within the pair, independent across edges). The m<N
latent fields, the serial LBFGS polish, and the final scoring evaluation all run clean.

**Per-step, not per-restart.** The controller fixed per-step resampling (smoothed
objective) as primary over noise held constant per restart (randomised-prior ensemble)
— these are different experiments, a decision the prior art flags explicitly
(`worktrees/fft-submission/experiments/exp13/staged/NOISE_ROBUSTNESS_PIPELINE.md` §5
Unit 1; D-FFT-14 in `worktrees/fft-submission/docs/DECISIONS.md`). D-FFT-14's placement
warning — noise must not blow up warm-started forward solves — was checked against THIS
codebase and does not bind: training here never simulates (CLAUDE.md §7c), and
`losses/total.total_loss_batched` calls `terms.steady_state_batched` with the default
`x0 = ones` every step, so the Newton solve is NOT warm-started across steps (source
checked 2026-08-19). The residual risk is noise-induced Newton failures killing batch
members; the existing kill-on-first-failure contract is kept (a member whose steady
state fails at perturbed parameters dies for good, same as the serial path's abandoned
restart), monitored via the smoke run, with per-restart held noise as the recorded
fallback if sigma=0.048 shows failure explosion or non-ignition. The exp13 deck's
surrogate numbers are ILLUSTRATIVE only and are not evidence about this pipeline
(that document's own §1 and §4 say so).

**The probe levels — UNCALIBRATED.** sigma_w in {0.048, 0.10, 0.20} plus the sigma_w=0
control, on three_gene_qvar/sample_0001, 8 seeds per level, the exact claim-5 c2_P
invocation (nc1, batched CUDA, cubic dispersion, 64 restarts, 400 Adam steps,
param_prior=1.0, turing=8.0, kstar=8.0). 0.048 mirrors Tica's measured 4.8% experimental
CV (the evaluation cloud's own smallest level), 0.10/0.20 span "moderate" to "large";
none is calibrated against a control because none exists — they are probe points for a
curve. weight_noise_seed per level: 5348 (0.048), 5310 (0.10), 5320 (0.20); the control
runs with NO weight-noise overrides at all, exercising the untouched default path.

**Pre-registered analysis (written before the runs).** Compare each sigma_w level to the
sigma_w=0 control on: `recovered_turing` count; the distributions (not just medians) of
`turing_volume_{1,4p8,10,20}pct`; `kstar_fft_rel_err` (the accuracy cost axis); and the
morphology columns. Expectation stated in advance: at 4.8%/10% the control is at ceiling
on this target (median 1.000, claim-4 fact), so the 20pct column is where a difference
can show (population mean 0.746). Seed-level spread reported honestly — claim-5 saw
non-monotonic seed effects; 8 seeds on one target cannot support a general claim either
way, and the report will say so.

**What was rejected.** (a) *Additive Gaussian noise on the raw theta directly* — a single
sigma in raw space delivers wildly uneven effective physical noise across families
(softplus is ~linear for large theta, ~exp for negative theta; the exp13 deck's
ROBUSTNESS_MEASUREMENT reference measured a ~14x spread of effective physical noise for
exactly this reason) and would not match the evaluation perturbation model. (b) *Perturbing
the gate logit too* — a different (sign-structure-perturbing) noise model, undefined as
"lognormal multiplicative", and mismatched to `_draw_JD_cloud`. (c) *Per-restart held
noise as primary* — kept as the documented fallback; it answers a different question
(randomised prior, not smoothed objective). (d) *A pass/fail bar* — no calibrated
threshold exists; CLAUDE.md §8. (e) *Restoring clean parameters by subtracting the noise*
— does not round-trip bit-exactly in floating point; the implementation saves and copies
back the clean bytes instead (tested bit-exact).

**Not independently validated:** the smoothed-objective estimator's variance/benefit
trade-off on THIS loss landscape is exactly what the runs measure; nothing about the
direction of the effect is assumed. With noise on, the training-history trace and the
dead-member test see the perturbed parameters (the ones that produced that row's loss)
— documented at the site.

**Where it lives:** `src/rngrn/config.py::TrainConfig.weight_noise_sigma/weight_noise_seed`;
`src/rngrn/recover.py::_weight_noise_perturb/_weight_noise_restore/_softplus_inverse` and
the two Adam loops; `src/rngrn/train.py::fit`. Tests: `tests/test_weight_noise.py`
(14 tests, TDD; suite 596 passed / 1 skipped). Runs:
`experiments/claim_wnoise/sigma_w_{0p00,0p20,0p048,0p10}/` (run in that order so the
extremes exist if time runs out), smoke at `experiments/claim_wnoise/smoke/` (plumbing
only, never a number source).

**Outcome (2026-08-19, all four cells complete — recorded against the pre-registered
analysis above, no re-decision).** The curve is decisively negative for the
noise-helps-robustness hypothesis at the probed levels: Turing-unstable recoveries
8/8 → 6/8 → 1/8 → 0/8 across σ_w = 0 → 0.048 → 0.10 → 0.20; rollout patterning
7/8 → 5/8 → 0/8 → 0/8; k*_fft rel-err median 0.031 → 0.059 → 0.621 → 0.977. The one
pro-noise signal — vol_20pct median 0.992 over the six σ_w=0.048 survivors vs 0.935 over
all eight control seeds — is survivor-biased and bought at ~2× k* error; recorded as a
hypothesis for a sub-0.048 sweep, not a claim. Full report:
`docs/PAPER_CLAIM_WNOISE.md`; tables `experiments/claim_wnoise/summary_table.json` +
`seed_level.csv`. Cell 0p20's first attempt never ran (flock-acquisition timeout behind
an unrelated R3-lane run, exit 4); re-launched same day with the identical invocation,
exit 0 — provenance in `experiments/claim_wnoise/README.md`. The documented fallback
(per-restart held noise) and a finer σ_w < 0.048 sweep remain unrun.


---

### D-REAL-1 — real stripe images enter the registry with the FFT peak written into `k_star`; every k\*-vs-truth score on this dataset is CIRCULAR by construction

**Date:** 2026-08-19 (branch `feature/real-stripes`, owner-requested real-data experiment).
**Status:** DECIDED (with one knob marked **UNCALIBRATED** below).
**Decided by:** the implementing agents under §10 delegated authority. The scope — 8 frames ×
1 seed ("indicative"), device `cuda` (CPU in use elsewhere) — was set by the owner, 2026-08-19.

**The context.** Owner request: train RNGRN on 8 pairs of real stripe-pattern snapshots of ONE
bacterial colony carrying an engineered 3-node Turing circuit; only 2 of the 3 nodes are imaged
(channels C1, C2), so this is partial observation, m=2 < N=3. Then a dynamical-lift simulation
on the learned weights and topology/parameter/dispersion comparison graphics. Ingestion is
`scripts/ingest_stripes.py`: 300×300 grayscale PNGs → per-channel least-squares 2nd-order
polynomial illumination detrend → LANCZOS downsample to 96×96 → [0,1] rescale → (2,96,96)
samples, C1 = channel 0, no mask (interior crops, no colony edge). Dataset
`data/datasets/stripes_colony_2ch/` (8 samples). `L = 1.0` is ARBITRARY — the images carry no
scale bar, so L only sets the units of D and k (§7c: L enters only as a unit).

**The decision.** Real data has no generating model and therefore NO answer key — but
`data/gate.py::from_registry` hard-requires a per-sample `k_star` attribute, defined there as
the generator's *linear* answer-key wavenumber, and a sample missing it RAISES
(`gate.py:205,228`). We write the FFT-measured peak of channel 0 into **both** `k_star` and
`k_star_fft` (`scripts/ingest_stripes.py:97-100`, disclosed at the write site and in the module
docstring).

**Consequence, announced loudly per §10.4:** on this dataset `AnswerKey.kstar` IS the same FFT
anchor the training objective already uses (`kstar_obs` from the observed frame). Every
`kstar_rel_err`-style score computed on these runs is therefore **circular** — the model's k\*
compared against the very measurement training was steered by — and must never be reported as
accuracy against truth. It is not comparable to any `kstar_rel_err` reported on a generated
dataset, where `k_star` is independent of the training signal. The reportable quantities here
are cross-frame/cross-run consistency of the learned topology, parameters, and dispersion, and
the lifted-rollout comparison against the real images.

**What was rejected and why:**
- *(a) Making `k_star` optional in `gate.py`.* A contract change at the firewall boundary
  (`RecoveryInput`/`AnswerKey` separation, §5) that touches every consumer of the loader — too
  invasive mid-deadline, and the hard requirement is the right default for generated data.
- *(b) Leaving the attribute absent.* The loader raises by design ("a payload with no manifest
  fails loud" applies to attributes too, §6); silencing that would weaken a guard to dodge a
  disclosure that costs nothing.

**Sub-decision — `loss.weights.resid = 0.1`, UNCALIBRATED.** m=2 < N=3 puts the hidden channel
in the objective only through the stationarity residual, so `recover.py:376-388` raises unless
`resid > 0` (otherwise the latent fields would sit at their initialisation while presenting as
"recovered"). But no calibration for this weight exists: hidden-channel recovery is a documented
open problem (TUNING.md Milestone 2, `[TUNE/IMPL]`), and the one measurement we have —
`experiments/exp06_residual_sweep.json`, cited at the `recover.py` guard — found the residual
*harmful* to Turing recovery when all channels are observed. `0.1` is an honest placeholder,
marked UNCALIBRATED in `configs/nc1_stripes_partial.yaml`. What would calibrate it: a resid
sweep on a *generated* partial-observation dataset with an answer key (Milestone 2's harness).

**Sub-decision — `n_restarts = 8` (vs the tuned 64), serial path, CUDA.** The batched path
refuses m<N (`recover.py:443-446` — latent fields have no batched residual form), so restarts
run serially; 8 is a deadline compute compromise consistent with the owner's "indicative"
scoping. Device `cuda` is the owner's call (CPU busy) — accepted knowing §7's measurement that
the serial path is ~3× slower on CUDA than CPU. Other weights follow the claim-5 tuned nc1
recipe (`param_prior 1.0, turing 8, kstar 8`; see the config header).

**Firewall.** `ingest_stripes` opens `payload.h5` and is importable by bare name from
`scripts/`, so it was added to `tests/test_firewall.py::FORBIDDEN` (line 68) per §5's
hand-maintained scripts/ rule — the completeness test cannot see `scripts/`.

**Evidence (preprocessing acceptance).** A windowed radial-spectrum estimate on the RAW 300 px
images during ingestion found a dominant stripe wavelength of ~100 px (3 periods/frame) in 13
of 16 channel-frames and ~60 px in 3, with spectral prominence ×250–×4000 over background
(session measurement, recorded here; no committed artifact). Consistent with the committed
payload: the registered samples' `k_star_fft` spans **11.44–19.93 rad/L ≈ 1.82–3.17
periods/frame** (read back from `payload.h5` attrs, 2026-08-19). A smoke run (1 restart, 5
Adam steps, `experiments/real_stripes/smoke/`) verified config → loader → trainer → scorer;
per §8 its numbers are not findings — the harness runs, nothing more.

**Not established.** Whether *any* run is Turing-unstable on this data — a real possibility,
and a reportable outcome rather than a harness failure if it happens; whether `resid=0.1`
helps or hurts the hidden channel; and recovery accuracy in any truth-anchored sense, which is
UNMEASURABLE here by construction.

**Where it lives:** `scripts/ingest_stripes.py` (docstring + attrs write);
`configs/nc1_stripes_partial.yaml`; `data/datasets/stripes_colony_2ch/manifest.json`;
`tests/test_firewall.py::FORBIDDEN`; runs under `experiments/real_stripes/` (see its README).

**Addendum (2026-08-19, post-run) — final scope is 5 of 8 frames, by owner decision after a
run-directory collision.** Run dirs are named `<run_name>_<UTC-second>_seedN`; the relaunched
frames 1, 3, 4, 5 started within the same wall-clock second and all wrote to
`stripes_partial_20260819_183655_seed0`, so the last finisher's (frame 1's) checkpoint
survived and frames 3, 4, 5's models were destroyed. (Evidence: `target_reports.jsonl` maps
all four samples to that one run_id; the surviving checkpoint's `kstar_obs` = 16.5936 =
sample_0001's anchor, ≠ the anchors of 3/4/5.) The owner declined a rerun for time, so **every
cross-frame claim from this experiment covers frames 1, 2, 6, 7, 8 only** — announced here per
§10.4 because it changes what "n frames" means in any summary. The per-frame *metric rows* for
3/4/5 in `target_reports.jsonl` were computed before the clobber and remain valid records;
only the on-disk models are gone. Mechanical fixes, both in `run_frames.sh`: per-frame timeout
raised 3600→7200 s (launch 1 killed 3 frames at the wire under 4-way GPU contention), and a
per-sample `-o tracking.run_name="stripes_${s}"` override making run dirs unique per frame —
mandatory for any rerun. `analyze.py` guards independently by cross-checking each checkpoint's
`kstar_obs` against the sample's own anchor and dropping mismatches loudly.

**Result summary (indicative — 1 seed, 8 restarts, resid UNCALIBRATED; k\* agreement is
self-consistency with the training anchor, NOT accuracy):** all 5 analysed frames recover
Turing-unstable models whose lifted (finite-gate μ = MU_BIO_CENTRAL) rollouts pattern, with
lifted k\* tracking model k\* (e.g. 16.6→15.7, 19.3→21.2, 11.2→10.0) and μ_crit = inf over
the probed gate range for all 5. `morphology_match` (QSS rollout vs observed frame class) is
true for frame 7 only. Numbers: `experiments/real_stripes/analysis_summary.json`.

Two test consequences, fixed on this branch: `tests/test_gate_contract.py::
test_kstar_fft_lands_on_the_half_bin_grid` is now scoped to generated data (`split != "real"`)
— the half-bin quantisation is a property of the generated-data annotator, and this dataset's
real 2D radial FFT peaks sit up to 0.36 bins off the grid with no linear k\* to diagnose
against; and `eval/lifted.py` was classified RECOVERY_SIDE in `tests/test_firewall.py`
(inherited main-branch red, same line as the paper branches' fix).
