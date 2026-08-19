# HANDOFF — R2 (model reconstruction + observability), branch `feature/redesign-model`

**Written for a zero-context reader.** What changed, what is verified, what is **not**, and
where to start. Written at the end of Task 16 (the R2 capstone), 2026-08-19.

**The one-sentence version:** the R2 machinery all works and is tested, and the first
Phase-I population run says the redesign's ignition objective **does not ignite** — 0/512
members Turing-unstable against 3/64 and 11/128 for the matched legacy control — because the
pinned, β-derived parameterization makes a **decoupled** network (no cross-regulation) an
exactly feasible, unpenalised solution, and the newly promoted `param_prior` overwhelms the
k\*-anchor. Both causes are measured and attributed, neither is patched.

---

## 1. What R2 was for

`docs/REDESIGN_rngrn.md` is the ratified clean-slate design. §7 splits it into rounds; **R2
is "model reconstruction + observability"**, and its deliverables are quoted there: fixed-point
pinning, box-sigmoid reparameterization, `kstar_si`, retired terms, the A0 baseline preserved
verbatim, the §4.8 infrastructure, and

> a Phase-I-only population run on the target read against the D5 row on the §1 comparable
> columns, the measured per-member ignition rate and throughput/RSS curve that set B and K.

The target is **`turing_labyrinth/sample_0000`** (N=3, competitive, 512², L = 185.006). It is
a *burned tuning sample*: per `docs/PLAN_fourier_training.md` §1, work on it supports **no
recovery and no generalisation claim**. Every sentence below is of the licensed form "the
objective changes the measured baseline statistics on the tuning sample, in this direction,
by this much".

## 2. Where to start

1. `docs/REDESIGN_rngrn.md` §1 (target + the three comparable columns), §3.2 (pinning),
   §3.3 (boxes), §4.4 (objective), §4.5 (population strategy).
2. `scripts/r2_ignition_run.py` — the Phase-I driver. Module docstring = the contract;
   `run_arm` = the arms; `_r2_terms` = the objective; `_invariant_block` = §3.4's coordinates.
3. `experiments/redesign_r2/phase1/results/phase1_summary.json` — every number, with the D5
   row carried inside it.
4. `docs/DECISIONS.md` **D-R2-1** (box-init re-centring, OPEN), **D-R2-2** (run config,
   DECIDED), **D-R2-3** (the ignition failure, OPEN).

## 3. What landed

| Task | What | Where |
|---|---|---|
| T8 | loss-term registry + enumeration contract | `src/rngrn/losses/term_registry.py` |
| T9/T10 | `TrainingHistory` events + invariants + `to_arrays` | `src/rngrn/history.py` |
| T11 | `viz` module | `src/rngrn/viz.py` |
| T12 | `pin_xstar` — β solved out of the steady-state condition | `src/rngrn/model.py::RNGRN` |
| T13 | `param_boxes` — box-sigmoid α/δ | `src/rngrn/model.py::RNGRN` |
| T14 | `kstar_si` — scale-invariant k\* anchor | `src/rngrn/losses/terms.py` |
| T15 | two thin-driver notebooks + thinness test | `notebooks/` |
| T16 | batched pinning/boxing + the Phase-I run | `src/rngrn/model.py::BatchedRNGRN`, `scripts/r2_ignition_run.py` |

**T16's `src/` change, additively:** `BatchedRNGRN` accepts `pin_xstar` and `param_boxes`
(T12/T13 raised `NotImplementedError` there and named T16). Pin and box are **shared across
members**; a mismatched member is refused loudly. A pinned batched model registers no
`theta_beta` and gains a per-member `beta_hinge() -> (B,)` — per-member, because a batch-wide
scalar would couple members' gradients, the defect `_clip_grad_norm_per_member` exists to
prevent. `pin_xstar=None, param_boxes=None` is unchanged. Tests:
`tests/test_batched_pinned_boxed.py` (12), plus the two superseded refusal tests rewritten in
`test_pinned_model.py` / `test_box_reparam.py`.

**Suite: 662 passed, 24 skipped, 5 min 46 s**, unsandboxed, against this worktree's own
`.venv` (T15 recorded 650/24; the +12 are the new batched pin/box tests).

---

## 4. The Phase-I run

All runs: CUDA (RTX 5070 Ti Laptop), `dispersion_backend='cubic'`, N=3, K=400, 1500 Adam
steps, lr 0.05, grad-clip 10.0, no LBFGS, `turing` staged (off 25 %, ramp 25 %), member seeds
`recover._restart_seed(0, r)` so the arms share members and the B rungs are nested.
Run dirs under `experiments/redesign_r2/phase1/`, git_sha `1676d52`.

### 4a. The D5-comparable columns

Only three D5 columns are comparable (§1): `turing_frac`, `n_distinct_structures`,
`kstar_fft_rel_err`. **A0 is the control; D5 is context** — D5 ran a different budget
(10 seeds × 4 restarts, 2000 Adam + 50 LBFGS, serial, `eig`).

| run | B | Turing | frac | distinct | modal frac | mean agree | kstar_fft_rel_err (med) |
|---|---|---|---|---|---|---|---|
| **r2** | 64 | 0 | 0.000 | 3 | 0.656 | 0.885 | 0.9769 |
| **r2** | 128 | 0 | 0.000 | 3 | 0.672 | 0.889 | 0.9769 |
| **r2** | 256 | 0 | 0.000 | 3 | 0.598 | 0.875 | 0.9769 |
| **r2** | 512 | 0 | 0.000 | 3 | 0.619 | 0.879 | 0.9769 |
| **a0** (control) | 64 | 3 | 0.047 | 44 | 0.078 | 0.593 | 0.9769 |
| **a0** (control) | 128 | 11 | 0.086 | 62 | 0.094 | 0.587 | 0.9769 |
| *D5* (context) | 10 seeds | 4 | 0.400 | 10 | 0.100 | 0.499 | 0.9769 |

**Read this table with three caveats, or it lies to you.**

1. **The r2 arm's "3 distinct structures / 0.62 modal" is NOT a consistency win.** It is the
   population collapsing onto the *trivial decoupled* network. The recovered J sign
   structures are diagonal-only with every off-diagonal scored 0 at `sign_zero_rtol=0.05`.
   Reporting 3-vs-10 as an improvement over D5 would be the single most misleading number in
   this document.
2. **`kstar_fft_rel_err` = 0.9769 is degenerate in every arm, including D5.** It is
   `|k_min − k*_fft| / k*_fft` where `k_min = k*_obs/50 + 1e-3 = 0.006678` is the
   **k-grid floor** — i.e. σ(k) has no interior maximum, so `argmax σ` sits on the first grid
   point. The r2 arm's mean equals its median to 15 digits (every member identical), and the
   value coincides with D5's median to 15 digits because D5's median seed was in the same
   state. It is not a wavenumber measurement; it is "no structured mode exists".
3. **D5's denominator is SEEDS (best-of-4 restarts), ours is MEMBERS.** §4.5 says outright
   that D5's 4/10 "bounds neither the per-member ignition probability nor the redesign arm's".
   The per-member rate is the new measurement: **a0 4.7 %–8.6 %, r2 0.0 %**.

### 4b. Throughput and RSS (the curve that sets B)

Telemetry OFF on every row, so the rungs are comparable (recording the trajectory + §3.4
invariants at stride 100 costs ~53 % at B=512: 6313 → 2957 member-steps/s).

**Caveat (I2/I3, added at review): these six rows come from different processes, not one
run, and run-to-run noise is at least as large as the effect below.** Each row is a separate
driver invocation; `phase1_r2_B512/PROVENANCE.md` documents ≥1.65× spread (4885 vs 2957
member-steps/s) measured on the SAME cell (B=512, r2, telemetry on) across two different
invocations, with no guard-log evidence for either the committed run.json's actual invocation
or for the telemetry-off `phase1_throughput/phase1_r2_B512` cell used in the row below. **The
+14 % B=256→512 throughput difference this table's next bullet reports is therefore NOT
established by these data** — a same-process, telemetry-off, back-to-back B=256/B=512 pair was
never measured, and the observed cross-process noise (1.65×) is an order of magnitude larger
than the reported effect (1.14×). The B=256 recommendation in §7 item 4 does not change: it
was already the conservative choice, made for throughput-saturation reasons that do not depend
on the exact +14 % figure holding up.

| arm | B | member-steps/s | ms per member-step | wall (s) | peak host RSS (MB) | CUDA alloc (MB) |
|---|---|---|---|---|---|---|
| r2 | 64 | 2168 | 0.461 | 44.3 | 1915 | 91.4 |
| r2 | 128 | 3737 | 0.268 | 51.4 | 1989 | 112.8 |
| r2 | 256 | 5527 | 0.181 | 69.5 | 1991 | 155.5 |
| r2 | 512 | **6313** | **0.158** | 121.6 | 1909 | 243.2 |
| a0 | 64 | 60 | 16.6 | 1598.5 | 1989 | 91.7 |
| a0 | 128 | 70 | 14.3 | 2758.5 | 1991 | 113.4 |

- **Throughput saturates between B=256 and B=512**: +100 % members buys +14 % throughput
  (this specific difference is UNESTABLISHED by cross-process data — see caveat above).
- **Host RSS is flat in B (~1.9–2.0 GB) and dominated by the CUDA context, not the batch.**
  CUDA allocation is 91 → 243 MB over an 8× range in B. So on GPU, **B is limited by
  throughput saturation, not memory**, and the §7a guard's 8 GB `MemAvailable` floor is
  satisfied by a single ~1.9 GB process with room to spare.
- **Pinning is worth 36–90× on throughput** (r2 0.158–0.461 ms vs a0 14.3–16.6 ms per
  member-step) because it deletes the per-step steady-state solve entirely, which is exactly
  what §3.2 predicted. The a0 cost is not constant across the run: it is the *known* damped-
  Newton relaxation fallback (`terms.steady_state_batched`'s own docstring records
  110–170 ms/step early, rising to 10,500–17,600 ms once parameters drift). **Measured here
  at B=256: >13 s per step.** That pathology is why the a0 B=256 and B=512 rungs were **not
  run** — see §6.

### 4c. Why the r2 arm does not ignite — attributed, not guessed

Two attribution ablations at B=64, same seeds
(`experiments/redesign_r2/phase1_ablation/`), each removing exactly one departure:

| arm | Turing | distinct | kstar_fft_rel_err (med) |
|---|---|---|---|
| r2 (pin + box + prior) | 0/64 | 3 | 0.9769 |
| r2_nobox (pin, softplus α/δ, prior) | 0/64 | 3 | 0.9769 |
| r2_noprior (pin + box, `param_prior`=0) | 0/64 | 5 | **0.0332** |

**Finding 1 — `param_prior` at weight 1.0 destroys the k\* anchor.** Turning it off moves the
median k\* error from **0.977 to 0.0332, a factor of 29**, i.e. with the prior off the model
*does* find a structured mode within 3.3 % of k\*_obs. The trace shows total prior dominance:
`d_ratio` is driven to **exactly 7.500** (5th and 95th percentiles both 7.5000 across all 512
members) while `L_kstar_si` moves only 0.0809 → 0.0791 over the whole run (**2 %**). §3.3
predicted the prior would "make the viability tension measurable"; the measurement is that at
weight 1.0 it does not merely measure the tension, it wins it. The promotion weight needs
calibrating — 1.0 was the neutral promotion, and it is wrong.

**Finding 2 — the pinned, β-derived parameterization makes a decoupled network free.**
Ignition fails in *all three* r2 variants, so neither the box nor the prior causes it. The
measured mechanism, from `experiments/redesign_r2/phase1/phase1_r2_B512/arrays/plot_arrays.npz`:

| quantity | a0 (control) | r2 |
|---|---|---|
| median max\|off-diag J\| / max\|diag J\| | **0.582** | **0.0027** (nobox 0.0018, noprior 0.0145) |
| median ‖J‖_F | 1.93 | **6.66** |
| binding budget `s = KA+KR`, median | — | 0.314 → **0.0133** (24× collapse) |
| promoter occupancy `KA·x*ⁿ/(1+…)`, median | — | 0.0244 → **5.7e-05** (430× collapse) |
| β/δ per species, final median | — | 0.4193, 0.2674, 0.3504 |

Compare that last row to the pin `x* = (0.41938, 0.75467, 0.35046)`: **β/δ → x\* to 0.03 % on
species 0 and 2.** That is the analytic signature of the decoupled solution. With the
production term `prod(x*) → 0`, the derived β is exactly `β_i = δ_i·x*_i`, which (a) satisfies
the pinned fixed point exactly, (b) satisfies the β ≥ 0 hinge comfortably, and (c) costs
nothing in any other term. **Nothing in the R2 ignition objective requires non-zero
cross-regulation**, and pinning removed the two terms that incidentally discouraged it
(`frame_scale_anchor`, retired by §3.2 because pinning subsumes the amplitude channel; and
`anticollapse`, retired by §3.3). Species 1 is the partial exception — its J diagonal sits at
−0.03 against −4.71/−4.72 for species 0 and 2, and it carries the only surviving occupancy
(max final 0.231) — so the collapse is near-total rather than total.

**And restoring `anticollapse` would not fix it** — this is computed from the measured J, not
guessed. `anticollapse` floors ‖J‖_F at `jac_floor = 1.0`; the r2 arm's ‖J‖_F is **6.66**, so
its hinge `softplus(1.0 − 6.66)` is ~0.003, inactive. §3.3's structural argument for retiring
it ("box-confined rates cannot let ‖J‖ collapse to zero") is *correct about ‖J‖ and irrelevant
to this failure*: the collapse is of the **off-diagonal**, through the binding budget `s`,
which §3.3 explicitly does **not** box ("D and s/g are NOT boxed in this design"). A term
bounding coupling — not norm — is what is missing.

**What the r2 arm did do:** `L_turing` fell 1.151 → 0.143 and `sig_max_pos` rose −1.149 →
−0.0329, approaching the +1e-3 ignition margin from below without crossing it in 1500 steps.
Whether a longer budget crosses it is **not measured** (§4.5's own citation for hit-rate
saturation by ~4000 steps is on a different sample family and flagged there as not
like-for-like).

---

## 5. What is verified

- `BatchedRNGRN` pinned/boxed == B independent serial pinned/boxed models, every constrained
  parameter to ≤1e-15; the pinned fixed point is exact (`|f(x*)| < 1e-12`); `beta_hinge`
  reduces per member and its gradient reaches `theta_delta/alpha/s/g`; mismatched pins and
  boxes are refused. 12 tests.
- The batched Jacobian at the pin equals the serial Jacobian (max abs diff 0.0 / 1.1e-16 on
  three seeds), including through the stride-0 `expand` used for the pin.
- The whole suite, unsandboxed: **662 passed, 24 skipped**.
- Every number in §4 traces to a run directory under `experiments/redesign_r2/`.

## 6. What is NOT verified — read this before quoting anything above

- **No ignited member exists in the redesign arm**, so every population statistic for r2 is a
  statistic *of a non-igniting population*. `topology_consistency`, `mean_agreement` and the
  distinct-structure count describe a collapse, not a recovery.
- **`a0` at B=256 and B=512 was not run.** The a0 arm costs >13 s/step at B=256 (the Newton
  relaxation fallback); the two rungs would have cost 5+ hours and were stopped deliberately.
  The a0 control therefore exists at **B=64 and B=128 only**. The r2-vs-a0 comparison in §4a
  is matched at those two rungs; there is no a0 number at B=256 or B=512 and none is implied.
- **The first sweep's B=64/128/256 cells and the later B=512 cell come from two invocations**
  of the driver. Between them the driver gained the ablation arms, `_arm_weights`, per-cell
  index writes, `--skip-existing` and `_invariant_block`. The `r2`/`a0` code paths are
  unchanged by all of those (the arm conditionals were widened from `== "r2"` to `!= "a0"`,
  identical for the two arms that ran). Stated because the run records `git_sha 1676d52`, the
  branch HEAD at launch — the driver was uncommitted at the time.
- **`a0` has a parameter trace but no §3.4 invariant trace.** The occupancy invariant needs
  x\*, which a pinned arm has for free and which `total_loss_batched` does not return;
  recovering it would mean re-running the pathological solve per recorded step.
- **Every weight in the ignition objective is UNCALIBRATED** (D-R2-2), and Finding 1 shows at
  least one of them is actively wrong.
- **`n_distinct_structures` is not comparable across B.** D5's is over K=10, ours over
  K=64…512. A larger population can only find more distinct structures, which makes r2's 3 at
  B=512 more striking and a0's 44/62 less so.
- **The notebooks are still committed unexecuted.** `redesign_plots.ipynb` can now be run
  (a real `arrays/plot_arrays.npz` exists at
  `experiments/redesign_r2/phase1/phase1_r2_B512/arrays/`), but T15's contract — enforced by
  `tests/test_notebooks_are_thin.py::test_notebooks_are_committed_unexecuted` — keeps stored
  outputs out of git. §7's "figures regenerated end-to-end from `redesign_plots.ipynb`" is
  therefore **runnable but not run**; that is a controller decision, not an oversight.

## 7. Where to start next

1. **Add a coupling term to the ignition objective, or re-derive why one is not needed.**
   Finding 2 is the blocker: no amount of budget fixes an objective whose optimum is the
   trivial network. Candidates: a floor on off-diagonal ‖J‖ (not ‖J‖_F), a floor on promoter
   occupancy (already computed as a §3.4 invariant, bounded in [0,1], so it is a natural
   hinge), or boxing `s`. All three are untested.
2. **Calibrate `param_prior`'s promoted weight** against the `r2_noprior` arm, which is the
   control that already exists. A sweep between 0 and 1 is the obvious first measurement.
3. **Resolve D-R2-1** (box-init re-centring) before any boxed number is compared to a legacy one.
4. **Set B from §4b**: the knee is at B≈256, and memory is not the constraint.

## 8. Reproducing

```bash
# the sweep (B rungs + A0 control) — ALWAYS under the §7a guard
bash scripts/guarded_run.sh .venv/bin/python -u scripts/r2_ignition_run.py \
    --out experiments/redesign_r2/phase1
# the attribution ablations
bash scripts/guarded_run.sh .venv/bin/python -u scripts/r2_ignition_run.py \
    --out experiments/redesign_r2/phase1_ablation --B 64 --arms r2_nobox r2_noprior
```

`--skip-existing` is the default, so an interrupted sweep resumes from the cells already on
disk, and `runs.jsonl` / `results/phase1_summary.json` are rebuilt from every cell's own
`results/run.json` after each cell.
