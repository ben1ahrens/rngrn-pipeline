# D-R2-1 — why the pinned + boxed R2 ignition objective does not ignite

**Task 18 of `docs/PLAN_redesign_R3.md`. Branch `feature/r2-ignition-diag`, BASE `db40995`.**
Diagnosis only. **Nothing here is a fix, and no objective was changed** — any change to the
objective is register-item class and goes to the owner via the controller.

> **THIS DOES NOT BLOCK THE PAPER.** The paper's claims rest on the legacy D5/FFT line and the
> lift ladder. The `r2` arm appears in no paper claim. This document is about the *redesign's*
> ignition objective, which is R3/R4 business.

**Everything below is re-derivable by `.venv/bin/python scripts/r2_ignition_diag.py`**, which
trains nothing: it reads the tracked `arrays/plot_arrays.npz` and `results/run.json` of runs
that already exist, and evaluates gradient probes at the r2 arm's own initialisation, rebuilt
from the tracked `config/frozen_config.yaml`. Its output is
`experiments/redesign_r2/ignition_diag/results/diagnosis.json`.

Which block emits what: **§1 → `step1_reproduction`, §3 → `step3_coupling_trajectory`,
§4 → `step4_term_gradients`**, and the supporting numbers quoted elsewhere — the §3a
β/binding-budget table, σ_max, the `w_turing` column, `L_kstar_si`, ‖J‖_F, `kstar_fft_rel_err`
(which lives in `results/run.json`, not the npz) and the a0 coincidence check —
**→ `step5_supporting`**. The staging schedule there is derived by calling
`rngrn.losses.weighting.staging_factor`, not asserted. *(An earlier revision of this paragraph
and of the script's docstring claimed the script covered every number when it covered only
Steps 1/3/4; `step5_supporting` was added so the claim is true rather than nearly true. A
docstring must not claim more than its code does.)*

---

## 0. The finding in one paragraph

The population does **not** start decoupled — it starts at coupling median **0.390**, which is
in the same range as the a0 control's final **0.582**. It is *driven* there, and **93.4 % of the
collapse happens before the `turing` term is ever switched on.** The `DataFirstStaging`
schedule inherited from D5 holds `turing` at weight **0 for steps 0–375** (first non-zero at
step 376, full at 750). Inside that window the
only term with any appreciable gradient on cross-regulation is **`beta_hinge`, which pushes the
off-diagonal binding budget DOWN for 100 % of 512 members**, and the only term opposing it,
`kstar_si`, is **314.9× too weak** to resist. By the time `turing` ramps in, coupling has fallen
**264×** and the population sits on the analytically-derivable decoupled optimum (β = δ·x\*,
matched to **0.17 %**). `turing` then does push coupling back up — by 3.94× — but from that
basin it never recovers: the **population-median** σ_max ends at **−0.031** and stays below zero
at every recorded step, and every one of the 512 members ends with **literally zero non-zero
off-diagonal signs**. (Two of the 512 members do end marginally positive, max **+0.0087**,
without meeting the ignition criterion — see §5.)

**So this is an objective-gradient story, not a parameterization story** — and the specific
defect is an *interaction* between the staging schedule and `beta_hinge` that T16 did not have
the trajectory data to see.

---

## 1. Step 1 — reproduction

Every headline number of T16 reproduces from the runs' own tracked arrays.
Source: `experiments/redesign_r2/phase1/`, `experiments/redesign_r2/phase1_ablation/`
(`git_sha 1676d52`; T16 report reviewed and passed 2026-08-19, so these are citable).

| arm | run path (under `experiments/redesign_r2/`) | Turing | rate | distinct signs | **non-zero off-diag signs** | coupling median (final) |
|---|---|---|---|---|---|---|
| r2 B=64 | `phase1/phase1_r2_B64` | 0/64 | 0.0 % | 3 | **0** | 0.002680 |
| r2 B=128 | `phase1/phase1_r2_B128` | 0/128 | 0.0 % | 3 | **0** | 0.002596 |
| r2 B=256 | `phase1/phase1_r2_B256` | 0/256 | 0.0 % | 3 | **0** | 0.002658 |
| r2 B=512 | `phase1/phase1_r2_B512` | 0/512 | 0.0 % | 3 | **0** | **0.002662** |
| **a0 B=64 (control)** | `phase1/phase1_a0_B64` | 3/64 | 4.7 % | 44 | 164 | 0.582209 |
| **a0 B=128 (control)** | `phase1/phase1_a0_B128` | 11/128 | 8.6 % | 62 | 328 | 0.582209 |
| r2_nobox B=64 | `phase1_ablation/phase1_r2_nobox_B64` | 0/64 | 0.0 % | 3 | **0** | 0.001815 |
| r2_noprior B=64 | `phase1_ablation/phase1_r2_noprior_B64` | 0/64 | 0.0 % | 5 | 12 | 0.014521 |

> The two a0 coupling medians being **identical to six digits is a real coincidence, not a
> copy error** — checked because it looks like one. The B rungs are nested by construction
> (shared `recover._restart_seed` members), the first 64 members' final Jacobians are
> **bit-identical** across the two runs (max abs difference **0.0**, a useful incidental check
> that `_clip_grad_norm_per_member` keeps members independent), and the 128-member median
> happens to straddle the same middle pair (0.5808728, 0.58354523) as the 64-member one.

T16's coupling median **0.00268** reproduces as **0.002662** at B=512. The **zero non-zero
off-diagonal signs** column is added here: it makes T16's "trap (c)" concrete and quantitative.
The r2 population's "3 distinct sign structures" is not reproducibility — the off-diagonal is
identically zero in every member, so only the three diagonal signs vary. The network is
**trivial**, and a trivial network cannot pattern by construction.

### 1a. The box re-centring (D-R2-1's literal subject) reproduces — and is NOT the ignition cause

Measured over the run's own 512 seeds, `configs/bio_box.yaml` = `alpha (0.5, 10.0)`,
`delta (0.4, 5.0)`:

| parameter | softplus init median | boxed init median | ratio |
|---|---|---|---|
| `alpha` | 0.6923 | 5.2457 | **7.578×** |
| `delta` | 0.6976 | 2.7103 | **3.885×** |

Reproduces T16's 7.58× / 3.88× exactly. The cause is as T16 stated: the raw scales
(`theta_alpha ~ N(0, 0.5)`, `theta_delta ~ N(0, 0.3)`) were chosen for `softplus`, and a raw
value near 0 lands at the **box midpoint** under `_box_sigmoid`.

**But it does not cause the ignition failure.** `r2_nobox` removes the box and still gives
0/64 — and in fact decouples *further* (final coupling 0.001815 vs r2's 0.002662). The box
re-centring is a **separate, real, still-open defect**: no boxed number is comparable to a
legacy one until it is resolved. Note that the brief's title ("why the pinned+boxed objective
does not ignite") and the label D-R2-1 (the box re-centring) name **two different problems**;
this document separates them.

---

## 2. Step 2 — the requested 2×2 is not constructible, and a better axis exists

**This is a returned plan defect.** The brief asks for {pinned, unpinned} × {boxed, unboxed}.
Three of the four cells cannot be made to mean what the design intends:

| cell | status |
|---|---|
| pinned + boxed | **exists** — `r2`, 0/64 |
| pinned + unboxed | **exists** — `r2_nobox`, 0/64 |
| unpinned + unboxed | **not the r2 objective.** `a0` is the *legacy* objective, not "r2 minus the pin" |
| unpinned + boxed | **does not exist and cannot be built** within the r2 objective |

The reason is structural and `scripts/r2_ignition_run.py:234-236` already states it: removing
the pin (a) reinstates the per-step Newton steady-state solve and (b) **deletes `beta_hinge`
entirely**, because only a pinned model derives β. An "unpinned" cell therefore changes three
things at once and would attribute nothing.

Step 4 below turns that from an inconvenience into the finding: **`beta_hinge` is the term doing
the damage**, so the pin axis is confounded with the culprit by construction. The informative
axis is not pin × box but **staging × `beta_hinge`**, and that axis *is* cleanly testable.

---

## 3. Step 3 — START or GO? It is GO, and it happens before `turing` exists

**Method.** The per-member Jacobian is reconstructed at every recorded step from `hist_params`
using the model's **own** autodiff kernel (`model._reaction_raw` + `torch.func.jacrev`), never a
hand-derived formula. Before any trajectory point is trusted, the reconstruction at the final
recorded step is validated against the independently-stored final `J`:

| arm | reconstruction vs stored `J`, max relative error |
|---|---|
| r2 B=512 | 1.176e-07 |
| r2_nobox B=64 | 1.168e-07 |
| r2_noprior B=64 | 5.379e-08 |

Agreement is at ~1e-7 rather than 0 because `hist_params` is stored `float32` while the run was
`float64`. That is the expected floor, and it is 4 orders below every effect discussed here.

**Coupling median trajectory** (`max|off-diag J| / max|diag J|`), from
`phase1/phase1_r2_B512/arrays/plot_arrays.npz`. The `w_turing` column is the staging weight as
recorded in the same file:

| step | `w_turing` | r2 (B=512) | r2_nobox (B=64) | r2_noprior (B=64) |
|---|---|---|---|---|
| 0 | 0.000 | **0.3897** | 0.2111 | 0.3931 |
| 100 | 0.000 | 0.0088 | 0.0020 | 0.0100 |
| 200 | 0.000 | 0.0037 | 0.0011 | 0.0046 |
| 300 | 0.000 | 0.0021 | 0.0008 | 0.0028 |
| **400** | 0.067 | **0.0015** | 0.0005 | 0.0019 |
| 500 | 0.333 | 0.0058 | 0.0020 | 0.0105 |
| 600 | 0.600 | 0.0060 | 0.0037 | 0.0084 |
| 800 | 1.000 | 0.0046 | 0.0031 | 0.0069 |
| 1000 | 1.000 | 0.0038 | 0.0025 | 0.0062 |
| 1500 | 1.000 | **0.0027** | 0.0018 | 0.0145 |

**Read this against the staging schedule, which is the whole point.** `off_frac=0.25,
ramp_frac=0.25` over 1500 steps means `turing` has weight **exactly 0 for steps 0–375**, first
becomes non-zero at **step 376**, and reaches full weight at **step 750**. (Fenceposts taken
from `weighting.staging_factor` itself, via `step5_supporting.staging_schedule`.)

1. **It is GO, not START.** Members begin at 0.390 — the same order as the a0 control's final
   0.582. They are not born decoupled.
2. **The collapse is essentially complete before `turing` is switched on.** 0.3897 → 0.0015 by
   step 400 is a **264× fall**. Step 300 is the last recorded step at which `turing`'s weight is
   strictly 0, and by then coupling is already 0.0021 — a **183×** fall, i.e.
   **log 183 / log 264 = 93.4 %** of the total collapse (measured on a log scale, which is the
   right scale for a quantity falling by two orders of magnitude) is complete before `turing`
   contributes anything at all. The decay is **monotone** across that window — the brief's own
   stated signature of "an objective-gradient story".
3. **`turing` does push back, and it is not enough.** The moment the weight becomes non-zero the
   trajectory reverses: 0.0015 → 0.0058 (≈4×) between steps 400 and 500. It then decays again
   for the remaining 1000 steps. A 3.94× recovery from a 264× hole does not reach ignition.
4. **`r2_noprior` is the one arm whose coupling grows late** (0.0062 at step 1000 → 0.0145 at
   1500, with p95 reaching 0.090). It is also the only arm with a real k\* (median
   `kstar_fft_rel_err` 0.0332). Consistent: with the prior off, there is budget to move.

### 3a. The decoupled optimum is reached analytically, and the binding budget is what collapses

From the same arrays. β is **derived** on a pinned model as `β = δ·x* − prod`; as `prod → 0` the
derived β tends to exactly `δ_i·x*_i`, which satisfies the pinned fixed point exactly and clears
the β ≥ 0 hinge at zero cost. That is precisely what happens:

| step | median \|β − δx\*\| / δx\* (species 0) | median binding budget `s` off-diag | `s` diag |
|---|---|---|---|
| 0 | 0.5009 | 0.3175 | 0.3083 |
| 100 | 0.0095 | 0.0673 | 0.0739 |
| **400** | **0.0017** | **0.0336** | 0.0386 |
| 1500 | 0.0003 | 0.0117 | 0.0195 |

By step 400 — still inside the staging window — β sits on the decoupled optimum to **0.17 %**,
and the off-diagonal binding budget has collapsed **9.4×** from its initial value. T16 reported
the endpoint of this (s: 0.314 → 0.0133, 24×); the trajectory shows **where** it happens.

---

## 4. Step 4 — which term moves the coupling, and in which direction

Gradient probes at the r2 arm's own initialisation (B=512, the run's own seeds, box, pin and
`kstar_obs`; CPU, float64). `theta_s` and `theta_g` are the only parameters that carry
cross-regulation. Descent moves `theta_s` by `−grad`, so a **negative** mean descent means the
term **shrinks** coupling.

| term | ‖g‖ total | ‖g‖ on `s` off-diag | mean descent on `s` off-diag | members pushed DOWN |
|---|---|---|---|---|
| `kstar_si` | 2.684e-01 | 2.632e-02 | **+9.850e-05** (up) | 18.8 % |
| `turing` | 2.073e+01 | 5.108e+00 | **+4.707e-02** (up) | 0.0 % |
| `param_prior` | 5.274e+01 | **0.000e+00** | 0.000e+00 (none) | 0.0 % |
| `beta_hinge` | 8.893e+00 | 3.832e+00 | **−3.102e-02** (DOWN) | **100.0 %** |

> **Read the last column carefully — it is `frac_members_pushed_down`, the fraction whose
> coupling the term SHRINKS.** So `kstar_si`'s 18.8 % means it pushes coupling *up* for the
> other **81.25 %**. An earlier revision of §4 point 3 and of fix D's rationale inverted this;
> both are corrected below. `param_prior`'s 0.0 % means **neither direction** — its gradient on
> these parameters is identically zero, not "up for 100 %".

**Five things follow, and together they are the diagnosis.**

1. **`beta_hinge` is the decoupling driver.** It pushes the off-diagonal binding budget down for
   **100 % of 512 members** — not a majority, all of them. Mechanistically this is forced: β is
   derived as `δ·x* − prod`, so the cheapest way to satisfy β ≥ 0 is to shrink `prod`, and
   `prod` *is* the cross-regulation.
2. **`param_prior` has exactly zero gradient on coupling.** Its norm is the largest of the four
   (52.7, 2.5× `turing`'s) but it lives entirely on `alpha`/`delta`/`D`. **This corrects the
   natural reading of T16's cause 1**: the prior dominating the objective is real and is why
   `d_ratio` pins to exactly 7.500, but the prior is *not* what decouples the network. The
   `r2_noprior` ablation already implied this (0/64 with the prior off); the gradient shows why.
3. **`kstar_si` is the only counterweight active during staging, and it is negligible** — **77.2×**
   below `turing` in total gradient norm and **194.0×** below it on the off-diagonal coupling
   specifically, while its coupling push is **314.9× weaker** than `beta_hinge`'s
   pull (9.85e-05 vs 3.10e-02). Its *direction* is nonetheless favourable: it pushes coupling
   **UP for 81.25 % of members** (down for 18.75 % — the `frac_members_pushed_down` column of
   the table above, which an earlier revision of this sentence read backwards). So `kstar_si`
   fails here purely on **magnitude**, not on sign. The brief's
   Step-4 hypothesis ("if `kstar_si` is orders down, that is the answer") is **confirmed on the
   magnitude**, with the refinement that the term it loses to is `beta_hinge`, not `param_prior`.
   Corroborated by the run itself: `L_kstar_si` moves **0.08085 → 0.07911 across all 1500
   steps — 2.1 %.** The term is effectively inert.
4. **`turing` is the only strong term that pushes coupling UP** (+4.707e-02, 0 % of members
   pushed down) — and it is **switched off for the first quarter of the run**, i.e. exactly when
   the collapse happens. Its magnitude (4.707e-02 up) is comparable to and slightly larger than
   `beta_hinge`'s (3.102e-02 down), so had it been active from step 0 the net sign on coupling
   would plausibly have been positive.
5. **The objective that is actually active during the staging window is decisively decoupling.**
   Summing the three live terms (`kstar_si` + `param_prior` + `beta_hinge`):
   mean descent on off-diagonal `s` = **−3.092e-02**, pushing **100 % of members down**, with
   the off-diagonal gradient budget split **`beta_hinge` 99.32 % / `kstar_si` 0.68 % /
   `param_prior` 0.00 %**.

### 4a. `kstar_si`'s UNCALIBRATED knobs — status, unchanged

The brief flags that `kstar_si`'s `temp` (inherited 60.0, never swept) and `eps` are
**UNCALIBRATED**, and that spec §4.4 assigned their sweep to R2, which did not happen. **That
remains true and remains unaddressed** — this task swept nothing. The measurements above say the
term is 77–315× too weak to matter at the current setting (77.2× below `turing` in total
norm, 194.0× on the off-diagonal, 314.9× below `beta_hinge`'s coupling pull), which makes calibrating it a live
candidate; they do **not** establish that any achievable `temp`/`eps` would fix ignition, and I
did not test one. See fix D below.

---

## 5. The mechanism, end to end

1. `DataFirstStaging(off_frac=0.25, ramp_frac=0.25)` holds `turing` at weight **0 for steps
   0–375** (first non-zero at 376, full at 750) — a schedule inherited unexamined from D5's
   frozen config.
2. In that window `beta_hinge` carries **99.32 %** of the gradient on off-diagonal coupling and
   pushes it **down for 100 % of members**; `param_prior` contributes exactly nothing to it; and
   `kstar_si`, the only opposing term, is **315× too weak**.
3. Coupling falls **0.3897 → 0.0015 (264×)** and the binding budget **9.4×**, monotonically. β
   lands on the analytic decoupled optimum `δ·x*` to **0.17 %**, which satisfies the pinned fixed
   point exactly and clears the β hinge at zero cost.
4. `turing` ramps in from step 376 and reverses the direction — coupling recovers **3.94×** — but
   from a 264× hole. The population-median σ_max rises only to **−0.031**, staying below zero at
   every recorded step; 2 of 512 members end marginally positive (max **+0.0087**) without
   meeting the ignition criterion (σ(0) < 0 **and** `sig_max_pos` > 1e-3).
5. Result: **0/512 Turing, zero non-zero off-diagonal signs, 3 "distinct" (diagonal-only)
   structures**, against a matched a0 control at 4.7–8.6 %.

**Why T16 could not see this.** T16 correctly identified that *no term requires* non-zero
cross-regulation. The trajectory adds the missing half: one term actively **penalises** it, and
the staging schedule removes the only counterweight for the first quarter of the run.

---

## 6. Ranked candidate fixes — RETURNED, NOT RULED

**No objective change was made. Each of these is register-item class and belongs to the owner
via the controller.** Ranked by strength of the evidence behind them, not by ease.

**A. Do not stage `turing` off in the pinned objective (strongest evidence, cheapest test).**
93.4 % of the collapse (log scale, §3) happens in the staging window; `turing` is the only strong term that
pushes coupling up, and it is disabled exactly when needed. There is also a *design* argument
independent of the measurement: `DataFirstStaging` means "fit the data before demanding
instability", but the r2 objective's **only** data-facing term is `kstar_si`, which moves 2.1 %
across the entire run. The staging window is therefore not "data-first" in any meaningful
sense — it is `beta_hinge` and `param_prior` running unopposed for 376 steps. Needs **no code
change**, but **not** via `--staging-ramp-frac 0`: `losses/weighting.py:150-151` validates
`0 < ramp_frac <= 1` and `r2_ignition_run.py:257-258` builds `DataFirstStaging`
unconditionally, so `ramp_frac=0` raises `ValueError` at construction. *(An earlier revision
registered exactly that command; it would have crashed. Corrected here and in §7.)* The
zero-code-change form is **`--staging-off-frac 0 --staging-ramp-frac 0.00067`**, which gives
`turing` weight 0.0 at step 0, 0.995 at step 1 and 1.0 from step 2 — i.e. live from step 1
rather than step 376. (No `ramp_frac` can make it live *at* step 0: with `off_frac=0` the factor
is `(step−off)/(ramp·total)`, which is 0 at `step=0` for any ramp.)
*Caveat:* this predicts improvement; the prediction was **registered but NOT TESTED** — see §7.

**B. Add a term that requires coupling (most robust; a genuinely new objective term).**
T16 already named three candidates in `HANDOFF_redesign_r2.md` §7 (an off-diagonal ‖J‖ floor; an
occupancy floor, already computed as a §3.4 invariant and bounded in [0,1] so a natural hinge;
boxing `s`). This is the only fix that removes the *structural* problem rather than
out-competing it — the decoupled network is currently a genuine optimum, and A merely makes the
optimiser less likely to fall into it. Note `anticollapse` would **not** serve: it floors ‖J‖_F,
and the measured ‖J‖_F is 6.66 against its floor of 1.0, so it is inactive. What collapses is
the off-diagonal, not the norm.

**C. Reformulate the pin so β ≥ 0 is not bought with coupling.** `beta_hinge` exists only
because the pin is solved *for β*, which makes β a decreasing function of `prod` and creates the
perverse gradient (99.32 % of the coupling gradient, 100 % of members). Solving the same
constraint for a different parameter, or enforcing β ≥ 0 by construction rather than by penalty,
would remove the pathway entirely rather than counterbalancing it. Deeper change than A or B and
interacts with T12/T13's pinned-model design.

**D. Calibrate `kstar_si`'s `temp` / `eps` (owed regardless).** Spec §4.4 assigned this sweep to
R2 and it did not happen; the knobs are UNCALIBRATED today. The measurements make it a live
candidate — the term is 77–315× too weak (§4).

> **Ranking rationale corrected.** An earlier revision ranked D below A–C on the grounds that
> `kstar_si` "pushes coupling up for only 18.8 % of members". **That read the sign backwards** —
> it pushes coupling **UP for 81.25 %**, i.e. its direction is *favourable*, the same direction
> `turing` pushes. The honest ranking argument is therefore **magnitude alone**: `kstar_si` is
> 314.9× below `beta_hinge`'s coupling pull and 194.0× below `turing`'s push on the same
> parameters, so closing that gap by tuning `temp`/`eps` would require roughly two to three
> orders of magnitude of gradient, and **nothing measured here establishes that any admissible
> `temp`/`eps` delivers it.** D stays below A–C on that basis and no other. Note this makes D
> *more* attractive than the original text implied — a directionally-correct term that is merely
> too weak is a better repair candidate than one pushing the wrong way — which is precisely why
> the erroneous reason had to be replaced rather than quietly dropped.

It should also be calibrated because it is owed, and because DP2 of T16 notes it must be
calibrated *jointly with* `param_prior`.

**E. Fix the box re-centring of the `default` init (separate defect, still open).** α 7.578×,
δ 3.885×. **Not** the ignition cause (`r2_nobox` is also 0/64), but no boxed number is
comparable to a legacy one until it is resolved. This is D-R2-1 as T16 originally labelled it.

---

## 7. What is NOT known — stated as plainly as the rest

- **Fix A is a prediction, and it was NOT TESTED.** This is the largest caveat in this document
  and it is stated first deliberately. The confirming pair was launched, sat **16 minutes queued
  behind the paper lane's `guarded_run` flock**, and was then **killed unrun** rather than left
  to land results after this report — the paper lane is deadline-critical and holds priority,
  and this task must not block it. `experiments/redesign_r2/ignition_diag/launch.log` records
  both the launch and the kill. **No `ignition_diag` training run exists**; the only output
  under that directory is the read-only `results/diagnosis.json`.

  The exact unrun command, so the next session can simply execute it:

  ```bash
  # baseline (CPU reproduction of 0/64)
  bash scripts/guarded_run.sh .venv/bin/python -u scripts/r2_ignition_run.py \
    --out experiments/redesign_r2/ignition_diag/staged --arms r2 --B 64 --steps 1500 \
    --device cpu --backend cubic --history-every 100
  # the mechanism test: `turing` live from step 1 (needs NO code change)
  bash scripts/guarded_run.sh .venv/bin/python -u scripts/r2_ignition_run.py \
    --out experiments/redesign_r2/ignition_diag/unstaged --arms r2 --B 64 --steps 1500 \
    --device cpu --backend cubic --history-every 100 \
    --staging-off-frac 0.0 --staging-ramp-frac 0.00067
  ```

  **Correction, recorded rather than silently patched.** This command was first registered with
  `--staging-ramp-frac 0.0`, which **crashes**: `losses/weighting.py:150-151` requires
  `0 < ramp_frac <= 1` and `r2_ignition_run.py:257-258` constructs `DataFirstStaging`
  unconditionally, so it raises `ValueError` before a single step runs. `0.00067` is the
  smallest clean value preserving the **zero-code-change** property: `turing` weight is 0.0 at
  step 0, 0.995 at step 1, 1.0 from step 2 (verified against `staging_factor` itself and
  recorded in `step5_supporting.staging_schedule.turing_live_alternative`). Making `turing` live
  *at* step 0 is impossible through the flags — with `off_frac=0` the factor is
  `(step−off)/(ramp·total)`, which is 0 at step 0 for any ramp — and would require relaxing the
  staging validation, i.e. a code change.

  **Registered prediction, written before any result exists:** the unstaged arm's coupling
  median at step 400 will be at least 10× above the staged arm's 0.0015, and its final coupling
  will exceed 0.0027. **Whether that produces any ignition at B=64 is genuinely unknown** — the
  a0 control ignites only 3/64, so B=64 has little power to resolve a small non-zero rate, and a
  null result there would be uninformative about fix A rather than evidence against it. Anyone
  running this for *ignition rate* rather than coupling should use the largest B affordable.
- **No claim that fix A alone is sufficient.** The decoupled network remains a true optimum of
  the objective (T16's D-R2-3). A changes the optimiser's path, not the landscape. B is the only
  candidate that changes the landscape.
- **The 99.32 % gradient share is a STEP-0 probe, not a per-step measurement.** §4's gradient
  decomposition is evaluated once, at the r2 initialisation. Its generalisation to the whole
  staging window rests on the monotone coupling trajectory of §3, not on re-measuring the share
  at each step — which was not done. The share almost certainly drifts as the parameters move;
  the claim it supports is the *direction and dominance at the start of the window*, which is
  where the collapse demonstrably begins.
- **`launch.log` records only the staged arm's launch line.** The unstaged arm never reached its
  own log line because the pair was killed while the first `guarded_run` was still queued, so the
  file shows one `### STAGED` header and the kill note — not two headers.
- **The `a0` control has no trajectory.** T16 recorded no parameter trace for either a0 cell, so
  the coupling *trajectory* comparisons in §3 have **no matched control** — only the a0
  endpoint (0.582) is available. Re-running a0 with telemetry is expensive for the known reason
  (the damped-Newton relaxation fallback, >13 s/step at B=256) and was not attempted here. This
  is the single biggest evidential gap in this document.
- **`r2_nobox` and `r2_noprior` recorded no invariant trace** (their `invariant_step` arrays are
  empty), so §3a's β/binding-budget table is available for the r2 arm only.
- **The CPU/CUDA question is untested.** All tracked runs are CUDA; the (unrun) confirmations were
  specified for CPU. A CPU reproduction of 0/64 would be a device-crossing reproduction, not a bit-identical
  one, and I have not established that the two agree.
- **`kstar_fft_rel_err` = 0.9769 is not a wavenumber measurement** and is not treated as one
  anywhere above. It is the k-grid floor constant `|k_min − k*_fft| / k*_fft` appearing in every
  arm including D5, as T16 §DP2(b) established. Only `r2_noprior`'s 0.0332 is a real k\*.
- **Nothing here is a statement about the legacy line.** "Turing-unstable" and "patterns" remain
  different claims, and this document establishes neither for the r2 arm — it establishes that
  the arm does not even reach Turing-instability (population-median σ_max ends at −0.031, with
  2 of 512 members marginally positive but failing the ignition criterion).
- **No threshold was introduced, weakened, or calibrated by this task.** The coupling metric is
  T16's, reused unchanged so the numbers stay comparable; no pass condition is attached to it.

---

## 8. Provenance

| item | value |
|---|---|
| runs read | `experiments/redesign_r2/phase1/`, `experiments/redesign_r2/phase1_ablation/` (`git_sha 1676d52`) |
| T16 status | report **reviewed and passed 2026-08-19**; numbers citable |
| analysis script | `scripts/r2_ignition_diag.py` (read-only; trains nothing; opens no `payload.h5`) |
| analysis output | `experiments/redesign_r2/ignition_diag/results/diagnosis.json` |
| confirming runs | **NONE — launched, never started, killed while queued. See §7** (`ignition_diag/launch.log`) |
| probe settings | B=512, seeds `recover._restart_seed(0, r)`, `kstar_obs` 0.28387817575592444, pin (0.41938, 0.75467, 0.35046), box α (0.5, 10.0) / δ (0.4, 5.0), CPU float64 |

`phase1_r2_B512`'s **throughput** provenance is disputed (`PROVENANCE.md` in that directory).
No throughput number is cited here — only its ignition, mechanism and trajectory arrays, which
that dispute does not touch.
