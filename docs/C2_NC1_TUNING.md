# C2 — tuning the NON-COMPETITIVE (`nc1`) form against the pre-registered criteria

**Status: measurement log. Binds nothing.** `docs/PREREGISTRATION.md` §3 is the authority;
nothing here moves a bar. Every number below names the runs-root, the targets and the seeds
that produced it, and every runs-root is committed under `experiments/`.

**Form:** `nc1`. **Data:** `three_gene_qvar` `sample_0000` … `sample_0004` — the TUNING half
of §1a's split. `sample_0008` … `sample_0033` untouched. **Seeds:** K = 8 independent
`train.seed` values (0…7), post-`f78ae74`, so they are 8 independent draws and not the
pre-fix K-identical replicate.

**Compute shape, fixed for every cell** (`scripts/c2_cell.sh`, so no cell can omit it):
`-o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic
-o train.n_restarts=64 -o train.lbfgs_steps=0`, float64.

**Why `nc1` matters more than `competitive`.** Stage 0 measured the biological box's
strictly-Turing acceptance at **2.5e-5 for `nc1` against 4.5e-4 for `competitive`** over
80,000 box-constrained draws — ~18× harder — and §3 requires **both** forms to pass. `nc1`
is the binding form.

---

## 0. The cells

| root | what moves | steps | targets | K |
|---|---|---|---|---|
| `c2_A_base400` | — (reference) | 400 | 0000–0004 | 8 |
| `c2_B_detach` | `loss.detach_xstar=true` | 400 | 0000–0004 | 8 |
| `c2_C_dinit` | `model.d_init_from_kstar=true` | 400 | 0000–0004 | 8 |
| `c2_D_turing8` | `loss.weights.turing=8.0` | 400 | 0000–0004 | 8 |
| `c2_E_nostage` | `loss.staging_keys=[]` | 400 | 0000–0004 | 8 |
| `c2_F_priorOFF` | `loss.weights.param_prior=0.0` | 400 | 0000–0004 | 8 |
| `c2_H_hill3` | `model.n_hill=3` | 400 | 0000–0004 | 8 |
| `c2_G_steps2000` | `train.adam_steps=2000` | 2000 | 0000–0004 | 8 |

Every arm is paired with `c2_A_base400` on the **same seeds and the same targets**, moving
exactly one override. That pairing is also the no-op control (§4).

*(Superseded: `experiments/c2_base400_priorON`, the killed session's 2-target × 4-seed
baseline. Kept in the record; `c2_A_base400` replaces it at the pre-registered K.)*

---

## 1. The baseline, and what it says the problem is

`experiments/c2_A_base400`, K = 8 seeds (0…7), `sample_0000` and `sample_0001`, 400 Adam
steps, 64 restarts, prior ON, batched/CUDA/cubic. Scored readout
(`scripts/c2_summary.py`):

| target | K | rec | **tur** | topo | ndist | tv10 | tv4.8 | kfft | triv | plaus |
|---|---|---|---|---|---|---|---|---|---|---|
| `sample_0000` | 8 | 8 | **0** | 0.250 | 7 | – | – | 0.979 | 1.000 | 1.000 |
| `sample_0001` | 8 | 8 | **1** | 0.250 | 7 | 0.405 | 0.610 | 0.403 | 0.250 | 1.000 |

**Turing rate 1/16 = 0.0625**, against competitive's 0.125 at the same K and the library
benchmark's 0.3684 (legacy data). `recovered_frac` is 16/16 = 1.000 and means nothing:
every seed "recovers", almost none patterns. `trivial_kstar_err` 1.000 / 0.250 confirms no
`L` leak, so these are honest k\* numbers, unlike the legacy benchmark.

### 1.1 Criterion 3.1 fails on BOTH halves (`scripts/c2_repro.py`, same runs)

| | r=0.02 | r=0.05 | r=0.10 |
|---|---|---|---|
| within-target, K=8 (`sample_0000`) | 0.250 | 0.250 | 0.125 |
| within-target, K=8 (`sample_0001`) | 0.250 | 0.250 | 0.125 |
| within, size-matched to control (n=2) | 0.522 | 0.516 | 0.500 |
| **CROSS-target control** (n=2, 2000 draws) | 0.525 | 0.512 | 0.518 |
| **gap** | **−0.003** | **+0.004** | **−0.018** |

3.1 needs within(K=8) ≥ 0.75 **and** gap ≥ 0.25. It gets 0.250 and ≈ 0.000. The gap being
*zero to three decimals at every tolerance* is the criterion's designed-for failure mode
firing exactly as written: seeds fitted to **different** targets agree with each other as
much as seeds fitted to the **same** target, so what agreement there is comes from the
model's inductive bias, not from the data. The conclusion is insensitive to the
UNCALIBRATED `sign_zero_rtol`, which is what reporting three tolerances was for.

### 1.2 THE DIAGNOSIS: nc1's failure is TWO failures, and only one of them is D

`scripts/c2_jacobian_probe.py` holds each run's recovered **J** fixed and scans **D** over
400 draws (`free`, and `box` = the biologically-scored near-immobile construction).
Full table: `experiments/c2_probe_A_base400.txt`. It separates cleanly into two populations
with nothing in between:

| population | seeds | max Re eig(J) vs `sig_obs` | free draws Turing | box draws Turing |
|---|---|---|---|---|
| **(a) DEAD J** — no Turing at ANY D | `0000`: 0,1,2,3,6,7 · `0001`: 0,6,7 → **9/16** | **equal** (σ peaks at k=0) | **0/400**, best σ < 0 | 0/400 |
| **(b) LIVE J** — Turing-capable, wrong D | `0000`: 4,5 · `0001`: 1,2,3,4,5 → **7/16** | −0.09…−0.18 vs −0.02…−0.07 | 26–95/400, best σ **> 0** | 0–67/400 |

This is the single most decision-relevant number in the unit. **9 of 16 nc1 seeds land on a
reaction Jacobian that admits no Turing instability for any diffusion matrix whatsoever.**
For those seeds the D init, the D prior and the step budget are all irrelevant by
construction — no amount of tuning on D can rescue a J whose σ(k) is maximised at k = 0.
The remaining 7/16 are a genuinely different defect: J *is* Turing-capable and recovery
simply put D in the wrong place, and for 4 of those 7 a **biologically in-box** D exists
(11, 28, 67, 57 of 400 draws) that would make them Turing.

Two consequences, both pre-registered-criteria-relevant:
* an upper bound — **no D-side intervention can take this baseline past 7/16 = 0.44**, and
  the D-only arm (`c2_C_dinit`) is therefore predicted to move `sample_0000` barely at all
  (6 of its 8 seeds are dead-J) and `sample_0001` more (5 of 8 live-J). That prediction is
  registered here *before* the arm's numbers are read.
* the axes that can move the rate past that bound are the ones that change **which J
  recovery lands on**: the Turing weight, staging, the Hill exponent, `detach_xstar`.

### 1.3 The near-immobile third node is not an artefact — it is where the box D lives

Every `box` scan's best-σ draw came back with `D_lo/D_mid` between 0.0000 and 0.0017 while
the **scored** ratio `D_hi/D_mid` sat at 38–59, i.e. pinned near the top of `bio_box`'s
[1, 60] row. The winners' own recovered D agree: median `D_lo/D_mid` 0.194 (`sample_0000`)
and 0.098 (`sample_0001`) against raw max/min ratios of 38.6 and 77.3
(`scripts/c2_restarts.py`). So the mechanism the one prior nc1 run showed
(`plausibility_d_ratio_value` 2.32 against a raw ratio of 723.8) reproduces across seeds and
targets, and the D-scan says it is not merely *tolerated* by the biological box but is the
region of the box where Turing capability actually lives. See §5.

### 1.4 The sensitive readout, and why the scored count needs one

`scripts/c2_restarts.py`, pooling all 64 restarts × 8 seeds = 512 per target:

| root | target | restarts>0 | rate | σ p90 | σ best | winner σ | D_lo/D_mid | raw D ratio |
|---|---|---|---|---|---|---|---|---|
| `c2_A_base400` | `sample_0000` | 0/512 | 0.0000 | −0.0465 | −0.0257 | −0.0429 | 0.194 | 38.6 |
| `c2_A_base400` | `sample_0001` | 1/512 | 0.0020 | −0.0442 | +0.0090 | −0.0410 | 0.098 | 77.3 |

`n_pos = 0` on `sample_0000` (not merely 0 *scored*) means recovery never reaches the
regime there at all — it is **not** a selection defect where the loss-argmin rule discards
a Turing restart. Cells are ranked on σ p90 / n_pos whenever the scored Turing count is 0
on both arms, and that is stated wherever it is done.

---

## 2. A CORRECTION TO THE RECORD, made before anything is built on it

Commit `927a77a` describes `c2_L_t8k8` as *"a single override `-o loss.weights.turing=8.0`
from the committed baseline"*. **It is not.** The frozen config of every one of that cell's
eight runs carries **both** `loss.weights.turing: 8.0` **and** `loss.weights.kstar: 8.0`
(`experiments/c2_L_t8k8/runs/*/config/frozen_config.yaml`), which is what
`experiments/c2_queue_sweep2.txt` line 1 actually asked for. The turing-8-**alone** arm is
`c2_D_turing8`, where `kstar` stays at its base 1.0. The mis-description propagated into the
relaunch brief, so it is corrected here rather than quietly fixed.

The correction matters because reading the two arms side by side says something the single
number hid. `sample_0000`, K = 8, 400 steps, prior ON, identical compute shape:

| arm | turing | kstar | **turing seeds** | **kfft** (3.3 bar 0.083) | topo (3.1 bar 0.75) | plaus |
|---|---|---|---|---|---|---|
| `c2_A_base400` | 1 | 1 | **0/8** | 0.979 | 0.250 | 1.000 |
| `c2_D_turing8` | 8 | 1 | **6/8** | **4.765** | 0.125 | 1.000 |
| `c2_L_t8k8` | 8 | 8 | **6/8** | **0.115** | 0.250 | 1.000 |

The Turing weight buys the **rate** and pays for it in **wavelength**; the k\* weight buys
the wavelength back **at no measured cost in rate** — 6/8 in both arms, and
4.765 → 0.115 is a 41× move on the criterion `c2_D_turing8` was failing by 57×. The working
point is therefore the **pair**, not the Turing weight alone, and §3.3 becomes a
**shortfall of 1.4×** (0.115 vs 0.083) rather than a blowout. `c2_D_turing8` is not a
candidate configuration under §3.3 however good its rate looks.

---

## 5. CRITERION 3.1 — the mechanism (JOB B)

`scripts/c2_initbias.py`. §3.1 was failing with within-target ≈ cross-target and the doc
read that as *"what agreement there is comes from the model's inductive bias"*. That
reading is now **measured, and it is wrong in a specific and useful way.**

### 5.1 Two instrument defects, fixed before any conclusion

1. **The size-matched control was saturated.** `c2_repro.py` matches the control's group
   size to `min(K, n_targets)`, which with two targets in a root is **2**. At group size 2
   the modal fraction can only be 0.5 or 1.0 — its floor is **0.500** and it has almost no
   power, so "within 0.516 vs cross 0.512" is two numbers pinned to the same floor, not a
   measured equality. `c2_initbias.py compare` draws the cross-target group at the
   **pre-registered K = 8** with a cap of `ceil(K/n_targets)` seeds per target, keeping the
   group both size-matched and genuinely mixed.
2. **There was no zero.** The absolute floor of the K = 8 modal fraction is **1/8 = 0.125**
   (eight mutually distinct structures). Nothing in the record read the statistic against it.

### 5.2 The measurement, `c2_A_base400`, K = 8, `sample_0000` + `sample_0001`

| rtol | within K=8 | cross K=8 | gap | **init-only K=8** | floor |
|---|---|---|---|---|---|
| 0.02 | 0.250 | 0.243 | +0.007 | **0.733** | 0.125 |
| 0.05 | 0.250 | 0.215 | +0.035 | **0.265** | 0.125 |
| 0.10 | 0.125 | 0.183 | −0.058 | 0.185 | 0.125 |

`init-only` is the same statistic over **256 untrained models**, built with the identical
`RNGRN(N, form, seed=_restart_seed(model_seed, r))` construction `recover()` uses, solved
to their own steady state and Jacobian-probed. **No frame is read and nothing is trained**,
so it is agreement the data cannot be credited with.

### 5.3 What it says — and it is not what the record said

* **The trained model sits just above the ABSOLUTE FLOOR, not on an inductive-bias
  plateau.** 0.250 against 0.125 at K = 8: essentially the level of "one coincidental pair
  in eight". The gap being ~0 is therefore not *"the model reproduces its own bias"* — it
  is **there is almost no agreement of any kind, within or across.**
* **The init is 3× MORE topologically reproducible than the trained model.** 0.733 vs 0.250
  at rtol 0.02. The untrained init distribution is nearly deterministic in sign — the single
  structure `-+++-+++-` occupies 188/256 draws — and **400 Adam steps destroy that
  concentration**. Reproducibility is not something training fails to add; it is something
  training **removes**.
* **Training genuinely moves J** (`c2_initbias.py moved`, n = 40 runs across all three
  cells): `‖J_rec − J_init‖/‖J_init‖` = 0.63 … 1.80, and the recovered structure is
  identical to its own init's in **0/40 runs at every one of rtol 0.02/0.05/0.10**. So the
  H1 "training never leaves the init basin" story is **refuted**; what is left is H3 — the
  objective does not pin the sign structure, and each seed's descent ends somewhere
  different. σ(k) = max Re eig(J − k²D) is a scalar function of a 9-entry J plus 3
  diffusivities; it cannot be expected to identify 3⁹ sign patterns, and the measurement is
  consistent with it identifying essentially none of them.

### 5.4 Node relabeling is real bookkeeping and it is NOT the explanation

Nothing in the objective pins node order — σ(k) is invariant under `J → PJPᵀ, D → PDPᵀ` —
so quotienting by the 6 relabelings (`canon_key`, lexicographic minimum over `S₃`) is the
honest upper bound on how much of the shortfall is bookkeeping. On `c2_A_base400`:

| rtol | within (quotiented) | cross (quotiented) | gap |
|---|---|---|---|
| 0.02 | 0.375 | 0.395 | −0.020 |
| 0.05 | 0.312 | 0.324 | −0.011 |
| 0.10 | 0.312 | 0.324 | −0.012 |

The quotient lifts within-target 0.250 → 0.312 at rtol 0.05 — and lifts the **control by
the same amount**, 0.215 → 0.324, so the gap stays at zero and even turns slightly
negative. Relabeling is a genuine ~0.06 accounting artefact and it explains **none** of the
failure. Consistent with C1's legacy measurement (0.125 → 0.375, still short of 0.75).
**The pre-registered raw statistic at rtol 0.05 remains the number read against the bar.**

### 5.5 The one thing that did move the init, and why it is a lead not a result

The Jacobian-diagonal defect of `STATE_OF_THE_SCIENCE` §10 reproduces exactly on `nc1`:
**0 of 256** default inits has any positive Jacobian diagonal entry, against 88/88 real
`three_gene` systems that do — and patterning needs self-activation. `init='low_basal'`
takes that to **80/249**. It also *lowers* the init's topological concentration (rtol 0.02
K=8 modal 0.733 → 0.235), i.e. it buys J-diversity at the cost of the trivially-high init
agreement. **Not measured under training in this unit** — recorded as a lead, not a claim.

One thing worth flagging for whoever picks it up. `docs/DECISIONS.md` D9 rejected
`low_basal` as a default on two grounds: it failed 40/40 restarts under training on
`three_gene_val/sample_0000`, **and** its own Newton steady-state solve converged only
**255/400 = 64 %** even before any objective was added. That second ground no longer
reproduces here: on `nc1`, `low_basal` converges **249/256 = 97 %**. Two things changed
between D9 and now — the form (D9 measured `competitive`) and unit B3's **multistart**
steady-state globalisation. If B3 is what fixed it, D9's training-time failure deserves
re-measuring before the axis is treated as closed. That is a hypothesis with a named test,
not a result.

---

## 6. THE NEAR-IMMOBILE THIRD NODE IS SYSTEMATIC (JOB C)

`scripts/c2_immobile.py`. A bare D ratio has no scale, so "immobile" is defined physically:
a species is immobile **at the recovered pattern's own wavenumber** when
`q_i = D_i · k*_model² / |J_ii| ≪ 1`, i.e. its diffusive loss at k\* is a small correction
to its own diagonal reaction rate. **q is only read on Turing seeds** — when k\*_model
collapses to ~0 (baseline `sample_0000`: 0.00587 on 8/8 seeds) q → 0 for *every* species and
would measure the collapse, not the mechanism.

| root | target | Turing seeds | q<0.1 | q<0.01 | med q_min | med D_lo/D_mid | med raw D ratio | med **scored** d_ratio |
|---|---|---|---|---|---|---|---|---|
| `c2_A_base400` | `sample_0001` | 1/8 | 1/1 | 1/1 | 6.6e−3 | 0.098 | 77.3 | 7.58 |
| `c2_D_turing8` | `sample_0000` | 6/8 | 5/6 | 5/6 | 4.4e−3 | 0.0002 | 39 862 | 7.82 |
| `c2_D_turing8` | `sample_0001` | 8/8 | 8/8 | 5/8 | 6.7e−3 | 0.0001 | 113 821 | 7.78 |
| `c2_L_t8k8` | `sample_0000` | 6/8 | 3/6 | 0/6 | 8.1e−2 | 0.0195 | 566 | 8.54 |

**17 of 21 Turing seeds are below q = 0.1 and 11 of 21 below q = 0.01.** So yes — `nc1`
finds the immobile-third-node route systematically, and `plausibility_score = 1.0` in every
cell is being earned that way: the *scored* d_ratio sits at 7.50–8.54, on the prior centre
of 7.5, while the *raw* max/min ratio spans 38.6 to 113 821. That is not the scorer being
gamed by accident — Stage 0 measured the same construction giving a 17× acceptance gain in
the biological box and 127/127 generator systems staying strictly Turing when their slowest
diffuser is immobilised (`docs/BIO_VIABILITY.md`), which is Tica et al.'s mechanism for
relaxing differential diffusion.

**And it is the link between JOB A and JOB C.** The immobile node is *how* `turing=8` alone
bought its rate, and it is *why* that arm's wavelength blew up: D_lo/D_mid 0.0002 with a raw
ratio of 39 862 is a system whose k\* is set by a near-singular diffusion matrix. Adding
`kstar=8` suppresses the mechanism by two orders of magnitude (D_lo/D_mid 0.0002 → 0.0195,
raw ratio 39 862 → 566, med q_min 4.4e−3 → 8.1e−2) **at no cost in rate** (6/8 in both), and
that is exactly the 41× recovery of `kfft`. The near-immobile node earns its own section in
the paper, *and* it is the thing the k\* term has to be strong enough to police.

---

## 4. No-op checks, per axis

*(filled in below as cells land)*

`c2_initbias.py` is a NEW instrument and its axes are proven live rather than assumed:
`floor` distinguishes `init=default` (positive-diagonal 0/256) from `init=low_basal`
(80/249) and the two give different modal fractions at every rtol, so the `init` argument
is not being ignored; `moved` returns a different winning restart index per seed (0…63, **18**
distinct values over 40 runs), so the `_restart_seed(model_seed, r)` reconstruction tracks
the run rather than returning a constant.
