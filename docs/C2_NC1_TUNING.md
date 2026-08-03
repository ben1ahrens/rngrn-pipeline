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

## 4. No-op checks, per axis

*(filled in below as cells land)*
