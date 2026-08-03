# C1 — tuning the COMPETITIVE form against the pre-registered criteria

**Form:** `competitive`. **Dataset:** `three_gene_qvar` (PRIMARY).
**Tuning targets (burned, per `PREREGISTRATION.md` §1a):** `sample_0000` … `sample_0004`.
**Held out and untouched by this unit:** `sample_0008` … `sample_0033`.
**Seeds:** K = 8 independent `train.seed` values (0…7), post-`f78ae74`, per §2.
**Shared execution block, never varied between cells:**

```
-o data.dataset_id=three_gene_qvar -o train.batched=true -o train.device=cuda \
-o model.dispersion_backend=cubic -o train.n_restarts=64 -o train.lbfgs_steps=0
```

This document applies **no** pass/fail of its own. The bars live in `PREREGISTRATION.md`
§3 and are read against them in priority order: **reproducibility (3.1) → robustness
(3.2) → pattern (3.3) → viability (3.4)**, never by loss.

---

## 0. What the previous attempt at this unit actually left behind

The commit message on `570f3c8` states *"Measured so far: baseline and steps2000 cells,
sample_0000 only, K=8 seeds."* **That is not what is on disk.** Both
`experiments/tune_comp_reports/{baseline,steps2000}/sample_0000.json` are **zero bytes**,
and all six `experiments/tune_comp/runs/*/` directories committed alongside them have an
**empty `results/`** — only `config/frozen_config.yaml` was ever written. The shell
redirect `> $OUT/$S.json` created the file at launch; the session exit killed the process
before any report was produced.

**No measurement from the previous attempt survived, and none is reused here.** Every
number below was produced by this session. The `model.init` no-op fix in `src/rngrn/train.py`
and its test *are* real and are inherited. This is recorded because a commit message
asserting results that do not exist is exactly the provenance defect this project keeps
finding, and it would otherwise be inherited silently.

---

## 1. No-op proof — every swept axis, on BOTH execution paths

**Binding requirement, not a formality.** Four silent no-ops have been found in this
codebase (`model.seed` pinned in `base.yaml` making `train.seed` inert; `losses/total.py`
never calling `terms.param_prior`; `fit()` never passing `cfg.model.init` to `recover()`;
the pre-push hook testing another worktree's source). A null result from an inert knob is
an artefact, so no axis is believed until it is shown to move something.

**Method** (`scripts/tune_comp_noop_check.py`): two runs at the **same `train.seed`**
differing **only** in the knob, on the same target, through **`rngrn.train.fit`** — the
config/CLI entry point the tuning cells use, *not* a direct `recover()` call, because
no-op 3 was precisely a defect of the config path that a direct call would have hidden.
The compared signature is the final loss, the recovered `J`, the recovered `D_model` and
`kstar_model`. Bit-identical ⇒ INERT.

Run twice, because the measurement cells run the **batched** assembler
(`losses/total.py::compute_terms_batched`) while the cheap check is natural on the serial
one, and a knob can be live in one and inert in the other:

| axis | override tested | serial path | **batched path** |
|---|---|---|---|
| `loss.detach_xstar` | `=true` | LIVE | **LIVE** |
| `model.d_init_from_kstar` | `=true` | LIVE | **LIVE** |
| `loss.weights.param_prior` | `=1.0` | LIVE | **LIVE** |
| `loss.weights.anchor` | `=0.5` | LIVE | **LIVE** |
| `model.init` | `=low_basal` | LIVE | **LIVE** |
| `loss.staging_off_frac` | `=0.05` | LIVE | **LIVE** |
| `loss.staging_ramp_frac` | `=0.05` | LIVE | **LIVE** |
| `train.adam_steps` | `=31` | LIVE | **LIVE** |

`sample_0000`, `train.seed=0`, `dispersion_backend=cubic`, `lbfgs_steps=0`, CPU float64;
serial arm `n_restarts=3, adam_steps=25`, batched arm `n_restarts=8, adam_steps=25`.
Run records under `experiments/runs/` (the check does not honour a `runs_root` override —
noted rather than hidden).

**All eight axes are LIVE on both paths.** `model.init=low_basal` is live on the config
path *only because of this branch's fix*; before it, that row would have read INERT.


---

## 2. The criteria table — pre-registered per cell, BEFORE the cells were run

Every cell is K = 8 independent seeds (0…7) on the stated targets, differing from
`baseline` by **exactly one** override. Ranked in `PREREGISTRATION.md` §3 priority order:

| rank | criterion | statistic | bar |
|---|---|---|---|
| 1 | 3.1 reproducibility | `topology_consistency` (= modal fraction over all 8 seeds) | ≥ **0.75** |
| 1 | 3.1 control | within-target *minus* cross-target `mean_agreement` | ≥ **0.25** |
| 2 | 3.2 robustness | median `turing_volume_10pct` / `turing_volume_4p8pct` | ≥ **0.90** / ≥ **0.95** |
| 3 | 3.3 pattern | `morphology_match`, `kstar_fft_rel_err`, beside `trivial_kstar_err` | err ≤ **0.083** |
| 4 | 3.4 viability | `plausibility_score` | = **1.0** |
| — | §4 rate | `recovered_frac`, `turing_frac` | reported regardless |

**Target choice, and why it is not the obvious one.** `BIO_VIABILITY.md` §4.2 measured, at
seed 0 / 400 steps, that `sample_0000/0001/0002` do **not** pattern while `sample_0003/0004`
do. Tuning on two non-patterning targets would leave criteria 3.2–3.4 uncomputable (they
are defined only over Turing-reaching seeds), and tuning on two patterning ones would hide
the binding problem. Each cell therefore runs **`sample_0000` (hard) and `sample_0003`
(patterns)** — one target that exposes the Turing-rate problem and one on which the
downstream criteria can actually be read. Both are inside the §1a TUNING range.

**A seed that fails to reach the Turing regime stays in the denominator** of
`topology_consistency`, per §2. It is not dropped.

---

## 3. Cell: `baseline` — the control every other cell is read against

Library defaults: `detach_xstar=false`, `param_prior=0.0`, `anchor=2.0`,
`d_init_from_kstar=false`, `init=default`, `staging_off/ramp=0.25/0.25`, 400 Adam steps.

### 3.1 `sample_0000` (K=8, `git_sha 570f3c8`, `config_id 30105ae5c671`)

| statistic | value |
|---|---|
| `recovered_frac` | **1.000** (8/8 scored, 0 raised) |
| `turing_frac` | **0.125** — 1 of 8 seeds (seed 3 only) |
| `topology_consistency` | **0.125** |
| `mean_agreement` | 0.579 |
| `turing_volume_10pct` / `4p8pct` (median) | 1.000 / 1.000 — **n = 1 seed** |
| `kstar_fft_rel_err` (median) | 0.979 |
| `trivial_kstar_err` | **1.000** — not leaked data |
| `plausibility_score` (mean) | 0.458 |

Per seed:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Turing | – | – | – | **yes** | – | – | – | – |
| `kstar_fft_rel_err` | 0.979 | 0.979 | 0.979 | **0.219** | 0.979 | 0.979 | 0.979 | 0.979 |
| `plausibility_score` | 0.33 | 0.67 | 0.67 | 0.33 | 0.67 | 0.33 | 0.33 | 0.33 |

**Read this against the library benchmark of `turing_frac` 0.3684** (40 seeds × 400 steps),
not against zero: on this target the competitive form is at **0.125**, a third of it.

**The seven non-Turing seeds share `kstar_fft_rel_err` = 0.979 to three digits.** That is
not scatter — it is one degenerate attractor reached from seven independent inits, the
`k* → 0` collapse `BIO_VIABILITY.md` §4.3 part 3 identified as a σ(k) monotone decreasing
and everywhere negative. `recovered_frac` = 1.0 therefore **overstates** success badly:
every seed converged, and seven converged to the same non-pattern.

**`topology_consistency` = 0.125 means all 8 seeds returned a DIFFERENT sign structure**
(modal fraction 1/8). Criterion 3.1 — the primary one — fails on this target at the
baseline by a wide margin, and it fails because the seeds that do not pattern have no
shared topology to agree on. Reproducibility and the Turing rate are not independent
problems here; the first is downstream of the second.

### 3.2 The SENSITIVE readout on the same 8 runs — the failure is REACHABILITY, not selection

`target-report` scores the **one** restart the loss-argmin returns, so `turing_frac` = 0.125
is a statistic with K = 8 samples behind it. Every run record also stores all
`n_restarts = 64` restarts with their own `sig_max_pos` — the max of σ(k) over the
instability hinge's support, i.e. exactly the quantity that must go positive for a Turing
regime — so pooling gives **512** samples instead of 8 (C2's instrument, imported here as
`scripts/c1_restarts.py`; run via `scripts/c1_analyse.py`).

| cell | target | K | pooled restarts | `sig_max_pos > 0` | rate | p90 | best | winner's `sig_max` (median) | `D_lo/D_mid` (median) |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | `sample_0000` | 8 | 512 | **2** | **0.0039** | −0.0413 | +0.1869 | −0.0304 | 0.512 |

**Two of five hundred and twelve.** `n_pos > 0` alongside a scored `turing_frac` of 0.125
means the selection rule is not the problem — it found one of the two — and the p90 at
−0.041 means the *typical* restart is not near the boundary either. **The Turing regime is
essentially never reached from a default init**, which is a different defect from "reached
and then discarded", and it has a different fix.

`STATE_OF_THE_SCIENCE.md` §10 has that root cause already measured, independently of this
unit: **the RNGRN's Jacobian diagonal is negative at every default init** — 0 of 200
converged inits had any positive diagonal (mean ≈ −0.63, max −0.27), against 88/88 real
`three_gene` systems that do — and Turing patterning requires self-activation. The same
section measures a low-basal init taking Turing-unstable inits from **0 % to 82 %**.

---

## 4. The cells, pre-registered here BEFORE they were run

Every cell below is K = 8 independent seeds on the stated target, differing from `baseline`
by **exactly one** override, on the shared execution block at the top of this document. The
ranking rule is §2's table and is not restated per cell. The queue that runs them —
`scripts/c1_queue2.sh` — is launched **detached** (`setsid nohup`) so that a session exit
costs polling, not compute; four runs across three sessions have now been lost to session
exits. A report is written only on `rc = 0` **and** non-empty, via a `.tmp` + `mv`, so a
killed process cannot leave a zero-byte file that a later reader takes for a measurement.

| # | cell | the single override | why it is where it is |
|---|---|---|---|
| 1 | `detach` | `loss.detach_xstar=true` | the largest untested difference between the library path and the exp05 experiment that measured `turing_frac` 0.3684; a gradient path through x\* lets the optimiser flatten σ(k), which is the k\*→0 attractor's mechanism |
| 2 | `lowbasal` | `model.init=low_basal` | §3.2's 2/512, plus `STATE_OF_THE_SCIENCE.md` §10's measured 0 %→82 % on the identical statistic |
| 3 | `legacy_control` | *(none — `three_gene_val`)* | one-off CONTROL, §5 |
| 4 | `steps2000` / `steps8000` | `train.adam_steps` | the curve, 2 seeds × 2 targets only |
| 5 | `dinit` | `model.d_init_from_kstar=true` | the dimensional D init starts ~29× too small |
| 6 | `prior0p3` | `loss.weights.param_prior=0.3` | a real axis for the first time; Stage 0 measured weight 1.0 buying `plausibility 1.0` at a k\* cost of 0.017→0.087 |
| 7 | `anchor4` / `anchor0p5` | `loss.weights.anchor` | 2.0 is inherited and was never swept |
| 8 | `staging005` | `loss.staging_off_frac` / `_ramp_frac` | both 0.25, inherited, never swept |

**`lowbasal` was promoted from position 6 to position 2 after §3.2 was measured, and this
is recorded rather than presented as the original plan.** The unit's brief lists
`detach_xstar` first and `model.init` sixth. §3.2 is new evidence — measured on the
already-committed baseline runs, not on any cell — that the binding failure is reachability
from the init, and §10 names the init as its measured cause. Reordering a queue changes no
threshold and drops no cell; the pre-registered bars in `PREREGISTRATION.md` §3 are
untouched.

## 5. The legacy control, and what it can and cannot say

The baseline's 0.125 is read against a library benchmark of 0.3684 — but that benchmark was
measured on **legacy** data, where `L = clip(6·2π/k*, 18, 220)` makes an image-blind
predictor score 0.0 % median k\* error. Without a control, 0.125 vs 0.3684 cannot separate
"`three_gene_qvar` is genuinely harder" from "something regressed since the benchmark".
`legacy_control` therefore runs the **baseline configuration unchanged** on
`three_gene_val/sample_0000`, K = 8, into a **separate runs-root**
(`experiments/tune_comp_legacy`).

**It is a CONTROL and nothing else.** `PREREGISTRATION.md` §1 forbids legacy data from
supporting any k\* claim, and its numbers are never pooled with `three_gene_qvar` numbers
anywhere in this document.

---

## 6. Cell `detach` — `loss.detach_xstar=true`. **The hypothesis does NOT hold.**

`sample_0000`, K = 8 (seeds 0…7), `git_sha a71efae`, `config_id 2bfcb176aafc`, 1275 s.
One override from `baseline` (`config_id 30105ae5c671`); everything else identical.

**The hypothesis under test.** `scripts/exp05` detached x\* for the dispersion-side terms
and measured `turing_frac` 0.3684; every library run has had it off, and the two had never
been compared. `STATE_OF_THE_SCIENCE.md` §2.4 traces the k\*-anchor's degenerate minimum:
while the model is non-Turing, σ(k) peaks at k = 0 and the anchor loss can be reduced by
**flattening σ** rather than relocating its peak. A gradient path through x\* into the
dispersion terms gives the optimiser one more way to do that flattening, so detaching it
was a live mechanistic explanation for the k\*→0 attractor.

### 6.1 The numbers, against the baseline control

| statistic | `baseline` | `detach` | |
|---|---|---|---|
| **`turing_frac`** | **0.125** (seed 3) | **0.125** (seed 3) | **no change** |
| pooled restarts with `sig_max_pos > 0` | **2 / 512** | **2 / 512** | **no change** |
| pooled `sig_max_pos` p90 | −0.0413 | −0.0430 | slightly worse |
| pooled `sig_max_pos` best | +0.1869 | +0.2068 | |
| `recovered_frac` | 1.000 (0 raised) | 1.000 (0 raised) | |
| `topology_consistency` | 0.125 | 0.250 | 2 of 8 seeds vs 1 of 8 |
| `mean_agreement` | 0.579 | 0.464 | **down** |
| `n_distinct_structures` | 8 | 7 | |
| `kstar_fft_rel_err` median | 0.979 | 0.979 | |
| `trivial_kstar_err` | 1.000 | 1.000 | not leaked |
| `plausibility_score` mean | 0.458 | 0.500 | |
| `turing_volume_10pct` / `4p8pct` median | 1.000 / 1.000 (**n = 1**) | 1.000 / 1.000 (**n = 1**) | |

### 6.2 The decisive detail

The seven non-Turing seeds return `kstar_model = 0.00587` **to five significant figures in
both arms**, and the same seed (3) is the only Turing seed in both. The runs are genuinely
different — seed 3's `kstar_model` moves 0.22024 → 0.22511, `D_lo/D_mid` 0.512 → 0.463, and
`config_id` differs — but the degenerate attractor is reached *identically* with and without
a gradient path through x\*.

**So the answer to the question this unit was asked to settle is: it does not hold.**
Detaching x\* does not lift the Turing rate, does not change the pooled reachability rate
(2/512 either way), and does not perturb the k\*→0 fixed point at all. Whatever produced
exp05's 0.3684, `detach_xstar` is not it. The direction of the p90 (−0.0413 → −0.0430) is
if anything mildly against detaching.

### 6.3 What moved, and why none of it is a win

`topology_consistency` 0.125 → 0.250 is **two seeds agreeing instead of one**, out of eight,
on a criterion whose bar is 0.75. On K = 8 that is one seed of movement and is not
distinguishable from noise; `mean_agreement` moves the *other* way (0.579 → 0.464), which is
what one expects when the change is noise rather than signal. `morphology_match_frac`
0.0 → 1.0 is **n = 1 compared seed** in both arms (morphology is scored `compared` only on
Turing-reaching seeds, of which there is one), and `morphology_distance` 1.259 → 1.108 —
a single seed flipping a boolean. Neither is reportable as an effect.

**Criterion 3.1 fails in both arms** (0.125 and 0.250 against 0.75), and 3.2's medians rest
on `robustness_n_used = 1` in both, so they are not yet measurements of anything.

### 6.4 The consequence for the unit

The reachability reading in §3.2 stands and is now measured on two independent
configurations: **4 of 1024 restarts** across `baseline` and `detach` reach `sig_max_pos > 0`
on `sample_0000`. The remaining rate axes must move the *init* or the *parameterisation*,
not the gradient path — which is where `lowbasal` and `d_init_from_kstar` sit, and is why
`lowbasal` was promoted before this cell finished.

---

## 7. Cell `legacy_control` — the control that reframes the unit

Baseline configuration **unchanged**, `three_gene_val/sample_0000`, K = 8, `git_sha 4246d6f`,
`config_id d3aab01611ae`, separate runs-root `experiments/tune_comp_legacy`, **370 s**.
Per `PREREGISTRATION.md` §1 this supports **no claim**; it is read only against the
`three_gene_qvar` baseline as a control, and its numbers are never pooled with qvar's.

| statistic | `baseline` (qvar `sample_0000`) | **`legacy_control`** (`three_gene_val`) |
|---|---|---|
| `turing_frac` | 0.125 (1/8) | **1.000 (8/8)** |
| pooled restarts `sig_max_pos > 0` | **2 / 512** (0.0039) | **174 / 512 (0.3398)** |
| pooled `sig_max_pos` p90 | −0.0413 | **+0.2217** |
| winner `sig_max` median | −0.0304 | **+0.2887** |
| `robustness_n_used` | 1 | **8** |
| `turing_volume_10pct` / `4p8pct` median | 1.000 / 1.000 (n=1) | 0.992 / 1.000 (n=8) |
| `kstar_fft_rel_err` median | 0.979 | 0.068 |
| **`trivial_kstar_err`** | **1.000** | **1.3e-16 — LEAKED** |
| `kstar_spread` | 2.170 | **0.0147** |
| **`topology_consistency`** | **0.125** | **0.125** |
| `mean_agreement` | 0.579 | 0.397 |
| `n_distinct_structures` | 8 | **8** |
| wall clock, same settings | 1275 s | **370 s** |

### 7.1 Nothing regressed. `three_gene_qvar` is harder by ~87× in reachability.

Same code, same config, same K, same 400 steps, same 64 restarts: **8 of 8** seeds reach the
Turing regime on the legacy sample against **1 of 8** on the qvar one, and at the restart
level **174/512 against 2/512**. That is the answer the control was run for. The 0.125 is
not a regression against the 0.3684 library benchmark — on legacy data this configuration
now *exceeds* that benchmark — it is a property of `three_gene_qvar`.

And `trivial_kstar_err = 1.3e-16` on the legacy sample is the documented leak, measured
here rather than assumed: an image-blind predictor using `L` alone is **exact** on this
target, so the legacy `kstar_fft_rel_err` of 0.068 carries no information about the model
and is not comparable to the qvar number beside it. This is exactly why §1 forbids legacy
k\* claims, and it is why the 0.3684 benchmark could never have been a fair target.

### 7.2 THE FINDING: criterion 3.1 is NOT downstream of the Turing rate

The unit's brief states that "reproducibility is downstream of the Turing rate — lift the
Turing rate and 3.1 follows". **That is refuted by this control.** On the legacy sample:

* every one of 8 seeds reaches the Turing regime (`turing_frac` 1.000),
* all 8 agree on k\* to 1.5 % (`kstar_spread` 0.0147 on a k\* of ~0.41),
* and `topology_consistency` is still **0.125**, with **8 distinct sign structures** — the
  worst possible value, identical to the value on the target where 7 of 8 seeds collapse to
  k\*→0.

Eight seeds that all pattern, all agree on the wavelength, and no two of which recover the
same network. **The Turing rate and the reproducibility of J are separate failures**, and
lifting the first will not lift the second. Anything in this project that reads
`turing_frac` as a proxy for 3.1 is reading it wrong.

### 7.3 How much of that is the node-label symmetry — a diagnostic, not a criterion

Nothing in the objective pins the node ordering: the loss is a function of σ(k), which is
invariant under relabeling the three nodes, so a seed may recover the *same* network with
its nodes permuted. `topology_consistency` compares sign structures entrywise and therefore
counts those as different structures. With N = 3 there are 6 relabelings.

Quotienting by them (`scripts/c1_analyse.py`, PERM block — **diagnostic only**):

| cell | target | raw @0.02 | canon @0.02 | raw @0.05 | canon @0.05 | raw @0.10 | canon @0.10 |
|---|---|---|---|---|---|---|---|
| `baseline` | qvar `sample_0000` | 0.250 | 0.625 | **0.125** | 0.125 | 0.250 | 0.500 |
| `detach` | qvar `sample_0000` | 0.125 | 0.250 | **0.250** | 0.375 | 0.250 | 0.375 |
| `legacy_control` | `three_gene_val` | 0.125 | 0.250 | **0.125** | 0.375 | 0.250 | 0.500 |

**The pre-registered statistic is the raw column at rtol 0.05 and it stays the number read
against the 0.75 bar.** The canonical column is reported because it says how much of the
failure is relabeling and how much is a genuinely different network — and the answer is
*some, but nowhere near enough*: the best canonical value anywhere in this table is 0.625,
and 3.1 fails on every row of both columns. The symmetry is a real confound in the
estimator and is worth recording for the paper; it is **not** a rescue, and no bar moves.

### 7.4 The cost profile, and the first evidence on the C2 cost anomaly

At **identical** settings the legacy sample took **370 s** and the qvar sample **1275 s** —
a **3.4×** spread, on the same GPU, same 8 seeds, same 4 workers. The cheap target is the
one where all 8 seeds pattern; the expensive one is the one where 7 of 8 collapse to k\*→0.
That is the direction the unit brief flagged as significant: the expensive targets are the
non-patterning ones, so the cost anomaly and the Turing-rate problem look like one problem —
the steady-state solve grinding on a flattening, near-singular Jacobian. Two targets is not
a measurement of that; `sample_0003` (which patterns) is the next data point.

---

## 8. JOB B — criterion 3.1 as its own problem: what the objective can and cannot see in J

§7.2 established, from the legacy control, that **3.1 is not downstream of the Turing
rate**: 8 of 8 seeds pattern, all agree on k\* to 1.5 %, and `topology_consistency` is
still 0.125 with 8 distinct sign structures. This section asks the next question directly —
*which directions in J does the objective actually constrain?* — and answers it with a
proof plus a measurement. Instrument: `scripts/c1_gauge.py`.

### 8.1 The objective sees J only through σ(k), and that has exact consequences

Every J-dependent term in the trained objective is one of three:

| term | how it sees J |
|---|---|
| `losses/terms.py::kstar_anchor` | only via σ(k) = max Re eig(J − k²D) |
| `losses/terms.py::turing_hinges_split` | only via σ(k) |
| `losses/terms.py::anticollapse` | `softplus_hinge(jac_floor − ‖J‖_F)`, `jac_floor` = 1.0 |

`frame_scale_anchor` sees x\* only, `param_prior` sees D/α/δ only, and `resid` has weight 0
and is not even computed on the batched path (`losses/total.py::compute_terms_batched`).

The `anticollapse` term is a **one-sided** hinge, so it is exactly flat with exactly zero
gradient everywhere ‖J‖_F > 1 — and every recovered network measured in this unit sits
there (‖J‖_F = 1.54–2.63 across all 24 committed runs). **On the region the optimiser
actually occupies, the objective is a function of σ(k) alone.**

Because **D is diagonal**, three transformations then leave σ(k) unchanged at every k:

1. **Transpose.** (J − k²D)ᵀ = Jᵀ − k²D, and a matrix and its transpose are isospectral.
   *The objective cannot distinguish a network from the one with every regulatory edge
   reversed.*
2. **Diagonal similarity.** For S = diag(s), s > 0: S(J − k²D)S⁻¹ = SJS⁻¹ − k²D, since a
   diagonal S commutes with a diagonal D. So J_ij → (s_i/s_j)·J_ij is invisible — an
   (N−1)-parameter **continuous** gauge freedom, 2-dimensional at N = 3.
3. **Node permutation** (with D permuted too) — the symmetry §7.3's PERM block measured.

A dimension count says these are the *whole* of the blindness, not merely some of it: σ(k)
for all k fixes the characteristic polynomial of J − k²D, whose coefficients depend on J
through 7 independent functions (the 3 diagonal entries, the 3 2×2 principal minors, and
det J). 9 entries − 7 constraints = **2**, exactly the dimension of the diagonal-similarity
group modulo its scalar centre.

**Two load-bearing premises checked against the code rather than assumed.** First, D really
is diagonal in the dispersion: `model.py::dispersion` builds
`M = J.unsqueeze(0) - k2 * torch.diag_embed(Dvec).unsqueeze(0)`, so the commutation argument
in (b) is exact and not an approximation of an off-diagonal D. Second, the dimension count's
"7 independent functions" is not this document's algebra imposed on the code — it is the
code's own reduction. `model.py::_sigma_max_cubic`, the backend every cell in this unit runs
(`dispersion_backend=cubic`), computes σ from exactly *"the characteristic polynomial of a
3x3 … whose coefficients (trace, sum of principal 2x2 minors, determinant)"* are formed from
M. The optimiser's entire view of J is those three coefficient families evaluated along the
k-grid, which is where the 7 comes from.

**This is verified numerically, not asserted** (`c1_gauge.py` BLOCK 1), on *real* recovered
(J, D) from the committed `legacy_control` runs — max |Δσ(k)| over the k-grid:

| seed | ‖J‖_F | transpose | diagonal similarity | permutation | σ scale |
|---|---|---|---|---|---|
| 0 | 2.038 | 6.66e−16 | 1.11e−15 | 1.93e−14 | 0.5080 |
| 1 | 2.504 | 4.53e−14 | 2.04e−14 | 4.75e−14 | 0.7823 |
| 2 | 2.099 | 5.05e−15 | 4.33e−15 | 3.94e−15 | 0.5706 |

Exact to machine precision against a σ scale of order 0.5–0.8.

### 8.2 But the gauge is NOT what is breaking 3.1 — a clean negative result

If the seeds were landing on different points of one gauge orbit, quotienting by the group
would collapse them onto one structure. **It does not.** The consistency ladder
(`c1_gauge.py` BLOCK 2), at the pre-registered rtol 0.05:

| cell | target | **raw (= criterion 3.1)** | +perm | +perm,transpose | +perm,transpose,balance |
|---|---|---|---|---|---|
| `legacy_control` | `three_gene_val/sample_0000` | **0.125** | 0.375 | 0.375 | 0.375 |
| `baseline` | qvar `sample_0000` | **0.125** | 0.125 | 0.125 | 0.375 |
| `detach` | qvar `sample_0000` | **0.250** | 0.375 | 0.375 | 0.250 |

Adding the transpose quotient moves **nothing** on any row. Adding the diagonal-similarity
gauge fixing (`scipy.linalg.matrix_balance`, Osborne) moves one row up, one row down, and
one not at all. **The exact symmetries proven in §8.1 are real and are not the explanation.**
The best value anywhere in the table is 0.375 against a bar of 0.75.

This is worth stating plainly because the opposite conclusion was the natural guess and it
is wrong: the seeds have not found the same network in different gauges.

### 8.3 What IS breaking it: the statistic's threshold sits at the median of the entries

`topology_consistency` calls J_ij an edge iff |J_ij| ≥ rtol·max|J|, a **hard** cut, then
compares sign matrices entrywise. `c1_gauge.py` BLOCK 3 measures where that cut lands
(`|J_ij|/max|J|` pooled over all K seeds):

| cell | target | q25 | **q50** | q75 | within a factor 3 of the cut | `topology_consistency` min→max over rtol ∈ [0.005, 0.5] |
|---|---|---|---|---|---|---|
| `baseline` | qvar `sample_0000` | 0.0107 | **0.0512** | 0.5277 | **0.333** (24/72) | 0.125 → 0.500 |
| `detach` | qvar `sample_0000` | 0.0089 | **0.0730** | 0.5597 | **0.361** (26/72) | 0.125 → 0.250 |
| `legacy_control` | `three_gene_val` | 0.0264 | 0.3143 | 0.7716 | 0.153 (11/72) | 0.125 → 0.250 |

**On `baseline` the pre-registered rtol of 0.05 sits on the median entry magnitude
(0.0512).** That is the single worst place a hard threshold can sit: it maximises the
number of entries whose edge/no-edge assignment is decided by numerical noise rather than
by dynamics. A third of all entries are within a factor of 3 of the cut.

And the criterion is correspondingly unstable in the tolerance. Recomputed across the sweep
0.005 / 0.01 / 0.02 / 0.03 / 0.05 / 0.08 / 0.10 / 0.15 / 0.20 / 0.30 / 0.50:

```
baseline        sample_0000   0.375 0.375 0.250 0.250 0.125 0.250 0.250 0.250 0.250 0.250 0.500
detach          sample_0000   0.250 0.125 0.125 0.125 0.250 0.250 0.250 0.250 0.250 0.125 0.250
legacy_control  sample_0000   0.250 0.125 0.125 0.125 0.125 0.250 0.250 0.125 0.250 0.125 0.250
```

Non-monotone, with no trend, spanning 0.125–0.500 — and the pre-registered 0.05 happens to
be at or near the **minimum** of `baseline`'s curve. §3.1 of `PREREGISTRATION.md` flagged
this tolerance UNCALIBRATED in advance and required 0.02/0.05/0.10 to be reported; this is
the measurement of *how* uncalibrated, and it is why the reported shortfall should be read
as "3.1 fails, and the estimator is also noisy", not as a precise distance from the bar.

**No bar moves and nothing here is a rescue.** 3.1 fails at every tolerance in the sweep on
every cell measured. The finding is that a material part of the *variance* in the number is
the estimator, which is a defect worth fixing in a future instrument and worth stating in
the paper — not a reason to reinterpret a failing measurement as a pass.

### 8.4 The gauge-invariant content, which is what σ(k) can actually pin

Given §8.1, the natural gauge-invariant summary of J at N = 3 is the sign of the diagonal
entries (self-activation/repression, 3 bits), the sign of the 2-cycle products J_ij·J_ji
(the feedback *sense* of each pair, 3 bits), and the sign of the 3-cycle product (1 bit) —
**7 sign bits, exactly the 7 functions of J that σ(k) fixes**, against the raw statistic's
9 entrywise signs. `c1_gauge.py` BLOCK 4, modal fraction over K = 8:

| cell | target | diag | pairs | 3-cycle | all 7 |
|---|---|---|---|---|---|
| `baseline` | qvar `sample_0000` | 0.875 | 0.500 | 0.750 | 0.500 |
| `detach` | qvar `sample_0000` | 0.875 | 0.500 | 1.000 | 0.500 |
| `legacy_control` | `three_gene_val` | **1.000** | 0.500 | 0.750 | 0.375 |

Seven of eight `baseline` seeds, and **eight of eight** `legacy_control` seeds, agree on the
sign pattern of the Jacobian diagonal — the property `STATE_OF_THE_SCIENCE.md` §10 identifies
as the binding one for patterning — while the pre-registered entrywise statistic scores the
same eight runs as eight distinct networks.

Note the *shape* of the failure, which splitting the statistic into columns makes visible
and a single 9-bit number cannot: agreement is **concentrated in the diagonal**
(0.875–1.000) and collapses on the off-diagonal pair senses (0.500 on all three cells).
What the seeds reproduce is *which nodes self-activate*; what they do not reproduce is
*who regulates whom*. The joint 7-bit column is 0.375–0.500 — still far below 0.75, so this
is not a criterion in disguise — but it localises the disagreement to the off-diagonal for
the first time, and that is the direction a future 3.1 fix has to attack.

**This is a DIAGNOSTIC and is not criterion 3.1**, and it is deliberately reported with its
limits attached: it is *coarser* than the pre-registered statistic (7 bits not 9, and node
identity is quotiented out), so a higher number is expected on granularity alone and the
comparison is not like-for-like. A statistic this coarse can also agree by the model's
inductive bias rather than from the target, which is exactly what §3.1's control exists to
catch — so **it means nothing until read against the size-matched cross-target control**,
which requires ≥ 2 targets in the same runs-root and is measured in §9.

### 8.5 The one J property that does separate patterning, measured on all 24 committed runs

`STATE_OF_THE_SCIENCE.md` §10 measured the Jacobian diagonal at **init**: negative at every
one of 200 converged default inits, against 88/88 real `three_gene` systems that have a
positive one. The same quantity is available at the **recovered solution** for every run
this unit has committed, and it separates the outcome perfectly:

| cell | target | seeds with max diag(J) > 0 | seeds scored Turing |
|---|---|---|---|
| `baseline` | qvar `sample_0000` | 1/8 (seed 3, +0.2276) | 1/8 (seed 3) |
| `detach` | qvar `sample_0000` | 1/8 (seed 3, +0.2535) | 1/8 (seed 3) |
| `legacy_control` | `three_gene_val` | **8/8** (+0.334 … +0.499) | **8/8** |

Pooled 2×2 over all 24 runs:

| | scored Turing | not Turing |
|---|---|---|
| **some J_ii > 0** | **10** | **0** |
| **all J_ii < 0** | **0** | **14** |

**Perfect separation, 24 of 24, with no off-diagonal cell.** The 14 non-patterning runs sit
at max diag(J) between −0.014 and −0.521, and every one of the 10 patterning runs is
positive.

**This is a necessary condition being confirmed, not an empirical surprise, and it is
reported that way.** Diffusion-driven instability with a diagonal D requires at least one
positive diagonal Jacobian entry — a standard result, not a finding of this project. What
the measurement adds is the *population* statement: on `three_gene_qvar` the optimiser
crosses into self-activating territory in **2 of 16** runs, while on the legacy sample it
does so in **8 of 8**. The reachability gap of §7.1 is, at the level of the recovered
network, exactly a self-activation gap.

The practical consequence is an instrument rather than a claim: `max diag(J) > 0` is a
one-number, zero-cost proxy for whether a configuration is reaching the patterning region,
and it is continuous (the *value*, not just the sign) where `turing_frac` is a floored count
of 8. It is used below as a secondary readout beside the pooled restart rate, never in place
of a pre-registered criterion.

It also ties the two jobs together. §8.4 found that what the seeds *do* agree on is the sign
of the diagonal (0.875–1.000) and what they do not agree on is the off-diagonal pair senses
(0.500). The diagonal is the part the objective must pin to pattern at all, and it is
pinned; the off-diagonal is the part σ(k) leaves loose, and it is loose. That is a single
coherent picture of both failures, and it predicts that **an axis which raises the Turing
rate will not, on its own, move criterion 3.1** — which is exactly what §7.2 measured and
what the `turing8` cell tests directly.

---

## 9. JOB A — cell `turing8` (`loss.weights.turing=8.0`). The rate problem is SOLVED on this target.

`sample_0000`, K = 8 (seeds 0…7), `git_sha 774ec7c`, `config_id 366484bd1f8c`, **2611 s**.
**One** override from `baseline` (`config_id 30105ae5c671`); the shared execution block and
everything else identical. All 8 run records are present and non-empty (21.3–21.7 KB each),
`seed_errors` is `{}`, and 0 seeds raised — this is a complete K = 8, not a partial cell.

The axis was proven LIVE on both the serial and the batched path before the cell was run
(§1). It is **not** in this unit's original eight axes; it was promoted on C2's cross-unit
measurement of the same override moving `nc1` from 0.0625 to 0.750.

### 9.1 The rate, against the `baseline` control

| statistic | `baseline` | **`turing8`** | |
|---|---|---|---|
| **`turing_frac`** | 0.125 (seed 3 only) | **1.000 — 8 of 8 seeds** | **8×** |
| pooled restarts `sig_max_pos > 0` | 2 / 512 (0.0039) | **18 / 512 (0.0352)** | **9×** |
| pooled `sig_max_pos` p90 | −0.0413 | **−0.0188** | |
| pooled `sig_max_pos` best | +0.1869 | **+0.3562** | |
| **winner's `sig_max` (median)** | −0.0304 | **+0.1801** | **crosses zero** |
| seeds with `max diag(J) > 0` | 1 / 8 | **8 / 8** | |
| `recovered_frac` | 1.000 (0 raised) | 1.000 (0 raised) | |
| `robustness_n_used` | **1** | **8** | 3.2 becomes computable |

**This is the largest rate movement measured anywhere in C1**, and it moves the *sensitive*
readout as well as the scored one, which `detach` did not: `detach` left the pooled restart
count at exactly 2/512, and this takes it to 18/512.

**But the two numbers must not be conflated, and the headline is the weaker of the two.**
`turing_frac` = 1.000 is a statistic over the **restart ensemble**: each seed runs
`n_restarts = 64` and the loss-argmin selects one. Per-restart reachability went from 0.39 %
to 3.5 % — a 9× improvement, not an 8× one — and at 3.5 % the chance that 64 independent
restarts contain at least one Turing member is ≈ 90 %. So `turing_frac` = 1.000 says two
things jointly: the regime is now reachable often enough that 64 restarts nearly always find
it, **and** the loss-argmin reliably selects it when present. The winner's median `sig_max`
crossing from −0.0304 to +0.1801 is the cleanest single statement of that. **A claim of
"8/8 seeds pattern" is conditional on `n_restarts = 64` and should always be quoted with
it.** The underlying regime is still reached by only 1 restart in 28.

### 9.2 The pre-registered criteria, in §3 priority order

| # | criterion | bar | `baseline` | **`turing8`** | verdict |
|---|---|---|---|---|---|
| 3.1 | `topology_consistency` @rtol 0.05 | ≥ 0.75 | 0.125 | **0.125** | **FAILS — and did not move at all** |
| 3.1 | within − cross-target gap | ≥ 0.25 | – | *(needs a 2nd qvar target — §10)* | not yet computable |
| 3.2 | median `turing_volume_10pct` | ≥ 0.90 | 1.000 (n=1) | **1.000 (n=8)** | **meets bar, now on a real n** |
| 3.2 | median `turing_volume_4p8pct` | ≥ 0.95 | 1.000 (n=1) | **1.000 (n=8)** | **meets bar, now on a real n** |
| 3.3 | `kstar_fft_rel_err` median | ≤ 0.083 | 0.979 | **0.245** | **FAILS** (4.0× better, still 3× the bar) |
| 3.3 | `trivial_kstar_err` | reported | 1.000 | **1.000** | not leaked — the k\* number is real |
| 3.3 | `morphology_match_frac` | – | 0.000 | 0.714 | |
| 3.4 | `plausibility_score` mean | = 1.0 | 0.458 | **0.375** | **FAILS**, and moves the wrong way |

§3.2 is the first criterion in this project to clear its bar on `three_gene_qvar` with
`robustness_n_used = 8` rather than 1 — at the baseline the medians were computed over the
single patterning seed and were not measurements of anything.

`plausibility_score` **falling** 0.458 → 0.375 while the rate rises is the trade-off
`PREREGISTRATION.md` §3.4 recorded *in advance* as the expected consequence of a D-ratio
prior centred at 7.5 against generators that sample 7.9–251. It is reported as a result, not
as a surprise, and the prior's centre does not move. `D_lo/D_mid` (median) falls 0.512 →
0.193, i.e. the recovered systems became more strongly differentially-diffusive — which is
what buys the instability and simultaneously what costs the plausibility score.

### 9.3 THE FINDING: the rate moved 8×, and criterion 3.1 did not move AT ALL

`topology_consistency` is **0.125 in both arms**, with **8 distinct sign structures** in
both, and `mean_agreement` moves the *wrong* way (0.579 → 0.409).

This is the **third independent confirmation** that 3.1 is not downstream of the Turing
rate, and it is the strongest of the three because it is the only one on the PRIMARY dataset
with the rate problem actually solved rather than absent:

1. `legacy_control` (§7.2): 8/8 pattern, k\* agreeing to 1.5 %, `topology_consistency` 0.125.
2. C2's `nc1` `turing=8.0` cell: rate ×12, `topology_consistency` unchanged at 0.250.
3. **this cell**: rate ×8 on qvar `sample_0000`, `topology_consistency` unchanged at 0.125.

Eight seeds that now *all* reach the Turing regime, all with a self-activating node, and no
two of which recover the same network. **Anything that reads `turing_frac` as a proxy for
reproducibility is reading it wrong**, and the earlier brief's claim that lifting the rate
would carry 3.1 with it is refuted on the primary dataset.

### 9.4 What the gauge instrument says about this cell

`c1_gauge.py` on `turing8`, beside `baseline`:

| block | `baseline` | `turing8` |
|---|---|---|
| ladder @0.05: raw → +perm → +transp → +balance | 0.125 → 0.125 → 0.125 → 0.375 | 0.125 → 0.250 → 0.250 → 0.375 |
| entry magnitude q50 (vs the rtol 0.05 cut) | 0.0512 | **0.1625** |
| entries within a factor 3 of the cut | 0.333 | **0.208** |
| BLOCK 4 `diag` / `pairs` / `3-cycle` / `all 7` | 0.875 / 0.500 / 0.750 / 0.500 | **1.000** / 0.500 / 0.625 / 0.375 |

Two things worth recording. First, the estimator's worst pathology is **relieved** here: the
median entry magnitude moves off the rtol cut (0.0512 → 0.1625) and the fraction of
coin-flip entries drops from a third to a fifth. So `turing8`'s 0.125 is a *better-measured*
0.125 than `baseline`'s — the failure is more real, not less.

Second, §8.5's prediction holds exactly. **All 8 seeds now agree on the diagonal sign
pattern (1.000, up from 0.875), and the off-diagonal pair senses stay pinned at 0.500.**
The rate lever fixed precisely the part of J that patterning requires and the objective
therefore constrains, and did nothing to the part σ(k) leaves loose. `max diag(J)` per seed
is now positive for all eight (+0.043 … +0.370) against baseline's 1 of 8, which extends
§8.5's perfect separation to **32 of 32** committed runs with still no off-diagonal cell.

### 9.5 Status

`turing8` is the best configuration C1 has measured and it is adopted as the base for the
remaining cells. It does not make the target pass: **3.1 and 3.3 and 3.4 all fail**, and
under `PREREGISTRATION.md` §3 a target passes only if all five criteria hold. What it does
is convert 3.2 from uncomputable to met, cut the k\* error 4-fold, and — most usefully —
remove the Turing rate as a confound so that the remaining failures can be attributed.

---

## 10. Cell `baseline` on `sample_0003` — and criterion 3.1's control, at last computable on the PRIMARY dataset

`three_gene_qvar/sample_0003`, K = 8 (seeds 0…7), `git_sha 9227636`,
`config_id 8681906d5a17`, **1024 s**. Baseline configuration unchanged. All 8 run records
present and non-empty (21.3–21.7 KB), `seed_errors` `{}`, 0 raised — a complete K = 8. (The
earlier in-flight attempt at this cell was 3-of-7 with 4 zero-byte results and was removed
rather than reported; this is a clean re-run of the whole cell, per §2.)

This is the target `BIO_VIABILITY.md` §4.2 identified as one that *patterns*, and it does:
`turing_frac` 0.875 (7 of 8; seed 7 is the miss), pooled restarts 31/512 = 0.0605 — **15×**
`sample_0000`'s baseline reachability of 2/512, on identical settings. It also cost 1024 s
against `sample_0000`'s 1275 s at the same K and workers, which is the third data point on
the cost anomaly and points the same way as §7.4: **the cheap targets are the patterning
ones.** Ranked by pooled reachability against wall-clock, the three baseline cells are
`legacy_control` 174/512 @ 370 s, `sample_0003` 31/512 @ 1024 s, `sample_0000` 2/512 @
1275 s — monotone in both. The expensive targets are the ones where the steady-state solve
grinds on a flattening, near-singular Jacobian.

### 10.1 CRITERION 3.1's CONTROL — the substantive half, and it is NEGATIVE

With two distinct `three_gene_qvar` targets in one runs-root, §3.1's size-matched
cross-target control is computable on the primary dataset for the first time in this unit:

| rtol | within-target | size-matched cross-target | **gap** | bar |
|---|---|---|---|---|
| 0.02 | 0.250 | 0.500 | **−0.250** | ≥ +0.25 |
| **0.05** | **0.250** | **0.510** | **−0.260** | **≥ +0.25** |
| 0.10 | 0.250 | 0.508 | **−0.258** | ≥ +0.25 |

**The gap is negative at every tolerance, and by roughly the full width of the bar in the
wrong direction.** Criterion 3.1 requires within-target agreement to exceed cross-target
agreement by ≥ 0.25; the measurement is that cross-target agreement is **twice** the
within-target value (0.510 vs 0.250).

`PREREGISTRATION.md` §3.1 states the reading in advance: *"If within-target ≈ cross-target,
the model is reproducing itself, not the target, and the criterion fails **regardless of the
absolute number**."* The measurement is worse than that condition — not approximate equality
but a systematic deficit. **Seeds fitted to different targets agree with each other more
than seeds fitted to the same target do.** That is the failure this control was placed first
to expose, and it is now measured on the primary dataset rather than inferred.

It also independently reproduces C2's finding on `nc1` (gap ≈ 0 at 0.250 within / 0.512
cross), across both regulation forms.

**This forecloses the one remaining escape route.** §8.4's gauge-invariant 7-bit statistic —
the coarser diagnostic that scored higher in absolute terms — carries its own control here,
and **it is negative too**: within 0.375, cross 0.500, gap **−0.125**. So the higher absolute
number was granularity, exactly as §8.4 warned, and not target information. No statistic
tried in this unit, at any tolerance, under any quotient, shows the model's recovered
topology carrying more information about *this* target than about a different one.

### 10.2 The other pre-registered criteria on this target

| # | criterion | bar | value | verdict |
|---|---|---|---|---|
| 3.1 | `topology_consistency` @0.05 | ≥ 0.75 | 0.250 (7 distinct structures) | **FAILS** |
| 3.1 | within − cross gap | ≥ 0.25 | **−0.260** | **FAILS** |
| 3.2 | median `turing_volume_10pct` | ≥ 0.90 | **0.990** (n = 7) | **MEETS** |
| 3.2 | median `turing_volume_4p8pct` | ≥ 0.95 | **1.000** (n = 7) | **MEETS** |
| 3.3 | `kstar_fft_rel_err` median | ≤ 0.083 | **0.0169** | **MEETS** |
| 3.3 | `trivial_kstar_err` (reported beside it) | — | **0.143** | **not leaked** |
| 3.3 | `morphology_match_frac` | true | 0.143 (1 of 7 compared) | **FAILS** |
| 3.4 | `plausibility_score` mean | = 1.0 | 0.417 | **FAILS** |

**`kstar_fft_rel_err` = 0.0169 against a bar of 0.083, with `trivial_kstar_err` = 0.143
beside it, is the first genuine k\* result in this project.** The model is 8.5× better than
the image-blind `L`-only predictor on this target, so — unlike every legacy number, where
the trivial predictor is exact to 1.3e−16 — this one carries real information about the
image. `kstar_spread` across the 8 seeds is 0.079.

So on `sample_0003` the pipeline meets **3.2 (both bars) and 3.3's k\* bar on non-leaked
data**, and fails **3.1 (both halves), 3.3's morphology requirement, and 3.4**. Under
§3 a target passes only if all five criteria hold, so `sample_0003` does not pass — but the
shape of the shortfall is now specific rather than global, and the binding failure is
unambiguously 3.1.

### 10.3 What this changes about the unit's conclusion

Combining §9 and §10, the two failures the brief separated are now measured independently of
each other on the primary dataset:

* **The rate is tractable.** `loss.weights.turing=8.0` took `sample_0000` from 1/8 to 8/8
  seeds and 2/512 to 18/512 restarts; `sample_0003` reaches 7/8 at the baseline already.
* **Criterion 3.1 is not.** It sits at 0.125–0.250 against 0.75 on every cell measured, it
  does not move when the rate moves 8×, it does not move under any symmetry quotient, and
  its control — the half `PREREGISTRATION.md` calls substantive — is **negative** on the
  primary dataset in both forms.

The mechanism in §8 says why these are different problems: the objective constrains J only
through σ(k), which fixes 7 of 9 functions of J and pins the diagonal that patterning needs,
while leaving the off-diagonal pair senses — *who regulates whom* — genuinely unconstrained.
Raising the Turing weight tightens exactly the constraint that is already binding. **No
weighting of the existing terms can fix 3.1, because the information 3.1 asks for is not in
the objective at all.** Fixing it requires a term that sees J beyond its spectrum, which is
future work and is recorded as such rather than attempted after the fact.
