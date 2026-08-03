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
