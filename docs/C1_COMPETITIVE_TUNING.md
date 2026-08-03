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
