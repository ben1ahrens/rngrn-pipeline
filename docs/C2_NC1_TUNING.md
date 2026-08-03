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

*(filled in below as cells land)*

---

## 4. No-op checks, per axis

*(filled in below as cells land)*
