# What the model actually sees

**Audit of every quantity that reaches the RNGRN model or its loss during recovery.**
Branch `feature/turing-training` @ `c237d30`. Read-only audit; nothing modified.

Every claim below carries a `file:line` citation. **Where the code and the docs disagree,
the code wins** and the discrepancy is called out explicitly in §6.

---

## 0. Verdict up front

| Question | Answer |
|---|---|
| Does any answer-key quantity (`x*`, `J`, `σ(k)`, `k*_true`, `k*_fft`, `σ_max`, `D`, generating coefficients) reach the model or the loss? | **No.** Neither in the library path nor in the exp04–exp09 recipe. Verified empirically (§5.3) and by grep (§2.4). |
| Is the firewall test as strong as the docs claim? | **No.** It is a 5-string substring match over the *import lines* of 11 hand-listed files. It catches the obvious breach and misses several realistic ones. See §5.1. |
| Is there anything alarming? | **Yes, one thing — but it is not an import leak.** `L` is a legal recovery input, and in **94.8% of the 287 registered samples `k*_true ≡ 6·2π/L` to machine precision** (§4). Recovery therefore receives the headline scoring target through a legal channel, as arithmetic. `scripts/exp08_trivial_baseline.py` exists precisely to measure this and it is the most important caveat on every k\* number the project reports. |
| Does anything currently constrain `D`? | **No.** No prior, no bound, no penalty. `D = exp(θ_D)`, `θ_D ~ N(0, 0.5²)` (`model.py:75,107`); `D` is only shaped indirectly, through `σ(k)` in the anchor and the hinges. |

---

## 1. The recovery input surface

### 1.1 `RecoveryInput` — the complete list

`src/rngrn/data/gate.py:32-38`. Four fields, frozen dataclass, nothing else:

| Field | Line | Type | Derived from |
|---|---|---|---|
| `frame` | `gate.py:35` | `(m,H,W)` ndarray | `_observe(frame_full, observed_idx)` (`gate.py:75-77`) — row-selects the observed channels out of the HDF5 `final_frame` dataset and `.copy()`s. **[IMAGE]** |
| `L` | `gate.py:36` | float | The **sample's own** `L` attribute, via `_resolve_L` (`gate.py:96-116`). **[GEOMETRY]** |
| `observed_idx` | `gate.py:37` | tuple | Caller/config (`train.py:29`: `cfg.model.observed_idx or list(range(cfg.model.m))`). **[CONFIG]** |
| `N` | `gate.py:38` | int | Caller/config `cfg.model.N` (`train.py:34,36,43,46`). An **assumption**, not truth — docstring `gate.py:38`. **[CONFIG]** |

`tests/test_firewall.py:53-59` pins this set exactly (`fields == {"frame","L","observed_idx","N"}`).

### 1.2 `L` is read from the sample, and the caller loses

`_resolve_L` (`gate.py:96-116`) implements a three-way contract:

- `L_arg is None` → file's `L` (`gate.py:109`). Normal path.
- `L_arg` agrees → file's `L`, silently.
- `L_arg` **disagrees** → `UserWarning` "PREFERRING THE FILE" and the **file wins** (`gate.py:110-116`).
- `L` missing from the sample → `KeyError` via `_require_attr` (`gate.py:80-93`, called at `gate.py:109`). No default.

Confirmed by `tests/test_gate_contract.py:61-77`. Config plumbing: `cfg.data.L` (`config.py:42`) is a **generator** parameter for `source=reference` only; `cfg.data.L_override` (`config.py:46`) is the cross-check passed to the gate at `train.py:43,47`.

### 1.3 Per-loader split: what goes to which side

All three loaders build the *same* `RecoveryInput` and differ only in where the answer key comes from.

| HDF5 key / attr | `from_cache` (`gate.py:119-147`) | `from_registry` (`gate.py:150-201`) | `from_3gene_hdf5` (`gate.py:204-235`) | Side |
|---|---|---|---|---|
| `final_frame` (or `manifest["observable_key"]`) | `:130` | `:174,180` | `:219` | **RECOVERY** (after `_observe`) |
| attr `L` | `:131` | `:187` | `:224` | **RECOVERY** |
| `answer_key/x_star` / `x_star` | `:135` | `:184` | `:221` | ANSWER KEY |
| `answer_key/J` / `jacobian` | `:136` | `:183` | `:220` | ANSWER KEY |
| attr `k_star` (group attr `kstar` in cache) | `:137` | `:191` | `:225` | ANSWER KEY |
| attr `k_star_fft` | `None` (`:139`) | `:192` | `:226` | ANSWER KEY |
| attr `sigma_max` | `:140` | `:193` | `:227` | ANSWER KEY |
| `answer_key/D` / `D` | `:141` | `:185` | `:222` | ANSWER KEY |
| `interaction_matrix` / `coefficients` | `:142` | `:186,195` | `:223,231` | ANSWER KEY |
| `n_species` / `frame_full.shape[0]` | `:132` | `:194` | `:228` | ANSWER KEY (`n_species_true`) |

`from_registry` additionally gates answer-key reads on the manifest's `answer_key_keys`
allow-list (`gate.py:175,181-182`; written by `registry.register`, default
`("jacobian","x_star","D","interaction_matrix","kstar")` at `data/registry.py:49-50`).

**Observed schema** (measured, 9 datasets / 287 samples): every sample group carries
datasets `final_frame (n,96,96)`, `D (n,)`, `jacobian (n,n)`, `x_star (n,)`; the
`three_gene_*` family adds `interaction_matrix (3,3)`, `trajectory (6,3,96,96)`,
`times (6,)`. Attrs include `L, k_star, k_star_fft, sigma_max, morphology, wavelength,
dx, grid, dt_sim, params_json` (+ family/topology labels). **`trajectory` is never read by
any loader** — the gate takes `final_frame` only, so the time series is inaccessible to
both sides as currently wired.

> **Sharp edge, not a leak.** `from_cache` reads `f.attrs["n_species"]` (`gate.py:132`)
> but the registry payloads have no file-level attrs (measured: `file_attrs=[]` on all 9);
> `two_gene_classical_*` carries `n_species` as a *sample* attr. Only `from_cache` is
> affected, and only for cache payloads, which do have it.

---

## 2. The full chain into the model

### 2.1 Library path — `recover.recover`

`recover.py:52-145`. Signature `recover.py:52-54`. Entry: `train.py:59` gets
`(ri, answer_key)` from the gate; `train.py:62-66` calls `recover(ri, ...)` passing
**only** config scalars; `answer_key` is untouched until `train.py:78` (scoring).

| # | Quantity | Tag | Computed at | Notes |
|---|---|---|---|---|
| 1 | `frame` → `torch.tensor` | **[IMAGE]** | `recover.py:61` | the `(m,H,W)` observed stack, float64 |
| 2 | `L` | **[GEOMETRY]** | `recover.py:62` | passed to the Laplacian, `terms.py:164` |
| 3 | `N`, `observed_idx`, `m` | **[CONFIG]** | `recover.py:62-63` | |
| 4 | `kstar_obs` | **[IMAGE]** | `recover.py:68` → **`observables.kstar_of(frame[0], L)`** (`observables.py:58-59` → `raps`, `observables.py:31-55`) | **channel 0 only.** Power-weighted centroid over the ±2 bins around the RAPS peak (`observables.py:51-54`), physical rad/length via `fftfreq(N, d=L/N)*2π` (`observables.py:40`) |
| 5 | `kgrid` | **[CONFIG]×[IMAGE]** | `recover.py:69` → `_kgrid_for` (`recover.py:39-41`) | `linspace(kstar_obs/50+1e-3, max(2, 8·kstar_obs), 400)` — **the grid itself is set by the image measurement** |
| 6 | model params `θ_s, θ_g, θ_α, θ_δ, θ_β, θ_D` | **[MODEL-SIDE]** | `recover.py:73` → `RNGRN(N, form, seed=seed+r)` (`model.py:56-75`) | **random init only**, `torch.Generator().manual_seed(seed)` (`model.py:63`) |
| 7 | latent fields (when `m<N`) | **[IMAGE]** + **[MODEL-SIDE]** | `recover.py:76-78` | `frame.mean(0)` repeated, times `U(0.8,1.2)` — seeded from the image, then a free `nn.Parameter` |
| 8 | `x*` | **[MODEL-SIDE]** | `terms.steady_state` (`terms.py:31-68`), polished by `steady_state_diff` (`terms.py:71-79`) | damped Newton from `x0 = ones(N)` (`terms.py:44`), relaxation fallback `terms.py:60-64`. Docstring `terms.py:37`: "NEVER returns the frame mean" — **true as written** |
| 9 | `J` | **[MODEL-SIDE]** | `model.jacobian` (`model.py:133-139`) | autodiff of `model.reaction`, `create_graph=True` |
| 10 | `σ(k)` | **[MODEL-SIDE]** | `model.dispersion` (`model.py:142-152`) | `max_i Re eig(J − k²diag(D))`, `torch.linalg.eigvals` |
| 11 | `D` | **[MODEL-SIDE]** | `model.py:107` — `D = exp(θ_D)` | see §4 |
| 12 | `tau`, `jac_floor`, weights, `strategy`, `adam_steps/lr`, `lbfgs_steps`, `grad_clip`, `n_restarts`, `seed`, `form` | **[CONFIG]** | `recover.py:52-54`, `config.py:64-86` | |

**No `[TRUTH]` row exists.** Nothing in the chain reads an answer-key quantity.

### 2.2 The four loss terms

`total.compute_terms` (`total.py:27-41`) is the only assembler on the library path; it is
called from `total.total_loss` (`total.py:44-56`), which `recover.py:86,111,121` calls.

| Term | Function | Inputs (tagged) |
|---|---|---|
| `kstar` | `terms.kstar_anchor` (`terms.py:120-138`), called `total.py:35` | `σ(k)` **[MODEL-SIDE]** (`terms.py:132`); `kgrid` **[CONFIG]×[IMAGE]**; `kstar_obs` **[IMAGE]**. Loss `= clamp(logsumexp(60σ)/60 − σ(k*_obs), min=0)` (`terms.py:133-135`). Reports `kstar_model = kgrid[argmax σ]` **detached** (`terms.py:136`) |
| `turing` | `terms.turing_hinges` (`terms.py:91-105`), called `total.py:36` | `J`, `σ(k)` **[MODEL-SIDE]** only. `softplus(σ(k₀)+margin)/β + softplus(−(max σ −margin))/β` (`terms.py:102-103`). **No image input at all** |
| `resid` | `terms.stationarity_residual` (`terms.py:144-186`), called `total.py:37` | `frame` **[IMAGE]** (`terms.py:154-157`); `L` **[GEOMETRY]** → `obs.laplacian_torch` (`terms.py:164`, `observables.py:131-144`, 5-point periodic); `D`, `f(x)` **[MODEL-SIDE]** (`terms.py:161-166`). Relative form: `mean(rhs²)/(mean((D·lap)²)+mean(f²)+1e-9)` (`terms.py:172-176`) |
| `anticollapse` | `terms.anticollapse` (`terms.py:189-201`), called `total.py:38` | `J` **[MODEL-SIDE]**; `jac_floor` **[CONFIG]**. `softplus(floor − ‖J‖_F)` |

Weighting: `strategy.combine` (`total.py:53`). `FixedWeighting` (`weighting.py:35-40`),
`ScheduledWeighting` (`weighting.py:43-53`). **`gradnorm` and `ntk` are stubs that silently
return base weights** (`weighting.py:56-71`, `weighting.py:74-90`) — their docstrings say so
(`weighting.py:60`, `:80`) but `build_strategy` (`weighting.py:93-95`) will happily hand you
one from a config.

### 2.3 Dead / inert knobs on the library path

- **`tau`** (`config.py:75`, default 0.12) threads `recover.py:53 → total.py:22,45,51 → terms.py:120` and is then **never referenced in the body of `kstar_anchor`** (verified: no `tau` token in `terms.py:121-138`). The docstring at `terms.py:122-131` describes a soft-max gap penalty with no tolerance band. **`loss.tau` is inert.**
- **`weights["morphology"]`** (`config.py:73`, default 0.1) — `compute_terms` builds
  `term_vals` with four keys only (`total.py:39`), so the morphology weight multiplies
  nothing. `terms.morphology_consistency` (`terms.py:207-212`) is numpy, non-differentiable,
  and uncalled. `config.py:68-71` documents this honestly.
- `terms.composite_loss` (`terms.py:221-238`) is a parallel, unused assembler (`recover.py`
  goes through `total.py`). It hardcodes `DEFAULT_WEIGHTS` (`terms.py:218`).

### 2.4 Grep evidence

`grep -rn "k_star|x_star|jacobian|interaction_matrix|sigma_max|answer_key|AnswerKey|rd_models|data.solver|data.cache"` over `src/rngrn/` excluding `data/`, `validate.py`, `scoring/`
returns only: `model.jacobian`/`pert.jacobian`/`m.jacobian` calls (model's own autodiff
Jacobian), the string `k_star` inside `observables.py` docstrings, `sigma_max` in
`rollout.py:8` and `config.py:94` (the model's *own* σ_max for a rollout horizon), and
`train.py`'s scoring-side lines. **No recovery-side module reads a truth quantity.**

---

## 3. The working recipe (exp04–exp09) — what actually runs

The library objective in `losses/` is **not** what produced the current results. The live
recipe is `scripts/exp05_pixel_minibatch.py::fit` (`exp05:70-111`); exp06/07/08/09 all
import it by file path and call it (`exp06:17-19,41`; `exp07:33-35,42`;
`exp08:26-28,61`; `exp09:32-34,40`).

### 3.1 Inputs to `exp05.fit`

`fit(seed, frame_t, L, kstar_obs, steps, batch, w_resid, anchor_w=2.0, lr=0.05, nk=200)` — `exp05:70`.
Callers load `final_frame` and attr `L` from the payload and compute `kstar_obs =
obs.kstar_of(frame[0], L)` themselves (`exp05:122-127`, `exp07:37-39`, `exp09:36-39`).
`k_star` is read at `exp04:99`, `exp05:125`, `exp06:31`, `exp07:58`, `exp08:47`, `exp09:61`
— **always at `__main__` scope, always only for the error metric, never passed into `fit`.**
Confirmed by the signature: `fit` takes `kstar_obs`, not `kstar_true`.

### 3.2 The four pieces of the live objective

| Piece | Line | Tag | Detail |
|---|---|---|---|
| `Lk = T.kstar_anchor(m, xsd, kg, kstar_obs)` | `exp05:85` | **[IMAGE]** target | same term as the library |
| **frame-scale anchor** `scale = 2.0·mean[(log(mean(frame)) − log(x*))²]` | `exp05:86` (`exp04:57` identical) | **[IMAGE]** | see §3.4 |
| `split_hinges(m, xsd, kg)` | `exp05:88` → `exp05:43-48` | **[MODEL-SIDE]** | **differs from the library term** — see §3.3 |
| `Lr` (optional, `w_resid>0`) | `exp05:89-95` | **[IMAGE]**+**[GEOMETRY]** | full-pixel `T.stationarity_residual` or `residual_minibatch` (`exp05:51-67`) over `batch` random pixels; the Laplacian is precomputed once on the whole field (`exp05:77`) because it is non-local |

`anticollapse` and `jac_floor` **do not appear anywhere in exp04–exp09** (grep: no hits).
The live objective has three terms, not four.

### 3.3 Staged schedule and the split hinge

- **Grid**: `kg = linspace(0.0, max(3.0, 4·kstar_obs), 200)` (`exp05:73`, `exp04:45`) — starts at **exactly k=0**, unlike the library's `kstar_obs/50+1e-3` (`recover.py:41`).
- **Stage schedule** (`exp05:87`): `w = 0` for `it < 0.25·steps`; then linear ramp to 1 over the next `0.25·steps`; then 1. So **data terms alone for the first quarter, hinges fully on from the halfway point.** This is exp04's `data_first` mode (`exp04:60-62`) promoted to the only mode; exp04's `physics_first` (`exp04:58-59`) is the exp03 baseline it beat.
- **Split hinge** (`exp05:43-48`, identical `exp04:35-40`): the uniform-stability hinge uses `sig[0]` = σ(0) exactly, but the instability hinge maximises over `sig[i0:]` with `i0 = max(1, int(0.1·len(kgrid)))` = index 20, i.e. **k ≳ 0.4·kstar_obs**. The library's `turing_hinges` (`terms.py:101`) takes `sig.max()` over the *whole* grid including k≈0, so the two hinges can be satisfied by the same mode. The split is the fix; it lives only in the scripts.
- **Steady state is detached** before the dispersion terms: `xsd = xs.detach()` (`exp05:83`), so `Lk` and the hinges give gradient through `J(x*)` and `D` but **not** through `∂x*/∂θ`. The frame-scale anchor at `exp05:86` uses the *undetached* `xs`, so it is the only term carrying `∂x*/∂θ`.

### 3.4 The frame-scale anchor — is it firewall-legal?

```python
obs_scale = float(frame_t.mean())                                   # exp05:72
scale = anchor_w * ((np.log(obs_scale) - torch.log(xs.clamp_min(1e-6)))**2).mean()   # exp05:86
```

**The statistic is the scalar mean over *all* observed channels and all pixels** of the
observed frame — `frame_t.mean()`, not a per-channel mean. It is compared, in log space,
against every component of the model's own `x*`, and averaged. `anchor_w = 2.0` by default
(`exp05:70`).

**Legality: LEGAL.** `frame_t` is `RecoveryInput.frame`; the mean of an image is an image
statistic. No answer-key read is involved.

**But it is a proxy for `x*`, and worth stating plainly.** For a converged Turing pattern
the spatial mean of a channel approximates that channel's homogeneous steady state.
Measured over all 287 samples (786 channel-instances):
`|mean_pixels(frame_c) − x*_c| / |x*_c|` has **median 8.8%, p90 37.4%, max 258%**. So the
anchor pulls `x*` toward a quantity that is within ~9% of the truth for a typical channel.
This is not a firewall breach — it is exactly the information a real experimentalist reads
off a stained image — but it means "recovery finds `x*`" is partly "recovery is told
approximately where `x*` is". Note also the collapse to a **single scalar**: for
`three_gene_val/sample_0000` the per-channel means are `[0.413, 1.854, 0.059]` while the
anchor target is the pooled `0.775`, so the term actively pushes all three components of
`x*` toward the same value.

### 3.5 Aside — untracked `scripts/exp10_power_dispersion.py`

Present in the working tree, untracked. It reads **true** `jacobian` and `D` from
`three_gene_{train,val,test}` (`exp10:71`) — but as **numerical test matrices** for
comparing `dispersion_power` against `dispersion_eig` (`exp10:81`), not as inputs to any
recovery run. It never constructs an `RNGRN` from them. **Legal**, on the same footing as
a unit test; flagged here only so it is not mistaken for a leak on a later read.

### 3.6 The scoring hook in the scripts

`exp05:108` (and `exp04:82`) call `eval.analysis.turing_ok(J, m.D)` (`analysis.py:19-32`)
on the **recovered** model, which returns `info["kstar"] = kgrid[argmax σ]` on a fixed
numpy grid `linspace(1e-3, 50, 4000)` (`analysis.py:23,30`). That `kstar` is the number
compared to `kstar_true` in the callers. It is a model-side quantity throughout.

---

## 4. Where a diffusion prior would attach

### 4.1 What `D` is now

```python
self.theta_D = nn.Parameter(randn(N) * 0.5)     # model.py:75  (comment: "diffusion (exp)")
@property
def D(self): return torch.exp(self.theta_D)     # model.py:107
```

`randn` is the seeded generator at `model.py:63-66`. So **`log D ~ N(0, 0.5²)` i.i.d. per
species**, i.e. `D` is log-normal with median 1.

### 4.2 Measured init distribution (200 seeds, not read off the code)

`RNGRN(N=N, seed=s)` for `s = 0..199`, `D = model.D`:

| N | median D | geo-mean D | sd(log D) | 5–95% of D | max D |
|---|---|---|---|---|---|
| 2 | 1.0095 | 1.0088 | 0.479 | 0.454 – 2.193 | 4.25 |
| 3 | 0.9715 | 0.9766 | 0.519 | 0.400 – 2.166 | 8.31 |

Per-seed **D-ratio** `Dmax/Dmin`:

| N | min | p25 | median | p75 | p95 | max | P(ratio>5) | P(ratio>10) | P(ratio<2) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.001 | 1.274 | **1.627** | 2.200 | 3.359 | 11.41 | 1.0% | 0.5% | 67.0% |
| 3 | 1.101 | 1.669 | **2.235** | 3.347 | 5.358 | 16.81 | 6.5% | 1.0% | 42.0% |

For N=3, `D₁/D₀` has median 1.047 (5–95%: 0.251–3.617) — **the init has no preferred
ordering of the diffusivities at all.**

**Contrast with the answer keys** (SCORING SIDE — quoted here only to size the prior, and
this table must never be read by recovery code): true `Dmax/Dmin` per dataset has median
**41.1 – 140.1**, minimum **10.0** across all 287 samples.

So the init sits at a D-ratio of ~2 while every generator used ≥10 and typically ~50–140.
**Recovery starts two orders of magnitude away from the required separation of scales.**

### 4.3 Does anything constrain `D`? No.

- Grep for `theta_D` / `model.D` across `src/`: the only *recovery-path* consumers are
  `model.py:148` (dispersion), `model.py:159` (2×2 self-test), `terms.py:165`
  (residual), and read-outs at `recover.py:142`, `analysis.py:39,62`,
  `rollout.py:44`, `topology.py:134`, `validate.py:215`.
- **No penalty, bound, clamp, or prior term anywhere.** `D` is shaped only *implicitly*:
  through `σ(k) = max Re eig(J − k²D)` in `kstar_anchor` and the hinges.
- `validate.py:215` uses `D_rec` only inside `turing_ok`; **`D` is not scored against the
  answer key at all** — there is no `D_rel_err` metric.

Measured gradient reaching `θ_D` from each live term at init (N=3, seed 0, `‖∂L/∂θ_D‖`):

| term | ‖grad θ_D‖ |
|---|---|
| `kstar_anchor` | 0.0816 |
| `split_hinges` | 0.0600 |
| `stationarity_residual` | 0.0072 |
| frame-scale anchor | **0.0** (exactly — `D` does not enter `x*`) |

And what the unconstrained optimiser does with that freedom — 12 seeds, 1200 steps of the
exp05 recipe on `three_gene_val/sample_0000` (true D-ratio 60.8):

- D-ratio at init: median 2.43, max 16.8
- D-ratio after fit: **median 118.2, spread 1.6 → 1483.9**

The recipe *does* drive the separation of scales up into the right neighbourhood, but with
two orders of magnitude of seed-to-seed spread and no upper restraint. **This is the gap a
prior would close.**

### 4.4 The four attachment points, ranked

| # | Where | Change | Reaches the live recipe? |
|---|---|---|---|
| **1** | `model.py:75` — the `θ_D` init scale | Re-centre/widen: `randn(N)*σ_D + μ_D`, or draw an *ordered* `θ_D` so `D₀ < D₁ < …`. **An init-side prior.** | **YES** — every script constructs `RNGRN(N=3, seed=seed)` (`exp05:74`, `exp04:46`) |
| **2** | `model.py:107` — the `D` property | Reparameterise: e.g. `D = D_ref · exp(θ_D)`, or a ratio parameterisation `D_i = D_0 · Π softplus(ρ_j)` that makes the *ratio* the learned quantity and lets you prior it directly. **A hard structural prior.** | **YES** — same reason |
| **3** | `losses/terms.py` new term + `total.py:39` `term_vals` + `config.py:72-73` weight | The clean, config-switchable **soft prior**, e.g. `L_D = (log(Dmax/Dmin) − log r₀)²`. | **NO** — `total.py` is the *library* objective; exp04–exp09 assemble their loss inline at `exp05:88`. It would need adding there too. |
| **4** | `scripts/exp05_pixel_minibatch.py:88` | Add the penalty to the live `loss = Lk + scale + w*split_hinges(...)`. | **YES**, but script-local and unaudited |

**Recommendation:** do #1 or #2 (they change the starting distribution, which is where the
two-orders-of-magnitude gap lives, and they are automatically picked up by every script),
and add #3 for the tunable soft version — but wire #3 into `exp05:88` at the same time or
it will silently do nothing to the numbers you are measuring.

### 4.5 Firewall rules for the prior — the trap

A diffusion prior is **legal** if its parameters come from **outside this dataset**:
published diffusivities, protein/morphogen size arguments, a stated modelling assumption
like "the inhibitor diffuses 10–100× faster". Hard-code the literature number.

A diffusion prior is a **breach** if its parameters are *fitted or tuned to the datasets'
own `D`* — including informally. Concretely, **do not** centre the prior on the measured
median true D-ratio (§4.2), and do not sweep `r₀` and pick the value that minimises
`kstar_rel_err` on samples whose truth you then report against. Both route
`AnswerKey.D` into recovery through your judgement rather than through an import, and
`tests/test_firewall.py` cannot see either.

The safe protocol: fix the prior's constants from a citation before running anything,
record the citation next to the constant, and if you must tune, tune on a
held-out split and report the tuning as part of the method.

---

## 5. Firewall verification

### 5.1 What `tests/test_firewall.py` ACTUALLY enforces

Two tests, 12 parametrised cases, all green (`pytest tests/test_firewall.py -q` → `12 passed`).

**Test 1** (`test_firewall.py:42-50`): for each of 11 hand-listed files
(`RECOVERY_SIDE`, `test_firewall.py:19-24`), `ast`-parse it, flatten the import
statements into **one space-joined string** (`_imports`, `test_firewall.py:30-39`), and
assert that none of 5 substrings appears:

```python
FORBIDDEN = ["rd_models", "data.solver", "data.cache", "AnswerKey", "answer_key"]   # :27
```

**Test 2** (`test_firewall.py:53-59`): `RecoveryInput` has exactly the four fields.

That is the whole enforcement. It is a **substring match over import lines**, so:

| Probe: this import added to a recovery-side module | Caught? |
|---|---|
| `from .data.rd_models import build_system` | **YES** (`rd_models`) |
| `from .data.gate import AnswerKey` | **YES** (`AnswerKey`) |
| `from .data import gate` → then call `gate.from_registry(...)` and read the returned `AnswerKey` | **NO** |
| `from .data.gate import from_registry` | **NO** |
| `from .data import registry as reg` (answer-key-side module by its own docstring, `data/registry.py:17`) | **NO** |
| `import h5py` → open `payload.h5` and read `attrs["k_star"]`, `["x_star"]`, `["D"]` directly | **NO** |
| `from .validate import score_recovery` | **NO** |
| `from .scoring import permutation` | **NO** by *this* test (caught by `tests/test_morphology_scoring.py:628-647`, verified passing) |
| `import importlib` → `importlib.import_module("rngrn.data.rd_models")` | **NO** |

**Coverage gaps beyond the string list:**

1. **`SRC = .../src/rngrn` (`test_firewall.py:16`) — `scripts/` is never parsed.** exp04–exp09, which *are* the current method, are entirely outside the audit. They legitimately open `payload.h5` and read `k_star` (`exp05:125` etc.); nothing structurally stops a future edit from passing it into `fit`.
2. **`RECOVERY_SIDE` is a hand-maintained list of 11 files.** 26 other `src/rngrn/*.py` files are unaudited, including `train.py`, `validate.py`, `optim/sweep.py`, `optim/benchmark.py`, and `cli.py`. A new recovery-side module is unaudited until someone remembers to add it. Two other tests (`test_morphology_scoring.py:636-641`, `test_permutation_scoring.py:288+`) maintain **their own copies** of the same list — deliberately, per `test_morphology_scoring.py:631-633` — so a new module must be added in three places.
3. **The direct-HDF5 route is wide open**, and the payloads keep truth in the same file as the frame.
4. Nothing checks *runtime* dataflow — only import text.

### 5.2 Docs vs code on the firewall

| Claim | Source | Status |
|---|---|---|
| "No recovery-side module may import `data.rd_models`, `data.solver`, `data.cache`, `AnswerKey`, **or `rngrn.scoring`**" | `docs/HANDOFF_identifiability.md:131-133` | **Overstated for `test_firewall.py`.** `"scoring"` is not in `FORBIDDEN` (`test_firewall.py:27`). It *is* enforced, but by `test_morphology_scoring.py:628-647`. The sentence attributes it to the wrong test. |
| "The only truth-derived inputs to recovery are `(frame, L, observed_idx)`. This is enforced by `tests/test_firewall.py` (a static import audit), not by comments." | `README.md:17-21` | **Half true.** The *set* is enforced (test 2, `test_firewall.py:53-59`) and it is `(frame, L, observed_idx, N)` — README omits `N`. The word "enforced" oversells a 5-string import check: §5.1 lists realistic breaches it passes. |
| "`RecoveryInput : (frame, L, observed_idx)` → handed to recovery. Nothing else." | `gate.py:6` | Docstring omits `N`, which the dataclass carries (`gate.py:38`) and the test requires. |
| "no recovery-side module (recover, train, losses, model, observables, optim, eval rollout) imports …" | `gate.py:15-17` | **`train.py` and `optim/` are NOT in `RECOVERY_SIDE`** (`test_firewall.py:19-24`) and are not audited. `train.py` in fact imports `data.rd_models` (`train.py:16`) and `data.cache` (`train.py:18`) — correctly, since it sits astride the boundary, but the gate docstring calls it recovery-side. |

### 5.3 Empirical check — one real sample

`gate.from_registry("data/datasets", "three_gene_val", "sample_0000", N=3, observed_idx=[0,1,2])`:

```
RecoveryInput  (dir() → ['L', 'N', 'frame', 'observed_idx'] — nothing else)
  frame        = ndarray(3, 96, 96) float32  min 0.05493  max 2.03542  mean 0.77525
  L            = 91.48997297253318
  observed_idx = (0, 1, 2)
  N            = 3

AnswerKey  (quarantined)
  x_star  = [0.4205 1.8744 0.0576]
  J       = [[0.4204 -0.2758 -0.0474] [7.9131 -1.2208 0.] [0.1369 -0.0129 -2.2324]]
  kstar   = 0.4120573065902579      kstar_fft = 0.446      sigma_max = 0.05577074930284631
  D       = [1.0  58.6314  60.8457]
  coefficients = {'interaction_matrix': ...}     n_species_true = 3
```

Derived from the frame by recovery:
`obs.kstar_of(frame[0], L) = 0.432021` (per-channel: 0.432021 / 0.423638 / 0.427741 — **only channel 0 is used**), `frame.mean() = 0.775249`.

**The `RecoveryInput` object physically cannot carry truth**: it is a frozen 4-field
dataclass and the answer key is a separate return value.

### 5.4 THE REAL EXPOSURE — `L` encodes `k*_true`

```
L = 91.48997297253318   →   6·2π/L = 0.412057
AnswerKey.kstar         =              0.412057
relative difference     =              1.3e-16
```

Measured across **all 287 registered samples: 94.8% satisfy `k*_true·L/2π = 6.000` to
within 1e-6.** Per dataset the median is exactly 6.0000 for all nine; only the
`*_classical_*` families contain any exceptions (max ratio 8.30, 8.05, 8.05).

`L` is a legal recovery input by the project's own reasoning (`gate.py:9-13`: it is
measurable geometry, not generator kinetics), and that reasoning is sound in isolation.
But because the generators set `L = clip(6·2π/k*_true, 18, 220)` (`exp08:5-8`), **the
headline scoring target is a deterministic function of a legal input.** A "recovery" that
ignores the image and returns `6·2π/L` scores ~0% error.

This is **not** an import leak and no code change closes it — it is a property of the
dataset. It is already documented and instrumented: `scripts/exp08_trivial_baseline.py`
compares `trivial` (`exp08:54`), `fft_only` (`exp08:55`), `recovery` (`exp08:65`) and
`shuffled_L` (`exp08:56-57,66`), the last being the discriminating control (`exp08:20`).
`tests/test_gate_contract.py:23-26` pins the 6-bin fact.

**Consequence for the D prior:** if the prior is expressed in terms of a length scale
derived from `L` — e.g. "`D` should give a Turing wavelength of `L/6`" — it re-injects
`k*_true` under a new name and will make `kstar_rel_err` look excellent for reasons that
have nothing to do with the image. Express the prior in terms of the **D-ratio**, which is
dimensionless and not determined by `L`.

---

## 6. Code vs docs — full list (code wins)

| # | Doc says | Code does | File:line |
|---|---|---|---|
| 1 | `RecoveryInput` is `(frame, L, observed_idx)` | It is `(frame, L, observed_idx, N)` | `gate.py:6` & `README.md:20` vs `gate.py:32-38`, `test_firewall.py:58` |
| 2 | Firewall forbids `rngrn.scoring` per `test_firewall.py` | `FORBIDDEN` has no `"scoring"`; that check lives in `test_morphology_scoring.py` | `docs/HANDOFF_identifiability.md:131-133` vs `test_firewall.py:27` |
| 3 | `train`, `optim` are recovery-side modules | Neither is in `RECOVERY_SIDE`; `train.py` imports `rd_models` and `cache` | `gate.py:15-17` vs `test_firewall.py:19-24`, `train.py:16,18` |
| 4 | `loss.tau` is the "k\* tolerance band" | `tau` is accepted and never used in `kstar_anchor`'s body | `config.py:75`, `terms.py:120` vs `terms.py:121-138` |
| 5 | `gradnorm` / `ntk` weighting strategies | Stubs returning base weights; selectable from config with no warning | `config.py:74` vs `weighting.py:56-71,74-90` |
| 6 | Design doc §5.6 morphology weight 0.1 | Not in the differentiable sum; the weight multiplies nothing | `config.py:73` vs `total.py:39` |
| 7 | `recover.py` docstring describes the objective | The scripts use a **different** objective (split hinge at `exp05:43-48` vs `terms.py:91-105`; frame-scale anchor `exp05:86` absent from the library; no `anticollapse`) | `recover.py:1-11` vs `exp05:70-100` |
| 8 | README "24 tests" | The suite is far larger now (140 per the branch history) | `README.md:65` |

---

## 7. One-page answer to "what does the model see?"

```
FROM THE DATA FILE                          FROM CONFIG              FROM ITSELF
──────────────────                          ───────────              ───────────
final_frame[observed_idx]  (m,96,96) ──┐    N, m, observed_idx       θ_s θ_g θ_α θ_δ θ_β θ_D
attr L                     scalar    ──┤    form, n_hill, seed        ↓ (random init, model.py:70-75)
                                       │    steps, lr, grad_clip     x*  (Newton, terms.py:31)
   ↓ obs.kstar_of(frame[0], L)         │    w_resid, batch, anchor_w J   (autodiff, model.py:133)
   kstar_obs  ── anchor target         │                             D   = exp(θ_D)  (model.py:107)
   ↓ frame.mean()                      │                             σ(k)= max Re eig(J−k²D)
   obs_scale  ── x* scale target       │                                   (model.py:142)
   ↓ obs.laplacian_torch(frame, L)     │
   lap        ── PDE residual  ────────┘

NEVER REACHES RECOVERY:  x*_true · J_true · D_true · k*_true · k*_fft · σ_max ·
                         interaction_matrix · params_json · trajectory
                         (all live in gate.AnswerKey, gate.py:41-72; consumed only by
                          validate.score_recovery, train.py:78)

CAVEAT:                  k*_true ≡ 6·2π/L for 94.8% of samples → L carries the target.
```
