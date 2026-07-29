# L-decoupled and multi-L datasets

Unit 11. Two new registered datasets that (a) break the domain-size leak and (b) create
an L axis to test generalisation across domain sizes, plus a tracked copy of the
generator that made them.

---

## 1. What was wrong with the existing data

Every generator — 2-gene and 3-gene, the same line character for character
(`data/staging/tg3/generator.py:95`) — set

```python
lam = 2*np.pi/p["k_star"]; L = float(np.clip(6*lam, 18, 220))
```

so the box always held exactly **six** wavelengths. Two consequences, both measured on
the registered payloads by `scripts/gen_tg3.py --mode audit`:

| dataset | n | periods-per-box (`k*·L/2π`) | median rel. err of the image-blind predictor `k̂ = 6·2π/L` | fraction within 1 % |
|---|---|---|---|---|
| `three_gene_train` | 88 | 6.000 (only value) | **0.0 %** | 1.000 |
| `three_gene_val`   | 19 | 6.000 (only value) | **0.0 %** | 1.000 |
| `three_gene_test`  | 20 | 6.000 (only value) | **0.0 %** | 1.000 |

`0.0 %` here is the same fact as the `1.4e-14 %` figure quoted in the wave brief — it is
float round-off on an exact algebraic identity, and the exact digits depend on how the
median is taken. **A predictor that never looks at the image beats recovery on
`kstar_rel_err`.** `kstar_rel_err` is therefore a gate, never evidence.

Second consequence, equally bad for the user's requirement: with periods-per-box pinned
at 6, every sample is *the same dimensionless system at a different physical scale*.
There is no L variation to generalise across, so L-generalisation could not be measured
at all on this data.

## 2. What was generated

Both sets use the same screening and the same nonlinear spectral-IMEX simulation as the
original generator (96×96, Tmax = 260, verified Turing-unstable, non-collapsed). The
**only** physics-facing change is the domain-size policy:

```
L = p * (2π / k*)     with p an integer drawn per system
```

`p` is free instead of fixed at 6, and candidates whose wavelength admits no `p` with
`18 ≤ L ≤ 220` are **rejected** rather than clipped — the original clipped, and at a clip
bound `L` is constant, which both destroys the decoupling and mixes under-/over-resolved
patterns into one set.

### `three_gene_qvar` — 34 samples, 34 distinct systems

`p` drawn uniformly from {3…14}. This is the L-decoupled replacement for
`three_gene_{train,val,test}`.

```
python scripts/gen_tg3.py --mode qvar --dataset-id three_gene_qvar --seed 11 --procs 9
rngrn scan-datasets --datasets-root data/datasets
```

- seed 11; 36 candidates screened (6 topologies × up to 8, multiplicative kinetics),
  34 produced a real pattern, 2 collapsed to homogeneous and were dropped.
- `activator_ring` yielded only 1 candidate in 4000 screening draws — the same weakness
  the original generator worked around by screening 14 000 draws per class.
- L ∈ [33.4, 214.3]; k* ∈ [0.141, 0.753]; all twelve `p` values from 3 to 14 present.
- morphology: 17 spots / 11 labyrinth / 6 stripes.

### `three_gene_multiL` — 92 samples = 23 systems × 4 domain sizes

The **same** screened system simulated at `p ∈ {4, 7, 10, 13}`. Every sample carries
`system_id` and `replicate`; a group's four samples share identical kinetics, identical
`k_star`, and identical initial-condition seed, and differ **only** in L (median L span
within a group: 3.25×, e.g. L = 58.2 / 101.8 / 145.4 / 189.0).

```
python scripts/gen_tg3.py --mode multiL --dataset-id three_gene_multiL --seed 11 \
    --procs 12 --want-per-topology 6
rngrn scan-datasets --datasets-root data/datasets
```

- seed 11; screening required feasibility at **all four** `p` (so no group is partial);
  24 candidates → 96 simulations → 94 real patterns → **2 orphan samples dropped** so
  that all 23 surviving groups are complete. An unbalanced group would make "the same
  network from every L" an unequal comparison across systems.
- morphology: 48 spots / 34 labyrinth / 10 stripes.

### `three_gene_qvar_smoke` — 3 samples

A 3-sample qvar set (seed 3) for fast pipeline smoke tests. Not for science.

## 3. Is the leak dead? — the headline number

`k̂ = 6·2π/L`, median relative error against the stored `k_star`, computed by
`python scripts/gen_tg3.py --mode audit --dataset-id <id>`:

| dataset | n | median rel. err of `6·2π/L` | within 1 % | oracle best fixed `p` |
|---|---|---|---|---|
| `three_gene_train/val/test` (**before**) | 127 | **0.0 %** | 100 % | p = 6 → 0.0 % |
| `three_gene_qvar` (**after**) | 34 | **45.5 %** | 8.8 % | p = 10 → 28.6 % |
| `three_gene_multiL` (**after**) | 92 | **45.0 %** | 0.0 % | p = 8 → 29.2 % |
| `three_gene_qvar_smoke` | 3 | 50.0 % | 0.0 % | p = 3 → 25.0 % |

The leak is dead. The residual 8.8 % of `three_gene_qvar` within 1 % is exactly the
3 samples that happened to draw `p = 6`; that is the expected 1-in-12 rate, not
structure.

The **oracle** column is a stronger control and is reported so the set cannot look
decoupled merely because the constant moved off 6: it is `k̂ = p·2π/L` with the single
integer `p` chosen *after seeing the answers* to minimise median error on that dataset.
No blind predictor could fit it. It still costs ~29 %, so no fixed periods-per-box
explains the new data. Both numbers are properties of the DATA, not model scores.

## 4. Storage and auditability

Registry layout, unchanged: `data/datasets/<dataset_id>/{payload.h5, manifest.json}`,
directory name **is** the `dataset_id`. `payload.h5` is gitignored; `manifest.json` and
`data/datasets/datasets.jsonl` are tracked.

Per-sample HDF5 contract — same as the existing `three_gene_*` family
(`final_frame (3,96,96)`, `trajectory`, `times`, `jacobian (3,3)`, `x_star (3,)`,
`D (3,)`, `interaction_matrix (3,3)`; attrs `L, k_star, k_star_fft, sigma_max,
wavelength, morphology, topology, reaction, n, grid, dx, dt_sim, anisotropy, area_frac,
cv0, n_components`) — **plus** four additions:

- `params_json` — the generating kinetics (`b, V, mu, K, n, D`, topology, reaction,
  interaction matrix). The existing `three_gene` family does **not** store this, which
  is why the kinetics behind its 127 samples are unrecoverable from the payload.
- `system_id`, `replicate`, `periods_per_box` — identify the four L of one system.
- `sim_seed` — the initial-condition seed, so a single sample can be reproduced alone.

None of these are read by `gate.from_registry` (it reads a fixed list of datasets and
attrs), so nothing new crosses the firewall. `params_json` is answer-key-side metadata.

### A reproducibility defect found and fixed

The staging generator seeded each screen with `abs(hash(topo))`. Python salts string
hashing per process, so **the recorded seed did not reproduce the screen** — two runs of
the original generator at the same seed drew different parameters. Observed directly
here: two runs of the smoke mode at seed 11 screened 3 and then 2 candidates.
`scripts/gen_tg3.py` uses a SHA-256-derived `stable_seed()` instead; both datasets above
were regenerated after the fix, so their seeds are honest.

## 5. What these sets are FOR

- `three_gene_qvar` — the set to quote `kstar_rel_err` on. On the old sets that metric
  is unfalsifiable; here it is not.
- `three_gene_multiL` — the L-generalisation test the user asked for. The measurement it
  supports: run recovery on each of a group's four L and ask whether the **same topology**
  comes back. That is a robustness/reproducibility question along the L axis, exactly the
  primary success metric.
- Neither set is claimed to be biologically viable. Screening ranges are unchanged from
  the original (D ratios reach ~240:1), and making the *priors* biological is a separate
  unit's job. What changed here is only the geometry.

## 6. What is NOT done

- No recovery has been run on either set. This unit produced data and measured properties
  of the data; it measured nothing about the model.
- `three_gene_qvar` has no train/val/test split (`splits: {}`, like the existing
  registered sets). Splitting — and whether to split `three_gene_multiL` by `system_id`
  so no system appears in two splits — is left to whoever consumes it.
- 34 samples is small. It is what finished cleanly in the budget; more can be made by
  raising `--want-per-topology`.
- Only `multiplicative` kinetics were screened. `--reaction additive` exists but, as the
  original generator's comment records, additive kinetics almost never yields a Turing
  instability, and screening it was not worth the compute here.
