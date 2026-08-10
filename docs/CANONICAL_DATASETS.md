# Canonical Turing datasets

`turing_spots` and `turing_labyrinth` — the high-resolution 3-gene sets generated once and
reused across simulated-data experiments. Written for a reader who has not seen this branch.

---

## 0. Status: THIS IS THE TRAINING DATA SOURCE

**Owner decision, 2026-08-10: from here on, these two datasets are the training data source
for simulated-data work.** Everything else in `data/datasets/` becomes historical:

| dataset | role from 2026-08-10 |
|---|---|
| `turing_spots`, `turing_labyrinth` | **PRIMARY. The training data source.** |
| `three_gene_qvar` | superseded as the primary set; retained as the provenance of the canonical systems and for continuity with runs already made against it |
| `three_gene_multiL` | cross-L transfer only (`PREREGISTRATION.md` §3.5a) |
| `three_gene_train/val/test` (legacy) | **may not support any k\* claim** — the domain-size leak; kept as the comparison baseline that shows what the leak looked like |
| `*_classical_*` | dormant; not 3-gene GRNs, generator not in version control |

Two consequences a reader should not have to work out for themselves:

1. **The entire future evidence base is 10 samples, 6 of them held out.** That is the whole
   set, not a sample of a larger pool. It is deliberate — the owner asked for the smallest
   number of datasets, one per pattern type — but it means a per-sample result is the unit
   of evidence here, and corpus-level medians over five samples are not meaningful. The
   pre-registered `three_gene_qvar` split had 26 held-out samples by comparison.
2. **The configs have not caught up.** `configs/m3_registry.yaml`, `nc1_m3_registry.yaml`,
   `expA_control_full.yaml` and `expA_hidden_channel.yaml` still name `three_gene_val`;
   `expB_*.yaml` still name `two_gene_classical_val`. No config names a `turing_*` set.
   Pointing them here is a separate change and has deliberately not been made as part of
   generating the data.

## 1. What they are, in one paragraph

Two registered datasets, five distinct 3-gene systems each, at **512×512**. Every sample is
a re-simulation of a system that already exists in `three_gene_qvar` or `three_gene_multiL`,
run at a larger domain and a much finer grid. They are not new physics: the kinetics, the
screening and the integrator are the tracked generator's, unchanged. What is new is the
resolution, the selection discipline, and the fact that the split was declared before the
data existed.

There is deliberately **no `turing_stripes`**. See §6.

## 2. Why re-simulate instead of screening fresh systems

Re-simulating at a new resolution needs the generating kinetics *and* the simulation seed.
Only two datasets carry both:

| family | n | kinetics | seed | re-simulatable |
|---|---|---|---|---|
| `three_gene_qvar` | 34 | yes | yes | **yes** |
| `three_gene_multiL` | 92 (23 systems × 4 L) | yes | yes | **yes** |
| `three_gene_train/val/test` | 127 | **no** | **no** | no — generator was gitignored |
| `*_classical_*` | 160 | yes | **no** | no — and not 3-gene GRNs |

So the eligible pool is **57 distinct systems**, not 413. Everything below selects from
those. Re-simulation was verified bit-exact at 96×96 before being trusted at 512×512.

## 3. How the five per class were chosen

Deterministic and seeded, in this order:

1. **Admission gates.** `peak_bin ≥ 3` (the spectrum must have an interior peak),
   `cv ≥ 0.30` (contrast well clear of the generator's 0.05 collapse floor), and a positive
   margin from the sample's own class boundary.
2. **Rank by margin** — distance from the nearest φ/A cut that defines the class.
3. **Label stability.** Re-simulate at a different box size and require the class to
   survive. For `multiL` systems this is free (they already exist at four domain sizes);
   for `qvar` it costs one extra 96×96 run.
4. **Split roles.** Tuning slots prefer systems recovery experiments have already burned;
   held-out slots are restricted to systems never run.

**The `peak_bin` gate is load-bearing, not decoration.** Ranking by margin alone puts
`three_gene_qvar/sample_0032` *first* among spots — its area fraction of 0.032 gives it the
largest possible distance below the 0.34 cut — and that sample is the one confirmed
non-Turing frame in the corpus (`D-TDPLOT-1`). Without the gate the worst available sample
would have become a canonical exemplar.

## 4. Resolution, and why 512 rather than 1024

The k\* estimator's precision floor is set by **periods per box**, not by pixel count: the
RAPS bin width is `2π/L`, so the relative half-bin precision is `1/2p`. Pixels-per-wavelength
is `N/p`. At a fixed grid those two fight — the legacy sets sit at exactly 16 px/wavelength
with an **8.3 %** k\* floor, and `qvar` buys precision by dropping to 6.9 px/wavelength. A
larger grid dissolves the conflict: at 512 with p ∈ [8, 40] the realised samples span
14.2–64.0 px/wavelength at a 1.4–6.2 % k\* floor — better than the legacy data on **both**
axes simultaneously, which no 96×96 dataset can be.

That matters because `PREREGISTRATION.md` §3.5a sets `kstar_phys_cv ≤ 0.10` and justifies it
as "one FFT bin's worth of scatter… ±8.3 % is the estimator's own floor." On the legacy data
the pre-registered bar sits *on* the instrument's noise floor. On these sets it does not.

1024 was rejected on measured cost, not taste. Recovery itself is nearly pixel-count-free —
every loss term lives on N×N objects rather than on the image — but the post-hoc ETDRK4
morphology rollout scales purely with pixel count, because its step count is reaction-rate
limited. That is ~45 s typical and ~17 min worst case at 512, against ~3 min and ~70+ min at
1024 with `eval/numerics._phi_contour` peaking at 4–6 GB, on a host with five recorded OOM
kills. `eval/lgen_eval.grid_for_L` also refuses grids above 512 by design.

## 5. The leak, and why periods-per-box is drawn

Every pre-2026-07-29 generator set `L = clip(6·2π/k*, 18, 220)`, so `k* = 6·2π/L` exactly and
the domain size *was* the label. `gen_tg3` fixed that by drawing `p` freely. These sets keep
the fix and tighten it.

The subtlety that cost a full regeneration: the leak predictor's error is `|q − p| / p`, a
**relative** quantity, so what protects against it is the spread of `p` in *log* space, not
its absolute magnitude or even its distinctness. A first attempt drew five distinct `p`
i.i.d. from {16..32} and produced `{17, 22, 23, 24, 28}` — which an oracle blind predictor
fits to **4.5 %** median error, far closer to the defect than to the fix. So `p` is now laid
out as a **geometric ladder** across {8..40} with a seeded sub-rung offset, and
`draw_periods` **checks** the result against a 25 % bar and rejects a ladder that fails it
rather than hoping. Realised:

| dataset | periods | spread | oracle blind-predictor error |
|---|---|---|---|
| `turing_spots` | 8, 11, 16, 24, 36 | 4.5× | **37.5 %** |
| `turing_labyrinth` | 8, 10, 15, 23, 35 | 4.4× | **33.3 %** |

For scale: the legacy sets score **0.0 %** and `three_gene_qvar` scores 45.5 % at n=34.

**The honest limit.** At n=5 no range decouples strongly, because an oracle single `q` can
always sit near the middle of five values — even the legacy {3..14} range only reaches 20 %
at n=5. Corpus-level `kstar_rel_err` medians are therefore not meaningful on a five-sample
dataset at any period range, and should not be quoted.

**What was considered and rejected: fixing `L` outright.** A constant `L` would make the leak
structurally absent rather than merely small, and would make the periods emergent from `k*`.
It was implemented, measured (L = 300 puts every candidate at 16.7–56.0 px/wavelength) and
then reverted on the owner's call. The reasoning: the *wavelength* is what the
reaction-diffusion system determines, and `λ = 2π/k*` is untouched either way — it comes from
the network's Jacobian and diffusivities. Choosing `L = p·λ` only decides how much of that
pattern is in view, which is a field-of-view choice of the kind a microscope makes, not a
transformation of the physics. The commit and its revert are both in the history if the
question is reopened.

One 96²-era assumption was relaxed to make this possible. `simulate_and_classify` asserted
`18 ≤ L ≤ 220`; every canonical sample exceeds it. Those bounds encode *resolution* at a
96×96 grid, and `L` enters the physics only as a unit (CLAUDE.md §7c) — the binding
constraint is pixels-per-wavelength, which `canon_generate` now enforces directly at
`PPW_MIN/PPW_MAX = 16/32`. The bound is a parameter defaulting to the old values, so every
existing caller is bit-for-bit unaffected.

## 6. Why there is no `turing_stripes`

Measured, not assumed: every stripes candidate in the re-simulatable corpus loses the label
when only the box size changes, and no sample at `p ≥ 11` is ever labelled stripes — against
a canonical range of 16–32. Anisotropy falls as the box grows (`r = −0.312`), because a small
box admits few orientations and forces the pattern onto one axis. Full argument, numbers and
rejected alternatives in `docs/DECISIONS.md` **D-CANON-2**; the figure is
`experiments/figures_report/canonical/s1_stripes_is_a_small_box_artefact.png`.

`holes` is absent for a different and older reason: it is structurally unreachable. Species 0
is the self-activator in all six topologies, so the observed channel is positively skewed by
construction, while `holes` (φ > 0.66) needs strong negative skew. The corpus contains zero
`holes` samples in 413.

## 7. What is stored

Per sample: `final_frame` (3 × 512 × 512, **all three channels**), `x_star`, `D`,
`interaction_matrix`, `jacobian`, and `cv_trace`/`cv_times`. Attributes carry `L` and
`k_star` (both mandatory — `gate.from_registry` raises without them), the generator's
morphology and shape statistics, the full `params_json`, `sim_seed`, `periods_per_box`,
`tmax`, and the provenance fields `source_dataset` / `source_key` / `role`.

**No `trajectory`.** Nothing under `src/rngrn/` reads it, so at 512² it would cost 6× the
storage for data no consumer touches. The `cv_trace` replaces it for the one purpose it
served here — confirming saturation.

**All three channels are kept** even though a real photograph gives one observable. That
makes `m=1` versus `m=3` a controlled comparison on identical data rather than two different
datasets.

## 8. The integration horizon varies per sample

`Tmax = 260` was chosen for 96×96 boxes and is not always enough at 512, where a sample holds
16–32 periods rather than 3–14. Each sample doubles its own horizon until the saturation gate
passes, capped at 8×. The horizon actually used is recorded per sample as the `tmax`
attribute, because it is needed to reproduce the frame.

The generator is *not* early-stopped when the cv plateaus. That would make the saturation
check trivially true afterwards — the gate would be testing the rule that produced the frame.
Restarting at a longer fixed horizon keeps it an independent check.

## 9. Reproducing

```bash
bash scripts/link_payloads.sh                       # provision the source corpus
.venv/bin/python scripts/canon_select.py            # freeze the selection (probes: ~15 min)
bash scripts/guarded_run.sh .venv/bin/python -u \
    scripts/canon_generate.py --procs 5             # generate at 512
.venv/bin/rngrn scan-datasets --datasets-root data/datasets
```

`data/canonical_selection.json` is the frozen record: which systems, at which `p`, on which
side of the split, and why each was admitted. It is tracked, so the path from corpus to
canonical set is auditable without re-running anything.

## 10. What these datasets do NOT establish

- **No recovery result.** Nothing is fitted here; no recovery metric is computed.
- **"Patterned" is not "Turing-unstable".** σ(k) is evaluated nowhere in this unit.
- **The `m<N` objective is still an open problem.** The plumbing for a single observed
  channel works, but `exp06` measured the stationarity residual — the only term latent
  channels enter — as harmful, with 9/9 swept cells collapsing. Storing three channels
  enables that work; it does not solve it.
- **`raps` still assumes a periodic tile.** It sets `dk = 2π/L` with no windowing or
  detrending, so a cropped real image will bias k\* *silently*. These sets are clean periodic
  masters precisely so that degradation can be applied downstream and measured, but the
  measurement has not been done.
- **The saturation tolerance is a convergence tolerance, not a calibrated threshold.** It is
  enforced as a fail-loud gate, never used to judge a result.
