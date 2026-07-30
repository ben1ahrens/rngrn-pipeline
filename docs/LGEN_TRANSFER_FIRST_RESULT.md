# First cross-L transfer measurement — 2026-07-30

**Status: one seed, one system, tuning data. Not evidence for any §3 claim.** This is the
first time the instrument has been run end to end on real data, and it is recorded to show
the measurement works and to fix the baseline it will be read against.

## What was done

`three_gene_multiL` system 13 exists at four real domain sizes with **identical kinetics**,
so it is ground truth for L-generalisation:

| sample | L | true k\* | true q | morphology |
|---|---|---|---|---|
| `sample_0052` | 55.58 | 0.4522 | 4.00 | spots |
| `sample_0053` | 97.27 | 0.4522 | 7.00 | spots |
| `sample_0054` | 138.96 | 0.4522 | 10.00 | spots |
| `sample_0055` | 180.65 | 0.4522 | 13.00 | spots |

k\* is constant and q scales exactly linearly — the signature a generalising network must
reproduce.

Recovery ran on **`sample_0052` only** (L = 55.58): competitive, batched CUDA, cubic
dispersion, 64 restarts, 400 Adam steps, `d_init_from_kstar=true`, `lbfgs_steps=0`.
Run `m3_registry_20260730_005701`, 9 min 24 s. Then that single model was simulated at all
four domain sizes with **nothing refitted**.

## Recovery, on its own target

`recovered_turing` true · `rollout_patterned` true · `morphology_scored` **compared** (not
`target_only`) · `morphology_match` **false** · `kstar_fft_rel_err` **4.6 %** ·
`trivial_kstar_err` **50 %** · `plausibility_score` 0.667 · loss 0.0677.

The `trivial_kstar_err` of 50 % is the point of the new datasets: on the legacy sets that
image-blind predictor scored ~0 %, which made any k\* number unreadable. Here it is
useless, so the 4.6 % is a real measurement.

## Transfer

| L | L/L_train | k\* model | k\* TRUE | err | q model | q TRUE | class | TRUE | px/λ |
|---|---|---|---|---|---|---|---|---|---|
| 55.58 | 1.00 | 0.4301 | 0.4522 | **4.9 %** | 3.80 | 4.00 | spots | spots | 33.6 |
| 97.27 | 1.75 | 0.3778 | 0.4522 | **16.4 %** | 5.85 | 7.00 | spots | spots | 21.9 |
| 138.96 | 2.50 | 0.4150 | 0.4522 | **8.2 %** | 9.18 | 10.00 | spots | spots | 13.9 |
| 180.65 | 3.25 | 0.4027 | 0.4522 | **10.9 %** | 11.58 | 13.00 | spots | spots | 11.1 |

Against the pre-registered §3.5a bars:

| quantity | measured | bar | |
|---|---|---|---|
| `kstar_phys_cv` | 0.0476 | ≤ 0.10 | pass |
| `periods_slope_rel_err` | 0.0143 | ≤ 0.15 | pass |
| `morphology_class_preserved` | 1.00 | ≥ 0.75 | pass |
| patterned at every L | 5/5 | all | pass |
| min pixels/wavelength | 11.1 | ≥ 6.0 | pass |

## What this does and does not show

**Does.** The model patterns at every domain size it was never trained on, keeps the
morphology class at all four, and its periods-per-box scales close to linearly (3.80 → 5.85
→ 9.18 → 11.58 against a true 4 → 7 → 10 → 13). The instrument runs, and the internal
criteria are met.

**Does not.** Against ground truth the k\* error is **4.9–16.4 %**, exceeding the 8.3 %
one-FFT-bin bar at three of four sizes. The model **systematically undershoots q at every
L** — its wavelength is too long by roughly 5–16 %. That is a real bias, not scatter, and
the internal `kstar_phys_cv` of 0.0476 does not see it, because a *consistent* offset is
invariant. This is exactly why §3.5a ranks the ground-truth comparison first and the CV
last: the CV passed while the comparison it exists to support did not.

**A resolution caveat that softens part of it.** `max_kstar_bin_width_rel` is 0.263 — at
L = 55.58 there are only ~4 periods in the box, so one FFT bin is 26 % of k\* and the 4.9 %
there is well inside the estimator's own resolution. The errors at the two largest L (8.2 %,
10.9 %), where the bin is finer, are the ones that are real.

## Reproduce

```bash
rngrn --runs-root experiments/lgen_transfer train --config configs/m3_registry.yaml \
  -o data.dataset_id=three_gene_multiL -o data.sample_key=sample_0052 \
  -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
  -o train.n_restarts=64 -o train.adam_steps=400 -o train.lbfgs_steps=0 \
  -o model.d_init_from_kstar=true

rngrn --runs-root experiments/lgen_transfer evaluate --config configs/m3_registry.yaml \
  -o data.dataset_id=three_gene_multiL --run-id <run_id> \
  --eval-L 55.58 97.27 138.96 180.65
```

The run directory, frozen config, results JSON and checkpoint are tracked in git, so the
numbers above trace to the run that produced them and the model can be re-simulated.

## Persisted artefacts (added 2026-07-30, unit P1)

The transfer table above used to exist only as stdout — `cmd_evaluate` printed it and stored
nothing. It is now recorded, so every number here is re-plottable without re-running the
9-minute recovery:

| artefact | what it holds |
|---|---|
| `experiments/lgen_transfer/lgen_eval.jsonl` | one flat row per (run_id, L) — the transfer table |
| `experiments/lgen_transfer/lgen_summary.jsonl` | one row per evaluation — the §3.5a quantities |
| `runs/m3_registry_20260730_005701/arrays/lgen_fields.npz` | the (3, 128, 128) field at each L |
| `runs/m3_registry_20260730_013119/arrays/plot_arrays.npz` | the full per-run arrays + training trajectory |

Re-running the `evaluate` command above reproduced the transfer table exactly
(`kstar_phys_cv` 0.047636, `periods_slope_rel_err` 0.014345,
`morphology_class_preserved` 1.00, `min_pixels_per_wavelength` 11.055).

`m3_registry_20260730_013119` is the **same recovery re-run** under the identical command with
array/trajectory recording switched on: loss, `kstar_model`, `kstar_fft_rel_err`,
`recovered_turing`, `morphology_match` and `plausibility_score` are **bit-identical** to
`m3_registry_20260730_005701`, which is the evidence that the recorder is side-effect-free. It
is kept because it is the only run here carrying a training trajectory — a trajectory exists
only while recovery runs and cannot be reconstructed from a checkpoint.

Draw the figures with:

```bash
python scripts/make_figures.py --runs-root experiments/lgen_transfer \
  --run-id m3_registry_20260730_013119
```

**System 13 is now burned for tuning.** A §3.5a claim must use systems untouched here.
