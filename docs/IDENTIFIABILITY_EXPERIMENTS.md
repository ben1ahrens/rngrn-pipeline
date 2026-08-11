# Identifiability experiments

> Picking this up cold? Read [HANDOFF_identifiability.md](HANDOFF_identifiability.md)
> for the engineering state: what changed, what is verified, and the caveats that matter.

Two validation experiments that both feed the model **2 observed channels** and ask it for
a **3×3 GRN**. They look similar and test opposite things.

|  | Experiment A | Experiment B |
|---|---|---|
| config | `expA_hidden_channel.yaml` | `expB_overparam.yaml` |
| data | 3-gene (`three_gene_val`) | 2-gene (`two_gene_classical_val`) |
| truth | a real 3rd gene exists, unobserved | **no 3rd gene exists** |
| model | N=3, m=2 | N=3, m=2 |
| true J | 3×3 | 2×2 |
| control | `expA_control_full.yaml` (m=3) | `expB_control_matched.yaml` (N=2) |
| asks | *can a hidden gene be recovered?* | *does spare capacity invent a gene?* |
| success | 3×3 correct up to relabelling the hidden species | spare species **inert** + observed 2×2 matches |

In both, recovery infers a `(1, H, W)` latent field for the unobserved species. Note that
`N=3` is an **assumption you impose** — the firewall means the model is never told the true
species count. In A that assumption is right; in B it is deliberately wrong by one.

## Why each experiment needs its own metric

**A — permutation.** Model species indices are arbitrary: recovered gene 3 need not be true
gene 3. Comparing J entry-by-entry under an arbitrary labelling is close to meaningless
(this is why the pre-fix pipeline reported `sign_match_frac ≈ 0.44` for A — a number with no
interpretation). `scoring/permutation.py` minimises the mismatch over permutations of the
**unobserved** species, holding observed indices pinned to their assigned rows.

> **Measured caveat — for the configured A, the permutation search is degenerate.**
> With N=3 and 2 observed channels there is exactly **one** unobserved index, so there is
> nothing to permute: `n_permutations_searched == 1` and the aligned score equals the
> identity score. Alignment only starts doing work at ≥2 hidden species (N=4, m=2 → 2
> candidates). A test asserts this degeneracy so it can never be quietly misread as
> "alignment was applied and helped". The residual labelling freedom in A is a **sign/scale
> gauge on the single latent field**, not a permutation — treat `latent_field_quality`
> (correlation, invariant to offset and rescale) as the honest identifiability read for A,
> and use `|r|` since phase inversion is part of that gauge.

**B — there is no correct 3×3.** The true J is 2×2, so any full-matrix comparison is a
category error; the old code returned a silent `NaN`. `scoring/overparam.py` asks the right
question instead: is the spare species **inert**?

- `extra_norm_frac` — fraction of ‖J‖_F carried by the spare species' row+column
  (union mask; the diagonal is not double-counted).
- `max_abs_coupling_to_extra` / `..._from_extra` — the strongest **single** edge. A mean
  hides one strong spurious edge behind many near-zero ones; the max does not.
- `spare_species_inert` — the headline pass/fail.
- `observed_block_agreement` — did it get the **real** 2×2 system right?

> **The threshold is uncalibrated.** `coupling_threshold` defaults to `0.05` (relative). That
> is a starting point, not a measured value — calibrate it on your CUDA runs by looking at
> the distribution of `max_abs_coupling_*` in the `expB_control_matched` arm, where by
> construction there is no spare species to be active.

## Reading the results table

```bash
rngrn --runs-root experiments benchmark --degradation
```

Rows are grouped by `(arm, dataset, n_true, n_model)`; controls sort first. Arms are
classified automatically in `fit()` as `fully_observed`, `hidden_channel`,
`overparameterised`, or `underparameterised`.

- **`subblock_sign_match_mean` is the one column comparable across every arm** — it restricts
  to the observed species, so it is well-defined whatever N the model assumed. Read this for
  cross-arm degradation.
- `sign_match_aligned_mean` applies only where a same-size true J exists (A and the controls).
- `spare_inert_frac` applies only to the over-parameterised arm.
- NaN in a column means *not applicable to this arm*, by construction — not a failure.

**Always read an arm against its control**, never against zero. `expA_control_full` is the
same data fully observed; `expB_control_matched` is the same data with a correctly-sized
model. Degradation is the difference.

## What a negative result looks like — and why it is still a result

This is a degenerate inverse problem; a null outcome is informative, not a failure:

- **A fails** (latent correlation near zero, observed sub-block no better than the control's
  spread) → a hidden gene is *not* identifiable from one frame under this objective. That is
  a real limit on the method, and worth knowing before trusting any recovered N-gene network
  where you cannot observe every gene.
- **B fails** (`spare_species_inert=False` with a confident third gene) → the method invents
  structure when over-parameterised. This directly calibrates how much to trust a recovered
  N=3 network when the true gene count is unknown — the situation in any real application.

Report either outcome as measured. Do not tune until the result flips.

## Running them

```bash
# one-time: drop the datasets in (docs/LOCAL_DATA_SETUP.md), then
rngrn scan-datasets

# The two CONTROL arms run as-is. Launch trainers through the memory guard (CLAUDE.md §7a).
for c in expA_control_full expB_control_matched; do
  bash scripts/guarded_run.sh rngrn --runs-root experiments train --config configs/$c.yaml
done
rngrn --runs-root experiments benchmark --degradation
```

> **Corrected 2026-08-04 — the two hidden-channel arms do NOT run as written.**
> `expA_hidden_channel` and `expB_overparam` are both `N=3, m=2`. `recover.py:376` refuses
> `m < N` when the stationarity residual has weight 0, and `resid: 0.0` is the default
> (`configs/base.yaml:33`, `strategy: fixed`). Both arms therefore raise `ValueError` before
> a single Adam step. The old four-arm loop above silently produced a `--degradation` table
> containing only the controls, which reads as "the experiments found nothing" rather than
> "the experiments did not run".
>
> To run them you must give the residual a non-zero weight — the only term the latent fields
> enter:
>
> ```bash
> bash scripts/guarded_run.sh rngrn --runs-root experiments train \
>   --config configs/expA_hidden_channel.yaml -o loss.weights.resid=<nonzero>
> ```
>
> There is **no known-good value** for it: `TUNING.md:102` records that hidden-channel
> (m &lt; N) recovery has no objective at `resid = 0` and that the alternative is unproven.
> Treat choosing one as a science decision under CLAUDE.md §10 —
> calibrate it against the matched control and record it in `docs/DECISIONS.md`.

Sweep across arms — `model.observed_idx` (rotate which channel is hidden) and
`data.sample_key` (more samples) are legal **experiment-arm axes** in `optim/sweep.py`;
data-*generation* params remain forbidden, since varying those compares different problems.

Use several seeds: rank arms by a seed aggregate and read `kstar_identifiability_std`, the
spread across seeds that all landed in-regime. For a degenerate inverse problem the spread
matters as much as the mean.

> **Fixed 2026-08-04 — this column was NaN on every row until today** (`DECISIONS.md`
> D-EVID-13). `benchmark.build_table` grouped on `config_id`, which hashes `train.seed`, so
> every K-seed replicate became K groups of one: `n_seeds` was always 1 and the spread had
> nothing to spread over. Grouping now keys on the seed-independent `arm_id`, and the table
> also records `seeds` — the list actually aggregated.
>
> **Runs recorded before today carry no `arm_id` and cannot be regrouped**, so they still
> report one row per seed with `arm_id = null`. Reading this column requires targets re-run
> after 2026-08-04.

## Status

Harness only — wired, dry-run on CPU, **not tuned**. The dry run uses 6 Adam steps and
recovers nothing meaningful; that is expected. Tuning and the actual quantitative results are
the CUDA stage. See TUNING.md.
