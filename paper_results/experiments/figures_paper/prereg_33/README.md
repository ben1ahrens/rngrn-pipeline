# PREREGISTRATION §3.3 — the within-class distance threshold, computed and adjudicated

**Status: closes the one pre-registered pass condition in §3.3 that had never been
computed.** Written 2026-08-18.

## Exact wording (docs/PREREGISTRATION.md §3.3, quoted verbatim)

> `morphology_match` true, **and** `morphology_distance` ≤ the 75th percentile of
> within-class distances in `three_gene_train`, **and** `kstar_fft_rel_err` ≤ **8.3 %**,
> **and** `trivial_kstar_err` reported in the same record.

This doc computes the second clause's threshold — "the 75th percentile of within-class
distances in `three_gene_train`" — which had no number attached anywhere in the repo
before this run, and adjudicates the c2_P campaign's 16 runs against it.

## Dataset

`dataset_id = three_gene_train` (`data/datasets/three_gene_train/`, main-checkout
canonical store; symlinked into this worktree for the run — this branch's declutter
commit removed non-canonical dataset directories, three_gene_train among them, but the
payload was never touched). 88 samples, labelled by the payload's own per-sample
`morphology` HDF5 attribute: **42 labyrinth / 42 spots / 4 stripes** — matches the count
already recorded in `src/rngrn/scoring/morphology.py`'s `TRAIN_CLASS_MEANS` docstring
(measured 2026-07-26), confirming this is the same split.

No dataset-id ambiguity: `three_gene_train` is a literal, exact `dataset_id` match to the
prereg's wording — no candidate dataset (`three_gene_classical_train`, `three_gene_val`,
etc.) shares that id.

## Method

For every unordered pair of samples **within the same class**, `morphology_distance(a, b)`
(`src/rngrn/scoring/morphology.py`, default `scale=TRAIN_SCALE` — itself data-derived from
this exact 88-sample split, so no external scale was introduced). Pairwise counts:
labyrinth 861, spots 861, stripes 6 (C(42,2), C(42,2), C(4,2)).

**Wording ambiguity, made explicit and both computed.** "The 75th percentile of
within-class distances in three_gene_train" grammatically names ONE distribution over ONE
number — it does not say "per class". Two honest readings exist:

- **POOLED (primary — matches the literal wording)**: all 1,728 within-class pairs from
  all three classes concatenated into one distribution, one 75th percentile.
- **PER-CLASS (context)**: within-class pairs computed separately per label, each with its
  own 75th percentile.

The adjudication below uses the **pooled** threshold as the pass/fail criterion (it is
what the sentence literally describes); per-class values are reported alongside for
context and are visibly very different — the stripes class in particular (75th percentile
computed from only 6 pairs drawn from 4 samples) is far higher and should be read as
weakly determined, consistent with the existing "stripes is weakly characterised"
limitation already documented in `morphology.py`.

## Result — threshold

| set | n_samples | n_pairs | p75 |
|---|---|---|---|
| **pooled (primary)** | 88 | 1728 | **2.1072** |
| labyrinth | 42 | 861 | 1.8409 |
| spots | 42 | 861 | 2.4159 |
| stripes | 4 | 6 | 6.1343 (weak — 6 pairs) |

## Adjudication — c2_P campaign (`experiments/c2_P_t8k8_consol/runs/*/results/train_results.json`)

16 runs. One run (`..._190648_seed3`) has `morphology_scored: "target_only"` — no
model-side field was comparable, so `morphology_match`/`morphology_distance` do not exist
for it and it fails the distance sub-condition by construction (cannot be evaluated as
passing).

- **Distance sub-condition alone** (`morphology_distance` ≤ pooled p75 = 2.1072):
  **7 / 16 pass**.
- **Full §3.3 conjunction** (all four sub-conditions — match, distance, `kstar_fft_rel_err`
  ≤ 8.3%, `trivial_kstar_err` reported): also **7 / 16 pass** — every run that passes the
  distance sub-condition also clears `kstar_fft_rel_err` ≤ 8.3% and reports
  `trivial_kstar_err`, and no run that fails distance is rescued into passing full §3.3.

**The result splits cleanly along the two seed batches in this campaign**: all 7 passing
runs are `nc1_m3_registry_20260803_190648_seed{0,1,2,4,5,6,7}` (distances 0.27–0.59, well
inside the threshold); every run in the `..._192351_seed*` batch fails the distance
sub-condition despite `morphology_match = True` for all 8 of them — their distances
(2.74–3.80) are 1.3–1.8x the pooled threshold, i.e. their recovered fields land the right
NEAREST-CENTROID class but are further from the target than typical same-class variation
in the training set. Per-run numbers, `morphology_pred_target`, and per-class-threshold
comparison are in `results.json`.

## Command

```
cd worktrees/lift-ladder
.venv/bin/python experiments/figures_paper/prereg_33/compute_prereg_33.py
```

Reads `data/datasets/three_gene_train/payload.h5` (main-checkout canonical store, via
symlink) and every `experiments/c2_P_t8k8_consol/runs/*/results/train_results.json` in the
main checkout. Writes `results.json` and `distance_distribution.png` in this directory.

## Firewall / evidence-discipline notes

- This script lives under `experiments/figures_paper/prereg_33/`, not `src/rngrn/` or
  `scripts/` — it is a standalone, scoring-side, post-hoc analysis, not part of the
  recovery path, and is not import-reachable by bare top-level name, so it is not subject
  to `tests/test_firewall.py`'s `FORBIDDEN` list (CLAUDE.md §5's `scripts/` gap does not
  apply here for that reason).
- The `morphology` ground-truth label is a plain per-sample HDF5 attribute on the payload
  (not part of `AnswerKey`/`answer_key_keys`), read directly via `h5py` — the same pattern
  `tests/test_morphology_scoring.py` already uses.
- This is a threshold computed from real data (not a plumbing check) and calibrated
  against the training set's own measured spread, per CLAUDE.md §8/§10. It should be
  recorded in `docs/DECISIONS.md` as closing the §3.3 threshold gap.
