# experiments/form_compare/ — paper comparison: `nc1` vs `competitive` robustness

Two cells, one target-form recipe borrowed from the `nc1` winning cell
(`c2_P_t8k8_consol`), run under `model.form=competitive` instead. Deliverable is a
**measured comparison with a disclosed confound**, not a pass/fail verdict — see
`docs/DECISIONS.md::D-FORMCOMP-1` for the full design, why the hyperparameters carry a
form-specific confound, and what was rejected.

## Exact commands

Common flags, run through `scripts/guarded_run.sh` (mandatory — see
`.claude/skills/run-training/SKILL.md`). Hyperparameters are copied verbatim from
`experiments/claim5_obs_noise/run_cells.sh` (the `nc1` `c2_P_t8k8_consol` recipe), with
`--form nc1` -> `--form competitive` and no observation-noise overrides:

```bash
bash scripts/guarded_run.sh timeout 2400 .venv/bin/python -m rngrn.cli \
  --runs-root experiments/form_compare/comp_XXXX target-report \
  --config configs/m3_registry.yaml --dataset-id three_gene_qvar --sample-key sample_XXXX \
  --form competitive --seeds 0 1 2 3 4 5 6 7 --workers 4 \
  -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
  -o train.n_restarts=64 -o train.lbfgs_steps=0 -o train.adam_steps=400 \
  -o loss.weights.param_prior=1.0 -o loss.weights.turing=8.0 -o loss.weights.kstar=8.0
```

| runs-root | sample_key | form |
|---|---|---|
| `comp_0001` | `sample_0001` | `competitive` |
| `comp_0004` | `sample_0004` | `competitive` |

The exact sequential driver used: `experiments/form_compare/run_cells.sh`. Cells run one at
a time (never concurrently) because they queue on the repo-global trainer flock, and a
second concurrent pool would exceed the host memory ceiling even after the flock clears
(CLAUDE.md §7a). Both cells ran through `configs/m3_registry.yaml --form competitive`, not
`configs/nc1_m3_registry.yaml -o model.form=competitive` — the two configs differ by more
than the `form` field textually (comments, `tracking.run_name`), but
`target_report.py::run_target_report` overrides `model.form`/`data.dataset_id`/
`data.sample_key` from the CLI flags onto the loaded config regardless of the file's own
values, so the two invocations are functionally identical. See D-FORMCOMP-1 for the full
reasoning.

## Seeds

`0 1 2 3 4 5 6 7`, identical across both targets and identical to the `nc1` comparison
(`c2_P_t8k8_consol`) — the same 8 restart-seed conditions, so any difference between forms
is attributable to the form (and the confound named below), not to a different restart-seed
draw.

## What this compares against

- **`nc1`:** the already-committed `experiments/c2_P_t8k8_consol/runs/*/results/train_results.json`
  in the MAIN checkout (`/home/benja/projects/personal/rngrn/rngrn-pipeline`), read
  read-only — 16 runs, `sample_0001` (x8) + `sample_0004` (x8). Not re-run.
- **Population baseline:** `/home/benja/projects/personal/rngrn/rngrn-pipeline/experiments/exp11_robustness_baseline.csv`,
  127 generator systems × 400 perturbation draws at 1%/4.8%/10%/20%.

## Analysis and figure

- `experiments/form_compare/analyze.py` — reads both forms' run records plus the baseline
  CSV, writes `numbers.json` / `numbers.md` (per-target and pooled: `recovered_frac`,
  `turing_frac`, `topology_consistency`, median/mean/min/per-seed `turing_volume_{1,4p8,10,20}pct`,
  `kstar_fft_rel_err`, `kstar_rel_err`, `trivial_kstar_err`, `morphology_match_frac`,
  `morphology_distance`, `plausibility_score_mean`).
- `experiments/form_compare/make_figures.py` — reads `numbers.json`, writes
  `experiments/form_compare/figures/1_form_robustness_comparison.png`: population
  distributions at the four perturbation levels with both forms' pooled recovered models
  overlaid, plus the pre-registered §3.2 bars at 4.8%/10%.

Run in order: `analyze.py` then `make_figures.py`, both from the repo root with the
worktree `.venv`.

## The disclosed confound — read before the numbers

`turing=8.0` / `kstar=8.0` were tuned **for `nc1`** (`docs/C2_NC1_TUNING.md` §2) to police a
mechanism measured on `nc1` — the near-immobile-third-node route by which `turing=8` alone
buys its Turing rate at the cost of an exploding k\* error. `competitive`'s own tuning
campaign (`docs/C1_COMPETITIVE_TUNING.md`) never swept a `kstar` axis at all; its own best
cell (`turing8`) uses `turing=8` alone, at the library default `kstar=1.0`, and was measured
only on `sample_0000`/`sample_0003` — not on the targets used here. **If `competitive`
underperforms `nc1` in this comparison, that must be read as "underperforms at
`nc1`-tuned hyperparameters", never as an intrinsic property of the `competitive` form** —
the honest alternative reading is that `competitive` needs its own `kstar` sweep, which this
unit does not have time to run before the paper deadline. See D-FORMCOMP-1 for the full
reasoning and for context (not a gate) on Stage-0's measured prior-difficulty gap between
the two forms.

## Caveats

- **Two independent targets, not more.** `sample_0001` is the same underlying system as
  `turing_labyrinth/sample_0000` at a different simulation grid (byte-identical Jacobian /
  `interaction_matrix` / `params_json` / `sim_seed`) — see
  `worktrees/paper-pack/docs/PAPER_RESULTS_PACK.md`, "three facts", fact 1. Only
  `sample_0004` is genuinely independent. This binds `nc1` and `competitive` identically
  here, since both use the same two targets.
- **`nc1`'s recovered models are the tuning-half data** (PREREGISTRATION §1a burned range),
  same as `competitive`'s here — neither side of this comparison is held-out evidence for a
  §3 pass condition.
- **The pre-registered §3.2 bars are reported, not newly gated.** Median
  `turing_volume_10pct` ≥ 0.90 and median `turing_volume_4p8pct` ≥ 0.95 are shown beside
  each form's numbers because they are pre-registered reference points already used
  elsewhere in the paper (claim 4) — this unit does not invent a per-form pass condition.
- **"Turing-unstable" and "patterns" are different claims** (CLAUDE.md §7c) —
  `turing_volume_*` / `recovered_turing` are dispersion claims; `morphology_match` is a
  rollout claim. Read them separately.
