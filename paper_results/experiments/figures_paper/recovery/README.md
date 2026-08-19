# RECOVERY PANELS — paper claim 1 (robust Turing recovery from an N=3 training example)

Figures for the paper's recovery-robustness claim, built from the c2_P campaign (MAIN
checkout, read-only) and the D5 restart-selection diagnostic (this worktree).

## Regenerate

```
.venv/bin/python experiments/figures_paper/recovery/make_recovery_figures.py
```

Reads absolute paths into the MAIN checkout
(`/home/benja/projects/personal/rngrn/rngrn-pipeline/experiments/c2_P_t8k8_consol/`) and
this worktree's own `experiments/diag_fft/d5/`. No `rngrn` import required — `plot_arrays.npz`
is read directly with `numpy.load` (same shape `rngrn.plotdata.load_plot_arrays` returns).

## Contents

- `figures/1_representative_panel.png` — one run's target frame vs recovered rollout
  (shared colour scale), RAPS overlay (target vs model, k* marked), and dispersion sigma(k)
  (recovered vs answer key, k* marked on both). Run: `nc1_m3_registry_20260803_190648_seed4`,
  chosen as the run closest to the campaign's own median `kstar_fft_rel_err`.
- `figures/2_campaign_summary.png` — all 16 c2_P runs: per-run `kstar_fft_rel_err` bars
  against the §3.3 pass line (8.3%), and `turing_volume_10pct` / `turing_volume_4p8pct` bars
  against the §3.2 pass floors (0.90 / 0.95).
- `figures/3_d5_contrast.png` — the D5 diagnostic: 10 seeds recovering one fixed target,
  4 landing on the interior k* (success) and 6 on the k-grid floor (failure), showing the
  outcome is bimodal restart-selection variance, not a spread — the reason recovery is
  reported best-of-64 restarts rather than best-of-1.
- `caption_data.md` — every number a caption needs, with run paths and the mandatory
  caveats (tuning-split, superseded-dataset, best-of-64, 2x8 design) spelled out
  verbatim-ready.

## Source data (read-only)

- `/home/benja/projects/personal/rngrn/rngrn-pipeline/experiments/c2_P_t8k8_consol/` — MAIN
  checkout, 16 runs, `runs.jsonl` + per-run `arrays/plot_arrays.npz`.
- `experiments/diag_fft/d5/` — this worktree, 10 runs, `runs.jsonl` + `analysis.json`.

PNGs are gitignored bulk elsewhere in this repo, but `figures_report/`-style paper figures
are tracked deliverables for this unit — see the commit for what was added.
