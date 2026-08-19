# paper_results/ — collected evidence behind PAPER_RESULTS_PACK.md

Everything `PAPER_RESULTS_PACK.md` cites — run data, figures, initial conditions, and the
scripts that drew the plots — copied from its four source branches into one directory on
one branch (`feature/paper-results`), so the pack's evidence can be read, re-plotted, and
extended without checking out four worktrees. Collected 2026-08-19. Nothing here is new
measurement; every file is a verbatim copy of tracked content at the commits below, except
the four figure scripts, whose *path constants only* were patched (see "What was modified").

## Provenance — where every file came from

| subtree here | source branch @ commit | original path |
|---|---|---|
| `experiments/figures_paper/` | `feature/lift-ladder` @ 2f50fff | same path |
| `experiments/lift_paper_demo/` | `feature/lift-ladder` @ 2f50fff | same path |
| `experiments/diag_fft/d5/` | `feature/lift-ladder` @ 2f50fff | same path |
| `experiments/lift_ladder/v4/` | `feature/lift-ladder` @ 2f50fff | same path |
| `experiments/c2_P_t8k8_consol/` | `main` @ 48441e4 | same path |
| `experiments/exp11_robustness_baseline.csv` | `main` @ 48441e4 | same path |
| `experiments/claim3_hidden_slow/` | `feature/paper-claim3-hidden-channel` @ 60fddfb | same path |
| `experiments/claim5_obs_noise/` | `feature/paper-claim5-noise` @ 4515ea1 | same path |
| `PAPER_RESULTS_PACK.md` | `docs/paper-results-pack` @ f91ee73 | `docs/PAPER_RESULTS_PACK.md` |
| `docs/DIAGNOSTICS_lift.md` | `feature/lift-ladder` @ 2f50fff | `docs/DIAGNOSTICS_lift.md` |
| `docs/ROBUSTNESS_MEASUREMENT.md` | `main` @ 48441e4 | `docs/ROBUSTNESS_MEASUREMENT.md` |
| `docs/DECISIONS_excerpts.md` | extracted from the three branches above | entries only; see its header |
| `scripts/lift_paper_demo.py`, `scripts/lift_ladder.py` | `feature/lift-ladder` @ 2f50fff | `scripts/` |
| `scripts/exp11_robustness_baseline.py` | `main` @ 48441e4 | `scripts/` |

The pack doc's own provenance map (`LL/`, `MAIN/`, `C3/`, `C5/` prefixes) maps onto this
directory as: **all four prefixes → `paper_results/experiments/`** — the source
`experiments/` subtrees are disjoint, so they merge here without collision.

## The scripts, and which ones run here

**Figure/analysis scripts — runnable from this branch as-is** (they only read data that is
in this pack, plus dataset payloads and `src/` from the enclosing checkout):

```bash
# from the repo root of this worktree, with its own .venv provisioned
# (payloads must be linked: bash scripts/link_payloads.sh)
.venv/bin/python paper_results/experiments/figures_paper/recovery/make_recovery_figures.py
.venv/bin/python paper_results/experiments/figures_paper/robustness/make_robustness_figures.py
.venv/bin/python paper_results/experiments/figures_paper/topology/make_topology_figures.py
.venv/bin/python paper_results/experiments/figures_paper/prereg_33/compute_prereg_33.py
```

Each writes into its own `figures/` (or `results.json`) next to itself — i.e. it
regenerates the committed outputs in place.

**Run/generation scripts — reference copies, NOT runnable from this branch alone:**

- `scripts/lift_paper_demo.py`, `scripts/lift_ladder.py`,
  `experiments/lift_paper_demo/domain_size/run_domain_panel.py` — need the
  `feature/lift-ladder` branch's `src/` (lift machinery beyond what `main` carries).
- `experiments/claim5_obs_noise/run_cells.sh` — needs `feature/paper-claim5-noise`'s
  `src/` (the `DataConfig.obs_noise_sigma`/`obs_noise_seed` knob at
  `data/gate.py::_apply_obs_noise` is not on `main` yet).
- `experiments/claim3_hidden_slow/` runs were launched by CLI invocations recorded in its
  `README.md` and D-PAPER-2 (no wrapper script); the serial m<N path likewise lives on its
  branch.
- `scripts/exp11_robustness_baseline.py` — runs on `main`'s code; regenerates
  `experiments/exp11_robustness_baseline.csv` (~127 systems × 400 draws, slow).
- All training/experiment launches go through `bash scripts/guarded_run.sh …` (CLAUDE.md §7a).

`tools_extract_decisions.py` is the one-shot extractor that produced
`docs/DECISIONS_excerpts.md`; re-run it if the source branches' DECISIONS.md entries move.

## What was modified relative to the sources

Only the four figure/analysis scripts, and only their root-path constants: they hardcoded
absolute paths to the main checkout (`MAIN`) and the lift-ladder worktree (`WT`/`WORKTREE`).
Both now resolve to this pack root via `__file__`; dataset-payload and `src/` paths resolve
to the enclosing repo checkout. Every data-reading subpath is unchanged. Each patch is
marked with a `# Collected into paper_results/` comment at the site. All four scripts were
re-run after patching (2026-08-19): every PNG regenerated **byte-identically**, and the
headline numbers reprinted match the pack (robustness medians, topology 26/26 distinct /
pooled 0.3333, prereg threshold 2.1072 and 7/16 conjunction). The only diffs were absolute
`payload_path`/`run_dir` provenance strings inside `prereg_33/results.json` and
`topology/data/topology_runs.json`, which now record this checkout; those two files were
restored to the source-branch versions so the committed provenance still names the
checkouts the runs actually came from.

## Initial conditions and data — where they are

- **Per-run configs (what actually ran):** every run directory here carries
  `config/frozen_config.yaml` and `results/train_results.json` (config + provenance,
  including code commit and seeds). Read those, not the launch command (D-EVID-1).
- **Datasets:** payloads are deliberately NOT in git (repo-wide rule). The experiments used
  `three_gene_qvar`, `turing_labyrinth`, and `three_gene_train`; their tracked
  `data/datasets/<id>/manifest.json` (id + checksum) identifies the exact bytes, and the
  canonical payload store is the main checkout's `data/datasets/` (link via
  `bash scripts/link_payloads.sh`). Claim 5's noisy frames are regenerable byte-identically
  from dataset checksum + σ + seed (D-CLAIM5-1) — no noisy payload copies exist on purpose.
- **Launch records:** `experiments/claim5_obs_noise/run_cells.sh` (claim 5),
  `experiments/claim3_hidden_slow/README.md` (claim 3), each figure directory's `README.md`
  and `caption_data.md` (figures), `docs/ROBUSTNESS_MEASUREMENT.md` (exp11 baseline).

## Branch context

This branch (`feature/paper-results`) is cut from `main` @ 48441e4 plus one cherry-pick:
`fix: classify eval/lifted.py as recovery-side` (6999f5f from `fix/main-green`) — without
it `main`'s test suite fails `test_every_loss_and_eval_module_is_classified` and the
pre-push hook blocks every push. Both paper experiment branches carry the identical fix.

None of the source branches are merged anywhere; the owner validates first. This branch
only *copies* their tracked artifacts — merging the source branches later will not
conflict with this directory (all paths here are new under `paper_results/`).
