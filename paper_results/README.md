# paper_results/ — all collected evidence from the 2026-08-19 paper sprint

Everything the paper sprint produced — run data, figures, initial conditions, and the
scripts that drove or plotted them — copied from six source branches into one directory on
one branch (`feature/paper-results`), so the evidence can be read, re-plotted, and extended
without checking out six worktrees. First collected 2026-08-19 (claims 1/2/3/4/5/6, the
`PAPER_RESULTS_PACK.md` set); extended the same day with two supplementary claims run on
separate branches after the pack was written: form-robustness (`PAPER_CLAIM_FORMCOMP.md`)
and weight-noise (`PAPER_CLAIM_WNOISE.md`, a negative result). Nothing here is new
measurement; every file is a verbatim copy of tracked content at the commits below, except
the four `figures_paper/` scripts, whose *path constants only* were patched (see "What was
modified"), and `experiments/claim_wnoise/figures/wnoise_curves.png`, which was untracked
(gitignored) on its source branch and is included here as the regenerated raw output.

## Experiments run — the key

| Claim | Doc | Data | Figures |
|---|---|---|---|
| 1 — Robust Turing pattern recovery, N=3 | `PAPER_RESULTS_PACK.md` §Claim 1 | `experiments/c2_P_t8k8_consol/` (16 runs), `experiments/diag_fft/d5/` (10 runs) | `experiments/figures_paper/recovery/figures/` |
| 2 — Topology/params robust under dynamical lift | `PAPER_RESULTS_PACK.md` §Claim 2, `docs/DECISIONS_excerpts.md::D-PAPER-1` | `experiments/lift_paper_demo/`, `experiments/lift_ladder/v4/` | `experiments/figures_paper/robustness/figures/2_lift_invariance.png`, `experiments/lift_paper_demo/figures/` |
| 3 — Recovery from partial observation (hidden slow channel) | `PAPER_RESULTS_PACK.md` §Claim 3, `docs/DECISIONS_excerpts.md::D-PAPER-2` | `experiments/claim3_hidden_slow/` (3 arms × 8 seeds) | none generated (dispersion/rollout metrics only) |
| 4 — Benchmark model robustness | `PAPER_RESULTS_PACK.md` §Claim 4 | `experiments/exp11_robustness_baseline.csv` (127 systems × 400 draws), `experiments/c2_P_t8k8_consol/` | `experiments/figures_paper/robustness/figures/1_robustness_benchmark.png` |
| 5 — Robustness to noisy training data (obs-noise) | `PAPER_RESULTS_PACK.md` §Claim 5, `docs/DECISIONS_excerpts.md::D-CLAIM5-1` | `experiments/claim5_obs_noise/` (4 sigma levels × 8 seeds) | none generated (table only) |
| 6 — GRN topology / non-identifiability | `PAPER_RESULTS_PACK.md` §Claim 6 | `experiments/figures_paper/topology/data/` | `experiments/figures_paper/topology/figures/` (7 PNGs) |
| Supplementary — form robustness (nc1 vs competitive) | `PAPER_CLAIM_FORMCOMP.md`, `docs/DECISIONS_excerpts.md::D-FORMCOMP-1` | `experiments/form_compare/` (2 targets × 8 seeds), plus a second copy of `experiments/claim5_obs_noise/` (identical, not re-copied) | `experiments/form_compare/figures/1_form_robustness_comparison.png` |
| Supplementary — weight noise (NEGATIVE result) | `PAPER_CLAIM_WNOISE.md`, `docs/DECISIONS_excerpts.md::D-WNOISE-1` | `experiments/claim_wnoise/` (4 sigma_w levels × 8 seeds + smoke) | `experiments/claim_wnoise/figures/wnoise_curves.png` (untracked source, included here) |
| Suite state / NOT-SUPPORTABLE list | `PAPER_RESULTS_PACK.md` §Suite state, §NOT-SUPPORTABLE | — | — |

Each claim's own `README.md` (inside its `experiments/<name>/` directory) has the exact
CLI commands, per-seed wall times, and caveats; each decision entry in
`docs/DECISIONS_excerpts.md` has the full design rationale and what was rejected.

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
| `experiments/form_compare/` | `feature/paper-form-robustness` @ bfb9474 | same path |
| `experiments/claim_wnoise/` | `feature/paper-weight-noise` @ 58445df | same path (incl. untracked `figures/wnoise_curves.png`) |
| `PAPER_RESULTS_PACK.md` | `docs/paper-results-pack` @ f91ee73 | `docs/PAPER_RESULTS_PACK.md` |
| `docs/PAPER_CLAIM_FORMCOMP.md` | `feature/paper-form-robustness` @ bfb9474 | same path |
| `docs/PAPER_CLAIM_WNOISE.md` | `feature/paper-weight-noise` @ 58445df | same path |
| `docs/DIAGNOSTICS_lift.md` | `feature/lift-ladder` @ 2f50fff | `docs/DIAGNOSTICS_lift.md` |
| `docs/ROBUSTNESS_MEASUREMENT.md` | `main` @ 48441e4 | `docs/ROBUSTNESS_MEASUREMENT.md` |
| `docs/DECISIONS_excerpts.md` | extracted from the five branches above | entries only; see its header |
| `scripts/lift_paper_demo.py`, `scripts/lift_ladder.py` | `feature/lift-ladder` @ 2f50fff | `scripts/` |
| `scripts/exp11_robustness_baseline.py` | `main` @ 48441e4 | `scripts/` |

The pack doc's own provenance map (`LL/`, `MAIN/`, `C3/`, `C5/` prefixes) maps onto this
directory as: **all four prefixes → `paper_results/experiments/`** — the source
`experiments/` subtrees are disjoint, so they merge here without collision. The two
supplementary claims (`form_compare/`, `claim_wnoise/`) were run on their own branches
after the pack's four-branch collection and are not part of that provenance map; they
merge in the same way, into their own disjoint subtrees.

Note: `feature/paper-form-robustness` @ bfb9474 also carries its own copy of
`experiments/claim5_obs_noise/` (it was built as the claim-5 recipe's follow-on branch) —
identical to the copy already collected here from `feature/paper-claim5-noise`, so it was
not re-copied.

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
- `experiments/form_compare/{run_cells.sh,analyze.py,make_figures.py}` — needs
  `feature/paper-form-robustness`'s `src/` (same obs-noise knob plus the `competitive`
  form config used for the comparison).
- `experiments/claim_wnoise/{run_cells.sh,driver_top.sh,analyze.py}` — needs
  `feature/paper-weight-noise`'s `src/` (`TrainConfig.weight_noise_sigma/weight_noise_seed`
  at `recover.py::_weight_noise_perturb`, not on `main` yet).
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
  and `caption_data.md` (figures), `docs/ROBUSTNESS_MEASUREMENT.md` (exp11 baseline),
  `experiments/form_compare/run_cells.sh` + driver logs (form-robustness),
  `experiments/claim_wnoise/{run_cells.sh,driver_top.sh}` + driver logs (weight-noise,
  including the 0p20 cell's flock-timeout retry — see `D-WNOISE-1`).

## Branch context

This branch (`feature/paper-results`) is cut from `main` @ 48441e4 plus one cherry-pick:
`fix: classify eval/lifted.py as recovery-side` (6999f5f from `fix/main-green`) — without
it `main`'s test suite fails `test_every_loss_and_eval_module_is_classified` and the
pre-push hook blocks every push. Both paper experiment branches carry the identical fix.

None of the source branches are merged anywhere; the owner validates first. This branch
only *copies* their tracked artifacts — merging the source branches later will not
conflict with this directory (all paths here are new under `paper_results/`).

`feature/paper-form-robustness` and `feature/paper-weight-noise` are two further branches
cut the same day, after `docs/paper-results-pack` was written — they are additional claims
under investigation, not part of the original six-claim pack. Both also carry the
`eval/lifted.py` firewall-classification fix.
