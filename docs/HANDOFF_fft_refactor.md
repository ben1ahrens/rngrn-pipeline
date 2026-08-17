# HANDOFF — `chore/fft-refactor`: decluttering to the fft-training minimum

Written for a zero-context reader. 2026-08-17.

## What this branch is

A **decluttering branch**, owner-requested. It removes from the working tree everything
not needed by the Fourier-space training effort: non-canonical datasets, unused configs,
pre-fft scripts, and pre-fft experiment runs. It changes **no source code** — `src/rngrn/`
and `tests/` are untouched.

Branched from `feature/fft-training` at `8139599`. Nothing here has been merged anywhere.

## The rule actually applied

> Delete everything named in the request, **except** what the test suite or a
> mechanically-enforced repo rule requires.

The literal request could not be satisfied in full without breaking the suite or the
mandated training path. Where those collided, the suite and the rules won, and the
exception is listed below rather than silently absorbed.

## Verified

- **Suite green, unsandboxed:** `607 passed, 24 skipped` in 298s on this branch.
- **Baseline before deletion, same worktree:** `630 passed, 1 skipped`.
- **Delta: 23 tests moved PASS → SKIP. Zero failures. Collection count unchanged (631).**
  The 23 are real-data tests that skip when their payload is absent; they are not broken,
  they are unexercised. This is a genuine loss of coverage, not a clean no-op.
- No real (non-symlink) `payload.h5` existed in this worktree before deletion
  (`find data/datasets -name payload.h5 -type f` → empty), so **no dataset bytes were
  destroyed**. The canonical store in the main checkout was never touched.
- 791 files / 143,225 lines deleted.

## What was kept, and why

### Datasets — 2 of 14
Kept `turing_spots`, `turing_labyrinth`. These are canonical **by the repo's own
definition**, not by my judgement: `docs/CANONICAL_DATASETS.md` §0, `docs/DECISIONS.md`
D-CANON-1 and D-CANON-4 ("from 2026-08-10, these are the training data source"). Stage 0 of
the fft plan uses `turing_labyrinth/sample_0000`, with `sample_0004` held out.

`data/three_gene_dataset/` (a README-only stub) was also removed.

### Configs — 8 of 15
**There is no "config that generates the canonical datasets."** Canonical generation is
driven entirely by CLI flags plus `data/canonical_selection.json` through
`scripts/canon_select.py` → `scripts/canon_generate.py`; neither script references
`configs/` at all. So that selection criterion picks the empty set, and the kept list is
instead "fft-owned plus test-required":

- `base.yaml` — fft-owned. Carries the five M1 spectral weight keys (`spec_shape`,
  `spec_aniso`, `spec_amp_mean`, `spec_amp_fluct`, `real_moments`), all `0.0`/inert.
- `bio_box.yaml`, `expA_control_full.yaml`, `expA_hidden_channel.yaml`,
  `expB_control_matched.yaml`, `expB_overparam.yaml`, `m3_registry.yaml`,
  `milestone1_schnak.yaml` — each is **loaded directly by tests**; deleting any hard-fails
  the suite.

Deleted: `milestone1_gm`, `milestone2_gm_partial`, `sweep_example`, and all four `nc1_*`.

### Scripts — 20 of 64
- **Canonical-generation chain** (the real answer to "keep what generates the canonical
  datasets"): `canon_select.py`, `canon_generate.py`, `canon_annotate.py`, `gen_tg3.py`,
  `td_figures.py`, `phase_topology.py`.
- **Rule-required infrastructure**: `guarded_run.sh` — `.claude/hooks/guard_trainer.py:49`
  admits a trainer command only if that literal string appears in it, so deleting the file
  would permanently block every `rngrn train/sweep/target-report` with no remediation.
  `link_payloads.sh` — the sanctioned provisioning path named in two skills.
- **All 12 `diag_fft_*.py`** — the fft branch's own diagnostics.

Deleted 44: `exp01`–`exp12` + `_runlog.py`, all `c1_*`/`c2_*`, all `tune_comp_*`,
`stage0*`, `make_figures.py`, `canon_stripes_evidence.py`.

### Experiments — `diag_fft` + one fixture run
- `experiments/diag_fft/` (65 files) — the only directory the fft branch added, and the
  evidence base for D-FFT-9 … D-FFT-13.
- `experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/` — **kept despite
  pre-dating the branch**, because `tests/test_forward_solve.py:36` and
  `tests/test_etdrk4_torch.py:25` hard-code it as a checkpoint fixture, one with an
  explicit `assert CKPT.exists()`. A literal "delete all non-fft experiments" would have
  hard-failed both of the fft branch's own test modules.

Everything else under `experiments/` was deleted (43 of 44 `tune_comp` runs, all other
top-level dirs, all loose `.json`/`.csv`/`.log`/`.jsonl` artifacts).

## NOT done — needs an owner decision

1. **`experiments/figures_report/` (108 files, 68M) still exists.** Deletion was refused
   repeatedly by the environment's permission classifier and I did not work around it.
   That refusal looks correct: `.gitignore` deliberately special-cases this directory
   (`!experiments/figures_report/**/*.png`) as the *report deliverable itself* — the Stage-0
   biological-viability visual check — expressly not a regenerable sweep figure. It is 68M
   of the 107M that `experiments/` still occupies. **Say the word and it goes.**

## Known consequences — read before merging

- **This branch severs the audit trail for many recorded claims.** `experiments/` is tracked
  in this repo *on purpose* (D-PLOT-1, `CLAUDE.md` §9) so that every number in a doc can be
  traced to the run behind it, because the project is headed for a paper. The deleted runs
  are cited as **Evidence** in `docs/DECISIONS.md`, `docs/C1_COMPETITIVE_TUNING.md`,
  `docs/C2_NC1_TUNING.md`, `docs/BIO_VIABILITY.md` and
  `docs/LGEN_TRANSFER_FIRST_RESULT.md`. After this branch those citations resolve only in
  git history, not in the working tree. **Merging this to `main` would break the paper's
  in-tree audit trail.** Recommendation: keep it as an unmerged working branch, or update
  the citing docs first.
- **23 tests are now skipped rather than run.** Green suite, less coverage. Restore with
  `bash scripts/link_payloads.sh` after `git checkout` of the dataset manifests.
- **`docs/` was not touched** — it was not in scope. Many docs now reference deleted
  scripts and runs.
- **`data/datasets/datasets.jsonl` is stale.** It is an append-only scan log and still
  lists all 14 datasets across three historical scans. `rngrn scan-datasets` reports only
  the two canonical sets ("already-indexed") and does not prune. I left the log intact
  rather than rewrite an audit record.
- **`tests/test_firewall.py::FORBIDDEN` still names deleted scripts.** This is deliberate
  and harmless: the entries are strings matched against `src/rngrn` import text, never
  imported. Per the standing non-negotiable, the firewall test was **not** edited.
- **Stale provenance comments.** `src/rngrn/losses/terms.py`, `eval/analysis.py` and
  `recover.py` cite `exp02`/`exp05`/`exp11` in comments; those scripts are gone.

## Reversing any of it

Everything is recoverable — this is a branch, and the canonical payload store was never
touched.

```bash
git checkout HEAD~1 -- <path>       # restore any deleted tracked file
bash scripts/link_payloads.sh       # re-provision dataset payload symlinks
```
