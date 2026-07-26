# CLAUDE.md — working conventions for this repository

Instructions for any AI agent working on `rngrn-pipeline`, and a reference for humans.
Two agents work on this tree: **Claude Code** (local terminal, git, refactors) and
**Claude Science** (experiments, measurement, figures, specification). This file is the
shared contract between them — it is the only place either can rely on the other having
read.

Read `README.md` for what the package does, `TUNING.md` for the list of knobs and stubs,
and `docs/` for per-branch handoffs.

---

## 1. Environment

- **Use the repo's own `.venv`.** It is created with `--system-site-packages` and holds
  `pip install -e ".[dev]"`. It has CUDA-capable torch (2.13.0+cu130).
- There is a conda env named `rngrn` that is **CPU-only and deprecated** — do not use it.
  (Claude Science: the registered environment name is `rngrn-dev`.)
- If `import torch` aborts with `OMP: Error #179`, set
  `KMP_AFFINITY=disabled OMP_NUM_THREADS=1`.
- Python ≥3.10. Entry point is the `rngrn` console script (`rngrn.cli:main`).
- CLI config overrides use `-o/--override key=value ...` — there is **no** `--set` flag,
  and `-v/--verbose` is a top-level option that must precede the subcommand.

## 2. Git

- **Conventional branch prefixes, always**: `feature/`, `fix/`, `chore/`, `docs/`.
  Never an unprefixed branch name.
- **Never commit directly to `main`.** Work on a branch and merge.
- **Do not commit on the user's behalf without asking**, unless the work was explicitly
  scoped as "commit it".
- Substantial branches get a `docs/HANDOFF_<topic>.md` written for a **zero-context
  reader**: what changed, what is verified, what is not, and where to start. This is a
  house convention, not a nicety — it is how the two agents hand work to each other.

## 3. Testing

- **The local pre-push hook is the authoritative test run.** Enable once per clone:
  `git config core.hooksPath .githooks`. It runs `pytest -q` against the repo `.venv`, so
  tests exercise the same torch build you develop against, and aborts the push on failure.
  Bypass deliberately with `git push --no-verify`.
- **GitHub Actions is not a signal here.** `.github/workflows/tests.yml` is
  `workflow_dispatch`-only; the `push`/`pull_request` triggers are commented out, not
  deleted. The account's Actions billing has lapsed, so runs are *skipped* and reported as
  failures that have nothing to do with the code. Do not diagnose those as code failures.
  Restoring CI = uncommenting the two triggers.
- Current suite: **77 tests**, all CPU, seconds to run. Keep them green.

## 4. House style

- **Fail loud.** Raise rather than returning a fallback, a zero, or a silent NaN. A NaN
  that *is* meaningful (a metric undefined for an experiment arm by construction) must be
  documented as deliberate at the site that produces it.
- **Docstrings must not claim more than the code does.** A prior audit caught overstated
  provenance in this repo. If a function delegates to something simpler, say so.
- **Run-index rows must be flat scalars** (sqlite/jsonl backed). Rows are built by
  `dict.update()` merge, never `**`-expansion into a `dict()` literal — scorers return
  keys that collide with run-identity keys. Run identity wins on collision.
- Label unproven code honestly. `TUNING.md` tags every knob `[TUNE]`, `[IMPL]`, or
  `[VALIDATE]`; scaffold modules say SCAFFOLD in their provenance line.

## 5. The data firewall

Recovery must never see ground truth. `tests/test_firewall.py` is a **static AST audit**,
not a convention — it parses each recovery-side module and asserts the forbidden names do
not appear in its imports.

- Recovery-side modules: `model.py`, `observables.py`, `recover.py`, `losses/{terms,
  total,weighting}.py`, `eval/{rollout,numerics,dynamical,analysis,topology}.py`.
- Forbidden imports: `rd_models`, `data.solver`, `data.cache`, `AnswerKey`, `answer_key`.
- The answer key's *contents* are read in **exactly one place**: `validate.score_recovery`.
  `train.fit` holds the key and passes it to the scorer — it touches only
  `n_species_true`, to classify the experiment arm — and `_resolve_recovery_input` is the
  firewall boundary where `(RecoveryInput, AnswerKey)` are separated. Neither is
  recovery-side.

**Known gap:** `rngrn.scoring` is *not* in the enforced `FORBIDDEN` list, though the
design intends it to be scoring-side only. No recovery-side module imports it today, so
the codebase is clean — but the rule is unguarded. Add it to `FORBIDDEN` when convenient.

## 6. Datasets

- A dataset's **directory name under `data/datasets/` is its `dataset_id`.** A config's
  `dataset_id` must match the directory exactly. (An id/directory mismatch silently
  disabled both Experiment B arms for the life of a branch.)
- `payload.h5` is gitignored; `manifest.json` and `data/datasets/datasets.jsonl` are
  **tracked**, so the repo records which dataset a run used, by id and checksum, without
  carrying the bytes.
- **Consequence to remember:** checking out a branch that predates the tracked manifests
  *deletes them from disk*. Payloads survive. Regenerate with `rngrn scan-datasets`
  (idempotent, inspects in place, no copying).
- A payload with no manifest fails loud — the loader will not guess.
- Per-machine setup is documented in `docs/LOCAL_DATA_SETUP.md`.

## 7. Compute reality

- **This package is CPU-only by construction.** `model.py` and `losses/terms.py` both call
  `torch.set_default_dtype(torch.float64)`, and there is **no device handling anywhere** —
  no `.to(device)`, no `cuda`, no `device` config field. `utils.py` only *records* whether
  CUDA exists, for provenance.
- Cost is dominated by the float64 autodiff Jacobian and the 400-point eigendecomposition
  in `model.dispersion()`, not by frame size — measured ~7 ms per Adam step at both
  N=2@128² and N=3@96². Consumer GPUs are weak at float64.
- The refactor that would make a GPU worthwhile is **batching restarts × samples** (vmap
  over the Jacobian, batched eigvals). It is not implemented: `recover()` loops restarts
  sequentially and `fit()` handles one frame at a time.
- **CPU runs are plumbing checks, not results.** A short-step run verifies that a config
  resolves, data loads, and scoring routes correctly. It recovers nothing meaningful.

## 8. Evidence discipline

The single most important rule in this file.

- **Dry-run numbers are never findings.** Say "the harness runs", never "recovery
  achieves X". Any number offered as a result must come from a tuned run on real data.
- **Thresholds are calibrated, never inherited.** A number in a doc is a placeholder until
  something measured it on *this* data. Two live examples: the `~15%` k\* tolerance in
  `TUNING.md` dates from the initial template commit and refers to reference frames that
  the current solver cannot even generate; `coupling_threshold=0.05` for spare-species
  inertness is explicitly uncalibrated. Calibrate both from a control arm.
- **Read an experiment arm against its control, never against zero.** Every experiment
  here ships a matched control for exactly this reason.
- **Verify, don't assert.** Turing instability is confirmed numerically, morphology is
  classified by measurement, and a claim about the code is checked against the code. If
  a doc and the source disagree, the source wins and the doc gets fixed.
- State what is *not* known as plainly as what is. A caveat that only appears after
  someone asks is a caveat that failed.

## 9. Where runs go

`experiments/<purpose>/` inside the repo, passed as `--runs-root` — e.g.
`experiments/dryrun/`, `experiments/tuning/`, `experiments/identifiability/`. The whole
`experiments/` tree is gitignored, so runs sit next to the configs that produced them
without entering version control. Name the subdirectory for the purpose, so a plumbing
check is never mistaken for a result later.

## 10. How far to run before checking in

**Run freely on mechanical work — stop when a science decision appears.**

Refactors, tests, plumbing, docs, packaging, and diagnosis: proceed without asking.

Stop and ask whenever a choice would bias what the results *mean*:

- defining or changing a metric
- choosing a threshold or pass condition
- choosing an estimator when the alternatives disagree materially
- deciding what counts as a control, or which arm is the reference
- anything that would make a later number non-comparable to an earlier one

Bring the evidence to that question rather than a bare choice: measure the alternatives
first, then ask. A decision the user makes on data stays made; one made on a guess gets
revisited.
