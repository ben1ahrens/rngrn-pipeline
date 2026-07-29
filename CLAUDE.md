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
- **New work → new branch → new worktree.** Any task that could edit files or add new
  ones starts with a fresh branch checked out in its own **git worktree**, never by
  switching branches in an existing checkout. Merge to `main` only after the tests pass
  **and the user has validated the results** — passing tests alone is not the bar.

  ```bash
  # from the main checkout
  git worktree add ../worktrees/<slug> -b feature/<slug>      # new branch + tree
  cd ../worktrees/<slug>
  python -m venv --system-site-packages .venv                 # see the gotcha below
  .venv/bin/pip install -e ".[dev]" -q
  ```

  Worktrees live in `../worktrees/<slug>` — a **sibling** of the checkout, never nested
  inside it (a nested tree lands inside the parent's working directory and git will try
  to track it). Remove a finished one with `git worktree remove ../worktrees/<slug>`.

  > **Gotcha — each worktree needs its own `.venv`.** An editable install writes an
  > `__editable__*.pth` file holding the **absolute path** of the tree it was installed
  > from. Reusing another tree's `.venv` therefore imports *that* tree's `src/` while you
  > edit this one — tests silently exercise the wrong code. Creating a venv per worktree
  > is cheap (~48 MB): `--system-site-packages` inherits the CUDA torch build from the
  > base conda env rather than downloading its own. Verify after setup with
  > `.venv/bin/python -c "import rngrn,os;print(os.path.dirname(rngrn.__file__))"` — the
  > path must be inside the worktree.

  > **Gotcha — a new worktree looks provisioned with datasets but is not.** Manifests and
  > `datasets.jsonl` are tracked, so they come across with the checkout and
  > `rngrn list-datasets` cheerfully lists every dataset. The `payload.h5` files are
  > gitignored and do **not**, so the first load fails with `FileNotFoundError` deep in
  > h5py. Symlink them from the main checkout rather than copying ~91 MB again:
  >
  > ```bash
  > MAIN=../../rngrn-pipeline
  > for d in "$MAIN"/data/datasets/*/; do
  >   n=$(basename "$d"); [ -f "$d/payload.h5" ] || continue
  >   mkdir -p "data/datasets/$n" && ln -sfn "$d/payload.h5" "data/datasets/$n/payload.h5"
  > done
  > ```
  >
  > Costs ~84 KB, loads correctly, and the symlinks stay gitignored like the payloads.
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

*Corrected 2026-07-29. This section previously said the package was "CPU-only by
construction" with "no device handling anywhere". That is no longer true and the source
wins over the doc (§8).*

- **Device handling exists.** `RNGRN` has `.device`/`.dtype` properties, `recover()` takes
  `device=`, and `model.dispersion_backend` selects `eig` (any N) or `cubic` (exact for
  N ≤ 3 only, rejected at construction otherwise). `torch.set_default_dtype(torch.float64)`
  is still global, for tight autodiff on small N.
- **The GPU loses on the serial path, and this is the measured reason.** One Adam step at
  N=3 is CPU 138 ms vs CUDA 418 ms — CUDA 3× *slower*. Not the maths: `terms.steady_state`
  is a damped Newton with up to 100 **sequential** 3×3 solves, and `torch.linalg.eigvals`
  on small **non-symmetric** matrices has no batched cuSOLVER kernel (~700 µs per matrix
  regardless of batch size).
- **Batched, the same maths strongly favours the GPU.** The closed-form cubic dispersion
  measured CUDA 0.97 ms vs eigvals 156.6 ms on 127 matrices (**162×**), with the crossover
  around 6,400 matrices and 38× at 1e6. So the unlock is **batching restarts × samples**
  — batched Newton, vmap Jacobian, batched cubic dispersion — which is what makes high
  iteration counts on a GPU affordable. This is the standing direction of travel.
- **CPU runs at short step counts are plumbing checks, not results.** A short run verifies
  that a config resolves, data loads, and scoring routes. It recovers nothing meaningful.
  20 cores are available; multiprocessing over seeds is the CPU throughput lever.

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

**Superseded 2026-07-29 by explicit owner instruction.** The previous version of this
section told agents to stop and ask on every science decision. The owner has withdrawn
that: *"You do not need to ask me for all science. If unsure, do some research to verify."*

**So: decide it yourself, and record it.** Run freely on mechanical work *and* on science
decisions — defining or changing a metric, choosing a threshold or pass condition, picking
an estimator, deciding what counts as a control. Do not block on the owner for these.

What replaces asking is **evidence plus a written record**, because this project is headed
for a paper (§8, and `docs/PREREGISTRATION.md`). Every science decision you take must:

1. **Be calibrated against something measured**, not chosen for convenience. A threshold
   needs a baseline or a control behind it. If you cannot calibrate it, mark it
   UNCALIBRATED in the code and in `docs/DECISIONS.md` — an honest placeholder is fine; a
   silent guess dressed up as a choice is not.
2. **Be researched when you are unsure.** Read the primary source rather than inferring
   from a secondary one. This has already caught real errors: four claims in
   `docs/STATE_OF_THE_SCIENCE.md` §2.7 were attributed to Matas-Gil & Endres when two came
   from Kho et al. and two were least-squares results rather than PINN results — traced by
   reading the paper *and* the authors' shipped code. When docs and source disagree the
   source wins; when a citation and the paper disagree the paper wins.
3. **Land in `docs/DECISIONS.md`** with the date, the evidence, and what was rejected.
4. **Be announced loudly if it changes what an existing number means.** A silently
   non-comparable number is worse than a missing one.

Two things still go to the owner rather than being decided unilaterally: anything that
would **weaken a pre-registered pass condition** in `docs/PREREGISTRATION.md` (that file is
binding, and is dated ahead of the runs it judges — which is the entire point of it), and
anything outside the technical scope of the work: publishing, sharing, spending money, or
touching systems beyond this repo.

## 11. Subagents and dynamic workflows

Spawning subagents inside a dynamic workflow is **allowed and encouraged**. Use it.

- **Tier the models.** Reserve the strongest model for genuinely open-ended work — a new
  objective, stiff-integrator numerics, a design decision with no prior art. Ports,
  retrofits, relabels, new metric modules written against a clear spec, and plumbing all
  run fine on a cheaper tier with the effort dialled down. Measured here: a 13-unit wave at
  4 opus + 9 sonnet produced the same quality of result as all-opus, far cheaper.
- **One unit, one worktree, one branch, one PR.** Create the worktrees *before* launching
  the wave — concurrent `git worktree add` contends on `.git/config`, and several agents in
  the first wave lost their upstream-tracking write to a lock collision.
- **Give every unit an anti-conflict rule** when units share files: new config fields
  appended at the end of the dataclass with a comment naming the unit; new metrics as a
  single appended `out.update(...)` line. Then expect to repair the merge anyway — git
  resolves at line granularity and cannot tell that two additions are independent. Keeping
  both sides is right for additive blocks and **wrong** inside a function signature or a
  dict literal, where it silently produces duplicate definitions of which only the last
  takes effect. That exact failure reverted `resid` to 0.3, dropped the anchor weight, and
  discarded `model_seed` in the phase-A merge.
- **`git stash` is repo-global, not per-worktree.** Two agents stashing concurrently will
  apply each other's changes. Do not use it in a parallel wave.
- **Subagents cannot ask the owner.** Whatever a unit needs to decide, it decides under §10
  and reports. A precisely diagnosed defect a unit could not fix is a better outcome than a
  green test that hides it — say which, explicitly.
- **A wave can die wholesale to an upstream outage.** Six phase-B units returned API 500s
  simultaneously. Check whether failures are server-side before diagnosing them as code.
