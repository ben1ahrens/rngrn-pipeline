# CLAUDE.md — working conventions for this repository

Instructions for any AI agent working on `rngrn-pipeline`, and a reference for humans. Two agents work on this tree: **Claude
Code** (local terminal, git, refactors) and **Claude Science** (experiments, measurement, figures, specification). This file
is the shared contract between them.

Read `README.md` for what the package does, `TUNING.md` for the list of knobs and stubs, and `docs/` for per-branch handoffs.

---

## 1. Environment

- **Use the repo's own `.venv`.** Created with `--system-site-packages`, holds `pip install -e ".[dev]"`. Has CUDA-capable
  torch (2.13.0+cu130).
- The conda env `rngrn` is **CPU-only and deprecated** — do not use it. (Claude Science: the registered environment name is
  `rngrn-dev`.)
- If `import torch` aborts with `OMP: Error #179`, set `KMP_AFFINITY=disabled OMP_NUM_THREADS=1`.
- Python ≥3.10. Entry point is the `rngrn` console script (`rngrn.cli:main`).
- CLI overrides use `-o/--override key=value ...` — there is **no** `--set` flag, and `-v/--verbose` must precede the
  subcommand.

## 2. Git

- **Conventional branch prefixes, always**: `feature/`, `fix/`, `chore/`, `docs/`. Never commit directly to `main` — work on
  a branch and merge, only after tests pass **and the user has validated the results**.
- **New work → new branch → new worktree**, never by switching branches in an existing checkout, as a **sibling** of the
  checkout (`../worktrees/<slug>`), never nested:

  ```bash
  git worktree add ../worktrees/<slug> -b feature/<slug>
  cd ../worktrees/<slug>
  python -m venv --system-site-packages .venv
  .venv/bin/pip install -e ".[dev]" -q
  ```
  Remove one with `git worktree remove ../worktrees/<slug>`.

  > **Gotcha — each worktree needs its own `.venv`.** An editable install's `__editable__*.pth` pins the absolute path of
  > the tree it was installed from; reusing another tree's `.venv` silently runs *that* tree's `src/`. Verify:
  > `.venv/bin/python -c "import rngrn,os;print(os.path.dirname(rngrn.__file__))"` must print a path inside this worktree.

  > **Gotcha — a new worktree looks provisioned but isn't.** Manifests are tracked (`rngrn list-datasets` looks complete)
  > but gitignored `payload.h5` is missing, so the first load throws `FileNotFoundError`. Symlink from the main checkout:
  > ```bash
  > MAIN=../../rngrn-pipeline
  > for d in "$MAIN"/data/datasets/*/; do
  >   n=$(basename "$d"); [ -f "$d/payload.h5" ] || continue
  >   mkdir -p "data/datasets/$n" && ln -sfn "$d/payload.h5" "data/datasets/$n/payload.h5"
  > done
  > ```
- **Do not commit on the user's behalf without asking**, unless explicitly scoped as "commit it". Substantial branches get
  a `docs/HANDOFF_<topic>.md` for a **zero-context reader**: what changed, what is verified, what is not, where to start.

## 3. Testing

- **The local pre-push hook is authoritative**: `pytest -q` against the repo `.venv`, aborting the push on failure. Bypass
  with `git push --no-verify`; run on demand with `git hook run pre-push`.
- **Enable once per REPOSITORY** (shared by every worktree via the one `.git/config`): `git config core.hooksPath .githooks`.
- **Claude Science cannot set this** — the sandboxed `.git/config` is read-only, so `git config` fails with
  `Device or resource busy`. Inject the setting per push instead: `git -c core.hooksPath=.githooks push ...`. Do not push
  from the sandbox without it, or the authoritative test run is silently skipped.
- **GitHub Actions is not a signal.** `.github/workflows/tests.yml` is `workflow_dispatch`-only (billing lapsed); runs are
  skipped and reported as unrelated failures. Restoring CI = uncommenting the `push`/`pull_request` triggers.
- Current suite: **462 passed, 1 skipped**, all CPU, ~3 minutes (measured 2026-08-04 on `feature/turing-training`).
  Re-measure rather than trust this line; update it when it moves.

## 4. House style

- **Fail loud.** Raise rather than a fallback, a zero, or a silent NaN. A NaN that *is* meaningful (undefined for an arm by
  construction) must be documented as deliberate at the site that produces it.
- **Docstrings must not claim more than the code does.** A prior audit caught overstated provenance here.
- **Run-index rows are flat scalars**, built by `dict.update()`, never `**`-expansion — scorer keys can collide with
  run-identity keys, and run identity must win.
- Label unproven code honestly: `TUNING.md` tags every knob `[TUNE]`, `[IMPL]`, or `[VALIDATE]`; scaffold modules say
  SCAFFOLD in their provenance line.

## 5. The data firewall

Recovery must never see ground truth. `tests/test_firewall.py` is a **static AST audit**: it parses each recovery-side
module and asserts forbidden names don't appear in imports.

- Recovery-side: `model.py`, `observables.py`, `recover.py`, `losses/{terms,total,weighting}.py`,
  `eval/{rollout,numerics,dynamical,analysis,topology}.py`.
- Forbidden: `rd_models`, `data.solver`, `data.cache`, `AnswerKey`, `answer_key`.
- The answer key's *contents* are read in exactly one place, `validate.score_recovery`. `train.fit` holds the key and
  touches only `n_species_true`; `_resolve_recovery_input` is the firewall boundary separating `(RecoveryInput, AnswerKey)`.

**Known gap.** `rngrn.scoring` *is* guarded (three tests assert `"scoring" not in imports` across the recovery-side list) —
only `test_firewall.py`'s own `FORBIDDEN` list omits it. `history.py:45` imports `.scoring.plausibility.d_ratio_of` into the
Adam loop (`recover.py:245,492`) — not a truth leak (`d_ratio_of` is a pure function of the model's own `D`) but unaudited
and unlisted anywhere. **The real gap: the audit is a name allowlist hand-copied into four files** (`test_firewall.py`,
`test_plot_arrays.py`, `test_permutation_scoring.py`, `test_morphology_scoring.py`, same eleven names each) — `history.py`
and `eval/lgen_eval.py` are in none of them, so a new module is unaudited by default and the suite stays green. Fix: define
`RECOVERY_SIDE` once, and test that every module under `losses/`, `eval/`, plus `history.py`, is on it or an explicit
`SCORING_SIDE` list.

## 6. Datasets

- A dataset's **directory name under `data/datasets/` is its `dataset_id`**; a config's `dataset_id` must match exactly (an
  id/directory mismatch once silently disabled both Experiment B arms for a branch's whole life).
- `payload.h5` is gitignored; `manifest.json` and `datasets.jsonl` are tracked, so the repo records which dataset a run
  used without carrying the bytes.
- Checking out a branch that predates the tracked manifests **deletes them from disk** (payloads survive). Regenerate with
  `rngrn scan-datasets` (idempotent, in place).
- A payload with no manifest fails loud. Per-machine setup: `docs/LOCAL_DATA_SETUP.md`.

### 6a. Generated payloads must be HARVESTED before a worktree dies

A unit once generated two datasets inside its own worktree; manifests are tracked but `payload.h5` is not, so removing the
worktree kept the manifest and lost the data. The standard symlink loop also **overwrites a real generated file with a
dangling symlink** if run against a payload not yet in the canonical store. Rules:
- `rngrn-pipeline/data/datasets/<id>/payload.h5` (MAIN checkout) is the canonical store; worktrees hold symlinks into it,
  never originals.
- A unit that GENERATES a dataset copies the payload into the canonical store as its last step, then verifies
  `registry._checksum` against the tracked `manifest.json`.
- Before removing any worktree, check for real (non-symlink) payloads inside it.
- Provisioning must never clobber a real file — link only when absent or already a link.
- Record the generating command and seed in a tracked doc (`docs/DATASETS_L.md` is the precedent) — SHA-256 seeding makes
  byte-identical regeneration possible.

## 7. Compute reality

Device handling exists (source wins over doc): `RNGRN` has `.device`/`.dtype`, `recover()` takes `device=`,
`model.dispersion_backend` selects `eig` (any N) or `cubic` (exact, N ≤ 3); `torch.set_default_dtype(torch.float64)` stays
global for tight autodiff.

- **GPU loses on the serial path** — one Adam step at N=3 is CPU 138ms vs CUDA 418ms: `terms.steady_state` is a damped
  Newton with up to 100 **sequential** 3×3 solves, and `torch.linalg.eigvals` on small non-symmetric matrices has no
  batched cuSOLVER kernel.
- **Batched, the same maths favours the GPU**: cubic dispersion measured CUDA 0.97ms vs eigvals 156.6ms on 127 matrices
  (162×), 38× at 1e6 — the unlock is batching restarts × samples (batched Newton, vmap Jacobian, batched cubic dispersion).
- **CPU runs at short step counts are plumbing checks, not results.** They verify a config resolves and data loads, nothing
  more. Scale multiprocessing workers to `len(os.sched_getaffinity(0))`.

### 7a. HOST RAM IS THE BINDING RESOURCE — launch trainers via `scripts/guarded_run.sh`.

Five sessions died 2026-07-29–08-03 to the Linux OOM killer (syslog: free swap 0kB, ~90MiB free RAM each time). Each
trainer is 1.47–1.68GiB RSS; two agents × (1 parent + `--workers 4`) × ~1.6GiB ≈ 16GiB against a 15.34GiB VM — over the
ceiling before anything else loads. **Run every trainer invocation as `bash scripts/guarded_run.sh <cmd…>`.** It serialises
sweeps across *all* worktrees with one `flock` (overcommit sums over agents, so per-agent limits can't help), waits for an
8192MB `MemAvailable` floor, and raises its own `oom_score_adj` so the kernel kills a trainer, not the session. Reducing
core count and per-agent worker limits alone do **not** fix this — the ceiling was raised 2026-08-03 (`.wslconfig`:
17.58GiB mem / 7.8GiB swap) but that is still below the 18.8GiB peak measured at the worst event, so the guard stays
load-bearing.

### 7b. `ps`, `pgrep`, `pkill` ARE BLIND INSIDE THE SANDBOX.

They return an **empty result**, not an error — indistinguishable from "nothing running." This already cost a live
trainer's stdout and committed run directories, deleted on a false-negative `ps`. Any process check runs with
`dangerouslyDisableSandbox: true`, or it means nothing; corroborate "job died" with what the sandbox can't hide — file
mtimes, `free -h`, `lsof` on `.trainer.lock`. A log's last line being a START is not evidence of death — the next line can
arrive 40 min later.

### 7c. There is NO discretised Laplacian in training — it's analytic, in Fourier space.

**Training never simulates.** `losses/terms.py` constrains the reaction pointwise; the spatial operator enters only
through Fourier eigenvalues, in the dispersion relation `sigma(k) = max Re eig( J - k^2 * diag(D) )`
(`model.py::dispersion`). The `-k^2 D` term **is** the Laplacian — no grid, no stencil, no discretisation error anywhere in
the objective. A real Laplacian appears only post-hoc, in `eval/numerics.py`, and it's **spectral** (`_spectral_k2`,
`fftfreq(n, d=L/n)*2*pi`) for the ETDRK4 rollout behind morphology comparison; the generator (`scripts/gen_tg3.py`)
matches.

1. **The training k-grid is continuous; the box is not** — a periodic domain of size L admits only `k = 2*pi*|m|/L`, but
   training hinges on 400 continuous wavenumbers (`recover._kgrid_for`) the domain may not support, so a model can be
   Turing-unstable in training and fail to pattern in rollout (mode-quantisation, `PREREGISTRATION` §3.5a).
2. **The grid is anchored to observed k\***: `kmin = kstar_obs/50 + 1e-3`, `kmax = max(8*kstar_obs, 2*kmin)` —
   firewall-legal (from an FFT of the observed frame).
3. **L enters only as a unit**: `L_model = 1.0 if nondim else L`; k scales as 1/L, the Laplacian as 1/L², so nondim is an
   exact change of variables — why §3.5b forbids it claiming credit for L-invariance.

So "Turing-unstable" and "patterns" are different claims — closing that gap is why `eval/rollout.py` exists and
`morphology_match` is scored separately.

## 8. Evidence discipline

The single most important rule in this file.

- **Dry-run numbers are never findings.** Say "the harness runs", never "recovery achieves X" — a number must come from a
  tuned run on real data.
- **Thresholds are calibrated, never inherited.** E.g. the `~15%` k\* tolerance in `TUNING.md` dates from the initial
  template and refers to frames the current solver can't generate; `coupling_threshold=0.05` is explicitly uncalibrated.
  Calibrate from a control arm.
- **Read an arm against its control, never against zero.** Every experiment ships one.
- **Verify, don't assert.** Instability is confirmed numerically, morphology is classified by measurement, code claims are
  checked against code. Source wins over doc.
- State what is *not* known as plainly as what is — a caveat that only appears after someone asks is a caveat that failed.

## 9. Where runs go

`experiments/<purpose>/`, passed as `--runs-root` (e.g. `experiments/dryrun/`, `experiments/tuning/`). The tree is
gitignored; name the subdirectory for the purpose so a plumbing check is never mistaken for a result later.

## 10. How far to run before checking in

Decide science decisions yourself, and record them — defining a metric, choosing a threshold, picking an estimator,
deciding what counts as a control. Don't block on the owner for these. What replaces asking is **evidence plus a written
record** (this project is headed for a paper — §8, `docs/PREREGISTRATION.md`). Every science decision must:

1. **Be calibrated against something measured.** If you can't, mark it UNCALIBRATED in both the code and
   `docs/DECISIONS.md` — an honest placeholder, not a silent guess.
2. **Be researched when unsure** — read the primary source, not a secondary one (this already caught misattributed claims
   in `STATE_OF_THE_SCIENCE.md` §2.7). Source wins over doc; paper wins over citation.
3. **Land in `docs/DECISIONS.md`** with date, evidence, and what was rejected.
4. **Be announced loudly if it changes what an existing number means.**

Two things still go to the owner: weakening a pre-registered pass condition in `docs/PREREGISTRATION.md` (binding, dated
ahead of the runs it judges), and anything outside technical scope — publishing, sharing, spending money, touching outside
systems.

## 11. Subagents and dynamic workflows

Spawning subagents inside a dynamic workflow is allowed and encouraged.

- **Tier the models.** Reserve the strongest model for genuinely open-ended work; ports, retrofits, and plumbing run fine
  cheaper — a 13-unit wave at 4 opus + 9 sonnet matched all-opus quality for less.
- **One unit, one worktree, one branch, one PR.** Create worktrees *before* launching the wave — concurrent
  `git worktree add` contends on `.git/config`.
- **Give every unit an anti-conflict rule** when units share files (e.g. new fields appended at the end of a dataclass)
  and expect to repair the merge anyway — git resolves at line granularity and can silently duplicate definitions inside a
  function signature or dict literal, keeping only the last. This once reverted `resid`, dropped the anchor weight, and
  discarded `model_seed` in a merge.
- **`git stash` is repo-global, not per-worktree** — never use it in a parallel wave.
- **Subagents cannot ask the owner** — a unit decides under §10 and reports. A precisely diagnosed defect it couldn't fix
  beats a green test that hides it.
- **A wave can die wholesale to an upstream outage** — check server-side status before diagnosing failures as code.

## 12. `.claude/` — the executable form of this file

This file is the contract; `.claude/` holds the parts that work better as a tool than a paragraph. It is tracked (only
`.claude/settings.local.json` is machine-local) — see `.claude/README.md` for the map. Where they could drift, this file
wins.

- **Agents** (`.claude/agents/`): `firewall-auditor` (§5), `evidence-auditor` (§8, §10), `numerics-reviewer` (§7, §7c),
  `merge-damage-hunter` (§11).
- **Skills** (`.claude/skills/`): `run-training` (§7a, §7b), `new-worktree` (§2, §11), `record-decision` (§10),
  `harvest-dataset` (§6, §6a).
- **Rules** (`.claude/rules/`): `pre-merge-checklist.md` and `reporting-numbers.md`.
- **Hook** (`.claude/hooks/guard_trainer.py`): refuses any Bash command launching `rngrn train`/`sweep`/`target-report` or
  `scripts/exp*.py` outside `scripts/guarded_run.sh`. `RNGRN_GUARD_OFF=1` bypasses it deliberately.

@.claude/rules/pre-merge-checklist.md
@.claude/rules/reporting-numbers.md
