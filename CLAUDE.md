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

- **The local pre-push hook is the authoritative test run.** It runs `pytest -q` against
  the repo `.venv`, so tests exercise the same torch build you develop against, and aborts
  the push on failure. Bypass deliberately with `git push --no-verify`. Run it on demand
  without pushing: `git hook run pre-push`.
- **Enable it once per REPOSITORY, not per worktree:**

  ```bash
  git config core.hooksPath .githooks      # from any worktree; covers them all
  ```

  Git's local config lives in the one `.git/config` and is **shared by every worktree** (a
  worktree's `.git` is a pointer file, not a config of its own). The value is a *relative*
  path and `.githooks/pre-push` is tracked, so it resolves against whichever worktree you
  push from, and the hook picks up that worktree's own `.venv`. Setting it again in a new
  worktree is a no-op — but it is also harmless.
- **Claude Science cannot set this** — in the sandbox `.git/config` (and each
  `.git/worktrees/*/config.worktree`) is bind-mounted read-only, so any `git config` write
  fails with `Device or resource busy`. The rest of `.git/` is writable, which is why
  commits work. An agent pushing from the sandbox must therefore inject the setting per
  command so the tests still gate the push:

  ```bash
  git -c core.hooksPath=.githooks push ...
  # or: GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath \
  #     GIT_CONFIG_VALUE_0=.githooks git push ...
  ```

  Both are verified to fire the hook. Do not push from the sandbox without one of them —
  otherwise the authoritative test run is silently skipped.
- **GitHub Actions is not a signal here.** `.github/workflows/tests.yml` is
  `workflow_dispatch`-only; the `push`/`pull_request` triggers are commented out, not
  deleted. The account's Actions billing has lapsed, so runs are *skipped* and reported as
  failures that have nothing to do with the code. Do not diagnose those as code failures.
  Restoring CI = uncommenting the two triggers.
- Current suite: **557 passed, 1 skipped**, all CPU, ~2 min 55 s to run (measured 2026-08-11 on
  `fix/reference-solver-cfl`; the +6 are the reference-solver regressions of D-EVID-17). Keep them green. The count in
  this line has been stale before — re-measure it rather than trusting it, and update it when
  it moves.
- **Run the suite with the sandbox DISABLED.** `payload.h5` is on the sandbox read-deny list,
  so a sandboxed run reports ~15 `PermissionError` failures and errors that look exactly like
  code faults and are not. Diagnosing those as code bugs wastes a session.

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

- Recovery-side modules: `model.py`, `observables.py`, `recover.py`, `history.py`,
  `losses/{terms,total,weighting}.py`,
  `eval/{rollout,numerics,dynamical,analysis,topology}.py`.
- Forbidden imports (`tests/test_firewall.py::FORBIDDEN`, 12 names — **re-read the list
  there rather than trusting this one**): `rd_models`, `data.solver`, `data.cache`,
  `data.gate`, `data.registry`, `AnswerKey`, `answer_key`, plus the payload-reading
  scripts `td_figures`, `gen_tg3`, `canon_select`, `canon_generate`, `canon_annotate`.
  The `data.*` names must stay **dotted**: a bare `registry` would false-positive on
  `rngrn/registry.py`, the component registry that `model.py` and `losses/weighting.py`
  legitimately import.
- The answer key's *contents* are read in **exactly one place**: `validate.score_recovery`.
  `train.fit` holds the key and passes it to the scorer — it touches only
  `n_species_true`, to classify the experiment arm — and `_resolve_recovery_input` is the
  firewall boundary where `(RecoveryInput, AnswerKey)` are separated. Neither is
  recovery-side.

**Known gap — restated 2026-08-04. The previous version of this paragraph was wrong in both
directions.** It said `rngrn.scoring` was unguarded and that no recovery-side module imports
it. Neither holds.

- *It is guarded, three times over.* `tests/test_permutation_scoring.py::test_no_recovery_side_module_imports_scoring`,
  `tests/test_morphology_scoring.py::test_no_recovery_side_module_imports_the_morphology_scorer`,
  and the equivalent in `tests/test_overparam_scoring.py` each assert `"scoring" not in
  imports` across the recovery-side list. It is absent from `test_firewall.py`'s `FORBIDDEN`
  only.
- *And the codebase is now clean — corrected 2026-08-11.* This bullet used to say
  `history.py:45` does `from .scoring.plausibility import d_ratio_of` and that "no audit
  covers it". Both halves are false against the source. `d_ratio_of` was moved to the
  side-neutral `rngrn/utils.py:71`, `history.py:55` now reads `from .utils import
  d_ratio_of`, and `history.py` is on `RECOVERY_SIDE` with `utils.py` on `SIDE_NEUTRAL`, so
  the audit *does* cover it — `history.py:41-48` documents the change at the site. No
  recovery-side module imports the scoring package today. It was never a truth leak either
  way (`d_ratio_of` is a pure function of the model's own `D`).

**The allowlist gap inside `src/` is CLOSED.** This paragraph used to describe the fix as a
to-do; it has been implemented and the description was stale. `tests/test_firewall.py` now
declares `RECOVERY_SIDE` (with `history.py` on it), `SCORING_SIDE` (with `eval/lgen_eval.py`)
and `SIDE_NEUTRAL` (`utils.py`), and
`test_every_loss_and_eval_module_is_classified` asserts that every module under `losses/` and
`eval/`, plus `history.py`, is on one of those lists — membership of none being a failure. A
new module under `src/rngrn/` is therefore no longer unaudited by default.

**The gap that remains is `scripts/`, and it bit on 2026-08-10.** The completeness test globs
`src/rngrn/{losses,eval}`, so it cannot see anything in `scripts/` — yet several scripts read
`payload.h5` (generating kinetics, `x_star`, the generator's `k_star`) *and* are importable by
bare top-level name, because the suite and the notebooks put `scripts/` on `sys.path`. Those
are covered only by the hand-maintained `FORBIDDEN` list: `td_figures`, `gen_tg3`,
`canon_select`, `canon_generate`, `canon_annotate`. Adding a sixth such script and forgetting
to list it leaves the suite green — which is exactly what happened when the canonical-dataset
scripts were written. **If you add a script that opens `payload.h5`, add its module name to
`FORBIDDEN`.** The durable fix would be a completeness test over `scripts/` that classifies
each module as payload-reading or not.

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

### 6a. Generated payloads must be HARVESTED before a worktree dies

Learned the hard way, 2026-07-29. A unit generated two datasets (`three_gene_qvar`,
`three_gene_multiL`) inside its own worktree. Manifests are tracked, `payload.h5` is not,
so when that worktree was removed the manifests survived and **the data did not**. Worse,
the standard provisioning line handed to every agent —

    for d in data/datasets/*/; do ln -sf $MAIN/$d/payload.h5 $d/payload.h5; done

— assumes every payload already lives in the canonical store. For a freshly generated
dataset it does not, so the loop creates a **dangling symlink**, and re-running it
**overwrites a real generated file with one**. Three of four downstream agents were left
unable to load the data at all, and one dataset had to be regenerated from scratch.

Rules:
- `rngrn-pipeline/data/datasets/<id>/payload.h5` (the MAIN checkout) is the canonical
  store. Worktrees hold symlinks into it, never originals.
- A unit that GENERATES a dataset copies the payload into the canonical store as its last
  step, then verifies `registry._checksum` against the tracked `manifest.json`.
- Before removing any worktree, check for real (non-symlink) payloads inside it.
- Provisioning must never clobber a real file. Link only when the target is absent or
  already a symlink, and never when the canonical payload does not exist.
- Record the generating command and seed in a tracked doc. This is what made recovery
  possible at all: `docs/DATASETS_L.md` carried the exact invocation, and unit 11 had
  already replaced the process-salted `abs(hash(topo))` seed with SHA-256, so the dataset
  could be regenerated byte-identically instead of merely similarly.

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
  Multiprocessing over seeds is the CPU throughput lever; scale worker counts to the
  machine's actual usable cores (e.g. `len(os.sched_getaffinity(0))`), not a fixed number.

### 7a. HOST RAM IS THE BINDING RESOURCE. Launch trainers through `scripts/guarded_run.sh`.

Five sessions died between 2026-07-29 and 2026-08-03, each killing hours of GPU compute.
Diagnosed from `/var/log/syslog`: the **Linux global OOM killer**, five times, with
**free swap 0 kB and ~90 MiB free RAM at every single event**. Not GPU, not Node, not
disk, not a UI action — `grep -icE "Xid|NVRM"` returns 0, the `claude` process was only
0.26–0.50 GiB RSS, and `/` is 7 % full. Two of the kills took down `setsid nohup` sweeps
that were detached precisely to survive a session exit, which is what proves the event
was system-wide rather than process-tree-local.

The arithmetic: each trainer is **1.47–1.68 GiB RSS**, and 17–21 ran concurrently. Two
agents × (1 parent + `--workers 4`) × ~1.6 GiB ≈ **16 GiB** against a VM `MemTotal` of
**15.34 GiB** — over the ceiling before anything else loads.

**Therefore: run every trainer invocation as `bash scripts/guarded_run.sh <cmd…>`.** It
serialises sweeps across *all* worktrees with one `flock` (per-agent limits cannot help —
the overcommit is the sum over agents, and no agent can see the others), waits for a
`MemAvailable` floor, and raises its own `oom_score_adj` so the kernel kills a trainer
rather than the session. Before that last part, trainers sat at adj 0 while session
`systemd`/`dbus-daemon` sat at 100–200, so a memory spike took down the whole user
session instead of one cell — exactly backwards.

Two things that do **not** fix this, recorded so they are not re-attempted:
- **The 20 → 14 core reduction.** `processors=` does not affect the memory ceiling, and
  the queue scripts pass `--workers` explicitly, so fewer cores does not shrink the pool.
  Failure #5 occurred *after* that change was applied.
- **Per-agent worker limits alone.** See above: the sum across agents is what matters.

**The ceiling was raised on 2026-08-03** — `.wslconfig` now sets `memory=18874368000`
(17.58 GiB `MemTotal`, from 15.34) and `swap=8388608000` (7.8 GiB, from 4). Verify with
`free -h` after any WSL restart.

That is a real improvement but **still below the 18.8 GiB peak demand** measured at the
worst of the five events (14.87 GiB RSS + 3.96 GiB swap). So the guard is **load-bearing,
not optional**: two concurrent trainer pools still do not fit, one does. The guard's
`MemAvailable` floor is 8192 MB for the same reason — a pool is 1 parent + `--workers`
children at ~1.6 GiB each, so `--workers 4` needs ~8 GiB, and a floor below the pool's own
footprint would let it launch into headroom it cannot fit in.

### 7b. `ps`, `pgrep` and `pkill` ARE BLIND INSIDE THE SANDBOX. Always disable it to check.

They do not error — they return an **empty result**, which reads exactly like "nothing is
running". On 2026-08-03 that cost 43 minutes: two queues had been training for 52 minutes,
every sandboxed `ps` said they were dead, and acting on that I deleted the live stdout file
of a running trainer (losing a 53-minute cell) and briefly deleted committed run
directories. `lsof` on `.trainer.lock` with the sandbox disabled showed the truth
immediately.

This is the same failure mode as the zero-byte reports: **absence of evidence rendered
indistinguishable from evidence of absence.** So:

- Any process check — `ps`, `pgrep`, `pkill`, `/proc` walks — runs with
  `dangerouslyDisableSandbox: true`, or its result means nothing.
- Before concluding a job died, corroborate with something the sandbox cannot hide: file
  mtimes, `free -h` (a dead pool frees GiB), or `lsof` on the lock.
- A log whose last line is a START is **not** evidence of death. The next line only arrives
  when the target finishes, which can be 40 minutes later.

### 7c. There is NO discretised Laplacian in training. It is analytic, in Fourier space.

Repeatedly re-derived from scratch, so it is written down once here.

**Training never simulates.** `losses/terms.py` constrains the reaction *pointwise*; the
spatial operator enters only through its Fourier eigenvalues, in the dispersion relation

```
sigma(k) = max Re eig( J - k^2 * diag(D) )        model.py::dispersion
```

The `-k^2 D` term **is** the Laplacian. No grid, no stencil, no FFT of any field, and
therefore no spatial discretisation error anywhere in the objective. Cost lives in the
steady-state Newton solve, never in a spatial operator — which is why the measured ~30x
per-target spread localises there.

**A real Laplacian appears only post-hoc**, in `eval/numerics.py`, and it is **spectral,
never finite-difference**: `_spectral_k2` / `_spectral_k2_half` build `|k|^2` from
`fftfreq(n, d=L/n) * 2*pi` for the ETDRK4 rollout behind the morphology comparison. The
generator (`scripts/gen_tg3.py`) uses the same spectral IMEX scheme.

Three consequences that matter:

1. **The training k-grid is continuous; the box is not.** A periodic domain of size L
   admits only `k = 2*pi*|m|/L`. Training hinges on 400 continuous wavenumbers
   (`recover._kgrid_for`), including ones the domain cannot support — so a model can be
   Turing-unstable *in training* and still fail to pattern in a rollout, because the
   realised mode must snap to the admissible lattice. This is exactly the mode-quantisation
   argument `PREREGISTRATION` §3.5a relies on to make cross-L transfer non-trivial. Milder
   in 2D than it sounds, since `|k| = 2*pi*sqrt(m^2+n^2)/L` is a denser set than the 1D
   picture suggests, but genuinely coarse at small L.
2. **The grid is anchored to the observed k\***: `kmin = kstar_obs/50 + 1e-3`,
   `kmax = max(8*kstar_obs, 2*kmin)`. Firewall-legal (it comes from an FFT of the observed
   frame). Unit B4 fixed a real defect here — the floor had been an absolute 2.0
   rad/length, which is not scale-free and silently dominated whenever `kstar_obs < 0.25`,
   i.e. 11 of 287 samples (3.8 %), pinning the grid to the wrong band.
3. **L enters only as a unit.** `L_model = 1.0 if nondim else L`. Since k scales as 1/L and
   the Laplacian as 1/L^2, the non-dimensional path is an EXACT change of variables — no
   approximation, only the units of D and k change. That is precisely why §3.5b forbids the
   nondim path from claiming credit for L-invariance: it is L-invariant by construction.

**So "Turing-unstable" and "patterns" are different claims.** Closing that gap is the entire
reason `eval/rollout.py` exists, and why `morphology_match` is scored separately from every
dispersion-derived criterion.

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
`experiments/tuning/`, `experiments/lgen_transfer/`, `experiments/stage0_bioviab/`. Name the
subdirectory for the purpose, so a plumbing check is never mistaken for a result later.

**Corrected 2026-08-11. This section used to say the whole `experiments/` tree is gitignored
and that runs never enter version control. That is the opposite of the truth today** — 792
files under `experiments/` are tracked. The reversal is deliberate and documented at length in
`.gitignore` (D-PLOT-1): the project is headed for a paper, so the run behind every number is
versioned alongside the claim. Tracked are the append-only indexes (`runs.jsonl`,
`target_reports.jsonl`, `lgen_eval.jsonl`), `config/frozen_config.yaml`, `results/*.json`,
`checkpoints/*.pt` and `arrays/*.npz`. Ignored is only regenerable bulk: `figures/`, `*.h5`,
bare `*.npy`, `*.png` (except `experiments/figures_report/`), and any `.npz` outside
`arrays/`. **Read `.gitignore` rather than this paragraph** — it carries the reasoning and the
measured costs.

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

## 12. `.claude/` — the executable form of this file

This file is the contract. `.claude/` holds the parts that work better as a tool than as a
paragraph. It is **tracked**; only `.claude/settings.local.json` is machine-local. See
`.claude/README.md` for the full map. Nothing there restates this file — where they could
drift, **this file wins and the other gets fixed**.

- **Agents** (`.claude/agents/`) — five read-only agents. Four review invariants the suite
  cannot check: `firewall-auditor` (§5), `evidence-auditor` (§8, §10), `numerics-reviewer`
  (§7, §7c), `merge-damage-hunter` (§11). The fifth, `repo-mapper`, is not a reviewer — it
  locates files and traces call paths off the `graphify-out/` graph, verifying every hit
  against the source, so the reviewers spend their context judging rather than searching. It
  makes no judgements of its own.
- **Skills** (`.claude/skills/`) — `run-training` (§7a, §7b), `new-worktree` (§2, §11),
  `record-decision` (§10), `harvest-dataset` (§6, §6a).
- **Rules** (`.claude/rules/`) — `pre-merge-checklist.md` and `reporting-numbers.md`, the
  checklist forms of §3/§5/§8/§11 and §8 respectively.
- **Hook** (`.claude/hooks/guard_trainer.py`) — refuses any Bash command that launches
  `rngrn train`/`sweep`/`target-report` or a `scripts/exp*.py` experiment outside
  `scripts/guarded_run.sh`. §7a is the one rule worth enforcing mechanically, because the
  agent that forgets the guard is the one that has not read §7a. `RNGRN_GUARD_OFF=1` still
  bypasses it deliberately.

@.claude/rules/pre-merge-checklist.md
@.claude/rules/reporting-numbers.md
