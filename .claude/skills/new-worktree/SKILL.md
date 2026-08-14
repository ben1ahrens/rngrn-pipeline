---
name: new-worktree
description: Use when starting any new unit of work in rngrn-pipeline that could edit or add files, or when launching a parallel wave of agent units. Covers branch naming, worktree creation, the per-worktree venv requirement, payload provisioning, and the anti-conflict rules that keep a parallel wave from destroying itself.
---

# Starting new work in rngrn-pipeline

**New work → new branch → new worktree.** Never by switching branches in an existing
checkout. Never commit directly to `main`.

Branch prefixes are mandatory: `feature/`, `fix/`, `chore/`, `docs/`. Never an unprefixed name.

## 1. Create it

```bash
# from the main checkout
git worktree add ../worktrees/<slug> -b feature/<slug>      # new branch + tree
cd ../worktrees/<slug>
python -m venv --system-site-packages .venv
.venv/bin/pip install -e ".[dev]" -q
```

Worktrees live in `../worktrees/<slug>` — a **sibling** of the checkout, never nested inside
it (a nested tree lands in the parent's working directory and git tries to track it). Remove
a finished one with `git worktree remove ../worktrees/<slug>`.

## 2. Each worktree needs its OWN venv — verify it

An editable install writes an `__editable__*.pth` file holding the **absolute path** of the
tree it was installed from. Reusing another tree's `.venv` therefore imports *that* tree's
`src/` while you edit this one, and the tests silently exercise the wrong code. This has
happened here and was measured: 4 tests for a change made in one worktree failed under the
pre-push hook and passed under `PYTHONPATH`, because the hook was exercising a sibling tree.

A venv per worktree is cheap (~48 MB) because `--system-site-packages` inherits the CUDA
torch build rather than downloading its own. **Always verify:**

```bash
.venv/bin/python -c "import rngrn,os;print(os.path.dirname(rngrn.__file__))"
```

The path must be inside *this* worktree. If it is not, nothing you test means anything.

Enable the test gate **once per repository**, not per worktree — local config is shared,
since a worktree's `.git` is a pointer file: `git config core.hooksPath .githooks`. The
pre-push hook is the authoritative test run.

> **In the sandbox that write FAILS.** `.git/config` (and each
> `.git/worktrees/*/config.worktree`) is bind-mounted read-only, so `git config` returns
> `Device or resource busy`. The rest of `.git/` is writable, which is why commits still
> work — and why a push can silently skip the authoritative test run. Push with:
> ```bash
> git -c core.hooksPath=.githooks push ...
> ```

**GitHub Actions is not a signal here** — `tests.yml` is `workflow_dispatch`-only, the
account's Actions billing has lapsed, and runs are skipped and reported as failures that
have nothing to do with the code. Do not diagnose those as code failures.

## 3. A new worktree LOOKS provisioned with datasets but is not

Manifests and `datasets.jsonl` are tracked, so they arrive with the checkout and
`rngrn list-datasets` cheerfully lists everything. The `payload.h5` files are gitignored and
do **not** arrive, so the first load fails with `FileNotFoundError` deep in h5py.

Use `bash scripts/link_payloads.sh` — it exists precisely because the naive one-liner is
dangerous. If you link by hand, obey these rules:

- `rngrn-pipeline/data/datasets/<id>/payload.h5` (the **main checkout**) is the canonical
  store. Worktrees hold symlinks into it, never originals.
- **Link only when the target is absent or already a symlink**, and never when the canonical
  payload does not exist. The naive loop `ln -sf $MAIN/$d/payload.h5 $d/payload.h5` assumes
  every payload already lives in the canonical store; for a freshly generated dataset it does
  not, so the loop creates a **dangling symlink** — and re-running it **overwrites a real
  generated file with one**. That left three of four downstream agents unable to load data at
  all and forced one dataset to be regenerated from scratch.

Also: checking out a branch that predates the tracked manifests **deletes them from disk**.
Payloads survive. Regenerate with `rngrn scan-datasets` (idempotent, inspects in place, no
copying). A payload with no manifest fails loud — the loader will not guess.

A dataset's **directory name under `data/datasets/` is its `dataset_id`**, and a config's
`dataset_id` must match the directory exactly. An id/directory mismatch once silently
disabled both Experiment B arms for the life of a branch.

## 4. Launching a parallel wave

- **One unit, one worktree, one branch, one PR.**
- **Create every worktree *before* launching the wave.** Concurrent `git worktree add`
  contends on `.git/config`; several agents in the first wave lost their upstream-tracking
  write to a lock collision.
- **`git stash` is repo-global, not per-worktree.** Two agents stashing concurrently will
  apply each other's changes. Do not use it in a parallel wave.
- **Tier the models.** Reserve the strongest model for genuinely open-ended work — a new
  objective, stiff-integrator numerics, a design decision with no prior art. Ports, retrofits,
  relabels, metric modules written against a clear spec, and plumbing run fine on a cheaper
  tier. Measured here: a 13-unit wave at 4 opus + 9 sonnet matched all-opus, far cheaper.
- **Give every unit an anti-conflict rule** when units share files: new config fields appended
  at the end of the dataclass with a comment naming the unit; new metrics as a single appended
  `out.update(...)` line.
- **Then expect to repair the merge anyway.** Git resolves at line granularity and cannot tell
  two additions are independent. Keeping both sides is right for additive blocks and **wrong**
  inside a function signature or a dict literal, where it silently produces duplicate
  definitions of which only the last takes effect. That exact failure reverted `resid` to 0.3,
  dropped the anchor weight, and discarded `model_seed`. Run the `merge-damage-hunter` agent
  after any wave integration.
- **A unit does not decide the science — it returns the decision point.** CLAUDE.md §11 (see
  `.claude/rules/orchestration.md` for the full framing) has the unit report what it found, the
  options, and its recommendation, then stop that thread; the orchestrator rules and records it
  under §10. A precisely diagnosed defect a unit could not fix is a better outcome than a green
  test that hides it — say which, explicitly.
- **A wave can die wholesale to an upstream outage.** Six phase-B units returned API 500s
  simultaneously. Check whether failures are server-side before diagnosing them as code.

## 5. Finishing

Merge to `main` only after the tests pass **and the user has validated the results** —
passing tests alone is not the bar. Committing and pushing on the branch's own prefixed name is
pre-authorised (CLAUDE.md §2) — a unit does not stop to ask. Three gates stay in force
regardless: never commit to `main`; merging still needs the user's validation of the results;
bypassing the pre-push hook with `git push --no-verify` stays a deliberate, explicit choice, not
a default. Substantial branches get a `docs/HANDOFF_<topic>.md` written for a **zero-context
reader**: what changed, what is verified, what is not, where to start.

Before removing any worktree, **check for real (non-symlink) payloads inside it** — see the
`harvest-dataset` skill.
