# Pre-merge checklist

Run before merging any branch to `main`. `CLAUDE.md` states the conventions; this is the
executable form. Every line is a check that has caught a real defect in this repo.

## Tests

- [ ] `git config core.hooksPath .githooks` is set — **once per repository**, not per
      worktree. In the sandbox `.git/config` is bind-mounted read-only, so that write fails
      with `Device or resource busy`; push with `git -c core.hooksPath=.githooks push ...`
      instead, or the authoritative test run is **silently skipped**. See `CLAUDE.md` §3.
- [ ] `.venv/bin/python -c "import rngrn,os;print(os.path.dirname(rngrn.__file__))"` prints a
      path **inside this worktree**. If it does not, every test result below is meaningless —
      an editable install's `.pth` is pinning a sibling tree's `src/`.
- [ ] `pytest -q` is green against that venv, with the count recorded in the merge message.
- [ ] The suite was run with the **sandbox DISABLED**. `payload.h5` is on the sandbox
      read-deny list, so a sandboxed run reports ~15 `PermissionError` failures that look
      exactly like code faults and are not. A "failing" suite is not a finding until it has
      been re-run unsandboxed. See `CLAUDE.md` §3.
- [ ] GitHub Actions was **not** consulted as a signal. `tests.yml` is `workflow_dispatch`-only
      and the account's Actions billing has lapsed; runs are skipped and reported as failures
      unrelated to the code.

## Merge damage (the class the suite cannot see)

- [ ] `merge-damage-hunter` agent run after any wave integration or multi-branch merge.
- [ ] No duplicate keys in a dict literal, duplicate keyword arguments, or a name defined
      twice at module/class scope where the second silently wins.
- [ ] No `**`-expansion building a run-index row. Rows are flat scalars built by
      `dict.update()`; run identity wins on collision with scorer keys.
- [ ] Values that a parent branch set are still set — check `resid`, anchor weight, and
      `model_seed` specifically; all three were silently lost in the phase-A merge.

## Firewall

- [ ] `firewall-auditor` agent run if the branch touched `model.py`, `observables.py`,
      `recover.py`, `losses/`, `eval/`, `scoring/`, `train.py` or `validate.py`.
- [ ] Any new module under `eval/` or `losses/` has been classified recovery-side or not, and
      `tests/test_firewall.py`'s module list updated to match. Inside `src/rngrn/` the
      completeness test enforces this for you; it will fail on an unclassified module.
- [ ] **Any new `scripts/` module that opens `payload.h5` was added to `FORBIDDEN`.** The
      completeness test globs `src/rngrn` only, so it is blind to `scripts/` — and scripts are
      importable by bare top-level name because the suite and notebooks put `scripts/` on
      `sys.path`. Forgetting this leaves the suite green, which is exactly what happened when
      the canonical-dataset scripts were written. See `CLAUDE.md` §5.
- [ ] No criterion used to *judge* recovery is derived from ground truth. `kstar_obs` from an
      FFT of the observed frame is legal; anything from the generating parameters is not.

## Evidence

- [ ] `evidence-auditor` agent run if the branch added a threshold, a metric, or a claim.
- [ ] Every new threshold is calibrated against a control, or explicitly marked `UNCALIBRATED`
      in **both** the code and `docs/DECISIONS.md`.
- [ ] Every science decision has a `docs/DECISIONS.md` entry with date, evidence, and what was
      rejected.
- [ ] No pre-registered pass condition in `docs/PREREGISTRATION.md` was weakened. Weakening one
      goes to the owner, never decided locally.
- [ ] Any number that changed meaning is announced loudly in the merge message.
- [ ] No docstring claims more than its code does; scaffold modules still say SCAFFOLD.

## Data

- [ ] `find data/datasets -name payload.h5 -type f` run in the worktree; any real (non-symlink)
      payload harvested to the canonical store in the main checkout and its checksum verified
      against the tracked `manifest.json`.
- [ ] Generating command and seed for any new dataset recorded in a tracked doc.

## Handoff

- [ ] Substantial branch has a `docs/HANDOFF_<topic>.md` written for a **zero-context reader**:
      what changed, what is verified, what is **not**, and where to start.
- [ ] The user has validated the results. Passing tests alone is not the bar.
