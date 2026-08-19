# PROVENANCE — `phase1_r2_B512` (B=512, arm `r2`, telemetry ON)

Written 2026-08-19 to reconcile review finding I2: the tracked `../launch.log` and the tracked
`results/run.json` in this directory report different throughput/RSS numbers for what purports
to be the same cell. **Reconciled, not re-run** — per the review instruction and CLAUDE.md §8,
a provenance gap is written down, not papered over by re-measuring on the GPU.

## The two measurements

| source | member-steps/s | peak host RSS (MB) | wall (s) | outcome |
|---|---|---|---|---|
| `../launch.log` lines 58-74 (guarded, logged) | 4885 | 1923 | ~157 (512×1500/4885, derived — the log prints only the rounded rate) | `guarded_run: END rc=0` |
| `results/run.json` (currently committed) | 2956.7477966347287 | 1936.37890625 | 259.7448456287384 | — |

Spread: 4885 / 2957 ≈ **1.65×** — this is the concrete instance of the "≥1.65× spread"
referenced by I3's cross-process caveat in `docs/HANDOFF_redesign_r2.md` §4b.

## Which invocation produced the committed `run.json`

The file on disk (2957 / 1936) does not match `launch.log`'s numbers (4885 / 1923). Evidence
for a later, third invocation that superseded the logged one:

- **File mtimes.** `../launch.log`'s last write (the `guarded_run: END rc=0` line) is
  2026-08-19 00:07:54. `results/run.json` and `config/frozen_config.yaml` in this directory are
  both dated 2026-08-19 00:18:45 — eleven minutes later. This is the original worktree the run
  executed in, not a fresh checkout, so these mtimes are believed to be real write times rather
  than checkout artifacts.
- **`verbose_every` mismatch.** `../results/phase1_summary.json`'s `args` block (as committed,
  before this fix session touched only the two `a0` entries elsewhere in `phase1/`) records
  `"verbose_every": 750`. `launch.log`'s B=512 cell prints at a stride of 250 steps (lines
  65-70), i.e. the CLI default (`--verbose-every` defaults to 250). A recorded stride of 750
  implies a CLI invocation that is not the one captured in the log.
  `_write_index_and_summary` (in `scripts/r2_ignition_run.py`) rewrites `phase1_summary.json`'s
  `args` block after every cell, from that invocation's own `argparse.Namespace` — so the `750`
  on disk is the args of whichever invocation against `--out experiments/redesign_r2/phase1`
  ran **last**, consistent with the 00:18:45 write above.
- **`--skip-existing` defaults True.** For a second invocation to overwrite an existing
  `results/run.json` for this cell (which the logged run had already written at 00:07:54), it
  must have passed `--no-skip-existing` explicitly.

Together: the run captured in `launch.log` (4885 / 1923, telemetry ON via the default
`history_every=100` since B=512 was that invocation's only/max B) completed cleanly and was
then superseded by a second, later invocation of the identical cell that produced the numbers
now committed (2957 / 1936).

## Guard evidence for the committed invocation

**None found.** Checked:

- `../launch.log` — no second `phase1_r2_B512` entry after the one reconciled above; the file
  ends at the `guarded_run: END rc=0` line for the logged (4885) run.
- Repo-wide search for other `*.log` / `*.out` / `nohup*` files with mtimes after
  2026-08-19 00:07 — none under `experiments/redesign_r2/` besides this directory's own
  siblings (see below).
- `~/.bash_history` and `~/.zsh_history` — both last modified June/July 2026, predating this run
  entirely. No evidence either way.
- The shared `.trainer.lock` (used by `scripts/guarded_run.sh` across every worktree) has mtime
  2026-08-19 01:01:41 — an hour after the write in question, and shared across every worktree's
  sweeps, so it neither confirms nor rules out this specific invocation.

**Plainly: there is no evidence that the invocation which produced the currently-committed
`results/run.json` for this cell ran under `scripts/guarded_run.sh`'s memory guard.** It may
have (the guard's own START/END lines are written by the *caller's* redirection into
`launch.log`, not guaranteed — `guarded_run.sh` itself writes only to its own stderr), or it may
have bypassed the guard entirely (`RNGRN_GUARD_OFF=1`, or a direct unwrapped invocation). The
evidence does not distinguish between these, and none is invented here.

## What this does not affect

- `docs/HANDOFF_redesign_r2.md` §4b's headline B=512 throughput number (**6313**
  member-steps/s) comes from a **third**, separate run directory,
  `experiments/redesign_r2/phase1_throughput/phase1_r2_B512/` (`history_every: 0`, genuinely
  telemetry-off) — not from either number in this file. That directory's invocation is also
  absent from `launch.log` and from every other log file found in the repo: the same evidence
  gap, on a different cell.
- §4b's "telemetry costs ~53 % at B=512 (6313 → 2957)" comparison *does* use the 2957 number
  from this directory as its telemetry-on reference point. See I3's caveat in §4b: that
  comparison is cross-process, and this file is the concrete instance of why.

## Do not re-run

Per the review instruction, this file reconciles the tracked record; it does not re-verify
either number on the GPU.
