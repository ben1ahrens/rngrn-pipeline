---
name: run-training
description: Use whenever launching, monitoring, or checking on an rngrn trainer, sweep, or target-report — anything that runs `rngrn train`, `rngrn sweep`, `rngrn target-report`, or a scripts/exp*.py experiment. Covers the mandatory memory guard, how to check whether a run is actually alive (the sandbox lies), and how to read a log without destroying it.
---

# Running trainers in rngrn-pipeline

Host RAM is the binding resource. Five sessions died to the Linux global OOM killer between
2026-07-29 and 2026-08-03, each costing hours of GPU compute. Everything below is the
consequence.

## 1. Every trainer invocation goes through the guard

```bash
bash scripts/guarded_run.sh <command> [args...]
```

Not optional. The guard does three things no per-agent limit can:

1. **Serialises sweeps across *all* worktrees** with one `flock` at
   `/home/benja/projects/personal/rngrn/.trainer.lock`. The overcommit is the *sum over
   concurrently running agents*, and no agent can see the others — which is why per-agent
   `--workers` caps alone do not fix it.
2. **Waits for `MemAvailable` to clear a floor** (default 8192 MB) before launching. The
   floor must exceed the pool's own footprint: a pool is 1 parent + `--workers` children at
   ~1.6 GiB each, so `--workers 4` needs ~8 GiB. A lower floor lets a pool launch into
   headroom it cannot fit in — that is how the original overcommit happened.
3. **Raises its own `oom_score_adj`** (default 500) so the kernel kills the trainer rather
   than the session. Before this, trainers sat at adj 0 while session `systemd`/`dbus-daemon`
   sat at 100–200, so a spike took down the whole user session instead of one cell.

Knobs, if you need them: `RNGRN_MEM_FLOOR_MB`, `RNGRN_MEM_WAIT_S`, `RNGRN_LOCK_WAIT_S`,
`RNGRN_LOCK_PATH`, `RNGRN_OOM_ADJ`. `RNGRN_GUARD_OFF=1` bypasses the guard entirely — if you
ever use it, **say so in writing** in whatever you report.

Guard exit codes are distinguishable and you should read them: `3` cannot open the lock,
`4` lock wait exhausted (another sweep is still running), `5` memory floor never cleared.
None of these mean your config is wrong.

**Two things that do NOT fix the memory problem** — recorded so they are not re-attempted:
the 20 → 14 core reduction (`processors=` does not affect the memory ceiling, and failure #5
happened after it), and per-agent worker limits alone.

The WSL2 ceiling was raised on 2026-08-03 to `memory=18874368000` (17.58 GiB `MemTotal`) and
`swap=8388608000` (7.8 GiB). Still below the 18.8 GiB peak demand measured at the worst
event, so the guard stays load-bearing. Verify with `free -h` after any WSL restart.

## 2. `ps`, `pgrep` and `pkill` are BLIND inside the sandbox

They do not error. They return an **empty result**, which reads exactly like "nothing is
running". On 2026-08-03 that cost 43 minutes: two queues had been training for 52 minutes,
every sandboxed `ps` reported them dead, and acting on that a live trainer's stdout file was
deleted (losing a 53-minute cell) along with committed run directories.

So:

- Any process check — `ps`, `pgrep`, `pkill`, a `/proc` walk — **must** run with
  `dangerouslyDisableSandbox: true`, or its result means nothing.
- Before concluding a job died, corroborate with something the sandbox cannot hide:
  - `lsof` on `.trainer.lock` (this is what showed the truth immediately),
  - file mtimes on the run directory or the log,
  - `free -h` — a dead pool frees GiB.
- **A log whose last line is a START is not evidence of death.** The next line only arrives
  when the target finishes, which can be 40 minutes later.

This is the same failure mode as a zero-byte report: absence of evidence rendered
indistinguishable from evidence of absence.

## 3. Never delete or truncate a file a run might be writing

Read logs; do not remove, rotate, move, or `>` them. Run directories under `experiments/`
are **tracked in git** (run records, frozen configs, results, checkpoints, and `arrays/*.npz`
are all versioned deliberately — the audit trail behind every number). Deleting one destroys
evidence, not scratch.

## 4. Where runs go

`experiments/<purpose>/`, passed as `--runs-root` — `experiments/dryrun/`,
`experiments/tuning/`, `experiments/identifiability/`. Name the subdirectory for its purpose
so a plumbing check is never mistaken for a result later.

## 5. What a short CPU run is worth

**A plumbing check, not a result.** It verifies that a config resolves, data loads, and
scoring routes. It recovers nothing meaningful. Say "the harness runs" — never "recovery
achieves X". Dry-run numbers are never findings.

The GPU *loses* on the serial path (one Adam step at N=3: CPU 138 ms vs CUDA 418 ms) because
`terms.steady_state` is a damped Newton with up to 100 sequential 3×3 solves and
`torch.linalg.eigvals` has no batched cuSOLVER kernel for small non-symmetric matrices.
Batched, CUDA wins only past a crossover. The 162× figure (156.6 ms vs 0.965 ms on 127
matrices, `experiments/exp10_cubic_dispersion.json`) is `eig_cuda` vs `cubic_cuda` — a backend
comparison at a fixed device, not CPU vs CUDA. At n=127 `cubic_cpu` (0.292 ms) actually *beats*
`cubic_cuda` (0.965 ms); the CPU→CUDA crossover for the cubic backend is around 6,400 matrices.
Stay on CPU below that. Multiprocessing over seeds is the CPU throughput lever; scale
`--workers` to `len(os.sched_getaffinity(0))`, not a fixed number — and remember the guard's
floor assumes ~1.6 GiB per worker.

## 6. Environment

Use the worktree's **own** `.venv`. If `import torch` aborts with `OMP: Error #179`, set
`KMP_AFFINITY=disabled OMP_NUM_THREADS=1`. CLI overrides are `-o/--override key=value ...`
(there is no `--set`), and `-v/--verbose` is top-level, so it must precede the subcommand.
