# `.claude/` — agent configuration for rngrn-pipeline

`CLAUDE.md` at the repo root is the **contract**: the prose that both Claude Code and Claude
Science are expected to have read. This directory is the **executable form** of it — the parts
that work better as a tool, a checklist, or a mechanical block than as a paragraph someone
might not reach.

Nothing here restates `CLAUDE.md`. Where the two could drift, `CLAUDE.md` wins and the file
here gets fixed.

## Layout

```
.claude/
├── agents/     four repo-specific reviewers plus repo-mapper, by name or proactively
├── skills/     four procedures that have gone wrong here before
├── rules/      two checklists and the dispatch rule, referenced from CLAUDE.md §12
├── hooks/      guard_trainer.py — mechanical enforcement of the memory guard
└── settings.json
```

## Agents

Each encodes an invariant the test suite cannot check, which is why they exist as reviewers
rather than as tests.

| Agent | Guards against | Run it when |
|---|---|---|
| `firewall-auditor` | Recovery seeing ground truth by a path the static AST audit cannot see | Any change to `model.py`, `recover.py`, `losses/`, `eval/`, `scoring/`, `train.py`, `validate.py` |
| `evidence-auditor` | Overclaiming, uncalibrated thresholds, decisions absent from `DECISIONS.md`, doc/source drift | Before any number enters a doc; before merging a branch that adds a threshold or metric |
| `numerics-reviewer` | The dispersion / Newton / k-grid / nondim maths silently changing meaning | Any change to `model.py`, `losses/terms.py`, `recover.py`, `eval/numerics.py` |
| `merge-damage-hunter` | Duplicate dict keys, shadowed definitions, values silently lost in a merge | Immediately after any wave integration or multi-branch merge |

`repo-mapper` is the fifth agent and the only non-reviewer: it answers *where is X and what
calls it*, off the `graphify-out/` graph, verifying every hit against the source. It makes no
judgements — dispatch it before a reviewer so the reviewer spends its context judging.

All five are **read-only**. They report `file:line` and evidence; they never edit. None has the
`Skill` tool, so never instruct one to invoke a skill (`.claude/rules/orchestration.md`).

## Skills

| Skill | Covers |
|---|---|
| `run-training` | The mandatory `guarded_run.sh` wrapper, why `ps`/`pgrep` lie inside the sandbox, and how to check liveness without destroying a live log |
| `new-worktree` | Branch/worktree creation, the per-worktree venv requirement and its verification, payload provisioning, parallel-wave anti-conflict rules |
| `record-decision` | The `docs/DECISIONS.md` entry format, what may be decided locally vs escalated, how to mark something `UNCALIBRATED` |
| `harvest-dataset` | Getting a generated `payload.h5` into the canonical store before a worktree dies, and never clobbering a real file with a symlink |

## The hook

`hooks/guard_trainer.py` blocks any Bash command that launches `rngrn train`, `rngrn sweep`,
`rngrn target-report`, or a `scripts/exp*.py` / stage-0 experiment **without**
`scripts/guarded_run.sh`.

This is the one rule worth enforcing mechanically: five sessions died to the OOM killer
because trainer pools summed past the VM ceiling across worktrees, and the agent most likely
to forget the guard is the one that has not read `CLAUDE.md` §7a. `RNGRN_GUARD_OFF=1` still
bypasses it deliberately, as the guard script itself documents.

It deliberately does **not** block `evaluate`, `analyze`, `export`, `benchmark`,
`list-datasets` or `scan-datasets` — those are cheap. It allows commands that merely *mention*
a trainer (`grep`, `echo`, `--help`), and it never blocks on malformed input.

## Settings

`settings.json` allowlists common read-only commands to cut permission prompts, and routes
three genuinely consequential operations to `ask`:

- `git push --no-verify` — bypasses the authoritative test gate.
- `git stash` — **repo-global, not per-worktree**. Two agents stashing concurrently apply each
  other's changes.
- `git worktree remove` — may destroy a generated `payload.h5` that exists nowhere else. See
  the `harvest-dataset` skill.

Machine-local overrides belong in `settings.local.json`, which is gitignored.
