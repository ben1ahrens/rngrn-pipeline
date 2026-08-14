# Orchestrating a unit of work

`CLAUDE.md` §11 is the rule — it carries the model-routing table, the delegation floor and the
wave floor. This is the executable form: which process chain governs, which skill fires when,
and what a dispatched agent must be told before it can be trusted with this repo.

**Standing opt-in.** This is a user rules file, so it is an instruction already given: use the
Agent tool and dynamic workflows here without asking again, as global `~/.claude/CLAUDE.md` §5
also grants. Agent teams and workflows are the **preferred** shape for implementation above the
§11 delegation floor — and are the wrong shape below it.

## Which doctrine governs

- **`superpowers` is the process chain** (plugin `superpowers@claude-plugins-official`; the
  roster below is pinned to 6.3.0). Follow it.
- **`~/.claude/rules/common/` is not authoritative here.** It mandates a `planner` →
  `tdd-guide` → `code-reviewer` chain, plus `security-reviewer` and `build-error-resolver`;
  **none of those five agents exist on this machine** (`~/.claude/agents/` is empty). Read the
  intent across instead: *planner* → `superpowers:writing-plans`, *tdd-guide* →
  `superpowers:test-driven-development`, *code-reviewer* → `superpowers:requesting-code-review`
  plus the real auditors in `.claude/agents/`. Its "GitHub code search first" step is also
  wrong for this repo — §10 says read the primary source, which here is a paper or
  `docs/SPEC_fourier_training.md`, not `gh search code`.
- **superpowers governs *how* work proceeds, never *when it may merge*.**
  `.claude/rules/pre-merge-checklist.md` is the merge gate and binds whatever produced the
  branch. In particular `superpowers:finishing-a-development-branch` offers "merge to base
  locally" straight off a green suite — **it may not**, because merging needs the owner's
  validation (§2, §3). Its "run the suite" step means an **unsandboxed** run.
- **`.claude/skills/new-worktree` supersedes `superpowers:using-git-worktrees`**, and the
  native `EnterWorktree` tool is not used here. Worktrees are **siblings** at
  `../worktrees/<slug>`, never nested, and need a per-worktree `.venv` plus payload symlinks
  that neither the skill nor the native tool provides. Skipping that silently tests another
  tree's `src/`.

## Decision points come back to the orchestrator — paste this into every dispatch

`superpowers:subagent-driven-development` says *"Rulings, not stalls. A running plan does not
wait on a human,"* and lists four things that stop a unit. §11 overrides that for decision
points: the unit returns them, the orchestrator gives the verdict. The reason is the register —
a ruling made inside a unit's context, recorded only in that unit's ledger, is a ruling
`docs/DECISIONS.md` never sees, and this repo is headed for a paper.

This block goes in the brief verbatim:

> **Settle these yourself:** naming, local structure, which helper to reuse, how to arrange a
> test — ordinary implementation choices inside your task.
>
> **Return these to the orchestrator — do not rule, do not proceed.** Report what you found,
> the options, and your recommendation, then stop that thread and wait for the verdict:
> - anything that would earn a `docs/DECISIONS.md` entry — defining or changing a metric,
>   picking a threshold or pass condition, choosing an estimator, deciding what counts as a
>   control, or marking something UNCALIBRATED;
> - the plan conflicting with what you actually found, a plan defect, or a cap you would have
>   exceeded;
> - **hard stop**, regardless of how small it looks: anything that would weaken a threshold,
>   tolerance, test or gate, or that touches `tests/test_firewall.py`,
>   `docs/PREREGISTRATION.md`, `docs/SPEC_fourier_training.md` or
>   `docs/PLAN_fourier_training.md`.
>
> Returning is not stalling. You stop that thread; the orchestrator answers and you resume.

The orchestrator then rules, records it under §10 (`.claude/skills/record-decision`), and hands
the ruling down. It escalates to the owner only for §10's reserved class: weakening a
pre-registered pass condition, or anything outside the technical scope of the work.

## Dispatched agents do not auto-invoke skills — you name them

`superpowers:using-superpowers` carries `<SUBAGENT-STOP>If you were dispatched as a subagent to
execute a specific task, ignore this skill.</SUBAGENT-STOP>`. A subagent therefore picks up
**none** of the skills below on its own, and the five agents in `.claude/agents/` could not
anyway — their `tools:` line has no `Skill`. **The dispatcher owns skill selection.** Every
brief states: the model, the skills to invoke by name, the success criterion, and the files the
unit may touch.

Two clauses belong verbatim in any brief that touches tests or process checks, because the
failure they prevent reads exactly like a code fault (§3, §7b):

> Run `pytest` and any `ps`/`pgrep`/`lsof` with the sandbox **disabled**. A sandboxed test run
> reports ~15 fake `PermissionError`s and a sandboxed process check returns empty rather than
> erroring. Neither is a finding.

## The work cycle

```
using-superpowers → brainstorming → writing-plans ─┬─ subagent-driven-development → finishing-a-development-branch
                            │                      └─ executing-plans ──────────────↗   (may NOT merge)
                            └─ .claude/skills/new-worktree FIRST   (supersedes using-git-worktrees)

  per task:  test-driven-development → requesting-code-review → receiving-code-review
  always:    verification-before-completion   before ANY "done", "fixed" or "passing"
  on a bug:  systematic-debugging → test-driven-development → verification-before-completion
  fan-out:   dispatching-parallel-agents (2+ independent units, and see §11's wave floor)
```

All 14 skills of **superpowers 6.3.0**, prefix `superpowers:`. This roster is pinned to that
version and is not owned by this repo — 6.1.1, 6.2.0 and 6.3.0 all shipped recently, so on a
plugin upgrade re-check it against
`~/.claude/plugins/cache/claude-plugins-official/superpowers/<version>/skills/` rather than
trusting the table. A stale roster reads as authoritative, which is the failure mode here.

| Skill | What it does | When it fires |
| --- | --- | --- |
| `using-superpowers` | How to find and invoke skills | Start of a session, before the first reply |
| `brainstorming` | Draws out intent and design before code exists | Before any new feature, metric or behaviour change |
| `writing-plans` | Turns a settled design into a task-by-task plan | Design agreed, before touching code |
| `using-git-worktrees` | Isolated workspace | **Superseded here** by `.claude/skills/new-worktree` |
| `subagent-driven-development` | Runs the plan task-by-task through dispatched agents | Default execution mode above the delegation floor |
| `executing-plans` | Runs a plan with review checkpoints in a separate session | When the plan is handed off rather than run now |
| `test-driven-development` | Failing test first, then the implementation | Inside every implementation task |
| `requesting-code-review` | Dispatches a reviewer against the stated requirements | After each task, and before the branch closes |
| `receiving-code-review` | Verifies feedback technically before acting on it | On any review response — no performative agreement |
| `verification-before-completion` | Forces command output to precede the claim | Before any completion claim, commit or PR |
| `systematic-debugging` | Root cause before fix | Any bug, test failure or surprise |
| `dispatching-parallel-agents` | Fan-out across independent work | 2+ units with no shared state or ordering |
| `finishing-a-development-branch` | Decides how the work integrates | Implementation complete — **but see the merge gate above** |
| `writing-skills` | Authoring and verifying skills | Only when changing `.claude/skills/` |

## What every delegated unit must be told

Assume it has **not** read `CLAUDE.md`. Paste the clauses its task can reach.

- **Trainers, sweeps, target-reports and `scripts/exp*.py` run only as
  `bash scripts/guarded_run.sh <cmd…>`** (§7a). `.claude/hooks/guard_trainer.py` refuses
  anything else, so a unit that does not know this reports a broken repo. Skill:
  `.claude/skills/run-training`.
- **A unit does not decide the science — it returns the decision point** (see above). The
  orchestrator rules and records it in `docs/DECISIONS.md` with the date, the evidence and what
  was rejected (`.claude/skills/record-decision`). A precisely diagnosed defect the unit could
  not fix beats a green test hiding one — say which.
- **One unit, one worktree, one branch, one PR** (§11). Worktrees are created *before* the wave
  launches. Never `git stash` in a parallel wave — it is repo-global. Never remove a worktree
  without checking for a real (non-symlink) `payload.h5` inside it (§6a).
- **Committing and pushing on its own prefixed branch is pre-authorised** (§2) — a unit does
  not stop to ask. It may **not** commit to `main`, and it may **not** merge: that needs the
  owner's validation of the results, which is also why
  `superpowers:finishing-a-development-branch` stops short here.
- **Numbers leaving a run directory pass `.claude/rules/reporting-numbers.md` first**, and the
  branch passes `.claude/rules/pre-merge-checklist.md` before it merges.
- **A wave can die wholesale to an upstream outage** (§11) — check whether failures are
  server-side before diagnosing them as code.
