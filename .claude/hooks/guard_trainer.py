#!/usr/bin/env python3
"""PreToolUse hook: refuse to launch an rngrn trainer outside scripts/guarded_run.sh.

WHY. Host RAM is the binding resource in this environment. Five sessions died to the Linux
global OOM killer between 2026-07-29 and 2026-08-03, each costing hours of compute, because
concurrent trainer pools across worktrees summed past the VM ceiling. scripts/guarded_run.sh
serialises sweeps with one flock, waits for a MemAvailable floor, and raises oom_score_adj so
the kernel kills a trainer rather than the session. CLAUDE.md 7a says to use it on every
invocation -- this makes that mechanical instead of remembered, because the agent that forgets
is exactly the agent that has not read CLAUDE.md 7a.

Deliberate bypass is still available and still honest: set RNGRN_GUARD_OFF=1 in the command,
which the guard itself documents, and say so in writing in whatever you report.
"""
import json
import re
import sys

# Subcommands that spawn a trainer pool. `evaluate`, `analyze`, `export`, `benchmark`,
# `list-datasets` and `scan-datasets` are cheap and deliberately absent.
TRAINER = re.compile(
    r"""(?x)
    (?:^|[;&|]|\s)                 # start, or after a shell separator
    (?:\S*rngrn|\S*python\S*\s+-m\s+rngrn\S*)\s+
    (?:[-\w=./]+\s+)*              # any leading options
    (?:train|sweep|target-report)\b
    """
)
# The experiment scripts drive the same trainers.
EXP_SCRIPT = re.compile(r"scripts/(?:exp\d+\w*|stage0_bio_viability|stage0_part4_prior_recovery)\.\w+")

# Talking *about* a command is not running one.
DISCUSSING = re.compile(r"(?x) \b(?:grep|rg|ag|ack|sed|awk|cat|head|tail|less|echo|printf)\b | --help | \s-h(?:\s|$)")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # never block on a hook parsing failure

    if payload.get("tool_name") != "Bash":
        return 0

    command = (payload.get("tool_input") or {}).get("command") or ""
    if not command:
        return 0

    if "guarded_run.sh" in command or "RNGRN_GUARD_OFF=1" in command:
        return 0
    if DISCUSSING.search(command):
        return 0
    if not (TRAINER.search(command) or EXP_SCRIPT.search(command)):
        return 0

    print(
        "BLOCKED: this launches an rngrn trainer without the memory guard.\n"
        "\n"
        "Host RAM is the binding resource here. Five sessions have already been killed by the\n"
        "Linux global OOM killer because concurrent trainer pools summed past the VM ceiling,\n"
        "and per-agent --workers caps cannot fix it: the overcommit is the sum over agents.\n"
        "\n"
        "Re-run it as:\n"
        "\n"
        "    bash scripts/guarded_run.sh <your command>\n"
        "\n"
        "The guard serialises sweeps across ALL worktrees with one flock, waits for a\n"
        "MemAvailable floor of 8192 MB (a pool is 1 parent + --workers children at ~1.6 GiB\n"
        "each), and raises oom_score_adj so the kernel kills the trainer rather than the\n"
        "session. See CLAUDE.md 7a and the run-training skill.\n"
        "\n"
        "To bypass deliberately, prefix RNGRN_GUARD_OFF=1 -- and say so in writing in whatever\n"
        "you report.",
        file=sys.stderr,
    )
    return 2  # blocking error: the message goes back to the model


if __name__ == "__main__":
    sys.exit(main())
