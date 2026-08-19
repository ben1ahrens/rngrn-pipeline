"""One-shot extractor: pull the DECISIONS.md entries cited by PAPER_RESULTS_PACK.md
out of their source branches, verbatim, into docs/DECISIONS_excerpts.md.

Run from the repo root of the paper-results worktree:
    python paper_results/tools_extract_decisions.py
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

SOURCES = [
    ("feature/lift-ladder", "2f50fff",
     ["D-CANON-4", "D-LIFT-1", "D-LIFT-2", "D-EVID-18", "D-PAPER-1"]),
    ("feature/paper-claim3-hidden-channel", "60fddfb", ["D-PAPER-2"]),
    ("feature/paper-claim5-noise", "4515ea1", ["D-CLAIM5-1"]),
    ("feature/paper-form-robustness", "bfb9474", ["D-FORMCOMP-1"]),
    ("feature/paper-weight-noise", "58445df", ["D-WNOISE-1"]),
    ("feature/real-stripes", "01b9bbf", ["D-REAL-1"]),
]


def extract(branch: str, entry_id: str) -> str:
    text = subprocess.run(
        ["git", "show", f"{branch}:docs/DECISIONS.md"],
        capture_output=True, text=True, check=True,
    ).stdout
    lines = text.splitlines()
    out, in_entry = [], False
    for line in lines:
        if line.startswith(f"### {entry_id} "):
            in_entry = True
        elif in_entry and (line.startswith("### ") or line.startswith("## ")):
            break
        if in_entry:
            out.append(line)
    if not out:
        raise SystemExit(f"entry {entry_id} not found on {branch}")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    parts = [
        "# DECISIONS.md excerpts — entries behind the paper-sprint claims\n",
        "Verbatim copies of the decision entries the pack (and the two supplementary "
        "claim docs, PAPER_CLAIM_FORMCOMP.md and PAPER_CLAIM_WNOISE.md) reference, "
        "extracted 2026-08-19.\nEach branch's full `docs/DECISIONS.md` remains the "
        "authoritative record. Sources:\n",
    ]
    for branch, commit, ids in SOURCES:
        parts.append(f"- `{branch}` @ {commit} — {', '.join(ids)}")
    parts.append("\n---\n")
    for branch, _commit, ids in SOURCES:
        for entry_id in ids:
            parts.append(extract(branch, entry_id))
            parts.append("\n---\n")
    body = "\n".join(parts)
    body = body.rstrip("\n-— \n") + "\n"
    (HERE / "docs" / "DECISIONS_excerpts.md").write_text(body)
    print("wrote", HERE / "docs" / "DECISIONS_excerpts.md")


if __name__ == "__main__":
    main()
