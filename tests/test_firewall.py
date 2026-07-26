"""test_firewall.py — THE firewall enforcement (not a comment, a test).

No recovery-side module may import the answer-key side (rd_models, solver, cache
answer-key access, or the AnswerKey type). The gate is the ONLY boundary; it is
allowed to touch both sides because it IS the boundary.

We audit by static import analysis (ast) of the recovery-side source files, so the
test fails LOUD if someone wires a truth quantity into recovery.
"""
import ast
import os
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "rngrn"

# recovery-side modules: everything that runs during recover()/fit() BEFORE scoring
RECOVERY_SIDE = [
    "model.py", "observables.py", "recover.py",
    "losses/terms.py", "losses/total.py", "losses/weighting.py",
    "eval/rollout.py", "eval/numerics.py", "eval/dynamical.py",
    "eval/analysis.py", "eval/topology.py",
]

# answer-key-side names that must never appear in a recovery-side import
FORBIDDEN = ["rd_models", "data.solver", "data.cache", "AnswerKey", "answer_key"]


def _imports(path):
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names += [f"{mod}.{a.name}" for a in node.names] + [mod]
    return names


@pytest.mark.parametrize("relpath", RECOVERY_SIDE)
def test_recovery_side_has_no_answer_key_import(relpath):
    path = SRC / relpath
    assert path.exists(), f"missing recovery-side module {relpath}"
    imports = " ".join(_imports(path))
    for forbidden in FORBIDDEN:
        assert forbidden not in imports, (
            f"FIREWALL BREACH: {relpath} imports '{forbidden}' "
            f"(answer-key side). Recovery may see only (frame, L, observed_idx).")


def test_recovery_input_is_minimal():
    """RecoveryInput must expose only frame/L/observed_idx/N — no truth fields."""
    from rngrn.data.gate import RecoveryInput
    import dataclasses
    fields = {f.name for f in dataclasses.fields(RecoveryInput)}
    assert fields == {"frame", "L", "observed_idx", "N"}, fields
    for bad in ("x_star", "J", "kstar", "sigma", "coefficients"):
        assert bad not in fields
