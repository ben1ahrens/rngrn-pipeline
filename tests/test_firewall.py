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
    "history.py",                                  # runs INSIDE the Adam loop (recover.py:245)
    "losses/terms.py", "losses/total.py", "losses/weighting.py",
    "eval/rollout.py", "eval/numerics.py", "eval/dynamical.py",
    "eval/analysis.py", "eval/topology.py", "eval/lifted.py",
]

# SCORING-side modules under losses/ and eval/. These MAY read the answer key; they are
# listed so `test_every_loss_and_eval_module_is_classified` can tell "deliberately
# scoring-side" apart from "nobody has classified this yet".
SCORING_SIDE = [
    "eval/lgen_eval.py",       # cross-L transfer scoring; imports scoring.morphology
]

# SIDE-NEUTRAL: imported by BOTH sides, so it must satisfy the recovery-side import rules
# even though it is not itself part of recovery. `utils.py` holds `d_ratio_of`, moved there
# so `history.py` (recovery-side) could stop importing the scoring package — which would
# have been pointless if utils were then free to import the answer key itself. The audit is
# per-file and non-transitive, so a neutral module in the middle of the chain is exactly
# where an unaudited hop would hide.
SIDE_NEUTRAL = ["utils.py"]

# answer-key-side names that must never appear in a recovery-side import.
# `data.gate` and `data.registry` added 2026-08-04: gate.from_registry returns the full
# (RecoveryInput, AnswerKey) pair, so a recovery-side module importing it could reach ground
# truth in two lines while passing every previous version of this audit. NOTE these names
# must stay DOTTED — a bare "registry" would false-positive on `rngrn/registry.py`, the
# component registry, which model.py and losses/weighting.py legitimately import.
# `td_figures` and `gen_tg3` added 2026-08-10 (unit training-data-plots). Both live in
# `scripts/`, so the completeness test below — which globs `src/rngrn` — is structurally
# blind to them, and both are importable by BARE TOP-LEVEL NAME because the inspection
# notebook and tests/test_td_figures.py put `scripts/` on sys.path for the whole pytest
# session. `td_figures.load_samples` reaches params_json, x_star, D and the generator's
# k_star in two lines. Nothing imports them from the recovery side today; these entries
# make sure that stays true instead of relying on it.
# `canon_select`, `canon_generate` and `canon_annotate` added 2026-08-10 alongside
# `td_figures`/`gen_tg3`, and for the same reason: all five live in `scripts/`, are therefore
# invisible to the completeness glob below, are importable by BARE TOP-LEVEL NAME because the
# test suite puts `scripts/` on sys.path, and all five read `payload.h5` — generating
# kinetics, `x_star`, and the generator's own `k_star`. A recovery-side import of any of them
# would reach ground truth in two lines.
#
# `phase_topology` is deliberately NOT here: it takes a bare 2-D array and reads no payload,
# so a recovery-side module could legitimately measure the OBSERVED frame with it.
FORBIDDEN = ["rd_models", "data.solver", "data.cache", "data.gate", "data.registry",
             "AnswerKey", "answer_key", "td_figures", "gen_tg3",
             "canon_select", "canon_generate", "canon_annotate"]


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


@pytest.mark.parametrize("relpath", SIDE_NEUTRAL)
def test_side_neutral_modules_are_import_clean(relpath):
    """A module both sides import must be as clean as the recovery side, or it becomes an
    unaudited hop: recovery -> utils -> anything. The per-file audit cannot see through it.
    """
    imports = " ".join(_imports(SRC / relpath))
    for forbidden in FORBIDDEN + ["scoring", "plotdata", "export"]:
        assert forbidden not in imports, (
            f"FIREWALL: side-neutral {relpath} imports '{forbidden}'. It is imported by "
            f"recovery-side code, so it must not reach the answer key or the scoring side.")


@pytest.mark.parametrize("relpath", RECOVERY_SIDE)
def test_recovery_side_has_no_answer_key_import(relpath):
    path = SRC / relpath
    assert path.exists(), f"missing recovery-side module {relpath}"
    imports = " ".join(_imports(path))
    for forbidden in FORBIDDEN:
        assert forbidden not in imports, (
            f"FIREWALL BREACH: {relpath} imports '{forbidden}' "
            f"(answer-key side). Recovery may see only (frame, L, observed_idx).")


def test_every_loss_and_eval_module_is_classified():
    """NO MODULE MAY BE UNCLASSIFIED. This is the audit's completeness guarantee.

    Every other firewall test is a NAME ALLOWLIST: it checks the modules someone remembered
    to list. That is exactly backwards for a guard against *new* code — `eval/lgen_eval.py`
    (564 lines) and `history.py` both landed on this branch, were recovery-relevant, and
    appeared in none of the four hand-copied RECOVERY_SIDE lists, so neither was audited and
    the suite stayed green. Adding one name would not have fixed that; making membership
    MANDATORY does.

    So: every module under losses/ and eval/, plus history.py, must be declared either
    RECOVERY_SIDE (and therefore import-audited above) or explicitly SCORING_SIDE. Being on
    neither list fails, which forces the classification decision at the moment a module is
    added rather than at the moment someone notices.
    """
    discovered = {"history.py"}
    for pkg in ("losses", "eval"):
        for path in sorted((SRC / pkg).glob("*.py")):
            if path.name == "__init__.py":
                continue
            discovered.add(f"{pkg}/{path.name}")

    classified = set(RECOVERY_SIDE) | set(SCORING_SIDE)
    unclassified = sorted(discovered - classified)
    assert not unclassified, (
        "UNCLASSIFIED MODULE(S): " + ", ".join(unclassified) + ".\n"
        "Every module under losses/ and eval/ (plus history.py) must be declared in "
        "tests/test_firewall.py as either RECOVERY_SIDE (runs during recover()/fit() and is "
        "import-audited against the answer key) or SCORING_SIDE (may read the answer key). "
        "Decide which, and add it to the right list — an unlisted module is an UNAUDITED "
        "module, which is how eval/lgen_eval.py went 564 lines without a firewall check.")

    # And the lists must be disjoint: a module cannot be both.
    both = sorted(set(RECOVERY_SIDE) & set(SCORING_SIDE))
    assert not both, f"module(s) on BOTH sides: {both}"

    # Every listed name must exist, so a rename cannot silently empty the audit.
    for relpath in RECOVERY_SIDE + SCORING_SIDE:
        assert (SRC / relpath).exists(), f"listed but missing: {relpath}"


def test_recovery_side_does_not_import_the_scoring_package():
    """The rule CLAUDE.md §5 states, enforced here rather than only in the scorers' tests.

    `rngrn.scoring` reads the answer key (scoring/overparam.py, scoring/permutation.py), so
    a recovery-side import of it is a route for truth to reach recovery. This was previously
    asserted only in test_{permutation,morphology,overparam}_scoring.py, each over its own
    hand-copied list — none of which contained history.py, which did import it.
    """
    for relpath in RECOVERY_SIDE:
        imports = " ".join(_imports(SRC / relpath))
        assert "scoring" not in imports, (
            f"FIREWALL: {relpath} is recovery-side and imports the scoring package. "
            "Move the shared helper to a neutral module (rngrn/utils.py) instead.")


def test_recovery_input_is_minimal():
    """RecoveryInput must expose only frame/L/observed_idx/N — no truth fields."""
    from rngrn.data.gate import RecoveryInput
    import dataclasses
    fields = {f.name for f in dataclasses.fields(RecoveryInput)}
    assert fields == {"frame", "L", "observed_idx", "N"}, fields
    for bad in ("x_star", "J", "kstar", "sigma", "coefficients"):
        assert bad not in fields
