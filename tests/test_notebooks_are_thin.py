"""test_notebooks_are_thin.py — the "thin drivers" contract of spec §4.8, made checkable.

Parses notebook JSON only — this suite NEVER executes a notebook. `redesign_pipeline.ipynb`
shells out to `scripts/r2_ignition_run.py` (which Task 16 built, so the reason is no longer
"the script is missing"): executing it here would launch a ~2-hour guarded GPU run from
inside a unit test. `redesign_plots.ipynb` needs a real run's `arrays/plot_arrays.npz`, which
now exists under `experiments/redesign_r2/phase1/` but is a run artefact a test must not
depend on. "Thin" is therefore checked structurally instead: per code cell,
  - no defined function is longer than ~10 lines (a notebook cell that hides real logic
    inside a helper function is not a thin driver, it is the module in disguise);
  - no cell imports `data.gate` or `AnswerKey` — the firewall boundary (CLAUDE.md §5) a
    notebook must never cross.
"""
from __future__ import annotations

import ast
import glob
import json
import os

import pytest

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
MAX_FUNCTION_LINES = 10
FORBIDDEN_IMPORTS = ("data.gate", "AnswerKey")


def _all_notebooks():
    """Every `.ipynb` under `notebooks/`, discovered rather than hard-coded so a future
    notebook is covered automatically."""
    return sorted(glob.glob(os.path.join(NOTEBOOKS_DIR, "*.ipynb")))


def _code_cell_sources(nb_path):
    with open(nb_path) as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            yield "".join(cell["source"])


def _function_line_counts(source: str):
    """(name, line_count) for every function a cell's source defines."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, (node.end_lineno - node.lineno + 1)


def _imported_names(source: str):
    """Every dotted module path and every `from X import Y` name a cell imports."""
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                names.add(f"{module}.{alias.name}" if module else alias.name)
                names.add(alias.name)
    return names


def test_the_two_redesign_notebooks_exist():
    expected = {"redesign_pipeline.ipynb", "redesign_plots.ipynb"}
    found = {os.path.basename(p) for p in _all_notebooks()}
    missing = expected - found
    assert not missing, f"missing notebook(s) under {NOTEBOOKS_DIR}: {missing}"


@pytest.mark.parametrize("notebook", ["redesign_pipeline.ipynb", "redesign_plots.ipynb"])
def test_no_cell_defines_a_function_longer_than_ten_lines(notebook):
    path = os.path.join(NOTEBOOKS_DIR, notebook)
    if not os.path.isfile(path):
        pytest.fail(f"missing {path}")
    for source in _code_cell_sources(path):
        for name, n_lines in _function_line_counts(source):
            assert n_lines <= MAX_FUNCTION_LINES, (
                f"{notebook}: function {name!r} is {n_lines} lines "
                f"(> {MAX_FUNCTION_LINES}) — not a thin driver")


@pytest.mark.parametrize("notebook", ["redesign_pipeline.ipynb", "redesign_plots.ipynb"])
def test_no_cell_imports_the_firewall_boundary(notebook):
    path = os.path.join(NOTEBOOKS_DIR, notebook)
    if not os.path.isfile(path):
        pytest.fail(f"missing {path}")
    for source in _code_cell_sources(path):
        imported = _imported_names(source)
        for forbidden in FORBIDDEN_IMPORTS:
            hit = {n for n in imported
                   if n == forbidden or n.endswith("." + forbidden)
                   or n.startswith(forbidden + ".")}
            assert not hit, f"{notebook} imports forbidden {forbidden!r}: {hit}"


@pytest.mark.parametrize("notebook", ["redesign_pipeline.ipynb", "redesign_plots.ipynb"])
def test_notebooks_are_committed_unexecuted(notebook):
    """Both notebooks are committed with no outputs and no execution_count, rather than
    carrying stale or fabricated output. UPDATED at Task 16: the original reason was that
    neither could be run at all (the launch script did not exist and no run directory did
    either). Both now exist, and the contract is kept anyway — a committed output is a
    snapshot that silently ages out of agreement with the run directory it came from, and
    `experiments/redesign_r2/phase1/` is the tracked record instead."""
    path = os.path.join(NOTEBOOKS_DIR, notebook)
    if not os.path.isfile(path):
        pytest.fail(f"missing {path}")
    with open(path) as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None, (
            f"{notebook}: a code cell carries an execution_count; it looks executed")
        assert cell.get("outputs") == [], (
            f"{notebook}: a code cell carries stored outputs; it looks executed")
