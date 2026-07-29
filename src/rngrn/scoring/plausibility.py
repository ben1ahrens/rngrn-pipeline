"""plausibility.py — biological viability of a RECOVERED model's OWN parameters.

WHY THIS EXISTS
---------------
None of the three stated goal components had a bound, prior, penalty, score, or run-index
column for biological plausibility anywhere in src/ (docs/CODE_REALITY.md: "no check that
recovered parameters lie in a plausible box, and no such column in the run index"). This
module supplies the missing SCORE side; losses/terms.py::param_prior supplies the matching
soft training-time prior, sharing this module's box loader and D-ratio definition.

FIREWALL: this module reads only the RECOVERED MODEL's own constrained parameters
(alpha, delta, beta, D — model.py properties). It never reads an answer key, a
ground-truth Jacobian, or anything from rd_models / data.solver / data.cache. It is safe
to import from recovery-side code (losses/terms.py does) as well as from scoring-side
code (validate.py does); tests/test_firewall.py enforces the recovery-side import
boundary on the modules that matter, and this module imports nothing forbidden.

THE BOX IS DATA, NOT CODE
--------------------------
Every bound and every citation lives in configs/bio_box.yaml, never here. This module
supplies zero fallback numbers: if the YAML has no row for a parameter, load_box's
caller gets a KeyError, not a guess.

USER DECISION ON RECORD
------------------------
Priors are centred on biologically viable LITERATURE values (D-ratio centre 7.5, the
measured Nodal/Lefty ratio), not on the values the synthetic training generators use
(D-ratio ~8-250, median ~135). A recovered D-ratio near the generators' population is
therefore EXPECTED to score poorly against this box — that is the intended consequence
of the decision, not a bug. See configs/bio_box.yaml and docs/STATE_OF_THE_SCIENCE.md
§11 for the full tension, and docs/ROBUSTNESS_MEASUREMENT.md §4.4 for the near-immobile
third node that makes a low D-ratio Turing-viable at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

DEFAULT_BOX_PATH = "configs/bio_box.yaml"


@dataclass(frozen=True)
class BoxRow:
    """One parsed row of bio_box.yaml. `low`/`high` are None for an unbounded or
    UNCITED row; `centre`/`spread` are only present for the D-ratio row's soft prior."""
    low: Optional[float]
    high: Optional[float]
    source: str
    centre: Optional[float] = None
    spread: Optional[float] = None

    @property
    def cited(self) -> bool:
        return self.source != "UNCITED"

    @property
    def bounded(self) -> bool:
        return self.cited and self.low is not None and self.high is not None


def load_box(path: str = DEFAULT_BOX_PATH) -> dict:
    """Load configs/bio_box.yaml into {name: BoxRow}. Fails loud on a malformed row
    (missing `source`) rather than silently treating it as UNCITED."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    box = {}
    for name, row in raw.items():
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {name!r} is not a mapping: {row!r}")
        if "source" not in row:
            raise ValueError(f"{path}: row {name!r} has no `source` — every row must "
                             f"cite a source, or be marked `source: UNCITED` explicitly")
        box[name] = BoxRow(
            low=row.get("low"), high=row.get("high"), source=row["source"],
            centre=row.get("centre"), spread=row.get("spread"))
    return box


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def d_ratio_of(D) -> float:
    """D_ratio = ratio of the two MOST MOBILE species' diffusivities: the largest
    divided by the second-largest, over ANY number of species N >= 2.

    [IMPL] design choice, not independently validated against an alternative
    definition (e.g. global max/min). Chosen specifically so a single near-immobile
    node — the mechanism docs/ROBUSTNESS_MEASUREMENT.md §4.4 measures as making 127/127
    three_gene systems stay strictly Turing when the SLOWEST diffuser is made immobile
    — never enters the ratio: for N=2 this reduces to the ordinary max/min ratio; for
    N>=3 the single smallest D (which an immobile node drives toward 0) is excluded by
    construction, so D_i -> 0 is never penalised here or in losses/terms.py::param_prior.
    """
    D = _to_numpy(D).ravel()
    if D.size < 2:
        raise ValueError(f"D_ratio needs at least 2 species, got {D.size}")
    if np.any(D <= 0):
        raise ValueError(f"D must be strictly positive, got {D!r}")
    sorted_D = np.sort(D)
    lo, hi = sorted_D[-2], sorted_D[-1]
    return float(hi / lo)


def _in_box(value: float, row: BoxRow) -> Optional[bool]:
    """True/False for a cited, bounded row; None ("not applicable") for an UNCITED or
    unbounded row — never a fabricated verdict."""
    if not row.bounded:
        return None
    return bool(row.low <= value <= row.high)


def plausibility_report(alpha, delta, beta, D, box: Optional[dict] = None,
                        box_path: str = DEFAULT_BOX_PATH) -> dict:
    """Score a RECOVERED model's own parameters against configs/bio_box.yaml.

    alpha, delta, beta, D : the model's own constrained parameters (RNGRN.alpha,
        .delta, .beta, .D — numpy or torch, any shape; each is flattened before
        comparison). NEVER pass an answer-key value here.
    box : a pre-loaded {name: BoxRow}; if None, loaded from box_path via load_box().

    Returns a flat dict of scalars, meant for a single out.update(...) call in
    validate.py::score_recovery:
      plausibility_<name>_min/_max  : observed range of that parameter in this run
      plausibility_<name>_in_box    : True/False/None (None = UNCITED or unbounded row)
      plausibility_<name>_source    : the citation string from bio_box.yaml
      plausibility_d_ratio_value    : d_ratio_of(D) for this run
      plausibility_n_scored         : how many of the 4 rows contributed a True/False
      plausibility_score            : mean of the contributing True/False rows, as a
                                       float in [0, 1]; NaN if every row is UNCITED or
                                       unbounded (never silently 0 or 1 in that case).
    """
    box = load_box(box_path) if box is None else box
    out = {}
    verdicts = {}

    for name, arr in (("alpha", alpha), ("delta", delta), ("beta", beta)):
        row = box.get(name)
        if row is None:
            raise KeyError(f"{box_path}: no row for {name!r}")
        a = _to_numpy(arr).ravel()
        # a whole-parameter row is in-box only if EVERY entry is; a single out-of-box
        # entry (e.g. one runaway alpha[i,j]) makes the parameter implausible.
        in_box = None if not row.bounded else bool(np.all((a >= row.low) & (a <= row.high)))
        out[f"plausibility_{name}_min"] = float(a.min())
        out[f"plausibility_{name}_max"] = float(a.max())
        out[f"plausibility_{name}_in_box"] = in_box
        out[f"plausibility_{name}_source"] = row.source
        verdicts[name] = in_box

    d_row = box.get("d_ratio")
    if d_row is None:
        raise KeyError(f"{box_path}: no row for 'd_ratio'")
    dr = d_ratio_of(D)
    dr_in_box = _in_box(dr, d_row)
    out["plausibility_d_ratio_value"] = dr
    out["plausibility_d_ratio_in_box"] = dr_in_box
    out["plausibility_d_ratio_source"] = d_row.source
    verdicts["d_ratio"] = dr_in_box

    scored = [v for v in verdicts.values() if v is not None]
    out["plausibility_n_scored"] = len(scored)
    out["plausibility_score"] = float(np.mean(scored)) if scored else float("nan")
    return out
