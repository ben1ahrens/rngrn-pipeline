"""utils.py — seeding, provenance, and side-neutral numeric helpers.

FIREWALL NOTE. This module is deliberately NEUTRAL: it reads no answer key and imports
nothing from `data/` or `scoring/`, so both recovery-side and scoring-side code may import
it. That is why `d_ratio_of` lives here (moved 2026-08-04) rather than in
`scoring/plausibility.py`: `history.py` runs inside the Adam loop and needs the D-ratio,
and importing the scoring package from recovery-side code is forbidden
(tests/test_firewall.py::test_recovery_side_does_not_import_the_scoring_package). Copying
the function instead would have let the two definitions drift, which matters because the
definition is a recorded science decision (docs/DECISIONS.md D2). Keep this module free of
answer-key dependencies.
"""
from __future__ import annotations
import os
import platform
import subprocess
import random


def seed_everything(seed: int):
    """Seed Python, NumPy and torch from one config field."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def set_deterministic(enabled: bool):
    """Turn on (or off) torch's strict determinism controls.

    CUBLAS_WORKSPACE_CONFIG must be set before CUDA is initialised to take effect; setting
    it here is a best-effort no-op once a CUDA context already exists, which is fine since
    this project defaults to CPU (see worker brief: CUDA is 3x slower per training step).
    PYTHONHASHSEED cannot be applied retroactively to an already-running interpreter; it is
    set here only so subprocesses inherit it.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        import torch
        torch.use_deterministic_algorithms(enabled)
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = not enabled
    except ImportError:
        pass


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _to_numpy(x):
    import numpy as np
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

    Recorded as docs/DECISIONS.md D2. Lives in utils (not scoring/plausibility.py, which
    re-exports it for backward compatibility) so that recovery-side `history.py` can use
    the SAME definition without importing the scoring package — see this module's docstring.

    NOTE this is NOT the only D-ratio in the codebase: `scoring/lgen.py::lgen_D_ratio` and
    `scoring/reproducibility.py::repro_D_ratio` are max/min over all species, a different
    statistic. Do not compare them.
    """
    import numpy as np
    D = _to_numpy(D).ravel()
    if D.size < 2:
        raise ValueError(f"D_ratio needs at least 2 species, got {D.size}")
    if np.any(D <= 0):
        raise ValueError(f"D must be strictly positive, got {D!r}")
    sorted_D = np.sort(D)
    lo, hi = sorted_D[-2], sorted_D[-1]
    return float(hi / lo)


def provenance(extra: dict | None = None) -> dict:
    """Record what produced a result: code revision, environment, device."""
    prov = dict(
        git_revision=git_revision(),
        python=platform.python_version(),
        platform=platform.platform(),
    )
    try:
        import torch
        prov["torch"] = torch.__version__
        prov["cuda_available"] = torch.cuda.is_available()
        prov["device"] = ("cuda:" + torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "cpu")
    except Exception:
        prov["device"] = "cpu"
    if extra:
        prov.update(extra)
    return prov
