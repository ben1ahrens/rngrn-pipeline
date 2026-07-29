"""utils.py — seeding and provenance (reproducibility hygiene)."""
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
