"""conftest.py — set OpenMP env BEFORE torch is imported.

Some sandboxed/containerised environments forbid pthread_setaffinity_np, which
makes the conda-forge torch build abort on import with 'OMP: Error #179'. Setting
these before the first torch import avoids the affinity call. Harmless on a normal
workstation/GPU box; delete if your environment doesn't need it.
"""
import os

os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_NUM_THREADS", "1")
