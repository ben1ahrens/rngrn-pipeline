#!/usr/bin/env python
"""lift_gpu_cost.py — measure the per-step cost of the lifted spatial integrator.

WHY THIS EXISTS. docs/REDESIGN_rngrn.md §5.2 carried an ESTIMATE of what a 512^2 lifted
run would cost. An estimate is not a measurement (CLAUDE.md §8), so this script produces
the number that replaces it, with enough metadata to be citable: device, grid, mu, dt,
steps timed, ms/step, and the numpy CPU leg beside it so the GPU figure is read against a
baseline rather than against nothing.

WHAT IS TIMED. `simulate_lifted*` returns `seconds` (the Strang loop alone, CUDA-synced
before it is read) and `nsteps_run`, so ms/step = 1e3 * seconds / nsteps_run excludes
one-time setup — coefficient build, host-to-device transfer, the steady-state solve — which
amortises to nothing over a real run and would otherwise inflate a short benchmark. One
untimed call precedes the timed one as warmup (allocator, cuFFT plan cache).

NOT a trainer, sweep or target-report — a one-off benchmark, so it runs directly under
.venv/bin/python and not through scripts/guarded_run.sh. It allocates one (N,N,n,n) gate
array (18.9 MB at N=3, n=512) plus the ETDRK4 work arrays; nowhere near the §7a ceiling.

Usage:
    .venv/bin/python scripts/lift_gpu_cost.py [--n 512] [--steps 200]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import time

import torch

from rngrn.eval import lifted, lifted_torch
from rngrn.eval.ladder import draw_models
from rngrn.eval.lifted import MU_BIO_CENTRAL
from rngrn.losses.terms import steady_state

torch.set_default_dtype(torch.float64)

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/lift_ladder/gpu_port/results/cost.json"

# The same draw tests/test_lifted_torch.py pins the CPU/GPU equivalence on, so the timed
# system is the one whose correctness is asserted in the suite.
FORM, SEED = "competitive", 23
L = 20.0


def _timed(run, steps):
    """(ms_per_step, nsteps_run) for `run(steps)`, after one untimed warmup call.

    `simulate_lifted*` floors nsteps at 200, so the denominator is the step count the run
    REPORTS, never the one requested.
    """
    run(steps)
    out = run(steps)
    return 1e3 * out["seconds"] / out["nsteps_run"], int(out["nsteps_run"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200,
                    help="steps per timed run (the integrator floors this at 200)")
    ap.add_argument("--no-cpu", action="store_true", help="skip the numpy baseline leg")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device — this benchmark measures the GPU leg and "
                           "refuses to report a CPU number as if it were one")

    model = draw_models(n=1, form=FORM, seed=SEED)[0]
    mu = MU_BIO_CENTRAL
    xs, ok = steady_state(model)
    assert ok, "benchmark draw has no converged steady state"
    xstar = xs.detach().cpu().numpy()
    D = model.D.detach().cpu().numpy()
    # dt from the shared step policy — the value a real run of this model would use.
    dt, sig_max, _, _, _ = lifted.step_policy(model, xstar, D, L, a.n, None, None, 200000)

    kw = dict(model=model, L=L, mu=mu, n=a.n, dt=dt, xstar=xstar, seed=0)
    cuda_ms, cuda_steps = _timed(
        lambda s: lifted_torch.simulate_lifted_torch(T=s * dt, device="cuda", **kw), a.steps)
    cpu_ms = cpu_steps = None
    if not a.no_cpu:
        cpu_ms, cpu_steps = _timed(
            lambda s: lifted.simulate_lifted(T=s * dt, **kw), a.steps)

    rec = dict(
        what="per-step wall cost of the lifted Strang integrator: "
             "eval/lifted_torch.simulate_lifted_torch (CUDA) vs "
             "eval/lifted.simulate_lifted (numpy CPU)",
        replaces="docs/REDESIGN_rngrn.md §5.2 ESTIMATE — this is the measured value",
        timed="the Strang loop only (`seconds`/`nsteps_run` from the run), one warmup run "
              "discarded; setup (coeffs, transfers, steady state) excluded",
        date=time.strftime("%Y-%m-%d"),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True).stdout.strip(),
        device=torch.cuda.get_device_name(0), host=platform.node(),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        dtype="float64",
        model=dict(form=FORM, draw_seed=SEED, N=int(model.N), n_hill=int(model.n_hill)),
        grid=dict(n=a.n, L=L), mu=float(mu), dt=float(dt), sig_max=float(sig_max),
        cuda=dict(steps_timed=cuda_steps, ms_per_step=float(cuda_ms)),
        cpu=(None if cpu_ms is None else
             dict(steps_timed=cpu_steps, ms_per_step=float(cpu_ms))),
        speedup_cpu_over_cuda=(None if cpu_ms is None else float(cpu_ms / cuda_ms)),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
