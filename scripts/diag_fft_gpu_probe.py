"""diag_fft_gpu_probe.py — CUDA fp64 ETDRK4 per-step cost at 256^2/512^2, B in (1,4).

Fills the gap in D2's torch_timings table (which stopped at 128^2): the per-step cost of
the training-relevant 512^2 grid on this machine's GPU, using D2's verified torch
integrator (bit-equivalent to the numpy reference at 1.1e-13, D2 torch_equivalence).
Evidence for the GPU-pipeline decision (owner direction 2026-08-12).

DIAGNOSTIC SCRIPT — no training, no payload.h5 (fixture = the tracked recovered
checkpoint), so NOT a tests/test_firewall.py::FORBIDDEN entry.
Writes experiments/diag_fft/gpu_probe/results.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
from diag_fft_d2 import (load_fixture, linear_kstar, _torch_reaction_builder,
                         torch_half_coeffs, integrate_etdrk4_rfft_torch)
from rngrn.losses.terms import steady_state

ckpt = ("experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
        "checkpoints/model.pt")
model = load_fixture(ckpt)
kstar_lin, sig_max, _ = linear_kstar(model)[0], None, None
# linear_kstar returns tuple; take first elem defensively
if isinstance(kstar_lin, tuple):
    kstar_lin = kstar_lin[0]
L = 8.0 * 2.0 * np.pi / float(kstar_lin)
xs, ok = steady_state(model)
assert ok
J = model.jacobian(xs, create_graph=False).detach()
jac_rate = float(torch.linalg.eigvals(J).abs().max())
dt = 0.2 / jac_rate
D = model.D.detach().numpy()
dev = torch.device("cuda")
reaction_t = _torch_reaction_builder(model, dev)
xstar = xs.detach().numpy()
rng = np.random.default_rng(0)

from rngrn.utils import git_revision  # noqa: E402

out_dir = "experiments/diag_fft/gpu_probe"
os.makedirs(out_dir, exist_ok=True)
report = dict(ckpt=ckpt, L=L, dt=dt, device=torch.cuda.get_device_name(0),
              dtype="float64", git_rev=git_revision(),
              torch_version=torch.__version__, timed_steps=200, warmup_steps=20,
              rows=[])

print(f"L={L:.2f} dt={dt:.4f} fp64 on {torch.cuda.get_device_name(0)}", flush=True)
for n in (256, 512):
    coeffs = torch_half_coeffs(D, n, L, dt, dev)
    for B in (1, 4):
        X0 = torch.from_numpy(
            xstar[:, None, None] + 1e-2 * rng.standard_normal((B, 3, n, n))
        ).to(dev)
        # warmup 20 steps (JIT/cuFFT plan), then timed 200
        integrate_etdrk4_rfft_torch(X0, reaction_t, n, dt, 20, coeffs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, blew = integrate_etdrk4_rfft_torch(X0, reaction_t, n, dt, 200, coeffs)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        vram = torch.cuda.max_memory_allocated() / 2 ** 30
        print(f"n={n} B={B}: {el / 200 * 1e3:.3f} ms/step "
              f"({el / 200 / B * 1e3:.3f} ms/step/member) blew={blew} "
              f"vram={vram:.2f} GiB", flush=True)
        report["rows"].append(dict(n=n, B=B, ms_per_step=el / 200 * 1e3,
                                   ms_per_step_per_member=el / 200 / B * 1e3,
                                   blew_up=bool(blew), vram_peak_gib=vram))
        torch.cuda.reset_peak_memory_stats()

with open(os.path.join(out_dir, "results.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"wrote {out_dir}/results.json", flush=True)
