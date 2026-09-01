"""r3_batch_curve.py — the B/K curve WITH R3's forward solve in the loop (Task 17).

WHY THIS IS A NEW MEASUREMENT, NOT A RE-READ OF T16 (R2's `task-16-report.md`,
`.superpowers/sdd/PLAN_redesign/task-16-report.md`). T16's curve is the *ignition-only*
path: a pinned, box-reparameterised model with `pin_xstar` set, so `beta` is derived and
**no steady-state Newton solve runs at all** — its B=64/128/256/512 rungs cost 0.158-0.461
ms/member-step and 1.9-2.0 GB RSS FLAT in B (host RSS there is dominated by the CUDA
context, not the batch). R3's Phase-II step is a completely different shape: a *batched*
ETDRK4 relax (`forward.relax_to_pattern_torch_batched`, which allocates
`_half_coeffs_batched`'s (b, N, n, n//2+1, 32) complex128 contour-integral tensor — ~58 MB
at b=8/n=96, ~230 MB at b=32/n=96, per batched solve, PER ADAM STEP, uncached because dt
and D move every step) followed by a per-member Newton polish and, for the path this task
measures as PRIMARY, a per-member truncated-unrolled differentiable segment
(`unrolled.unrolled_relax`, no batched twin exists — `unrolled.py`'s own docstring says so).
T16's curve therefore BOUNDS NOTHING about this one; it is read into this script's output
only as a labelled reference row (Step 1).

REGISTER ITEM 8 (owner ruling, ledgered same day as this task): the UNROLLED path is the
DEFAULT gradient path for ALL members in R4. This script therefore measures the UNROLLED
arm as the PRIMARY curve. The ADJOINT arm (`forward.BatchedPatternSolver.solve_subset`,
which is what a training step would use for a member whose Newton actually converges) is
measured as a SECONDARY comparison column, at a small subset of B, exactly as
`scripts/r3_fd_ab.py` selects paths explicitly rather than going through
`recover._spectral_solve_with_stall_switch` (that switch is hard-wired to the CALLER's
grid and is R4's integration, not this task's).

FIREWALL (CLAUDE.md §5). This script opens NO `payload.h5` and reads NO `AnswerKey`. Its
only input is the tracked known-Turing checkpoint `r3_fd_ab.py` already uses; every batch
member is a small perturbation of it, and every loss target is a synthetic surrogate frame
built the same way `r3_fd_ab.py.surrogate_frame` does. It therefore owes no entry in
`tests/test_firewall.py::FORBIDDEN` — same reasoning as `r3_fd_ab.py`, which is not listed
either. Reused directly from that module (see the import below) rather than re-derived:
`load_fixture`, `theta_dim`, `theta_params`, `operating_point`, `surrogate_frame`. This is
DRY, not a new dependency — both scripts live in `scripts/`, neither is recovery-side, and
neither touches the data firewall.

MEASUREMENT DESIGN. Each cell (one arm, one B) runs in a FRESH SPAWNED PROCESS (the
`r3_fd_ab.py::measure_cost` pattern) so `ru_maxrss` is that cell's own peak, not a
high-water mark inherited from an earlier, smaller cell. Threads pinned to 1 for BOTH torch
and the BLAS/FFT libraries the CPU relax and Newton legs use (Task 11's ~215x finding).
ONE step per cell (no warm-up, no repeats): `_spectral_solve_with_stall_switch`'s own
docstring says a real R3 step is "NOT warm-started across calls ... fresh relax on every
call, by design", and `r3_stall_survey.py` measured that this costs ~70s per solve on this
machine — an order beyond `r3_fd_ab.py`'s warm-started reps=3 convention, so a single
measurement per cell is what stays affordable, and it is what the design actually is doing
every stalled or freshly-ignited call.

THE §7a MEMORY GUARD, per cell. Before launching a cell, this script reads the current
host `MemAvailable` and, once at least one same-arm cell has been measured, LINEARLY
EXTRAPOLATES that cell's peak RSS from the most recent measured cells (`_half_coeffs_batched`'s
own cost note says the dominant term is B-linear). If `MemAvailable - projected_peak <
--mem-floor-mb` (default 8192, the §7a guard's own floor), the cell is REFUSED and the
sweep stops there WITHOUT launching it — recorded as a finding, not a failure. After a
cell actually runs, its own measured `mem_available_mb_after` is checked against the same
floor as a second, empirical stop condition.

Run (guarded; this is the whole point of the measurement):

    bash scripts/guarded_run.sh .venv/bin/python scripts/r3_batch_curve.py \\
        --out experiments/redesign_r3/batch_curve
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import resource
import sys
import time

# BEFORE numpy/torch (and before importing r3_fd_ab, which imports them too) — see that
# module's own comment. `setdefault` so an explicit environment still wins.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import r3_fd_ab as fdab  # noqa: E402  (path must be set first)

from rngrn.forward import (BatchedPatternSolver,               # noqa: E402
                           relax_to_pattern_torch_batched)
from rngrn.losses.spectral import (SpectralConfig, build_frame_targets,  # noqa: E402
                                   spectral_terms_batched)
from rngrn.model import BatchedRNGRN                             # noqa: E402
from rngrn.unrolled import SEGMENT_STEPS_DEFAULT, unrolled_relax  # noqa: E402

REPO = fdab.REPO
PERT_SCALE = 1e-3          # r3_fd_ab.py's own cost-measurement convention


# ------------------------------------------------------------------------------- helpers

def _mem_available_mb() -> float:
    """`/proc/meminfo`'s MemAvailable, in MiB — the §7a guard's own currency."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    raise RuntimeError("/proc/meminfo has no MemAvailable line")


def _build_members(batch: int, seed: int) -> tuple:
    """`batch` members, NESTED by index (member b's perturbation does not depend on
    `batch`, so a B=8 cell's first member is identical to a B=32 cell's first member —
    T16's own nesting convention). Member 0 is the unperturbed fixture; the rest are
    `PERT_SCALE`-displaced, matching `r3_fd_ab.py::_cost_child`'s own convention, which
    keeps every member Turing-unstable and inside the shared commensurate box's validity.

    Returns (members, ops, dropped) — `ops` is each member's own `operating_point()`
    (xstar, gamma, ...); `dropped` records any member whose steady state or Turing check
    failed (excluded from the batch, not silently substituted).
    """
    rng = np.random.default_rng(seed)
    dim = fdab.theta_dim(fdab.load_fixture())
    members, ops, dropped = [], [], []
    for b in range(batch):
        pert = None if b == 0 else PERT_SCALE * rng.standard_normal(dim)
        m = fdab.load_fixture(pert)
        try:
            op = fdab.operating_point(m)
        except RuntimeError as exc:
            dropped.append({"member": b, "reason": str(exc)[:160]})
            continue
        members.append(m)
        ops.append(op)
    return members, ops, dropped


# --------------------------------------------------------------------------- one cell

def _cell_child(q, arm: str, batch: int, segment_steps: int, threads: int,
                seed: int) -> None:
    """One (arm, B) cell's forward+backward, in a FRESH process — see module docstring."""
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(threads))

    base = fdab.load_fixture()
    base_op = fdab.operating_point(base)
    n, L = base_op["n"], base_op["L"]
    cfg = SpectralConfig()
    frame = fdab.surrogate_frame(base_op, seed)
    targets = build_frame_targets(frame, L, base_op["khat"], cfg)

    members, ops, dropped = _build_members(batch, seed + 5000)
    b_built = len(members)
    rss_setup = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if b_built == 0:
        q.put(dict(arm=arm, batch=int(batch), b_built=0, b_dropped=len(dropped),
                   dropped=dropped, error="no member stayed Turing-unstable",
                   rss_setup_mb=rss_setup, peak_rss_mb=rss_setup,
                   mem_available_mb_after=_mem_available_mb()))
        return

    bmodel = BatchedRNGRN(members)
    xstar_batch = torch.stack([torch.from_numpy(o["xstar"]) for o in ops])
    dt_arr = np.array([0.2 / o["gamma"] for o in ops])
    seeds_list = list(range(b_built))
    idx_all = torch.arange(b_built, dtype=torch.long)

    if arm == "adjoint":
        # `BatchedPatternSolver.solve_subset`, PRODUCTION API, fresh (no `_warm` seeded) —
        # so pass 2 of its state machine runs the SAME batched relax (`_half_coeffs_batched`
        # included) that the unrolled arm below runs directly, then a per-member Newton
        # polish, then the LSMR adjoint in `BatchedPatternSolve.backward`.
        solver = BatchedPatternSolver(bmodel, n=n, L=L, seeds=seeds_list,
                                      warm_mode="newton")

        def step():
            u, ok_members, reasons = solver.solve_subset(list(range(b_built)),
                                                          xstar_batch)
            if u is None:
                raise RuntimeError(f"cell solved nothing: {reasons}")
            vals = spectral_terms_batched(u, targets, cfg, members=ok_members)[0]
            loss = sum(v.sum() for v in vals.values())
            torch.autograd.grad(loss, fdab.theta_params(bmodel))
            return len(ok_members), reasons
    else:
        # UNROLLED (primary, register item 8). No batched twin of `unrolled_relax` exists
        # (`unrolled.py`'s own docstring), so the batched RELAX is run directly to get every
        # member's saturated field (bypassing Newton entirely — the unrolled path's caller
        # contract needs a saturated warm state, not a Newton-polished one), then each
        # member's differentiable segment + backward runs in a SERIAL loop. Because each
        # member's graph is freed (no `retain_graph`) before the next member starts, the
        # segment's own memory (Task 14: +433-516 MB at S=128 checkpointed) does not
        # accumulate across B — only the batched relax's `_half_coeffs_batched` term does.
        def step():
            fields, reasons = relax_to_pattern_torch_batched(
                bmodel, idx_all, xstar_batch.detach().cpu().numpy(), n, L, dt_arr,
                seeds_list, torch.device("cpu"))
            solved = 0
            for j, m in enumerate(members):
                if fields[j] is None:
                    continue
                u = unrolled_relax(m, fields[j].detach(), n, L, float(dt_arr[j]),
                                   segment_steps=segment_steps, checkpoint_every=1)
                vals = spectral_terms_batched(u[None], targets, cfg, members=[0])[0]
                loss = sum(v.sum() for v in vals.values())
                torch.autograd.grad(loss, fdab.theta_params(m))
                solved += 1
            return solved, {str(j): r for j, r in enumerate(reasons) if r is not None}

    t0 = time.perf_counter()
    solved, reasons = step()
    seconds = time.perf_counter() - t0
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    q.put(dict(
        arm=arm, batch=int(batch), b_built=int(b_built), b_dropped=len(dropped),
        dropped=dropped, members_solved=int(solved),
        s_per_step=seconds,
        member_steps_per_s=(solved / seconds) if seconds > 0 else float("nan"),
        peak_rss_mb=rss_peak, rss_setup_mb=rss_setup,
        rss_step_increment_mb=rss_peak - rss_setup,
        mem_available_mb_after=_mem_available_mb(),
        cuda_max_alloc_mb=float("nan"),          # CPU only — see module docstring
        threads=int(threads),
        segment_steps=(int(segment_steps) if arm == "unrolled" else None),
        reasons=reasons,
    ))


def measure_cell(arm: str, batch: int, segment_steps: int, threads: int, seed: int,
                 timeout: float) -> dict:
    """One cell, isolated in a fresh process. A per-cell TIMEOUT is reported as a `row`
    with `"error"` set — same shape as a real in-process failure — rather than raised,
    because an uncaught timeout would abort `main()` before the JSON is written and lose
    every cell already measured. This is a WALL-CLOCK safety valve (`--cell-timeout-s`),
    distinct from the §7a MEMORY guard: a cell that times out here has not necessarily
    violated the memory floor, and the report must not conflate the two."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_cell_child, args=(q, arm, batch, segment_steps, threads, seed))
    p.start()
    try:
        row = q.get(timeout=timeout)
    except Exception as exc:                    # queue.Empty on timeout, or a broken pipe
        if p.is_alive():
            p.terminate()
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
                p.join()
        row = dict(arm=arm, batch=int(batch),
                   error=f"cell timed out or died after {timeout:g}s ({type(exc).__name__}: "
                         f"{exc}) — a script-level wall-clock safety valve, NOT the §7a "
                         "memory guard")
    else:
        p.join()
    if "error" in row:
        print(f"  cell [{arm} B={batch}] FAILED: {row['error']}", flush=True)
    else:
        print(f"  cell [{arm} B={batch}] {row['s_per_step']:.2f} s/step, "
              f"{row['member_steps_per_s']:.3f} member-steps/s, "
              f"solved {row['members_solved']}/{row['b_built']} "
              f"(dropped {row['b_dropped']} pre-batch), "
              f"peak RSS {row['peak_rss_mb']:.0f} MB, "
              f"MemAvailable after {row['mem_available_mb_after']:.0f} MB", flush=True)
    return row


# ---------------------------------------------------------------------- memory projection

def _project_next_rss(history: list, b_next: int) -> float | None:
    """Linear extrapolation of `peak_rss_mb` vs `B` from prior MEASURED cells of the SAME
    arm. `_half_coeffs_batched`'s own cost note (~58 MB @ b=8, ~230 MB @ b=32 at n=96) is
    linear in B to within the arithmetic given there, so a linear fit is the right shape,
    not merely a convenient one."""
    if not history:
        return None
    if len(history) == 1:
        b0, r0 = history[0]
        return r0 * (float(b_next) / float(b0))
    pts = history[-3:]
    Bs = np.array([p[0] for p in pts], dtype=float)
    Rs = np.array([p[1] for p in pts], dtype=float)
    slope, intercept = np.polyfit(Bs, Rs, 1)
    return float(intercept + slope * b_next)


# --------------------------------------------------------------------------- reporting

def _render_report_table(payload: dict) -> str:
    """`batch_curve.json` -> the markdown table `task-17-report.md` Step 2 wants — written
    now so filling in the report, once the guarded sweep produces the JSON, is mechanical
    rather than hand-transcribed. KEY SET matches T16's own (module docstring): `s_per_step`,
    `member_steps_per_s`, `peak_rss_mb`, `cuda_max_alloc_mb`, `mem_available_mb_after`.
    """
    lines = ["| arm | B | s/step | member-steps/s | peak RSS MB | CUDA alloc MB | "
             "MemAvailable after MB | solved/built |",
             "|---|---|---|---|---|---|---|---|"]
    for arm_name, arm_rows in (("unrolled", payload.get("unrolled_sweep", [])),
                               ("adjoint", payload.get("adjoint_comparison", []))):
        for r in arm_rows:
            if "error" in r:
                lines.append(f"| {arm_name} | {r['batch']} | FAILED: {r['error']} | | | | | |")
                continue
            lines.append(
                f"| {arm_name} | {r['batch']} | {r['s_per_step']:.2f} | "
                f"{r['member_steps_per_s']:.3f} | {r['peak_rss_mb']:.0f} | "
                f"{r['cuda_max_alloc_mb']!r} | {r['mem_available_mb_after']:.0f} | "
                f"{r['members_solved']}/{r['b_built']} |")
    if payload.get("refused"):
        lines.append("")
        lines.append("**Refused rungs (memory-projected, never launched):**")
        for r in payload["refused"]:
            lines.append(
                f"- arm={r['arm']} B={r['batch']}: projected peak RSS "
                f"{r['projected_peak_rss_mb']:.0f} MB against MemAvailable "
                f"{r['mem_available_mb']:.0f} MB (floor {r['mem_floor_mb']:.0f} MB) — "
                f"basis: {r['basis']}")
    lines.append("")
    lines.append(f"Largest unrolled B measured: **{payload.get('largest_unrolled_b_measured')}**")
    return "\n".join(lines)


def _build_payload(a, rows: list, adjoint_rows: list, refused: list,
                   largest_unrolled_b: int, t_start: float) -> dict:
    return {
        "what": "R3 Phase-II B/K curve WITH the batched forward solve in the loop "
                "(PLAN_redesign_R3 Task 17). Primary arm: unrolled (register item 8's "
                "ruling). Secondary: adjoint, at a subset of B.",
        "not_this_curve": "the T16/R2 ignition-only curve — read separately from "
                          "task-16-report.md (.superpowers/sdd/PLAN_redesign/), it "
                          "parameterizes nothing here (see module docstring).",
        "method": {
            "checkpoint": str(fdab.CKPT.relative_to(REPO)),
            "key_set": ["s_per_step", "member_steps_per_s", "peak_rss_mb",
                       "cuda_max_alloc_mb", "mem_available_mb_after"],
            "device": "cpu", "threads": a.threads, "torch": torch.__version__,
            "segment_steps": a.segment_steps, "checkpoint_every": 1,
            "perturbation_scale": PERT_SCALE, "seed": a.seed,
            "mem_floor_mb": a.mem_floor_mb, "cell_timeout_s": a.cell_timeout_s,
            "reps_per_cell": 1,
            "reps_note": "ONE step per cell, no warm-up: a real R3 step is not "
                         "warm-started across calls (fresh relax every call, by design; "
                         "r3_stall_survey.py measured ~70s/solve for exactly this reason) "
                         "— r3_fd_ab.py's reps=3/warm-started convention does not apply "
                         "here and would misrepresent the cost this task measures.",
            "argv": list(sys.argv),
        },
        "unrolled_sweep": rows,
        "adjoint_comparison": adjoint_rows,
        "refused": refused,
        "largest_unrolled_b_measured": largest_unrolled_b,
        "total_seconds": time.perf_counter() - t_start,
    }


# -------------------------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=False,
                    help="run directory; results land in <out>/results/batch_curve.json. "
                         "Required unless --report-from is given.")
    ap.add_argument("--report-from", default=None,
                    help="skip the sweep entirely; read an existing batch_curve.json and "
                         "print the Step-2 markdown table for the report, then exit")
    ap.add_argument("--b-start", type=int, default=1)
    ap.add_argument("--b-max", type=int, default=512,
                    help="T16's own ceiling (register item 14) — do not exceed without "
                         "re-reading whether it is still the right comparison point")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch AND BLAS/FFT threads; pinned at 1 (Task 11's 215x)")
    ap.add_argument("--segment-steps", type=int, default=SEGMENT_STEPS_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-floor-mb", type=float, default=8192.0,
                    help="the §7a guard's own MemAvailable floor")
    ap.add_argument("--adjoint-batches", type=int, nargs="*", default=[1, 8],
                    help="B values at which to ALSO measure the adjoint arm as a "
                         "secondary comparison column (register item 8: unrolled is "
                         "primary). Only attempted for B <= the largest B the unrolled "
                         "sweep actually reached.")
    ap.add_argument("--cell-timeout-s", type=float, default=1800.0)
    a = ap.parse_args()

    if a.report_from:
        payload = json.loads(pathlib.Path(a.report_from).read_text())
        print(_render_report_table(payload))
        return
    if not a.out:
        ap.error("--out is required unless --report-from is given")

    t_start = time.perf_counter()
    out_dir = pathlib.Path(a.out)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    print(f"host MemAvailable at start: {_mem_available_mb():.0f} MB, "
          f"floor {a.mem_floor_mb:.0f} MB", flush=True)

    # ---------------------------------------------------- primary: UNROLLED path sweep --
    rows: list = []
    refused: list = []
    history: list = []
    out_path = out_dir / "results" / "batch_curve.json"

    def _save() -> None:
        """Write the CURRENT state after every cell, not just at the end. A cell that
        blows `--cell-timeout-s` (a script-level wall-clock valve — see `measure_cell`)
        used to abort `main()` before the one-shot final write, silently discarding every
        cell already measured. Cheap (a handful of small dicts) and makes the run safe to
        interrupt at any point."""
        payload = _build_payload(a, rows, adjoint_rows, refused,
                                 max((r["batch"] for r in rows if "error" not in r),
                                     default=0),
                                 t_start)
        out_path.write_text(json.dumps(payload, indent=2))

    adjoint_rows: list = []          # populated below; declared here so _save() can close over it
    b = a.b_start
    while b <= a.b_max:
        mem_now = _mem_available_mb()
        projected = _project_next_rss(history, b)
        if projected is not None and mem_now - projected < a.mem_floor_mb:
            entry = dict(arm="unrolled", batch=b, projected_peak_rss_mb=projected,
                        mem_available_mb=mem_now, mem_floor_mb=a.mem_floor_mb,
                        basis="linear extrapolation from prior cells: "
                              + ", ".join(f"B={h[0]} rss={h[1]:.0f}MB" for h in history))
            print(f"REFUSING B={b} (unrolled): projected peak RSS {projected:.0f} MB would "
                  f"leave MemAvailable {mem_now - projected:.0f} MB < floor "
                  f"{a.mem_floor_mb:.0f} MB — not launching", flush=True)
            refused.append(entry)
            break
        row = measure_cell("unrolled", b, a.segment_steps, a.threads, a.seed,
                           a.cell_timeout_s)
        rows.append(row)
        _save()
        if "error" not in row:
            history.append((b, row["peak_rss_mb"]))
            if row["mem_available_mb_after"] < a.mem_floor_mb:
                print(f"STOPPING sweep: B={b} left MemAvailable "
                      f"{row['mem_available_mb_after']:.0f} MB < floor "
                      f"{a.mem_floor_mb:.0f} MB — this rung is the last one measured",
                      flush=True)
                break
        else:
            print(f"STOPPING sweep: B={b} failed outright ({row.get('error')})",
                  flush=True)
            break
        b *= 2
    largest_unrolled_b = max((r["batch"] for r in rows if "error" not in r), default=0)

    # ------------------------------------------------- secondary: ADJOINT comparison ----
    for ab in sorted(set(a.adjoint_batches)):
        if ab > largest_unrolled_b:
            print(f"skipping adjoint B={ab}: exceeds the largest B the unrolled sweep "
                  f"reached ({largest_unrolled_b})", flush=True)
            continue
        mem_now = _mem_available_mb()
        # Conservative projection: reuse the UNROLLED row at this same B (relax cost is
        # shared; Newton+LSMR typically costs less than the unrolled segment does), so this
        # is an upper-bound proxy, not a separately-fit adjoint curve.
        proxy = next((r["peak_rss_mb"] for r in rows
                     if r.get("batch") == ab and "error" not in r), None)
        if proxy is not None and mem_now - proxy < a.mem_floor_mb:
            print(f"REFUSING adjoint B={ab}: unrolled-proxy peak RSS {proxy:.0f} MB would "
                  f"leave MemAvailable {mem_now - proxy:.0f} MB < floor "
                  f"{a.mem_floor_mb:.0f} MB — not launching", flush=True)
            refused.append(dict(arm="adjoint", batch=ab, projected_peak_rss_mb=proxy,
                                mem_available_mb=mem_now, mem_floor_mb=a.mem_floor_mb,
                                basis=f"unrolled row at the same B={ab}"))
            continue
        row = measure_cell("adjoint", ab, a.segment_steps, a.threads, a.seed,
                           a.cell_timeout_s)
        adjoint_rows.append(row)
        _save()

    # ------------------------------------------------------------------------- output ---
    _save()
    print(f"\nwrote {out_path}", flush=True)
    print(f"largest unrolled B measured: {largest_unrolled_b}", flush=True)
    if refused:
        print(f"refused {len(refused)} cell(s) on memory grounds — see 'refused' in the "
              f"output", flush=True)


if __name__ == "__main__":
    main()
