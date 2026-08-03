"""history.py — the TRAINING TRAJECTORY of a recovery, recorded at a thinned cadence.

WHY. `recover()` used to keep one summary row per RESTART (total, sig_max, sig_max_pos,
kstar_model, rel_err) and nothing per step. So "how did the loss and the parameters evolve"
was unplottable: a run that reached a good loss by step 100 and one that crawled there by
step 1900 left identical records, and the D-ratio trajectory — the project's open question
is whether recovery reaches biologically plausible diffusivities — was invisible.

WHAT IS RECORDED, per recorded step and per MEMBER (restart):
  * every loss TERM separately (`L_kstar`, `L_turing`, ... whatever `losses.total` produced)
    and the total;
  * the WEIGHTS actually in force at that step (`w_*`). Load-bearing, not decoration:
    DataFirstStaging holds `turing` at 0 for the first `staging_off_frac` of the budget and
    then ramps it, and RatioWeighting recomputes every `ratio_update_every` steps — a
    loss-vs-step curve read without its weight is misleading;
  * the CONSTRAINED PHYSICAL parameters KA, KR, alpha, delta, beta, D (not raw theta —
    theta is a link-function artefact, the physical parameters are what a paper plots),
    plus the derived D-ratio;
  * the per-step diagnostics recovery already computes: kstar_model, sig_max, sig_max_pos,
    rel_err;
  * steady-state health: `ss_converged` per member, and `death_step` — the step at which a
    member's steady state diverged and it was abandoned — so a member dying mid-run reads as
    an EVENT rather than as an unexplained gap.

THINNING (docs/DECISIONS.md D-PLOT-2). Every step x 64 restarts x 36 physical parameters is
~921k floats for a 400-step run. `every` (config `train.history_every`, default 10) thins the
step axis; step 0, the last training step and the final evaluation step are recorded
UNCONDITIONALLY whatever the stride, so the endpoints of every curve are measured rather than
interpolated.

ALL MEMBERS ARE KEPT, thinned — never member 0 relabelled as "the run". On the batched path
the reported result is the BEST member, whose index is only known after the last step, so
recording one lane during training would either be the wrong lane or require guessing.

FIREWALL. Recovery-side: numpy only, no answer-key import, nothing here reads truth. Note
that `recover.py` does NOT import this module — `train.fit` constructs the recorder and passes
it in, and recovery only calls methods on whatever object it is handed, so the training loop's
import graph is unchanged. `tests/test_plot_arrays.py::test_history_is_recovery_side_safe`
audits this module's own imports statically.
"""
from __future__ import annotations

import numpy as np

from .scoring.plausibility import d_ratio_of

# The CONSTRAINED physical parameters, with their rank (2 = (N,N) matrix, 1 = (N,) vector).
# Order fixes the column order of the parameter trace; do not reorder without bumping
# plotdata.ARRAY_SCHEMA_VERSION.
PARAM_SPEC = (("KA", 2), ("KR", 2), ("alpha", 2), ("delta", 1), ("beta", 1), ("D", 1))

# Per-step diagnostics `losses.total` already puts in `parts`. Absent keys are recorded NaN
# only if they are absent on the FIRST recorded step too — see TrainingHistory._names.
DIAG_KEYS = ("total", "sig_max", "sig_max_pos", "kstar_model", "rel_err")

HIST_DTYPE = np.float32   # a trajectory is plotted, not differentiated; see plotdata.py


def param_names(N: int) -> list:
    """Column names of one member's parameter trace, e.g. 'KA[0,1]', 'delta[2]'."""
    names = []
    for name, rank in PARAM_SPEC:
        if rank == 2:
            names += [f"{name}[{i},{j}]" for i in range(N) for j in range(N)]
        else:
            names += [f"{name}[{i}]" for i in range(N)]
    return names


def _param_vector(model) -> np.ndarray:
    """One serial model's physical parameters, flattened in PARAM_SPEC order. (P,)"""
    return np.concatenate([
        np.asarray(getattr(model, name).detach().cpu().numpy(), dtype=float).ravel()
        for name, _rank in PARAM_SPEC])


def _param_block(bmodel) -> np.ndarray:
    """A BatchedRNGRN's physical parameters for ALL members at once, (B, P).

    One device->host transfer per PARAMETER, not per (parameter, member). On CUDA the
    per-member form cost 6*B tiny synchronising copies per recorded step (15744 for a
    400-step, 64-restart run at stride 10); this is 6 per recorded step.
    """
    B = bmodel.B
    return np.concatenate([
        np.asarray(getattr(bmodel, name).detach().cpu().numpy(), dtype=float).reshape(B, -1)
        for name, _rank in PARAM_SPEC], axis=1)


def _d_slice(vec: np.ndarray, N: int) -> np.ndarray:
    """The D entries of a flattened parameter vector (D is last in PARAM_SPEC)."""
    return vec[-N:]


class TrainingHistory:
    """Accumulates (step, member) records and renders them as dense numpy arrays.

    every        : step stride. Must be >= 1; `train.history_every <= 0` means "no history"
                   and the caller simply does not construct a recorder.
    total_steps  : the Adam budget, so the last training step and the final evaluation step
                   can be recorded unconditionally.
    n_members    : n_restarts — the member axis length, fixed up front so a member that dies
                   at step 3 still has a lane (of NaN) rather than shifting the others.
    N            : species count, which fixes the parameter column names.
    """

    def __init__(self, *, every: int, total_steps: int, n_members: int, N: int):
        if int(every) < 1:
            raise ValueError(f"history every must be >= 1; got {every!r}. To disable the "
                             f"recorder, do not construct one (train.history_every <= 0).")
        self.every = int(every)
        self.total_steps = int(total_steps)
        self.n_members = int(n_members)
        self.N = int(N)
        self.param_names = param_names(self.N)
        self._scalar_names = None
        self._rows: dict = {}            # (step, member) -> (scalar vec, param vec)
        self._deaths: dict = {}          # member -> step at which it was abandoned

    # ---- cadence --------------------------------------------------------------------
    def should_record(self, step: int) -> bool:
        """The thinning rule: stride `every`, plus the endpoints unconditionally.

        `step >= total_steps - 1` covers BOTH the last training step and the post-training
        evaluation pass (called with step=total_steps), so the final parameters — the ones
        that actually get checkpointed — are always in the trace.
        """
        return step == 0 or step >= self.total_steps - 1 or step % self.every == 0

    # ---- recording ------------------------------------------------------------------
    def _names(self, scalars: dict) -> list:
        """Freeze the scalar column order on the first record; fail loud if it changes."""
        if self._scalar_names is None:
            self._scalar_names = sorted(scalars)
        elif sorted(scalars) != self._scalar_names:
            raise ValueError(
                f"training-history scalar keys changed mid-run: first record had "
                f"{self._scalar_names}, this one has {sorted(scalars)}. A trace whose columns "
                f"shift cannot be rendered as one array.")
        return self._scalar_names

    def _add(self, step: int, member: int, scalars: dict, params: np.ndarray):
        names = self._names(scalars)
        vec = np.array([float(scalars[k]) for k in names], dtype=float)
        self._rows[(int(step), int(member))] = (vec, np.asarray(params, dtype=float))

    @staticmethod
    def _scalars_from(parts: dict, member=None, d_ratio=float("nan")) -> dict:
        """The scalar columns of one member: diagnostics, per-term losses, live weights.

        `member` slices (B,) batched entries; None reads the serial floats. Every `L_*` key
        `losses.total` wrote is picked up by name, so a term added later appears without a
        change here — `L_param_prior` did exactly that when the biological prior was finally
        wired into compute_terms (docs/DECISIONS.md D-EVID-5). This docstring used to say
        param_prior was "not yet wired into compute_terms"; that is no longer true.
        """
        def scalar(v):
            if hasattr(v, "shape") and getattr(v, "ndim", 0) == 1 and member is not None:
                v = v[member]
            return float(v)

        out = {}
        for k in DIAG_KEYS:
            out[k] = scalar(parts[k]) if k in parts else float("nan")
        for k, v in parts.items():
            if k.startswith("L_"):
                out[k] = scalar(v)
        conv = parts.get("ss_converged")
        if conv is None:
            out["ss_converged"] = float("nan")
        elif hasattr(conv, "shape") and getattr(conv, "ndim", 0) == 1 and member is not None:
            out["ss_converged"] = float(bool(conv[member]))
        else:
            out["ss_converged"] = float(bool(conv))
        for k, w in (parts.get("weights_used") or {}).items():
            out[f"w_{k}"] = float(w)
        out["d_ratio"] = float(d_ratio)
        return out

    def record_serial(self, step: int, member: int, parts: dict, model):
        """One serial restart's state at `step`. Call BEFORE `opt.step()` so the recorded
        parameters are the ones that produced the recorded loss."""
        import torch
        with torch.no_grad():
            pvec = _param_vector(model)
        self._add(step, member, self._scalars_from(
            parts, member=None, d_ratio=d_ratio_of(_d_slice(pvec, self.N))), pvec)

    def record_batched(self, step: int, parts: dict, bmodel, alive=None):
        """All B members' state at `step`. `alive` (a (B,) bool tensor/array) skips members
        already abandoned, so a dead lane stays NaN instead of recording frozen parameters
        that are no longer being optimised."""
        import torch
        with torch.no_grad():
            block = _param_block(bmodel)
        live = None if alive is None else np.asarray(
            alive.detach().cpu().numpy() if hasattr(alive, "detach") else alive, dtype=bool)
        for b in range(bmodel.B):
            if live is not None and not bool(live[b]):
                continue
            pvec = block[b]
            self._add(step, b, self._scalars_from(
                parts, member=b, d_ratio=d_ratio_of(_d_slice(pvec, self.N))), pvec)

    def record_death(self, member: int, step: int):
        """A member whose steady state diverged and was abandoned at `step`."""
        self._deaths[int(member)] = int(step)

    # ---- rendering ------------------------------------------------------------------
    def to_arrays(self) -> dict:
        """Dense arrays for `plotdata.save_plot_arrays`, or {} when nothing was recorded.

        `hist_scalars` / `hist_params` are (S, B, Q) / (S, B, P) with NaN in every cell no
        record exists for — a member that died at step 30 of 400 is a lane that stops, which
        is the honest rendering of what happened.
        """
        if not self._rows:
            return {}
        steps = sorted({s for s, _ in self._rows})
        S, B = len(steps), self.n_members
        Q, P = len(self._scalar_names), len(self.param_names)
        si = {s: i for i, s in enumerate(steps)}
        scal = np.full((S, B, Q), np.nan)
        parm = np.full((S, B, P), np.nan)
        for (s, b), (svec, pvec) in self._rows.items():
            scal[si[s], b, :] = svec
            parm[si[s], b, :] = pvec
        deaths = np.full(B, np.nan)
        for b, s in self._deaths.items():
            deaths[b] = float(s)
        return {
            "hist_step": np.asarray(steps, dtype=np.int64),
            "hist_member": np.arange(B, dtype=np.int64),
            "hist_scalar_names": np.array(self._scalar_names),
            "hist_scalars": scal.astype(HIST_DTYPE),
            "hist_param_names": np.array(self.param_names),
            "hist_params": parm.astype(HIST_DTYPE),
            "hist_death_step": deaths.astype(np.float64),
        }

    def meta(self, best_member=None) -> dict:
        """History provenance for the npz `meta` blob."""
        steps = sorted({s for s, _ in self._rows})
        return dict(history_every=self.every,
                    history_total_steps=self.total_steps,
                    history_n_steps=len(steps),
                    history_n_members=self.n_members,
                    history_best_member=(None if best_member is None else int(best_member)),
                    history_dtype=str(np.dtype(HIST_DTYPE)))
