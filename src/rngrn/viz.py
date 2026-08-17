"""viz.py — figures drawn from a `TrainingHistory.to_arrays()` dict.

Top-level, outside the firewall (`CLAUDE.md` §5): imports no `data/`, no `scoring/`, no
answer key, and reads exactly the dict shape `history.py::TrainingHistory.to_arrays()`
produces — never a run directory, never ground truth. Its import surface is deliberately
narrow: numpy, matplotlib, and `history.DIAG_KEYS` for the two scalar-column names it plots
by name.

An empty or legacy-only dict (missing the Task-9 `events`/`invariant_*` keys, or `{}` for a
history that recorded nothing) draws a figure with an "empty" annotation rather than
raising — that is the expected shape of an early or population-events-less run. A dict whose
*present* keys have inconsistent shapes (e.g. `hist_scalars` not matching
`hist_scalar_names`) raises, per the house fail-loud rule (`CLAUDE.md` §4): that is a
malformed dict, not a legitimate empty state.
"""
from __future__ import annotations

import matplotlib

if matplotlib.get_backend().lower() not in ("agg",):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from .history import DIAG_KEYS                                     # noqa: E402

TOTAL_LOSS_KEY = DIAG_KEYS[0]        # "total"     -- losses/total.py's combined loss
SIG_MAX_POS_KEY = DIAG_KEYS[2]       # "sig_max_pos" -- losses.spectral.is_ignited's diagnostic

_MAX_LEGEND_MEMBERS = 12             # more than this and a per-member legend is just noise


def _empty_axis(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig, out_png: str) -> str:
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def _scalar_column(arrays: dict, key: str):
    """`(step, values)` for one named `hist_scalars` column, or `(None, None)` if the
    history is empty or never recorded that column — the legitimate-empty case."""
    step = arrays.get("hist_step")
    names = arrays.get("hist_scalar_names")
    scal = arrays.get("hist_scalars")
    if step is None or names is None or scal is None or len(step) == 0:
        return None, None
    names = list(names)
    if key not in names:
        return None, None
    return step, scal[:, :, names.index(key)]


def loss_curves(arrays: dict, out_png: str) -> str:
    """One line per member: the total loss (`history.DIAG_KEYS[0]`) vs recorded step."""
    fig, ax = plt.subplots()
    step, values = _scalar_column(arrays, TOTAL_LOSS_KEY)
    if step is None:
        _empty_axis(ax, "no loss history")
    else:
        B = values.shape[1]
        for b in range(B):
            ax.plot(step, values[:, b], label=f"member {b}")
        ax.set_xlabel("step")
        ax.set_ylabel(TOTAL_LOSS_KEY)
        if B <= _MAX_LEGEND_MEMBERS:
            ax.legend(fontsize="small")
    ax.set_title("loss curves")
    return _save(fig, out_png)


def invariant_trajectories(arrays: dict, out_png: str) -> str:
    """One stacked subplot per named invariant (spec §3.4), one line per member."""
    inv_step = arrays.get("invariant_step")
    inv_names = arrays.get("invariant_names")
    inv = arrays.get("invariants")
    if (inv_step is None or inv_names is None or inv is None
            or len(inv_step) == 0 or len(inv_names) == 0):
        fig, ax = plt.subplots()
        _empty_axis(ax, "no invariants recorded")
        ax.set_title("invariant trajectories")
        return _save(fig, out_png)

    inv_names = list(inv_names)
    Q = len(inv_names)
    B = inv.shape[1]
    fig, axes = plt.subplots(Q, 1, sharex=True, squeeze=False, figsize=(6.0, 2.2 * Q))
    for q, name in enumerate(inv_names):
        ax = axes[q, 0]
        for b in range(B):
            ax.plot(inv_step, inv[:, b, q], label=(f"member {b}" if q == 0 else None))
        ax.set_ylabel(name)
    axes[-1, 0].set_xlabel("step")
    if B <= _MAX_LEGEND_MEMBERS:
        axes[0, 0].legend(fontsize="small")
    fig.suptitle("invariant trajectories")
    return _save(fig, out_png)


def event_timeline(arrays: dict, out_png: str) -> str:
    """A scatter of population-management events (`history.EVENT_KINDS`): step on x,
    member on y, one marker style per kind."""
    fig, ax = plt.subplots()
    events = arrays.get("events")
    if events is None or len(events) == 0:
        _empty_axis(ax, "no events recorded")
    else:
        markers = "o^svPXD*"
        kinds = sorted(set(events["kind"].tolist()))
        for i, kind in enumerate(kinds):
            mask = events["kind"] == kind
            ax.scatter(events["step"][mask], events["member"][mask],
                       label=kind, marker=markers[i % len(markers)])
        ax.set_xlabel("step")
        ax.set_ylabel("member")
        ax.legend(fontsize="small")
    ax.set_title("population events")
    return _save(fig, out_png)


def spectral_trace(arrays: dict, floor: float, out_png: str) -> str:
    """Per-member `sig_max_pos` (`history.DIAG_KEYS[2]`, the growth-rate diagnostic
    `losses.spectral.is_ignited` compares against its ignition margin) vs step, with
    `floor` drawn as a horizontal reference line."""
    fig, ax = plt.subplots()
    step, values = _scalar_column(arrays, SIG_MAX_POS_KEY)
    if step is None:
        _empty_axis(ax, "no spectral diagnostic recorded")
    else:
        B = values.shape[1]
        for b in range(B):
            ax.plot(step, values[:, b], label=f"member {b}")
        ax.axhline(float(floor), color="k", linestyle="--", label="floor")
        ax.set_xlabel("step")
        ax.set_ylabel(SIG_MAX_POS_KEY)
        if B <= _MAX_LEGEND_MEMBERS:
            ax.legend(fontsize="small")
    ax.set_title("spectral trace")
    return _save(fig, out_png)
