"""plotdata.py — the PLOTTABLE arrays of a run, persisted next to its scalars.

WHY THIS MODULE EXISTS. Until now a run recorded scalars (runs.jsonl, target_reports.jsonl),
the recovered parameters and Jacobian as text, and a checkpoint. That is enough to audit a
number and to re-run a model. It is not enough to draw a figure: you cannot plot a pattern,
a dispersion relation or a power spectrum from a scalar, and every array a paper on inverse
Turing patterns needs was being discarded the moment the run ended. This module writes those
arrays to one compressed `.npz` per run so a figure can be drawn — and redrawn — without
re-running a 9-minute recovery.

WHICH SIDE OF THE FIREWALL. This module is on the SCORING/bookkeeping side, like
validate.py: `build_plot_arrays` accepts the ANSWER KEY's (J, D) so the true dispersion
curve can be overlaid on the recovered one, which is the canonical Turing figure. It never
imports the answer-key side itself (the arrays arrive as plain numpy from the caller), and
no recovery-side module may import it — tests/test_plot_arrays.py audits that statically,
the same way tests/test_firewall.py audits the recovery side.

DTYPES, and why they are split (measured, see docs/DECISIONS.md D-PLOT-1):
  FIELDS and SPECTRA  -> float32. A field is plotted, not differentiated; float32 carries
                         ~7 significant digits, which is far beyond what a colour map or a
                         log-y spectrum can show, and it halves the file.
  k AXES and sigma(k) -> float64. These are read as NUMBERS (k* is their argmax, and a
                         dispersion curve near onset has sigma ~ 1e-3 against a J of order
                         1), so they stay at the precision they were computed in.

NOTHING HERE IS A FALLBACK. A rollout that did not pattern gets NO `model_field` key at all
(and `meta['rollout_status']` says why) rather than a decayed-noise field saved as if it
were a pattern — the project already recorded one false `morphology_match` from exactly that
mistake (see train._morphology_rollout).
"""
from __future__ import annotations
import json
import os

import numpy as np

from . import observables as OBS
from .scoring.morphology import FEATURE_ORDER

# Bumped when the KEY SET changes, so a stray npz can say which contract it satisfies.
ARRAY_SCHEMA_VERSION = 1

FIELD_DTYPE = np.float32     # fields + spectra: plotted, never differentiated
AXIS_DTYPE = np.float64      # k axes + sigma(k): read as numbers


def plot_arrays_path(run_dir: str) -> str:
    """`<run_dir>/arrays/plot_arrays.npz` — created lazily by `save_plot_arrays`."""
    return os.path.join(run_dir, "arrays", "plot_arrays.npz")


def lgen_fields_path(run_dir: str) -> str:
    """`<run_dir>/arrays/lgen_fields.npz` — the per-L fields of a cross-L evaluation."""
    return os.path.join(run_dir, "arrays", "lgen_fields.npz")


# --------------------------------------------------------------------------------------
# the dispersion relation, as numpy
# --------------------------------------------------------------------------------------
def sigma_of(J, D, kgrid) -> np.ndarray:
    """sigma(k) = max_i Re eig(J - k^2 diag(D)) on a numpy (J, D) over a k grid.

    The same quantity `eval.analysis.turing_ok` maxes over and `model.dispersion` computes
    differentiably, written out here because a FIGURE needs the whole curve rather than its
    maximum, and because this one must run on the ANSWER KEY's (J, D) — plain arrays that
    carry no torch model.
    """
    J = np.asarray(J, dtype=float)
    D = np.asarray(D, dtype=float)
    kgrid = np.asarray(kgrid, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"J must be square (N, N); got shape {J.shape}")
    if D.shape != (J.shape[0],):
        raise ValueError(f"D must be (N,) matching J's {J.shape[0]} species; got {D.shape}")
    Dm = np.diag(D)
    return np.array([np.max(np.real(np.linalg.eigvals(J - k ** 2 * Dm))) for k in kgrid],
                    dtype=float)


def plot_kgrid(kstar_obs: float) -> np.ndarray:
    """The k grid both sigma(k) curves are evaluated on: recovery's OWN grid.

    Reused verbatim from `recover._kgrid_for` rather than invented here, so a figure shows
    sigma(k) over exactly the band the objective's hinges saw — a second, differently-shaped
    grid would be a new uncalibrated choice and would not be "the k-grid used".
    """
    from .recover import _kgrid_for
    return np.asarray(_kgrid_for(float(kstar_obs)).detach().cpu().numpy(), dtype=float)


# --------------------------------------------------------------------------------------
# per-run plot arrays
# --------------------------------------------------------------------------------------
def build_plot_arrays(*, target_frame, L, J_recovered, D_recovered, kstar_obs,
                      morph_vector_target, model_field=None, rollout_status="unknown",
                      morph_vector_model=None, J_true=None, D_true=None,
                      meta=None) -> dict:
    """Assemble every array a per-run figure needs. Pure: computes, writes nothing.

    target_frame : the OBSERVED frame, (m, H, W) or (H, W). The left panel of every figure.
    L            : the sample's own domain size, in the units k is reported in (rad/length).
    J_recovered, D_recovered : the recovered model's PHYSICAL Jacobian and diffusivities
                   (RecoveryResult.params['J'-equivalent] / .D_phys — both are physical on
                   the non-dimensional path too).
    kstar_obs    : the wavenumber measured off the target, which sets the k grid.
    morph_vector_target / morph_vector_model : the 4-D morphology vectors ALREADY computed
                   by scoring (validate._morphology_metrics writes them into the run row).
                   Passed in rather than recomputed so the npz and the index row cannot
                   disagree.
    model_field  : the recovered model's rolled-out field, (N, H, W), or None. Supply it
                   ONLY when the rollout ran and PATTERNED; a non-patterned field is float
                   noise and saving it would invite the same false-positive morphology call
                   the rollout guard exists to prevent.
    rollout_status : train._morphology_rollout's verdict, recorded in `meta` so an absent
                   `model_field` says why it is absent.
    J_true, D_true : the ANSWER KEY's Jacobian and diffusivities, or None. Present ->
                   `dispersion_sigma_true` is written on the SAME k grid, which is what
                   makes the recovered-vs-true dispersion overlay drawable.
    meta         : run identity (run_id, git sha, dataset_id, sample_key, form, ...). Merged
                   with the keys computed here and stored as a JSON string.

    Returns a dict of numpy arrays ready for `save_plot_arrays`.
    """
    tgt = np.asarray(target_frame, dtype=float)
    if tgt.ndim == 2:
        tgt = tgt[None]
    if tgt.ndim != 3:
        raise ValueError(f"target_frame must be (H, W) or (m, H, W); got shape {tgt.shape}")
    L = float(L)
    if not np.isfinite(L) or L <= 0:
        raise ValueError(f"L must be finite and > 0 to put k in physical units; got {L!r}")

    out: dict = {"target_frame": tgt.astype(FIELD_DTYPE)}

    # --- the RAPS of target and model, on one shared k axis. k* IS the argmax of these, so
    #     a reader can see where the headline number came from.
    raps_k, raps_tgt, kstar_tgt = OBS.raps(tgt[0], L)
    out["raps_k"] = np.asarray(raps_k, dtype=AXIS_DTYPE)
    out["raps_target"] = np.asarray(raps_tgt, dtype=FIELD_DTYPE)

    kstar_mdl = float("nan")
    if model_field is not None:
        mdl = np.asarray(model_field, dtype=float)
        if mdl.ndim != 3:
            raise ValueError(f"model_field must be (N, H, W); got shape {mdl.shape}")
        if mdl.shape[-2:] != tgt.shape[-2:]:
            raise ValueError(
                f"model_field {mdl.shape} and target_frame {tgt.shape} are on different "
                f"grids; a side-by-side figure and a shared RAPS k axis both require the "
                f"same (H, W).")
        out["model_field"] = mdl.astype(FIELD_DTYPE)
        raps_k_m, raps_mdl, kstar_mdl = OBS.raps(mdl[0], L)
        if not np.allclose(raps_k_m, raps_k):
            raise ValueError("target and model RAPS k axes differ despite a matching grid")
        out["raps_model"] = np.asarray(raps_mdl, dtype=FIELD_DTYPE)

    # --- sigma(k): the canonical Turing figure. Recovered always; TRUE when the dataset
    #     carries an answer-key (J, D), on the same grid so the two overlay directly.
    kgrid = plot_kgrid(kstar_obs)
    out["dispersion_k"] = kgrid.astype(AXIS_DTYPE)
    sig_rec = sigma_of(J_recovered, D_recovered, kgrid)
    out["dispersion_sigma"] = sig_rec.astype(AXIS_DTYPE)
    kstar_disp_rec = float(kgrid[int(np.argmax(sig_rec))])
    kstar_disp_true = float("nan")
    if J_true is not None and D_true is not None:
        sig_true = sigma_of(J_true, D_true, kgrid)
        out["dispersion_sigma_true"] = sig_true.astype(AXIS_DTYPE)
        kstar_disp_true = float(kgrid[int(np.argmax(sig_true))])

    # --- morphology vectors, as computed by scoring.
    out["morph_feature_order"] = np.array(list(FEATURE_ORDER))
    out["morph_vector_target"] = np.asarray(morph_vector_target, dtype=AXIS_DTYPE)
    if morph_vector_model is not None:
        out["morph_vector_model"] = np.asarray(morph_vector_model, dtype=AXIS_DTYPE)

    full_meta = dict(
        schema_version=ARRAY_SCHEMA_VERSION,
        L=L,
        rollout_status=str(rollout_status),
        has_model_field=bool(model_field is not None),
        kstar_raps_target=float(kstar_tgt),
        kstar_raps_model=float(kstar_mdl),
        kstar_dispersion_recovered=kstar_disp_rec,
        kstar_dispersion_true=kstar_disp_true,
        field_dtype=str(np.dtype(FIELD_DTYPE)),
        axis_dtype=str(np.dtype(AXIS_DTYPE)),
    )
    if meta:
        full_meta.update(meta)
    out["meta"] = np.array(json.dumps(full_meta, default=str))
    return out


def save_plot_arrays(path: str, arrays: dict) -> int:
    """Write `arrays` to a compressed npz at `path`. Returns the file size in bytes."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"          # np.savez_compressed appends .npz if absent
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return os.path.getsize(path)


def load_plot_arrays(path: str) -> dict:
    """Read an npz written by `save_plot_arrays`. `meta` comes back as a dict."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    if "meta" in out:
        out["meta"] = json.loads(str(out["meta"].item()))
    return out


# --------------------------------------------------------------------------------------
# cross-L evaluation records
# --------------------------------------------------------------------------------------
def lgen_rows(out: dict, *, run_id: str, git_sha: str) -> tuple:
    """Flatten `eval.lgen_eval.evaluate_across_L`'s result into index rows.

    Returns (per_L_rows, summary_row): FLAT SCALARS only, built with `dict.update()` per the
    house rule, one per-L row per (run_id, L) plus one summary row. This is what makes the
    cross-L measurement re-plottable — `cmd_evaluate` used to print it to stdout and store
    nothing, so docs/LGEN_TRANSFER_FIRST_RESULT.md's table could not be redrawn without
    re-running a 9-minute recovery.

    `run_id` and `git_sha` ride on every row so a row joins back to the run that produced
    the model, and to the code that evaluated it.
    """
    shared = dict(run_id=run_id, git_sha=git_sha, L_train=float(out["L_train"]),
                  n_grid=int(out["n_grid"]), grid_rule=str(out["grid_rule"]),
                  ppw_floor=float(out["ppw_floor"]))

    per_L_rows = []
    for r in out["per_L"]:
        row = dict(shared)
        vec = r.get("morph_vector")
        for k, v in r.items():
            if k in ("morph_vector", "morph_feature_order", "field"):
                continue
            row[k] = v
        # the 4-D morphology vector, one flat column per feature (None when the row did not
        # pattern, so an absent class and absent features stay visibly absent)
        for i, name in enumerate(FEATURE_ORDER):
            row[f"morph_{name}"] = (None if vec is None else float(vec[i]))
        per_L_rows.append(row)

    summary = dict(shared)
    for k, v in out["summary"].items():
        if k in ("verdict_components", "L_over_L_train_values"):
            continue                       # nested / vector: recoverable from the per-L rows
        summary[k] = v
    return per_L_rows, summary


def save_lgen_fields(path: str, out: dict, *, meta=None) -> int:
    """Write the per-L FIELDS of a cross-L evaluation to one compressed npz.

    "The pattern at each domain size" is itself a figure, so the fields are persisted, not
    just the scalars. One key per L (`field_00`, `field_01`, ... in the order of `per_L`)
    because under grid_rule='constant_dx' the grids differ between L and cannot be stacked;
    `L_values` and `n_values` give the ordering and the grid of each.

    RAISES if any row has no `field` — call `evaluate_across_L(..., keep_fields=True)`.
    """
    rows = list(out["per_L"])
    arrays: dict = {}
    for i, r in enumerate(rows):
        if r.get("field") is None:
            raise ValueError(
                f"per-L row {i} (L={r.get('L')!r}) carries no 'field'; call "
                f"evaluate_across_L(..., keep_fields=True) to keep the simulated fields.")
        arrays[f"field_{i:02d}"] = np.asarray(r["field"], dtype=FIELD_DTYPE)
    arrays["L_values"] = np.array([float(r["L"]) for r in rows], dtype=AXIS_DTYPE)
    arrays["n_values"] = np.array([int(r["n"]) for r in rows], dtype=np.int64)
    arrays["kstar_phys"] = np.array([float(r["kstar_phys"]) for r in rows], dtype=AXIS_DTYPE)
    arrays["periods_per_box"] = np.array([float(r["periods_per_box"]) for r in rows],
                                         dtype=AXIS_DTYPE)
    full_meta = dict(schema_version=ARRAY_SCHEMA_VERSION,
                     L_train=float(out["L_train"]), grid_rule=str(out["grid_rule"]),
                     n_grid=int(out["n_grid"]),
                     morph_class=[r.get("morph_class") for r in rows],
                     patterned=[bool(r["patterned"]) for r in rows],
                     # WHICH row is the reference: `summarise_across_L` reads the morphology
                     # reference class off it, so a figure must be able to label it too.
                     is_L_train=[bool(r["is_L_train"]) for r in rows],
                     field_dtype=str(np.dtype(FIELD_DTYPE)))
    if meta:
        full_meta.update(meta)
    arrays["meta"] = np.array(json.dumps(full_meta, default=str))
    return save_plot_arrays(path, arrays)
