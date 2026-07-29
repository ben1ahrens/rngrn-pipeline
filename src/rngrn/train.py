"""train.py — fit(): config -> RecoveryInput -> recovery -> validation metric.

fit() is the single composable unit the outer loops (optim/sweep.py) wrap. It
RETURNS the validation metric (does not merely write to disk), so a sweep is just
a loop over fit(). It resolves the frame through the firewall gate, runs recovery,
scores against the answer key (scoring only), and records a run.

The answer key enters fit() ONLY for scoring, and only AFTER recovery has finished
on the RecoveryInput. recover() itself never sees it.
"""
from __future__ import annotations
import os

import numpy as np

from .config import Config
from .data import gate
from .data.rd_models import build_system
from .data.spec import spec_from_config
from .data import cache
from .eval.rollout import simulate
from . import recover as R
from .losses.weighting import build_strategy
from . import io as IO
from .utils import seed_everything, provenance, set_deterministic
from .validate import score_recovery


def _resolve_recovery_input(cfg: Config):
    """Firewall gate: produce (RecoveryInput, AnswerKey) from the config's data source."""
    dc = cfg.data
    obs_idx = cfg.model.observed_idx or list(range(cfg.model.m))
    if dc.source == "reference":
        system = build_system(dc.system, L=dc.L)
        spec = spec_from_config(dc, system)
        cache.generate(dc.cache_root, spec, system)   # idempotent
        return gate.from_cache(dc.cache_root, spec.hash(), cfg.model.N, obs_idx)
    if dc.source == "cache":
        return gate.from_cache(dc.cache_root, dc.dataset_hash, cfg.model.N, obs_idx)
    # For file-backed sources the sample's own L is authoritative; `data.L` is passed
    # through only as an EXPLICIT cross-check when the config sets one (the gate warns and
    # prefers the file if they disagree). `L_override` is None on these configs by default,
    # which is why the hardcoded `L: 100.0` was dropped from them — see gate._resolve_L.
    if dc.source == "registry":
        return gate.from_registry(dc.datasets_root, dc.dataset_id, dc.sample_key,
                                  cfg.model.N, obs_idx, dc.L_override,
                                  backend=cfg.tracking.index_backend)
    if dc.source == "hdf5_3gene":
        return gate.from_3gene_hdf5(dc.hdf5_path, dc.sample_key, cfg.model.N, obs_idx,
                                    dc.L_override)
    raise ValueError(f"unknown data.source '{dc.source}'")


def _morphology_rollout(cfg: Config, result, ri):
    """Simulate the recovered model on the TARGET's geometry. Returns (field, row).

    `field` is the (N, H, W) simulated state, or None when there is nothing comparable —
    in which case `row` says why and scoring stays at "target_only". The row is a flat
    dict of `rollout_*` scalars that merges straight into the run index.

    THE GRID AND L COME FROM THE TARGET FRAME, not from solver.n_grid. Morphology is only
    defined between two fields on the same grid (scoring/morphology raises on a shape
    mismatch, deliberately, because the angular features live on the Fourier grid), and
    `ri.L` is the sample's own box length. Both are recovery-side inputs — the firewall
    permits exactly (frame, L, observed_idx) — so nothing new crosses it here.

    A rollout that blows up, goes non-finite, or fails to produce a pattern at all is a
    REAL RESULT about the recovered model, not an error. All three are checked explicitly
    (never caught as exceptions) and reported as `rollout_status`: morphology_vector
    legitimately raises on a field with no spatial variance, and fit() must not die on a
    model that simply failed to pattern.
    """
    row = {"rollout_status": "disabled"}
    if not cfg.solver.morphology_rollout:
        return None, row

    frame = np.asarray(ri.frame, dtype=float)
    if frame.ndim != 3 or frame.shape[1] != frame.shape[2]:
        raise ValueError(
            f"morphology rollout needs a square (m, H, W) target frame to fix the grid; "
            f"got shape {frame.shape}. Set solver.morphology_rollout=false for this source.")
    n_grid = int(frame.shape[1])

    res = simulate(result.model, L=float(ri.L), n=n_grid, seed=cfg.train.seed,
                   noise=cfg.solver.noise, xstar=result.xstar,
                   integrator=cfg.solver.morphology_integrator,
                   horizon_growth_times=cfg.solver.horizon_growth_times,
                   dt=cfg.solver.dt,
                   max_steps=cfg.solver.morphology_max_steps,
                   early_stop=cfg.solver.morphology_early_stop,
                   check_every=cfg.solver.morphology_check_every,
                   saturation_tol=cfg.solver.morphology_saturation_tol,
                   saturation_window=cfg.solver.morphology_saturation_window)
    row = dict(rollout_status="ok",
               rollout_grid=n_grid, rollout_L=float(ri.L),
               rollout_integrator=res["integrator"], rollout_dt=res["dt"],
               rollout_nsteps=res["nsteps"], rollout_nsteps_run=res["nsteps_run"],
               rollout_stopped_reason=res["stopped_reason"],
               rollout_sig_max=res["sig_max"], rollout_seconds=res["seconds"],
               rollout_amplitude=float(res["amplitude"]),
               rollout_kstar=float(res["kstar"]),
               rollout_patterned=bool(res["patterned"]))

    if res["blew_up"]:
        row["rollout_status"] = "blew_up"
        return None, row
    field = res["fields"]
    ch0 = field[0]
    if not np.isfinite(ch0).all():
        row["rollout_status"] = "non_finite"
        return None, row
    # THE GUARD THAT MATTERS, and the one an earlier draft of this function got wrong.
    # A first e2e run on three_gene_val/sample_0000 recovered a linearly stable model whose
    # field decayed to amplitude 1.9e-12 — pure float noise — and, because the guard was an
    # absolute 1e-12, that noise was classified, matched against the target's 'stripes' and
    # recorded as morphology_match=True. A morphology call on decayed noise is meaningless
    # and a match on it is a FALSE POSITIVE, which is worse than no number at all.
    #
    # The gate is therefore `patterned`, the verdict rollout.simulate already computes from
    # its own threshold (amplitude > max(1e-3, 0.02*|x*_0|)) — no new threshold is invented
    # here. An unpatterned model records rollout_patterned=False and morphology_scored stays
    # "target_only".
    #
    # DECISION LEFT OPEN, deliberately: whether an unpatterned model should instead be
    # recorded as morphology_match=False (it produces no pattern, so it certainly does not
    # produce the TARGET's pattern) is a metric definition, not a mechanical detail. It
    # changes what the headline morphology number means across a whole benchmark, so it is
    # not being decided here. rollout_patterned carries the fact either way.
    if not res["patterned"]:
        row["rollout_status"] = "unpatterned"
        return None, row
    return field, row


def fit(cfg: Config, runs_root: str = "experiments", run_id: str | None = None,
        verbose: bool = False) -> dict:
    """Run one recovery + scoring. Returns the metric dict (and writes a run row)."""
    set_deterministic(cfg.train.deterministic)
    seed_everything(cfg.train.seed)
    run_id = run_id or IO.new_run_id(cfg.tracking.run_name)
    rdir = IO.run_dir(runs_root, run_id)
    cfg.to_yaml(os.path.join(rdir, "config", "frozen_config.yaml"))

    ri, answer_key = _resolve_recovery_input(cfg)   # <-- the firewall boundary

    strategy = build_strategy(cfg.loss)
    result = R.recover(ri, form=cfg.model.form, strategy=strategy, tau=cfg.loss.tau,
                       jac_floor=cfg.loss.jac_floor, n_restarts=cfg.train.n_restarts,
                       adam_steps=cfg.train.adam_steps, adam_lr=cfg.train.adam_lr,
                       lbfgs_steps=cfg.train.lbfgs_steps, grad_clip=cfg.train.grad_clip,
                       seed=cfg.train.seed, verbose=verbose,
                       # promoted objective knobs (unit 1)
                       split_hinges=cfg.loss.split_hinges,
                       hinge_k_min_frac=cfg.loss.hinge_k_min_frac,
                       staging_keys=tuple(cfg.loss.staging_keys),
                       staging_off_frac=cfg.loss.staging_off_frac,
                       staging_ramp_frac=cfg.loss.staging_ramp_frac,
                       detach_xstar=cfg.loss.detach_xstar,
                       nondim=cfg.model.nondim,   # unit 12: default False = unchanged path
                       model_seed=cfg.model.seed,                       # unit 10
                       dispersion_backend=cfg.model.dispersion_backend,  # unit 10
                       batched=cfg.train.batched,                        # unit b2
                       device=cfg.train.device)                          # unit b2

    # Scoring uses the answer key; recovery did not. `ri.frame` is passed as target_frame
    # so MORPHOLOGY — the owner's primary criterion — is recorded on every run. That is
    # free: it is the image recovery already trained on, and it is on the recovery side of
    # the firewall, so handing it to scoring adds no truth quantity to anything.
    #
    # model_frame is now supplied too (unit 7), which is what turns morphology_scored from
    # "target_only" into "compared" and makes morphology_match / morphology_distance /
    # spectral_distance_2d real numbers rather than absent keys. It used to be omitted
    # because the rollout cost ~6.5-10 min per field; that cost was a horizon bug (see
    # eval/rollout.py) and is now 0.9-1.7 s. Set solver.morphology_rollout=false to go back
    # to target-only scoring.
    model_frame, rollout_row = _morphology_rollout(cfg, result, ri)
    # L is passed so the k* LEAK CONTROLS populate (unit 8). Without it trivial_kstar_err
    # and kstar_fft_bin_width are NaN on every row, and a k* number is then unreadable:
    # on the legacy three_gene sets an image-blind 6*2pi/L predictor scores ~0 % error, so
    # recovery's k* must always be quoted next to what ignoring the image would have got.
    # L is already a legitimate recovery input, so this adds no truth quantity to scoring.
    metric = score_recovery(result, answer_key,
                            observed_idx=(cfg.model.observed_idx or list(range(cfg.model.m))),
                            target_frame=ri.frame, model_frame=model_frame, L=ri.L)
    metric.update(rollout_row)

    # ---- experiment-arm identity (scoring/bookkeeping side) -------------------------
    # Classify this run so the benchmark can compare like with like. n_true comes from the
    # ANSWER KEY (it is truth, and is used only here for labelling/scoring — never by
    # recovery, which was already finished above).
    obs_idx_used = list(cfg.model.observed_idx or range(cfg.model.m))
    n_true = getattr(answer_key, "n_species_true", None)
    hidden_idx = [i for i in range(cfg.model.N) if i not in obs_idx_used]
    if n_true is not None and cfg.model.N > n_true:
        arm = "overparameterised"      # Experiment B: model has more species than exist
    elif n_true is not None and cfg.model.N < n_true:
        arm = "underparameterised"     # model has fewer species than exist
    elif len(hidden_idx) > 0:
        arm = "hidden_channel"         # Experiment A: right N, but some channels unobserved
    else:
        arm = "fully_observed"         # control
    metric["arm"] = arm
    metric["n_true"] = n_true
    metric["n_model"] = cfg.model.N
    metric["observed_idx"] = str(obs_idx_used)   # single source of truth for this key

    # `nondim`/`L` ride along because the checkpointed model's D is in the units recovery
    # ran in: dimensionless (D/L**2) on the non-dimensional path. Anyone reloading it for a
    # rollout must know which, or they will simulate the wrong diffusivity silently. # unit 12
    IO.save_checkpoint(rdir, result.model,
                       extra=dict(kstar_obs=result.kstar_obs,
                                  nondim=result.nondim, L=result.L))
    IO.save_results(rdir, "train_results.json",
                    dict(loss=result.loss, kstar_model=result.kstar_model,
                         kstar_obs=result.kstar_obs, restarts=result.restarts,
                         metric=metric, provenance=provenance()))
    # Record the TRUE data identity per source. `system` (a reference-kinetics name) is
    # only meaningful when we generated from one; for registry/hdf5 it is unused config
    # default noise, so we log None there and a source-appropriate label instead.
    # `system` is authoritative ONLY for source=reference (we generated from that
    # reference-kinetics config). source=cache resolves purely by dataset_hash, so its
    # config `system` field is unvalidated — label cache runs by the hash instead.
    src = cfg.data.source
    system_label = cfg.data.system if src == "reference" else None
    if src == "reference":
        dataset_label = system_label
    elif src == "cache":
        dataset_label = cfg.data.dataset_hash
    elif src == "registry":
        dataset_label = cfg.data.dataset_id
    elif src == "hdf5_3gene":
        dataset_label = f"{os.path.basename(cfg.data.hdf5_path or '')}:{cfg.data.sample_key}"
    else:
        dataset_label = None
    # Build the run-index row explicitly. The scorers return their own keys (and may echo
    # observed_idx / loss / n_true), so MERGE rather than **-expand into a dict() literal —
    # that raises TypeError on any duplicate key. Run identity wins over scorer echoes.
    row = {k: (v if isinstance(v, (int, float, bool, str)) or v is None else str(v))
           for k, v in metric.items()}          # flat scalars only, for sqlite/jsonl
    row.update(
        run_id=run_id, config_id=cfg.config_id(),
        source=src, dataset_label=dataset_label,
        dataset_hash=(cfg.data.dataset_hash or getattr(ri, "dataset_hash", None)),
        dataset_id=cfg.data.dataset_id, sample_key=cfg.data.sample_key,
        system=system_label, N=cfg.model.N, m=cfg.model.m, form=cfg.model.form,
        # experiment-arm identity: which observation regime, and how the model's assumed
        # species count relates to the truth.
        hidden_idx=str(hidden_idx), strategy=cfg.loss.strategy, loss=result.loss,
    )
    # objective identity (unit 1): which promoted terms/schedule produced this row. Without
    # these, runs.jsonl cannot distinguish two rows fitted with different objectives.
    row.update(
        split_hinges=cfg.loss.split_hinges, hinge_k_min_frac=cfg.loss.hinge_k_min_frac,
        staging_off_frac=cfg.loss.staging_off_frac,
        staging_ramp_frac=cfg.loss.staging_ramp_frac,
        staging_keys=str(list(cfg.loss.staging_keys)), detach_xstar=cfg.loss.detach_xstar,
        w_anchor=float(cfg.loss.weights.get("anchor", 0.0)),
        w_resid=float(cfg.loss.weights.get("resid", 0.0)),
    )
    IO.append_run_index(runs_root, row, backend=cfg.tracking.index_backend)
    metric["run_id"] = run_id
    metric["loss"] = result.loss
    return metric
