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
from .history import TrainingHistory
from . import plotdata as PD
from .scoring.morphology import FEATURE_ORDER
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


def _best_restart(restarts) -> int | None:
    """Index of the restart/member recover() reported as the winner: lowest finite total.

    Recomputed here from the restart log rather than returned by recover(), which already
    reports the winner's MODEL and not its index. It is needed so the training trajectory can
    say which lane became the run's answer — on the batched path all B lanes are recorded and
    the winner is only known after the last step.
    """
    best, best_i = None, None
    for e in restarts:
        t = e.get("total")
        if t is None or not np.isfinite(t):
            continue
        if best is None or t < best:
            best, best_i = t, int(e["restart"])
    return best_i


def _save_run_arrays(cfg: Config, rdir: str, run_id: str, ri, result, J_rec,
                     model_frame, rollout_row, answer_key, metric, hist) -> int:
    """Write <run_dir>/arrays/plot_arrays.npz. Returns its size in bytes.

    SCORING SIDE, deliberately. The answer key's (J, D) are read here so
    `dispersion_sigma_true` can be written on the same k grid as the recovered curve —
    recovered-vs-true sigma(k) is the canonical Turing figure and was previously impossible
    to draw. Recovery finished long before this line; nothing here reaches back into it.
    """
    morph_tgt = [metric[f"morphology_{k}_target"] for k in FEATURE_ORDER]
    morph_mdl = ([metric[f"morphology_{k}_model"] for k in FEATURE_ORDER]
                 if metric.get("morphology_scored") == "compared" else None)
    best_member = _best_restart(result.restarts)
    meta = dict(run_id=run_id, git_sha=provenance()["git_revision"],
                config_id=cfg.config_id(), source=cfg.data.source,
                dataset_id=cfg.data.dataset_id, sample_key=cfg.data.sample_key,
                form=cfg.model.form, N=cfg.model.N, m=cfg.model.m,
                seed=int(cfg.train.seed), arm=metric.get("arm"),
                kstar_model=float(result.kstar_model), kstar_obs=float(result.kstar_obs),
                kstar_true=metric.get("kstar_true"),
                kstar_fft_true=metric.get("kstar_fft_true"),
                loss=float(result.loss))
    if hist is not None:
        meta.update(hist.meta(best_member=best_member))
    arrays = PD.build_plot_arrays(
        target_frame=ri.frame, L=ri.L, J_recovered=J_rec, D_recovered=result.D_phys,
        kstar_obs=result.kstar_obs, morph_vector_target=morph_tgt,
        model_field=model_frame, rollout_status=rollout_row.get("rollout_status"),
        morph_vector_model=morph_mdl,
        J_true=getattr(answer_key, "J", None), D_true=getattr(answer_key, "D", None),
        meta=meta)
    if hist is not None:
        arrays.update(hist.to_arrays())
    return PD.save_plot_arrays(PD.plot_arrays_path(rdir), arrays)


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
    # The plottable training trajectory (unit P1). Constructed here, before recovery, and read
    # back afterwards — recover() takes it as an argument rather than returning it, so the
    # trace survives even a recovery that raises. adam_steps == 0 (init-only / determinism
    # checks) has no trajectory to record.
    hist = (TrainingHistory(every=cfg.train.history_every, total_steps=cfg.train.adam_steps,
                            n_members=cfg.train.n_restarts, N=cfg.model.N)
            if (cfg.train.history_every > 0 and cfg.train.adam_steps > 0) else None)
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
                       d_init_from_kstar=cfg.model.d_init_from_kstar,   # unit B4
                       # unit C1: `model.init` round-tripped into frozen_config.yaml and was
                       # written onto the run-index `model_init` column, but was NEVER handed
                       # to recover() -- so `-o model.init=low_basal` was a silent NO-OP and
                       # the index column asserted an init the run had not used. Threading it
                       # is byte-identical on the default path (init="default").
                       init=cfg.model.init,
                       batched=cfg.train.batched,                        # unit b2
                       device=cfg.train.device,                          # unit b2
                       # unit 5: the biological prior's own knobs. These were already
                       # recorded on every run-index row (dratio_centre/dratio_spread) but
                       # were never handed to recovery, so the prior could not act.
                       dratio_centre=cfg.loss.dratio_centre,
                       dratio_spread=cfg.loss.dratio_spread,
                       bio_box_path=cfg.loss.bio_box_path,
                       history=hist)                                     # unit P1

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
    # THE RECOVERED NETWORK, AS PLAIN TEXT. Until now the recovered parameters existed only
    # inside checkpoints/model.pt — a binary blob that needs a matching torch to read. The
    # recovered J and D *are* the result of this project, so they belong in a diffable,
    # human-readable record that survives a torch upgrade and can be cited directly. These
    # are the CONSTRAINED (physical) parameters, not raw theta, because those are the
    # quantities every claim is expressed in.
    recovered = {k: (v.tolist() if hasattr(v, "tolist") else v)
                 for k, v in (result.params or {}).items()}
    recovered["x_star"] = (result.xstar.tolist() if hasattr(result.xstar, "tolist")
                           else result.xstar)
    # The JACOBIAN is the recovered gene regulatory network — the object this whole project
    # exists to recover, and the thing every topology and sign-structure claim is computed
    # from. It is derivable from the parameters above, but "derivable" is not "recorded":
    # reconstructing it needs a matching torch and the right x*. Written out explicitly.
    import torch as _torch
    J_rec = result.model.jacobian(
        _torch.as_tensor(result.xstar), create_graph=False).detach().cpu().numpy()
    recovered["J"] = J_rec.tolist()
    # THE PLOTTABLE ARRAYS (unit P1). Everything above this line is a scalar or a number as
    # text; none of it can draw a pattern, a dispersion relation, a spectrum or a learning
    # curve. `plot_arrays_bytes` goes on the run row so the cost of keeping them is itself
    # auditable rather than a surprise in `du`.
    plot_arrays_bytes = None
    if cfg.solver.save_plot_arrays:
        plot_arrays_bytes = _save_run_arrays(cfg, rdir, run_id, ri, result, J_rec,
                                             model_frame, rollout_row, answer_key, metric,
                                             hist)
    IO.save_results(rdir, "train_results.json",
                    dict(loss=result.loss, kstar_model=result.kstar_model,
                         kstar_obs=result.kstar_obs, restarts=result.restarts,
                         recovered=recovered,
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
    # unit B3: how many restarts this run LOST to a diverged steady state. Before B3 this was
    # 31/32 for nc1 at adam_steps=2000 and INVISIBLE in the index — a row could report a
    # 4-restart run that in truth kept one. Recorded so the multistart fix stays auditable
    # and any regression shows up as a non-zero count, not as a quietly worse loss.
    row.update(
        n_restarts_run=len(result.restarts),
        n_restarts_ss_failed=sum(1 for e in result.restarts if e.get("steady_state_failed")),
    )
    # EXECUTION PATH identity. The frozen config records these, but the run index is what
    # gets aggregated and compared, and without them a row cannot say whether it came from
    # the serial or the batched optimiser, on CPU or CUDA, through which dispersion
    # backend, or with which init. Those are not cosmetic: batched and serial are the same
    # METHOD but not the same random draw (float associativity diverges them over a long
    # run), the cubic backend is exact only for N<=3, and d_init_from_kstar / low_basal
    # change where recovery starts. Pooling rows that differ on any of them would be
    # comparing different experiments.
    row.update(
        batched=bool(cfg.train.batched), device=str(cfg.train.device),
        dispersion_backend=str(cfg.model.dispersion_backend),
        model_init=str(cfg.model.init), nondim=bool(cfg.model.nondim),
        d_init_from_kstar=bool(cfg.model.d_init_from_kstar),
        deterministic=bool(cfg.train.deterministic),
        adam_steps=int(cfg.train.adam_steps), n_restarts_requested=int(cfg.train.n_restarts),
    )
    # THE REMAINING TUNING AXES (unit P1). Several phase-C sweeps are over exactly these, and
    # a hyperparameter-vs-outcome plot can only be drawn from columns that are ON the row:
    # frozen_config.yaml has them, but the index is what gets aggregated. `seed` in particular
    # was absent, which made a seed-replicate sweep — the project's standard design — impossible
    # to disaggregate from the index alone. `git_sha` joins a row to the code that produced it.
    row.update(
        # model_seed is the EFFECTIVE value recovery used, not the config's. cfg.model.seed
        # is None by default meaning 'derive from train.seed', so recording the raw config
        # value would put None on every row and hide which inits a run actually drew.
        seed=int(cfg.train.seed),
        model_seed=int(cfg.train.seed if cfg.model.seed is None else cfg.model.seed),
        model_seed_explicit=bool(cfg.model.seed is not None),
        adam_lr=float(cfg.train.adam_lr), lbfgs_steps=int(cfg.train.lbfgs_steps),
        grad_clip=float(cfg.train.grad_clip), tau=float(cfg.loss.tau),
        jac_floor=float(cfg.loss.jac_floor), dratio_centre=float(cfg.loss.dratio_centre),
        dratio_spread=float(cfg.loss.dratio_spread),
        ratio_update_every=int(cfg.loss.ratio_update_every),
        w_kstar=float(cfg.loss.weights.get("kstar", 0.0)),
        w_turing=float(cfg.loss.weights.get("turing", 0.0)),
        w_anticollapse=float(cfg.loss.weights.get("anticollapse", 0.0)),
        w_morphology=float(cfg.loss.weights.get("morphology", 0.0)),
        w_param_prior=float(cfg.loss.weights.get("param_prior", 0.0)),
        n_grid=int(cfg.solver.n_grid), morphology_rollout=bool(cfg.solver.morphology_rollout),
        history_every=int(cfg.train.history_every),
        git_sha=provenance()["git_revision"],
        plot_arrays_bytes=plot_arrays_bytes,
    )
    IO.append_run_index(runs_root, row, backend=cfg.tracking.index_backend)
    metric["run_id"] = run_id
    metric["loss"] = result.loss
    return metric
