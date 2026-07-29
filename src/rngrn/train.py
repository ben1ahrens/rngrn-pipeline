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

from .config import Config
from .data import gate
from .data.rd_models import build_system
from .data.spec import spec_from_config
from .data import cache
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
                       seed=cfg.train.seed, model_seed=cfg.model.seed,
                       dispersion_backend=cfg.model.dispersion_backend, verbose=verbose)

    # Scoring uses the answer key; recovery did not. `ri.frame` is passed as target_frame
    # so MORPHOLOGY — the owner's primary criterion — is recorded on every run. That is
    # free: it is the image recovery already trained on, and it is on the recovery side of
    # the firewall, so handing it to scoring adds no truth quantity to anything.
    # model_frame is deliberately NOT supplied here: a field simulated from the recovered
    # model costs a rollout (measured ~4.2 ms/step at 96x96, with the step count derived
    # from the model's own sigma_max — ~128k steps, i.e. ~9 min, for an untrained N=3
    # model), which fit() must not silently add to every run. Callers that want the full comparison run
    # the rollout themselves and re-score, or use eval/. Runs therefore record
    # morphology_scored="target_only".
    metric = score_recovery(result, answer_key,
                            observed_idx=(cfg.model.observed_idx or list(range(cfg.model.m))),
                            target_frame=ri.frame)

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

    IO.save_checkpoint(rdir, result.model, extra=dict(kstar_obs=result.kstar_obs))
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
    IO.append_run_index(runs_root, row, backend=cfg.tracking.index_backend)
    metric["run_id"] = run_id
    metric["loss"] = result.loss
    return metric
