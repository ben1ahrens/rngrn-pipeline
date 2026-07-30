"""cli.py — the thin command-line surface.

Subcommands:
  generate-data  resolve a config's dataset spec -> content-addressed cache (idempotent)
  train          run fit() for one config
  evaluate       lift-and-simulate a saved run's model at the CHECKPOINT's own L, report
                 patterning + k*; with --eval-L/--l-factors, evaluate across domain sizes
  analyze        linear stability + topology + robustness cloud for a saved run
  sweep          run a sweep from a sweep YAML (base config + axes + seeds)
  benchmark      aggregate the run index into a comparison table (markdown/CSV)
  target-report  run K seeds on one target (dataset_id + sample_key + form), emit ONE
                 auditable reproducibility/robustness/pattern/viability report
  export         the indexes as TIDY/long CSV (one observation per row), for plotting

Every subcommand takes --config (a YAML) and optional dotted overrides (-o key=val).
"""
from __future__ import annotations
import argparse
import json
import os
import warnings

from .config import load_config, apply_overrides
from . import io as IO


def _load(args):
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    return cfg


def cmd_generate_data(args):
    from .data.rd_models import build_system
    from .data.spec import spec_from_config
    from .data import cache
    cfg = _load(args)
    system = build_system(cfg.data.system, L=cfg.data.L)
    spec = spec_from_config(cfg.data, system)
    d = cache.generate(cfg.data.cache_root, spec, system, overwrite=args.overwrite)
    print(json.dumps(dict(dataset_hash=spec.hash(), path=d), indent=2))


def cmd_train(args):
    from .train import fit
    cfg = _load(args)
    metric = fit(cfg, runs_root=args.runs_root, verbose=args.verbose)
    print(json.dumps(metric, indent=2, default=str))


def _persist_lgen(args, cfg, out):
    """Write a cross-L evaluation into the run-index machinery and an npz. Prints nothing.

    THE DEFECT THIS FIXES. `evaluate_across_L` returns a rich per-L table and this command
    PRINTED it to stdout and stored nothing, so the measurement in
    docs/LGEN_TRANSFER_FIRST_RESULT.md could not be re-plotted without re-running a 9-minute
    recovery. The per-L rows and the summary now go through the SAME machinery
    optim/target_report.py uses (index.open_index on a named table), as flat scalar rows
    carrying run_id and the git sha so they join back to the run; the per-L FIELDS go to
    <run_dir>/arrays/lgen_fields.npz because the pattern at each domain size is itself a
    figure.
    """
    from .index import open_index
    from . import plotdata as PD
    from .utils import provenance
    git_sha = provenance()["git_revision"]
    per_L_rows, summary = PD.lgen_rows(out, run_id=args.run_id, git_sha=git_sha)
    backend = cfg.tracking.index_backend
    idx = open_index(args.runs_root, "lgen_eval", backend)
    for row in per_L_rows:
        idx.append(row)
    open_index(args.runs_root, "lgen_summary", backend).append(summary)
    if cfg.solver.save_plot_arrays:
        rdir = os.path.join(args.runs_root, "runs", args.run_id)
        PD.save_lgen_fields(PD.lgen_fields_path(rdir), out,
                            meta=dict(run_id=args.run_id, git_sha=git_sha,
                                      dataset_id=cfg.data.dataset_id,
                                      sample_key=cfg.data.sample_key))


def cmd_evaluate(args):
    """Re-simulate a saved run's model, at its OWN domain size by default.

    THE DEFECT THIS FIXES. This used to call `simulate(model, L=cfg.data.L, ...)`, and
    `cfg.data.L` is a GENERATOR parameter that file-backed configs deliberately do not set
    (config.DataConfig.L: "Do not add `L:` to a file-backed config expecting it to apply") —
    so for every registry config it was the base.yaml default 100.0 while the sample's real
    L, recorded in the checkpoint, was something else entirely (78.014 for one measured
    m3_registry run: a 28% error in the length scale). Re-running an archived model therefore
    reproduced nothing. The checkpoint's stored L is now the default and `cfg.data.L` is
    only a cross-check that warns.

    --eval-L / --l-factors evaluate the SAME model on DIFFERENT domain sizes via
    eval.lgen_eval.evaluate_across_L, which is the cross-domain-size generalisation test.
    Both forms are mutually exclusive; --l-factors are multiples of the checkpoint's L.
    """
    from .eval import simulate
    from .eval.lgen_eval import evaluate_across_L, physical_model_from_checkpoint
    model, payload = IO.load_checkpoint(os.path.join(args.runs_root, "runs", args.run_id))
    cfg = _load(args)
    # The checkpointed theta_D is in the units recovery ran in; on the non-dimensional path
    # that is D/L**2. Converting here is not optional — simulating the raw reloaded model
    # would integrate the wrong diffusivity silently.
    model, L_train = physical_model_from_checkpoint(model, payload)
    if abs(float(cfg.data.L) - L_train) > 1e-6 * max(abs(L_train), 1.0):
        warnings.warn(
            f"config data.L={cfg.data.L!r} disagrees with the checkpoint's stored "
            f"L={L_train!r}; USING THE CHECKPOINT'S. data.L is a generator parameter and is "
            f"the base.yaml default for file-backed configs, so it is not the domain size "
            f"this run was recovered on.", RuntimeWarning, stacklevel=2)

    if args.eval_L and args.l_factors:
        raise ValueError("pass either --eval-L (absolute domain sizes) or --l-factors "
                         "(multiples of the checkpoint's L), not both")
    L_values = args.eval_L or ([f * L_train for f in args.l_factors]
                               if args.l_factors else None)
    if L_values is not None:
        out = evaluate_across_L(model, L_train, L_values, n_grid=cfg.solver.n_grid,
                                seed=cfg.train.seed, integrator=cfg.solver.integrator,
                                horizon_growth_times=cfg.solver.horizon_growth_times,
                                noise=cfg.solver.noise,
                                keep_fields=cfg.solver.save_plot_arrays)
        _persist_lgen(args, cfg, out)
        # the fields are numpy arrays and belong in the npz, not in stdout
        printable = dict(out, per_L=[{k: v for k, v in r.items() if k != "field"}
                                     for r in out["per_L"]])
        print(json.dumps(printable, indent=2, default=str))
        return

    res = simulate(model, L=L_train, n=cfg.solver.n_grid,
                   integrator=cfg.solver.integrator,
                   horizon_growth_times=cfg.solver.horizon_growth_times,
                   noise=cfg.solver.noise)
    res.pop("fields", None)
    res["L"] = L_train
    print(json.dumps(res, indent=2, default=str))


def cmd_analyze(args):
    from .eval.analysis import linear_stability, robustness_cloud
    from .losses.terms import steady_state
    model, payload = IO.load_checkpoint(os.path.join(args.runs_root, "runs", args.run_id))
    cfg = _load(args)
    xs, _ = steady_state(model)
    lin = linear_stability(model, xs.detach().cpu().numpy())
    rob = robustness_cloud(model, n_samples=cfg.solver.robustness_samples,
                           sigma_log=cfg.solver.robustness_sigma_log)
    print(json.dumps(dict(linear_stability=lin, robustness=rob), indent=2, default=str))


def cmd_sweep(args):
    import yaml
    from .optim.sweep import run_sweep
    with open(args.sweep) as fh:
        spec = yaml.safe_load(fh)
    base = load_config(spec["base_config"])
    rows = run_sweep(base, spec["axes"], seeds=tuple(spec.get("seeds", [0])),
                     runs_root=args.runs_root, name=spec.get("name", "sweep"),
                     verbose=args.verbose)
    print(json.dumps(dict(n_runs=len(rows)), indent=2))


def cmd_register_data(args):
    """Register an existing/downloaded HDF5 payload into the dataset store by name."""
    from .data import registry as reg
    prov = {"source": args.provenance} if args.provenance else {}
    man = reg.register(args.datasets_root, args.dataset_id, args.payload,
                       provenance=prov, backend=args.index_backend, overwrite=args.overwrite)
    print(json.dumps({k: man[k] for k in
                      ("dataset_id", "n_samples", "frame_shape", "splits", "checksum")},
                     indent=2, default=str))


def cmd_scan_datasets(args):
    """Index datasets you placed manually at <datasets_root>/<id>/payload.h5."""
    from .data import registry as reg
    rows = reg.scan(args.datasets_root, backend=args.index_backend,
                    observable_key=args.observable_key, refresh=args.refresh)
    if not rows:
        print(json.dumps({"found": 0, "hint":
                          f"put a payload.h5 at {args.datasets_root}/<dataset_id>/payload.h5"},
                         indent=2))
        return
    print(json.dumps(rows, indent=2, default=str))


def cmd_list_datasets(args):
    from .data import registry as reg
    rows = reg.list_datasets(args.datasets_root, backend=args.index_backend)
    print(json.dumps(rows, indent=2, default=str))


def cmd_benchmark(args):
    from .optim.benchmark import (build_table, to_markdown, COLUMNS,
                                  degradation_table, degradation_markdown,
                                  DEGRADATION_COLUMNS)
    if getattr(args, "degradation", False):
        table = degradation_table(runs_root=args.runs_root, backend=args.index_backend)
        if args.format == "markdown":
            print(degradation_markdown(table)); return
        import csv, sys
        w = csv.DictWriter(sys.stdout, fieldnames=DEGRADATION_COLUMNS); w.writeheader()
        for row in table: w.writerow({c: row.get(c) for c in DEGRADATION_COLUMNS})
        return
    table = build_table(runs_root=args.runs_root, backend=args.index_backend)
    if args.format == "markdown":
        print(to_markdown(table))
    else:
        import csv, sys
        w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        w.writeheader()
        for row in table:
            w.writerow({c: row.get(c) for c in COLUMNS})


def cmd_export(args):
    """Write the indexes out as TIDY (long) CSV — one observation per row — for plotting."""
    from .export import export_all
    counts = export_all(args.runs_root, args.out_dir, backend=args.index_backend,
                        history_members=args.history_members)
    print(json.dumps(dict(out_dir=args.out_dir, rows=counts), indent=2))


def cmd_target_report(args):
    from .optim.target_report import run_target_report
    cfg = _load(args)
    report = run_target_report(cfg, dataset_id=args.dataset_id, sample_key=args.sample_key,
                               form=args.form, seeds=args.seeds, runs_root=args.runs_root,
                               n_workers=args.workers, verbose=args.verbose)
    print(json.dumps(report, indent=2, default=str))


def build_parser():
    p = argparse.ArgumentParser("rngrn")
    p.add_argument("--runs-root", default="experiments")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_cfg(sp):
        sp.add_argument("--config", required=True)
        # action="extend" is load-bearing. With the previous nargs="*" alone, argparse
        # REPLACED args.override on every occurrence, so `-o a=1 -o b=2` silently kept only
        # b=2 and the run used the default for a — a config that differs from the one the
        # command line asked for, with nothing in the output saying so. Both grouped
        # (`-o a=1 b=2`) and repeated (`-o a=1 -o b=2`) forms now accumulate.
        sp.add_argument("-o", "--override", nargs="*", action="extend", default=[])

    sp = sub.add_parser("generate-data"); add_cfg(sp); sp.add_argument("--overwrite", action="store_true"); sp.set_defaults(func=cmd_generate_data)
    sp = sub.add_parser("train"); add_cfg(sp); sp.set_defaults(func=cmd_train)
    sp = sub.add_parser("evaluate"); add_cfg(sp); sp.add_argument("--run-id", required=True)
    sp.add_argument("--eval-L", type=float, nargs="+", default=None,
                    help="absolute domain sizes to evaluate the recovered model at "
                         "(cross-L generalisation; the checkpoint's own L is always added)")
    sp.add_argument("--l-factors", type=float, nargs="+", default=None,
                    help="the same, as MULTIPLES of the checkpoint's L, e.g. 0.5 1 2 4")
    sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("analyze"); add_cfg(sp); sp.add_argument("--run-id", required=True); sp.set_defaults(func=cmd_analyze)
    sp = sub.add_parser("sweep"); sp.add_argument("--sweep", required=True); sp.set_defaults(func=cmd_sweep)
    sp = sub.add_parser("benchmark"); sp.add_argument("--format", choices=["markdown", "csv"], default="markdown"); sp.add_argument("--index-backend", choices=["jsonl", "sqlite"], default="jsonl"); sp.add_argument("--degradation", action="store_true", help="identifiability degradation table (experiment arms)"); sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("register-data")
    sp.add_argument("--datasets-root", default="data/datasets")
    sp.add_argument("--dataset-id", required=True)
    sp.add_argument("--payload", required=True, help="path to an HDF5 payload to register")
    sp.add_argument("--provenance", default=None, help="free-form source note")
    sp.add_argument("--index-backend", choices=["jsonl", "sqlite"], default="jsonl")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_register_data)
    sp = sub.add_parser("scan-datasets", help="index datasets placed manually in datasets_root")
    sp.add_argument("--datasets-root", default="data/datasets")
    sp.add_argument("--index-backend", choices=["jsonl", "sqlite"], default="jsonl")
    sp.add_argument("--observable-key", default="final_frame")
    sp.add_argument("--refresh", action="store_true", help="rebuild existing manifests")
    sp.set_defaults(func=cmd_scan_datasets)
    sp = sub.add_parser("target-report", help="run K seeds on one target, emit one auditable report")
    add_cfg(sp)
    sp.add_argument("--dataset-id", required=True)
    sp.add_argument("--sample-key", required=True)
    sp.add_argument("--form", required=True, choices=["competitive", "nc1"])
    sp.add_argument("--seeds", type=int, nargs="+", required=True)
    sp.add_argument("--workers", type=int, default=1, help="ProcessPoolExecutor workers over seeds (1 = sequential)")
    sp.set_defaults(func=cmd_target_report)
    sp = sub.add_parser("export", help="indexes -> tidy/long CSV (one observation per row)")
    sp.add_argument("--out-dir", default="exports")
    sp.add_argument("--index-backend", choices=["jsonl", "sqlite"], default="jsonl")
    sp.add_argument("--history-members", choices=["best", "all"], default="best",
                    help="training-history rows: only the winning member (default) or every "
                         "recorded restart (much larger; the npz holds all of them either way)")
    sp.set_defaults(func=cmd_export)
    sp = sub.add_parser("list-datasets")
    sp.add_argument("--datasets-root", default="data/datasets")
    sp.add_argument("--index-backend", choices=["jsonl", "sqlite"], default="jsonl")
    sp.set_defaults(func=cmd_list_datasets)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
