"""cli.py — the thin command-line surface.

Subcommands:
  generate-data  resolve a config's dataset spec -> content-addressed cache (idempotent)
  train          run fit() for one config
  evaluate       lift-and-simulate a saved run's model, report patterning + k*
  analyze        linear stability + topology + robustness cloud for a saved run
  sweep          run a sweep from a sweep YAML (base config + axes + seeds)
  benchmark      aggregate the run index into a comparison table (markdown/CSV)

Every subcommand takes --config (a YAML) and optional dotted overrides (-o key=val).
"""
from __future__ import annotations
import argparse
import json
import os

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


def cmd_evaluate(args):
    from .eval import simulate
    model, payload = IO.load_checkpoint(os.path.join(args.runs_root, "runs", args.run_id))
    cfg = _load(args)
    res = simulate(model, L=cfg.data.L, n=cfg.solver.n_grid,
                   integrator=cfg.solver.integrator,
                   horizon_growth_times=cfg.solver.horizon_growth_times,
                   noise=cfg.solver.noise)
    res.pop("fields", None)
    print(json.dumps(res, indent=2, default=str))


def cmd_analyze(args):
    from .eval.analysis import linear_stability, robustness_cloud
    from .losses.terms import steady_state
    model, payload = IO.load_checkpoint(os.path.join(args.runs_root, "runs", args.run_id))
    cfg = _load(args)
    xs, _ = steady_state(model)
    lin = linear_stability(model, xs.detach().numpy())
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
    from .optim.benchmark import build_table, to_markdown, COLUMNS
    table = build_table(runs_root=args.runs_root, backend=args.index_backend)
    if args.format == "markdown":
        print(to_markdown(table))
    else:
        import csv, sys
        w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        w.writeheader()
        for row in table:
            w.writerow({c: row.get(c) for c in COLUMNS})


def build_parser():
    p = argparse.ArgumentParser("rngrn")
    p.add_argument("--runs-root", default="experiments")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_cfg(sp):
        sp.add_argument("--config", required=True)
        sp.add_argument("-o", "--override", nargs="*", default=[])

    sp = sub.add_parser("generate-data"); add_cfg(sp); sp.add_argument("--overwrite", action="store_true"); sp.set_defaults(func=cmd_generate_data)
    sp = sub.add_parser("train"); add_cfg(sp); sp.set_defaults(func=cmd_train)
    sp = sub.add_parser("evaluate"); add_cfg(sp); sp.add_argument("--run-id", required=True); sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("analyze"); add_cfg(sp); sp.add_argument("--run-id", required=True); sp.set_defaults(func=cmd_analyze)
    sp = sub.add_parser("sweep"); sp.add_argument("--sweep", required=True); sp.set_defaults(func=cmd_sweep)
    sp = sub.add_parser("benchmark"); sp.add_argument("--format", choices=["markdown", "csv"], default="markdown"); sp.add_argument("--index-backend", choices=["jsonl", "sqlite"], default="jsonl"); sp.set_defaults(func=cmd_benchmark)
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
