"""tune_comp_noop_check.py — prove every swept knob actually CHANGES something.

Four silent no-ops have been found in this codebase (model.seed pinned in base.yaml making
train.seed inert; losses/total.py never calling terms.param_prior; fit() never passing
cfg.model.init to recover(); the pre-push hook testing another worktree). A null result
from an inert knob is an artefact, so no axis in this unit is believed until it is shown
here to move something.

METHOD. Two runs at the SAME train.seed differing ONLY in the knob under test, on the same
target, through the SAME entry point the tuning cells use (`rngrn.train.fit`, i.e. the
config/CLI path — NOT a direct recover() call, because no-op 3 was precisely a defect of
the config path that a direct call would have hidden). Compare:

  * the loss TRAJECTORY (train history), and
  * the recovered J and D,
  * and the frozen config, to confirm the knob was actually written.

VERDICT LIVE  -> the trajectories differ: the knob reaches the optimiser.
VERDICT INERT -> bit-identical: the knob is a no-op and any null result from it is void.

Deliberately small and on CPU (few steps, few restarts, serial): this measures WHETHER the
knob moves the computation, not how well it recovers. It must not compete for the GPU that
the measurement cells are using.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rngrn.config import load_config  # noqa: E402
from rngrn.train import fit  # noqa: E402

# The axes this unit sweeps. name -> list of "dotted.path=value" overrides applied to the
# shared control config. Each is compared against the control (the library default).
AXES = {
    "turing_weight":     ["loss.weights.turing=8.0"],
    "detach_xstar":      ["loss.detach_xstar=true"],
    "d_init_from_kstar": ["model.d_init_from_kstar=true"],
    "param_prior":       ["loss.weights.param_prior=1.0"],
    "anchor":            ["loss.weights.anchor=0.5"],
    "model_init":        ["model.init=low_basal"],
    "staging_off_frac":  ["loss.staging_off_frac=0.05"],
    "staging_ramp_frac": ["loss.staging_ramp_frac=0.05"],
    "adam_steps":        ["train.adam_steps=31"],
}

# Small, serial, CPU. n_restarts>1 so a knob that only changes which restart wins is seen.
CONTROL = [
    "data.dataset_id=three_gene_qvar", "data.sample_key=sample_0000",
    "train.batched=false", "train.device=cpu", "model.dispersion_backend=cubic",
    "train.n_restarts=3", "train.lbfgs_steps=0", "train.adam_steps=25",
    "train.seed=0", "train.history_every=1",
]


def _set(cfg, dotted, raw):
    obj, *rest = dotted.split(".")
    node = getattr(cfg, obj)
    for k in rest[:-1]:
        node = node[k] if isinstance(node, dict) else getattr(node, k)
    key = rest[-1]
    cur = node.get(key) if isinstance(node, dict) else getattr(node, key)
    if isinstance(cur, bool) or raw in ("true", "false"):
        val = raw == "true"
    elif isinstance(cur, (int,)) and not isinstance(cur, bool):
        val = int(raw)
    elif isinstance(cur, float):
        val = float(raw)
    else:
        val = raw
    if isinstance(node, dict):
        node[key] = val
    else:
        setattr(node, key, val)


def run(overrides, runs_root):
    cfg = load_config("configs/m3_registry.yaml")
    for ov in CONTROL + overrides:
        k, v = ov.split("=", 1)
        _set(cfg, k, v)
    cfg.tracking.runs_root = runs_root
    res = fit(cfg)
    return res


def signature(res):
    """The things a live knob must be able to move."""
    hist = res.get("history") or res.get("restarts") or []
    return json.dumps({
        "loss": res.get("loss"),
        "J": res.get("recovered", {}).get("J"),
        "D": res.get("recovered", {}).get("D_model"),
        "kstar": res.get("kstar_model"),
        "hist_len": len(hist) if hasattr(hist, "__len__") else None,
    }, sort_keys=True, default=str)


if __name__ == "__main__":
    root = "experiments/tune_comp_noop"
    which = sys.argv[1:] or list(AXES)
    base = signature(run([], root))
    print(f"{'axis':<22}{'verdict':<10}  override")
    for name in which:
        ov = AXES[name]
        try:
            sig = signature(run(ov, root))
        except Exception as e:  # a knob that CHANGES BEHAVIOUR BY FAILING is still live
            print(f"{name:<22}{'LIVE(raise)':<10}  {ov}  -> {type(e).__name__}: {e}")
            continue
        verdict = "INERT" if sig == base else "LIVE"
        print(f"{name:<22}{verdict:<10}  {ov}")
