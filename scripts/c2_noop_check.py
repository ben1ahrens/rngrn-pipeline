"""c2_noop_check.py — prove every swept knob actually changed something.

FOUR silent no-ops have been found in this codebase (model.seed pinned in base.yaml making
train.seed inert; losses/total.py never calling terms.param_prior; fit() never passing
cfg.model.init to recover(); the pre-push hook testing another worktree). A null result on
an unverified knob is worth nothing, so each arm is checked against the baseline cell it
was paired with -- SAME seeds, SAME targets, one override moved.

For each (target, seed) present in both roots this reports whether the recovered J, the
final loss, and the winning restart's sig_max are bit-identical. Bit-identical across every
pair means the knob is INERT and any "no effect" reading from that arm is an artefact.
`config_id` is reported too, but a differing config_id only proves the override was PARSED,
not that it reached the optimiser -- that is exactly how no-op 3 hid. The run-level
comparison is the load-bearing one.

Usage:  python scripts/c2_noop_check.py <baseline_root> <arm_root> [arm_root ...]
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(root):
    idx = {}
    p = Path(root) / "runs.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                idx[r["run_id"]] = r
    out = {}
    for rd in sorted(Path(root).glob("runs/*")):
        f = rd / "results" / "train_results.json"
        if not f.exists() or rd.name not in idx:
            continue
        d = json.loads(f.read_text())
        r = idx[rd.name]
        out[(r["sample_key"], r["seed"])] = dict(
            J=np.array(d["recovered"]["J"], dtype=float),
            D=np.array(d["recovered"]["D"], dtype=float),
            loss=float(d["loss"]),
            sig=float(d["metric"]["recovered_sig_max"]),
            config_id=r["config_id"],
        )
    return out


def main(base_root, arm_roots):
    base = load(base_root)
    print(f"baseline: {Path(base_root).name}  ({len(base)} runs)")
    for ar in arm_roots:
        arm = load(ar)
        shared = sorted(set(base) & set(arm))
        if not shared:
            print(f"{Path(ar).name:<22} NO SHARED (target, seed) CELLS -- cannot check")
            continue
        ident = 0
        dl, dj = [], []
        for k in shared:
            b, a = base[k], arm[k]
            same = (b["loss"] == a["loss"] and np.array_equal(b["J"], a["J"])
                    and b["sig"] == a["sig"])
            ident += int(same)
            dl.append(abs(a["loss"] - b["loss"]))
            dj.append(float(np.abs(a["J"] - b["J"]).max()))
        cid = sorted({a["config_id"] for a in arm.values()})
        verdict = "INERT (NO-OP)" if ident == len(shared) else "LIVE"
        print(f"{Path(ar).name:<22} n={len(shared):>3} identical={ident:>3} "
              f"max|dloss|={max(dl):.3e} max|dJ|={max(dj):.3e}  cfg={cid[0][:8]}  {verdict}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
