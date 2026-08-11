"""tune_comp_kstar_bias.py — SIGNED k* error per cell, from the run index.

docs/LGEN_TRANSFER_FIRST_RESULT.md found the recovered model UNDERSHOOTS k* (wavelength
5-16 % too long) at every domain size — a systematic bias, not scatter. `kstar_fft_rel_err`
is an ABSOLUTE error and cannot show it. This reads `kstar_model` and `kstar_fft_true`
straight off experiments/tune_comp/runs.jsonl and reports the SIGNED relative error
(negative = undershoot = wavelength too long), grouped by config_id, so any axis that moves
the bias is visible.

Only Turing-reaching rows are used: k* of a non-patterning recovery is not a wavelength.
"""
import json, sys, os
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "experiments", "tune_comp", "runs.jsonl")

rows = [json.loads(l) for l in open(IDX)]
# config_id -> the tuning axis that cell varied; recovered from the columns train.py writes.
def cell_of(r):
    return (f"steps={r.get('adam_steps')}", f"detach={r.get('detach_xstar')}",
            f"dinit={r.get('d_init_from_kstar')}", f"prior={r.get('w_param_prior')}",
            f"anchor={r.get('w_anchor')}", f"init={r.get('model_init')}",
            f"offfrac={r.get('staging_off_frac')}")

g = defaultdict(list)
for r in rows:
    if not r.get("recovered_turing"):
        continue
    km, kt = r.get("kstar_model"), r.get("kstar_fft_true")
    if km is None or kt in (None, 0):
        continue
    g[(cell_of(r), r.get("sample_key"))].append((float(km) - float(kt)) / float(kt))

print(f"{'cell':<95}{'sample':<14}{'n':>4}{'signed_med':>12}{'signed_mean':>13}{'n_under':>9}")
for (cell, samp), v in sorted(g.items(), key=lambda kv: str(kv[0])):
    a = np.array(v)
    print(f"{','.join(cell):<95}{str(samp):<14}{len(a):>4}"
          f"{np.median(a):>12.4f}{a.mean():>13.4f}{int((a < 0).sum()):>9}")
