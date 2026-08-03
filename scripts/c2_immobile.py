"""c2_immobile.py — how often does nc1 recovery park one node as EFFECTIVELY IMMOBILE?

JOB C. The one prior nc1 run reported `plausibility_d_ratio_value` 2.32 against a RAW
max/min D ratio of 723.8, because scoring/plausibility.py::d_ratio_of deliberately scores
the two most mobile species and lets the third sit below them. That is not a metric
loophole being exploited by accident: Stage 0 measured the same construction giving a 17x
acceptance gain inside the biological box, and 127/127 generator systems stay strictly
Turing when their slowest diffuser is immobilised (docs/BIO_VIABILITY.md). It is Tica et
al.'s mechanism for relaxing the differential-diffusion requirement. If nc1 finds it
systematically, that is a result about the model, not an artefact of the scorer.

THE THRESHOLD IS PHYSICAL, NOT A ROUND NUMBER. "Immobile" cannot be a bare D ratio --
a ratio has no scale. A species is immobile AT THE PATTERN'S OWN WAVENUMBER when its
diffusive loss there is negligible against its own reaction timescale:

    q_i = D_i * kstar_model**2 / |J_ii|

q_i << 1 means that at k = k*, the species' diffusion term is a small correction to its own
diagonal reaction rate, i.e. removing its diffusion entirely would barely move sigma(k).
Reported at q < 0.1 and q < 0.01 so the conclusion's dependence on where the line is drawn
is visible rather than hidden. k*_model, not k*_obs: the question is about the pattern the
RECOVERED system makes.

Usage:  python scripts/c2_immobile.py experiments/<root> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(roots):
    # q IS ONLY MEANINGFUL ON A PATTERNING SEED. q_i = D_i k*^2 / |J_ii| goes to 0 for
    # EVERY species when k*_model collapses to ~0, which is exactly what the non-Turing
    # seeds do (baseline sample_0000: k*_model = 0.00587 on 8/8 seeds). Counting those as
    # "immobile" would measure the collapse, not the mechanism. Rows are therefore split:
    # `tur` is the Turing subset and is the one the JOB C claim is read from.
    print(f"{'root':<24} {'target':<12} {'n':>3} {'tur':>4} {'q<0.1|tur':>10} "
          f"{'q<0.01|tur':>11} {'med q_min|tur':>13} {'med Dlo/Dmid':>12} "
          f"{'med rawD':>10} {'med scoredD':>11}")
    for root in roots:
        smap = {}
        p = Path(root) / "runs.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    smap[r["run_id"]] = r.get("sample_key", "?")
        by = {}
        for rd in sorted(Path(root).glob("runs/*")):
            f = rd / "results" / "train_results.json"
            if not f.exists() or f.stat().st_size == 0:
                continue
            d = json.loads(f.read_text())
            D = np.array(d["recovered"]["D"], dtype=float)
            J = np.array(d["recovered"]["J"], dtype=float)
            ks = float(d.get("kstar_model", float("nan")))
            m = d.get("metric", {})
            q = D * ks ** 2 / np.abs(np.diag(J))
            s = np.sort(D)
            by.setdefault(smap.get(rd.name, "?"), []).append(
                (float(np.min(q)), s[0] / s[1], s[2] / s[0],
                 float(m.get("plausibility_d_ratio_value", float("nan"))),
                 float(bool(m.get("recovered_turing", False)))))
        for t, rows in sorted(by.items()):
            a = np.array(rows, dtype=float)
            n = len(a)
            tur = a[a[:, 4] > 0]
            nt = len(tur)
            if nt:
                c1 = f"{int((tur[:, 0] < 0.1).sum())}/{nt}"
                c2 = f"{int((tur[:, 0] < 0.01).sum())}/{nt}"
                qm = f"{np.median(tur[:, 0]):.2e}"
            else:
                c1 = c2 = qm = "-"
            print(f"{Path(root).name:<24} {t:<12} {n:>3} {nt:>4} {c1:>10} {c2:>11} "
                  f"{qm:>13} {np.median(a[:, 1]):>12.4f} "
                  f"{np.median(a[:, 2]):>10.1f} {np.median(a[:, 3]):>11.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
