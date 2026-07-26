"""scoring/ — SCORING-SIDE metrics for the identifiability experiments.

Everything here reads the quarantined answer key and therefore runs ONLY after recovery
has finished. No recovery-side module may import this package; tests/test_firewall.py and
the per-module ast audits enforce that.

  permutation.py : Experiment A — hidden-channel identifiability. Aligns the recovered J
                   to the truth over permutations of the UNOBSERVED species (observed
                   indices pinned), plus the observed-subblock comparison that stays valid
                   when model N differs from true N, plus a latent-field correlation
                   diagnostic.
  overparam.py   : Experiment B — over-parameterisation robustness. Measures whether a
                   spare species stays inert (norm fraction, strongest single edge,
                   decoupling) and whether the observed sub-block matches the truth.

Import from the submodules directly (`from rngrn.scoring import permutation as P`) — this
__init__ deliberately imports nothing so the two modules stay independently loadable and
a failure in one cannot mask the other.
"""
