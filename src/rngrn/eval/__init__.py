"""eval/ — post-training analysis & validation (RECOVERY SIDE, reads only the model).

dynamical : QSS->lift and the mu->0 consistency check
numerics  : ETDRK4 / IMEX / BDF1 stiff integrators
rollout   : lift-and-simulate (the real success test)
analysis  : linear stability, Turing conditions, robustness cloud
topology  : learned-GRN diagram
lgen_eval : cross-domain-size evaluation — one recovered model rolled out on several box
            sizes. Deliberately NOT re-exported below: it uses scoring/morphology for the
            class call, so like scoring/* it runs strictly after recovery. Import it
            explicitly (`from rngrn.eval.lgen_eval import evaluate_across_L`).

None of these import the answer-key side.
"""
from .rollout import simulate                       # noqa: F401
from .dynamical import lift_check                    # noqa: F401
from .analysis import turing_ok, linear_stability, robustness_cloud, robustness_volumes  # noqa: F401
