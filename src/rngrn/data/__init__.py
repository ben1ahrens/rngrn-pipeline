"""data/ — the answer-key side + the firewall gate.

rd_models, solver, cache, spec are ANSWER-KEY SIDE (generators + truth).
gate is THE BOUNDARY: it is the only module that may hand data to recovery, and
it hands over exactly (frame, L, observed_idx). Recovery-side modules import
`from rngrn.data.gate import RecoveryInput` only — never rd_models/solver/cache.
"""
from .spec import DatasetSpec, spec_from_config  # noqa: F401
from .gate import RecoveryInput, AnswerKey        # noqa: F401
from . import registry                            # noqa: F401
