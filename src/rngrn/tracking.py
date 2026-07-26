"""tracking.py — a Tracker protocol with a no-op default.

Turning on an experiment tracker is a config flag, not a code change. The W&B
implementation is lazily imported so the dependency is optional.
"""
from __future__ import annotations
from typing import Protocol, Any


class Tracker(Protocol):
    def log_config(self, cfg: dict) -> None: ...
    def log_scalars(self, step: int, scalars: dict) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def finish(self) -> None: ...


class NoOpTracker:
    def log_config(self, cfg): pass
    def log_scalars(self, step, scalars): pass
    def log_artifact(self, path): pass
    def finish(self): pass


class WandbTracker:
    def __init__(self, project: str, run_name: str):
        import wandb  # lazy
        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name)

    def log_config(self, cfg): self._run.config.update(cfg)
    def log_scalars(self, step, scalars): self._wandb.log(scalars, step=step)
    def log_artifact(self, path): self._run.log({"artifact": self._wandb.Image(path)}
                                                if path.endswith(".png") else {})
    def finish(self): self._run.finish()


def build_tracker(cfg) -> Tracker:
    if cfg.backend == "wandb":
        return WandbTracker(cfg.project, cfg.run_name)
    return NoOpTracker()
