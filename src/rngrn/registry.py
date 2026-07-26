"""A tiny name -> implementation registry.

Used for the three varying concerns (model architectures, loss terms, weighting
strategies) so that adding a variant is a decorator, never an edit to the
training loop. Mirrors the pattern the reference codebase already applied to
regulation forms and targets.
"""
from __future__ import annotations
from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if key in self._items:
                raise KeyError(f"{self.name}: '{key}' already registered")
            self._items[key] = obj
            return obj
        return deco

    def get(self, key: str) -> T:
        if key not in self._items:
            raise KeyError(
                f"{self.name}: unknown key '{key}'. Registered: {sorted(self._items)}"
            )
        return self._items[key]

    def keys(self):
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items
