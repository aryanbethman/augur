"""Cluster routers: pick an engine for each released call.

Placement is outside vLLM. These are simple built-in policies; the vLLM
Router's policies plug in behind the same interface later (design doc,
Phase 2).
"""

from __future__ import annotations

from typing import Protocol, Sequence


class EngineView(Protocol):
    engine_id: int

    def load(self) -> int: ...


class Router(Protocol):
    def route(self, call, engines: Sequence[EngineView]) -> int: ...


class RoundRobin:
    def __init__(self) -> None:
        self._next = 0

    def route(self, call, engines):
        i = self._next % len(engines)
        self._next += 1
        return engines[i].engine_id


class LeastLoaded:
    """Fewest waiting + running requests; ties go to the lowest engine id.

    Matches the fork's default router, which considers request counts only.
    """

    def route(self, call, engines):
        return min(engines, key=lambda e: (e.load(), e.engine_id)).engine_id


ROUTERS = {"round-robin": RoundRobin, "least-loaded": LeastLoaded}
