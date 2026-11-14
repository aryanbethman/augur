"""Step cost models.

A cost model turns one engine step (the batch vLLM's scheduler chose) into a
duration in nanoseconds. The executor hands it a ``StepShape`` built from the
``SchedulerOutput``; it never sees vLLM objects, so cost models stay independent
of vLLM internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class SeqWork:
    """One request's share of a step."""

    num_new: int  # tokens computed this step
    num_computed_before: int  # tokens already in the KV cache (incl. prefix hits)
    is_decode: bool  # one new token past the known tokens, not a prefill chunk


@dataclass(frozen=True)
class StepShape:
    seqs: tuple[SeqWork, ...] = field(default_factory=tuple)
    # Token count the forward pass runs at after vLLM's CUDA-graph padding
    # (vLLM's own CudagraphDispatcher decides it). None: no padding.
    padded_tokens: int | None = None

    @property
    def forward_tokens(self) -> int:
        return max(self.total_tokens, self.padded_tokens or 0)

    @property
    def total_tokens(self) -> int:
        return sum(s.num_new for s in self.seqs)

    @property
    def num_decode(self) -> int:
        return sum(1 for s in self.seqs if s.is_decode)

    @property
    def num_prefill(self) -> int:
        return sum(1 for s in self.seqs if not s.is_decode)


class CostModel(Protocol):
    def step_ns(self, shape: StepShape) -> int: ...


@dataclass
class FixedCost:
    """Constant step time, plus an optional per-token term. For plumbing tests."""

    base_ns: int = 25_000_000
    per_token_ns: int = 0

    def step_ns(self, shape: StepShape) -> int:
        return self.base_ns + self.per_token_ns * shape.total_tokens
