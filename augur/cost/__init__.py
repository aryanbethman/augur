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
    is_decode: bool  # query length 1 (vLLM's decode/prefill split), not a multi-token chunk
    # Of its ``num_computed_before`` tokens of KV, those held in cache blocks that a sequence before it in this step
    # also references (prefix caching: one physical block, read by several requests). 0: nothing shared, or unknown.
    shared_kv: int = 0


@dataclass(frozen=True)
class StepShape:
    seqs: tuple[SeqWork, ...] = field(default_factory=tuple)
    # Token count the forward pass runs at after vLLM's CUDA-graph padding
    # (vLLM's own CudagraphDispatcher decides it). None: no padding.
    padded_tokens: int | None = None
    # An idle DP rank's dummy forward (vLLM execute_dummy_batch): runs the
    # layers and the EP collectives, but no logits or sampler.
    dummy: bool = False

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

    @property
    def step_class(self) -> str:
        """One of STEP_CLASSES: no prefill sequence, only prefill sequences, or both."""
        if self.num_prefill == 0:
            return "decode"
        return "prefill" if self.num_decode == 0 else "mixed"


STEP_CLASSES = ("decode", "prefill", "mixed")


def parse_class_scales(spec: str | None) -> dict[str, float]:
    """Per-class GPU time scales from a JSON file (``python -m augur.verify.calibrate --output``, or just
    {"decode": 1.01, ...}) or inline text ``decode=1.01,mixed=0.97``. A class left out keeps 1.0.
    Called by: augur.cli, augur.verify.calibrate."""
    if not spec:
        return {}
    if "=" in spec:
        out = {k.strip(): float(v) for k, v in (kv.split("=", 1) for kv in spec.split(","))}
    else:
        import json
        with open(spec) as f:
            out = json.load(f)
        out = out.get("scales", out)
    bad = set(out) - set(STEP_CLASSES)
    if bad:
        raise ValueError(f"unknown step class {sorted(bad)}; classes are {list(STEP_CLASSES)}")
    return {k: float(v) for k, v in out.items()}


class CostModel(Protocol):
    def step_ns(self, shape: StepShape) -> int: ...


@dataclass
class FixedCost:
    """Constant step time, plus an optional per-token term. For plumbing tests."""

    base_ns: int = 25_000_000
    per_token_ns: int = 0

    def step_ns(self, shape: StepShape) -> int:
        return self.base_ns + self.per_token_ns * shape.total_tokens
