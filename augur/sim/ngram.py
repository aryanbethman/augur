"""Token-faithful ngram speculation: the drafter of vLLM's model runner over a request's real context.

The simulated model emits the recorded output ids (augur.trace.workload), so a request's context at any step is its
prompt plus the ids emitted so far, as on a GPU.  After each step the drafter proposes from that context and the next
step accepts the longest prefix of the drafts that equals the next recorded ids (exact greedy verification);
acceptance is the text's, not a rate.

The drafter is the one the runner in use has (``vllm_config.use_v2_model_runner``, as vLLM selects it):

* V2 runner (0.33's default): ``NgramGPUSpeculator``, two Triton kernels.  Triton can run them on CPU tensors
  (``TRITON_INTERPRET=1``), but at ~3 ms per program that is hours per run, so ``v2_drafts`` repeats them in numpy;
  tests/test_token_faithful.py runs the kernels themselves (interpreted) against it.  Against the real
  ShareGPT 300 runs' step logs (sync and async) it reproduces every logged request-step's accepted count.
* V1 runner: vLLM's CPU ``NgramProposer``, run as is.  It takes the earliest of equal longest matches where the V2
  kernel takes the latest, and returns fewer than k drafts when it finds none.
"""

from __future__ import annotations

import numpy as np

from vllm.v1.spec_decode.ngram_proposer import NgramProposer

_INT32_MAX = (1 << 31) - 1


def as_int32(ids) -> np.ndarray:
    """Ids as int32 (the proposer's dtype).  Filler ids beyond int32 (augur.trace.workload) are unique within their
    request and never recur, so each becomes a distinct negative number that can match nothing."""
    a = np.asarray(ids, dtype=np.int64)
    big = (a > _INT32_MAX) | (a < 0)
    if big.any():
        a = a.copy()
        a[big] = -1 - np.flatnonzero(big)
    return a.astype(np.int32)


def v2_drafts(context: np.ndarray, k: int, min_n: int, max_n: int) -> list[int]:
    """``NgramGPUSpeculator``'s drafts for one row (vllm/v1/worker/gpu/spec_decode/ngram/speculator.py,
    ``_ngram_scan_kernel`` and ``_ngram_finalize_kernel``): the longest suffix n-gram (max_n down to min_n) that
    occurs earlier (its occurrence must end before the last token), the latest such occurrence; the k ids after it,
    each slot past the end of the context (and all of them without a match) the last id."""
    L, last = len(context), int(context[-1])
    for n in range(max_n, min_n - 1, -1):
        if L - n < 1:
            continue
        hit = np.ones(L - n, dtype=bool)
        for j in range(n):
            hit &= context[j:L - n + j] == context[L - n + j]
        where = np.flatnonzero(hit)
        if where.size:
            start = int(where[-1]) + n
            return [int(context[start + i]) if start + i < L else last for i in range(k)]
    return [last] * k


class Context:
    """A request's token ids (prompt + emitted) in a growing int32 buffer."""

    def __init__(self, ids, spare: int):
        first = as_int32(ids)
        self.buf = np.empty(len(first) + spare, dtype=np.int32)
        self.buf[:len(first)] = first
        self.n = len(first)

    def append(self, ids) -> None:
        more = as_int32(ids)
        if self.n + len(more) > len(self.buf):
            self.buf = np.resize(self.buf, 2 * (self.n + len(more)))
        self.buf[self.n:self.n + len(more)] = more
        self.n += len(more)

    @property
    def ids(self) -> np.ndarray:
        return self.buf[:self.n]


class NgramDrafter:
    """The runner's ngram drafter for the requests that emitted a token this step."""

    def __init__(self, vllm_config):
        sc = vllm_config.speculative_config
        self.k, self.min_n, self.max_n = sc.num_speculative_tokens, sc.prompt_lookup_min, sc.prompt_lookup_max
        self.v2 = vllm_config.use_v2_model_runner
        self.proposer = None if self.v2 else NgramProposer(vllm_config)

    def propose(self, contexts: list[Context]) -> list[list[int]]:
        """The drafts (k of them, or fewer from the V1 proposer) for each context."""
        if self.v2:
            return [v2_drafts(c.ids, self.k, self.min_n, self.max_n) for c in contexts]
        lens = np.array([c.n for c in contexts], dtype=np.int32)
        tokens = np.zeros((len(contexts), int(lens.max())), dtype=np.int32)
        for i, c in enumerate(contexts):
            tokens[i, :c.n] = c.ids
        return self.proposer.propose(self.k, [[int(c.ids[-1])] for c in contexts], lens, tokens)
