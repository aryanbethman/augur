"""CPU-tier eviction policies for vLLM 0.33's OffloadingConnector, written as an out-of-tree plugin.

vLLM resolves the CPU tier's eviction policy by name (vllm/v1/kv_offload/cpu/policies/factory.py), importing it
from ``cache_policy_module_path`` when it is not built in. Nothing in vLLM is patched. The same file loads in a real
server and in augur:

    vllm serve ... --kv-offloading-size 2 --kv-offloading-backend native \\
      --kv-transfer-config '{"kv_connector_extra_config": {"eviction_policy": "ToolWaitPolicy",
                              "cache_policy_module_path": "examples.kv_offload_policies"}}'

    python -m augur.cli ... --kv-offload-gib 2 \\
      --kv-offload-extra-config '{"eviction_policy": "ToolWaitPolicy",
                                  "cache_policy_module_path": "examples.kv_offload_policies"}'

(put the checkout on PYTHONPATH). The built-ins to compare against are ``lru`` (the default) and ``arc``.

Both policies below keep vLLM's LRU rank, (request clock, -token position): a request's tail goes before its head,
so a prefix never loses a block in front of one it keeps. They differ in what they put in front of the rank.

* ``ToolWaitPolicy``: a session with a request in flight has its KV on the GPU as well, so its CPU copy is the one
  to give up; a session waiting on a tool call has nothing but the CPU copy and its next call is coming. Blocks of
  sessions with no request in flight are evicted last. The session is read from the request id
  (``augur.<session>#<turn>.``, which real vLLM wraps as ``cmpl-augur.<session>#<turn>.-0``).
* ``ReuseFrequencyPolicy``: blocks that served the fewest requests go first, then LRU. A block reused by many
  turns of a session or by many sessions outlives one written once.

State is logical (counters), never a clock, so the policies decide the same on a virtual clock and a real one.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus, order_request_keys


def session_of(req_id: str) -> str:
    """The session in a request id: ``cmpl-augur.<session>#<turn>.-0`` and ``augur.<session>#<turn>.`` give
    <session>; any other id is its own session."""
    s = req_id.split("augur.", 1)[1] if "augur." in req_id else req_id
    s = s.rsplit(".-", 1)[0].rstrip(".")
    return s.rsplit("#", 1)[0]


@dataclass(slots=True)
class _Req:
    owner: object
    session: str
    clock: int  # this request's recency rank until it finishes
    finished: bool = False


class _RankedPolicy(CachePolicy):
    """LRU's bookkeeping (recency rank per chunk, lazy min-heap of the evictable ones) with an overridable
    priority: the chunk with the lowest ``priority(key)`` is evicted first."""

    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self._rank: dict[OffloadKey, tuple[int, int]] = {}
        self._uses: dict[OffloadKey, int] = {}  # requests that reused the chunk
        self._prio: dict[OffloadKey, tuple] = {}  # what each evictable chunk's live heap entry carries
        self._session: dict[OffloadKey, str] = {}  # the session that stored it
        self._by_session: dict[str, set[OffloadKey]] = {}
        self._evictable: set[OffloadKey] = set()
        self._heap: list[tuple[tuple, OffloadKey]] = []
        self._clock = 0

    # -- what a subclass decides ------------------------------------------------------------------
    def priority(self, key: OffloadKey) -> tuple:
        return (self._rank[key],)

    def request_started(self, session: str) -> None:
        """A request of this session reached the tier."""

    def request_done(self, session: str) -> None:
        """A request of this session finished."""

    # -- bookkeeping ------------------------------------------------------------------------------
    def _request(self, ctx: ReqContext) -> _Req:
        st = ctx.get_state(_Req)
        if st is None or st.owner is not self:
            self._clock += 1
            st = _Req(self, session_of(ctx.req_id), self._clock)
            ctx.set_state(st)
            self.request_started(st.session)
        return st

    def _push(self, key: OffloadKey) -> None:
        p = self._prio[key] = self.priority(key)
        heapq.heappush(self._heap, (p, key))
        if len(self._heap) > max(64, 2 * len(self._evictable)):  # drop stale entries
            self._heap = [(self._prio[k], k) for k in self._evictable]
            heapq.heapify(self._heap)

    def _repush(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            if key in self._evictable:
                self._push(key)

    def _drop(self, key: OffloadKey) -> None:
        del self.chunks[key]
        self._rank.pop(key, None)
        self._uses.pop(key, None)
        self._prio.pop(key, None)
        self._evictable.discard(key)
        s = self._session.pop(key, None)
        if s is not None:
            self._by_session[s].discard(key)

    def _set_rank(self, key: OffloadKey, clock: int, ctx: ReqContext) -> None:
        pos = ctx.get_offload_key_position(key)
        self._rank[key] = (clock, 0 if pos is None else -pos)

    # -- CachePolicy ------------------------------------------------------------------------------
    def get(self, key):
        return self.chunks.get(key)

    def insert(self, key, chunk, req_context) -> None:
        req = self._request(req_context)
        self.chunks[key] = chunk
        self._uses[key] = 0
        self._session[key] = req.session
        self._by_session.setdefault(req.session, set()).add(key)
        self._set_rank(key, req.clock, req_context)
        if chunk.ref_cnt == 0:
            self._evictable.add(key)
            self._push(key)

    def remove(self, key) -> None:
        self._drop(key)

    def touch(self, keys, req_context) -> None:
        req = self._request(req_context)
        for key in keys:
            if key in self.chunks:
                self._set_rank(key, req.clock, req_context)
                self._repush([key])

    def on_request_access(self, keys, req_context) -> None:
        self.touch(keys, req_context)

    def on_request_finished(self, key_groups: Sequence[Sequence[OffloadKey]], insertion_only_keys: set[OffloadKey],
                            reused_keys: set[OffloadKey], req_context: ReqContext) -> None:
        req = self._request(req_context)
        if req.finished:
            return
        req.finished = True
        self._clock += 1
        for key in order_request_keys(key_groups, req_context):
            if key in self.chunks:
                self._set_rank(key, self._clock, req_context)
                if key in reused_keys:
                    self._uses[key] += 1
                self._repush([key])
        self.request_done(req.session)

    def mark_evictable(self, key) -> None:
        self._evictable.add(key)
        self._push(key)

    def mark_non_evictable(self, key) -> None:
        self._evictable.discard(key)

    def clear(self) -> None:
        for d in (self.chunks, self._rank, self._uses, self._prio, self._session, self._by_session):
            d.clear()
        self._evictable.clear()
        self._heap.clear()
        self._clock += 1

    def evict(self, n: int, protected: set[OffloadKey]):
        if n == 0:
            return []
        taken, deferred, seen = [], [], set()
        while self._heap and len(taken) < n:
            entry = heapq.heappop(self._heap)
            p, key = entry
            if key not in self._evictable or self._prio.get(key) != p or key in seen:
                continue  # stale
            if key in protected:
                deferred.append(entry)
                continue
            taken.append(entry)
            seen.add(key)
        for entry in deferred + (taken if len(taken) < n else []):
            heapq.heappush(self._heap, entry)
        if len(taken) < n:
            return None
        out = []
        for _, key in taken:
            chunk = self.chunks[key]
            assert chunk.ref_cnt == 0
            self._drop(key)
            out.append((key, chunk))
        return out


class ToolWaitPolicy(_RankedPolicy):
    """Evict blocks of sessions with a request in flight first (their KV is on the GPU too), blocks of sessions
    waiting on a tool call last, LRU within each."""

    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)
        self._inflight: dict[str, int] = {}

    def priority(self, key):
        return (0 if self._inflight.get(self._session[key]) else 1, self._rank[key])

    def request_started(self, session: str) -> None:
        self._inflight[session] = self._inflight.get(session, 0) + 1
        if self._inflight[session] == 1:
            self._repush(self._by_session.get(session, ()))

    def request_done(self, session: str) -> None:
        self._inflight[session] -= 1
        if not self._inflight[session]:
            del self._inflight[session]
            self._repush(self._by_session.get(session, ()))


class ReuseFrequencyPolicy(_RankedPolicy):
    """Evict the blocks that served the fewest requests first, LRU among equals."""

    def priority(self, key):
        return (self._uses[key], self._rank[key])
