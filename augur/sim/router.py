"""Cluster routers: pick an engine for each released call.

Placement is outside vLLM. ``round-robin`` and ``least-loaded`` are augur's
simple policies (least-loaded matches LLMServingSim's default router). The
others port the vLLM Router's policies (github.com/vllm-project/router,
src/policies, main as of 2026-11-18) with its CLI defaults:

* ``random``: uniform over the engines.
* ``power-of-two``: two distinct engines drawn at random; the one with fewer
  requests in flight wins, ties to the first drawn.
* ``cache-aware`` (the router's default): a prefix tree per engine of the
  prompts sent to it, keyed on token ids as the router does in its normal
  mode. Unless in-flight counts are imbalanced, the engines with the best
  match go first when that match is above ``cache_threshold``; among the
  candidates the fewest in flight wins, ties to the lowest engine. Every
  ``eviction_interval_s`` each tree drops its least recently used prompts
  until it is within ``max_tree_size``.
* ``consistent-hash``: a ring with 160 virtual nodes per engine.
* ``rendezvous-hash``: the highest hash of (key, engine) wins.
* ``sticky-least-loaded``: a session keeps the engine it was first given
  (the one with the fewest sessions then) until it is idle past its expiry.

The hash policies key a call on its session id (the router reads the
``x-session-id`` header) or, without one, on its prompt. "In flight" is the
router's own count: from routing until the response stream ends
(``release``). Not ported: the router's hash function (FurcHash over
MurmurHash64A; engine URLs are synthetic here, so any good 64-bit hash
places keys alike), health checks and circuit breakers (every engine is
healthy), and the kv-aware policy (it needs the engines' KV events).
"""

from __future__ import annotations

import bisect
import hashlib
import heapq
import inspect
import random
from array import array
from collections import defaultdict
from typing import Protocol, Sequence

S = 1_000_000_000  # ns per second
TOKEN_WIDTH = 8  # the router writes a token id as 8 hex characters; max_tree_size counts characters
VIRTUAL_NODES = 160
SWEEP_NS = 60 * S  # sticky-least-loaded drops expired sessions at most this often


class EngineView(Protocol):
    engine_id: int

    def load(self) -> int: ...


class Router(Protocol):
    def route(self, call, engines: Sequence[EngineView], now_ns: int) -> int: ...

    def release(self, call, at_ns: int = 0) -> None: ...


class _Policy:
    """Counts each engine's requests in flight as the vLLM Router does, from
    routing until the response stream ends. Called by: Coordinator._pick
    (route), Coordinator._finish and _on_step_done (release). Calls: choose."""

    def __init__(self) -> None:
        self.in_flight: dict[int, int] = defaultdict(int)
        self._engine_of: dict[str, int] = {}
        self._releases: list[tuple[int, int, str]] = []  # (at_ns, order, call id)
        self._order = 0

    def route(self, call, engines: Sequence[EngineView], now_ns: int) -> int:
        self._apply_releases(now_ns)
        eid = self.choose(call, engines, now_ns)
        self.in_flight[eid] += 1
        self._engine_of[call.call_id] = eid
        return eid

    def release(self, call, at_ns: int = 0) -> None:
        """The call's response stream ends at at_ns. Applied at the first
        routing decision at or after it, so the coordinator needs no event
        of its own for it."""
        self._order += 1
        heapq.heappush(self._releases, (at_ns, self._order, call.call_id))

    def _apply_releases(self, now_ns: int) -> None:
        while self._releases and self._releases[0][0] <= now_ns:
            _, _, cid = heapq.heappop(self._releases)
            eid = self._engine_of.pop(cid, None)
            if eid is not None:
                self.in_flight[eid] -= 1

    def choose(self, call, engines: Sequence[EngineView], now_ns: int) -> int:
        raise NotImplementedError


class RoundRobin(_Policy):
    def __init__(self) -> None:
        super().__init__()
        self._next = 0

    def choose(self, call, engines, now_ns):
        i = self._next % len(engines)
        self._next += 1
        return engines[i].engine_id


class LeastLoaded(_Policy):
    """Fewest waiting + running requests in the engine; ties go to the lowest
    engine id. Matches the fork's default router, which counts requests only."""

    def choose(self, call, engines, now_ns):
        return min(engines, key=lambda e: (e.load(), e.engine_id)).engine_id


class RandomChoice(_Policy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.rng = random.Random(seed)

    def choose(self, call, engines, now_ns):
        return engines[self.rng.randrange(len(engines))].engine_id


class PowerOfTwo(_Policy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.rng = random.Random(seed)

    def choose(self, call, engines, now_ns):
        if len(engines) == 1:
            return engines[0].engine_id
        i = self.rng.randrange(len(engines))
        j = self.rng.randrange(len(engines))
        while j == i:
            j = self.rng.randrange(len(engines))
        a, b = engines[i].engine_id, engines[j].engine_id
        return a if self.in_flight[a] <= self.in_flight[b] else b


class CacheAware(_Policy):
    """The vLLM Router's cache_aware policy, normal (token id) path. Calls: PrefixTree."""

    def __init__(self, cache_threshold: float = 0.3, balance_abs_threshold: int = 64,
                 balance_rel_threshold: float = 1.5, eviction_interval_s: float = 120.0,
                 max_tree_size: int = 1 << 25) -> None:
        super().__init__()
        self.cache_threshold = cache_threshold
        self.balance_abs_threshold = balance_abs_threshold
        self.balance_rel_threshold = balance_rel_threshold
        self.eviction_ns = int(eviction_interval_s * S)
        self.max_tree_tokens = max_tree_size // TOKEN_WIDTH
        self.trees: dict[int, PrefixTree] = defaultdict(PrefixTree)
        self._next_eviction = self.eviction_ns

    def choose(self, call, engines, now_ns):
        self._evict_due(now_ns)
        key = PrefixTree.key(call.prompt_ids)
        n = len(call.prompt_ids)
        loads = [self.in_flight[e.engine_id] for e in engines]
        lo, hi = min(loads), max(loads)
        imbalanced = hi - lo > self.balance_abs_threshold and hi > lo * self.balance_rel_threshold
        rates = [self.trees[e.engine_id].match(key) / n if n else 0.0 for e in engines]
        best = max(rates)
        use_prefix = not imbalanced and best > self.cache_threshold
        pick = min((i for i in range(len(engines)) if not use_prefix or rates[i] == best), key=lambda i: loads[i])
        eid = engines[pick].engine_id
        self.trees[eid].insert(key, now_ns)
        return eid

    def _evict_due(self, now_ns: int) -> None:
        """The router evicts on a timer from its start. Passes that fell
        between two routing decisions collapse into one: with no insert in
        between, a second pass changes nothing."""
        if self.eviction_ns <= 0 or now_ns < self._next_eviction:
            return
        for tree in self.trees.values():
            tree.evict(self.max_tree_tokens)
        self._next_eviction = (now_ns // self.eviction_ns + 1) * self.eviction_ns


class _Node:
    __slots__ = ("label", "children", "parent", "stamp")

    def __init__(self, label: bytes, parent: "_Node | None", stamp: int):
        self.label = label
        self.children: dict[bytes, _Node] = {}
        self.parent = parent
        self.stamp = stamp


class PrefixTree:
    """One engine's prompts as a radix tree over token ids (8 bytes each).
    Each node holds when a prompt last went through it, for LRU eviction of
    leaves as in the router's tree. Called by: CacheAware."""

    def __init__(self) -> None:
        self.root = _Node(b"", None, 0)
        self.size = 0  # tokens held

    @staticmethod
    def key(token_ids) -> bytes:
        return array("Q", token_ids).tobytes()

    def match(self, key: bytes) -> int:
        """Tokens of key that a prompt in the tree starts with."""
        node, pos = self.root, 0
        while pos < len(key):
            child = node.children.get(key[pos:pos + 8])
            if child is None:
                break
            k = _common_prefix(child.label, key, pos)
            pos += k
            if k < len(child.label):
                break
            node = child
        return pos // 8

    def insert(self, key: bytes, stamp: int) -> None:
        node, pos = self.root, 0
        while pos < len(key):
            first = key[pos:pos + 8]
            child = node.children.get(first)
            if child is None:
                node.children[first] = _Node(key[pos:], node, stamp)
                self.size += (len(key) - pos) // 8
                return
            k = _common_prefix(child.label, key, pos)
            if k < len(child.label):
                child = self._split(node, child, k)
            child.stamp = stamp
            node, pos = child, pos + k

    @staticmethod
    def _split(parent: _Node, child: _Node, k: int) -> _Node:
        mid = _Node(child.label[:k], parent, child.stamp)
        child.label = child.label[k:]
        child.parent = mid
        mid.children[child.label[:8]] = child
        parent.children[mid.label[:8]] = mid
        return mid

    def evict(self, max_tokens: int) -> None:
        """Drop least recently used leaves until the tree holds max_tokens or fewer."""
        if self.size <= max_tokens:
            return
        heap = [(n.stamp, id(n), n) for n in self._leaves()]
        heapq.heapify(heap)
        while heap and self.size > max_tokens:
            _, _, node = heapq.heappop(heap)
            parent = node.parent
            del parent.children[node.label[:8]]
            self.size -= len(node.label) // 8
            if parent is not self.root and not parent.children:
                heapq.heappush(heap, (parent.stamp, id(parent), parent))

    def _leaves(self) -> list[_Node]:
        out, stack = [], list(self.root.children.values())
        while stack:
            n = stack.pop()
            if n.children:
                stack.extend(n.children.values())
            else:
                out.append(n)
        return out


def _common_prefix(label: bytes, key: bytes, pos: int) -> int:
    """Bytes label shares with key[pos:], in whole tokens. The first token is
    known to match (children are keyed by it)."""
    n = min(len(label), len(key) - pos) // 8
    if label[:n * 8] == key[pos:pos + n * 8]:
        return n * 8
    lo, hi = 1, n  # the first lo tokens match, the first hi do not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if label[:mid * 8] == key[pos:pos + mid * 8]:
            lo = mid
        else:
            hi = mid
    return lo * 8


def _hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def _url(engine_id: int) -> str:
    return f"http://augur-engine-{engine_id}"


def _prompt_key(call) -> str:
    return "request_hash:" + hashlib.blake2b(PrefixTree.key(call.prompt_ids), digest_size=8).hexdigest()


def _hash_key(call, session_ids: bool) -> str:
    """The router's key: the x-session-id header if the client sends one, else the request body."""
    if session_ids and call.session_id is not None:
        return f"header:x-session-id:{call.session_id}"
    return _prompt_key(call)


class ConsistentHash(_Policy):
    def __init__(self, session_ids: bool = True) -> None:
        super().__init__()
        self.session_ids = session_ids
        self._engine_ids: tuple = ()
        self._points: list[int] = []
        self._owners: list[int] = []

    def _build_ring(self, engines) -> None:
        ids = tuple(e.engine_id for e in engines)
        if ids == self._engine_ids:
            return
        ring = {}
        for eid in ids:  # on a collision the later engine takes the point, as in the router
            for i in range(VIRTUAL_NODES):
                ring[_hash64(f"{_url(eid)}:{i}")] = eid
        self._points = sorted(ring)
        self._owners = [ring[p] for p in self._points]
        self._engine_ids = ids

    def choose(self, call, engines, now_ns):
        self._build_ring(engines)
        i = bisect.bisect_left(self._points, _hash64(_hash_key(call, self.session_ids)))
        return self._owners[i % len(self._points)]


class RendezvousHash(_Policy):
    def __init__(self, session_ids: bool = True) -> None:
        super().__init__()
        self.session_ids = session_ids

    def choose(self, call, engines, now_ns):
        key = _hash_key(call, self.session_ids)
        best_score, best = -1, None
        for e in engines:  # the last of equal scores wins, as Rust's max_by_key
            score = _hash64(f"{key}:{_url(e.engine_id)}")
            if score >= best_score:
                best_score, best = score, e.engine_id
        return best


class StickyLeastLoaded(_Policy):
    def __init__(self, session_expiry_s: float = 7200.0, session_ids: bool = True) -> None:
        super().__init__()
        self.expiry_ns = int(session_expiry_s * S)
        self.session_ids = session_ids
        self.sessions: dict[str, list] = {}  # session id -> [engine id, last access ns]
        self.active: dict[int, int] = defaultdict(int)  # sessions per engine
        self._last_sweep: int | None = None

    def choose(self, call, engines, now_ns):
        self._sweep(now_ns)
        sid = call.session_id if self.session_ids else None
        if sid is None:
            return self._fewest_sessions(engines, _prompt_key(call))  # not remembered
        held = self.sessions.get(sid)
        if held is not None:
            if held[0] in {e.engine_id for e in engines} and now_ns - held[1] < self.expiry_ns:
                held[1] = now_ns
                return held[0]
            self._drop(sid)
        eid = self._fewest_sessions(engines, sid)
        self.sessions[sid] = [eid, now_ns]
        self.active[eid] += 1
        return eid

    def _fewest_sessions(self, engines, tie_key: str) -> int:
        """Fewest active sessions; among those the highest hash of (tie_key, engine)."""
        best_key, best = None, None
        for e in engines:
            k = (self.active[e.engine_id], -_hash64(f"{tie_key}:{_url(e.engine_id)}"))
            if best_key is None or k < best_key:
                best_key, best = k, e.engine_id
        return best

    def _sweep(self, now_ns: int) -> None:
        if self._last_sweep is not None and now_ns - self._last_sweep < SWEEP_NS:
            return
        self._last_sweep = now_ns
        for sid in [s for s, (_, t) in self.sessions.items() if now_ns - t >= self.expiry_ns]:
            self._drop(sid)

    def _drop(self, sid: str) -> None:
        eid, _ = self.sessions.pop(sid)
        self.active[eid] -= 1


ROUTERS = {
    "round-robin": RoundRobin,
    "least-loaded": LeastLoaded,
    "random": RandomChoice,
    "power-of-two": PowerOfTwo,
    "cache-aware": CacheAware,
    "consistent-hash": ConsistentHash,
    "rendezvous-hash": RendezvousHash,
    "sticky-least-loaded": StickyLeastLoaded,
}


def make_router(name: str, **options) -> Router:
    """The named policy with the options it takes; the rest are ignored, so
    the CLI can pass every router option to any policy. Called by: cli.main."""
    cls = ROUTERS[name]
    accepted = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in options.items() if k in accepted})
