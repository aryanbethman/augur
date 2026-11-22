"""Shared transfer resources: every KV transfer is a flow through a path of resources.

Concurrent flows share each resource max-min fairly (progressive filling), so a
transfer slows when another one uses the same PCIe root, host memory, tier or
network link. The simulator's own transfer models (the offload worker's lanes,
the memory tiers' lanes, PD pulls) stay as they are; with ``--contention on``
their bandwidth becomes the fabric's share instead of a private constant.

Rules (all times integer ns, bandwidth in bytes per ns = GB/s):

* A flow has a fixed ``latency_ns`` before it consumes bandwidth.
* A flow alone on its path takes exactly ``latency_ns + int(nbytes / min(gbps))``,
  the cost ``OffloadLink`` charges. Flows submitted into one *stream* run one
  after another (a worker's FIFO lane): the next starts when the previous ends,
  and not before its ``earliest`` time.
* Rates are recomputed whenever the set of flows consuming bandwidth changes.
  Everything is processed in insertion order, so a run is deterministic.
* ``rounding="ceil"`` finishes a flow at ``ceil(remaining / rate)`` and treats
  half a byte or less as done (``--pd-contention decode``); with one link it
  matches the processor-sharing reference in tests/test_fabric.py to the ns.
* ``forced_end`` (replay): the flow ends at max(its start, forced_end), whatever
  its rate, like the offload worker's ``replay_done_ns``.

The fabric advances itself between ``advance(now)`` calls: flows that start
later (a load queued behind a forward pass) and finish in between are handled at
their own times, so a caller only ever needs to advance to the present.
``info(key)`` gives (start, end) of a flow, projected for one still running on the
assumption that no flow joins later. The projection simulates only the flows linked
to it (through resources, couplings or streams) and is cached until the next
``submit``/``cancel``.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass

INF = float("inf")
QUEUED, LATENT, ACTIVE = 0, 1, 2


class _Flow:
    __slots__ = ("key", "stream", "path", "nbytes", "remaining", "latency", "earliest", "state", "start", "lat_end",
                 "forced_end", "rnum", "rden", "t0", "fin", "rounding", "payload", "notify")

    def __init__(self, key, stream, path, nbytes, earliest, latency, forced_end, rounding, payload, notify):
        self.key, self.stream, self.path = key, stream, path
        self.nbytes = nbytes
        self.remaining = float(nbytes)
        self.latency, self.earliest = latency, earliest
        self.state = QUEUED
        self.start = None
        self.lat_end = None
        self.forced_end = forced_end
        self.rnum, self.rden = 0.0, 1  # rate = rnum / rden bytes per ns (kept as a fraction: see _recompute)
        self.t0 = 0  # when `remaining` was last brought up to date (only when the flows using bandwidth change)
        self.fin = INF
        self.rounding = rounding
        self.payload = payload
        self.notify = notify


class Fabric:
    def __init__(self, resources: dict[str, float]):
        for name, gbps in resources.items():
            if gbps <= 0:
                raise ValueError(f"fabric resource {name}: bandwidth must be > 0 (got {gbps})")
        self.cap: dict[str, float] = dict(resources)
        # (trigger, victim, penalty): while any flow uses ``trigger``, ``victim``'s capacity is (1 - penalty) of its own
        self.couplings: list[tuple[str, str, float]] = []
        self.time = 0
        self.flows: dict = {}  # key -> _Flow, every flow not yet done, in submission order
        self.streams: dict = {}  # stream -> deque of keys, in order
        self.stream_free: dict = {}  # stream -> when its last flow ended
        self.done: dict = {}  # key -> (start, end)
        self.completed: list = []  # (key, end, payload) of finished flows submitted with notify=True
        self.epoch = 0
        self._proj: tuple | None = None
        self.recomputes = 0  # diagnostic: rate recomputations

    def couple(self, trigger: str, victim: str, penalty: float) -> None:
        """Capacity of ``victim`` drops by ``penalty`` (a fraction) while a flow uses ``trigger``. Used for
        the PCIe duplex loss: measured on an RTX 4060 Ti, D2H falls 13.2 -> 11.1 GB/s (-15%) while an H2D copy
        runs on the same GPU, H2D about 1%."""
        if trigger not in self.cap or victim not in self.cap or not 0 <= penalty < 1:
            raise ValueError(f"coupling {trigger} -> {victim} ({penalty}) is not valid")
        self.couplings.append((trigger, victim, penalty))
        self.epoch += 1

    # ---- flows ------------------------------------------------------------
    def submit(self, key, stream, path, nbytes: int, earliest: int, latency_ns: int = 0, payload=None,
               notify: bool = False, rounding: str = "floor", forced_end: int | None = None) -> None:
        if key in self.flows or key in self.done:
            raise KeyError(f"flow {key!r} submitted twice")
        for r in path:
            if r not in self.cap:
                raise KeyError(f"flow {key!r}: no resource {r!r}")
        if stream is None:
            stream = ("flow", key)
        f = _Flow(key, stream, tuple(path), int(nbytes), int(earliest), int(latency_ns), forced_end, rounding,
                  payload, notify)
        self.flows[key] = f
        self.streams.setdefault(stream, deque()).append(key)
        self.epoch += 1

    def cancel(self, key) -> None:
        """Drop a flow that has not finished (its bytes are not moved)."""
        f = self.flows.pop(key, None)
        if f is None:
            return
        q = self.streams[f.stream]
        was_head = q[0] == key
        q.remove(key)
        if was_head:
            self.stream_free[f.stream] = max(self.stream_free.get(f.stream, 0), self.time)
        if not q:
            del self.streams[f.stream]
        self.done[key] = (f.start if f.start is not None else self.time, self.time)
        self.epoch += 1
        self._recompute(set(f.path))

    # ---- queries ----------------------------------------------------------
    def rate(self, key) -> float:
        f = self.flows.get(key)
        return f.rnum / f.rden if f is not None and f.state == ACTIVE else 0.0

    def rates_through(self, resource: str) -> float:
        """Sum of the rates of the flows using bandwidth on a resource now."""
        return sum(f.rnum / f.rden for f in self.flows.values() if f.state == ACTIVE and resource in f.path)

    def info(self, key) -> tuple[int, int]:
        """(start, end) of a flow: exact once finished, else projected."""
        d = self.done.get(key)
        if d is not None:
            return d
        if self._proj is None or self._proj[0] != self.epoch:
            self._proj = (self.epoch, {})
        if key not in self._proj[1]:
            self._proj[1].update(self._projection(key))
        return self._proj[1][key]

    def next_event(self) -> int | None:
        """The next time a flow starts, begins using bandwidth or ends."""
        t = self._next_time()
        return None if t is None else int(t)

    def pop_completed(self) -> list:
        out, self.completed = self.completed, []
        return out

    # ---- time ---------------------------------------------------------------
    def advance(self, now: int) -> None:
        """Run the fabric to ``now``: every event up to and including it happens at its own time."""
        if now < self.time:
            return
        while True:
            t = self._next_time()
            if t is None or t > now:
                break
            self._process(int(t))
        self._move(now)

    def _move(self, t: int) -> None:
        if t > self.time:
            self.time = t

    def _left(self, f: _Flow, t: int) -> float:
        """Bytes a flow using bandwidth still has at t, at the rate it has had since its last update. Updated
        only when the flows using bandwidth change, so the arithmetic does not depend on when the fabric is
        advanced (and, with one link, matches tests/test_fabric.py's SharedLink to the byte)."""
        return f.remaining - (t - f.t0) * f.rnum / f.rden

    def _next_time(self):
        t = INF
        for q in self.streams.values():
            f = self.flows[q[0]]
            if f.state == QUEUED:
                t = min(t, max(f.earliest, self.stream_free.get(f.stream, 0), self.time))
        for f in self.flows.values():
            if f.state == LATENT:
                t = min(t, f.lat_end)
            elif f.state == ACTIVE:
                t = min(t, max(f.fin, self.time))
        return None if t == INF else t

    def _process(self, t: int) -> None:
        self._move(t)
        changed = True
        any_change = False  # the flows using bandwidth changed: rates must be recomputed
        touched: set = set()  # resources a flow started or stopped using
        while changed:
            changed = False
            for f in list(self.flows.values()):
                if f.state == ACTIVE and self._is_done(f, t):
                    self._finish(f, t)
                    touched.update(f.path)
                    changed = any_change = True
            for f in list(self.flows.values()):
                if f.state == LATENT and f.lat_end <= t:
                    self._activate(f, t)
                    touched.update(f.path)
                    changed = any_change = True
            for q in list(self.streams.values()):
                f = self.flows[q[0]]
                if f.state == QUEUED and max(f.earliest, self.stream_free.get(f.stream, 0)) <= t:
                    f.start = t
                    changed = True
                    if f.latency > 0:
                        f.state, f.lat_end = LATENT, t + f.latency
                    else:
                        self._activate(f, t)
                        touched.update(f.path)
                        any_change = True
        if any_change:
            self._recompute(touched)

    def _is_done(self, f: _Flow, t: int) -> bool:
        if f.forced_end is not None:
            return t >= max(f.start, f.forced_end)
        if f.nbytes == 0 or f.fin <= t:
            return True
        return f.rounding == "ceil" and self._left(f, t) <= 0.5

    def _activate(self, f: _Flow, t: int) -> None:
        f.state = ACTIVE
        f.t0 = t
        f.fin = INF
        if f.forced_end is not None:
            f.fin = max(f.start, f.forced_end)

    def _finish(self, f: _Flow, t: int) -> None:
        del self.flows[f.key]
        q = self.streams[f.stream]
        q.popleft()
        if not q:
            del self.streams[f.stream]
        self.stream_free[f.stream] = t
        self.done[f.key] = (f.start, t)
        if f.notify:
            self.completed.append((f.key, t, f.payload))

    def _recompute(self, touched: set) -> None:
        """Max-min fair rates by progressive filling over the flows using bandwidth. A flow whose rate
        changes, or that shares a resource with a flow that started or stopped (``touched``), is first brought
        up to date at its old rate; any other keeps its arithmetic."""
        self.recomputes += 1
        t = self.time
        active = [f for f in self.flows.values() if f.state == ACTIVE]
        left = {}
        for f in active:
            for r in f.path:
                left.setdefault(r, self.cap[r])
        if self.couplings:
            used = {r for f in active for r in f.path}
            for trigger, victim, penalty in self.couplings:
                if trigger in used and victim in left:
                    left[victim] *= 1.0 - penalty
        new: dict = {}
        unfixed = list(active)
        while unfixed:
            count: dict = {}
            for f in unfixed:
                for r in f.path:
                    count[r] = count.get(r, 0) + 1
            best_r, best = None, INF
            for r, n in count.items():
                share = left[r] / n
                if share < best:
                    best_r, best = r, share
            if best_r is None:  # a flow with an empty path: unconstrained
                for f in unfixed:
                    new[f.key] = (INF, 1)
                break
            rest = []
            num, den = left[best_r], count[best_r]  # the share, as a fraction
            for f in unfixed:
                if best_r in f.path:
                    new[f.key] = (num, den)
                    for r in f.path:
                        left[r] -= best
                else:
                    rest.append(f)
            unfixed = rest
        for f in active:
            rate = new[f.key]
            if rate == (f.rnum, f.rden) and f.fin != INF and touched.isdisjoint(f.path):
                continue
            f.remaining, f.t0 = self._left(f, t), t
            f.rnum, f.rden = rate
            if f.forced_end is not None:
                f.fin = max(f.start, f.forced_end)
            elif f.nbytes == 0 or f.rnum == INF:
                f.fin = t
            elif f.rnum > 0:
                if f.rounding == "ceil":  # as SharedLink.next_finish
                    f.fin = t + max(0, int(-(-f.remaining * f.rden // f.rnum)))
                else:
                    f.fin = t + int(f.remaining * f.rden / f.rnum)
            else:
                f.fin = INF

    # ---- projection -------------------------------------------------------
    def _projection(self, key) -> dict:
        """(start, end) of ``key`` and every flow linked to it, simulated to the end. Flows that share no
        resource, coupling or stream with them, directly or through other flows, cannot change their rates."""
        coupled: dict = {}
        for trigger, victim, _ in self.couplings:
            coupled.setdefault(trigger, set()).add(victim)
            coupled.setdefault(victim, set()).add(trigger)
        res, streams, linked = set(), {self.flows[key].stream}, set()
        grew = True
        while grew:
            grew = False
            for k, f in self.flows.items():
                if k not in linked and (f.stream in streams or not res.isdisjoint(f.path)):
                    linked.add(k)
                    streams.add(f.stream)
                    for r in f.path:
                        res.add(r)
                        res.update(coupled.get(r, ()))
                    grew = True
        sim = copy.copy(self)
        sim.flows = {k: copy.copy(f) for k, f in self.flows.items() if k in linked}
        sim.streams = {s: deque(q) for s, q in self.streams.items() if s in streams}
        sim.stream_free = {s: self.stream_free[s] for s in streams if s in self.stream_free}
        # (couplings and capacities are shared: a projection never changes them)
        sim.done = {}
        sim.completed = []
        sim._proj = None
        while sim.flows:
            t = sim._next_time()
            if t is None:
                raise RuntimeError("fabric: flows that can never finish (zero bandwidth)")
            sim._process(int(t))
        return sim.done


# ---- topology -------------------------------------------------------------

@dataclass
class Net:
    """One engine's view of the fabric: the paths its transfers take (its first GPU's links)."""

    fabric: Fabric
    engine_id: int
    gpu: int
    host: int
    pd_mode: str = "nic"

    def store_path(self) -> list[str]:
        return [f"gpu{self.gpu}.d2h", f"host{self.host}.pcie_root", f"host{self.host}.mem"]

    def load_path(self) -> list[str]:
        return [f"host{self.host}.mem", f"host{self.host}.pcie_root", f"gpu{self.gpu}.h2d"]

    def tier_write_path(self, tier: str) -> list[str]:
        return [f"host{self.host}.mem", f"host{self.host}.tier.{tier}"]

    def tier_read_path(self, tier: str) -> list[str]:
        return [f"host{self.host}.tier.{tier}", f"host{self.host}.mem"]

    def hbm_resources(self) -> list[str]:
        return [f"gpu{self.gpu}.h2d", f"gpu{self.gpu}.d2h"]

    def interference_rate(self) -> float:
        """Bytes per ns now moving through this GPU's PCIe links."""
        return sum(self.fabric.rates_through(r) for r in self.hbm_resources())


class Topology:
    """The default fabric of a cluster (see ``build``) and each engine's paths in it."""

    def __init__(self, fabric: Fabric, gpu_of: dict[int, int], gpus_per_host: int, pd_mode: str = "nic"):
        self.fabric = fabric
        self.gpu_of = gpu_of
        self.gpus_per_host = gpus_per_host
        self.pd_mode = pd_mode

    @classmethod
    def build(cls, engine_tps: list[tuple[int, int]], gpus_per_host: int = 8, h2d_gbps: float = 25.0,
              d2h_gbps: float | None = None, root_gbps: float | None = None, mem_gbps: float = 256.0,
              tiers: dict[str, float] | None = None, nic_gbps: float = 16.0, net_gbps: float | None = None,
              pd_mode: str = "nic", overrides: dict[str, float] | None = None,
              d2h_duplex_penalty: float = 0.15, host_mem_background_gbps: float = 0.0) -> "Topology":
        """Per host: ``gpu{i}.h2d`` / ``.d2h`` (PCIe, one per direction), ``host{h}.pcie_root`` (default two GPUs'
        links in both directions), ``host{h}.mem``, ``host{h}.tier.{name}`` for each secondary tier; per GPU ``gpu{i}.nic``
        and one ``net`` for PD pulls (GPU-direct RDMA, GPU to GPU: nothing on the host is used) whose
        default is every GPU's NIC at once (non-blocking). With pd_mode "decode-link" every engine
        instead has one link ``pd.{engine}`` of nic_gbps its pulls share (what --pd-contention decode
        models). Engines are packed onto GPUs in order by TP size; an engine's transfers use its first
        GPU's links. ``overrides`` replaces or adds resources by name.

        Calibration (RTX 4060 Ti host, scripts/contention_bench.py, 2026-11-21): one GPU moves 12.7 GB/s H2D and
        13.1-13.2 GB/s D2H whatever the size (one KV block to 64 MB); with both directions at once D2H falls 13.2 ->
        11.1 (-15%, ``d2h_duplex_penalty``) and H2D about 1%; two H2D streams on one GPU split the link evenly; two
        GPUs with an idle host do not slow each other, so ``pcie_root`` and ``mem`` must not bind below about
        40 GB/s for two GPUs (the defaults assume an idle host). With four host processes streaming memory,
        same-direction copies on two GPUs fell to 14 GB/s in all and four directions to 24 GB/s: host memory is
        the resource they share under CPU load; ``host_mem_background_gbps`` is what other work takes of it."""
        if host_mem_background_gbps >= mem_gbps:
            raise ValueError("host memory background load must be below the host memory bandwidth")
        mem_gbps = mem_gbps - host_mem_background_gbps
        d2h = h2d_gbps if d2h_gbps is None else d2h_gbps
        root = 2 * (h2d_gbps + d2h) if root_gbps is None else root_gbps  # two GPUs, both directions
        gpu_of, g = {}, 0
        for eid, tp in engine_tps:
            gpu_of[eid] = g
            g += tp
        n_gpus = max(g, 1)
        res: dict[str, float] = {}
        for h in range((n_gpus + gpus_per_host - 1) // gpus_per_host):
            res[f"host{h}.pcie_root"] = root
            res[f"host{h}.mem"] = mem_gbps
            for name, gbps in (tiers or {}).items():
                res[f"host{h}.tier.{name}"] = gbps
        for i in range(n_gpus):
            res[f"gpu{i}.h2d"] = h2d_gbps
            res[f"gpu{i}.d2h"] = d2h
            res[f"gpu{i}.nic"] = nic_gbps
        res["net"] = nic_gbps * n_gpus if net_gbps is None else net_gbps
        if pd_mode == "decode-link":
            for eid, _ in engine_tps:
                res[f"pd.{eid}"] = nic_gbps
        res.update(overrides or {})
        fabric = Fabric(res)
        if d2h_duplex_penalty:
            for i in range(n_gpus):
                fabric.couple(f"gpu{i}.h2d", f"gpu{i}.d2h", d2h_duplex_penalty)
        return cls(fabric, gpu_of, gpus_per_host, pd_mode)

    @classmethod
    def from_file(cls, path: str, engine_tps: list[tuple[int, int]], **defaults) -> "Topology":
        """A topology file: JSON ``{"gpus_per_host": 8, "pd": "nic" | "decode-link", "resources": {name: GB/s}}``.
        The defaults of ``build`` apply to everything the file does not name."""
        import json
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
        unknown = set(spec) - {"gpus_per_host", "pd", "resources"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        if "gpus_per_host" in spec:
            defaults["gpus_per_host"] = int(spec["gpus_per_host"])
        if "pd" in spec:
            defaults["pd_mode"] = spec["pd"]
        return cls.build(engine_tps, overrides={k: float(v) for k, v in spec.get("resources", {}).items()}, **defaults)

    def net(self, engine_id: int) -> Net:
        g = self.gpu_of[engine_id]
        return Net(self.fabric, engine_id, g, g // self.gpus_per_host, self.pd_mode)

    def pd_path(self, src: int, dst: int) -> list[str]:
        if self.pd_mode == "decode-link":
            return [f"pd.{dst}"]
        return [f"gpu{self.gpu_of[src]}.nic", "net", f"gpu{self.gpu_of[dst]}.nic"]
