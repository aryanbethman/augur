"""Tiered KV offload: simulated secondary tiers under vLLM's TieringOffloadingManager.

vLLM's tiering (kv_offload/tiering) puts secondary tiers behind the CPU
tier: chunks stored to the CPU cascade to every secondary tier, and a chunk
found only in a secondary tier is promoted back to the CPU before it can be
loaded to the GPU. All of that runs unmodified here; augur supplies

* ``SimMemoryTier``: a secondary tier of memory (CXL, a remote DRAM pool)
  with a capacity, LRU eviction, and transfers costing latency + bytes /
  bandwidth on two FIFO lanes (writes to the tier, reads from it), timed on
  the virtual clock. A chunk is readable once its write completes; chunks
  being read are not evicted.
* ``SimTieringOffloadingSpec``: vLLM's TieringOffloadingSpec with the CPU
  tier's buffer as an anonymous private mapping, not a /dev/shm file (a
  simulator never touches it, and many engines' tiers would not fit
  /dev/shm's free-space check).

Tier calls happen inside an engine step; ``tier_clock_ns`` on the engine's
SimContext says when on the virtual clock (the step's start while it
schedules, the step's outputs landing while vLLM applies them).
"""

from __future__ import annotations

import contextlib
import mmap
import weakref
from collections import Counter, OrderedDict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from typing_extensions import override
from vllm.v1.kv_offload.base import LookupResult, Medium, OffloadKey, ReqContext, RequestOffloadingContext
from vllm.v1.kv_offload.tiering import spec as tiering_spec
from vllm.v1.kv_offload.tiering.base import JobResult, SecondaryTierManager, TransferJob
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

from augur.sim.offload import OffloadLink

GIB = 1 << 30
_MAP_NORESERVE = getattr(mmap, "MAP_NORESERVE", 0x4000)  # Linux value; Python's mmap module does not export it

# Engine clocks by key: the tier config carries the key (it must be plain data).
_clocks: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()


def register_clock(key: str, clock: Any) -> None:
    """Make an engine's clock (an object with ``tier_clock_ns`` and a ``tiers``
    list, its SimContext) available to the tiers its config names it in.
    Called by: SimEngine.__init__."""
    _clocks[key] = clock


# Named tier presets: ``--kv-offload-tier NAME:GIB`` takes (GB/s, latency ns) from here.
# "cxl" is LLMServingSim's CXL memory (cluster config key cxl_mem), per device:
#   configs/cluster/single_node_cxl_instance.json   "cxl_mem": {"mem_size": 1024, "mem_latency": 250,
#                                                              "mem_bw": 60, "num_devices": 4}
#   docs/docs/examples/memory-tiers/cxl-memory.mdx  documents mem_bw as GB/s per device and mem_latency as ns
# (the fork's config_builder defaults num_devices to 1; its example uses 4, which adds up to 240 GB/s, so give
# that as --kv-offload-tier cxl:GIB:240:250). Those are the fork's example values, not a measured CXL device.
TIER_PRESETS = {"cxl": (60.0, 250)}


class _NoPreset(ValueError):
    pass


@dataclass(frozen=True)
class TierSpec:
    """One simulated secondary tier, as ``--kv-offload-tier NAME:GIB[:GBPS[:LATENCY_NS]]`` gives it
    (NAME:GIB alone needs NAME in TIER_PRESETS)."""

    name: str
    capacity_bytes: int
    gbps: float
    latency_ns: int = 0

    @classmethod
    def parse(cls, text: str) -> "TierSpec":
        """Called by: cli.main."""
        parts = text.split(":")
        try:
            if len(parts) not in (2, 3, 4) or not parts[0]:
                raise ValueError
            if len(parts) == 2:
                if parts[0] not in TIER_PRESETS:
                    raise _NoPreset
                gib = float(parts[1])
                gbps, latency = TIER_PRESETS[parts[0]]
            else:
                gib, gbps = float(parts[1]), float(parts[2])
                latency = int(parts[3]) if len(parts) == 4 else 0
            if gib <= 0 or gbps <= 0 or latency < 0:
                raise ValueError
        except _NoPreset:
            raise ValueError(f"tier {text!r}: {parts[0]!r} has no preset (presets: {', '.join(TIER_PRESETS)}); "
                             "give NAME:GIB:GBPS[:LATENCY_NS]") from None
        except ValueError:
            raise ValueError(f"tier {text!r}: expected NAME:GIB[:GBPS[:LATENCY_NS]] (name, capacity in GiB > 0, "
                             "bandwidth in GB/s > 0, latency in ns >= 0; GBPS and LATENCY_NS only "
                             "for a name without a preset)") from None
        return cls(parts[0], int(gib * GIB), gbps, latency)

    def tier_config(self, clock_key: str) -> dict:
        """The entry for vLLM's ``secondary_tiers`` list. Called by: SimEngine."""
        return {"type": SimMemoryTier.__name__, "module_path": __name__, "name": self.name,
                "capacity_bytes": self.capacity_bytes, "gbps": self.gbps, "latency_ns": self.latency_ns,
                "clock": clock_key}


@dataclass
class _Job:
    job_id: int
    is_load: bool
    keys: list  # stored (store) or found (load)
    missing: list  # keys a load did not find, or a store could not place
    nbytes: int
    start_ns: int
    end_ns: int


class _Lane:
    """One direction of the tier's link: transfers run one after another."""

    def __init__(self) -> None:
        self.free_ns = 0


class SimMemoryTier(SecondaryTierManager):
    """A secondary tier of memory, simulated. Called by: vLLM's
    TieringOffloadingManager (lookup, submit_store, submit_load,
    get_finished_jobs, touch, drain_jobs), Coordinator (next_end,
    deliverable). Calls: OffloadLink."""

    medium: ClassVar[Medium] = Medium.CPU

    def __init__(self, offloading_spec, primary_kv_view: memoryview, tier_type: str, name: str,
                 capacity_bytes: int, gbps: float, latency_ns: int, clock: str,
                 backpressure_detector=None) -> None:
        super().__init__(offloading_spec=offloading_spec, primary_kv_view=primary_kv_view, tier_type=tier_type,
                         backpressure_detector=backpressure_detector)
        self.name = name
        self.link = OffloadLink(gbps=gbps, latency_ns=latency_ns)
        self.capacity = capacity_bytes // self.block_size_bytes  # chunks
        self._clock = _clocks[clock]
        self._clock.tiers.append(self)
        self._present: OrderedDict[OffloadKey, None] = OrderedDict()  # readable, least recently used first
        self._writing: set[OffloadKey] = set()  # slots taken by writes in flight
        self._reading: Counter = Counter()  # keys pinned by reads in flight
        self._writes, self._reads = _Lane(), _Lane()
        self._jobs: list[_Job] = []

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return LookupResult.HIT if key in self._present else LookupResult.MISS

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        now = self._clock.tier_clock_ns
        new = list(dict.fromkeys(k for k in job_metadata.keys if k not in self._present and k not in self._writing))
        self._make_room(len(new))
        room = max(self.capacity - len(self._present) - len(self._writing), 0)
        stored, dropped = new[:room], new[room:]
        self._writing.update(stored)
        self._add_job(self._writes, job_metadata.job_id, False, stored, dropped, now)

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        now = self._clock.tier_clock_ns
        found = [k for k in job_metadata.keys if k in self._present]
        missing = [k for k in job_metadata.keys if k not in self._present]
        for k in found:
            self._reading[k] += 1
            self._present.move_to_end(k)
        self._add_job(self._reads, job_metadata.job_id, True, found, missing, now)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        now = self._clock.tier_clock_ns
        done = sorted((j for j in self._jobs if j.end_ns <= now), key=lambda j: (j.end_ns, j.job_id))
        self._jobs = [j for j in self._jobs if j.end_ns > now]
        return [self._finish(j) for j in done]

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        for k in keys:
            if k in self._present:
                self._present.move_to_end(k)

    @override
    def has_pending_work(self) -> bool:
        return bool(self._jobs)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def drain_jobs(self) -> None:
        """Every job in flight completes now (there is no real I/O to wait for)."""
        now = self._clock.tier_clock_ns
        for j in self._jobs:
            j.end_ns = min(j.end_ns, now)

    def next_end(self) -> int | None:
        """When the next job in flight completes: vLLM polls for it at its next step."""
        return min((j.end_ns for j in self._jobs), default=None)

    def deliverable(self, now_ns: int) -> bool:
        end = self.next_end()
        return end is not None and end <= now_ns

    def _make_room(self, needed: int) -> None:
        """Evict least recently used chunks, never ones being read, until
        ``needed`` more fit (or nothing more can go)."""
        excess = len(self._present) + len(self._writing) + needed - self.capacity
        for k in list(self._present):
            if excess <= 0:
                return
            if self._reading[k]:
                continue
            del self._present[k]
            excess -= 1

    def _add_job(self, lane: _Lane, job_id: int, is_load: bool, keys: list, missing: list, now: int) -> None:
        nbytes = len(keys) * self.block_size_bytes
        start = end = now
        if keys:  # nothing to move costs nothing and does not hold the lane
            start = max(now, lane.free_ns)
            end = lane.free_ns = start + self.link.transfer_ns(nbytes)
        self._jobs.append(_Job(job_id, is_load, keys, missing, nbytes, start, end))

    def _finish(self, j: _Job) -> JobResult:
        if j.is_load:
            for k in j.keys:
                self._reading[k] -= 1
                if not self._reading[k]:
                    del self._reading[k]
        else:
            for k in j.keys:
                self._writing.discard(k)
                self._present[k] = None
        partial = j.is_load and j.missing and j.keys
        return JobResult(job_id=j.job_id, success=not j.missing, successful_keys=list(j.keys) if partial else None,
                         transfer_time=(j.end_ns - j.start_ns) / 1e9, transfer_bytes=j.nbytes)


class _AnonymousRegion:
    """Stands in for vLLM's SharedOffloadRegion on the scheduler side: the CPU
    tier's buffer as a private anonymous mapping. Pages are only allocated if
    written, and augur never writes them. Called by: TieringOffloadingSpec.get_manager
    (through SimTieringOffloadingSpec), CPUPrimaryTierOffloadingManager."""

    def __init__(self, engine_id: str, num_chunks: int, rank: int | None, kv_bytes_per_chunk: int,
                 cpu_page_size: int, **_: Any) -> None:
        self.num_chunks = num_chunks
        self._row = kv_bytes_per_chunk
        self._mmap: mmap.mmap | None = mmap.mmap(-1, max(num_chunks * kv_bytes_per_chunk, mmap.PAGESIZE),
                                                 flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_NORESERVE,
                                                 prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self._array: np.ndarray | None = np.frombuffer(self._mmap, dtype=np.int8,
                                                       count=num_chunks * kv_bytes_per_chunk)

    def create_kv_memoryview(self) -> memoryview:
        return memoryview(self._array.reshape(self.num_chunks, self._row))

    def cleanup(self) -> None:
        self._array = None
        if self._mmap is not None:
            with contextlib.suppress(BufferError):  # a view someone still holds keeps it mapped until freed
                self._mmap.close()
            self._mmap = None


class SimTieringOffloadingSpec(TieringOffloadingSpec):
    """vLLM's TieringOffloadingSpec with the CPU tier's buffer from
    ``_AnonymousRegion``. Called by: vLLM's OffloadingSpecFactory
    (``spec_name``/``spec_module_path`` in the connector's extra config)."""

    @override
    def get_manager(self):
        original = tiering_spec.SharedOffloadRegion
        tiering_spec.SharedOffloadRegion = _AnonymousRegion
        try:
            return super().get_manager()
        finally:
            tiering_spec.SharedOffloadRegion = original
