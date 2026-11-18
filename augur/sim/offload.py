"""KV offload to CPU memory: the GPU-worker half of vLLM's OffloadingConnector.

vLLM's ``--kv-offloading-size`` runs its OffloadingConnector: the scheduler
half (which blocks to store, CPU prefix hits, which loads to issue, LRU
eviction of the CPU tier) runs unmodified inside vLLM's scheduler here too.
Its worker half moves blocks over PCIe on CUDA streams; ``SimOffloadWorker``
stands in for it with the same rules (vLLM 0.33, v2 model runner,
kv_offload/cpu/gpu_worker.py):

* Three lanes, each FIFO: stores (GPU -> CPU), loads, expedited (sync) loads.
  A transfer starts once the GPU work queued before it is done and the lane
  is free.
* Before the forward: last step's stores are submitted; stores the scheduler
  flushes (a preempted request's, or blocks being reallocated) are waited
  for; if any load is sync, all loads are submitted and the sync ones waited
  for. Waits hold the step.
* After the forward: loads not yet submitted are, behind this forward; this
  step's stores wait for the next step. Then the worker polls: every transfer
  finished by then is reported (job ids to the scheduler; requests whose
  async load finished, as finished_recving).

A transfer takes latency + bytes / bandwidth (``OffloadLink``), the cost
LLMServingSim gives a kv_load / kv_evict row on its CPU memory. Lanes do not
share bandwidth. With ``replay_done_ns`` (job id -> poll time in a real run's
step log) a job finishes when the real one was reported.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec


@dataclass(frozen=True)
class OffloadLink:
    """GPU <-> CPU memory: bandwidth in GB/s (= bytes per ns) and latency per transfer."""

    gbps: float
    latency_ns: int = 0

    def transfer_ns(self, nbytes: int) -> int:
        return self.latency_ns + int(nbytes / self.gbps)


@dataclass
class _Transfer:
    job_id: int
    req_id: str | None  # an async load's request (finished_recving); None for stores and sync loads
    is_load: bool
    nbytes: int
    start_ns: int
    end_ns: int


class _Lane:
    """One CUDA stream of transfers, each starting when the previous ends."""

    def __init__(self) -> None:
        self.queue: deque[_Transfer] = deque()
        self.free_ns = 0

    def finished_by(self, t: int) -> list[_Transfer]:
        out = []
        while self.queue and self.queue[0].end_ns <= t:
            out.append(self.queue.popleft())
        return out


class SimOffloadWorker:
    """Stands in for OffloadingConnectorWorker and its CPU OffloadingWorker.
    Called by: SimWorker.execute_model (pre_forward, post_forward),
    Coordinator (next_end, deliverable). Calls: OffloadLink."""

    def __init__(self, link: OffloadLink, block_bytes: int, replay_done_ns: dict[int, int] | None = None):
        self.link = link
        self.block_bytes = block_bytes
        self.replay_done_ns = replay_done_ns or {}
        self.stores, self.loads, self.sync_loads = _Lane(), _Lane(), _Lane()
        self._deferred_stores: list[tuple[int, TransferJob]] = []
        self._ends: dict[int, int] = {}  # submitted, unreported job -> end
        self._loads_submitted = False

    def pre_forward(self, meta: OffloadingConnectorMetadata, now_ns: int, gpu_free_ns: int) -> int:
        """Hooks before the forward (handle_preemptions, sync start_load_kv).
        Returns when the forward can be launched."""
        flush = meta.jobs_to_flush or set()
        self._deferred_stores += [(jid, job) for jid, job in meta.store_jobs.items() if jid in flush]
        for jid, job in self._deferred_stores:
            self._submit(self.stores, jid, job, None, max(now_ns, gpu_free_ns))
        self._deferred_stores = []
        ready = max([now_ns] + [self._ends[j] for j in flush if j in self._ends])
        self._loads_submitted = any(job.sync for job in meta.load_jobs.values())
        if self._loads_submitted:
            waits = self._submit_loads(meta, max(ready, gpu_free_ns))
            ready = max([ready] + waits)
        return ready

    def post_forward(self, meta: OffloadingConnectorMetadata, forward_end_ns: int, poll_ns: int):
        """Hooks after the forward (async start_load_kv, wait_for_save,
        get_finished). Returns (finished_recving, worker metadata or None)."""
        if not self._loads_submitted:
            self._submit_loads(meta, forward_end_ns)
        flush = meta.jobs_to_flush or set()
        self._deferred_stores += [(jid, job) for jid, job in meta.store_jobs.items() if jid not in flush]
        done = self.stores.finished_by(poll_ns) + self.loads.finished_by(poll_ns) + self.sync_loads.finished_by(poll_ns)
        if not done:
            return set(), None
        out = OffloadingWorkerMetadata()
        finished_recving = set()
        for t in done:
            del self._ends[t.job_id]
            out.mark_completed(t.job_id)
            stats = out.transfer_stats.load if t.is_load else out.transfer_stats.store
            stats.record(t.nbytes, (t.end_ns - t.start_ns) / 1e9)
            if t.req_id is not None:
                finished_recving.add(t.req_id)
        return finished_recving, out

    def next_end(self) -> int | None:
        """When the next submitted transfer finishes (the engine must poll then)."""
        return min(self._ends.values(), default=None)

    def deliverable(self, now_ns: int) -> bool:
        """A transfer has finished and is waiting for the engine's next poll."""
        end = self.next_end()
        return end is not None and end <= now_ns

    def _submit_loads(self, meta: OffloadingConnectorMetadata, after_ns: int) -> list[int]:
        """Submit every load; returns the sync ones' end times (the step waits for them)."""
        waits = []
        for jid, job in meta.load_jobs.items():
            if job.sync:
                waits.append(self._submit(self.sync_loads, jid, job, None, after_ns))
            else:
                self._submit(self.loads, jid, job, job.req_id, after_ns)
        return waits

    def _submit(self, lane: _Lane, jid: int, job: TransferJob, req_id: str | None, after_ns: int) -> int:
        gpu = job.src_spec if isinstance(job.src_spec, GPULoadStoreSpec) else job.dst_spec
        nbytes = len(gpu.block_ids) * self.block_bytes
        start = max(after_ns, lane.free_ns)
        end = start + self.link.transfer_ns(nbytes)
        if jid in self.replay_done_ns:
            end = max(start, self.replay_done_ns[jid])  # earlier only if the replay already diverged
        lane.free_ns = end
        lane.queue.append(_Transfer(jid, req_id, lane is not self.stores, nbytes, start, end))
        self._ends[jid] = end
        return end
