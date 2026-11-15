"""ASTRA-Sim as an oracle for communication and memory time.

LLMServingSim prices a step by running its trace through ASTRA-Sim. Compute
rows are plain delays, executed in order; collectives and remote-memory reads
take whatever ASTRA-Sim's network and memory backends say. To reuse that model
exactly without running every step through ASTRA-Sim, augur splits a step:

    step = sum(compute latencies) + ASTRA(the same trace with compute set to 0)

The second term depends only on the collective and memory rows (types and
sizes), so it is cached on them and ASTRA-Sim runs once per distinct pattern.
``validate`` checks the split against a full ASTRA run of the same trace.

Each query starts a fresh ASTRA-Sim process with the fork's own config builder,
graph generator and controller protocol (legacy, file-based), so the network,
system and memory configs are exactly those LLMServingSim would use.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import sys
import tempfile


class AstraOracle:
    def __init__(self, fork_root: str, cluster_config: str, workdir: str | None = None):
        self.fork_root = os.path.abspath(fork_root)
        self.run_dir = os.path.join(self.fork_root, "serving")
        if self.fork_root not in sys.path:
            sys.path.insert(0, self.fork_root)
        self.astra_sim = os.path.join(self.fork_root, "astra-sim")
        self.workdir = os.path.abspath(workdir or tempfile.mkdtemp(prefix="augur-astra-"))
        cluster_config = os.path.abspath(cluster_config)
        cwd = os.getcwd()
        os.chdir(self.run_dir)
        try:
            from serving.core.config_builder import build_cluster_config
            from serving.core.controller import Controller
            from serving.core.graph_generator import generate_graph
            from serving.core.trace_generator import TraceData, generate_event
            from serving.core.utils import get_workload
            self._Controller = Controller
            self._generate_graph = generate_graph
            self._generate_event = generate_event
            self._get_workload = get_workload
            self._TraceData = TraceData
            self.cluster = build_cluster_config(self.astra_sim, cluster_config,
                                                False, False, inputs_root=self.workdir)
        finally:
            os.chdir(cwd)
        self.binary = os.path.join(self.astra_sim, "build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra")
        self.total_npu = self.cluster["total_npu"]
        self._ids = itertools.count(1)
        self.cache: dict[tuple, int] = {}
        self.runs = 0
        self._event_wl = None

    @staticmethod
    def comm_key(rows) -> tuple:
        """What the comm/memory time depends on: the collectives, the tensors in
        CPU ("REMOTE") memory, and the row structure. LOCAL tensor sizes do not
        change ASTRA-Sim's time (checked: equal comm time at very different KV
        lengths), so per-step attention sizes stay out of the key."""
        key = [len(rows)]
        for i, r in enumerate(rows):
            if len(r) != 11:
                key.append((i, r))
                continue
            if r[8] != "NONE":
                key.append((i, r[8], r[9]))
            for loc, size in ((r[2], r[3]), (r[4], r[5]), (r[6], r[7])):
                if not loc.startswith("LOCAL"):
                    key.append((i, loc, size))
        return tuple(key)

    def comm_ns(self, rows, header_line: str, model: str, hardware: str, num_npus: int) -> int:
        key = (self.comm_key(rows), header_line, num_npus)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        # The converter drops zero-latency compute nodes, so each compute row
        # gets 1 ns, removed again below (compute nodes run serially).
        unit = [(r[0], "1", *r[2:]) if len(r) == 11 else r for r in rows]
        n_comp = sum(1 for r in rows if len(r) == 11)
        ns = self.run(unit, header_line, model, hardware, num_npus) - n_comp
        self.cache[key] = ns
        return ns

    def run(self, rows, header_line: str, model: str, hardware: str, num_npus: int) -> int:
        """Run one trace on NPUs 0..num_npus-1 (instance 0) and return its duration in ns."""
        return self.run_group([(rows, header_line, 0)], model, hardware)[0]

    def run_group(self, members, model: str, hardware: str, first_member: int = 0) -> list[int]:
        """Run one trace per instance together (a DP round). Returns, per
        member, when its last NPU finished, measured from the round's first
        handover. ``members``: (rows, header_line, instance_id).
        As in LLMServingSim's DP path, all members' graphs share one workload
        folder; each NPU picks its own rank's graph."""
        self.runs += 1
        cwd = os.getcwd()
        os.chdir(self.run_dir)
        try:
            return self._run(members, model, hardware, first_member)
        finally:
            os.chdir(cwd)

    def _run(self, members, model, hardware, first_member=0):
        inputs = self.workdir
        if self._event_wl is None:  # the start-up graph never changes: build it once
            ev = self._generate_event(1, inputs_root=inputs)
            self._generate_graph(None, None, self.total_npu, event=True, inputs_root=inputs, trace=ev)
            self._event_wl = self._get_workload(None, None, event=True, inputs_root=inputs)
        event_wl = self._event_wl

        class _B:  # what generate_graph / get_workload read from a batch
            pass
        insts = self.cluster["instances"]
        npu_of = self.cluster["inst2npu_mapping"]
        node_of = self.cluster["inst2node_mapping"]
        round_id = next(self._ids)
        shared = f"{hardware}/{model}/augur_round{round_id}" if len(members) > 1 else None
        wl_of_npu: dict[int, str] = {}
        npus_of_member: list[range] = []
        first_npu = None
        for rows, header_line, inst_id in members:
            b = _B()
            b.batch_id = round_id
            b.model = model
            b.workload_name = shared
            n = insts[inst_id]["num_npus"]
            off = npu_of[inst_id]
            trace = self._TraceData(header_line=header_line, rows=list(rows),
                                    path=os.path.join(inputs, "trace", hardware, model,
                                                      f"instance{inst_id}_batch{round_id}.txt"))
            self._generate_graph(b, hardware, n, node_of[inst_id], inst_id, off, workload_name=shared,
                                 inputs_root=inputs, trace=trace)
            wl = self._get_workload(b, hardware, inst_id, workload_name=shared, inputs_root=inputs)
            # ASTRA-Sim polls only an instance's first and last NPU (the
            # --start/--end-npu-ids it is given); the NPUs between them never
            # report. Hand the workload to and wait on those two only, as
            # LLMServingSim's loop does. (Waiting on all NPUs hung at TP4.)
            reporting = sorted({off, off + n - 1})
            for npu in reporting:
                wl_of_npu[npu] = wl
            npus_of_member.append(reporting)
            if len(npus_of_member) - 1 == first_member:
                first_npu = off
        num_npus = len(wl_of_npu)

        args = [self.binary, "--workload-configuration=" + event_wl,
                "--system-configuration=" + os.path.join(inputs, "system", "system.json"),
                "--network-configuration=" + os.path.join(inputs, "network", "network.yml"),
                "--memory-configuration=" + os.path.join(inputs, "memory", "memory_expansion.json")]
        if self.cluster.get("start_npu_ids") != "":
            args.append("--start-npu-ids=" + self.cluster["start_npu_ids"])
        if self.cluster.get("end_npu_ids") != "":
            args.append("--end-npu-ids=" + self.cluster["end_npu_ids"])
        if os.environ.get("AUGUR_ASTRA_DEBUG"):
            print("ASTRA args:", " ".join(args), file=sys.stderr, flush=True)
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        ctl = self._Controller(self.total_npu)
        given: dict[int, int] = {}  # npu -> cycle when our workload was handed over
        done: dict[int, int] = {}
        debug = os.environ.get("AUGUR_ASTRA_DEBUG")
        try:
            for _ in range(100_000):
                out = ctl.read_wait(p)
                rec = ctl.parse_output(out[-2])
                if debug:
                    print("ASTRA>", [l.rstrip() for l in out[-3:]], "->", rec, file=sys.stderr)
                if rec is None:
                    ctl.write_flush(p, "pass")
                    continue
                npu, cycle = rec["sys"], rec["cycle"]
                if npu in wl_of_npu and npu not in given:
                    # LLMServingSim hands a batch to the instance's first NPU
                    # first; the others pass until then. The order matters:
                    # the NPUs share one CPU-memory device.
                    if npu != first_npu and first_npu not in given:
                        ctl.write_flush(p, "pass")
                        continue
                    given[npu] = cycle
                    ctl.write_flush(p, wl_of_npu[npu])
                    continue
                if npu in given and npu not in done and rec["id"] > 0:
                    done[npu] = cycle
                if len(done) == num_npus:
                    ctl.write_flush(p, "exit")
                    break
                ctl.write_flush(p, "done")
            else:
                raise RuntimeError("ASTRA oracle: no completion after 100k exchanges")
        finally:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        if os.environ.get("AUGUR_ASTRA_DEBUG"):
            print("ASTRA given", given, "done", done, file=sys.stderr, flush=True)
        # Each member's step runs from the round's first handover to its own
        # last NPU finishing.
        t0 = min(given.values())
        return [max(done[n] for n in npus) - t0 for npus in npus_of_member]

    def close(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)
