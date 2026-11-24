"""The one step-price function every simulator in the same-step-time test uses.

A step's price is augur's own: ``TableCost`` of the profile shipped for the
hardware and model, on the batch padded to vLLM's CUDA-graph size (vLLM's own
``CudagraphDispatcher``, as ``augur.sim.executor`` and ``augur.verify.stepcheck``
do), plus augur's per-step CPU term (``--step-cpu-us``, ``--step-cpu-per-request-us``).
A batch is a list of ``(num_new, num_computed_before, is_decode)``, one per
scheduled request.

In process::

    p = Pricer(hardware="RTX4060Ti", model="Qwen/Qwen3-1.7B", max_model_len=8192)
    p.gpu_ns([(2048, 0, False), (1, 900, True)])   # GPU time, ns
    p.cpu_ns(2)                                    # the CPU term, ns

As a server for a Go (or any) simulator, one JSON object per line on stdin, one
answer per line on stdout::

    {"reqs": [[2048, 0, 0], [1, 900, 1]], "spec_k": 0}  ->  {"gpu_ns": 18234567, "cpu_ns": 2000000}

    PYTHONPATH=<worktree> ~/augur/.venv/bin/python scripts/baselines/pricer.py serve \\
        --max-model-len 8192 --step-cpu-us 2000

With ``--durations <real step log>`` the server prices nothing: step k takes the duration step k took in
the real run (``augur.verify.replay.ReplayCost``, as ``augur --replay-durations``), the replay in which an
exact copy of vLLM's scheduler reproduces the real steps.

Check that the price equals ``augur.verify.stepcheck``'s predicted ns on every
logged step of a real run (through the server protocol, as the Go side calls it)::

    PYTHONPATH=<worktree> ~/augur/.venv/bin/python scripts/baselines/pricer.py check \\
        out/real/sharegpt300/steps.<pid>.jsonl --max-model-len 8192
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile


class Pricer:
    """GPU price of a batch (augur's tables) and the per-step CPU term."""

    def __init__(self, hardware: str = "RTX4060Ti", model: str = "Qwen/Qwen3-1.7B", max_num_seqs: int = 128,
                 max_num_batched_tokens: int = 2048, max_model_len: int = 8192, step_cpu_us: float = 2000.0,
                 step_cpu_per_request_us: float = 0.0, spec_us_per_request: float = 180.0,
                 fork: str | None = None, class_scales: dict | None = None) -> None:
        from augur.cost.profiles import default_fork
        from augur.cost.tables import bundled_profile
        from augur.verify.stepcheck import make_pricer

        bundled_profile(hardware, model)  # raises without a shipped profile (make_pricer would use the fork's)
        self.cost, self.cudagraph = make_pricer(model, hardware, fork or default_fork(), max_num_seqs,
                                                max_num_batched_tokens, max_model_len)
        self.cpu_fixed_ns = int(step_cpu_us * 1e3)  # as augur.cli: --step-cpu-us
        self.cpu_per_request_ns = int(step_cpu_per_request_us * 1e3)
        self.spec_ns = int(spec_us_per_request * 1e3)  # as augur: per request verified with drafts
        self.class_scales = class_scales or {}  # as augur --step-class-scales: GPU time of a decode, prefill or mixed step

    def gpu_ns(self, reqs, spec_k: int = 0) -> int:
        """Price of one step. ``spec_k``: speculative tokens per decode (the executor's uniform-decode query length)."""
        from augur.cost import SeqWork, StepShape, adjust_gpu_ns

        seqs = tuple(SeqWork(num_new=int(n), num_computed_before=int(c), is_decode=bool(d)) for n, c, d in reqs)
        if not seqs:
            return 0
        total = sum(s.num_new for s in seqs)
        uniform = all(s.num_new == 1 + spec_k for s in seqs)
        _, desc = self.cudagraph.dispatch(num_tokens=total, uniform_decode=uniform)
        shape = StepShape(seqs=seqs, padded_tokens=desc.num_tokens)
        return adjust_gpu_ns(self.cost.step_ns(shape), shape, class_scales=self.class_scales,
                             spec_ns_per_request=self.spec_ns)

    def cpu_ns(self, n_reqs: int) -> int:
        return (self.cpu_fixed_ns + self.cpu_per_request_ns * n_reqs) if n_reqs else 0


class Client:
    """The server above as a function call, for a simulator in another Python environment (stdlib only).

        c = Client(python="~/augur/.venv/bin/python", args=["--max-model-len", "8192"], pythonpath="<worktree>")
        gpu_ns, cpu_ns = c.price([(2048, 0, 0), (1, 900, 1)])
    """

    def __init__(self, python: str | None = None, args: list[str] | None = None, pythonpath: str | None = None) -> None:
        python = os.path.expanduser(python or os.environ.get("AUGUR_PYTHON", "~/augur/.venv/bin/python"))
        env = dict(os.environ, VLLM_LOGGING_LEVEL="ERROR")
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
        self.proc = subprocess.Popen([python, os.path.abspath(__file__), "serve", *(args or [])], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, env=env)
        self.proc.stdout.readline()

    def price(self, reqs, spec_k: int = 0) -> tuple[int, int]:
        self.proc.stdin.write(json.dumps({"reqs": [list(map(int, r)) for r in reqs], "spec_k": spec_k}) + "\n")
        self.proc.stdin.flush()
        m = json.loads(self.proc.stdout.readline())
        return m["gpu_ns"], m["cpu_ns"]

    def close(self) -> None:
        self.proc.stdin.write('{"quit": 1}\n')
        self.proc.stdin.flush()
        self.proc.wait()


def _add_args(p) -> None:
    p.add_argument("--hardware", default="RTX4060Ti")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--step-cpu-us", type=float, default=2000.0)
    p.add_argument("--step-cpu-per-request-us", type=float, default=0.0)
    p.add_argument("--spec-us-per-request", type=float, default=180.0)  # augur.cli's default
    p.add_argument("--step-class-scales", default=None, help="augur --step-class-scales: a calibrate.py JSON or decode=1.01,...")


def _make(a) -> Pricer:
    from augur.cost import parse_class_scales
    return Pricer(a.hardware, a.model, a.max_num_seqs, a.max_num_batched_tokens, a.max_model_len, a.step_cpu_us,
                  a.step_cpu_per_request_us, a.spec_us_per_request, class_scales=parse_class_scales(a.step_class_scales))


def serve(a) -> int:
    import logging
    logging.disable(logging.CRITICAL)
    out = sys.stdout
    if a.durations:
        # Replay: step k takes what step k took in the real run (as augur --replay-durations), nothing is priced.
        from augur.verify.replay import ReplayCost
        durations, p = ReplayCost.from_log(a.durations).durations, None
    else:
        p = _make(a)
    out.write('{"ready": 1}\n')
    out.flush()
    k = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get("quit"):
            break
        reqs = m["reqs"]
        if p is None:
            # past the log's last step (the simulator took more steps than the real run): that step's duration again
            ans = {"gpu_ns": durations[min(k, len(durations) - 1)], "cpu_ns": 0}
            k += 1
        else:
            ans = {"gpu_ns": p.gpu_ns(reqs, m.get("spec_k", 0)), "cpu_ns": p.cpu_ns(len(reqs))}
        out.write(json.dumps(ans) + "\n")
        out.flush()
    return 0


def check(a) -> int:
    """Price every logged step through the server and compare with stepcheck's predicted_ns."""
    from augur.verify import stepcheck
    from augur.verify.replay import read_log

    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, "stepcheck.csv")
        argv = [a.log, "--model", a.model, "--hardware", a.hardware, "--max-num-seqs", str(a.max_num_seqs),
                "--max-num-batched-tokens", str(a.max_num_batched_tokens), "--max-model-len", str(a.max_model_len),
                "--csv", csv]
        stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
        try:
            rc = stepcheck.main(argv)
        finally:
            sys.stdout = stdout
        if rc:
            return rc
        predicted = [int(l.split(",")[6]) for l in open(csv).read().splitlines()[1:]]
    shapes = [s for s, _ in stepcheck.real_steps(read_log(a.log))]
    cmd = [sys.executable, os.path.abspath(__file__), "serve", "--model", a.model, "--hardware", a.hardware,
           "--max-num-seqs", str(a.max_num_seqs), "--max-num-batched-tokens", str(a.max_num_batched_tokens),
           "--max-model-len", str(a.max_model_len)]
    srv = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=os.environ)
    srv.stdout.readline()
    bad = 0
    for sh, want in zip(shapes, predicted):
        srv.stdin.write(json.dumps({"reqs": [[s.num_new, s.num_computed_before, int(s.is_decode)] for s in sh.seqs]}) + "\n")
        srv.stdin.flush()
        got = json.loads(srv.stdout.readline())["gpu_ns"]
        bad += got != want
    srv.stdin.write('{"quit": 1}\n')
    srv.stdin.flush()
    srv.wait()
    print(json.dumps({"log": a.log, "steps": len(shapes), "stepcheck_rows": len(predicted), "mismatches": bad,
                      "identical": bad == 0 and len(shapes) == len(predicted)}))
    return 0 if bad == 0 and len(shapes) == len(predicted) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pricer")
    sub = p.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="JSON-lines price server")
    _add_args(sv)
    sv.add_argument("--durations", default=None, help="price step k with the duration of step k in this real step log")
    c = sub.add_parser("check", help="compare with augur.verify.stepcheck on a real step log")
    c.add_argument("log")
    _add_args(c)
    a = p.parse_args(argv)
    return serve(a) if a.cmd == "serve" else check(a)


if __name__ == "__main__":
    sys.exit(main())
