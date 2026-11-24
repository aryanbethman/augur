"""Workload loader: JSONL traces in LLMServingSim's format.

Every line is either a flat request or an agentic session (it has
``sub_requests``); a file may mix them. Semantics follow the fork's router:

* A flat request arrives at ``arrival_time_ns``.
* A session's first call arrives at the session's ``arrival_time_ns``. Call
  ``i+1`` is released at call ``i``'s completion plus call ``i``'s
  ``tool_duration_ns`` (the wait *after* that call).
* ``priority`` (optional, default 0; lower runs first under ``--scheduling-policy priority``) goes to vLLM with
  the request; a session's calls each carry their own.
* ``input_toks`` / ``output_toks`` set the prompt and output lengths.
  ``input_tok_ids`` drive prefix caching only: the prompt's ids are the
  recorded ones, and any tokens past them are filled with ids unique to the
  request, so that tail never hits the cache (the fork leaves it unhashed).
  ``output_tok_ids`` are what the simulated model emits; past them, unique ids.  A call whose outputs
  are all recorded is ``recorded``: ngram speculation then verifies its drafts against them.

The file is streamed: sessions are read in arrival order, and a session's
calls are kept only until the session ends.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import zlib
from dataclasses import dataclass, field
from typing import Iterator

# Filler ids live far above any real vocabulary, each request in its own range.
_FILLER_BASE = 1 << 40
_FILLER_STRIDE = 1 << 24


@dataclass
class Call:
    call_id: str
    session_id: str | None
    index: int  # position in the session (0 for flat requests)
    arrival_ns: int  # set when released
    prompt_ids: list[int]
    output_toks: int
    output_ids: list[int]
    tool_wait_ns: int  # wait after this call before the session's next one
    after_step_start: bool = False  # replay only: reached the engine just after a step began
    priority: int = 0  # vLLM request priority (only --scheduling-policy priority looks at it)
    recorded: bool = False  # output_ids are all recorded ids (none is filler): speculation can verify against them


@dataclass
class Session:
    session_id: str
    arrival_ns: int
    sub_requests: list[dict]
    next_index: int = 0
    shift: int = 0  # --replicate-sessions: added to every recorded token id of this copy (mod REPLICA_VOCAB)


class _Filler:
    """Token ids for the parts of a prompt/output the trace did not record.

    Default: ids far above any vocabulary, each request in its own range
    (never equal to anything, so never a false prefix-cache hit). With
    ``vocab``: pseudo-random ids inside the vocabulary, seeded by the call id
    and part, which a real vLLM server accepts; augur and a real run then
    build identical prompts whatever order calls are released in."""

    def __init__(self, vocab: int | None = None) -> None:
        self._n = itertools.count()
        self.vocab = vocab

    def ids(self, count: int, key: str = "") -> list[int]:
        if self.vocab:
            rng = random.Random(zlib.crc32(key.encode()))
            return [rng.randrange(1000, self.vocab) for _ in range(count)]
        base = _FILLER_BASE + next(self._n) * _FILLER_STRIDE
        return list(range(base, base + count))


@dataclass
class Workload:
    """Streams top-level entries (flat requests and sessions) in file order."""

    path: str
    max_entries: int | None = None
    # False: the simulated model emits ids unique to the request instead of the
    # recorded outputs, as when a real model generates its own text (a later
    # turn then hits the prefix cache on the earlier prompt only).
    recorded_outputs: bool = True
    # Set to (at most) the model's regular vocabulary to keep filler ids valid
    # for a real server, e.g. 150000 for Qwen3 (its special tokens start at 151643).
    filler_vocab: int | None = None
    # Output ids a real run generated (augur.verify.client --tokens-out), by call id: they replace the trace's
    # (or the unique ids) so that calls whose generations coincide share prefix-cache blocks, as they did there.
    output_tokens: str | None = None
    # K > 1 (--replicate-sessions): every session K times, as ``<id>-r<k>``, arriving up to ``replica_jitter_s`` later
    # than the original (copy 0 is the original). Copies k > 0 have every recorded token id shifted by a copy-specific
    # amount, so they share no prefix with each other while keeping the original's sharing inside a copy. A copy
    # holds the original's records, not its own: a cluster of hundreds of engines costs no more memory per session.
    replicas: int = 1
    replica_jitter_s: float = 5.0
    _filler: _Filler = field(default_factory=_Filler)
    _generated: dict = field(default_factory=dict)

    def __post_init__(self):
        self._filler = _Filler(self.filler_vocab)
        if self.output_tokens:
            with open(self.output_tokens) as f:
                self._generated = {r["call_id"]: r["output_tok_ids"] for r in map(json.loads, f)}

    def _replicated(self) -> Iterator[Session]:
        rng = random.Random(0)
        with open(self.path) as f:
            base = [(i, json.loads(line)) for i, line in enumerate(f) if line.strip()]
        copies = []
        for k in range(self.replicas):
            for line_no, d in base:
                if "sub_requests" not in d:
                    raise ValueError("--replicate-sessions replicates sessions; the workload has flat requests")
                if not d["sub_requests"]:
                    continue
                jitter = int(rng.random() * self.replica_jitter_s * 1e9) if k else 0
                copies.append((int(d["arrival_time_ns"]) + jitter, k, line_no, d))
        copies.sort(key=lambda c: c[:3])
        for n, (arrival, k, line_no, d) in enumerate(copies):
            if self.max_entries is not None and n >= self.max_entries:
                return
            sid = str(d.get("session_id", f"session_{line_no}"))
            yield Session(session_id=f"{sid}-r{k}", arrival_ns=arrival, sub_requests=d["sub_requests"],
                          shift=REPLICA_STRIDE * k)

    def entries(self) -> Iterator[Call | Session]:
        if self.replicas > 1:
            yield from self._replicated()
            return
        with open(self.path) as f:
            n = 0
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if self.max_entries is not None and n >= self.max_entries:
                    return
                d = json.loads(line)
                if "sub_requests" in d:
                    subs = d["sub_requests"]
                    if not subs:  # the fork skips empty sessions silently
                        continue
                    sid = str(d.get("session_id", f"session_{line_no}"))
                    yield Session(session_id=sid, arrival_ns=int(d["arrival_time_ns"]), sub_requests=subs)
                else:
                    yield self.make_call(d, call_id=f"req_{line_no}", session_id=None, index=0,
                                         arrival_ns=int(d["arrival_time_ns"]))
                n += 1

    def make_call(self, d: dict, call_id: str, session_id: str | None, index: int, arrival_ns: int,
                  shift: int = 0) -> Call:
        n_in = int(d["input_toks"])
        n_out = int(d["output_toks"])
        rec_in = list(d.get("input_tok_ids") or ())[:n_in]
        if shift:
            rec_in = [(t + shift) % REPLICA_VOCAB for t in rec_in]
        if len(rec_in) < n_in:
            rec_in += self._filler.ids(n_in - len(rec_in), call_id + "/in")
        rec_out = list(d.get("output_tok_ids") or ())[:n_out] if self.recorded_outputs else []
        if shift:
            rec_out = [(t + shift) % REPLICA_VOCAB for t in rec_out]
        rec_out = self._generated.get(call_id, rec_out)[:n_out]
        recorded = n_out > 0 and len(rec_out) == n_out
        if len(rec_out) < n_out:
            rec_out += self._filler.ids(n_out - len(rec_out), call_id + "/out")
        return Call(call_id=call_id, session_id=session_id, index=index, arrival_ns=arrival_ns,
                    prompt_ids=rec_in, output_toks=n_out, output_ids=rec_out, recorded=recorded,
                    tool_wait_ns=int(d.get("tool_duration_ns", 0) or 0), priority=int(d.get("priority", 0) or 0))

    def next_call(self, s: Session, arrival_ns: int) -> Call:
        i = s.next_index
        s.next_index += 1
        return self.make_call(s.sub_requests[i], call_id=f"{s.session_id}#{i}", session_id=s.session_id,
                              index=i, arrival_ns=arrival_ns, shift=s.shift)


REPLICA_VOCAB = 151643  # Qwen's regular vocabulary: a shifted id stays below the special tokens
REPLICA_STRIDE = 104729  # a prime, so two copies' ids differ by more than a small offset


_BENCH_RID = re.compile(r"^(?:cmpl-)?bench-[0-9a-f]{8}-(\d+)-")


def strip_pd_suffix(rid: str) -> str:
    """The call id of a PD half (<call>@p, <call>@d). A session id may itself
    contain "@". Called by: call_id, Coordinator._cid."""
    return rid[:-2] if rid.endswith(("@p", "@d")) else rid


def call_id(rid: str) -> str:
    """The workload call id inside a request id.

    augur's trace client sends ``augur.<call_id>.``, which real vLLM wraps as
    ``cmpl-augur.<call_id>.-0`` (plus any internal suffix). Inside augur the
    request id is the call id, with ``@p`` / ``@d`` for PD halves."""
    if "augur." in rid:
        return rid.split("augur.", 1)[1].split(".", 1)[0]
    m = _BENCH_RID.match(rid)
    if m:  # `vllm bench serve`: request i of the benchmark is augur's req_<i>
        return f"req_{m.group(1)}"
    return strip_pd_suffix(rid)
