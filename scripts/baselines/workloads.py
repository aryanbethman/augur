"""Turn one real vLLM run into every simulator's input for the same-step-time test.

``prepare`` reads the run's step log (augur.verify.hooks format), its server.log and
the workload file the run replayed, and writes ``<out>/requests.json``: one row per
call, in arrival order, with

* ``arrival_s``: when the call reached the engine in the real run (the step log's ``add`` event,
  seconds after the log's first step starts, never before 0). The replay rows of every simulator take
  these. No simulator models a front end (HTTP, tokenizer);
* ``arrival_trace_s``, ``tool_wait_s``, ``frontend_s``, ``delivery_s``: augur's normal arrival process: a flat
  request or a session's first call is sent at the trace's time and reaches the engine a front end
  (3.4 ms + 3 us per prompt token) later; a session's next call is sent the previous call's tool wait plus
  8 ms (delivery) after that call completes, and reaches the engine a front end later;
* ``arrival_aug_s`` (``arrivals``, after augur's normal run): when each call reached the engine in that run.
  Vidur and Frontier, which cannot loop a session through their own completions, and BLIS's flat requests take
  these; BLIS loops sessions itself (``think_time_us``);
* ``input``, ``output``: prompt and output token counts;
* ``blocks``: the prompt's full 16-token blocks as chained hashes (vLLM's prefix-cache
  keys), from the workload's token ids;
* ``prefix_group``, ``prefix_len``: the same sharing, as BLIS's trace states it (below);
* ``priority``: the call's vLLM priority (the workload's ``priority``, 0 when it has none);
* ``real``: first-token, finish and output-token numbers from the step log. Times are the
  engine's: a request's first token is the update of the step that emitted it.

and ``<out>/meta.json`` (engine settings from server.log: async or sync, speculative decoding,
KV blocks, max-model-len, KV offload, the measured n-gram acceptance, the real prefix hit rate
and how well each simulator's prefix mapping reproduces the token-id sharing).

Then one subcommand per simulator writes that simulator's input from requests.json::

    python scripts/baselines/workloads.py prepare out/real/swe50 $W/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl out/baselines/swe50
    python scripts/baselines/workloads.py vidur   out/baselines/swe50      # vidur.csv
    python scripts/baselines/workloads.py frontier out/baselines/swe50     # frontier.csv
    python scripts/baselines/workloads.py blis    out/baselines/swe50      # blis.yaml, blis.csv, scen/scenario.yaml

The engine settings come from the run's server.log (max-model-len, max-num-batched-tokens, max-num-seqs, KV blocks,
async scheduling, prefix caching, scheduling policy); ``prepare`` takes ``--max-model-len``, ``--max-num-batched-tokens``,
``--max-num-seqs``, ``--num-gpu-blocks``, ``--prefix-caching on|off``, ``--scheduling-policy fcfs|priority`` and
``--async-scheduling sync|async`` to state them instead (a run directory without a server.log needs them all).

How faithful each input is
* Vidur has no prefix caching, no sessions, no token ids and no priority: arrival and the two lengths only.
* Frontier takes ``block_hash_ids``, the prompt's chained block hashes: the same keys vLLM's
  prefix cache uses, so sharing is exact at block granularity. Sessions are flattened: each call
  arrives at its recorded engine arrival. Under the priority policy it also takes a ``priority`` column.
* BLIS takes a ``prefix_group`` per request and prepends the group's one prefix (the first
  record's ``prefix_length``, the same for every member) to the request's ``input_tokens``. For
  flat requests we process them in arrival order, join the group of the earlier request with the
  longest block-aligned common prefix, and give the group the shortest match among its members
  (members that share more lose the difference). Sessions (call ids ``<session>#<i>``) use BLIS's
  ``session_context_growth: accumulate`` trace: round n's prompt is round n-1's prompt, its
  output and a delta, at the recorded arrival (``replay --session-mode fixed-accumulate``), so
  a round reuses the whole previous round, output ids included (real vLLM reuses the previous
  prompt only: its outputs differ from the recorded ones). A round whose recorded prompt is shorter than
  the previous prompt plus output is a compaction in that format, a reset to a fresh prompt: it
  shares nothing in BLIS although it shares most of the previous prompt in vLLM (64 of the 715 later
  rounds of swe50). Sharing across sessions is not mapped. ``meta.json`` reports the tokens the mapping shares against the tokens
  token-id sharing shares (no eviction), and the hit rate the real server logged.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys

from augur.trace.workload import Session, Workload, call_id
from augur.verify.replay import ReplayWorkload, is_async, read_log, steps

BLOCK = 16
MASK = (1 << 62) - 1


def _block_hashes(ids: list[int]) -> list[int]:
    """vLLM-style chained hashes of the prompt's full blocks."""
    out, parent = [], 0
    for i in range(0, len(ids) - len(ids) % BLOCK, BLOCK):
        parent = hash((parent, tuple(ids[i:i + BLOCK]))) & MASK
        out.append(parent)
    return out


def _server_meta(run_dir: str) -> dict:
    path = os.path.join(run_dir, "server.log")
    text = open(path).read() if os.path.exists(path) else ""
    m = re.search(r"non-default args: (\{.*\})", text)
    args = ast.literal_eval(m.group(1)) if m else {}
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
    hit = re.findall(r"Prefix cache hit rate: ([\d.]+)%", text)
    spec = args.get("speculative_config") or {}
    return {"prefix_caching": args.get("enable_prefix_caching", True),
            "scheduling_policy": args.get("scheduling_policy", "fcfs"),
            "max_model_len": args.get("max_model_len", 8192), "max_num_seqs": args.get("max_num_seqs", 128),
            "max_num_batched_tokens": args.get("max_num_batched_tokens", 2048),
            "kv_tokens": int(kv[-1].replace(",", "")) if kv else None,
            "num_gpu_blocks": int(kv[-1].replace(",", "")) // BLOCK if kv else None,
            "kv_offloading_gib": args.get("kv_offloading_size"),
            "spec_k": spec.get("num_speculative_tokens", 0), "spec": spec,
            "real_prefix_hit_rate": float(hit[-1]) / 100 if hit else None}


def _real_requests(events: list[dict]) -> dict:
    """call id -> engine-side add time, first token time, finish time, tokens, preemptions (log ns)."""
    r: dict[str, dict] = {}
    for e in events:
        if e["ev"] == "add" and not e["rid"].endswith("@d"):
            r.setdefault(call_id(e["rid"]), {"add": e["t"], "rid": e["rid"], "out": 0, "first": None, "fin": None,
                                             "pre": 0})
    by_rid = {v["rid"]: v for v in r.values()}
    for e in events:
        if e["ev"] == "sched":
            for rid in e["pre"]:
                if rid in by_rid:
                    by_rid[rid]["pre"] += 1
        elif e["ev"] == "upd":
            for rid, n in e["out"].items():
                v = by_rid.get(rid)
                if v is not None and n:
                    v["out"] += n
                    if v["first"] is None:
                        v["first"] = e["t"]
            for rid in e["fin"]:
                if rid in by_rid:
                    by_rid[rid]["fin"] = e["t"]
    return r


def _acceptance(events: list[dict], k: int) -> dict:
    """Measured n-gram acceptance: tokens emitted per decode request-step, and accepted / scheduled drafts."""
    emitted = steps_n = drafts = accepted = 0
    seen: set[str] = set()  # requests that have emitted a token: their later steps are decodes
    for s, u in steps(events):
        for rid, n in s["tok"].items():
            o = u["out"].get(rid, 0)
            if rid in seen and o and n <= 1 + k:  # (a longer step is the recompute of a preempted request)
                # a decode: scheduled n = 1 + drafts, emitted o = 1 + accepted drafts
                steps_n += 1
                emitted += o
                drafts += n - 1
                accepted += o - 1
            if o:
                seen.add(rid)
    return {"decode_request_steps": steps_n, "mean_tokens_per_step": emitted / steps_n if steps_n else None,
            "draft_tokens_scheduled": drafts, "accepted_per_scheduled": accepted / drafts if drafts else None}


def prepare(a) -> int:
    log = sorted(glob.glob(os.path.join(a.run_dir, "steps.*.jsonl")))[-1]
    events = read_log(log)
    meta = _server_meta(a.run_dir)
    for k, v in (("max_model_len", a.max_model_len), ("max_num_batched_tokens", a.max_num_batched_tokens),
                 ("max_num_seqs", a.max_num_seqs), ("num_gpu_blocks", a.num_gpu_blocks),
                 ("scheduling_policy", a.scheduling_policy)):
        if v is not None:
            meta[k] = v
    if a.num_gpu_blocks is not None:
        meta["kv_tokens"] = a.num_gpu_blocks * BLOCK
    if a.prefix_caching is not None:
        meta["prefix_caching"] = a.prefix_caching == "on"
    meta.update(async_scheduling=is_async(events) if a.async_scheduling is None else a.async_scheduling == "async")
    meta.update(step_log=os.path.abspath(log), workload=os.path.abspath(a.workload),
                run_dir=os.path.abspath(a.run_dir))
    wl = ReplayWorkload(a.workload, log_path=log, recorded_outputs=False, filler_vocab=150000)
    calls = list(wl.entries())  # arrival_ns: engine arrival, session chaining off
    # What the trace itself says, for the normal arrival process: when each flat request and each session's first
    # call is sent, and how long a session waits after each call (tool time).
    plain = Workload(a.workload, recorded_outputs=False, filler_vocab=150000)
    sent, wait = {}, {}
    for e in plain.entries():
        if isinstance(e, Session):
            for i in range(len(e.sub_requests)):
                c = plain.next_call(e, 0)
                wait[c.call_id] = c.tool_wait_ns / 1e9
            sent[f"{e.session_id}#0"] = e.arrival_ns / 1e9
        else:
            sent[e.call_id] = e.arrival_ns / 1e9
    real = _real_requests(events)
    rows = []
    for c in calls:
        fe = (a.frontend_latency_ms * 1e3 + a.frontend_us_per_token * len(c.prompt_ids)) / 1e6
        rows.append({"call_id": c.call_id, "arrival_s": c.arrival_ns / 1e9, "input": len(c.prompt_ids),
                     "output": c.output_toks, "blocks": _block_hashes(c.prompt_ids), "priority": c.priority,
                     "real": real.get(c.call_id),
                     # a flat request's and a session's first call's engine arrival in augur's normal run: sent at the
                     # trace's time, a front end earlier than it reaches the engine; None for a later call
                     "arrival_trace_s": sent[c.call_id] + fe if c.call_id in sent else None,
                     "tool_wait_s": wait.get(c.call_id, 0.0), "frontend_s": fe,
                     "delivery_s": a.delivery_latency_ms / 1e3})
    # Time zero is the first step's start (the engine adds a request just before the schedule that takes it): a
    # simulator whose first step starts at its first arrival then has every step on the real run's timeline, and
    # a request is never added at the very instant a step starts.
    t_first = min(e["t"] for e in events)
    t0 = min(e["t"] for e in events if e["ev"] == "sched")
    shift = (t0 - t_first) / 1e9
    for r in rows:
        r["arrival_s"] = max(r["arrival_s"] - shift, 0.0)
    rows.sort(key=lambda r: (r["arrival_s"], r["call_id"]))
    for r in rows:
        rr = r["real"]
        r["real"] = None if rr is None else {
            "add_s": (rr["add"] - t0) / 1e9, "first_s": None if rr["first"] is None else (rr["first"] - t0) / 1e9,
            "fin_s": None if rr["fin"] is None else (rr["fin"] - t0) / 1e9, "out": rr["out"], "pre": rr["pre"]}
    _map_prefixes(rows, meta, a.blis_encoding)
    # top-level entries of the trace the real run sent (a flat request, or a whole session), for augur's --num-entries
    meta["num_entries"] = len({r["call_id"].rsplit("#", 1)[0] if "#" in r["call_id"] else r["call_id"] for r in rows})
    if meta["spec_k"]:
        meta["acceptance"] = _acceptance(events, meta["spec_k"])
        acc = meta["acceptance"]["mean_tokens_per_step"] - 1  # accepted drafts per decode request-step
        # augur's --spec-accept-rate is a per-position probability: p + p^2 + ... + p^k accepted drafts per step
        lo, hi = 0.0, 1.0
        for _ in range(60):
            p = (lo + hi) / 2
            lo, hi = (p, hi) if sum(p ** i for i in range(1, meta["spec_k"] + 1)) < acc else (lo, p)
        meta["acceptance"]["accepted_per_step"] = acc
        meta["acceptance"]["spec_p"] = round((lo + hi) / 2, 4)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"requests": rows}, open(os.path.join(a.out, "requests.json"), "w"))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("spec",)}, indent=1))
    return 0


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _map_prefixes(rows: list[dict], meta: dict, encoding: str = "greedy") -> None:
    """BLIS's prefix_group / prefix_len per request (see the module docstring) and the sharing it gives.
    ``encoding`` "best" then replaces the greedy mapping with ``_encode_best`` (the greedy one's numbers stay in
    ``meta["prefix"]`` as ``greedy_*``)."""
    by_first: dict[int, list[int]] = {}  # first block hash -> earlier requests starting with it
    group: list[int] = []
    plen: list[int] = [0] * len(rows)  # blocks
    ideal = mapped = total = 0
    for i, r in enumerate(rows):
        blocks = r["blocks"]
        cap = (r["input"] - 1) // BLOCK  # vLLM never hits the whole prompt
        sid = r["call_id"].split("#")[0] if "#" in r["call_id"] else None
        best, bj, best_any = 0, -1, 0
        for j in by_first.get(blocks[0], []) if blocks else []:
            n = min(_lcp(blocks, rows[j]["blocks"]), cap)
            best_any = max(best_any, n)
            # A session's calls chain; sharing across sessions is not mapped (one prefix group per request).
            if n > best and (sid is None or rows[j]["call_id"].split("#")[0] == sid):
                best, bj = n, j
        total += r["input"]
        ideal += best_any * BLOCK
        if bj >= 0:
            group.append(group[bj])
            plen[i] = best
            plen[bj] = max(plen[bj], best)
        else:
            group.append(i)
        if blocks:
            by_first.setdefault(blocks[0], []).append(i)
    members: dict[int, list[int]] = {}
    for i, g in enumerate(group):
        members.setdefault(g, []).append(i)
    # BLIS gives every member of a group the group's first prefix_length: one length per group, the shortest
    # (what every member really shares with the others); sessions use BLIS's accumulate mode instead (see blis()).
    for g, idx in members.items():
        if len(idx) > 1 and "#" not in rows[idx[0]]["call_id"]:
            m = min(plen[i] for i in idx)
            for i in idx:
                plen[i] = m
    for i, r in enumerate(rows):
        r["prefix_group"] = f"g{group[i]}" if plen[i] and "#" not in r["call_id"] else ""
        r["prefix_len"] = plen[i] * BLOCK if r["prefix_group"] else 0
    for g, idx in members.items():
        if "#" in rows[idx[0]]["call_id"]:
            # accumulate mode: round n reuses round n-1's prompt and output (BLIS's own output ids, so a little more
            # than real vLLM, whose outputs differ from the recorded ones)
            for a, b in zip(idx, idx[1:]):
                if rows[b]["input"] >= rows[a]["input"] + rows[a]["output"]:  # else a reset: a fresh prompt, nothing shared
                    mapped += min((rows[a]["input"] + rows[a]["output"]) // BLOCK, (rows[b]["input"] - 1) // BLOCK) * BLOCK
        else:
            for k, i in enumerate(idx):
                if k:
                    mapped += min(plen[i], (rows[i]["input"] - 1) // BLOCK) * BLOCK
    meta["prefix"] = {"prompt_tokens": total, "token_id_sharing_tokens": ideal, "blis_mapping_tokens": mapped,
                      "token_id_sharing_rate": ideal / total if total else 0.0,
                      "blis_mapping_rate": mapped / total if total else 0.0}
    if encoding == "best":
        greedy = meta["prefix"]
        _encode_best(rows, meta)
        meta["prefix"].update(greedy_mapping_tokens=greedy["blis_mapping_tokens"],
                              greedy_mapping_rate=greedy["blis_mapping_rate"])


def _encode_best(rows: list[dict], meta: dict) -> None:
    """The best a user of BLIS's trace format can do with the prompts' token ids (``--blis-encoding best``).

    BLIS states sharing as one ``prefix_group`` per request with one ``prefix_length`` per group (the group's first
    record's), and a session as each round's previous prompt and output plus a delta, or a reset to a fresh prompt.
    Here:

    * A session round whose prompt starts with the whole previous prompt (and is at least as long as that prompt and
      output) continues the session. Any other round (a context trimmed or summarised: it shares only a head with the
      previous prompt, and the greedy encoding's reset shares nothing) starts a new BLIS session at its own engine
      arrival, so that it can carry a prefix group.
    * Every request and every such start is a unit, and the units are partitioned into groups with a length each so
      as to minimise the sum over units of |tokens the group states - tokens the prompt really reuses from an earlier
      prompt| (the first unit of a group states 0). Start from the greedy partition (join the group of the earlier unit
      sharing the most), pick each group's length best, then move one unit at a time to the group that lowers the
      sum, until none does. A unit's sharing with units of other groups, and a tree of prefixes
      (a preamble for everyone, then a section for some), is what this cannot state."""
    n = len(rows)
    last: dict[str, int] = {}
    cont = [False] * n
    sid_of = [r["call_id"].rsplit("#", 1)[0] if "#" in r["call_id"] else None for r in rows]
    for i, r in enumerate(rows):
        s = sid_of[i]
        if s is None:
            continue
        p = last.get(s)
        if p is not None:
            q = rows[p]
            cont[i] = r["input"] >= q["input"] + q["output"] and _lcp(r["blocks"], q["blocks"]) >= len(q["blocks"])
        last[s] = i
    cap = [(r["input"] - 1) // BLOCK for r in rows]
    by_first: dict[int, list[int]] = {}
    true = [0] * n  # blocks the prompt really reuses from an earlier prompt
    lcp: dict[tuple[int, int], int] = {}  # between units, both orders, capped by neither
    units = [i for i in range(n) if not cont[i]]
    for i, r in enumerate(rows):
        b = r["blocks"]
        for j in by_first.get(b[0], []) if b else []:
            n_ij = _lcp(b, rows[j]["blocks"])
            true[i] = max(true[i], min(n_ij, cap[i]))
            lcp[i, j] = lcp[j, i] = n_ij
        if b:
            by_first.setdefault(b[0], []).append(i)

    def best_len(members: list[int]) -> tuple[int, int]:
        """(cost, length in blocks) of a group of units (indices into rows, ascending): the first states nothing,
        so the length that minimises the others' |length - true| is their median, within every member's cap."""
        first, rest = members[0], members[1:]
        if not rest:
            return true[first], 0
        top = min(cap[m] for m in members)
        ts = sorted(true[m] for m in rest)
        length = min(ts[len(ts) // 2], top)
        return true[first] + sum(abs(length - t) for t in ts), length

    group_of: dict[int, int] = {}
    members: dict[int, list[int]] = {}
    for u in units:  # the greedy partition: join the group of the earlier unit sharing the most
        best, bj = 0, -1
        for j in units:
            if j >= u:
                break
            if lcp.get((u, j), 0) > best:
                best, bj = lcp[u, j], j
        g = group_of[bj] if bj >= 0 else u
        group_of[u] = g
        members.setdefault(g, []).append(u)
    cost = {g: best_len(m)[0] for g, m in members.items()}
    fresh = n  # labels for the groups a move creates

    def move(us: list[int], src: int, dst: int) -> None:
        nonlocal fresh
        if dst == -1:
            dst, fresh = fresh, fresh + 1
            members[dst] = []
        keep = [m for m in members[src] if m not in us]
        if keep:
            members[src], cost[src] = keep, best_len(keep)[0]
        else:
            del members[src], cost[src]
        members[dst] = sorted(members[dst] + us)
        cost[dst] = best_len(members[dst])[0]
        for m in us:
            group_of[m] = dst

    for _ in range(12):
        moved = False
        seen: set[tuple] = set()  # (subset, destination) already tried against the current groups
        for u in units:
            g0 = group_of[u]
            mem = members[g0]
            near = {v: lcp.get((u, v), 0) for v in mem if v != u}
            others = {group_of[v] for v in units if v != u and lcp.get((u, v), 0) > 0} - {g0}
            depths = sorted({x for x in near.values() if x > 0})
            depths = depths if len(depths) <= 24 else depths[::len(depths) // 24 + 1]
            tries = []
            # (a) take with u the members of its group sharing at least d blocks with it (d = None: u alone), and
            # make them a group of their own or add them to another group u shares a prefix with; (b) merge u's
            # group into another one
            for d in [None, *depths]:
                us = tuple([u] if d is None else sorted([u] + [v for v, x in near.items() if x >= d]))
                if (us, g0) in seen:
                    continue
                seen.add((us, g0))
                keep = [m for m in mem if m not in us]
                c_keep = best_len(keep)[0] if keep else 0
                if keep:
                    tries.append((c_keep + best_len(list(us))[0] - cost[g0], us, -1))
                for g in others:
                    tries.append((c_keep + best_len(sorted(members[g] + list(us)))[0] - cost[g0] - cost[g], us, g))
            for g in others:
                tries.append((best_len(sorted(members[g] + mem))[0] - cost[g] - cost[g0], tuple(mem), g))
            if tries:
                d, us, dst = min(tries, key=lambda t: t[0])
                if d < 0:
                    move(list(us), g0, dst)
                    moved = True
                    seen.clear()
        if not moved:
            break
    plen = {}
    for g, m in members.items():
        length = best_len(m)[1] if len(m) > 1 else 0
        for u in m:
            plen[u] = length
    seg: dict[str, int] = {}  # session -> its current BLIS session (a start of a new one counts up)
    rnd: dict[str, int] = {}
    last_row: dict[str, int] = {}
    ideal = mapped = error = total = 0
    first_of = {g: m[0] for g, m in members.items()}
    for i, r in enumerate(rows):
        s = sid_of[i]
        total += r["input"]
        if cont[i]:
            p = last_row[s]
            e = min((rows[p]["input"] + rows[p]["output"]) // BLOCK, cap[i])
            rnd[s] += 1
            r.update(prefix_group="", prefix_len=0, blis_sid=f"{s}~{seg[s]}" if seg[s] else s, blis_round=rnd[s], blis_cont=True)
        else:
            g = group_of[i]
            e = plen[i] if first_of[g] != i else 0
            if s is not None:
                seg[s] = seg.get(s, -1) + 1
                rnd[s] = 0
            r.update(prefix_group=f"g{g}" if plen[i] else "", prefix_len=plen[i] * BLOCK,
                     blis_sid="" if s is None else (f"{s}~{seg[s]}" if seg[s] else s), blis_round=0, blis_cont=False)
        if s is not None:
            last_row[s] = i
        ideal += true[i] * BLOCK
        mapped += e * BLOCK
        error += abs(e - true[i]) * BLOCK
    meta["prefix"] = {"prompt_tokens": total, "token_id_sharing_tokens": ideal, "blis_mapping_tokens": mapped,
                      "token_id_sharing_rate": ideal / total if total else 0.0,
                      "blis_mapping_rate": mapped / total if total else 0.0, "blis_error_tokens": error,
                      "groups": sum(1 for m in members.values() if len(m) > 1 and plen[m[0]])}


def blis_groups(call_ids: list[str], prompts: list[list[int]], outputs: list[int]) -> dict:
    """The sharing both encodings state for calls in arrival order (``meta["prefix"]``), from their prompt ids."""
    rows = [{"call_id": c, "blocks": _block_hashes(p), "input": len(p), "output": o}
            for c, p, o in zip(call_ids, prompts, outputs)]
    meta: dict = {}
    _map_prefixes(rows, meta, "best")
    return meta["prefix"]


def _load(out: str) -> tuple[list[dict], dict]:
    return json.load(open(os.path.join(out, "requests.json")))["requests"], json.load(open(os.path.join(out, "meta.json")))


def arrivals(a) -> int:
    """After augur's normal run: when each call reached the engine in it (seconds after its first add), as
    ``arrival_aug_s``. A simulator that cannot loop a session through its own completions takes these."""
    log = sorted(glob.glob(os.path.join(a.out, "augur_steps.*.jsonl")))[-1]
    rows, meta = _load(a.out)
    idx = {r["call_id"]: i for i, r in enumerate(rows)}
    add = {idx[call_id(e["rid"])]: e["t"] for e in read_log(log) if e["ev"] == "add" and call_id(e["rid"]) in idx}
    t0 = min(add.values())
    for i, r in enumerate(rows):
        r["arrival_aug_s"] = (add[i] - t0) / 1e9 if i in add else None
    json.dump({"requests": rows}, open(os.path.join(a.out, "requests.json"), "w"))
    return 0


def _arrival(r: dict, replay: bool) -> float:
    """The engine arrival a simulator is given: the real run's (replay), else augur's normal run's."""
    return r["arrival_s"] if replay or r.get("arrival_aug_s") is None else r["arrival_aug_s"]


def vidur(a) -> int:
    rows, _ = _load(a.out)
    for v in ("", "_replay"):
        with open(os.path.join(a.out, f"vidur{v}.csv"), "w") as f:
            f.write("arrived_at,num_prefill_tokens,num_decode_tokens\n")
            for r in rows:
                f.write(f"{_arrival(r, bool(v)):.9f},{r['input']},{r['output']}\n")
    return 0


def frontier(a) -> int:
    rows, meta = _load(a.out)
    prio = meta["scheduling_policy"] == "priority"
    for v in ("", "_replay"):
        sessions: dict[str, int] = {}
        with open(os.path.join(a.out, f"frontier{v}.csv"), "w") as f:
            f.write("arrived_at,num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids"
                    + (",priority" if prio else "") + "\n")
            for i, r in enumerate(rows):
                sid = r["call_id"].rsplit("#", 1)[0] if "#" in r["call_id"] else r["call_id"]
                f.write(f"{_arrival(r, bool(v)):.9f},{r['input']},{r['output']},{sessions.setdefault(sid, len(sessions))},"
                        f"{'|'.join(map(str, r['blocks']))}" + (f",{r['priority']}" if prio else "") + "\n")
    return 0


def blis(a) -> int:
    """blis.yaml/blis.csv: the normal arrival process (a flat request at augur's engine arrival; a session's first
    call likewise and each later call think_time_us after the previous one completes, closed loop). blis_replay.*:
    every call at its real arrival (``replay --session-mode fixed-accumulate`` for sessions)."""
    rows, meta = _load(a.out)
    sessions = any("#" in r["call_id"] for r in rows)
    prio = meta["scheduling_policy"] == "priority"
    cache = meta["prefix_caching"]
    for v in ("", "_replay"):
        replay = bool(v)
        think = sessions and not replay
        open(os.path.join(a.out, f"blis{v}.yaml"), "w").write(
            "trace_version: 4\ntime_unit: microseconds\nmode: converted\nworkload_seed: 1\n"
            + ("session_context_growth: accumulate\n" if sessions else ""))
        cols = ("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,"
                "input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,"
                "deadline_us,arrival_time_us,first_chunk_time_us,last_chunk_time_us,status"
                + (",think_time_us" if think else "") + (",input_tokens_reset" if sessions else "") + "\n")
        last: dict[str, dict] = {}  # session -> previous round
        with open(os.path.join(a.out, f"blis{v}.csv"), "w") as f:
            f.write(cols)
            for i, r in enumerate(rows):
                sid, rnd, tail, tt = "", 0, "", ""
                if "blis_sid" in r:  # --blis-encoding best: a continuing round is a delta, any other a session's start
                    sid, rnd = r["blis_sid"], r["blis_round"]
                    if r["blis_cont"]:
                        prev = last[sid]
                        n_in = r["input"] - prev["input"] - prev["output"]
                        if think:
                            tt = str(round((prev["tool_wait_s"] + r["delivery_s"] + r["frontend_s"]) * 1e6))
                    else:
                        n_in = r["input"] - (r["prefix_len"] if cache else 0)
                    tail = (f",{tt}," if think else ",") if sessions else ""
                    if sid:
                        last[sid] = r
                elif "#" in r["call_id"]:
                    sid, idx = r["call_id"].rsplit("#", 1)
                    rnd = int(idx)
                    prev = last.get(sid)
                    grown = prev["input"] + prev["output"] if prev else 0
                    # the trace's delta law: round n's input = round n-1's input + output + delta; a round that
                    # is shorter than that is a compaction, stated as a reset to its absolute length
                    if prev is None:
                        n_in, reset = r["input"], ""
                    elif r["input"] >= grown:
                        n_in, reset = r["input"] - grown, ""
                    else:
                        n_in, reset = 0, str(r["input"])
                    if think:
                        # as augur's loop: the previous call's tool wait and delivery, then a front end
                        tt = "" if prev is None else str(round((prev["tool_wait_s"] + r["delivery_s"] + r["frontend_s"]) * 1e6))
                        tail = f",{tt},{reset}"
                    else:
                        tail = f",{reset}"
                    last[sid] = r
                else:
                    # input_tokens is what follows the shared prefix (the replay prepends the group's prefix); with
                    # the prefix cache off nothing is shared, so no group
                    n_in = r["input"] - (r["prefix_len"] if cache else 0)
                grp, plen = (r["prefix_group"], r["prefix_len"]) if cache else ("", 0)
                # a numeric SLO class is the request's vLLM priority (blis_priority.patch)
                f.write(f"{i},c,,{r['priority'] if prio else ''},{sid},{rnd},{grp},{plen},false,{n_in},{r['output']},0,0,0,0,0,,0,"
                        f"{round(_arrival(r, replay) * 1e6)},0,0,ok{tail}\n")
    spec = ""
    if meta["spec_k"]:
        s = meta["spec"]
        spec = ("      speculative_config: {method: ngram, num_speculative_tokens: %d}\n"
                % s["num_speculative_tokens"])
    knobs = ("" if cache else "      enable_prefix_caching: false\n") + \
        ("      scheduling_policy: priority\n" if prio else "")  # (the engine's defaults otherwise)
    offload, tiers = "", ""
    if meta["kv_offloading_gib"]:
        tiers = ", cost-model-offload-tiers"
        offload = ("offload:\n  connector: OffloadingConnector\n  eviction_policy: lru\n  tiers:\n"
                   f"    - tier: cpu_dram\n      bytes: {int(meta['kv_offloading_gib'] * (1 << 30))}\n")
    # Qwen3-1.7B and an RTX 4060 Ti are not in BLIS's catalog. The price comes from augur's pricer, and the KV block
    # count is overridden, so the model (dense, full attention, GQA, like Qwen3-14B) and the chip only name the engine.
    os.makedirs(os.path.join(a.out, "scen"), exist_ok=True)
    open(os.path.join(a.out, "scen", "scenario.yaml"), "w").write(f"""kind: Scenario
name: baseline
engine: {{kind: vllm, version: "0.33.0"}}
model: qwen3-14b
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-attention, cost-model-recurrent, cost-model-memory{tiers}]
cluster:
  hardware: h100
  nodes: 1
  gpus_per_node: 1
---
kind: Deployment
name: baseline
pools:
  - role: colocated
    nodes: 1
    parallel: {{tp: 1, pp: 1, dp: 1}}
    engine:
      block_size: 16
      max_num_batched_tokens: {meta['max_num_batched_tokens']}
      max_num_seqs: {meta['max_num_seqs']}
      max_model_len: {meta['max_model_len']}
      gpu_memory_utilization: 0.85
      async_scheduling: {'true' if meta['async_scheduling'] else 'false'}
{knobs}{spec}{offload}""")
    return 0


def get(a) -> int:
    """Print one value of meta.json (a dotted key), for the run script."""
    v = json.load(open(os.path.join(a.out, "meta.json")))
    for k in a.key.split("."):
        v = v.get(k) if isinstance(v, dict) else None
    print("" if v is None else (str(v).lower() if isinstance(v, bool) else v))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="workloads")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("run_dir")
    pr.add_argument("workload")
    pr.add_argument("out")
    # the run's settings, where they are not in its server.log (or to override it)
    pr.add_argument("--max-model-len", type=int, default=None)
    pr.add_argument("--max-num-batched-tokens", type=int, default=None)
    pr.add_argument("--max-num-seqs", type=int, default=None)
    pr.add_argument("--num-gpu-blocks", type=int, default=None)
    pr.add_argument("--prefix-caching", choices=("on", "off"), default=None)
    pr.add_argument("--scheduling-policy", choices=("fcfs", "priority"), default=None)
    pr.add_argument("--async-scheduling", choices=("sync", "async"), default=None)
    # how BLIS's prefix groups and sessions are stated (see _encode_best): greedy is the encoding of the earlier runs
    pr.add_argument("--blis-encoding", choices=("greedy", "best"), default="greedy")
    # augur's front-end and delivery defaults (augur.cli), which its normal run applies to every call
    pr.add_argument("--frontend-latency-ms", type=float, default=3.4)
    pr.add_argument("--frontend-us-per-token", type=float, default=3.0)
    pr.add_argument("--delivery-latency-ms", type=float, default=8.0)
    sub.add_parser("arrivals").add_argument("out")
    for n in ("vidur", "frontier", "blis"):
        sub.add_parser(n).add_argument("out")
    g = sub.add_parser("get")
    g.add_argument("out")
    g.add_argument("key")
    a = p.parse_args(argv)
    return {"prepare": prepare, "vidur": vidur, "frontier": frontier, "blis": blis, "get": get, "arrivals": arrivals}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
