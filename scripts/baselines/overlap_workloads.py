"""Seeded workloads whose prompts overlap in part, for the baselines test (scripts/baselines/run.sh).

    python scripts/baselines/overlap_workloads.py gen tenants out/overlap/tenants.jsonl [--seed 0]
    python scripts/baselines/overlap_workloads.py gen all out/overlap          # the four, <pattern>.jsonl each
    python scripts/baselines/overlap_workloads.py stats out/overlap/tenants.jsonl

Every prompt is built from token ids already on the host (the ShareGPT trace's prompts and outputs, and the
SWE-bench trace's last calls), cut into disjoint pieces, so two prompts share a token only where the pattern puts
the same piece in both. The patterns (sizes in the module's ``PATTERNS``):

* ``tenants``: a platform preamble shared by every tenant, then the tenant's own system prompt (a different length
  per tenant; some tenants have two versions that agree for part of their length), then the user's turn.
  Tenants are Zipf-popular.
* ``fewshot``: three tasks, each with a header and a bank of examples; a prompt is the header, a subset of the
  bank (Zipf-popular examples, in bank order) and the query. Two prompts share the examples they both lead with.
* ``rag``: a platform prompt, then retrieved documents from one topic's corpus (popular documents first, ranked
  with some noise), then the question. Documents are only shared where the whole run of documents before them is.
* ``agent``: sessions with one system prompt in common and a task of their own; each call appends the previous
  output and a tool result, and now and then the context is trimmed (older turns dropped) or summarised (replaced
  by a new summary): the next prompt then shares the head of the previous one and nothing after it.

Four more, for held-out real runs (``HELDOUT``; ``gen all`` leaves them out): ``bursty`` (the ShareGPT trace's requests
with gamma-process arrivals, CV 4), ``longctx`` (12k-24k-token prompts cut from the SWE-bench trace), ``swefast`` (the
SWE-bench sessions arriving at another rate) and ``peruser`` (short prompts: a shared preamble, then one of a few
personal prefixes of the user that agree for part of their length, then the user's turn).

``stats`` reports each workload's sharing from the token ids: the tokens of each prompt reusable from earlier
prompts (the longest run of full 16-token blocks it has in common with one earlier prompt, as vLLM's prefix cache
would hit it with unlimited memory), against what a prefix-group trace (BLIS) can state: workloads.py's
``blis_groups``, one prefix length per group.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WL = os.path.join(os.environ.get("AUGUR_FORK", os.path.expanduser("~/llmss-vllm-sim-20261114")), "workloads")
SHAREGPT = os.path.join(WL, "sharegpt-qwen3-32b-300-sps10.jsonl")
SWEBENCH = os.path.join(WL, "swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl")
BLOCK = 16
MAX_PROMPT = 7000  # + the output must fit max-model-len 8192

# requests (sessions for agent), arrivals per second (sessions per second for agent)
PATTERNS = {"tenants": (400, 6.0), "fewshot": (350, 5.0), "rag": (350, 5.0), "agent": (28, 0.45)}
# requests, arrivals per second (sessions per second for swefast)
HELDOUT = {"bursty": (300, 1.2), "longctx": (100, 0.2), "swefast": (50, 1.0), "peruser": (400, 5.0)}
SWE_RATE = 0.2  # the SWE-bench trace's sessions per second
BURST_CV = 4.0
LONGCTX_PROMPT = (12000, 23800)  # + the output must fit max-model-len 24576
PERUSER_MAX = 800


class Pool:
    """Disjoint pieces of real token ids: documents in a seeded shuffle, concatenated, handed out in order (the
    walk wraps when the ids run out)."""

    def __init__(self, docs: list[list[int]], rng: random.Random):
        docs = list(docs)
        rng.shuffle(docs)
        self.ids = [t for d in docs for t in d]
        self.pos = 0

    def take(self, n: int) -> list[int]:
        out = []
        while len(out) < n:
            k = min(n - len(out), len(self.ids) - self.pos)
            out += self.ids[self.pos:self.pos + k]
            self.pos = (self.pos + k) % len(self.ids)
        return out


def sharegpt_docs(path: str = SHAREGPT) -> list[list[int]]:
    docs = []
    for line in open(path):
        r = json.loads(line)
        docs += [r["input_tok_ids"], r["output_tok_ids"]]
    return [d for d in docs if len(d) >= 64]


def swebench_docs(path: str = SWEBENCH) -> list[list[int]]:
    """Each session's longest call: one long run of code, issue and tool-output text."""
    return [max((c["input_tok_ids"] for c in json.loads(line)["sub_requests"]), key=len) for line in open(path)]


def zipf_choice(rng: random.Random, n: int, s: float = 1.0) -> int:
    return rng.choices(range(n), weights=[1 / (i + 1) ** s for i in range(n)])[0]


def arrivals(rng: random.Random, n: int, rate: float) -> list[int]:
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(int(t * 1e9))
    return out


def request(t_ns: int, ids: list[int], rng: random.Random) -> dict:
    return {"input_toks": len(ids), "output_toks": rng.randint(100, 300), "arrival_time_ns": t_ns, "input_tok_ids": ids}


def tenants(rng: random.Random, n: int, rate: float, chat: Pool, _swe: Pool) -> list[dict]:
    preamble = chat.take(768)
    versions = []  # per tenant: a list of (version weight, system prompt)
    for k in range(24):
        body = chat.take(rng.randint(200, 2600))
        vs = [(3, body)]
        if k % 2 == 0:  # a revised system prompt: the same start, another end
            cut = int(len(body) * rng.uniform(0.3, 0.9))
            vs.append((1, body[:cut] + chat.take(rng.randint(100, 900))))
        versions.append(vs)
    out = []
    for t in arrivals(rng, n, rate):
        vs = versions[zipf_choice(rng, len(versions))]
        body = rng.choices([v for _, v in vs], weights=[w for w, _ in vs])[0]
        user = chat.take(rng.randint(100, 1500))
        out.append(request(t, (preamble + body + user)[:MAX_PROMPT], rng))
    return out


def fewshot(rng: random.Random, n: int, rate: float, chat: Pool, _swe: Pool) -> list[dict]:
    tasks = [(chat.take(rng.randint(150, 350)), [chat.take(rng.randint(150, 450)) for _ in range(24)])
             for _ in range(3)]
    out = []
    for t in arrivals(rng, n, rate):
        header, bank = tasks[zipf_choice(rng, len(tasks), 0.7)]
        pick = set()
        while len(pick) < rng.randint(4, 10):
            pick.add(zipf_choice(rng, len(bank)))
        ids = header + [x for i in sorted(pick) for x in bank[i]] + chat.take(rng.randint(150, 1200))
        out.append(request(t, ids[:MAX_PROMPT], rng))
    return out


def rag(rng: random.Random, n: int, rate: float, chat: Pool, _swe: Pool) -> list[dict]:
    system = chat.take(300)
    topics = [[chat.take(rng.randint(250, 900)) for _ in range(10)] for _ in range(8)]
    out = []
    for t in arrivals(rng, n, rate):
        docs = topics[zipf_choice(rng, len(topics), 0.8)]
        pick = set()
        while len(pick) < rng.randint(3, 6):
            pick.add(zipf_choice(rng, len(docs), 0.8))
        order = sorted(pick)  # most popular first, as a retriever's ranking mostly agrees on
        for i in range(len(order) - 1):  # ranking noise
            if rng.random() < 0.3:
                order[i], order[i + 1] = order[i + 1], order[i]
        ids = system + [x for i in order for x in docs[i]] + chat.take(rng.randint(50, 300))
        out.append(request(t, ids[:MAX_PROMPT], rng))
    return out


def agent(rng: random.Random, n: int, rate: float, chat: Pool, swe: Pool) -> list[dict]:
    system = swe.take(1000)
    out = []
    for t0 in arrivals(rng, n, rate):
        task = swe.take(rng.randint(600, 1500))
        turns: list[list[int]] = []  # (output + tool result) of each call so far
        prompt: list[int] = []
        calls = []
        for i in range(rng.randint(8, 16)):
            n_out = rng.randint(100, 300)
            if i == 0:
                prompt = system + task
            else:
                turns.append(calls[-1]["output_ids"] + swe.take(rng.randint(200, 900)))
                if len(system) + len(task) + sum(map(len, turns)) > 6000 or (i >= 3 and rng.random() < 0.2):
                    if rng.random() < 0.5:  # trim: the oldest turns go
                        turns = turns[-2:]
                    else:  # summarise: one new summary for everything but the last turn
                        turns = [swe.take(rng.randint(250, 500))] + turns[-1:]
                prompt = system + task + [x for tn in turns for x in tn]
            calls.append({"input_toks": len(prompt), "output_toks": n_out, "input_tok_ids": prompt,
                          "tool_duration_ns": int(rng.uniform(0.2, 2.0) * 1e9),
                          "output_ids": chat.take(n_out)})
        calls[-1]["tool_duration_ns"] = 0
        for c in calls:  # the trace's output ids are the ones in the next prompt
            c["output_tok_ids"] = c.pop("output_ids")
        out.append({"session_id": f"agent_{len(out)}", "arrival_time_ns": t0, "sub_requests": calls})
    return out


def peruser(rng: random.Random, n: int, rate: float, chat: Pool, _swe: Pool) -> list[dict]:
    """60 users with 2-4 personal prefixes each: the user's text, or its first 30-90% and another end. Every prompt
    is the shared preamble, one prefix and the turn: 200-800 tokens, so two prompts of a user agree for the
    preamble and as much of the prefix as their versions have in common."""
    preamble = chat.take(96)
    users = []
    for _ in range(60):
        base = chat.take(rng.randint(120, 520))
        vs = [(3, base)]
        for _ in range(rng.randint(1, 3)):
            cut = int(len(base) * rng.uniform(0.3, 0.9))
            vs.append((1, base[:cut] + chat.take(rng.randint(40, 160))))
        users.append(vs)
    out = []
    for t in arrivals(rng, n, rate):
        vs = users[zipf_choice(rng, len(users), 0.6)]
        body = rng.choices([v for _, v in vs], weights=[w for w, _ in vs])[0]
        out.append(request(t, (preamble + body + chat.take(rng.randint(30, 150)))[:PERUSER_MAX], rng))
    return out


def longctx(rng: random.Random, n: int, rate: float, _chat: Pool, swe: Pool) -> list[dict]:
    """Prompts of 12k-24k tokens, each the next run of the SWE-bench pool (the pool wraps, at an offset that is
    not the same twice, so no two prompts share a block-aligned prefix)."""
    return [request(t, swe.take(rng.randint(*LONGCTX_PROMPT)), rng) for t in arrivals(rng, n, rate)]


def bursty(rng: random.Random, n: int, rate: float, src: list[dict]) -> list[dict]:
    """The trace's requests in order, arriving as a gamma renewal process of mean ``rate`` and CV ``BURST_CV``:
    shape 1/CV^2, so bursts of near-simultaneous requests between long gaps."""
    shape = 1 / BURST_CV**2
    t, out = 0.0, []
    for d in src[:n]:
        t += rng.gammavariate(shape, 1 / (rate * shape))
        out.append({**d, "arrival_time_ns": int(t * 1e9)})
    return out


def swefast(_rng: random.Random, n: int, rate: float, src: list[dict]) -> list[dict]:
    """The trace's sessions with their arrival times scaled to ``rate`` sessions per second (the tool waits inside
    a session stay)."""
    k = SWE_RATE / rate
    return [{**d, "arrival_time_ns": int(d["arrival_time_ns"] * k)} for d in src[:n]]


GENERATORS = {"tenants": tenants, "fewshot": fewshot, "rag": rag, "agent": agent, "peruser": peruser,
              "longctx": longctx}
SOURCED = {"bursty": (bursty, SHAREGPT), "swefast": (swefast, SWEBENCH)}  # generators of a trace's own lines


def generate(pattern: str, seed: int = 0, n: int | None = None, rate: float | None = None,
             chat: list[list[int]] | None = None, swe: list[list[int]] | None = None,
             src: list[dict] | None = None) -> list[dict]:
    """The workload's JSONL lines (flat requests, or sessions for ``agent`` and ``swefast``). ``chat`` and ``swe``
    are the source documents, ``src`` the trace's own lines for ``bursty`` and ``swefast`` (default: the files on
    this host)."""
    dn, dr = {**PATTERNS, **HELDOUT}[pattern]
    rng = random.Random(f"{pattern}/{seed}")
    if pattern in SOURCED:
        fn, path = SOURCED[pattern]
        return fn(rng, n or dn, rate or dr, src if src is not None else [json.loads(line) for line in open(path)])
    chat_pool = Pool(chat if chat is not None else sharegpt_docs(), rng)
    swe_pool = Pool(swe if swe is not None else swebench_docs(), rng)
    return GENERATORS[pattern](rng, n or dn, rate or dr, chat_pool, swe_pool)


def calls(lines: list[dict]) -> list[tuple[str, int, int, list[int], int]]:
    """(call id, arrival-order key, session-or-flat, prompt ids, output tokens) as the loader numbers them."""
    out = []
    for no, d in enumerate(lines):
        if "sub_requests" in d:
            for i, c in enumerate(d["sub_requests"]):
                out.append((f"{d['session_id']}#{i}", d["arrival_time_ns"] + i, no, c["input_tok_ids"], c["output_toks"]))
        else:
            out.append((f"req_{no}", d["arrival_time_ns"], no, d["input_tok_ids"], d["output_toks"]))
    return out


def _blocks(ids: list[int]) -> list[tuple[int, ...]]:
    return [tuple(ids[i:i + BLOCK]) for i in range(0, len(ids) - len(ids) % BLOCK, BLOCK)]


def reusable(prompts: list[list[int]]) -> list[int]:
    """Per prompt (in the order given), the tokens an unbounded prefix cache would hit: the full blocks it shares
    with the best earlier prompt, never the last token (vLLM recomputes one token)."""
    from workloads import _lcp  # noqa: PLC0415 (the sibling script)
    seen: dict[tuple, list[list[tuple]]] = {}
    out = []
    for ids in prompts:
        bl = _blocks(ids)
        cap = (len(ids) - 1) // BLOCK
        best = max((min(_lcp(bl, o), cap) for o in seen.get(bl[0], [])), default=0) if bl else 0
        out.append(best * BLOCK)
        if bl:
            seen.setdefault(bl[0], []).append(bl)
    return out


def stats(lines: list[dict]) -> dict:
    """Sharing structure by token ids against what prefix groups can state (see the module docstring)."""
    from workloads import blis_groups  # noqa: PLC0415
    cs = sorted(calls(lines), key=lambda c: (c[1], c[0]))
    prompts = [c[3] for c in cs]
    total = sum(map(len, prompts))
    share = sum(reusable(prompts))
    enc = blis_groups([c[0] for c in cs], prompts, [c[4] for c in cs])
    best, greedy = enc["blis_mapping_tokens"], enc["greedy_mapping_tokens"]
    return {"calls": len(cs), "sessions": sum("sub_requests" in d for d in lines), "prompt_tokens": total,
            "mean_prompt": round(total / len(cs)), "max_prompt": max(map(len, prompts)),
            "reusable_tokens": share, "reusable_fraction": round(share / total, 4),
            "prefix_group_tokens": best, "prefix_group_fraction": round(best / total, 4),
            "prefix_group_error_tokens": enc["blis_error_tokens"], "groups": enc["groups"],
            "greedy_group_tokens": greedy, "greedy_group_fraction": round(greedy / total, 4)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="overlap_workloads")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("pattern", choices=[*PATTERNS, *HELDOUT, "all"])
    g.add_argument("out", help="a file, or for 'all' a directory")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--rate", type=float, default=None)
    s = sub.add_parser("stats")
    s.add_argument("workload", nargs="+")
    a = p.parse_args(argv)
    if a.cmd == "gen":
        names = list(PATTERNS) if a.pattern == "all" else [a.pattern]
        if a.pattern == "all":
            os.makedirs(a.out, exist_ok=True)
        for name in names:
            path = os.path.join(a.out, f"{name}.jsonl") if a.pattern == "all" else a.out
            lines = generate(name, a.seed, a.n, a.rate)
            with open(path, "w") as f:
                for d in lines:
                    f.write(json.dumps(d, separators=(",", ":")) + "\n")
            print(path, len(lines))
    else:
        for path in a.workload:
            print(path, json.dumps(stats([json.loads(line) for line in open(path)]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
