"""Token-faithful ngram speculation: with recorded output ids, drafts come from the drafter of vLLM's model runner
over the real context and are accepted exactly where they equal the next recorded ids; without them
--spec-accept-rate decides.

GPU-free: constructed traces, a fixed-cost engine, an independent model of the drafter and the verification, the V2
runner's Triton kernels run on CPU (TRITON_INTERPRET), and requests of the real ShareGPT 300 ngram runs
(tests/data/ngram_real_*.json.gz: prompt and generated ids, and the tokens each step of the real log emitted)."""

import csv
import glob
import gzip
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from augur.sim.ngram import Context, as_int32, v2_drafts
from augur.verify.replay import compare

K, MIN_N, MAX_N = 3, 2, 4
BASE = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed", "--num-gpu-blocks", "400", "--max-num-seqs", "16",
        "--max-model-len", "4096", "--max-num-batched-tokens", "1024", "--fixed-step-ms", "8",
        "--fixed-per-token-us", "15", "--step-cpu-us", "0", "--quiet", "--num-speculative-tokens", str(K)]
SCHEDULING = {"sync": "--no-async-scheduling", "async": "--async-scheduling"}


def ids(rng, n):
    return [rng.randrange(1000, 100_000) for _ in range(n)]


def write_trace(path, calls):
    """calls: (arrival_s, prompt ids, output ids or None)."""
    with open(path, "w") as f:
        for t, prompt, out in calls:
            d = {"input_toks": len(prompt), "output_toks": len(out), "arrival_time_ns": int(t * 1e9),
                 "input_tok_ids": prompt, "output_tok_ids": out}
            f.write(json.dumps(d) + "\n")
    return str(path)


def augur(workload, *args, tmp_path=None, **env):
    out = tmp_path / "out.csv"
    r = subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", workload, "--output", str(out), *args],
                       capture_output=True, text=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING", **env))
    assert r.returncode == 0, r.stderr[-2000:]
    with open(out) as f:
        return list(csv.DictReader(f))


def emitted_per_step(log, rid="req_0"):
    """Tokens `rid` emitted at each step of augur's own step log, steps where it emitted nothing skipped."""
    events = [json.loads(line) for line in open(glob.glob(f"{log}.*.jsonl")[0])]
    return [e["out"][rid] for e in events if e["ev"] == "upd" and e["out"].get(rid)]


def drafts(context, v2=True):
    """The drafts for the context, by brute force: the longest suffix n-gram (MAX_N down to MIN_N) that occurred
    earlier, and the k ids after it.  V2 runner (NgramGPUSpeculator): the latest such occurrence, padded to k with
    the last id.  V1 runner (NgramProposer): the earliest, and fewer than k when the context ends first."""
    L = len(context)
    for n in range(MAX_N, MIN_N - 1, -1):
        hits = [p for p in range(L - n) if context[p:p + n] == context[L - n:]]
        if hits:
            start = (hits[-1] if v2 else hits[0]) + n
            d = context[start:start + K]
            break
    else:
        d = []
    return d + [context[-1]] * (K - len(d)) if v2 else d


def expected_emits(prompt, out, v2=True):
    """Tokens emitted per step: exact greedy verification of those drafts against the recorded output."""
    context, emits, sent = list(prompt), [], 0
    pending = []  # drafts for the next step (none before the first token)
    while sent < len(out):
        accepted = 0
        while accepted < len(pending) and sent + accepted < len(out) and pending[accepted] == out[sent + accepted]:
            accepted += 1
        n = min(accepted + 1, len(out) - sent)
        context += out[sent:sent + n]
        emits.append(n)
        sent += n
        pending = drafts(context, v2)
    return emits


@pytest.fixture(scope="module")
def repeating():
    """Outputs that repeat a 6-id phrase (drafts are right while the phrase repeats), and outputs with no repeats."""
    rng = random.Random(7)
    prompt = ids(rng, 150)
    phrase = ids(rng, 6)
    return prompt, ids(rng, 5) + phrase * 10, ids(rng, 80)


@pytest.mark.parametrize("mode", SCHEDULING)
def test_acceptance_per_step_is_the_text_s(tmp_path, repeating, mode):
    prompt, loop, noise = repeating
    for name, out in (("loop", loop), ("noise", noise)):
        wl = write_trace(tmp_path / f"{name}.jsonl", [(0.0, prompt, out)])
        augur(wl, SCHEDULING[mode], "--step-log-vllm", str(tmp_path / name), tmp_path=tmp_path)
        got = emitted_per_step(str(tmp_path / name))
        want = expected_emits(prompt, out)
        assert sum(got) == len(out)
        assert got == want, (name, got, want)
    assert max(expected_emits(prompt, loop)) == K + 1 and set(expected_emits(prompt, noise)) == {1}


def test_a_request_never_runs_past_its_recorded_length(tmp_path, repeating):
    prompt, loop, _ = repeating
    for n in range(len(loop) - 3, len(loop)):  # ends mid-phrase, with accepted drafts to spare
        wl = write_trace(tmp_path / "cut.jsonl", [(0.0, prompt, loop[:n])])
        rows = augur(wl, "--no-async-scheduling", "--step-log-vllm", str(tmp_path / f"cut{n}"), tmp_path=tmp_path)
        assert int(rows[0]["output"]) == n and sum(emitted_per_step(str(tmp_path / f"cut{n}"))) == n


def test_without_recorded_tokens_the_rate_decides(tmp_path, repeating):
    prompt, loop, _ = repeating
    wl = write_trace(tmp_path / "t.jsonl", [(0.0, prompt, loop)])
    for rate, steps in (("0", len(loop) - 1), ("1", -(-(len(loop) - 1) // (K + 1)))):
        augur(wl, "--no-async-scheduling", "--no-recorded-outputs", "--spec-accept-rate", rate,
              "--step-log-vllm", str(tmp_path / f"r{rate}"), tmp_path=tmp_path)
        got = emitted_per_step(str(tmp_path / f"r{rate}"))
        assert len(got) == steps + 1 and sum(got) == len(loop), (rate, got)
    augur(wl, "--no-async-scheduling", "--spec-rate-only", "--spec-accept-rate", "0",
          "--step-log-vllm", str(tmp_path / "only"), tmp_path=tmp_path)
    assert set(emitted_per_step(str(tmp_path / "only"))) == {1}  # --spec-rate-only: the rate even with recorded ids


def test_calls_are_drafted_independently_in_a_mixed_batch(tmp_path, repeating):
    prompt, loop, noise = repeating
    rng = random.Random(3)
    calls = [(0.0, ids(rng, 200), loop), (0.0, ids(rng, 200), noise), (0.01, ids(rng, 300), loop[::-1])]
    wl = write_trace(tmp_path / "mix.jsonl", calls)
    augur(wl, "--no-async-scheduling", "--step-log-vllm", str(tmp_path / "mix"), tmp_path=tmp_path)
    for i, (_, p, out) in enumerate(calls):
        assert emitted_per_step(str(tmp_path / "mix"), f"req_{i}") == expected_emits(p, out), i


def test_generated_text_shares_prefix_cache_blocks(tmp_path, repeating):
    """A later call whose prompt holds an earlier call's generated text hits the cache on it (block hashes follow
    the emitted ids), also when the first call decoded with speculation."""
    prompt, loop, _ = repeating
    follow = prompt + loop + ids(random.Random(1), 40)
    wl = write_trace(tmp_path / "chat.jsonl", [(0.0, prompt, loop), (5.0, follow, loop[:20])])
    for flags in ([], ["--no-recorded-outputs"]):
        rows = augur(wl, "--no-async-scheduling", *flags, tmp_path=tmp_path)
        cached = int(sorted(rows, key=lambda r: int(r["input"]))[1]["cached_tokens"])
        assert cached == len(prompt if flags else prompt + loop) // 16 * 16


@pytest.mark.parametrize("mode", SCHEDULING)
def test_replay_of_own_log_is_exact(tmp_path, repeating, mode):
    rng = random.Random(5)
    prompt, loop, noise = repeating
    calls = [(0.02 * i, ids(rng, rng.randrange(60, 900)), (loop, noise, loop[:30])[i % 3]) for i in range(30)]
    wl = write_trace(tmp_path / "many.jsonl", calls)
    augur(wl, SCHEDULING[mode], "--step-log-vllm", str(tmp_path / "orig"), tmp_path=tmp_path)
    (orig,) = glob.glob(str(tmp_path / "orig.*.jsonl"))
    augur(wl, SCHEDULING[mode], "--replay-durations", orig, "--step-log-vllm", str(tmp_path / "rep"), tmp_path=tmp_path)
    (rep,) = glob.glob(str(tmp_path / "rep.*.jsonl"))
    r = compare(orig, rep)
    assert r["steps_a"] > 50 and r["identical_steps"] == r["steps_a"] == r["steps_b"], r["at_first_difference"]


def test_filler_ids_beyond_int32_match_nothing():
    a = as_int32([5, 1 << 40, 7, (1 << 40) + 1, 5])
    assert a.dtype == np.int32 and len(set(a[[1, 3]])) == 2 and (a[[1, 3]] < 0).all()
    assert list(a[[0, 2, 4]]) == [5, 7, 5]
    c = Context([1, 2, 3], spare=0)
    c.append([4, 5, 6, 7])
    assert list(c.ids) == [1, 2, 3, 4, 5, 6, 7]


def test_the_v1_runner_uses_vllm_s_cpu_proposer(tmp_path, repeating):
    prompt, loop, noise = repeating
    wl = write_trace(tmp_path / "t.jsonl", [(0.0, prompt, loop)])
    augur(wl, "--no-async-scheduling", "--step-log-vllm", str(tmp_path / "v1"), tmp_path=tmp_path,
          VLLM_USE_V2_MODEL_RUNNER="0")
    got = emitted_per_step(str(tmp_path / "v1"))
    assert got == expected_emits(prompt, loop, v2=False)


KERNELS = """
import json, os, sys
os.environ["TRITON_INTERPRET"] = "1"
import torch
from vllm.v1.worker.gpu.spec_decode.ngram import speculator as S
rows, K, MIN_N, MAX_N, L, BL = json.load(sys.stdin)
B, NB = len(rows), -(-L // BL)
tok = torch.zeros((B, L), dtype=torch.int32)
for i, r in enumerate(rows):
    tok[i, :len(r)] = torch.tensor(r, dtype=torch.int32)
total = torch.tensor([len(r) for r in rows], dtype=torch.int32)
idx = torch.arange(B, dtype=torch.int64)
last = torch.tensor([r[-1] for r in rows], dtype=torch.int64)
sampled = torch.ones(B, dtype=torch.int32)
scratch = torch.zeros((B, NB), dtype=torch.int64)
drafts = torch.zeros((B, K), dtype=torch.int64)
S._ngram_scan_kernel[(B, NB)](tok, tok.stride(0), idx, total, sampled, scratch, scratch.stride(0), L, MIN_N, MAX_N,
                              max(1, 1 << (MAX_N - 1).bit_length()), BL)
S._ngram_finalize_kernel[(B,)](tok, tok.stride(0), idx, total, sampled, last, scratch, scratch.stride(0), drafts, L,
                               NB, K, max(1, 1 << (K - 1).bit_length()), max(1, 1 << (NB - 1).bit_length()))
print(json.dumps(drafts.tolist()))
"""


@pytest.mark.parametrize("k,min_n,max_n", [(3, 2, 4), (5, 1, 3)])
def test_the_numpy_drafter_matches_the_v2_triton_kernels(k, min_n, max_n):
    """vLLM's own kernels, run on CPU by Triton's interpreter, give the drafts v2_drafts does (small alphabets make
    matches, ties and contexts shorter than the n-gram frequent; the context spans several kernel blocks)."""
    rng = random.Random(11)
    rows = [[rng.randrange(alphabet) for _ in range(n)]
            for alphabet, n in ((3, 40), (5, 90), (8, 150), (30, 200), (2, 7), (4, 300), (1, 20), (6, 1), (50, 120))]
    r = subprocess.run([sys.executable, "-c", KERNELS], input=json.dumps([rows, k, min_n, max_n, 512, 128]),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout.splitlines()[-1]) == [v2_drafts(as_int32(c), k, min_n, max_n) for c in rows]


def real_slice(name):
    with gzip.open(os.path.join(os.path.dirname(__file__), "data", f"ngram_real_{name}.json.gz"), "rt") as f:
        return json.load(f)


@pytest.mark.parametrize("name", ["sync", "async"])
def test_the_drafter_reproduces_every_logged_step_of_real_requests(name):
    """Requests of a real vLLM 0.33 run (ngram k=3, V2 runner): from the context after each logged step, the
    drafter and the exact verification give the number of tokens that step emitted."""
    calls = real_slice(name)
    for c in calls:
        prompt, out = c["prompt_ids"], c["output_ids"]
        assert expected_emits(prompt, out) == c["emitted"], c["call_id"]
        # and the numpy drafter agrees with the brute force at every step
        done = 0
        for n in c["emitted"][:-1]:
            done += n
            context = prompt + out[:done]
            assert v2_drafts(as_int32(context), K, MIN_N, MAX_N) == drafts(context)


@pytest.mark.parametrize("name", ["sync", "async"])
def test_the_simulator_reproduces_the_real_acceptance_per_step(tmp_path, name):
    """The same requests through augur's engine (their recorded real generations): each emits what it did in the
    real log, step for step, in sync and async scheduling."""
    calls = real_slice(name)
    wl = write_trace(tmp_path / "real.jsonl", [(0.0, c["prompt_ids"], c["output_ids"]) for c in calls])
    augur(wl, SCHEDULING[name], "--step-log-vllm", str(tmp_path / "real"), tmp_path=tmp_path)
    for i, c in enumerate(calls):
        assert emitted_per_step(str(tmp_path / "real"), f"req_{i}") == c["emitted"], c["call_id"]
