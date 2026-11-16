"""Replay an augur/LLMServingSim JSONL trace against a real `vllm serve`.

Prompts are sent as the same token ids augur would use (augur.trace.workload builds
them), with exact output lengths (`min_tokens = max_tokens`, `ignore_eos`) and
request ids `augur.<call_id>.`, so a real run lines up with an augur run call
for call (start the server with `--scheduler-cls augur.verify.hooks.Logging...`
and AUGUR_STEP_LOG set to also get its step log). Flat requests are sent at
their arrival times; a session's next call is sent when the previous one
finishes plus its recorded tool wait.

Writes per-call results in augur's CSV format, plus the generated token ids
(JSONL), which differ from the trace's recorded outputs: the model generates
its own tokens.

    python -m augur.verify.client --url http://localhost:8000 --model Qwen/Qwen3-1.7B \\
        --workload trace.jsonl --output out/real.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp

from augur.trace.metrics import CallRecord, Recorder
from augur.trace.workload import Call, Session, Workload


class Client:
    def __init__(self, url: str, model: str, workload: Workload, tokens_out: str | None):
        self.url = url.rstrip("/") + "/v1/completions"
        self.model = model
        self.workload = workload
        self.recorder = Recorder()
        self.failed: list[tuple[str, str]] = []
        self._ids = 0
        self._t0 = 0
        self._tok_f = open(tokens_out, "w") if tokens_out else None

    def _now(self) -> int:
        return time.monotonic_ns() - self._t0

    async def _send(self, http: aiohttp.ClientSession, call: Call) -> None:
        rec = CallRecord(num_id=self._ids, call_id=call.call_id, instance_id=0, model=self.model,
                         input_toks=len(call.prompt_ids), output_toks=call.output_toks,
                         arrival_ns=self._now(), session_id=call.session_id, index=call.index)
        self._ids += 1
        body = {"model": self.model, "prompt": call.prompt_ids, "max_tokens": call.output_toks,
                "min_tokens": call.output_toks, "ignore_eos": True, "temperature": 0.0,
                "stream": True, "return_token_ids": True, "request_id": f"augur.{call.call_id}."}
        generated: list[int] = []
        async with http.post(self.url, json=body) as resp:
            if resp.status != 200:
                self.failed.append((call.call_id, f"HTTP {resp.status}: {(await resp.text())[:200]}"))
                return
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices", []):
                    ids = choice.get("token_ids") or []
                    n = len(ids) if ids else (1 if choice.get("text") else 0)
                    if n:
                        rec.token_times.extend([self._now()] * n)
                        generated.extend(ids)
        rec.end_ns = self._now()
        if len(rec.token_times) != call.output_toks:
            self.failed.append((call.call_id, f"{len(rec.token_times)} of {call.output_toks} tokens"))
            return
        self.recorder.done.append(rec)
        if self._tok_f:
            self._tok_f.write(json.dumps({"call_id": call.call_id, "output_tok_ids": generated}) + "\n")

    async def _session(self, http, s: Session) -> None:
        while s.next_index < len(s.sub_requests):
            call = self.workload.next_call(s, self._now())
            await self._send(http, call)
            if s.next_index < len(s.sub_requests) and call.tool_wait_ns:
                await asyncio.sleep(call.tool_wait_ns / 1e9)

    async def run(self) -> Recorder:
        timeout = aiohttp.ClientTimeout(total=None)
        # No connection cap: aiohttp's default of 100 would queue calls in the client.
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as http:
            self._t0 = time.monotonic_ns()
            tasks = []
            for e in self.workload.entries():
                wait = e.arrival_ns - self._now()
                if wait > 0:
                    await asyncio.sleep(wait / 1e9)
                tasks.append(asyncio.create_task(
                    self._session(http, e) if isinstance(e, Session) else self._send(http, e)))
            await asyncio.gather(*tasks)
        if self._tok_f:
            self._tok_f.close()
        return self.recorder


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="augur.verify.client")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--workload", required=True)
    p.add_argument("--num-entries", type=int, default=None)
    p.add_argument("--output", required=True, help="per-call CSV (augur / fork format)")
    p.add_argument("--tokens-out", default=None, help="generated token ids per call (JSONL)")
    p.add_argument("--filler-vocab", type=int, default=150000,
                   help="unrecorded prompt tokens get in-vocabulary ids below this (default suits Qwen3)")
    a = p.parse_args(argv)
    client = Client(a.url, a.model, Workload(a.workload, max_entries=a.num_entries, filler_vocab=a.filler_vocab),
                    a.tokens_out)
    rec = asyncio.run(client.run())
    rec.write_csv(a.output)
    summary = rec.summary()
    summary["failed"] = len(client.failed)
    print(json.dumps(summary, indent=2, default=float))
    for cid, why in client.failed[:5]:
        print("failed:", cid, why, file=sys.stderr)
    return 1 if client.failed else 0


if __name__ == "__main__":
    sys.exit(main())
