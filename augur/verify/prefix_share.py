"""Which KV a real run's requests shared, rebuilt from its step log and workload.

A step log says which requests ran and how many tokens of each were in the KV cache, not which cache blocks.  With
prefix caching, requests whose prompts start with the same tokens share blocks, and ``TableCost`` prices a decode
batch lower when its requests read the same blocks (``SeqWork.shared_kv``).  augur's own engine reads the block ids
from vLLM's scheduler; for a real run this module rebuilds them from the prompts in the workload the run was driven
by.  A block is named by the hash of every prompt token up to its end, as vLLM's prefix cache names it, and requests
whose prompts have the block's name in common hold one physical block.  Checked against the block ids of augur's own
engine replaying a real log (tenants and agent runs): the tokens shared among the decodes of a step were equal in
every step.  Only prompt tokens are hashed, so blocks of generated text count as private.
"""

from __future__ import annotations

from augur.trace.workload import Session, Workload

BLOCK_SIZE = 16


def block_chain(ids, block_size: int = BLOCK_SIZE) -> list[int]:
    """The id of each full block of ``ids``: a hash of every token up to the end of the block, so equal ids mean
    equal prefixes. Called by: prompt_chains, scripts/baselines/workloads.py."""
    out, h = [], 0
    for i in range(len(ids) // block_size):
        h = hash((h, tuple(ids[i * block_size:(i + 1) * block_size])))
        out.append(h)
    return out


def prompt_chains(path: str, block_size: int = BLOCK_SIZE) -> dict[str, list[int]]:
    """Call id -> ``block_chain`` of its prompt."""
    workload = Workload(path=path)
    chains = {}
    for entry in workload.entries():
        if isinstance(entry, Session):  # every turn of a session, as the real client sends them
            for i, sub in enumerate(entry.sub_requests):
                call = workload.make_call(sub, f"{entry.session_id}#{i}", entry.session_id, i, 0)
                chains[call.call_id] = block_chain(call.prompt_ids, block_size)
        else:
            chains[entry.call_id] = block_chain(entry.prompt_ids, block_size)
    return chains


def shared_tokens(decodes, block_size: int = BLOCK_SIZE) -> dict[str, int]:
    """For the decodes of one step, ``(request id, its block ids, tokens of KV already computed)``: request id -> tokens
    of that KV in blocks an earlier decode of the step also holds (the total does not depend on the order). Only full
    blocks of the computed KV count. Called by: SimWorker.execute_model (vLLM's block ids),
    augur.verify.stepcheck.real_steps (``prompt_chains``)."""
    seen: set[int] = set()
    out: dict[str, int] = {}
    for rid, blocks, before in decodes:
        held = set(blocks[:before // block_size])
        if held:
            out[rid] = len(held & seen) * block_size
            seen |= held
    return out
