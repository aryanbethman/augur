"""The KV-cache spec comes from vLLM's own model definition (augur.sim.kvspec): sliding-window, MLA and mixed
layers get the spec vLLM's GPU model runner would build, and vLLM's KV-cache manager acts on it.

Tiny configs in tests/data/kvspec; no weights, no GPU."""

import json
import os

import pytest
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, SlidingWindowSpec

from augur.cost import FixedCost
from augur.sim.engine import EngineSpec, SimEngine
from augur.sim.kvspec import _cache, layer_kv_specs
from augur.trace.workload import Call

DATA = os.path.join(os.path.dirname(__file__), "data", "kvspec")
BLOCK, BF16 = 16, 2
KV_BYTES = 8 << 20


@pytest.fixture(autouse=True)
def fresh_build():
    """Every test builds the model itself rather than reading a cached build."""
    _cache.clear()


def model_dir(tmp_path, name, **changes):
    """A directory holding the named config (with ``changes`` applied) as config.json, where vLLM reads it."""
    with open(os.path.join(DATA, f"{name}.json")) as f:
        config = {**json.load(f), **changes}
    path = tmp_path / f"{name}-{len(list(tmp_path.iterdir()))}"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def vllm_config(path, **args):
    return EngineArgs(model=path, skip_tokenizer_init=True, dtype="bfloat16", max_model_len=2048,
                      **args).create_engine_config()


def test_sliding_window_layers_get_sliding_window_specs(tmp_path):
    specs = layer_kv_specs(vllm_config(model_dir(tmp_path, "sliding")))
    assert sorted(specs) == [f"model.layers.{i}.self_attn.attn" for i in range(4)]
    for spec in specs.values():
        assert type(spec) is SlidingWindowSpec and spec.sliding_window == 64
        assert (spec.num_kv_heads, spec.head_size, spec.block_size) == (2, 64, BLOCK)
        assert spec.page_size_bytes == 2 * BLOCK * 2 * 64 * BF16  # K and V


def test_mla_layers_cache_one_latent_vector(tmp_path):
    specs = layer_kv_specs(vllm_config(model_dir(tmp_path, "mla")))
    assert len(specs) == 2
    for spec in specs.values():
        assert type(spec) is MLAAttentionSpec
        assert (spec.num_kv_heads, spec.head_size) == (1, 128 + 64)  # kv_lora_rank + qk_rope_head_dim
        assert spec.page_size_bytes == BLOCK * (128 + 64) * BF16  # no separate K and V


def test_mixed_layers_follow_the_config_layer_types(tmp_path):
    specs = layer_kv_specs(vllm_config(model_dir(tmp_path, "mixed")))
    kinds = [type(specs[f"model.layers.{i}.self_attn.attn"]) for i in range(4)]
    assert kinds == [SlidingWindowSpec] * 3 + [FullAttentionSpec]
    assert {spec.page_size_bytes for spec in specs.values()} == {2 * BLOCK * 4 * 64 * BF16}


def test_a_model_vllm_cannot_build_is_an_error_naming_it(tmp_path, monkeypatch):
    def refuse(**kwargs):
        raise RuntimeError("no such layer")
    monkeypatch.setattr("augur.sim.kvspec.initialize_model", refuse)
    with pytest.raises(ValueError, match="MistralForCausalLM.*no such layer"):
        layer_kv_specs(vllm_config(model_dir(tmp_path, "sliding")))


def engine(tmp_path, name, tp=1, **changes):
    spec = EngineSpec(model_dir=model_dir(tmp_path, name, **changes), model_name=f"test/{name}",
                      kv_cache_bytes=KV_BYTES, max_model_len=2048, max_num_batched_tokens=512, max_num_seqs=8,
                      tensor_parallel_size=tp)
    return SimEngine(0, spec, FixedCost(base_ns=1_000_000))


def groups(e):
    return e.core.scheduler.kv_cache_config.kv_cache_groups


def test_kv_manager_gets_the_sliding_window_spec(tmp_path):
    e = engine(tmp_path, "sliding")
    (group,) = groups(e)
    assert type(group.kv_cache_spec) is SlidingWindowSpec and len(group.layer_names) == 4
    assert e.vllm_config.cache_config.num_gpu_blocks == KV_BYTES // (4 * group.kv_cache_spec.page_size_bytes)


def test_kv_heads_are_per_tensor_parallel_rank(tmp_path):
    e = engine(tmp_path, "sliding", tp=2)
    (group,) = groups(e)
    assert group.kv_cache_spec.num_kv_heads == 1
    assert e.ctx.kv_block_bytes == 4 * 2 * BLOCK * 1 * 64 * BF16


def test_kv_manager_groups_mixed_layers_by_kind(tmp_path):
    kinds = {type(g.kv_cache_spec) for g in groups(engine(tmp_path, "mixed"))}
    assert kinds == {FullAttentionSpec, SlidingWindowSpec}


def test_mla_blocks_are_smaller_so_more_fit(tmp_path):
    e = engine(tmp_path, "mla")
    (group,) = groups(e)
    block_bytes = 2 * BLOCK * (128 + 64) * BF16
    assert type(group.kv_cache_spec) is MLAAttentionSpec
    assert e.vllm_config.cache_config.num_gpu_blocks == KV_BYTES // block_bytes
    assert e.ctx.kv_block_bytes == block_bytes


def blocks_held_by_a_long_prompt(e, n_prompt=480):
    ids = list(range(1000, 1000 + n_prompt))
    e.add(Call(call_id="r", session_id=None, index=0, arrival_ns=0, prompt_ids=ids, output_toks=64, output_ids=[],
               tool_wait_ns=0), 0)
    for _ in range(12):
        e.step()
    pool = e.core.scheduler.kv_cache_manager.block_pool
    return pool.num_gpu_blocks - 1 - pool.get_num_free_blocks()  # block 0 is vLLM's null block


def test_sliding_window_frees_blocks_behind_the_window(tmp_path):
    full = blocks_held_by_a_long_prompt(engine(tmp_path, "sliding", sliding_window=None))
    window = blocks_held_by_a_long_prompt(engine(tmp_path, "sliding"))
    assert full >= 480 // BLOCK  # the whole prompt (a block spans all layers)
    assert window <= 64 // BLOCK + 2  # a window's worth of blocks


def test_engines_sharing_a_model_build_it_once_and_get_their_own_dict(tmp_path):
    path = model_dir(tmp_path, "sliding")
    first = layer_kv_specs(vllm_config(path))
    first["draft.layers.0.self_attn.attn"] = None  # an engine adds its draft layers to its copy
    assert "draft.layers.0.self_attn.attn" not in layer_kv_specs(vllm_config(path))
    assert len(_cache) == 1
    layer_kv_specs(vllm_config(path, kv_cache_dtype="fp8"))
    layer_kv_specs(vllm_config(model_dir(tmp_path, "mla")))
    assert len(_cache) == 3
