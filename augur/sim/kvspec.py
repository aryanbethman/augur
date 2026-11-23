"""The KV-cache spec of every layer, asked of vLLM's own model definition instead of assumed.

vLLM's GPU model runner builds the spec from the attention modules the model constructs
(``GPUModelRunner.get_kv_cache_spec``): sliding-window, MLA, Mamba, chunked-local and full-attention layers each
answer with their own spec, and the KV-cache manager's hybrid coordinator groups them.  ``layer_kv_specs`` runs the
same code: it builds the model class vLLM would load (``initialize_model``) on the meta device, so no weights or
GPU memory exist, with the parallel groups stood in for by single-process stubs of the configured sizes, then
collects the specs exactly as the runner does.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from torch.overrides import TorchFunctionMode

import vllm.distributed.parallel_state as parallel_state
from vllm.config import set_current_vllm_config
from vllm.config.vllm import get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec
from vllm.v1.worker.utils import customize_attention_spec

# Module-level groups the model constructors read (get_tp_group() and friends), set to stubs while building.
_SHARED_EXPERTS = "vllm.model_executor.layers.fused_moe.runner.shared_experts"
_GROUPS = ("_WORLD", "_TP", "_PP", "_DP", "_EP", "_PCP", "_DCP", "_ETP", "_EPLB")


@contextmanager
def _parallel_groups(vllm_config):
    """Stub parallel groups: each reports its configured size and rank 0, and does no communication."""
    pc = vllm_config.parallel_config

    def group(size: int):
        process_group = SimpleNamespace(size=lambda: size, rank=lambda: 0)
        return SimpleNamespace(world_size=size, rank_in_group=0, rank=0, local_rank=0, is_first_rank=True,
                               is_last_rank=True, ranks=list(range(size)), device_group=process_group,
                               cpu_group=process_group, device_communicator=None, size=lambda: size)
    sizes = {"_TP": pc.tensor_parallel_size, "_ETP": pc.tensor_parallel_size, "_DP": pc.data_parallel_size,
             "_PP": pc.pipeline_parallel_size}
    saved = {name: getattr(parallel_state, name) for name in _GROUPS}
    try:
        for name in _GROUPS:
            setattr(parallel_state, name, group(sizes.get(name, 1)))
        yield
    finally:
        for name, value in saved.items():
            setattr(parallel_state, name, value)


class _CudaAsMeta(TorchFunctionMode):
    """Tensors a layer asks for on "cuda" (attention backends allocate their workspaces so) land on meta too."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if kwargs.get("device") is not None and torch.device(kwargs["device"]).type == "cuda":
            kwargs = {**kwargs, "device": "meta"}
        return func(*args, **kwargs)


@contextmanager
def _on_meta_device(vllm_config):
    """Build on a machine with no GPU: tensors land on the meta device (also where a model asks the device config
    or a CUDA device for one), and nothing reads the device: attention backends see a GPU of one SM,
    MoE layers need no CUDA stream for shared experts, and importing flashinfer starts no CUDA context."""
    device_config = vllm_config.device_config
    device = device_config.device
    device_config.device = torch.device("meta")
    try:
        with torch.device("meta"), _CudaAsMeta(), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(multi_processor_count=1)), \
                patch(f"{_SHARED_EXPERTS}.aux_stream", return_value=None), \
                patch("torch.cuda.device_count", return_value=0):
            yield
    finally:
        device_config.device = device


def layer_kv_specs(vllm_config) -> dict[str, KVCacheSpec]:
    """{layer name: KV spec} of the model in ``vllm_config``, as vLLM's GPU model runner builds it.

    Raises ValueError when vLLM's model class cannot be built here, or a layer answers with something that is not
    a KV-cache spec."""
    arch = vllm_config.model_config.architecture
    compilation = vllm_config.compilation_config
    registered, compilation.static_forward_context = compilation.static_forward_context, {}
    try:
        with _parallel_groups(vllm_config), set_current_vllm_config(vllm_config), _on_meta_device(vllm_config):
            with set_default_torch_dtype(vllm_config.model_config.dtype):
                initialize_model(vllm_config=vllm_config)
            layers = get_layers_from_vllm_config(vllm_config, cast(type[Any], AttentionLayerBase))
            specs = _collect(vllm_config, layers)
    except Exception as e:
        raise ValueError(f"cannot derive the KV-cache spec of {arch} from vLLM's model definition: "
                         f"{type(e).__name__}: {e}") from e
    finally:
        compilation.static_forward_context = registered
    if not specs:
        raise ValueError(f"{arch} has no layer that takes KV cache")
    return specs


def _collect(vllm_config, layers) -> dict[str, KVCacheSpec]:
    """Mirror of GPUModelRunner.get_kv_cache_spec over the constructed layers."""
    specs: dict[str, KVCacheSpec] = {}
    for name, module in layers.items():
        if isinstance(module, Attention) and module.kv_sharing_target_layer_name:
            continue  # uses another layer's cache: vLLM's KV manager does not see it
        spec = module.get_kv_cache_spec(vllm_config)
        if spec is None:
            continue  # e.g. encoder-only attention
        if not isinstance(spec, KVCacheSpec):
            raise ValueError(f"layer {name} ({type(module).__name__}) returned {type(spec).__name__}, not a KV spec")
        if isinstance(spec, AttentionSpec):
            spec = customize_attention_spec(module.get_attn_backend(), spec)
        specs[name] = spec
    return specs
