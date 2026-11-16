"""vLLM's NIXL connector with the network side switched off.

The scheduler half (which requests wait for remote KV, block pinning on the
prefill side, the kv_transfer_params it returns) is vLLM's own and runs
unchanged. Only the side-channel handshake is disabled: at engine start it
binds a ZeroMQ listener for worker-to-worker NIXL metadata, which a simulator
has no use for and which collides when many engines share one process. The
worker half is never built; augur's executor stands in for it.
"""

from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlConnector


class SimNixlConnector(NixlConnector):
    def set_xfer_handshake_metadata(self, *args, **kwargs) -> None:
        return None

    def set_xfer_handshake_metadata_pp_aware(self, *args, **kwargs) -> None:
        return None
