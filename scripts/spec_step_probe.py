"""Where does a speculative decode step's extra time come from? (1) CPU: vLLM's ngram proposer on a batch;
(2) GPU: FlashAttention decode with query length 1 vs 1+k at the same context."""
import sys, time
import numpy as np, torch

def cpu():
    from vllm.config import VllmConfig, SpeculativeConfig, ModelConfig
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer
    mc = ModelConfig("Qwen/Qwen3-1.7B", max_model_len=8192)
    cfg = VllmConfig(model_config=mc, speculative_config=SpeculativeConfig(method="ngram", num_speculative_tokens=3, prompt_lookup_max=4, prompt_lookup_min=2))
    p = NgramProposer(cfg)
    rng = np.random.default_rng(0)
    for n_req, ctx in ((48, 1200), (48, 3000), (128, 1200)):
        toks = rng.integers(0, 150000, size=(n_req, 8192), dtype=np.int32)
        ntok = np.full(n_req, ctx, dtype=np.int32)
        valid = list(range(n_req))
        toks[:, :] = rng.integers(0, 150, size=toks.shape, dtype=np.int32)  # small vocab: matches found, as repetitive output
        t = []
        for _ in range(25):
            t0 = time.perf_counter()
            p.batch_propose(n_req, valid, ntok, toks, 3)
            t.append(time.perf_counter() - t0)
        print(f"ngram propose, {n_req} requests x {ctx} ctx: median {np.median(t)*1e3:.2f} ms")

def gpu():
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    dev = "cuda"; H, KVH, D, bs = 16, 8, 128, 16
    for B, ctx in ((48, 1200), (48, 3000), (128, 1200)):
        nblk = (ctx + 15) // 16 + 1
        kc = torch.randn(B * nblk, bs, KVH, D, device=dev, dtype=torch.bfloat16); vc = torch.randn_like(kc)
        bt = torch.arange(B * nblk, device=dev, dtype=torch.int32).view(B, nblk)
        seq = torch.full((B,), ctx, device=dev, dtype=torch.int32)
        res = {}
        for q in (1, 4):
            qq = torch.randn(B * q, H, D, device=dev, dtype=torch.bfloat16)
            cu = torch.arange(0, (B + 1) * q, q, device=dev, dtype=torch.int32)
            f = lambda: flash_attn_varlen_func(qq, kc, vc, cu_seqlens_q=cu, max_seqlen_q=q, seqused_k=seq, max_seqlen_k=ctx,
                                               block_table=bt, causal=True)
            for _ in range(5): f()
            torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(50): f()
            e1.record(); torch.cuda.synchronize(); res[q] = e0.elapsed_time(e1) / 50
        kv_gb = B * ctx * KVH * D * 2 * 2 / 1e9
        print(f"attention one layer, B={B} ctx={ctx}: q=1 {res[1]*1e3:.0f} us, q=4 {res[4]*1e3:.0f} us (x{res[4]/res[1]:.2f}); KV {kv_gb*1e3:.0f} MB = {kv_gb/288*1e6:.0f} us at 288 GB/s")

if __name__ == "__main__":
    {"cpu": cpu, "gpu": gpu}[sys.argv[1]]()
