#!/usr/bin/env python3
"""Single-node NVFP4 TwoTower — both towers on ONE GB10 (cuda:0).

The distributed runner split the towers across .3/.4 because bf16 towers were
59 GB each. NVFP4 quantization dropped each to ~21 GB, so both now fit on one
GPU (42 GB weights + caches) — this is the node-plan unlock that frees a node
and lets TwoTower co-reside with another serve (e.g. Nemotron-Omni) on .3.

Loads context_tower.* from tt-context-nvfp4 and denoiser_tower.*/heads from
tt-denoiser-nvfp4 into a single NemotronHTwoTowerForCausalLM, attaches grouped
NVFP4 experts on BOTH towers, and runs NVIDIA's own generate_mask_diffusion
(which already handles the two towers in one process — with both on cuda:0 the
cross-device copies become no-ops).
"""
import argparse
import json
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))  # mamba_ssm shim + tt_kernels + tt_nvfp4

import torch  # noqa: E402
import tt_kernels  # noqa: E402
from twotower_dist import make_model_class, patch_fast_moe  # noqa: E402


def load_both_towers(ctx_dir, den_dir, device, compile_kernels=True):
    sys.path.insert(1, ctx_dir)
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from configuration_nemotron_h import NemotronHConfig
    import modeling_nemotron_h as mh
    import modeling_nemotron_twotower as ttm
    from tt_nvfp4 import attach_quant_experts

    patch_fast_moe(mh)
    if compile_kernels:
        try:
            tt_kernels.ssd_scan_block = torch.compile(tt_kernels.ssd_scan_block, dynamic=False)
            tt_kernels.conv1d_block = torch.compile(tt_kernels.conv1d_block, dynamic=False)
            print("[single] tt_kernels compiled", flush=True)
        except Exception as e:
            print(f"[single] compile skipped: {e}", flush=True)

    cfg = NemotronHConfig.from_pretrained(ctx_dir)
    cls = make_model_class(ttm)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with init_empty_weights():
        model = cls(cfg)
    torch.set_default_dtype(prev)

    sd, qsd = {}, {}

    def ingest(d, prefixes):
        idx = json.load(open(Path(d) / "model.safetensors.index.json"))
        for shard in sorted(set(idx["weight_map"].values())):
            with safe_open(Path(d) / shard, framework="pt", device=device) as f:
                for k in f.keys():
                    if not k.startswith(prefixes):
                        continue
                    (qsd if k.endswith((".qweight", ".scales")) else sd)[k] = f.get_tensor(k)

    t0 = time.perf_counter()
    ingest(ctx_dir, ("context_tower.", "context_lm_head."))
    ingest(den_dir, ("denoiser_tower.", "lm_head.", "t_embedder.",
                     "t_block.", "scale_shift_tables."))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    bad = [k for k in missing if k.replace(".weight", ".qweight") not in qsd]
    assert not bad, f"missing non-quant keys: {bad[:5]}"
    assert not unexpected, f"unexpected: {unexpected[:5]}"
    attach_quant_experts(model, ["context_tower", "denoiser_tower"], qsd, cfg, device)
    # any residual meta params (e.g. time-cond created but unused) -> device
    for n, p in model.named_parameters():
        if p.device.type == "meta":
            raise RuntimeError(f"still meta: {n}")
    model.eval()
    print(f"[single] both towers loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    return model, cfg, ttm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="/models/ctx")
    ap.add_argument("--den", default="/models/den")
    ap.add_argument("--prompt", default="France is a country ")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps-per-block", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--confidence-threshold", type=float, default=0.8)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    device = "cuda:0"
    torch.set_grad_enabled(False)

    model, cfg, _ = load_both_towers(a.ctx, a.den, device,
                                     compile_kernels=not a.no_compile)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ctx)
    ids = tok(a.prompt, return_tensors="pt").input_ids.to(device)

    # warmup (compile) + timed run
    for label in ("warmup", "timed"):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = model.generate_mask_diffusion(
            ids, max_new_tokens=(32 if label == "warmup" else a.max_new_tokens),
            block_size=a.block_size, steps_per_block=a.steps_per_block,
            mask_token_id=3, temperature=a.temperature,
            confidence_threshold=a.confidence_threshold,
            eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n = out.shape[1] - ids.shape[1]
        peak = torch.cuda.max_memory_allocated() / 1e9
        resv = torch.cuda.max_memory_reserved() / 1e9
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        nfe = getattr(model, "_last_nfe", None)
        print(f"\n[{label}] {n} tok, {nfe} NFE, {dt:.2f}s, {n/dt:.2f} tok/s, "
              f"peak_alloc={peak:.1f}GB peak_reserved={resv:.1f}GB", flush=True)
        if label == "timed":
            print("OUTPUT:", text, flush=True)


if __name__ == "__main__":
    main()
