#!/usr/bin/env python3
"""Quantize routed-expert weights to NVFP4-style W4A16.

fp4 e2m1 values (grid {0,.5,1,1.5,2,3,4,6} +/- sign) packed 2/byte along K,
fp16 scales per 16-element K-block (scale = blockwise amax / 6, RTN).
Only `*.mixer.experts.<e>.(up|down)_proj.weight` tensors are converted:
  <name>.weight  ->  <name>.qweight (uint8, N x K/2)  +  <name>.scales (fp16, N x K/16)
Everything else is copied through unchanged.

Usage: python3 quantize_experts.py SRC_DIR DST_DIR
"""
import json
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT_RE = re.compile(r"\.mixer\.experts\.\d+\.(up|down)_proj\.weight$")
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# midpoints for RTN bucketing (ties round toward the lower code, fine for RTN)
MIDS = (GRID[1:] + GRID[:-1]) / 2


def quantize_w4(w):
    """w (N, K) -> qweight (N, K//2) uint8, scales (N, K//16) fp16."""
    N, K = w.shape
    assert K % 16 == 0
    wf = w.float().cuda()
    blocks = wf.view(N, K // 16, 16)
    amax = blocks.abs().amax(dim=-1)                       # (N, K/16)
    scales = (amax / 6.0).clamp(min=1e-8)
    v = (blocks / scales.unsqueeze(-1)).reshape(N, K)      # in [-6, 6]
    sign = (v < 0).to(torch.uint8)
    mag = v.abs()
    idx = torch.bucketize(mag, MIDS.to(mag.device)).to(torch.uint8)  # 0..7
    code = (sign << 3) | idx                               # 4-bit code
    lo = code[:, 0::2]
    hi = code[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()
    return packed.cpu(), scales.half().cpu()


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    idx = json.load(open(src / "model.safetensors.index.json"))
    weight_map = idx["weight_map"]
    new_map = {}
    n_q = 0
    t0 = time.perf_counter()
    for shard in sorted(set(weight_map.values())):
        out = {}
        with safe_open(src / shard, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if EXPERT_RE.search(k):
                    base = k[: -len(".weight")]
                    qw, sc = quantize_w4(t)
                    out[base + ".qweight"] = qw
                    out[base + ".scales"] = sc
                    new_map[base + ".qweight"] = shard
                    new_map[base + ".scales"] = shard
                    n_q += 1
                else:
                    out[k] = t
                    new_map[k] = shard
        save_file(out, dst / shard, metadata={"format": "pt"})
        print(f"  {shard}: done ({n_q} experts quantized so far, "
              f"{time.perf_counter()-t0:.0f}s)", flush=True)
    json.dump({"metadata": {"quant": "nvfp4-style w4a16 experts"},
               "weight_map": new_map},
              open(dst / "model.safetensors.index.json", "w"))
    for aux in ["config.json", "configuration_nemotron_h.py",
                "modeling_nemotron_h.py", "modeling_nemotron_twotower.py",
                "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "generation_config.json"]:
        if (src / aux).exists():
            shutil.copy(src / aux, dst / aux)
    print(f"DONE: {n_q} expert mats quantized in "
          f"{time.perf_counter()-t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
