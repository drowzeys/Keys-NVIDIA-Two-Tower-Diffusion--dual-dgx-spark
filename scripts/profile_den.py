#!/usr/bin/env python3
"""Profile denoiser NFE: optimized (fast-MoE + optional compiled kernels)
vs original MoE, with a correctness check between the two."""
import json, os, sys, time
from pathlib import Path

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import torch

sys.argv = ["profile_den.py"]


class A:
    role = "den"
    model = "/model"


import importlib.util as iu
spec = iu.spec_from_file_location("twotower_dist", str(WORK / "twotower_dist.py"))
td = iu.module_from_spec(spec)
spec.loader.exec_module(td)

torch.set_grad_enabled(False)
COMPILE = bool(int(os.environ.get("COMPILE", "1")))
model, cfg, _ = td.load_model(A, "cuda:0", fast_moe=True,
                              compile_kernels=COMPILE)
import modeling_nemotron_h as mh

B, CTX = 1, 64
mc = td.MirrorCache(cfg.num_hidden_layers)
g = torch.Generator(device="cuda").manual_seed(7)
def rnd(shape, dtype):
    return (torch.randn(shape, generator=g, device="cuda",
                        dtype=torch.float32) * 0.02).to(dtype)
for i, kind in enumerate(td.layer_kinds(cfg)):
    if kind == "M":
        mc.conv_states[i] = rnd(td.conv_shape(cfg, B), torch.bfloat16)
        mc.ssm_states[i] = rnd(td.ssm_shape(cfg, B), torch.float32)
    elif kind == "*":
        mc.key_cache[i] = rnd(td.kv_shape(cfg, B, CTX), torch.bfloat16)
        mc.value_cache[i] = rnd(td.kv_shape(cfg, B, CTX), torch.bfloat16)

xt = torch.full((B, 16), 3, dtype=torch.long, device="cuda")
t_vec = torch.tensor([1.0], device="cuda")


def bench(label, n=5, warmup=2):
    for _ in range(warmup):
        model._run_denoiser_step_diffusion(xt, {"ctx_len": CTX}, t=t_vec, den_cache=mc)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = model._run_denoiser_step_diffusion(xt, {"ctx_len": CTX}, t=t_vec, den_cache=mc)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"{label}: {ms:.1f} ms/NFE", flush=True)
    return ms, out


t_warm0 = time.perf_counter()
model._run_denoiser_step_diffusion(xt, {"ctx_len": CTX}, t=t_vec, den_cache=mc)
torch.cuda.synchronize()
print(f"first NFE (incl. compile warmup): {time.perf_counter()-t_warm0:.1f}s",
      flush=True)

ms_fast, logits_fast = bench("OPTIMIZED (fast-MoE%s)" %
                             (" + compiled kernels" if COMPILE else ""))

fast_fn = mh.NemotronHMOE.moe
mh.NemotronHMOE.moe = mh.NemotronHMOE._orig_moe
ms_orig, logits_orig = bench("ORIGINAL MoE", n=3, warmup=1)
mh.NemotronHMOE.moe = fast_fn

diff = (logits_fast - logits_orig).abs().max().item()
rel = diff / logits_orig.abs().max().item()
print(f"correctness: max|diff|={diff:.4e} (rel {rel:.2e}) "
      f"{'OK' if rel < 1e-2 else 'SUSPECT'}", flush=True)
print("RESULT " + json.dumps({"opt_ms": ms_fast, "orig_ms": ms_orig,
                              "speedup": ms_orig / ms_fast,
                              "maxdiff": diff, "compile": COMPILE}))
