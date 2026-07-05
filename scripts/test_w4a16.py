#!/usr/bin/env python3
"""Unit test + microbench for the grouped W4A16 MoE path (no model needed)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
from quantize_experts import quantize_w4
from tt_nvfp4 import grouped_moe, dequant_ref

torch.manual_seed(0)
dev = "cuda"
E, HID, INT, K_TOP = 128, 2688, 1856, 6

print("building synthetic expert bank...", flush=True)
W_up = [torch.randn(INT, HID, dtype=torch.bfloat16, device=dev) * 0.03 for _ in range(E)]
W_dn = [torch.randn(HID, INT, dtype=torch.bfloat16, device=dev) * 0.03 for _ in range(E)]
uq, us, dq, ds = [], [], [], []
for e in range(E):
    q, s = quantize_w4(W_up[e]); uq.append(q); us.append(s)
    q, s = quantize_w4(W_dn[e]); dq.append(q); ds.append(s)
up_q = torch.stack(uq).to(dev); up_s = (torch.stack(us) / 4.0).to(dev)
down_q = torch.stack(dq).to(dev); down_s = (torch.stack(ds) / 4.0).to(dev)
# dequantized reference banks
W_up_dq = [dequant_ref(uq[e], us[e], HID).to(dev) for e in range(E)]
W_dn_dq = [dequant_ref(dq[e], ds[e], INT).to(dev) for e in range(E)]


def ref_moe(x2, idx, w):
    out = torch.zeros(x2.shape[0], HID, dtype=torch.float32, device=dev)
    for t in range(x2.shape[0]):
        for j in range(idx.shape[1]):
            e = int(idx[t, j])
            h = F.relu(x2[t:t+1].float() @ W_up_dq[e].float().T)
            h = h * h
            out[t] += (h @ W_dn_dq[e].float().T).squeeze(0) * float(w[t, j])
    return out.to(x2.dtype)


for T in (16, 3, 200):
    x = torch.randn(T, HID, dtype=torch.bfloat16, device=dev)
    idx = torch.stack([torch.randperm(E, device=dev)[:K_TOP] for _ in range(T)])
    w = torch.rand(T, K_TOP, device=dev)
    w = w / w.sum(-1, keepdim=True)
    y = grouped_moe(x, idx, w, up_q, up_s, down_q, down_s, HID, INT)
    y_ref = ref_moe(x, idx, w)
    err = (y.float() - y_ref.float()).abs().max().item()
    rng = y_ref.float().abs().max().item()
    ok = err / max(rng, 1e-6) < 3e-2
    print(f"T={T:3d}: max|err|={err:.4f} rel={err/rng:.2e} {'OK' if ok else 'FAIL'}")
    assert ok

# Microbench: T=16, fresh random routing each iter (defeats L2 reuse of the
# same experts; the full 128-expert bank is 400+MB so reads are mostly cold).
x = torch.randn(16, HID, dtype=torch.bfloat16, device=dev)
idxs = [torch.stack([torch.randperm(E, device=dev)[:K_TOP] for _ in range(16)])
        for _ in range(50)]
w = torch.full((16, K_TOP), 1.0 / K_TOP, device=dev)
for i in range(10):
    grouped_moe(x, idxs[i % 50], w, up_q, up_s, down_q, down_s, HID, INT)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(50):
    grouped_moe(x, idxs[i], w, up_q, up_s, down_q, down_s, HID, INT)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 50 * 1e3
print(f"grouped quant MoE layer (T=16): {ms:.2f} ms")

# bf16 reference: the fast_moe-style loop over unique experts
def bf16_moe(x2, idx, wts):
    T = x2.shape[0]
    flat_e = idx.reshape(-1)
    flat_t = torch.arange(T, device=dev).repeat_interleave(K_TOP)
    flat_w = wts.reshape(-1)
    order = flat_e.argsort()
    se, st, sw = flat_e[order], flat_t[order], flat_w[order]
    uniq, counts = torch.unique_consecutive(se, return_counts=True)
    out = torch.zeros(T, HID, dtype=torch.float32, device=dev)
    pos = 0
    for e, c in zip(uniq.tolist(), counts.tolist()):
        rows = st[pos:pos+c]
        h = F.relu(x2[rows] @ W_up[e].T); h = h * h
        out.index_add_(0, rows, (h @ W_dn[e].T).float() * sw[pos:pos+c, None].float())
        pos += c
    return out

for i in range(5):
    bf16_moe(x, idxs[i % 50], w)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(50):
    bf16_moe(x, idxs[i], w)
torch.cuda.synchronize()
ms_ref = (time.perf_counter() - t0) / 50 * 1e3
print(f"bf16 fast-moe loop      (T=16): {ms_ref:.2f} ms  -> speedup {ms_ref/ms:.2f}x")
print("ALL-OK")
