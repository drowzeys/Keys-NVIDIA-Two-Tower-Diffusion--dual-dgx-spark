# NVFP4-style W4A16 expert path for the TwoTower MoE layers.
#
# Design: per MoE layer, ALL active experts run in ONE triton launch per
# projection (grouped GEMM over token segments sorted by expert), instead of
# ~2 launches x ~60-90 experts. Weights are fp4 e2m1 packed 2/byte along K
# with fp16 block-16 scales, dequantized in-register via integer shifts —
# only 0.625 bytes/element cross the memory bus (vs 2 for bf16) on a
# 273 GB/s GB10.
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Grouped W4A16 GEMM: out[seg] = x[seg] @ dequant(qw[eid]).T  per segment
# ---------------------------------------------------------------------------

@triton.jit
def _w4a16_grouped_kernel(x_ptr, out_ptr, seg_ptr, eid_ptr,
                          qw_ptr, sc_ptr,
                          N, K,
                          stride_xm, stride_xk,
                          stride_om, stride_on,
                          stride_qe, stride_qn, stride_qk,
                          stride_se, stride_sn, stride_sk,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    seg_i = tl.program_id(0)
    pid_n = tl.program_id(1)
    r0 = tl.load(seg_ptr + seg_i)
    r1 = tl.load(seg_ptr + seg_i + 1)
    eid = tl.load(eid_ptr + seg_i)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    qw_base = qw_ptr + eid.to(tl.int64) * stride_qe
    sc_base = sc_ptr + eid.to(tl.int64) * stride_se

    for m0 in range(r0, r1, BLOCK_M):
        rm = m0 + tl.arange(0, BLOCK_M)
        m_mask = rm < r1
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
                        mask=m_mask[:, None] & (rk[None, :] < K), other=0.0)
            rkb = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
            qw = tl.load(qw_base + rn[:, None] * stride_qn + rkb[None, :] * stride_qk,
                         mask=(rn[:, None] < N) & (rkb[None, :] < K // 2),
                         other=0)
            lo = (qw & 0x0F).to(tl.int32)
            hi = (qw >> 4).to(tl.int32)
            code = tl.interleave(lo, hi)               # (BLOCK_N, BLOCK_K)
            man = code & 1
            exp = (code >> 1) & 3
            # e2m1 magnitude x4 via integer shifts: exp=0 -> 2*man,
            # else (2+man) << exp  ->  {0,2,4,6,8,12,16,24}
            m4 = tl.where(exp == 0, 2 * man, (2 + man) << exp)
            val = tl.where((code >> 3) & 1 == 1, -m4, m4).to(tl.float32)
            # scales are pre-divided by 4 at attach time
            rks = (k0 // 16) + tl.arange(0, BLOCK_K // 16)
            sc = tl.load(sc_base + rn[:, None] * stride_sn + rks[None, :] * stride_sk,
                         mask=(rn[:, None] < N) & (rks[None, :] < K // 16),
                         other=0.0).to(tl.float32)
            w = tl.reshape(val, (BLOCK_N, BLOCK_K // 16, 16)) * sc[:, :, None]
            w = tl.reshape(w, (BLOCK_N, BLOCK_K))
            acc += tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)),
                          out_dtype=tl.float32)
        out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
        tl.store(out_ptrs, acc.to(tl.bfloat16),
                 mask=m_mask[:, None] & (rn[None, :] < N))


def w4a16_grouped(x, seg, eid, qw, sc, N, n_seg):
    """x (P, K) rows sorted by expert segment; seg (n_seg+1) int32 row
    offsets; eid (n_seg) expert ids; qw (E, N, K/2) uint8; sc (E, N, K/16)
    fp16 (pre-divided by 4). Returns (P, N) bf16."""
    P, K = x.shape
    out = torch.empty((P, N), dtype=torch.bfloat16, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 128, 128
    grid = (n_seg, triton.cdiv(N, BLOCK_N))
    _w4a16_grouped_kernel[grid](
        x, out, seg, eid, qw, sc, N, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        qw.stride(0), qw.stride(1), qw.stride(2),
        sc.stride(0), sc.stride(1), sc.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3)
    return out


def grouped_moe(x2, topk_indices, topk_weights, up_q, up_s, down_q, down_s,
                hidden, inter):
    """Full routed-expert MoE over quantized stacked weights.
    x2 (T, hidden); returns (T, hidden) in topk_weights dtype semantics of
    the original (accumulate fp32, cast at the end)."""
    T = x2.shape[0]
    k = topk_indices.shape[1]
    flat_e = topk_indices.reshape(-1)
    flat_t = torch.arange(T, device=x2.device).repeat_interleave(k)
    flat_w = topk_weights.reshape(-1)
    order = flat_e.argsort()
    se, st, sw = flat_e[order], flat_t[order], flat_w[order]
    uniq, counts = torch.unique_consecutive(se, return_counts=True)
    n_seg = int(uniq.shape[0])                      # single host sync
    seg = torch.zeros(n_seg + 1, dtype=torch.int32, device=x2.device)
    seg[1:] = torch.cumsum(counts, 0).to(torch.int32)
    eid = uniq.to(torch.int32)

    xg = x2[st]                                      # (P, hidden) gather
    h = w4a16_grouped(xg, seg, eid, up_q, up_s, inter, n_seg)
    h = torch.nn.functional.relu(h)
    h = h * h                                        # relu2
    y = w4a16_grouped(h, seg, eid, down_q, down_s, hidden, n_seg)
    out = torch.zeros(T, hidden, dtype=torch.float32, device=x2.device)
    out.index_add_(0, st, y.float() * sw.float().unsqueeze(-1))
    return out.to(x2.dtype)


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------

def attach_quant_experts(model, tower_names, qsd, cfg, device):
    """Per MoE layer: stack the 128 experts' quant tensors into contiguous
    (E, N, K/2)+(E, N, K/16) buffers on `mixer`, drop the meta expert Linears,
    and swap mixer.forward for the grouped quantized path."""
    import types
    hidden = cfg.hidden_size
    inter = cfg.moe_intermediate_size
    n_layers = 0

    def quant_forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        x2 = hidden_states.view(-1, orig_shape[-1])
        y = grouped_moe(x2, topk_indices, topk_weights,
                        self.up_q, self.up_s, self.down_q, self.down_s,
                        hidden, inter)
        y = y.view(*orig_shape)
        return y + self.shared_experts(residuals)

    for tower_name in tower_names:
        tower = getattr(model, tower_name)
        for li, blk in enumerate(tower.layers):
            if getattr(blk, "block_type", None) != "moe":
                continue
            mixer = blk.mixer
            E = len(mixer.experts)
            base = f"{tower_name}.layers.{li}.mixer.experts"
            up_q = torch.stack([qsd[f"{base}.{e}.up_proj.qweight"] for e in range(E)])
            up_s = torch.stack([qsd[f"{base}.{e}.up_proj.scales"] for e in range(E)]) / 4.0
            down_q = torch.stack([qsd[f"{base}.{e}.down_proj.qweight"] for e in range(E)])
            down_s = torch.stack([qsd[f"{base}.{e}.down_proj.scales"] for e in range(E)]) / 4.0
            mixer.up_q = up_q.to(device).contiguous()
            mixer.up_s = up_s.to(torch.float16).to(device).contiguous()
            mixer.down_q = down_q.to(device).contiguous()
            mixer.down_s = down_s.to(torch.float16).to(device).contiguous()
            for e in range(E):
                del qsd[f"{base}.{e}.up_proj.qweight"]
                del qsd[f"{base}.{e}.up_proj.scales"]
                del qsd[f"{base}.{e}.down_proj.qweight"]
                del qsd[f"{base}.{e}.down_proj.scales"]
            mixer.experts = torch.nn.ModuleList()   # drop meta Linears
            mixer.forward = types.MethodType(quant_forward, mixer)
            n_layers += 1
    print(f"[opt] attached NVFP4 W4A16 grouped experts on {n_layers} MoE "
          f"layers", flush=True)
    return n_layers


# ---------------------------------------------------------------------------
# Reference helpers (tests)
# ---------------------------------------------------------------------------

def dequant_ref(qweight, scales, K):
    """Reference dequant: (N, K) bf16. `scales` = checkpoint scales (amax/6)."""
    N = qweight.shape[0]
    lo = (qweight & 0x0F).to(torch.int32)
    hi = (qweight >> 4).to(torch.int32)
    code = torch.stack([lo, hi], dim=-1).reshape(N, K)
    man = (code & 1).float()
    exp = ((code >> 1) & 3).float()
    mag = torch.where(exp == 0, 0.5 * man,
                      torch.exp2(exp - 1.0) * (1.0 + 0.5 * man))
    sgn = torch.where((code >> 3) & 1 == 1, -1.0, 1.0)
    w = (sgn * mag).reshape(N, K // 16, 16)
    w = w * scales.float().unsqueeze(-1)
    return w.reshape(N, K).to(torch.bfloat16)
