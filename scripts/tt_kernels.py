# Pure-torch replacements for the two mamba_ssm kernels used by the two-tower
# block paths (_denoiser_block_mamba). Blocks are <=16 tokens, so a sequential
# fp32 scan is exact and cheap; no sm_121a compiled kernels on the path.
import torch
import torch.nn.functional as F


def conv1d_block(xBC, weight, bias, init_conv, activation="silu"):
    """Depthwise causal conv over a short block, seeded from context state.

    xBC:       (B, L, C) raw conv inputs (block)
    weight:    (C, K)    conv1d.weight.squeeze(1)
    bias:      (C,) or None
    init_conv: (B, C, K-1) trailing context inputs, or None (zeros)
    Returns:   (B, L, C) activated conv outputs (same as causal_conv1d_fn)
    """
    B, L, C = xBC.shape
    K = weight.shape[-1]
    xt = xBC.transpose(1, 2)  # (B, C, L)
    if init_conv is None:
        init_conv = xt.new_zeros(B, C, K - 1)
    inp = torch.cat([init_conv.to(xt.dtype), xt], dim=-1)  # (B, C, L+K-1)
    out = F.conv1d(inp, weight.unsqueeze(1).to(xt.dtype),
                   bias.to(xt.dtype) if bias is not None else None,
                   groups=C)  # valid conv -> (B, C, L)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"unsupported activation {activation}")
    return out.transpose(1, 2)


def ssd_scan_block(x, dt, A, B_proj, C_proj, D=None, dt_bias=None,
                   dt_softplus=True, initial_states=None):
    """Sequential fp32 Mamba2/SSD recurrence over a short block.

    x:              (B, L, H, P)
    dt:             (B, L, H)
    A:              (H,)  already -exp(A_log)
    B_proj, C_proj: (B, L, G, N)
    D:              (H,) or None
    dt_bias:        (H,) or None
    initial_states: (B, H, P, N) or None
    Returns: y (B, L, H, P) fp32, final_state (B, H, P, N) fp32

    Recurrence (matches mamba_chunk_scan_combined):
        dt' = softplus(dt + dt_bias)
        h_t = exp(dt'_t * A) * h_{t-1} + dt'_t * x_t B_t^T
        y_t = C_t h_t + D * x_t
    """
    Bsz, L, H, P = x.shape
    G, N = B_proj.shape[2], B_proj.shape[3]
    xf = x.float()
    dtf = dt.float()
    if dt_bias is not None:
        dtf = dtf + dt_bias.float()[None, None, :]
    if dt_softplus:
        dtf = F.softplus(dtf)
    hpg = H // G  # heads per group (head i -> group i // hpg)
    Bh = B_proj.float().repeat_interleave(hpg, dim=2)  # (B, L, H, N)
    Ch = C_proj.float().repeat_interleave(hpg, dim=2)
    state = (initial_states.float().clone() if initial_states is not None
             else xf.new_zeros(Bsz, H, P, N))
    ys = []
    Af = A.float()
    for t in range(L):
        decay = torch.exp(dtf[:, t] * Af[None, :])                 # (B, H)
        state = (state * decay[:, :, None, None]
                 + dtf[:, t][:, :, None, None]
                 * xf[:, t][:, :, :, None] * Bh[:, t][:, :, None, :])
        ys.append(torch.einsum("bhpn,bhn->bhp", state, Ch[:, t]))
    y = torch.stack(ys, dim=1)                                     # (B, L, H, P)
    if D is not None:
        y = y + D.float()[None, None, :, None] * xf
    return y, state
