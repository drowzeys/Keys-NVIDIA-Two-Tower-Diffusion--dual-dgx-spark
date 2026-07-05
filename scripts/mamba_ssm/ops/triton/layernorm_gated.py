# Pure-torch shim for mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn.
# modeling_nemotron_h.py hard-imports this one function (MambaRMSNormGated);
# everything else in mamba_ssm is optional there (falls back to torch_forward).
# Semantics match mamba_ssm: norm_before_gate=False -> out = norm(x * silu(z)) * w,
# with grouped RMS statistics over contiguous channel groups of size group_size.
import torch
import torch.nn.functional as F


def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None,
               norm_before_gate=True):
    dtype = x.dtype
    xf = x.float()
    if z is not None and not norm_before_gate:
        xf = xf * F.silu(z.float())
    d = xf.shape[-1]
    if group_size is None or group_size == d:
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + eps)
    else:
        xg = xf.view(*xf.shape[:-1], d // group_size, group_size)
        var = xg.pow(2).mean(-1, keepdim=True)
        xf = (xg * torch.rsqrt(var + eps)).view(*xf.shape)
    out = xf * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None and norm_before_gate:
        out = out * F.silu(z.float())
    return out.to(dtype)
