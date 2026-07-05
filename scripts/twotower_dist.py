#!/usr/bin/env python3
"""Cross-node two-tower mask-diffusion inference for Nemotron-Labs-TwoTower-30B.

NVIDIA's reference runs the two towers on two GPUs in one process
(place_towers_on_devices('cuda:0','cuda:1')). Here each tower runs on its own
DGX Spark (GB10), talking over the 200G fabric via torch.distributed (gloo):

  rank 0 = denoiser tower  (node .4, ~/aeon27b/models/tt-denoiser)
  rank 1 = context tower   (node .3, ~/aeon27b/models/tt-context)

v2: multi-request serve loop (one weight-load amortizes a whole bench+eval
suite), fast-MoE patch (single sync instead of 128 per layer), optional
torch.compile of the pure-torch block kernels (COMPILE=1), per-layer streamed
context-extension deltas.

Protocol per request:
  ctx -> den : [ctx_len, num_blocks] header  ([0,0] = shutdown)
  ctx -> den : initial denoiser cache (mamba conv/ssm + full attention KV)
  per block:
    den -> ctx : committed block tokens
    ctx -> den : flag (1 = continue)
    ctx -> den : per-layer delta, streamed as each layer's extension finishes
  den -> ctx : total NFE for the request
"""
import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK_DIR))  # fake mamba_ssm shim + tt_kernels

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import tt_kernels  # noqa: E402

DEN_RANK, CTX_RANK = 0, 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["ctx", "den"], required=True)
    p.add_argument("--model", required=True, help="tower checkpoint dir")
    p.add_argument("--master", default="10.100.10.4")
    p.add_argument("--port", type=int, default=29613)
    p.add_argument("--prompt", default="France is a country ")
    p.add_argument("--prompt-file", default=None,
                   help='jsonl: {"text":..., "max_new_tokens":..., "tag":...}')
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--steps-per-block", type=int, default=16)
    p.add_argument("--mask-token-id", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--confidence-threshold", type=float, default=0.9)
    p.add_argument("--mode", choices=["mask_diffusion", "ar"],
                   default="mask_diffusion",
                   help="ar = single-node context-tower AR baseline (no dist)")
    p.add_argument("--fast-moe", type=int, default=1)
    p.add_argument("--compile", type=int,
                   default=int(os.environ.get("COMPILE", "0")))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Optimizations
# ---------------------------------------------------------------------------

def patch_fast_moe(mh):
    """Replace NemotronHMOE.moe: iterate only over ACTIVE experts with ONE
    host sync, instead of 128 iterations x (one_hot + torch.where + .numel()
    sync) per layer. Inference-only (drops the training dummy-forward path)."""

    def fast_moe(self, hidden_states, topk_indices, topk_weights):
        T = hidden_states.shape[0]
        k = topk_indices.shape[1]
        flat_e = topk_indices.reshape(-1)
        flat_t = torch.arange(T, device=flat_e.device).repeat_interleave(k)
        flat_w = topk_weights.reshape(-1)
        order = flat_e.argsort()
        se, st, sw = flat_e[order], flat_t[order], flat_w[order]
        uniq, counts = torch.unique_consecutive(se, return_counts=True)
        uniq_l = uniq.tolist()          # the single host sync
        counts_l = counts.tolist()
        out = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        pos = 0
        for e, c in zip(uniq_l, counts_l):
            idx = st[pos:pos + c]
            y = self.experts[e](hidden_states[idx]) * sw[pos:pos + c].unsqueeze(-1)
            out.index_add_(0, idx, y)
            pos += c
        return out.type(hidden_states.dtype)

    mh.NemotronHMOE._orig_moe = mh.NemotronHMOE.moe
    mh.NemotronHMOE.moe = fast_moe
    print("[opt] fast MoE patched (single-sync active-expert loop)", flush=True)


def maybe_compile_kernels(enable):
    """torch.compile the two pure-torch block kernels. Shapes are static
    (L=16 blocks), so the 16-step SSD python loop unrolls into fused kernels
    and repeated launches collapse (this is where the CUDA-graph-style win
    lives for the mamba path)."""
    if not enable:
        return
    try:
        tt_kernels.ssd_scan_block = torch.compile(
            tt_kernels.ssd_scan_block, dynamic=False)
        tt_kernels.conv1d_block = torch.compile(
            tt_kernels.conv1d_block, dynamic=False)
        print("[opt] tt_kernels compiled (inductor, static shapes)", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[opt] torch.compile unavailable, staying eager: {e}", flush=True)


# ---------------------------------------------------------------------------
# Wire helpers (gloo, fixed deterministic tensor order, uint8 on the wire)
# ---------------------------------------------------------------------------

def send_t(t, dst):
    t = t.detach().to("cpu").contiguous().reshape(-1)
    dist.send(t.view(torch.uint8), dst=dst)


def recv_t(shape, dtype, src, device):
    numel = 1
    for s in shape:
        numel *= s
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    buf = torch.empty(nbytes, dtype=torch.uint8)
    dist.recv(buf, src=src)
    return buf.view(dtype).reshape(shape).to(device)


def send_longs(vals, dst):
    send_t(torch.tensor(vals, dtype=torch.int64), dst)


def recv_longs(n, src):
    return recv_t((n,), torch.int64, src, "cpu").tolist()


# ---------------------------------------------------------------------------
# Model subclass: pure-torch block mamba kernel + streaming context extension
# ---------------------------------------------------------------------------

def make_model_class(ttm):
    class DistTwoTower(ttm.NemotronHTwoTowerForCausalLM):
        def _denoiser_block_mamba(self, mixer, hidden, init_conv, init_ssm,
                                  return_states=False):
            d_inner = mixer.intermediate_size
            ngroups = mixer.n_groups
            d_state = mixer.ssm_state_size
            headdim = mixer.head_dim
            conv_dim = mixer.conv_dim
            d_conv = mixer.conv_kernel_size

            proj = mixer.in_proj(hidden)
            z, xBC, dt = torch.split(
                proj, [d_inner, conv_dim, mixer.num_heads], dim=-1)

            xBC_conv = tt_kernels.conv1d_block(
                xBC, mixer.conv1d.weight.squeeze(1), mixer.conv1d.bias,
                init_conv, activation=mixer.activation)

            x, B_proj, C_proj = torch.split(
                xBC_conv, [d_inner, ngroups * d_state, ngroups * d_state], dim=-1)
            Bsz, L = x.shape[:2]
            x = x.view(Bsz, L, mixer.num_heads, headdim)
            B_proj = B_proj.view(Bsz, L, ngroups, d_state)
            C_proj = C_proj.view(Bsz, L, ngroups, d_state)

            A = -torch.exp(mixer.A_log.float())
            y, new_ssm = tt_kernels.ssd_scan_block(
                x, dt, A, B_proj, C_proj, D=mixer.D, dt_bias=mixer.dt_bias,
                dt_softplus=True, initial_states=init_ssm)
            y = y.reshape(Bsz, L, d_inner).to(z.dtype)
            y = mixer.norm(y, z)
            out = mixer.out_proj(y)
            if not return_states:
                return out
            # New conv state: last d_conv raw xBC inputs, most-recent at -1.
            if L >= d_conv:
                new_conv = xBC[:, -d_conv:, :].transpose(1, 2).contiguous()
            else:
                hist = (init_conv if init_conv is not None
                        else xBC.new_zeros(Bsz, conv_dim, d_conv - 1))
                comb = torch.cat([hist.transpose(1, 2), xBC], dim=1)
                new_conv = comb[:, -d_conv:, :].transpose(1, 2).contiguous()
            return out, new_conv, new_ssm

        def extend_context_streaming(self, new_tokens, cache_state, dst,
                                     block: int):
            """Block-wise context extension that STREAMS each layer's cache
            delta to `dst` as soon as that layer finishes — transfer overlaps
            the remaining layers' compute. Mirrors _extend_context_cache
            (block_wise=True) exactly."""
            ctx_cache = cache_state["ctx_cache"]
            pattern = self.config.hybrid_override_pattern
            ctx_len = cache_state["ctx_len"]
            tower = self.context_tower
            ctx_device = next(tower.parameters()).device
            L = new_tokens.shape[1]
            tokens = new_tokens.to(ctx_device)

            new_s2 = {}
            for i in range(self.config.num_hidden_layers):
                if pattern[i] == "M":
                    new_s2[i] = (ctx_cache.conv_states[i].clone(),
                                 ctx_cache.ssm_states[i].clone())
            cache_state["mamba_s2"] = new_s2
            ctx_cache.has_previous_state = True

            cache_position = torch.arange(ctx_len, ctx_len + L, device=ctx_device)
            hidden = tower.embeddings(tokens)
            causal_mask = tower._update_causal_mask(None, hidden, cache_position)

            for layer_idx, blk in enumerate(tower.layers):
                residual = hidden
                h = blk.norm(hidden.to(dtype=blk.norm.weight.dtype))
                if blk.residual_in_fp32:
                    residual = residual.to(torch.float32)

                if blk.block_type == "mamba":
                    d_conv = blk.mixer.conv_kernel_size
                    init_conv = ctx_cache.conv_states[layer_idx][..., -(d_conv - 1):]
                    init_ssm = ctx_cache.ssm_states[layer_idx].contiguous()
                    h, new_conv, new_ssm = self._denoiser_block_mamba(
                        blk.mixer, h, init_conv, init_ssm, return_states=True)
                    ctx_cache.conv_states[layer_idx] = new_conv
                    ctx_cache.ssm_states[layer_idx] = new_ssm
                    send_t(new_conv.to(torch.bfloat16), dst)
                    send_t(new_ssm.float(), dst)
                elif blk.block_type == "attention":
                    h, _, _ = blk.mixer(
                        h, attention_mask=causal_mask,
                        past_key_value=ctx_cache, cache_position=cache_position)
                    send_t(ctx_cache.key_cache[layer_idx][:, :, -block:, :]
                           .to(torch.bfloat16), dst)
                    send_t(ctx_cache.value_cache[layer_idx][:, :, -block:, :]
                           .to(torch.bfloat16), dst)
                elif blk.block_type in ["mlp", "moe"]:
                    h = blk.mixer(h)
                else:
                    raise ValueError(f"Unknown block_type: {blk.block_type}")
                hidden = residual + h

            cache_state["ctx_len"] = ctx_len + L
            return cache_state

    return DistTwoTower


# ---------------------------------------------------------------------------
# Loading: meta init, materialize only this rank's tower
# ---------------------------------------------------------------------------

def load_model(args, device, fast_moe=True, compile_kernels=False):
    sys.path.insert(1, args.model)
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from configuration_nemotron_h import NemotronHConfig
    import modeling_nemotron_h as mh
    import modeling_nemotron_twotower as ttm

    if fast_moe:
        patch_fast_moe(mh)
    maybe_compile_kernels(compile_kernels)

    cfg = NemotronHConfig.from_pretrained(args.model)
    cls = make_model_class(ttm)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with init_empty_weights():
        model = cls(cfg)
    torch.set_default_dtype(prev)

    prefixes = (("context_tower.", "context_lm_head.") if args.role == "ctx"
                else ("denoiser_tower.", "lm_head.", "t_embedder.",
                      "t_block.", "scale_shift_tables."))
    idx = json.load(open(Path(args.model) / "model.safetensors.index.json"))
    weight_map = idx["weight_map"]
    sd = {}
    t0 = time.perf_counter()
    for shard in sorted(set(weight_map.values())):
        with safe_open(Path(args.model) / shard, framework="pt",
                       device=device) as f:
            for k in f.keys():
                if k.startswith(prefixes):
                    sd[k] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    bad = [k for k in missing if k.startswith(prefixes)]
    assert not bad, f"missing keys for this tower: {bad[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    tower = model.context_tower if args.role == "ctx" else model.denoiser_tower
    metas = [n for n, p in tower.named_parameters() if p.device.type == "meta"]
    assert not metas, f"still-meta params: {metas[:5]}"
    model.eval()
    print(f"[{args.role}] loaded {len(sd)} tensors in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    return model, cfg, ttm


# ---------------------------------------------------------------------------
# Cache wire schema
# ---------------------------------------------------------------------------

def layer_kinds(cfg):
    pat = cfg.hybrid_override_pattern
    assert len(pat) == cfg.num_hidden_layers
    return pat  # 'M' mamba, '*' attention, 'E' moe, '-' mlp


def conv_shape(cfg, B):
    conv_dim = (cfg.mamba_num_heads * cfg.mamba_head_dim
                + 2 * cfg.n_groups * cfg.ssm_state_size)
    return (B, conv_dim, cfg.conv_kernel)


def ssm_shape(cfg, B):
    return (B, cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state_size)


def kv_shape(cfg, B, seqlen):
    return (B, cfg.num_key_value_heads, seqlen, cfg.head_dim)


def ctx_send_full_cache(cache, cfg, B, dst):
    for i, kind in enumerate(layer_kinds(cfg)):
        if kind == "M":
            send_t(cache.conv_states[i].to(torch.bfloat16), dst)
            send_t(cache.ssm_states[i].float(), dst)
        elif kind == "*":
            send_t(cache.key_cache[i].to(torch.bfloat16), dst)
            send_t(cache.value_cache[i].to(torch.bfloat16), dst)


class MirrorCache:
    """Denoiser-side read-only mirror of the context cache."""

    def __init__(self, n_layers):
        self.conv_states = [None] * n_layers
        self.ssm_states = [None] * n_layers
        self.key_cache = [None] * n_layers
        self.value_cache = [None] * n_layers
        self.has_previous_state = True


def den_recv_full_cache(cfg, B, ctx_len, src, device):
    mc = MirrorCache(cfg.num_hidden_layers)
    for i, kind in enumerate(layer_kinds(cfg)):
        if kind == "M":
            mc.conv_states[i] = recv_t(conv_shape(cfg, B), torch.bfloat16, src, device)
            mc.ssm_states[i] = recv_t(ssm_shape(cfg, B), torch.float32, src, device)
        elif kind == "*":
            mc.key_cache[i] = recv_t(kv_shape(cfg, B, ctx_len), torch.bfloat16, src, device)
            mc.value_cache[i] = recv_t(kv_shape(cfg, B, ctx_len), torch.bfloat16, src, device)
    return mc


def den_recv_delta(mc, cfg, B, block, src, device):
    for i, kind in enumerate(layer_kinds(cfg)):
        if kind == "M":
            mc.conv_states[i] = recv_t(conv_shape(cfg, B), torch.bfloat16, src, device)
            mc.ssm_states[i] = recv_t(ssm_shape(cfg, B), torch.float32, src, device)
        elif kind == "*":
            k = recv_t(kv_shape(cfg, B, block), torch.bfloat16, src, device)
            v = recv_t(kv_shape(cfg, B, block), torch.bfloat16, src, device)
            mc.key_cache[i] = torch.cat([mc.key_cache[i], k], dim=2)
            mc.value_cache[i] = torch.cat([mc.value_cache[i], v], dim=2)


# ---------------------------------------------------------------------------
# Denoiser rank
# ---------------------------------------------------------------------------

def denoise_block(model, args, mc, ctx_len, device, B=1):
    """One block of confidence-unmasking (mirrors generate_mask_diffusion)."""
    mask_id = args.mask_token_id
    xt = torch.full((B, args.block_size), mask_id, dtype=torch.long, device=device)
    nfe = 0
    for step_idx in range(args.steps_per_block):
        is_masked = (xt == mask_id)
        if is_masked.sum().item() == 0:
            break
        t_model = is_masked.float().mean()
        t_vec = t_model.expand(B).to(device)

        logits = model._run_denoiser_step_diffusion(
            xt, {"ctx_len": ctx_len}, t=t_vec, den_cache=mc)
        nfe += 1

        log_x_theta = model._mdlm_forward(logits, xt, mask_id)
        x_theta = log_x_theta.exp()
        if args.temperature <= 0:
            predicted = log_x_theta.argmax(dim=-1)
        else:
            scaled = logits.clone()
            scaled[..., mask_id] = -1e12
            scaled = scaled / args.temperature
            scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
            unmasked = (xt != mask_id)
            if unmasked.any():
                scaled[unmasked] = -1e12
                scaled[unmasked, :].scatter_(-1, xt[unmasked].unsqueeze(-1), 0.0)
            predicted = model._gumbel_sample(scaled)

        confidence = x_theta.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
        confidence[~is_masked] = float("inf")
        n_masked_int = is_masked.sum(-1)
        if step_idx == args.steps_per_block - 1:
            tokens_to_commit = n_masked_int
        else:
            remaining = max(1, args.steps_per_block - step_idx)
            num_above = ((confidence > args.confidence_threshold) & is_masked).sum(-1)
            tokens_to_commit = torch.where(
                num_above > 0, num_above, torch.ones_like(num_above))
            min_commit = (n_masked_int.float() / remaining).ceil().long()
            tokens_to_commit = torch.clamp(
                torch.max(tokens_to_commit, min_commit), max=n_masked_int)

        output = torch.where(is_masked, predicted, xt)
        num_to_remask = n_masked_int - tokens_to_commit
        for b in range(B):
            if num_to_remask[b] > 0:
                mi = is_masked[b].nonzero(as_tuple=True)[0]
                _, order = confidence[b, mi].sort()
                output[b, mi[order[:num_to_remask[b]]]] = mask_id
        xt = output
    return xt, nfe


def run_denoiser(model, cfg, args, device):
    B = 1
    req = 0
    while True:
        ctx_len, num_blocks = recv_longs(2, CTX_RANK)
        if num_blocks == 0:
            print("[den] shutdown", flush=True)
            break
        req += 1
        mc = den_recv_full_cache(cfg, B, ctx_len, CTX_RANK, device)
        total_nfe = 0
        block_idx = 0
        while True:
            t0 = time.perf_counter()
            xt, nfe = denoise_block(model, args, mc, ctx_len, device, B)
            total_nfe += nfe
            if args.verbose:
                print(f"[den] req {req} block {block_idx}: {nfe} NFE in "
                      f"{time.perf_counter() - t0:.2f}s", flush=True)
            send_t(xt.to(torch.int64), CTX_RANK)
            (flag,) = recv_longs(1, CTX_RANK)
            if flag == 0:
                break
            den_recv_delta(mc, cfg, B, args.block_size, CTX_RANK, device)
            ctx_len += args.block_size
            block_idx += 1
        send_longs([total_nfe], CTX_RANK)
        print(f"[den] req {req} done: {total_nfe} NFE", flush=True)


# ---------------------------------------------------------------------------
# Context rank
# ---------------------------------------------------------------------------

def load_requests(args):
    if args.prompt_file:
        reqs = []
        for line in open(args.prompt_file):
            line = line.strip()
            if line:
                d = json.loads(line)
                reqs.append({
                    "text": d["text"],
                    "max_new_tokens": int(d.get("max_new_tokens",
                                                args.max_new_tokens)),
                    "tag": d.get("tag", f"req{len(reqs)}"),
                })
        return reqs
    return [{"text": args.prompt, "max_new_tokens": args.max_new_tokens,
             "tag": "prompt"}]


def run_context(model, cfg, args, device):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    B = 1
    requests = load_requests(args)
    eos = tok.eos_token_id
    results = []

    for ridx, req in enumerate(requests):
        assert req["max_new_tokens"] % args.block_size == 0
        num_blocks = req["max_new_tokens"] // args.block_size
        ids = tok(req["text"], return_tensors="pt").input_ids.to(device)
        S = ids.shape[1]
        print(f"[ctx] req {ridx+1}/{len(requests)} ({req['tag']}): "
              f"{S} prompt tokens, {req['max_new_tokens']} new", flush=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            cache_state = model._build_context_cache(ids)
        prefill_s = time.perf_counter() - t0

        send_longs([S, num_blocks], DEN_RANK)
        ctx_send_full_cache(cache_state["ctx_cache"], cfg, B, DEN_RANK)

        context_ids = ids.clone()
        t_gen = time.perf_counter()
        for block_idx in range(num_blocks):
            xt = recv_t((B, args.block_size), torch.int64, DEN_RANK, device)
            context_ids = torch.cat([context_ids, xt], dim=1)
            if args.verbose:
                print(f"[ctx] block {block_idx}: "
                      f"{tok.decode(xt[0], skip_special_tokens=False)!r}",
                      flush=True)
            stop = (block_idx == num_blocks - 1) or \
                   (eos is not None and (xt == eos).any().item())
            if stop:
                send_longs([0], DEN_RANK)
                break
            send_longs([1], DEN_RANK)
            with torch.no_grad():
                cache_state = model.extend_context_streaming(
                    xt, cache_state, DEN_RANK, args.block_size)
        gen_s = time.perf_counter() - t_gen
        (nfe,) = recv_longs(1, DEN_RANK)

        gen_ids = context_ids[0, S:]
        n_new = int(gen_ids.shape[0])
        text = tok.decode(gen_ids, skip_special_tokens=True)
        r = {"tag": req["tag"], "prompt": req["text"], "tokens": n_new,
             "nfe": nfe, "prefill_s": round(prefill_s, 3),
             "gen_s": round(gen_s, 3), "tok_s": round(n_new / gen_s, 2),
             "text": text}
        results.append(r)
        print(f"\n--- {req['tag']} | {nfe} NFE | {n_new} tok | "
              f"{gen_s:.2f}s | {r['tok_s']} tok/s ---", flush=True)
        print(text, flush=True)

    send_longs([0, 0], DEN_RANK)  # shutdown
    print("\n" + "=" * 70)
    print("RESULTS_JSON " + json.dumps(results))
    print("=" * 70)
    for r in results:
        print(f"  {r['tag']:12s} {r['tokens']:4d} tok  {r['nfe']:4d} NFE  "
              f"{r['gen_s']:7.2f}s  {r['tok_s']:6.2f} tok/s")


def run_context_ar(model, cfg, args, device):
    """Single-node context-tower AR baseline on the SAME eager path."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    results = []
    for req in load_requests(args):
        ids = tok(req["text"], return_tensors="pt").input_ids.to(device)
        S = ids.shape[1]
        print(f"[ar] {req['tag']}: generating {req['max_new_tokens']}...",
              flush=True)
        t0 = time.perf_counter()
        out = model.generate_ar(ids, max_new_tokens=req["max_new_tokens"],
                                temperature=args.temperature,
                                eos_token_id=tok.eos_token_id)
        elapsed = time.perf_counter() - t0
        gen_ids = out[0, S:]
        n_new = int(gen_ids.shape[0])
        text = tok.decode(gen_ids, skip_special_tokens=True)
        r = {"tag": req["tag"], "tokens": n_new, "gen_s": round(elapsed, 3),
             "tok_s": round(n_new / elapsed, 2), "text": text}
        results.append(r)
        print(f"--- {req['tag']} | {n_new} tok | {elapsed:.2f}s | "
              f"{r['tok_s']} tok/s ---", flush=True)
        print(text, flush=True)
    print("RESULTS_JSON " + json.dumps(results))


def main():
    args = parse_args()
    device = "cuda:0"
    torch.set_grad_enabled(False)
    if args.mode == "ar":
        assert args.role == "ctx", "--mode ar runs on the context node"
        model, cfg, _ = load_model(args, device, fast_moe=bool(args.fast_moe),
                                   compile_kernels=bool(args.compile))
        run_context_ar(model, cfg, args, device)
        return
    rank = CTX_RANK if args.role == "ctx" else DEN_RANK
    dist.init_process_group(
        "gloo", init_method=f"tcp://{args.master}:{args.port}",
        rank=rank, world_size=2, timeout=timedelta(hours=12))
    print(f"[{args.role}] dist init ok (rank {rank})", flush=True)
    model, cfg, _ = load_model(args, device, fast_moe=bool(args.fast_moe),
                               compile_kernels=bool(args.compile))
    dist.barrier()
    if args.role == "ctx":
        run_context(model, cfg, args, device)
    else:
        run_denoiser(model, cfg, args, device)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
