# keys-Nvidia — TRUE Two-Tower Diffusion on TWO DGX Sparks

Running **NVIDIA `Nemotron-Labs-TwoTower-30B-A3B-Base`** in its **full two-tower
mask-diffusion mode** — the way NVIDIA envisioned it for the diffusion base — with the
**context tower on one DGX Spark and the denoiser tower on a second DGX Spark**, talking
over a 200G RoCE fabric.

> **Status: WORK IN PROGRESS (2026-07-04).** The cross-node runner loads both towers,
> completes prefill, and is in first end-to-end validation now. Results, benchmarks, and
> the full wall-ledger land here as they are measured. The single-Spark AR baseline is
> already published:
> [Keys-NVIDIA-Two-Tower-AR--single-dgx-spark](https://github.com/drowzeys/Keys-NVIDIA-Two-Tower-AR--single-dgx-spark).

## Why two Sparks

NVIDIA's reference implementation is one process with two ~80 GB GPUs:

```python
model.place_towers_on_devices("cuda:0", "cuda:1")   # context | denoiser
outputs = model.generate_mask_diffusion(...)
```

Each tower is ~59 GB BF16 (~30B params, 3B active). A DGX Spark (GB10) is **one** GPU with
128 GB unified memory — one tower fits comfortably, two do not (118 GB + runtime + caches).
So the true two-tower diffusion base maps naturally onto **two Sparks, one tower each**:

| Rank | Node | Tower | Role |
|---|---|---|---|
| 0 | Spark A | **denoiser** (`denoiser_tower.*`, `lm_head`, `t_embedder`, `t_block`, `scale_shift_tables`) | runs the 16-step confidence-unmasking loop per block |
| 1 | Spark B | **context** (`context_tower.*`, `context_lm_head`) | prefills the prompt, commits blocks, owns the KV/Mamba cache |

## The cross-node design

`place_towers_on_devices("cuda:0","cuda:1")` becomes `torch.distributed` (gloo) over the
fabric. The tower boundary in NVIDIA's own code is clean — only cache tensors and token
blocks cross it — so the protocol is small:

1. **Prefill (once):** context rank builds its cache, ships the diffusion denoiser cache
   (23 Mamba conv/ssm states + full KV for the 6 attention layers) to the denoiser rank.
   With 2 KV heads × 128 head-dim, KV is ~6 KB/token — a few MB total.
2. **Per 16-token block:** the denoiser cache is *read-only within a block*, so the
   denoiser rank runs the entire mask-diffusion loop locally (up to 16 NFE) and sends back
   only the committed block (128 bytes). The context rank extends its cache block-wise and
   returns a delta: Mamba states replaced (~25 MB) + 16 tokens of KV appended (~100 KB).
3. **EOS/last block:** context rank flags stop; denoiser reports total NFE.

Cross-node traffic is ~25 MB per 16-token block — milliseconds on a 200G fabric. The
interconnect is *not* the bottleneck; this is why the two-tower split works across
machines at all.

## The GB10 walls (ledger so far)

| Wall | Symptom | Fix |
|---|---|---|
| HF `causal_conv1d`/`mamba_ssm` sm_121a kernels emit garbage | `" and, and, and…"` degeneration | don't install them at all: shim the one hard import (`rmsnorm_fn`, pure torch), let HF take its `torch_forward` fallback for prefill |
| `mamba_ssm` hard-import in `modeling_nemotron_h.py` | `ImportError` despite torch fallback existing | fake `mamba_ssm` package (`scripts/mamba_ssm/`) + dist-info so `importlib.metadata` version checks pass, fast-path names stubbed to `None` |
| Block kernels (`mamba_chunk_scan_combined`, `causal_conv1d_fn`) needed for ≤16-token block paths | no sm_121a build of either | exact pure-torch fp32 replacements (`scripts/tt_kernels.py`) — blocks are ≤16 tokens, so a sequential scan is exact and cheap |
| `torch_forward` decode branch: `cache_params.ssm_states.device` on a per-layer **list** | `AttributeError` at prefill pass 2 | shadowed `modeling_nemotron_h.py` with per-layer-indexed fix (upstream copy-paste bug) |
| 118 GB of towers vs 128 GB/node | OOM | meta-device init (`accelerate init_empty_weights`) + load **only this rank's tower** from the split checkpoint (`tt-context` / `tt-denoiser`, 59 GB each) |

## Files

```
scripts/
  twotower_dist.py       # the cross-node runner (both ranks; --role ctx|den)
  tt_kernels.py          # pure-torch fp32 block kernels (conv1d + SSD scan)
  tt-dist-launch.sh      # per-node docker launcher (gpu-clear + OOM watchdog)
  mamba_ssm/             # fake package: shims rmsnorm_fn, disables fast path
  mamba_ssm-2.2.4.dist-info/
```

Run (den rank hosts the rendezvous — start it first):

```bash
# Spark A (denoiser):
ROLE=den scripts/tt-dist-launch.sh --max-new-tokens 128 --block-size 16 \
  --steps-per-block 16 --temperature 0.1 --confidence-threshold 0.8
# Spark B (context):
ROLE=ctx scripts/tt-dist-launch.sh --max-new-tokens 128 --block-size 16 \
  --steps-per-block 16 --temperature 0.1 --confidence-threshold 0.8
```

## Credits

- Model: `nvidia/Nemotron-Labs-TwoTower-30B` (NVIDIA Open Model License).
- Recipe, scripts, measurements: MIT (see [`LICENSE`](LICENSE)).
- Companion AR baseline: [Keys-NVIDIA-Two-Tower-AR--single-dgx-spark](https://github.com/drowzeys/Keys-NVIDIA-Two-Tower-AR--single-dgx-spark).
