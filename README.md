# keys-Nvidia — TRUE Two-Tower Diffusion on TWO DGX Sparks

Running **NVIDIA `Nemotron-Labs-TwoTower-30B-A3B-Base`** in its **full two-tower
mask-diffusion mode** — the way NVIDIA envisioned it for the diffusion base — with the
**context tower on one DGX Spark and the denoiser tower on a second DGX Spark**, talking
over a 200G RoCE fabric.

> **Status: VALIDATED (2026-07-04).** Coherent end-to-end cross-node mask-diffusion:
> **128 tokens in 72 NFE at 6.82 tok/s** (block 16, threshold 0.8, temp 0.1), context
> tower on one Spark, denoiser on the other. Full results below. The single-Spark AR
> baseline is published separately:
> [Keys-NVIDIA-Two-Tower-AR--single-dgx-spark](https://github.com/drowzeys/Keys-NVIDIA-Two-Tower-AR--single-dgx-spark).

## Results (first validated run, 2026-07-04)

Prompt `"France is a country "`, 128 new tokens, block_size 16, steps_per_block 16,
confidence_threshold 0.8, temperature 0.1 (NVIDIA's reference settings):

| Metric | Value |
|---|---|
| Output | coherent fluent prose (see below) |
| Total NFE (denoiser forwards) | **72** for 128 tokens = **1.78 tokens/NFE** |
| End-to-end throughput | **6.82 tok/s** (HF eager, cross-node) |
| Prefill (5-token prompt) | 1.33 s |
| Initial cache transfer over fabric | 0.06 s |
| Per-block denoise time | 4.7 s (block 0, cold) → 1.0–1.7 s steady-state |

Confidence unmasking works as designed: early blocks needed the full 16 steps, but once
context accumulated, blocks converged in **5–8 NFE** — the diffusion win NVIDIA's card
describes, reproduced across two machines.

Output sample (base model, low temperature):

```
It is in Europe.
It is beautiful.
It is romantic.
It is expensive.
It is delicious.
It is famous.
It is historic.
...
```

The interconnect is a non-factor: per-block fabric traffic (~25 MB) transfers in
milliseconds against 1.0–1.7 s of denoiser compute. Two Sparks over 200G behave like
NVIDIA's intended two-GPU box for this workload.

### vs the like-for-like AR baseline (same stack, same hardware)

`--mode ar` runs the context tower alone, single node, through the *same* HF-eager
torch path (one token per forward):

| Mode | tok/s | Forwards for 128 tok | Output |
|---|---|---|---|
| Cross-node two-tower mask-diffusion (2 Sparks) | 6.82 | **72 NFE** | coherent |
| Context-tower AR, single Spark, same eager stack | 7.39 | 128 | coherent (Fukushima/nuclear-policy prose) |

Honest read: **parity (0.92×), not NVIDIA's 2.42×** — on this unoptimized HF-eager
stack a 16-token denoiser forward costs ~1.9× a single-token decode, and per-block
context-extension is serialized with denoising. The NFE win (72 vs 128) is real and
reproduced; converting it into wall-clock speedup on GB10 needs the standard inference
optimizations (torch.compile/CUDA graphs, overlapping context extension with the next
block's early NFEs, fp32→bf16 scan where safe). That is the optimization runway, and
exactly where NVIDIA's H100 Megatron numbers come from.

Bonus result: this AR run is (to our knowledge) the **first coherent HF-Transformers
output from this model on GB10** — proving the historical "garbage output" was entirely
the compiled `causal_conv1d`/`mamba_ssm` sm_121a kernels, not the HF code path itself.
Remove the packages, fix the one list-indexing bug, and eager transformers is correct
on sm_121a.

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
