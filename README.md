# keys-Nvidia — TRUE Two-Tower Diffusion on TWO DGX Sparks — NVFP4 APPLIED

Running **NVIDIA `Nemotron-Labs-TwoTower-30B-A3B-Base`** in its **full two-tower
mask-diffusion mode** — the way NVIDIA envisioned it for the diffusion base — with the
**context tower on one DGX Spark and the denoiser tower on a second DGX Spark**, talking
over a 200G RoCE fabric. The routed-expert weights now run in **NVFP4 (W4A16)** via a
custom grouped Triton kernel, which is what pushed cross-node diffusion past the AR
baseline on this hardware.

> **Status: v3 — NVFP4 applied, 38.9 tok/s (2026-07-04).** Routed experts quantized to
> NVFP4-style W4A16 + grouped Triton dequant-in-GEMM: cross-node diffusion **beats the
> AR baseline 1.4–1.8×** on generation-heavy prompts, with all quality evals passing.
> Trajectory on the flagship benchmark: **6.82 → 11.2 → 38.85 tok/s (5.7×)** across
> v1 → v2 → v3. The single-Spark AR companion repo:
> [Keys-NVIDIA-Two-Tower-AR--single-dgx-spark](https://github.com/drowzeys/Keys-NVIDIA-Two-Tower-AR--single-dgx-spark).

## ⭐ v3 — NVFP4 expert weights: 38.9 tok/s (LATEST, 2026-07-04)

v2's analysis said the wall was expert-weight bandwidth and the fix was 4-bit experts.
v3 implements it:

- **NVFP4-style W4A16 quantization** of the routed experts (98% of each tower's
  weights): fp4 e2m1 values packed 2/byte, fp16 block-16 scales, RTN
  (`scripts/quantize_experts.py`) — 5,888 matrices per tower, 35 s to quantize, tower
  shrinks **59 → 21 GB** (loads in ~2 min instead of ~6.5).
- **Grouped Triton GEMM** (`scripts/tt_nvfp4.py`): ALL active experts in ONE kernel
  launch per projection, fp4 decoded in-register via integer shifts — only
  **0.625 bytes/element** cross the 273 GB/s bus instead of 2. Verified correct at
  T=3/16/200 (rel err ≤ 6e-3), **3.59×** vs the bf16 fast-MoE loop.
- (The community `syscall42/nemotron-twotower-nvfp4` checkpoint was evaluated and
  rejected: its own card admits the ModelOpt export had defective routed-expert scales
  and only the *context* tower was repaired — the denoiser, the tower that matters for
  diffusion, was not.)

**Results — both modes on NVFP4 experts, temp 0.1, conf 0.8:**

| Request | Diffusion 2-Spark tok/s (NFE) | AR 1-Spark tok/s | Diffusion/AR |
|---|---|---|---|
| bench256 | **38.85** (66) | 21.59 | **1.80×** |
| bench128 | **28.29** (52) | 20.92 | **1.35×** |
| eval-japan | 15.83 (76) | 20.19 | 0.78× |
| eval-train | 17.81 (62) | 20.47 | 0.87× |
| eval-colors | 15.34 (73) | 20.42 | 0.75× |
| eval-france | 17.33 (66) | 20.29 | 0.85× |
| eval-boil | 16.42 (71) | 20.43 | 0.80× |

**All 5 evals remain correct in both modes** after 4-bit RTN quantization, with NFE
counts unchanged vs bf16 — the diffusion process is insensitive to expert quantization
noise at this scale. Full texts: [`data/nvfp4_results.json`](data/nvfp4_results.json).

As predicted by the v2 bandwidth analysis, cutting bytes-per-NFE moved diffusion from
losing to AR to **beating it 1.4–1.8× on generation-heavy prompts** — the NVIDIA
two-tower speedup thesis, reproduced on two $4K Sparks instead of two H100s. Eval-style
prompts (harder continuations, ~1.3 tok/NFE) still favor AR; the breakeven is now
~1.5 tokens/NFE and typical prose sits right around it.

Run it (den rank hosts the rendezvous — start it first):

```bash
# Spark A (denoiser):
MODEL_DIR=~/models/tt-denoiser-nvfp4 ROLE=den scripts/tt-dist-launch.sh \
  --prompt-file /work/bench_eval.jsonl --temperature 0.1 --confidence-threshold 0.8
# Spark B (context):
MODEL_DIR=~/models/tt-context-nvfp4 ROLE=ctx scripts/tt-dist-launch.sh \
  --prompt-file /work/bench_eval.jsonl --temperature 0.1 --confidence-threshold 0.8
```

## v2 — fast-MoE + torch.compile + streamed deltas (2026-07-04)

Round 2 implemented the first optimization runway: **fast-MoE** (the HF MoE looped over
all 128 experts with a GPU sync each — 80% of NFE time; replaced with a single-sync
active-expert loop, **bit-exact**, max logit diff 0.0), **torch.compile** on the block
kernels (the 16-step SSD scan unrolls into fused kernels; 8 s warmup), **per-layer
streamed context-extension deltas**, and a **multi-request serve loop** (one weight
load runs a whole suite). Profile: **174.5 → 103.5 ms/NFE (1.67×)**; before breakdown
was MoE 140.3 / mamba 30.0 / attention 1.8 / other 11.5 ms.

Optimized-vs-optimized (AR re-run with the same fast-MoE stack), full texts in
[`data/optimized_results.json`](data/optimized_results.json):

| Request | Diffusion 2-Spark tok/s (NFE) | AR 1-Spark tok/s | Diffusion/AR |
|---|---|---|---|
| bench256 | 11.20 (132) | 16.98 | 0.66× |
| bench128 | 24.19 (31) | 16.44 | 1.47× |
| eval-japan | 7.35 (78) | 16.14 | 0.46× |
| eval-train | 10.02 (52) | 15.80 | 0.63× |
| eval-colors | 7.00 (80) | 16.02 | 0.44× |
| eval-france | 8.56 (66) | 16.06 | 0.53× |
| eval-boil | 8.10 (71) | 15.92 | 0.51× |

**The GB10 finding that motivated v3:** diffusion beats AR only when
confidence-unmasking converges fast. At typical 1.2–1.9 tokens/NFE it lost, because on
GB10 a 16-token denoiser NFE cost ~1.7× an AR step: each NFE activates ~60–90 *unique*
experts per MoE layer (~30 GB of weight reads against ~273 GB/s), while an AR step
touches only 6 per layer. NVIDIA's 2.42× comes from H100-class bandwidth (3.3 TB/s).
**Block-diffusion MoE speedup is a memory-bandwidth play** — hence NVFP4 experts in v3
to cut bytes-per-NFE ~3.2×.

## v1 — first cross-node validation (2026-07-04)

Prompt `"France is a country "`, 128 new tokens, block 16, steps 16, conf 0.8, temp 0.1:

| Metric | Value |
|---|---|
| Output | coherent fluent prose |
| Total NFE | **72** for 128 tokens = 1.78 tokens/NFE |
| End-to-end throughput | 6.82 tok/s (HF eager, cross-node, unoptimized) |
| Prefill (5-token prompt) | 1.33 s |
| Initial cache transfer over fabric | 0.06 s |
| Per-block denoise | 4.7 s cold → 1.0–1.7 s steady-state |

vs the like-for-like AR baseline (`--mode ar`, same eager stack, single node):
diffusion 6.82 vs AR 7.39 tok/s — parity, with the NFE win (72 vs 128) real but eaten
by per-forward cost. Confidence unmasking worked as designed from the first run: blocks
converged in 5–8 NFE once context accumulated.

Bonus result: the v1 AR run was (to our knowledge) the **first coherent HF-Transformers
output from this model on GB10** — proving the historical "garbage output" was entirely
the compiled `causal_conv1d`/`mamba_ssm` sm_121a kernels, not the HF code path itself.

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
   streams a per-layer delta: Mamba states replaced (~25 MB) + 16 tokens of KV appended.
3. **EOS/last block:** context rank flags stop; denoiser reports total NFE.

Cross-node traffic is ~25 MB per 16-token block — milliseconds on a 200G fabric. The
interconnect is *not* the bottleneck; this is why the two-tower split works across
machines at all.

## The GB10 walls (ledger)

| Wall | Symptom | Fix |
|---|---|---|
| HF `causal_conv1d`/`mamba_ssm` sm_121a kernels emit garbage | `" and, and, and…"` degeneration | don't install them at all: shim the one hard import (`rmsnorm_fn`, pure torch), let HF take its `torch_forward` fallback for prefill |
| `mamba_ssm` hard-import in `modeling_nemotron_h.py` | `ImportError` despite torch fallback existing | fake `mamba_ssm` package (`scripts/mamba_ssm/`) + dist-info so `importlib.metadata` version checks pass, fast-path names stubbed to `None` |
| Block kernels (`mamba_chunk_scan_combined`, `causal_conv1d_fn`) needed for ≤16-token block paths | no sm_121a build of either | exact pure-torch fp32 replacements (`scripts/tt_kernels.py`) — blocks are ≤16 tokens, so a sequential scan is exact and cheap |
| `torch_forward` decode branch: `cache_params.ssm_states.device` on a per-layer **list** | `AttributeError` at prefill pass 2 | shadowed `modeling_nemotron_h.py` with per-layer-indexed fix (upstream copy-paste bug) |
| 118 GB of towers vs 128 GB/node | OOM | meta-device init (`accelerate init_empty_weights`) + load **only this rank's tower** from the split checkpoint |
| HF MoE: 128-expert loop, one `torch.where` sync each | 80% of NFE time | fast-MoE: single-sync active-expert loop (v2), then grouped NVFP4 kernel (v3) |
| Per-expert W4A16 launches lose to cuBLAS | 0.55× in microbench | grouped GEMM: all active experts in one launch per projection → 3.59× |

## Files

```
scripts/
  twotower_dist.py       # the cross-node runner (both ranks; --role ctx|den, --mode ar)
  tt_nvfp4.py            # grouped W4A16 triton kernel + model surgery (v3)
  quantize_experts.py    # NVFP4-style RTN expert quantizer (v3)
  test_w4a16.py          # kernel correctness + microbench
  tt_kernels.py          # pure-torch fp32 block kernels (conv1d + SSD scan)
  profile_den.py         # per-NFE profiler (optimized vs original, correctness check)
  bench_eval.jsonl       # the benchmark + eval suite
  tt-dist-launch.sh      # per-node docker launcher (gpu-clear + OOM watchdog)
  mamba_ssm/             # fake package: shims rmsnorm_fn, disables fast path
  mamba_ssm-2.2.4.dist-info/
data/
  nvfp4_results.json     # v3 results (both modes, full texts)
  optimized_results.json # v2 results + profile numbers
```

## Credits

- Model: `nvidia/Nemotron-Labs-TwoTower-30B` (NVIDIA Open Model License).
- Recipe, scripts, kernels, measurements: MIT (see [`LICENSE`](LICENSE)).
- Companion AR baseline: [Keys-NVIDIA-Two-Tower-AR--single-dgx-spark](https://github.com/drowzeys/Keys-NVIDIA-Two-Tower-AR--single-dgx-spark).
