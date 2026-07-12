# 10 — FA4 CuTeDSL (Active Development)

FlashAttention-4 is the contribution target: Python CuTeDSL kernels JIT-compiled to PTX/CUBIN for Hopper and Blackwell (also Sm80/Sm120 paths).

Package: `flash-attn-4` · code: [`flash_attn/cute/`](../flash_attn/cute/) · paper: [`assets/fa4_paper.pdf`](../assets/fa4_paper.pdf).

Install (dev):

```bash
pip install -e "flash_attn/cute[dev]"
```

## Why CuTeDSL

| Goal | How FA4 addresses it |
|------|----------------------|
| Track NVIDIA archs quickly | Python kernels + `cute.compile` |
| Research hooks | `score_mod` / `mask_mod` as `@cute.jit` callables |
| Perf | Still generates specialized CUBIN; disk/JIT cache |
| Readability vs FA3 instantiations | One `.py` per arch instead of thousands of `.cu` clones |

The **algorithm is still** tiled online softmax. The novelty is the programming model and Blackwell features.

## Public API

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

out = flash_attn_func(q, k, v, causal=True)
```

Defined in [`interface.py`](../flash_attn/cute/interface.py), exported from [`__init__.py`](../flash_attn/cute/__init__.py).

Layout: `(batch, seqlen, num_heads, head_dim)`, last dim contiguous, 16-byte aligned.

## Arch dispatch

`_get_device_arch()` uses `torch.cuda.get_device_capability()` (override with `FLASH_ATTENTION_ARCH`):

| Capability | Forward class |
|------------|---------------|
| 8.x | `FlashAttentionForwardSm80` — [`flash_fwd.py`](../flash_attn/cute/flash_fwd.py) |
| 9.x | `FlashAttentionForwardSm90` — [`flash_fwd_sm90.py`](../flash_attn/cute/flash_fwd_sm90.py) |
| 10.x / 11.x | `FlashAttentionForwardSm100` — [`flash_fwd_sm100.py`](../flash_attn/cute/flash_fwd_sm100.py) |
| 12.x | `FlashAttentionForwardSm120` — [`flash_fwd_sm120.py`](../flash_attn/cute/flash_fwd_sm120.py) |

Special cases (MLA, large head dim) also live in `interface.py`. Prefer reading dispatch there before diving into a kernel file.

## Major forward kernels — what / why / vs naïve

### Sm80 — [`flash_fwd.py`](../flash_attn/cute/flash_fwd.py)

- **What:** Ampere-style CuTeDSL forward (cp.async-era patterns)
- **Why structured this way:** shared FA4 abstractions (`softmax`, `mask`, `block_info`) even on older GPUs
- **Hardware:** SM80 MMA + async copy style constraints
- **Vs naïve:** fused tile loop; no HBM scores

### Sm90 — [`flash_fwd_sm90.py`](../flash_attn/cute/flash_fwd_sm90.py)

- **What:** Hopper forward with TMA/WGMMA, stages, PackGQA, paged KV
- **Why:** expose Hopper async engines explicitly (like FA3), but in CuTeDSL
- **Hardware:** TMA alignment, warp-group MMA, pipeline SMEM stages ([`pipeline.py`](../flash_attn/cute/pipeline.py); Hopper helpers via `cutlass.utils.hopper_helpers`, plus [`ampere_helpers.py`](../flash_attn/cute/ampere_helpers.py) for Sm80 patterns)
- **Vs naïve / vs FA2:** producer-consumer overlap; larger tiles via Hopper MMA

### Sm100 — [`flash_fwd_sm100.py`](../flash_attn/cute/flash_fwd_sm100.py)

- **What:** Blackwell forward — fullest feature set
- **Why complex:** UMMA, **2CTA**, SplitKV, persistent scheduling, paged KV
- **Hardware:** cluster mbarriers, `tx_count` × `cta_group_size` for 2CTA, CLC/persistent tile pull ([`tile_scheduler.py`](../flash_attn/cute/tile_scheduler.py), [`blackwell_helpers.py`](../flash_attn/cute/blackwell_helpers.py))
- **Vs naïve:** same math; parallelism across KV (SplitKV) and clusters

SplitKV combine: [`flash_fwd_combine.py`](../flash_attn/cute/flash_fwd_combine.py).

## Shared abstractions (read these early)

| Module | Role |
|--------|------|
| [`softmax.py`](../flash_attn/cute/softmax.py) | `row_max` / `row_sum`, rescale, score_mod |
| [`mask.py`](../flash_attn/cute/mask.py) | Causal, window, block sparse, mask_mod |
| [`block_info.py`](../flash_attn/cute/block_info.py) | Legal n-block ranges |
| [`seqlen_info.py`](../flash_attn/cute/seqlen_info.py) | Varlen offsets |
| [`paged_kv.py`](../flash_attn/cute/paged_kv.py) | Paged KV manager |
| [`pack_gqa.py`](../flash_attn/cute/pack_gqa.py) | GQA packing |

Dependency picture: [diagrams/dependency-map.md](diagrams/dependency-map.md).

## Compilation and caching

1. Build a `compile_key` from dtype, dims, flags, mod hashes, tiles, arch
2. `cute.compile` → cached callable
3. Optional disk cache: `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` → `/tmp/$USER/flash_attention_cute_dsl_cache/`

See [`cache_utils.py`](../flash_attn/cute/cache_utils.py), [`cute_dsl_utils.py`](../flash_attn/cute/cute_dsl_utils.py).

Debug env vars: `CUTE_DSL_KEEP_PTX=1`, `CUTE_CUBIN_PATH`, `CUTE_DSL_LINEINFO=1`.

## Backward

- [`flash_bwd.py`](../flash_attn/cute/flash_bwd.py) — Sm80 base patterns
- [`flash_bwd_sm90.py`](../flash_attn/cute/flash_bwd_sm90.py), [`flash_bwd_sm100.py`](../flash_attn/cute/flash_bwd_sm100.py)
- Pre/postprocess helpers alongside

Learn forward thoroughly before bwd — bwd recomputes attention using LSE and is denser.

## Constexpr and mods

Compile-time constants use `cutlass.Constexpr[...]`. User `score_mod` / `mask_mod` are injected at compile time so the kernel stays fused (no callback into Python per element at runtime).

## Reading order inside FA4

1. `interface.py` — `flash_attn_func` + arch branch (~30 min skim)
2. `softmax.py` — online softmax + score_mod
3. `block_info.py` + `mask.py` — causal ranges
4. `flash_fwd_sm90.py` **or** `flash_fwd_sm100.py` matching your GPU
5. `pipeline.py` / helpers as needed when you hit barriers

## Next

[11 — Dispatch, shapes, layouts](11-dispatch-shapes-layouts.md)
