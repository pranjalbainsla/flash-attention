# 05 — Papers and History

Crosswalk between publications and this monorepo. Read papers for *why*; read code for *how*.

## Timeline (conceptual)

| Era | Artifact | Repo location today |
|-----|----------|---------------------|
| 2022 | FlashAttention (FA1) paper + CUDA | Algorithm lives on in all gens; no separate FA1 tree |
| ~2022 | Triton tutorial fused attention | [`flash_attn/flash_attn_triton_og.py`](../flash_attn/flash_attn_triton_og.py) |
| 2023 | FlashAttention-2 | [`csrc/flash_attn/`](../csrc/flash_attn/) + [`flash_attn/flash_attn_interface.py`](../flash_attn/flash_attn_interface.py) |
| Experimental | Triton FA variant | [`flash_attn/flash_attn_triton.py`](../flash_attn/flash_attn_triton.py) (not production FA2) |
| 2024 | FlashAttention-3 (Hopper beta) | [`hopper/`](../hopper/) package `flash_attn_3` |
| 2025+ | FlashAttention-4 (CuTeDSL) | [`flash_attn/cute/`](../flash_attn/cute/) package `flash-attn-4` |

Root [`README.md`](../README.md) still leads with FA1/FA2 citations; FA3/FA4 sections document the newer stacks.

## Paper → concept → code

### FlashAttention (FA1) — [arxiv:2205.14135](https://arxiv.org/abs/2205.14135)

| Paper idea | Where to see it in code |
|------------|-------------------------|
| HBM vs SRAM IO analysis | This guide [01](01-gpu-primer.md)–[02](02-theory-io-and-tiling.md); comments/structure in all fwd kernels |
| Tiled attention | `BLOCK_M`/`BLOCK_N`, `kBlockM`/`kBlockN`, FA4 tile heuristics in `interface.py` |
| Online softmax | [03](03-theory-online-softmax.md); `softmax.h` / `softmax.py` |
| Exact attention | Tests vs PyTorch reference (`attention_ref` in cute bench/test utils) |

**Suggested paper skim:** intro + IO complexity + Algorithm 1 (forward). Skip proofs on first pass.

### FlashAttention-2 — [flash2.pdf](https://tridao.me/publications/flash2/flash2.pdf)

| Paper idea | Code |
|------------|------|
| Parallelism over sequence | Grid in `flash_fwd_launch_template.h` |
| Work partitioning inside block | `flash_fwd_kernel.h` + `kernel_traits.h` |
| Reduced non-matmul overhead | Inner loop in fwd kernel / `softmax_rescale_o` |

**Suggested skim:** parallelism & work partitioning sections; compare figures to FA2 launch grid.

### FlashAttention-3 — [flash3.pdf](https://tridao.me/publications/flash3/flash3.pdf) · [blog](https://tridao.me/blog/2024/flash3/)

| Paper idea | Code |
|------------|------|
| Producer/consumer warp specialization | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` |
| TMA + WGMMA pipeline | Same mainloop + FA4 `flash_fwd_sm90.py` |
| FP8 forward | Hopper FP8 benches / instantiations; FA4 FP8 paths on SM100 |
| Inter-warp ping-pong / async | Pipeline state in hopper + FA4 `pipeline.py` |

**Suggested skim:** blog first (intuition), then paper sections on asynchrony and low precision.

### FlashAttention-4 — [`assets/fa4_paper.pdf`](../assets/fa4_paper.pdf)

| Paper idea | Code |
|------------|------|
| CuTeDSL implementation | Entire `flash_attn/cute/` |
| Blackwell / 2CTA | `flash_fwd_sm100.py`, `AI/DEBUG_2CTA.md` |
| Extensibility (mods, sparsity) | `score_mod` / `mask_mod` / block sparsity tests |
| Persistent / CLC scheduling | `tile_scheduler.py`, `AI/CLC_TRACE_DEBUG.md` |

## Generations are separate entry points

There is **no** single runtime that auto-picks FA2 vs FA3 vs FA4 for you:

```python
# FA2 (root package flash_attn)
from flash_attn import flash_attn_func

# FA3 (install hopper/)
from flash_attn_3 import flash_attn_interface

# FA4
from flash_attn.cute import flash_attn_func
```

Within FA4, **arch** dispatch (SM80/90/100/120) *does* happen automatically in [`interface.py`](../flash_attn/cute/interface.py) via `_get_device_arch()`.

## Historical artifacts useful for learning

| Artifact | Use |
|----------|-----|
| `flash_attn_triton_og.py` | Shortest readable fused attention; great first kernel |
| `flash_attn_triton.py` | Fuller Triton FA (bias, bwd); still experimental |
| `assets/flash2_*.png`, `assets/flash3_*.png` | Perf plots from README |
| `usage.md` | Adoption list (ecosystem history, not API docs) |

## What “legacy” means here

- **Legacy for contributors focusing on FA4:** FA2 CUDA templates, FA3 instantiation explosion, Triton experimental files — still valuable to *read*, rarely the place to *patch* unless fixing FA2/FA3 users
- **Legacy / ignore for onboarding:** blocksparse FA1-era APIs, ROCm CK trees, training model zoo (see [06](06-repo-map.md))

## Citation

When you publish work that uses FlashAttention, cite the papers appropriate to the generation you use (BibTeX in root [`README.md`](../README.md) §Citation). Prefer citing FA2/FA3/FA4 if those are the kernels you actually ran.

## Next

[06 — Repo map](06-repo-map.md) — where everything lives and what to ignore
