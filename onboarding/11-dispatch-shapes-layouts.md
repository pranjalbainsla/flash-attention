# 11 — Dispatch, Tensor Shapes, and Memory Layouts

How a user call becomes a kernel launch, and what layouts kernels assume.

## Three packages, three dispatch stories

```text
flash_attn_func  ──► FA2 CUDA extension (flash_attn_2_cuda)
flash_attn_3...  ──► FA3 extension (_C) + ARCH_SWITCH
flash_attn.cute  ──► CuTeDSL compile cache + SM80/90/100/120 class
```

There is **no** cross-generation auto-selector. Pick the package intentionally. Within FA4, arch selection is automatic.

Full graphs: [diagrams/call-graphs.md](diagrams/call-graphs.md).

## FA4 call path (detail)

1. User calls `flash_attn_func(q, k, v, ...)` in [`interface.py`](../flash_attn/cute/interface.py)
2. Validate dtype, device, contiguity, head dims, optional mods / block sparsity
3. `_get_device_arch()` → choose forward (and later backward) implementation class
4. Tile heuristics (`m_block_size`, `n_block_size`, stages, num_threads, …)
5. Build compile key → hit JIT cache or `cute.compile`
6. Launch kernel(s); if SplitKV, launch [`FlashAttentionForwardCombine`](../flash_attn/cute/flash_fwd_combine.py)

FakeTensor mode (`FLASH_ATTENTION_FAKE_TENSOR=1`) stops after compile for CI pass 1.

## Tensor shapes (FA4 / FA2 common convention)

| Tensor | Shape | Notes |
|--------|-------|-------|
| `q` | `(B, S_q, H_q, D)` | Head dim last, contiguous |
| `k` | `(B, S_k, H_k, D)` | `H_q` ≥ `H_k` for GQA |
| `v` | `(B, S_k, H_k, D_v)` | `D_v` may differ (MLA) |
| `o` | `(B, S_q, H_q, D_v)` | |
| `lse` | `(B, H_q, S_q)` | Layout may vary by API — check callers |

**Varlen:** packed sequences with `cu_seqlens_q` / `cu_seqlens_k` (int32 prefix sums), total length as batch dim flattened.

**Paged KV:** `k`/`v` as paged cache + `page_table` — see [`paged_kv.py`](../flash_attn/cute/paged_kv.py).

### Alignment

Kernels assume **16-byte alignment** along the contiguous head dimension (e.g. `D % 8 == 0` for FP16). Violating this is a common integration bug.

## Strides mentally

Kernels receive not only shapes but **strides** so they can support non-trivial batch/seq layouts as long as the last dim is contiguous. When debugging wrong outputs, print `tensor.stride()` beside shape.

## Tile sizes (who decides)

| Stack | Where |
|-------|-------|
| Triton | `BLOCK_M`/`BLOCK_N` in autotune configs |
| FA2 | `kernel_traits` + `run_mha_fwd_hdim*` |
| FA3 | [`hopper/tile_size.h`](../hopper/tile_size.h) |
| FA4 | Heuristics in [`interface.py`](../flash_attn/cute/interface.py) (e.g. `_tile_size_fwd_sm90`) + overrides |

Causal often prefers different `kBlockN` than bidirectional — softmax + masking overhead vs MMA shape.

## Feature flags that change dispatch

Examples (FA4): `causal`, `window_size_left/right`, `softcap`, `score_mod`, `mask_mod`, `block_sparse_tensors`, `num_splits`, `pack_gqa`, page table present, FP8 descale tensors.

Each flag either:

- becomes part of the **compile key** (specialized kernel), or
- selects a **different code path** / kernel class

That is why adding a feature has a compile-cache and testing cost.

## Work partitioning recap

| Axis | Mechanism |
|------|-----------|
| Q sequence | m-blocks across CTAs |
| Batch / heads | grid Z or folded into scheduler |
| KV sequence | n-loop inside CTA; SplitKV across CTAs |
| GQA | PackGQA or head broadcasting |
| Persistent | tile scheduler pulls next work (FA4 SM100) |

## Checklist when adding a parameter

1. Python API surface (`interface.py`)
2. Validation / shape rules
3. Compile key + Constexpr plumbing into kernel
4. Host-side tensor args to launch
5. Tests in `tests/cute/` (include a negative / edge case)
6. Bench if it affects the hot path

## Next

[12 — Benchmarks and performance](12-benchmarks-and-perf.md)
