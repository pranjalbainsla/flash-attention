# 07 — Triton Kernels (Pedagogical Bridge)

Triton is the best first *code* contact with FlashAttention: the algorithm is visible in one Python file without CUTLASS traits or CuTe layouts. These kernels are **experimental / historical** for NVIDIA — production paths are FA2 CUDA, FA3, and FA4. Still: if you can read Triton FA, FA2/FA4 become pattern matching.

Paper concepts: same as FA1 — tiling + online softmax ([arxiv:2205.14135](https://arxiv.org/abs/2205.14135)).

## Files

| File | Role |
|------|------|
| [`flash_attn/flash_attn_triton_og.py`](../flash_attn/flash_attn_triton_og.py) | OpenAI tutorial-style fused attention; smallest |
| [`flash_attn/flash_attn_triton.py`](../flash_attn/flash_attn_triton.py) | Dao-lab experimental FA in Triton (fwd+bwd, bias, causal) |

Also exists: AMD Triton backends via ROCm / `third_party/aiter` — **ignore until you need AMD**.

## Why start here vs CUDA

| Concern | Triton | FA2 CUDA |
|---------|--------|----------|
| Expressing `BLOCK_M` loops | Obvious | Hidden in templates |
| Online softmax | Clear `m_i` / `lse_i` | Inlined in `softmax.h` + macros |
| Launch grid | `program_id(0/1)` | `flash_fwd_launch_template.h` |
| Performance | Good teaching vehicle | Production Ampere+ |

## Anatomy of `_fwd_kernel` ([`flash_attn_triton.py`](../flash_attn/flash_attn_triton.py))

### Work partitioning

```python
start_m = tl.program_id(0)   # which Q-block
off_hb  = tl.program_id(1)   # fused batch*head
```

Each program instance owns **one Q-row tile** (`BLOCK_M` rows) for one batch/head, then loops over KV tiles of size `BLOCK_N`.

**Vs naïve:** naïve would write a full `(seqlen_q, seqlen_k)` score matrix; here scores are a transient `BLOCK_M × BLOCK_N` register tile.

### State (online softmax)

```python
lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
```

Matches [03-theory-online-softmax.md](03-theory-online-softmax.md).

### Inner loop (structure)

1. Load K (and later V) tile with masks for sequence tails (`EVEN_M` / edge cases)
2. `qk = Q @ K.T` (often FP32 accumulate)
3. Optional bias / scale
4. Causal: limit `end_n` and mask future positions
5. Update `m_ij`, `p = exp(...)`, rescale `acc_o`, accumulate `p @ V`
6. Update `lse_i`

### Final store

Normalize `acc_o` with `exp(m_i - lse_i)`, write O and LSE to HBM **once**.

## Hardware constraints Triton exploits (implicitly)

- **SRAM/register tiles** sized by `BLOCK_M`/`BLOCK_N` so the compiler keeps working sets on-chip
- **`num_warps` / `num_stages`** in configs: occupancy and software pipelining of loads
- **FP32 accumulators** for softmax stability while inputs may be FP16/BF16

Triton does not expose TMA producer warps the way FA3 does; the compiler schedules memory. That is why FA3/FA4 code looks “more complicated” for the same math — they manually orchestrate Hopper/Blackwell async engines.

## Causal loop difference vs non-causal

```python
end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
```

Fewer KV iterations for early Q-blocks. Inside a tile, lanes corresponding to `q_idx < k_idx` get `-inf` scores.

**Pitfall:** off-by-one in causal limits → wrong attention near the diagonal only. Always test causal + non-causal.

## Backward kernels (skim)

Bwd recomputes attention probabilities using stored LSE (and Q/K/V), then forms `dQ`, `dK`, `dV`. It is more complex than fwd (atomics / sequencing for `dK`/`dV`). For intuition: **recompute beats storing P**. Read after you are solid on forward.

## Exercise (no GPU required)

1. In `_fwd_kernel`, list every HBM store. Confirm scores are not stored.
2. Trace one rescale: find `acc_o_scale = tl.exp(m_i - m_ij)` and the multiply into `acc_o`.
3. Compare `flash_attn_triton_og.py` — which statistics does it keep (`M`/`L` vs `lse_i`)? Same algebra?

## How this maps to FA2/FA4

| Triton | FA2 | FA4 |
|--------|-----|-----|
| `program_id` grid | `flash_fwd_launch_template.h` grid | Tile scheduler + grid in interface/kernel |
| `BLOCK_M/N` | `kBlockM/N` in traits | Tile args / heuristics in `interface.py` |
| `m_i`, `lse_i`, `acc_o` | `softmax.h` | `softmax.py` |
| `@triton.jit` | `__global__` templates | `@cute.jit` / `cute.compile` |

## Next

[08 — FA2 CUDA](08-fa2-cuda.md)
