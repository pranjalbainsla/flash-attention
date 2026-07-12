# 03 — Online Softmax

Paper anchor: [FlashAttention](https://arxiv.org/abs/2205.14135) algorithm section; also Milakov & Gimelshein “Online normalizer calculation for softmax” (the streaming softmax technique FA builds on).

Tiling raises a problem: **softmax needs the max (and sum of exps) over the whole row**, but you only see one `Br × Bc` tile at a time. Online softmax solves that without storing the full row.

## Standard softmax (one row)

For a row of scores `x_0, …, x_{N-1}`:

```
m = max_j x_j
p_j = exp(x_j - m)
ℓ = sum_j p_j
output uses p_j / ℓ
```

Subtracting `m` is for **numerical stability**. The issue for tiling: you do not know the global `m` until you have seen all tiles.

## Online / streaming update

Maintain running statistics as tiles arrive. For tile `i` with local scores `x^{(i)}`:

```
m_i     = max(m_{i-1}, max(x^{(i)}))
ℓ_i     = exp(m_{i-1} - m_i) * ℓ_{i-1} + sum(exp(x^{(i)} - m_i))
```

When the max grows, **rescale** the previous sum by `exp(m_{i-1} - m_i)`.

The output accumulator `O` must rescale the same way: previous contributions were computed under the old max / normalization. FlashAttention tracks:

| State | Meaning |
|-------|---------|
| `m_i` / `row_max` | Running max of scores for this Q row |
| `ℓ_i` / `row_sum` | Running sum of `exp(score - max)` |
| `acc_o` | Unnormalized (or partially normalized) output accumulator |
| `LSE` | Log-sum-exp `m + log(ℓ)` written for backward |

When a new tile arrives with a larger max, multiply `acc_o` by `exp(m_old - m_new)` before adding `P_tile @ V_tile`.

See [diagrams/online-softmax-flow.md](diagrams/online-softmax-flow.md).

## Why this is exact

Algebraically, streaming updates compute the same `m` and `ℓ` as a full-row pass (again, up to FP rounding order). You never need the full score row in memory — only the current tile plus a few scalars per row.

## LSE (log-sum-exp) and the backward pass

Forward often stores **LSE** per row:

```
LSE = m + log(ℓ)
```

Backward needs the attention probabilities again. Recomputing them from Q/K and LSE is cheaper than storing the full `P` matrix — another IO win, and why FA kernels are careful about LSE layout (and about **SplitKV combine**, which must merge partial LSEs correctly).

## Where this appears in the repo

| Stack | File | Symbols |
|-------|------|---------|
| Triton (experimental) | [`flash_attn/flash_attn_triton.py`](../flash_attn/flash_attn_triton.py) | `m_i`, `lse_i`, `acc_o`, `acc_o_scale` |
| FA2 CUDA | [`csrc/flash_attn/src/softmax.h`](../csrc/flash_attn/src/softmax.h) | `Softmax`, `softmax_rescale_o` |
| FA3 | [`hopper/softmax.h`](../hopper/softmax.h) | Same idea, CUTLASS tensors |
| FA4 | [`flash_attn/cute/softmax.py`](../flash_attn/cute/softmax.py) | `Softmax` with `row_max` / `row_sum`, score_mod hooks |

Reading **one** of these thoroughly (start with Triton or FA4 `softmax.py`) transfers to all of them.

## Implementation sketch (matches Triton `_fwd_kernel`)

```python
# Pseudocode for one Q-block
m_i = -inf
lse_i = -inf   # or track l_i separately
acc_o = 0

for each K/V block:
    qk = Q_tile @ K_tile.T          # Br × Bc, often in FP32 accum
    # optional: apply scale, mask, score_mod inside tile
    m_ij = max(m_i, rowmax(qk))
    p = exp(qk - m_ij)              # or exp(qk * scale - m_ij)
    acc_o *= exp(m_i - m_ij)        # rescale old output
    acc_o += p @ V_tile
    lse_i = m_ij + log(exp(lse_i - m_ij) + rowsum(p))
    m_i = m_ij

O_tile = acc_o * exp(m_i - lse_i)   # final normalize
# store O_tile and lse_i
```

FA2/FA4 may use `exp2` with a log2-base rescale for speed (`fast_math`); the structure is identical.

## Causal masking inside the tile

For causal attention, some entries in the `Br × Bc` tile are invalid (future keys). Kernels set those scores to `-inf` (or skip them) **before** the max/exp. Getting the wrong `-inf` pattern is a classic bug: silent wrong answers near the diagonal, not a crash.

## Softcap / score_mod

Extra per-score transforms (tanh softcap, ALiBi, custom `score_mod`) apply **on the score tile** before online softmax. FA4 injects user `@cute.jit` callables at compile time ([`softmax.py`](../flash_attn/cute/softmax.py) `call_score_mod`). That preserves fusion: still no full `S` in HBM.

## Checkpoint questions

1. Why rescale `acc_o` when `m` increases?
2. What is stored to HBM for the backward pass instead of `P`?
3. Open [`flash_attn_triton.py`](../flash_attn/flash_attn_triton.py) and find the three lines that update `m_i`, `lse_i`, and `acc_o_scale`.

## Next

[04 — Parallelism evolution](04-theory-parallelism.md)
