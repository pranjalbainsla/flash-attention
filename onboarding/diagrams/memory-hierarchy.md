# Diagram: Memory Hierarchy and Tiling

Companion to [01-gpu-primer.md](../01-gpu-primer.md) and [02-theory-io-and-tiling.md](../02-theory-io-and-tiling.md).

## HBM vs SRAM vs registers

```mermaid
flowchart TB
  subgraph hbm [HBM_device_DRAM]
    Q[Q_full]
    K[K_full]
    V[V_full]
    O[O_full]
    LSE[LSE_rows]
  end
  subgraph smem [SRAM_shared_memory_per_CTA]
    Qt[Q_tile_Br_x_d]
    Kt[K_tile_Bc_x_d]
    Vt[V_tile_Bc_x_d]
  end
  subgraph regs [Registers]
    S[Score_tile_Br_x_Bc]
    M[row_max]
    L[row_sum_or_LSE]
    Acc[acc_o_Br_x_d]
  end
  Q -->|load_once_per_Q_block| Qt
  K -->|stream_tiles| Kt
  V -->|stream_tiles| Vt
  Qt --> S
  Kt --> S
  S --> M
  S --> L
  S --> Acc
  Vt --> Acc
  Acc -->|write_once| O
  L -->|write_once| LSE
```

**Naïve attention** would also place a full `S` / `P` of shape `N×N` in HBM between steps. FlashAttention never does.

## One CTA’s traffic pattern (forward)

```text
HBM Q ──► SMEM/regs Q_tile ─────────────────────────────► (reuse across KV loop)
HBM K ──► SMEM K_tile_0 ──► scores ──► softmax upd ──► acc_o
HBM V ──► SMEM V_tile_0 ───────────────────────────────►
HBM K ──► SMEM K_tile_1 ──► scores ──► rescale+upd ──► acc_o
HBM V ──► SMEM V_tile_1 ───────────────────────────────►
 ...
acc_o, LSE ──► HBM O, LSE
```

Bytes from HBM scale like **O(N d)** per head for Q/K/V/O, not **O(N²)** for scores.

## Tile geometry

```text
                 KV sequence (N_k)
            ┌──────┬──────┬──────┐
            │ Bc   │ Bc   │ Bc   │
     ┌──────┼──────┼──────┼──────┤
Q    │ Br   │ S00  │ S01  │ S02  │  ← one CTA often owns one Br row-block
seq  ├──────┼──────┼──────┼──────┤     and loops across Bc columns
(N_q)│ Br   │ S10  │ S11  │ S12  │
     └──────┴──────┴──────┴──────┘

Sij lives only ephemerally in registers/SMEM for that step.
```

Typical teaching sizes: `Br = Bc = 64` or `128` (exact values vary by arch, head dim, causal, and generation).

## Capacity constraint (why tiles cannot be huge)

Rough shared-memory budget for a CTA (illustrative):

```text
bytes ≈ Br*d*sizeof(elem)      # Q
      + stages * Bc*d*sizeof   # K (and V) pipeline stages
      + extras (O buffers, barriers, metadata)
```

Must fit in the SM’s SMEM limit (and leave room for occupancy). Register pressure from `acc_o` (`Br × d` accumulators in FP32) is the other hard constraint.
