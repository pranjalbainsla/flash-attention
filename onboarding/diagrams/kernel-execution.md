# Diagram: Kernel Execution Across Generations

Companion to [04-theory-parallelism.md](../04-theory-parallelism.md).

## FA2-style grid (Ampere CUDA)

```text
gridDim = (num_m_blocks, batch, num_heads)

        batch
       ┌─────────────────────►
 heads │  CTA(m0,b0,h0) CTA(m1,b0,h0) ...
       │  CTA(m0,b0,h1) ...
       ▼

Each CTA:
  - loads Q[m_block] for (b, h)
  - loops n_blocks over K/V
  - writes O[m_block], LSE[m_block]
```

```mermaid
flowchart TB
  Launch[Kernel_launch]
  Launch --> CTA[CTA_owns_Q_row_block]
  CTA --> LoadQ[Load_Q_tile_SMEM]
  LoadQ --> Loop[For_each_KV_block]
  Loop --> CopyKV[cp_async_K_V]
  CopyKV --> MMA1[MMA_QK]
  MMA1 --> Soft[Online_softmax]
  Soft --> MMA2[MMA_PV]
  MMA2 --> Loop
  Loop --> Store[Store_O_LSE]
```

## FA3 / FA4 Hopper: warp-specialized pipeline

```text
Within one CTA (simplified):

  Producer warps          Consumer warps
  ──────────────          ──────────────
  issue TMA Q/K/V    →    wait buffer ready
  advance pipeline        WGMMA QK
                          softmax + rescale
                          WGMMA PV
                          signal buffer free
```

Ping-pong / multi-stage SMEM buffers hide HBM latency behind MMA. See `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` and FA4 `flash_fwd_sm90.py` + `pipeline.py`.

```mermaid
sequenceDiagram
  participant Prod as Producer_warps
  participant SMEM as SMEM_stages
  participant Cons as Consumer_warps
  Prod->>SMEM: TMA_store_K_V_stage_i
  Cons->>SMEM: wait_stage_i
  Cons->>Cons: MMA_softmax_MMA
  Cons->>SMEM: release_stage_i
  Prod->>SMEM: TMA_store_stage_i_plus_1
```

## FA4 Blackwell extras

| Mechanism | Role |
|-----------|------|
| Persistent CTA / CLC scheduler | CTA pulls next tiles instead of one-tile-per-CTA only |
| SplitKV | Multiple CTAs on one Q-block’s KV range |
| 2CTA (cluster) | Two CTAs cooperate for large MMA / hdim patterns |
| UMMA | Blackwell matrix units |

Debug hangs in 2CTA/barrier code with [`AI/DEBUG_2CTA.md`](../../AI/DEBUG_2CTA.md).

## Thread roles (mental model)

| Generation | Who loads | Who computes softmax | Sync |
|------------|-----------|----------------------|------|
| Triton | Same program (compiler schedules) | Same | Implicit barriers |
| FA2 | Threads issue `cp.async` | Same CTA warps | CTA barriers |
| FA3/FA4 SM90 | Producer warps + TMA | Consumer warps | Pipeline mbarriers |
| FA4 SM100 | TMA/UMMA paths + optional 2CTA | Consumer / MMA paths | Cluster + pipeline barriers |
