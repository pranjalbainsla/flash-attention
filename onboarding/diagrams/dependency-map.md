# Diagram: FA4 Module Dependency Map

Companion to [10-fa4-cute.md](../10-fa4-cute.md).

## Core dependency graph

```mermaid
flowchart TB
  init["__init__.py exports"]
  iface["interface.py API_dispatch"]
  init --> iface

  iface --> fwd80["flash_fwd.py Sm80"]
  iface --> fwd90["flash_fwd_sm90.py"]
  iface --> fwd100["flash_fwd_sm100.py"]
  iface --> fwd120["flash_fwd_sm120.py"]
  iface --> combine["flash_fwd_combine.py"]
  iface --> bwd["flash_bwd*.py"]

  fwd90 --> softmax["softmax.py"]
  fwd100 --> softmax
  fwd80 --> softmax
  fwd90 --> mask["mask.py"]
  fwd100 --> mask
  fwd90 --> binfo["block_info.py"]
  fwd100 --> binfo
  fwd90 --> seqlen["seqlen_info.py"]
  fwd100 --> seqlen
  fwd90 --> pipe["pipeline.py"]
  fwd100 --> pipe
  fwd90 --> sched["tile_scheduler.py"]
  fwd100 --> sched
  fwd90 --> hop["cutlass.utils.hopper_helpers"]
  fwd80 --> amp["ampere_helpers.py"]
  fwd100 --> blk["blackwell_helpers.py"]
  fwd100 --> paged["paged_kv.py"]
  fwd90 --> pack["pack_gqa.py"]
  fwd100 --> pack

  iface --> cache["cache_utils.py"]
  iface --> cuteutil["cute_dsl_utils.py"]
  softmax --> utils["utils.py"]
  mask --> utils
```

## Responsibility cheat sheet

| Module | Owns |
|--------|------|
| `interface.py` | Public API, arch/tile heuristics, compile + cache, tensor checks |
| `flash_fwd*.py` | Architecture-specific forward kernels |
| `flash_bwd*.py` | Backward (+ preprocess/postprocess helpers) |
| `softmax.py` | Online softmax, score_mod hooks |
| `mask.py` | Causal / local / block-sparse / mask_mod |
| `block_info.py` | n/m block ranges for causal, local, split |
| `pipeline.py` | Multi-stage copy-compute circular buffers |
| `tile_scheduler.py` | Persistent / varlen / CLC-style scheduling |
| `paged_kv.py` | Paged KV cache loads |
| `pack_gqa.py` | Pack Q heads sharing KV |
| `cache_utils.py` | In-memory + disk JIT caches |

## What depends on what when you change code

| If you change… | Re-read / retest |
|----------------|------------------|
| Softmax numerics | All fwd/bwd; combine; score_mod tests |
| Causal block ranges | `block_info.py`, mask tests, varlen |
| Tile sizes in `interface.py` | Perf benches + correctness across hdim |
| Pipeline barriers | Hang risk — [`AI/DEBUG_2CTA.md`](../../AI/DEBUG_2CTA.md) |
| Compile key / cache fingerprint | `tests/cute/test_cache_utils.py` |
