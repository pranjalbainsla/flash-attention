# 12 — Benchmarks and Performance

Goal: measure attention kernels the way maintainers do — **milliseconds and TFLOPS**, with warm caches, correct FLOP formulas, and apples-to-apples comparisons.

## Where the benches live

| Path | Focus |
|------|-------|
| [`benchmarks/benchmark_attn.py`](../benchmarks/benchmark_attn.py) | Cross-impl (FA2/FA4/cuDNN/ref) |
| [`benchmarks/benchmark_flash_attention.py`](../benchmarks/benchmark_flash_attention.py) | Classic FA2 vs others |
| [`benchmarks/bench_sm90.py`](../benchmarks/bench_sm90.py) | FA4 SM90 sweeps |
| [`flash_attn/cute/benchmark.py`](../flash_attn/cute/benchmark.py) | Timer helpers |
| [`flash_attn/cute/bench_utils.py`](../flash_attn/cute/bench_utils.py) | FLOPS, bandwidth, refs |
| [`hopper/benchmark_*.py`](../hopper/) | FA3-specific |
| [`tests/cute/benchmark_mask_mod.py`](../tests/cute/benchmark_mask_mod.py) | Mod / sparsity benches |

## How timing works

Typical pattern:

1. Allocate inputs on GPU
2. **Warmup** launches (JIT compile + clock stabilize)
3. Time `N` iterations with CUDA events or `torch.utils.benchmark.Timer`
4. Report mean ms and derived TFLOPS

Never trust the first launch after a code change — FA4 may compile for seconds.

## FLOP counting (attention)

Rough forward FLOPs for standard attention (one commonly used formula):

```text
FLOPs_fwd ≈ 4 * B * S * S * H * D
```

(two matmuls QKᵀ and PV, each ~`2 B S S H D`). **Causal** often counted at ~half. Backward is a larger multiple (commonly ~2.5× forward in FA reporting — check the specific bench’s helper).

TFLOPS:

```text
TFLOPS = FLOPs / time_seconds / 1e12
```

If two implementations disagree on the FLOP formula, their TFLOPS are not comparable even when ms are.

## What “good” looks like

- Compare against **cuDNN / PyTorch SDPA / FA2 / FA3** on the **same GPU**
- Sweep sequence lengths: short `S` may be launch-bound; long `S` shows IO/algorithm wins
- Sweep head dims (64, 128, …) — tile heuristics differ
- Causal vs non-causal separately
- For FA4: note arch (SM90 vs SM100) — never compare H100 numbers to B200 as “regressions”

Assets in [`assets/`](../assets/) show published FA2/FA3 speedup plots for calibration.

## Perf analysis checklist

When a change is “slow”:

1. **Compile time vs run time** — disk cache enabled?
2. **Tile sizes** — compare to `tile_size.h` / `interface.py` heuristics for that hdim
3. **Occupancy / SMEM** — too-large tiles
4. **Extra features** — score_mod, mask_mod, softcap, dropout paths
5. **SplitKV** — helps or hurts depending on `S` and machine
6. **GQA packing** — PackGQA on/off
7. **PTX/SASS** — `CUTE_DSL_KEEP_PTX=1`, [`AI/SASS_MMA_ANALYSIS.md`](../AI/SASS_MMA_ANALYSIS.md)

## Bandwidth vs compute bound

Estimate bytes moved (Q+K+V+O roughly `2*B*S*H*D * sizeof` × factors for rereads). If achieved GB/s ≈ HBM peak, you are memory-bound; optimize IO/tiling. If GB/s is low but TFLOPS ≪ peak MMA, look for pipeline bubbles, poor occupancy, or scalar overhead in the inner loop (FA2/FA3 paper themes).

## Practical commands

```bash
# Example: cross-attn bench (adjust args per script --help)
python benchmarks/benchmark_attn.py

# FA4 SM90-oriented
python benchmarks/bench_sm90.py

# Keep GPU free
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
```

## Interpreting regressions

| Symptom | Likely cause |
|---------|----------------|
| ms↑ only on first run | Compile / cold cache |
| ms↑ all runs after small kernel edit | Real hot-path regression |
| Wrong numerics + “fast” | Masking skipped / wrong causal range |
| Only varlen slow | Scheduler / preprocess path |

## Next

[13 — Testing infrastructure](13-testing-infrastructure.md)
