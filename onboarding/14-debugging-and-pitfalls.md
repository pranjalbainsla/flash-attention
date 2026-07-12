# 14 — Debugging and Pitfalls

Kernel bugs split into three families: **wrong numerics**, **hangs/deadlocks**, and **perf cliffs**. Use different tools for each.

## Wrong numerics (silent)

### Common causes

| Pitfall | Symptom | Where to look |
|---------|---------|---------------|
| Causal / window block range off-by-one | Errors near diagonal | `block_info.py`, mask code, Triton `end_n` |
| Forgetting `-inf` inside partial tiles | Leakage from padded KV | mask / predicate loads |
| LSE combine wrong (SplitKV) | Diffs grow with `num_splits` | `flash_fwd_combine.py` |
| Softmax rescale missing | Exploding / vanishing rows | `softmax.py` / `softmax.h` |
| Wrong GQA head index | Every q-head same/wrong KV | pack_gqa / head broadcast |
| Non-contiguous head dim | Garbage or assert | interface validation |
| FP8 descale | Huge error | descale tensors in interface |

### Strategy

1. Bisect features: dense non-causal → causal → varlen → mods → split
2. Shrink to `B=1, H=1, S=128, D=64` and compare elementwise to reference
3. Dump LSE as well as O — LSE bugs show up clearly
4. For mods, test identity `score_mod` / always-true mask first

## Hangs and deadlocks

Especially on **Hopper WS** and **Blackwell 2CTA**:

- Pipeline stage index / phase mismatch
- Missing barrier arrive/wait
- 2CTA: `tx_count` not multiplied by `cta_group_size`
- Only some CTAs take a code path that waits on a cluster barrier

**Playbook:** [`AI/DEBUG_2CTA.md`](../AI/DEBUG_2CTA.md) — printf bisection with thread guards (`tidx % 32 == 0`, `elect_one()`).

CLC scheduling issues: [`AI/CLC_TRACE_DEBUG.md`](../AI/CLC_TRACE_DEBUG.md) (`FA_LOG_LEVEL`).

## Sanitizers and false positives

`compute-sanitizer --tool=racecheck` is useful but can **false-positive** on raw TMA / `cp.async.bulk` patterns. Read [`AI/RACECHECK_TMA_HAZARD.md`](../AI/RACECHECK_TMA_HAZARD.md) before “fixing” a race that is a known hazard report.

## PTX / SASS inspection

```bash
export CUTE_DSL_KEEP_PTX=1
export CUTE_DSL_LINEINFO=1
# optional CUBIN dump
export CUTE_CUBIN_PATH=/tmp/cubins
```

See [`AI/SASS_MMA_ANALYSIS.md`](../AI/SASS_MMA_ANALYSIS.md) and [`tools/sass_diff.py`](../tools/sass_diff.py).

## Implementation pitfalls by topic

### Online softmax

- Updating `m` but forgetting to rescale `acc_o`
- Mixing log2/exp2 paths with natural log inconsistently (`fast_math`)
- Row reduce incomplete across warps → wrong max

### Tiling / edges

- Sequence not multiple of `BLOCK_M/N` — predicate loads/stores
- Varlen: wrong `cu_seqlens` or tile preprocess ([`AI/VARLEN_PREPROCESS_TILE_BUG.md`](../AI/VARLEN_PREPROCESS_TILE_BUG.md))

### Memory layouts

- Assuming `(B, H, S, D)` when API is `(B, S, H, D)`
- Ignoring stride when only shape “looks right”

### JIT cache

- Stale mental model: cache keys include source fingerprints — but if you bypass cache incorrectly you may run old CUBIN during weird experiments
- First-run timeout in CI — use two-pass FakeTensor

## Debug toolchain summary

| Problem | Tool |
|---------|------|
| Wrong O | Tiny pytest vs reference |
| Hang | Guarded `cute.printf`, 2CTA doc |
| Race reports | compute-sanitizer + TMA hazard doc |
| Slow | Benchmarks ch. 12 + SASS notes |
| Compile fail | Keep PTX, shrink Constexpr configs |

## Next

[15 — Extensions and contributing](15-extensions-and-contributing.md)
