# 09 — FA3 Hopper Kernels

FlashAttention-3 targets Hopper (H100/H800) with TMA, WGMMA, and warp-specialized pipelines. Code lives entirely under [`hopper/`](../hopper/), installed as `flash_attn_3`.

Paper: [FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf) · [blog](https://tridao.me/blog/2024/flash3/).

## What changed vs FA2 (same math, new schedule)

| FA2 (Ampere) | FA3 (Hopper) |
|--------------|--------------|
| Threads issue `cp.async` | **TMA** hardware copy engine |
| Uniform warps | **Warp specialization** (producer vs consumer) |
| MMA in-CTA | **WGMMA** warp-group MMA |
| FP16/BF16 focus | **FP8** forward paths |
| Single main coding style | CUTLASS 3 collective mainloops + huge instantiation grid |

FA3 also keeps an SM80-style mainloop for non-Hopper arches in the same tree.

## Entry and dispatch

| File | Role |
|------|------|
| [`hopper/flash_attn_interface.py`](../hopper/flash_attn_interface.py) | Python API (varlen, paged KV, etc.) |
| [`hopper/flash_api.cpp`](../hopper/flash_api.cpp) | `ARCH_SWITCH`, headdim/dtype/Split/Paged/PackGQA/FP8 |
| [`hopper/tile_size.h`](../hopper/tile_size.h) | Heuristics e.g. hdim128 causal → `{128,128}` |

**Why heuristics matter:** Hopper SMEM and WGMMA shapes reward different tiles for causal vs dense and for different head dims. Wrong tiles → leaving TFLOPS on the table without failing tests.

## The SM90 forward mainloop

Primary file: [`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`](../hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp)

**Structure (why it looks intimidating):**

1. **Producer warps** issue TMA loads of Q/K/V into multi-stage SMEM; advance pipeline indices
2. **Consumer warps** wait on barriers, run WGMMA for QK, online softmax ([`hopper/softmax.h`](../hopper/softmax.h)), WGMMA for PV
3. Named barriers / mbarriers enforce stage ownership so producers and consumers overlap

**Hardware constraints exploited:**

- TMA throughput and alignment rules
- WGMMA’s large tile shapes (amortize softmax overhead)
- Separate async proxy so math warps are not stuck issuing loads

**Vs naïve:** still no `S` in HBM; additionally, **copy and math overlap in time** — FA2 overlaps via `cp.async`, but Hopper’s TMA + WS is a deeper producer/consumer split.

Kernel shell: [`hopper/flash_fwd_kernel_sm90.h`](../hopper/flash_fwd_kernel_sm90.h) wires collective mainloop + epilogue + tile scheduler.

See [diagrams/kernel-execution.md](diagrams/kernel-execution.md).

## Instantiations and codegen

`hopper/instantiations/*.cu` + [`generate_kernels.py`](../hopper/generate_kernels.py) produce the combinatorial explosion of templates. **Do not hand-edit dozens of instantiations** — change the collective/header and regenerate as the FA3 workflow expects.

## Features you will see in FA3 APIs

- Paged KV cache
- PackGQA
- Softcap
- FP8 forward
- SplitKV variants

Many of these reappear in FA4 with CuTeDSL implementations — reading FA3 once makes FA4’s feature checklist familiar.

## Reading strategy

1. Blog post for intuition (30 min)
2. `tile_size.h` — notice causal vs non-causal differences
3. Skim `mainloop_fwd_sm90_tma_gmma_ws.hpp` searching for softmax and pipeline wait/commit — do not memorize CUTLASS types on day one
4. Compare to FA4 [`flash_fwd_sm90.py`](../flash_attn/cute/flash_fwd_sm90.py): same roles (TMA, stages, softmax), different language

## Pitfalls specific to Hopper WS kernels

- **Barrier / stage mismatch** → hang (not wrong numerics)
- **Assuming FA2 tile sizes** on Hopper → large silent perf cliffs
- **FP8 scaling** — wrong descale → large error norms; check interface scaling tensors

## Relation to FA4

FA4’s Hopper path is the CuTeDSL descendant of this design. For **new** Hopper/Blackwell work, prefer FA4 (`flash_attn/cute/`). Use FA3 when maintaining the `flash_attn_3` package or comparing CUTLASS collectives to CuTeDSL.

## Next

[10 — FA4 CuTeDSL](10-fa4-cute.md)
