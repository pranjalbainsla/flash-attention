# 16 — Progressive Reading Order

A concrete 2–4 week path. Adjust pace; keep the **checkpoints**. Skip ROCm, packaging, and model zoo the entire time.

## Week 0 / Day 0 — Orient (2–3 hours)

| Step | Material | Checkpoint |
|------|----------|------------|
| 1 | [README hub](README.md) | Know FA2/FA3/FA4 entry imports |
| 2 | [06-repo-map.md](06-repo-map.md) | List 5 files you will open first; list 3 ignore zones |
| 3 | [`CLAUDE.md`](../CLAUDE.md) architecture section | Name fwd kernels for SM90 and SM100 |
| 4 | [diagrams/memory-hierarchy.md](diagrams/memory-hierarchy.md) | Sketch Q tile + KV loop from memory |

## Days 1–2 — Theory

| Step | Material | Checkpoint |
|------|----------|------------|
| 1 | [01-gpu-primer.md](01-gpu-primer.md) | Explain HBM vs SRAM to a rubber duck |
| 2 | [02-theory-io-and-tiling.md](02-theory-io-and-tiling.md) | Why O(N²) score storage hurts |
| 3 | FA1 paper skim ([arxiv](https://arxiv.org/abs/2205.14135)) — IO + Algorithm 1 | Match paper steps to tiling story |
| 4 | [03-theory-online-softmax.md](03-theory-online-softmax.md) + [online-softmax diagram](diagrams/online-softmax-flow.md) | Hand-simulate 2 tiles of rescale |
| 5 | [04-theory-parallelism.md](04-theory-parallelism.md) + [05-papers-and-history.md](05-papers-and-history.md) | One sentence per generation |

## Days 3–4 — Triton bridge

| Step | Material | Checkpoint |
|------|----------|------------|
| 1 | [07-triton-kernels.md](07-triton-kernels.md) | — |
| 2 | Read [`flash_attn_triton_og.py`](../flash_attn/flash_attn_triton_og.py) end-to-end | Mark loads/stores |
| 3 | Read `_fwd_kernel` in [`flash_attn_triton.py`](../flash_attn/flash_attn_triton.py) | Find `m_i`, `lse_i`, `acc_o_scale` |
| 4 | Mental experiment: change `BLOCK_M`; what HBM traffic changes? | Answer in one paragraph |

## Days 5–7 — FA2 CUDA

| Step | Material | Checkpoint |
|------|----------|------------|
| 1 | [08-fa2-cuda.md](08-fa2-cuda.md) + [call-graphs](diagrams/call-graphs.md) | — |
| 2 | `flash_attn_interface.py` → `flash_api.cpp` skim | Follow one fwd call |
| 3 | `softmax.h` `softmax_rescale_o` | Map to Triton lines |
| 4 | Skim `flash_fwd_kernel.h` main loop | Identify Q load, KV loop, store |
| 5 | FA2 paper parallelism section ([pdf](https://tridao.me/publications/flash2/flash2.pdf)) | Relate to grid `(m, batch, head)` |

## Week 2 — FA3 skim → FA4 deep

| Step | Material | Checkpoint |
|------|----------|------------|
| 1 | FA3 [blog](https://tridao.me/blog/2024/flash3/) + [09-fa3-hopper.md](09-fa3-hopper.md) | Producer vs consumer warps |
| 2 | Skim `mainloop_fwd_sm90_tma_gmma_ws.hpp` for pipeline waits | Don’t memorize types |
| 3 | [10-fa4-cute.md](10-fa4-cute.md) + [dependency-map](diagrams/dependency-map.md) | — |
| 4 | `interface.py`: `_get_device_arch` + `flash_attn_func` | Which class on your GPU? |
| 5 | `softmax.py` + `block_info.py` + `mask.py` | Causal n-range story |
| 6 | Read `flash_fwd_sm90.py` **or** `flash_fwd_sm100.py` (your arch) | Map to FA3 roles |
| 7 | [11-dispatch-shapes-layouts.md](11-dispatch-shapes-layouts.md) | Layout `(B,S,H,D)` checklist |

## Week 3 — Practice on hardware

| Step | Material | Checkpoint |
|------|----------|------------|
| 1 | Dev install FA4 | `pip install -e "flash_attn/cute[dev]"` |
| 2 | [13-testing-infrastructure.md](13-testing-infrastructure.md) | — |
| 3 | `pytest tests/cute/test_flash_attn_fast.py -x` | Green |
| 4 | Optional two-pass compile on larger suite | Cache dir populated |
| 5 | [12-benchmarks-and-perf.md](12-benchmarks-and-perf.md); run one bench script | Record ms + TFLOPS |
| 6 | Read [`AI/DEBUG_2CTA.md`](../AI/DEBUG_2CTA.md) (even if unused yet) | Know hang playbook exists |

## Week 4 — First contribution-sized change

| Step | Material | Checkpoint |
|------|----------|------------|
| 1 | [14-debugging-and-pitfalls.md](14-debugging-and-pitfalls.md) | — |
| 2 | [15-extensions-and-contributing.md](15-extensions-and-contributing.md) | Pick one arc |
| 3 | Implement: e.g. a new edge test, a score_mod example, or a doc fix | PR-ready diff |
| 4 | Re-run fast tests + relevant `-k` filter | Green |
| 5 | Mentoring checklist in ch. 15 | All boxes |

## Condensed weekend path (minimum viable onboarding)

If you only have 2 days:

1. Ch. 01–03 + memory & softmax diagrams  
2. Triton `_fwd_kernel`  
3. FA4 `interface.py` + `softmax.py` + one `flash_fwd_sm*.py` skim  
4. `test_flash_attn_fast.py` + one benchmark  

Then schedule Week 2–4 properly before large kernel edits.

## What you should not do early

- Port or debug ROCm CK
- Edit `csrc/cutlass/` submodule
- Rewrite FA2 instantiations for a FA4 feature
- Land a tile-size change without benches
- Trust racecheck blindly on TMA kernels

## After the guide

- Keep [`AI/`](../AI/) as the debug library
- Re-read FA3/FA4 papers when touching WS or 2CTA
- Use `agent_space/` for experiments; promote winners into `tests/cute/`
