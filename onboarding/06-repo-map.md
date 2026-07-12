# 06 — Repo Map

A maintainer’s map of the monorepo: what exists, what matters, what to ignore.

## Generations at a glance

| Gen | Package | Code home | Status for new contributors |
|-----|---------|-----------|------------------------------|
| FA2 | `flash_attn` | [`csrc/flash_attn/`](../csrc/flash_attn/), [`flash_attn/flash_attn_interface.py`](../flash_attn/flash_attn_interface.py) | Stable; read for CUDA patterns |
| FA3 | `flash_attn_3` | [`hopper/`](../hopper/) | Hopper beta; read for TMA/WS |
| FA4 | `flash-attn-4` | [`flash_attn/cute/`](../flash_attn/cute/) | **Active development** |
| Triton exp. | (modules only) | [`flash_attn/flash_attn_triton*.py`](../flash_attn/) | Learning bridge |

## Top-level directories

| Path | Purpose | Onboarding priority |
|------|---------|---------------------|
| `flash_attn/cute/` | FA4 CuTeDSL kernels + API | **Primary** |
| `csrc/flash_attn/` | FA2 CUDA kernels | High (read) |
| `hopper/` | FA3 CUDA/CUTLASS | High (skim) |
| `tests/cute/` | FA4 pytest suite | **Primary** |
| `tests/` (root) | FA2 + ops/models tests | Medium |
| `benchmarks/` | Perf scripts | High |
| `AI/` | Kernel debug notes | High when debugging |
| `assets/` | Plots + `fa4_paper.pdf` | Reference |
| `onboarding/` | This guide | You are here |
| `examples/` | Inference examples | Low |
| `training/` | GPT training demo | Ignore initially |
| `tools/ci/` | FA4 CI two-pass | When running CI locally |
| `third_party/` | AMD aiter submodule | Ignore initially |
| `.github/` | CI / publish | Ignore initially |

## The ~25 most important files

### Theory bridge (Triton)

| # | File | Why |
|---|------|-----|
| 1 | [`flash_attn/flash_attn_triton_og.py`](../flash_attn/flash_attn_triton_og.py) | Shortest fused-attention kernel; teach tiling + softmax |
| 2 | [`flash_attn/flash_attn_triton.py`](../flash_attn/flash_attn_triton.py) | Fuller Triton FA (fwd/bwd, bias); still readable |

### FA2 CUDA

| # | File | Why |
|---|------|-----|
| 3 | [`flash_attn/flash_attn_interface.py`](../flash_attn/flash_attn_interface.py) | FA2 Python API → C++ extension |
| 4 | [`csrc/flash_attn/flash_api.cpp`](../csrc/flash_attn/flash_api.cpp) | Host dispatch, params setup |
| 5 | [`csrc/flash_attn/src/flash.h`](../csrc/flash_attn/src/flash.h) | `Flash_fwd_params` / pointers / strides |
| 6 | [`csrc/flash_attn/src/kernel_traits.h`](../csrc/flash_attn/src/kernel_traits.h) | Tile sizes, SMEM layouts, MMA atoms |
| 7 | [`csrc/flash_attn/src/flash_fwd_launch_template.h`](../csrc/flash_attn/src/flash_fwd_launch_template.h) | Grid launch + compile-time switches |
| 8 | [`csrc/flash_attn/src/flash_fwd_kernel.h`](../csrc/flash_attn/src/flash_fwd_kernel.h) | `compute_attn_1rowblock` — the algorithm |
| 9 | [`csrc/flash_attn/src/softmax.h`](../csrc/flash_attn/src/softmax.h) | Online softmax + O rescale |
| 10 | [`csrc/flash_attn/src/flash_bwd_kernel.h`](../csrc/flash_attn/src/flash_bwd_kernel.h) | Backward structure |

### FA3 Hopper

| # | File | Why |
|---|------|-----|
| 11 | [`hopper/flash_attn_interface.py`](../hopper/flash_attn_interface.py) | FA3 Python API |
| 12 | [`hopper/flash_api.cpp`](../hopper/flash_api.cpp) | Arch / feature dispatch |
| 13 | [`hopper/tile_size.h`](../hopper/tile_size.h) | Tile heuristics |
| 14 | [`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`](../hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp) | TMA + warp-specialized mainloop |
| 15 | [`hopper/softmax.h`](../hopper/softmax.h) | FA3 online softmax |

### FA4 CuTeDSL

| # | File | Why |
|---|------|-----|
| 16 | [`flash_attn/cute/interface.py`](../flash_attn/cute/interface.py) | API, arch select, tiles, compile |
| 17 | [`flash_attn/cute/flash_fwd.py`](../flash_attn/cute/flash_fwd.py) | Sm80 forward |
| 18 | [`flash_attn/cute/flash_fwd_sm90.py`](../flash_attn/cute/flash_fwd_sm90.py) | Hopper forward |
| 19 | [`flash_attn/cute/flash_fwd_sm100.py`](../flash_attn/cute/flash_fwd_sm100.py) | Blackwell forward (richest features) |
| 20 | [`flash_attn/cute/softmax.py`](../flash_attn/cute/softmax.py) | Shared softmax + score_mod |
| 21 | [`flash_attn/cute/block_info.py`](../flash_attn/cute/block_info.py) | Causal/local/split block ranges |
| 22 | [`flash_attn/cute/mask.py`](../flash_attn/cute/mask.py) | Masking + mask_mod |
| 23 | [`flash_attn/cute/pipeline.py`](../flash_attn/cute/pipeline.py) | Copy-compute pipeline state |
| 24 | [`flash_attn/cute/tile_scheduler.py`](../flash_attn/cute/tile_scheduler.py) | Persistent / varlen scheduling |

### Practice

| # | File | Why |
|---|------|-----|
| 25 | [`tests/cute/test_flash_attn.py`](../tests/cute/test_flash_attn.py) | Main correctness suite |
| — | [`benchmarks/benchmark_attn.py`](../benchmarks/benchmark_attn.py) | Cross-impl timing |
| — | [`AI/DEBUG_2CTA.md`](../AI/DEBUG_2CTA.md) | Hang/deadlock playbook |

(25 core + 2 practice companions = the “start here” set.)

## Safely ignore initially

| Area | Paths | Why defer |
|------|-------|-----------|
| ROCm / AMD | `csrc/flash_attn_ck/`, `csrc/composable_kernel/`, `third_party/aiter`, `*triton_amd*` | Separate backend |
| Windows | win32 / MSVC branches in root `setup.py` | Linux-first |
| Packaging / wheels | root & hopper `setup.py` publish details, `.github/workflows/publish*.yml`, `MANIFEST.in` | Distribution, not algorithm |
| Training stack | `training/` | End-to-end demo |
| Model zoo | `flash_attn/models/`, `modules/`, `layers/`, `losses/` | Wrappers around FA2 |
| Side CUDA libs | `csrc/fused_dense_lib/`, `csrc/layer_norm/` | Not attention core |
| Legacy blocksparse | `flash_attn/flash_blocksparse_*` | Old sparse API |
| Vendor dump | wholesale `csrc/cutlass/` | Dependency; don’t study end-to-end |
| Adoption list | `usage.md` | Marketing/history |

## Suggested first-hour tour

1. This file + [README hub](README.md)
2. [`CLAUDE.md`](../CLAUDE.md) architecture section
3. [`flash_attn/cute/interface.py`](../flash_attn/cute/interface.py) — skim `flash_attn_func` and `_get_device_arch`
4. Open [diagrams/call-graphs.md](diagrams/call-graphs.md)

## Next

[07 — Triton kernels](07-triton-kernels.md)
