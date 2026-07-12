# 08 — FA2 CUDA Kernels

Production FlashAttention-2 for Ampere and newer (SM80+), implemented in CUDA/CUTLASS-style templates under [`csrc/flash_attn/`](../csrc/flash_attn/).

Paper: [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf).

## Why this stack is structured the way it is

FA2 must ship **many specialized binaries**: head dims (32…256), FP16/BF16, causal vs non-causal, dropout, local windows, SplitKV, etc. C++ **templates + explicit `.cu` instantiations** bake those choices at compile time so the inner loop has no dead branches. That is why the tree looks huge compared to Triton — same algorithm, more specialization knobs.

Layering:

```text
Python API → flash_api.cpp → launch templates → kernel headers → softmax / traits
```

See [diagrams/call-graphs.md](diagrams/call-graphs.md).

## Entry points

| Layer | File | Job |
|-------|------|-----|
| Python | [`flash_attn/flash_attn_interface.py`](../flash_attn/flash_attn_interface.py) | `flash_attn_func`, varlen, KV-cache wrappers; calls `flash_attn_2_cuda` |
| Host C++ | [`csrc/flash_attn/flash_api.cpp`](../csrc/flash_attn/flash_api.cpp) | Fill `Flash_fwd_params`, dtype/headdim switches, launch |
| Params | [`csrc/flash_attn/src/flash.h`](../csrc/flash_attn/src/flash.h) | Pointers, strides, sizes, flags (causal, softcap, …) |

## Traits: tiles and SMEM layouts

[`kernel_traits.h`](../csrc/flash_attn/src/kernel_traits.h) defines `Flash_fwd_kernel_traits`:

- `kBlockM`, `kBlockN`, `kHeadDim`
- Shared memory layouts for Q/K/V/O (often **swizzled** to avoid bank conflicts)
- MMA atom and copy atom types (`cp.async` on SM80)

**Why:** the compiler and tensor-core pipelines need static shapes. Swizzling is a hardware constraint (SMEM bank conflicts), not a math concern.

**Vs naïve:** naïve PyTorch uses generic GEMM + softmax ops with whatever workspace allocator; FA2 picks tiles that maximize SRAM reuse under Ampere limits.

## Launch template

[`flash_fwd_launch_template.h`](../csrc/flash_attn/src/flash_fwd_launch_template.h):

- Grid roughly `(num_m_block, batch, num_heads)` — FA2 parallelism story
- Boolean template switches: even sequence lengths, local attention, ALiBi, softcap, etc.
- SplitKV path + combine when enabled

**Why so many switches:** each removed runtime `if` in the inner loop is saved latency on every tile. Compile-time specialization is an Ampere-era performance tactic.

## Forward kernel: `compute_attn_1rowblock`

[`flash_fwd_kernel.h`](../csrc/flash_attn/src/flash_fwd_kernel.h) is the algorithmic heart:

1. CTA identifies its `(m_block, batch, head)`
2. Load Q tile into SMEM / registers
3. Loop `n` blocks: async copy K/V, MMA QK, **online softmax**, MMA PV
4. Epilogue: normalize and store O, store LSE

**Hardware exploited:**

- Tensor core MMA for QK and PV
- `cp.async` to overlap HBM→SMEM with compute
- Register-held `acc_o` and softmax stats

**Vs naïve:** no HBM `S`/`P`; one fused kernel; fixed tile pipeline.

## Softmax

[`softmax.h`](../csrc/flash_attn/src/softmax.h):

- Thread-local reduce then warp/quad allreduce for row max/sum
- `softmax_rescale_o` applies `exp(m_old - m_new)` to `acc_o`

Same math as Triton; implementation uses CuTe tensors and unrolled loops for ILP.

## Backward

[`flash_bwd_kernel.h`](../csrc/flash_attn/src/flash_bwd_kernel.h) + launch templates:

- Recomputes probabilities using Q, K, V, LSE, `dO`
- Produces `dQ`, `dK`, `dV` (with careful accumulation)
- Pre/postprocess kernels prepare LSE-related quantities and finalize grads

**Why harder than fwd:** multiple outputs, atomics or sequenced reductions on `dK`/`dV`, and still no full `P` stored.

## Instantiations

`csrc/flash_attn/src/flash_fwd_hdim*_*.cu` explicitly instantiate templates per head dim / dtype / causal. **Ignore the bulk of these files** once you understand one — they exist for the build system, not for new logic.

## Reading strategy (half day)

1. Skim `flash_attn_interface.py` for one forward call
2. In `flash_api.cpp`, find `run_mha_fwd` and how headdim is switched
3. Read `kernel_traits.h` for one headdim’s `kBlockM/N`
4. Read `softmax_rescale_o` in `softmax.h`
5. Skim the main loop in `flash_fwd_kernel.h` with Triton still open in another window

## Common FA2 pitfalls

- **Wrong strides / non-contiguous last dim** — API expects head-dim contiguous, 16-byte aligned
- **Varlen `cu_seqlens` off-by-one** — classic silent wrong results
- **Editing one `.cu` instantiation but not the header** — logic belongs in `.h` templates

## Next

[09 — FA3 Hopper](09-fa3-hopper.md)
