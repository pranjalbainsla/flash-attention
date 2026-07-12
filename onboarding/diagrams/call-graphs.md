# Diagram: Call Graphs

Companion to [11-dispatch-shapes-layouts.md](../11-dispatch-shapes-layouts.md).

## FA2 (root `flash_attn`)

```mermaid
flowchart TD
  User[User_code]
  User --> PyAPI["flash_attn.flash_attn_func"]
  PyAPI --> Iface["flash_attn_interface.py"]
  Iface --> Ext["flash_attn_2_cuda extension"]
  Ext --> ApiCpp["csrc/flash_attn/flash_api.cpp"]
  ApiCpp --> Switch["dtype_headdim_causal_SplitKV_switches"]
  Switch --> Launch["flash_fwd_launch_template.h"]
  Launch --> Kernel["flash_fwd_kernel.h compute_attn_1rowblock"]
  Kernel --> Traits["kernel_traits.h tiles_SMEM"]
  Kernel --> Soft["softmax.h"]
```

Backward: `flash_api.cpp` → `flash_bwd_launch_template.h` → `flash_bwd_kernel.h` (+ preprocess/postprocess for dQ/dK/dV).

## FA3 (`hopper/` → `flash_attn_3`)

```mermaid
flowchart TD
  User[User_code]
  User --> PyAPI["flash_attn_3.flash_attn_interface"]
  PyAPI --> Ext["flash_attn_3._C"]
  Ext --> ApiCpp["hopper/flash_api.cpp"]
  ApiCpp --> Arch["ARCH_SWITCH SM90_vs_SM80"]
  Arch --> Fwd90["flash_fwd_kernel_sm90.h"]
  Arch --> Fwd80["flash_fwd_kernel_sm80.h"]
  Fwd90 --> Main90["mainloop_fwd_sm90_tma_gmma_ws.hpp"]
  Main90 --> Soft["hopper/softmax.h"]
```

## FA4 (`flash_attn.cute`)

```mermaid
flowchart TD
  User[User_code]
  User --> PyAPI["flash_attn.cute.flash_attn_func"]
  PyAPI --> Iface["cute/interface.py"]
  Iface --> Arch["_get_device_arch"]
  Arch --> Sm80["FlashAttentionForwardSm80"]
  Arch --> Sm90["FlashAttentionForwardSm90"]
  Arch --> Sm100["FlashAttentionForwardSm100"]
  Arch --> Sm120["FlashAttentionForwardSm120"]
  Sm80 --> Compile["cute.compile + JIT cache"]
  Sm90 --> Compile
  Sm100 --> Compile
  Sm120 --> Compile
  Compile --> Cubin[PTX_CUBIN]
  Cubin --> Launch[Kernel_launch]
  Iface --> Combine["FlashAttentionForwardCombine if SplitKV"]
```

Compile keys include dtype, head dim, causal, tile sizes, mask/score_mod hashes, arch — see [`cache_utils.py`](../../flash_attn/cute/cache_utils.py).

## Triton experimental (learning path only)

```mermaid
flowchart LR
  User --> TritonPy["flash_attn_triton.py"]
  TritonPy --> Fwd["_fwd_kernel @triton.jit"]
  TritonPy --> Bwd["_bwd_* kernels"]
```

Not on the FA2/FA3/FA4 production dispatch path for NVIDIA CUDA installs.
