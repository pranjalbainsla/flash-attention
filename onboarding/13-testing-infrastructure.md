# 13 — Testing Infrastructure

Correctness is non-negotiable: FlashAttention must match reference attention within tight tolerances. This chapter maps the pytest layout and the FA4 two-pass compile workflow.

## Suites by generation

### FA4 — [`tests/cute/`](../tests/cute/) (primary)

| File | Coverage |
|------|----------|
| `test_flash_attn.py` | Main fwd/bwd, KV-cache, MLA, SplitKV, edges |
| `test_flash_attn_fast.py` | Smoke subset for rapid iteration |
| `test_flash_attn_varlen.py` | Variable-length packing |
| `test_flash_attn_combine.py` | SplitKV combine |
| `test_mask_mod.py` / `test_score_mod.py` | FlexAttention-style mods |
| `test_block_sparsity.py` | Block-sparse masks |
| `test_clc_fuzz.py` | Tile scheduler stress |
| `test_cache_utils.py` | JIT disk cache behavior |
| `conftest.py` | xdist / GPU logging helpers |

### FA2 — [`tests/test_flash_attn.py`](../tests/test_flash_attn.py)

Packed/unpacked, varlen, KV-cache, ALiBi, rotary; plus `tests/ops/`, `tests/modules/`, `tests/models/` (ignore models initially).

### FA3 — [`hopper/test_flash_attn.py`](../hopper/test_flash_attn.py)

Run with `hopper/` on `PYTHONPATH` after FA3 install.

## Two-pass FA4 testing (compilation dominates)

Documented in [`CLAUDE.md`](../CLAUDE.md) and [`tools/ci/README.md`](../tools/ci/README.md):

```bash
# Pass 1: compile in parallel, no real GPU alloc
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: execute with warm disk cache
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  pytest -x tests/cute/test_flash_attn.py
```

| Env var | Effect |
|---------|--------|
| `FLASH_ATTENTION_FAKE_TENSOR=1` | FakeTensorMode — compile only |
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` | Persist CUBIN under `/tmp/$USER/flash_attention_cute_dsl_cache/` |
| `FLASH_ATTENTION_ARCH` | Override detected arch |
| `CUDA_VISIBLE_DEVICES` | Pick a free GPU if OOM |

CI driver: [`tools/ci/run_fa4_ci.py`](../tools/ci/run_fa4_ci.py).

## What tests compare against

Typically a PyTorch reference (`attention_ref` / FlexAttention for mods). Failures print max diffs — learn the project’s rtol/atol norms from existing asserts before tightening or loosening them.

Parametrization usually covers: dtype (fp16/bf16), head dim, sequence length, causal, MHA/GQA/MQA.

## Fast local loop

```bash
pip install -e "flash_attn/cute[dev]"
pytest tests/cute/test_flash_attn_fast.py -x
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output" -x
```

## Linting

Pre-commit runs **ruff** on `flash_attn/cute/`, excluding huge kernels (`flash_fwd.py`, `flash_fwd_sm100.py`, `interface.py`, …) from autoformat.

```bash
ruff check flash_attn/cute/ --fix
ruff format flash_attn/cute/
```

## Writing a good new test

1. Minimal repro shapes (small `S` first)
2. Compare to reference with the same mask/mod semantics
3. Include an edge: `S` not divisible by tile, `S=0`/`S=1`, GQA, causal boundary
4. Prefer adding to `tests/cute/` over one-off scripts in `agent_space/` for anything that should regress-gate

## Next

[14 — Debugging and pitfalls](14-debugging-and-pitfalls.md)
