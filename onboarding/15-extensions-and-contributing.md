# 15 — Extensions and Contributing

Where to plug in new behavior, and how to land a change as a new contributor.

## Extension points (FA4-first)

### 1. `score_mod` — transform scores before softmax

Injected at compile time via [`softmax.py`](../flash_attn/cute/softmax.py) (`call_score_mod`). Use for softcap-like transforms, ALiBi-style biases, research attention variants.

- Tests: [`tests/cute/test_score_mod.py`](../tests/cute/test_score_mod.py)
- Fixtures: [`tests/cute/score_mod_definitions.py`](../tests/cute/score_mod_definitions.py)

### 2. `mask_mod` — flexible masking

[`mask.py`](../flash_attn/cute/mask.py) + [`tests/cute/test_mask_mod.py`](../tests/cute/test_mask_mod.py). Prefer structured mods over materializing full masks.

### 3. Block sparsity

Block-level skip of KV tiles — large speed win when masks are structured. See `test_block_sparsity.py` and interface `block_sparse_tensors`.

### 4. SplitKV + combine

Parallelize over KV for long contexts / decode. Kernel: fwd with `num_splits`; merge: [`flash_fwd_combine.py`](../flash_attn/cute/flash_fwd_combine.py).

### 5. Paged KV

Inference caches: [`paged_kv.py`](../flash_attn/cute/paged_kv.py).

### 6. PackGQA

[`pack_gqa.py`](../flash_attn/cute/pack_gqa.py) — keep MMA shapes healthy under GQA.

### 7. Tile heuristics

[`interface.py`](../flash_attn/cute/interface.py) tile pickers — high-impact, needs careful benches across shapes ([`AI/SM90_BLOCK_SIZE_TUNING.md`](../AI/SM90_BLOCK_SIZE_TUNING.md)).

### 8. Tests and docs

Often the best first PR: reproduce a bug with a minimal `tests/cute/` case, or improve onboarding/AI notes.

## Contribution opportunities (good first arcs)

| Arc | Why valuable |
|-----|----------------|
| Edge-case tests (empty seq, odd `S`, GQA+causal) | Prevents regressions |
| score_mod / mask_mod examples + benches | Research usability |
| Tile heuristic sweeps + documented configs | Perf without rewriting MMA |
| Varlen / paged KV bugfixes | Production inference |
| Docs (this folder, FA4 README clarity) | Onboarding leverage |
| CI FakeTensor cache reliability | Developer velocity |

Avoid as a first change: wholesale CUTLASS vendor updates, ROCm ports, or reformatting excluded mega-kernel files without a functional need.

## Suggested PR workflow

1. **Dev install** FA4: `pip install -e "flash_attn/cute[dev]"`
2. Branch from latest `main`
3. Minimal fix + **pytest** (`test_flash_attn_fast.py` then targeted `-k`)
4. If perf-related: before/after numbers on the same GPU, shapes listed
5. Ruff on touched cute files
6. Clear PR description: motivation, algorithm impact, test plan

There is no dedicated `CONTRIBUTING.md`; conventions live in READMEs, pre-commit, and CI. Cite the relevant paper section if your change embodies a paper idea.

## Agent scratch space

Use [`agent_space/`](../agent_space/) (create if needed) for disposable repros and profiles — do not commit large binaries or one-off dumps to product paths.

## Mentoring checklist before you claim “I can contribute”

- [ ] Explain IO bottleneck and online softmax without notes
- [ ] Trace FA4 `flash_attn_func` to the arch-specific fwd class
- [ ] Point to softmax rescale in code
- [ ] Run fast tests + one benchmark
- [ ] Know where 2CTA debug docs live
- [ ] Have a scoped PR idea from the table above

## Next

[16 — Reading order](16-reading-order.md)
