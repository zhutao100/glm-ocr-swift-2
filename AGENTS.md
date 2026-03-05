# Agent Operating Guide (AGENTS.md)

This repository is intended to be safely evolvable by **agentic coding tools** across many sessions.

## Mission

Build a **fully native macOS (Apple Silicon) GLM-OCR app** in Swift.

## Current reality

- The repo builds and tests cleanly (`swift test`).
- `ModelStore` snapshot download + HF cache resolution is implemented.
- End-to-end OCR runs locally (MLX Swift) for:
  - a single image or a PDF (single/multi-page) (CLI + App),
  - optional layout mode (PP-DocLayout-V3 → region OCR → merged Markdown + structured `OCRDocument`).

## Non-negotiables

1. **Docs must stay in sync**
   - Update `docs/architecture.md` whenever module boundaries or public API shape changes.
   - Update `docs/dev_plans/*` checklists as milestones complete (keep them truthful).
   - Add ADRs in `docs/decisions/` for decisions that affect future work (interfaces, caching layout, tokenization/chat template scheme, etc.).

2. **Prefer deterministic, testable primitives**
   - Keep pure functions for preprocessing and prompt/template logic where possible.
   - Add unit tests for: prompt splitting, cache path rules, JSON schema task formatting, and any token budgeting logic.

## Formatting / linting (optional but preferred)

This repo includes configs for local tooling:

- SwiftFormat: `.swiftformat`
- SwiftLint: `.swiftlint.yml`
- pre-commit: `.pre-commit-config.yaml`

Typical commands (if installed):

```bash
swiftformat --config .swiftformat .
swiftlint --config .swiftlint.yml
pre-commit run -a
```

## Coding conventions

- Swift 6 strict concurrency is enabled for all targets.
- Default to `Sendable` value types; use `actor` for mutable shared state.
- Avoid global singletons (except lightweight statics for constants).
- Fail with typed errors (`enum: Error`) rather than `fatalError`, unless the failure is truly unrecoverable.
- When working with MLX,
  - no compound assignment on tensors unless you can prove non-aliasing.
  - prefer out-of-place ops in residual paths (`x = x + y`, not `x += y`) to avoid accidental aliasing drift.

## MLX vs PyTorch (MPS) dtype quirks (parity-critical)

- **MPS defaults to FP16 often**: PyTorch/Transformers commonly runs models + `pixel_values` in `float16` on MPS; parity runs in Swift should force **both weights and inputs** to `.float16`.
- **Mixed-dtype comparisons can differ**: PyTorch MPS may effectively compare FP16 tensors against FP32 scalars (e.g. `eps`) in FP32; if a mask depends on thresholds, match the reference comparison dtype explicitly (often by casting the tensor to FP32 for the compare).
- **Sentinel magnitudes must be dtype-aware**: casting `Float.greatestFiniteMagnitude` to FP16 becomes `inf`; prefer `Float16.greatestFiniteMagnitude` (or an explicit finite FP16 max) for masking to avoid `inf`/`NaN` cascades.
- **FP16 border sensitivity is real**: operations like `grid_sample(align_corners=false)` are very sensitive near `[0, 1]`; FP16 rounding can flip in/out-of-bounds. When writing golden checks, prefer stable probe indices and record any internal selection indices (see `docs/golden_checks.md`).


## References

- `GLM-OCR` model insights: `docs/GLM-OCR_model.md`

## Useful Tools / Resources

- Inspect `.safetensors` structure:
  - `~/bin/stls.py --format toon <file.safetensors>`
  - If missing: `curl https://gist.githubusercontent.com/zhutao100/cc481d2cd248aa8769e1abb3887facc8/raw/89d644c490bcf5386cb81ebcc36c92471f578c60/stls.py > ~/bin/stls.py`
- Model snapshot cache (common location `~/.cache/huggingface/hub/`):
  - `zai-org--GLM-OCR`: `~/.cache/huggingface/hub/models--zai-org--GLM-OCR/snapshots`
  - `PaddlePaddle/PP-DocLayoutV3_safetensors`: `~/.cache/huggingface/hub/models--PaddlePaddle--PP-DocLayoutV3_safetensors`
  - Use shell command `hf cache ls` to list model caches, `hf cache download [model-org]/[model-id]` to download models as needed.
- The official github repo [GLM-OCR](https://github.com/zai-org/GLM-OCR/): accessible at `../GLM-OCR`
- when inspecting the reference implementation in Python, use the virtual env `venv313` by pretending `PYENV_VERSION=venv313 pyenv exec ` to the command.

## Example parity workflow

Generate `examples/result/` and evaluate it against `examples/reference_result/` + `examples/golden_result/`:

```bash
git submodule update --init --recursive tools/example_eval
scripts/run_examples.sh
scripts/eval_examples.sh
```

Reports are written under `.build/example_eval/`.
