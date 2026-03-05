# Architecture

This article describes the Swift port architecture and runtime flow.

## Module map

`GlmOCRSwift` composes several targets:

- `GlmOCRSwift`: public API and orchestration actor (`GlmOCRPipeline`)
- `GlmOCRLayoutMLX`: PP-DocLayoutV3 inference, preprocessing, and postprocessing
- `GlmOCRRecognizerMLX`: GLM-OCR recognizer runtime, tokenizer/config/model loading, generation
- `GlmOCRModelDelivery`: model resolution, download, cache verification, persisted model state
- `GlmOCRPDFium` (macOS): PDFium-backed rasterization
- `CPDFium` / `PDFiumBinary`: PDFium bridge and binary

## End-to-end flow

`GlmOCRPipeline.parse(_:options:)` follows this sequence:

1. Validate `GlmOCRConfig` and `ParseOptions`.
2. Load pages from input (`image`, `imageData`, `pdfData`).
3. Branch on `config.enableLayout`.
4. Recognize content.
5. Format page regions to markdown + structured output.
6. Build diagnostics and return `OCRDocumentResult`.

## Input loading and PDF behavior

`PipelinePageLoader` handles input decoding:

- `.image`: pass-through single page
- `.imageData`: `CGImageSource` decode
- `.pdfData`:
  - macOS: PDFium (`PDFiumDocument` + `PDFiumPageRasterizer`)
  - iOS: CoreGraphics PDF rendering fallback

Page limits:

- `ParseOptions.maxPages` and `GlmOCRConfig.defaultMaxPages` are combined with `min` when both are present.
- `defaultMaxPages` applies only to PDF inputs.

## Model delivery

`SandboxModelManager` resolves recognizer/layout model directories before runtime starts:

- accepts direct local paths
- otherwise fetches from Hugging Face via `swift-huggingface`
- validates required files (`config.json`) and `.safetensors` presence
- persists integrity metadata in `model-delivery-state.json`
- supports custom endpoint via `HF_ENDPOINT`

The bundled manifest pins default model revisions (`model-manifest.json`).

## Layout path (`enableLayout = true`)

The layout backend runs through `PPDocLayoutMLXRunner`:

1. Resize and normalize page to fixed layout tensor input (`[1,3,800,800]`).
2. Run PP-DocLayout model (Swift MLX implementation).
3. Decode logits/boxes/order/masks.
4. Apply thresholding, NMS, containment filtering, box merge rules, and task mapping.
5. Return normalized regions for OCR preprocessing.

Each region is then cropped (`PipelineRegionCropper`) using normalized `bbox2D` and optional polygon masks.

Tasks map to OCR behavior:

- `text`, `table`, `formula`: sent to recognizer
- `skip`, `abandon`: excluded from recognition queue

## No-layout path (`enableLayout = false`)

Each page is recognized as a single region using `prompts.noLayoutPrompt`.

This path skips layout model execution and region cropping.

## Recognition runtime

`GLMRegionRecognizer` uses `GlmOcrRecognizerRuntime`:

1. Load model bundle (`config.json`, processor config, generation config).
2. Load tokenizer (`AutoTokenizer`) and model weights (`.safetensors`).
3. Prepare multimodal input:
  - chat-template prompt
  - smart image resize
  - normalization
  - patchification + image token expansion
4. Run generation with KV cache, optional prefill chunking, and sampling controls:
  - `temperature`
  - `topP`
  - `topK`
  - `repetitionPenalty`

`GlmOCRPipeline` limits parallel recognition with `AsyncLimiter` (`maxConcurrentRecognitions`).

## Formatting and output assembly

`PipelineFormatter` normalizes labels and content, then builds:

- `OCRPageResult` per page (`OCRRegion` entries)
- merged markdown document
- optional `OCRMarkdownBundle` (in-memory) with:
  - rewritten markdown image references (`figures/*.heic`)
  - figure binaries (`OCRFigureAsset.data`)
  - JSON sidecar (`documentJSON`) including pages/diagnostics/figure manifest

Formatting includes:

- heading normalization (`doc_title`, `paragraph_title`)
- formula block normalization and formula-number tag merge
- text cleanup and bullet normalization
- default image placeholders (`![Image <page>-<idx>](page=...,bbox=...)`) that are rewritten when markdown bundle export is enabled

## Diagnostics and warnings

When `ParseOptions.includeDiagnostics == true`, `ParseDiagnostics` includes:

- `warnings`: recoverable crop/recognition issues
- `timingsMs`: stage timings (`page_load`, layout, OCR, total)
- `metadata`: runtime context (prompt hashes, page counts, PDF render debug metadata)

Recognition/crop failures are non-fatal for the full document; failed regions are emitted with empty content and warning entries.

## Cancellation and concurrency model

- `GlmOCRPipeline` is an actor.
- Long-running loops use `Task.checkCancellation()`.
- Recognition work runs in task groups and merges per-region results deterministically by page/region key.

## Debug environment variables

- `GLMOCR_DEBUG_PAGE_RENDER_DUMP`
- `GLMOCR_DEBUG_OCR_PREPROCESS_DUMP`
- `GLMOCR_DEBUG_OCR_PREPROCESS_ONLY`
- `GLMOCR_DEBUG_OCR_POSTPROCESS_INPUT_DUMP`
- `GLMOCR_DEBUG_DUMP_CROPS_DIR`
- `GLMOCR_DEBUG_NATIVE_TRACE`
- `GLMOCR_DEBUG_NATIVE_TOKENS`
- `GLMOCR_DEBUG_PIXEL_TRACE`
- `GLMOCR_DEBUG_INTERPOLATION`
- `GLMOCR_DEBUG_VIMAGE_HQ`
- `GLMOCR_DEBUG_TENSOR_TRACE`
- `GLMOCR_DEBUG_POSITION_TRACE`
- `GLMOCR_MAX_TOKENS`
- `GLMOCR_TEMPERATURE`
- `GLMOCR_PREFILL_STEP_SIZE`
- `GLMOCR_TOP_P`
- `GLMOCR_TOP_K`
- `GLMOCR_REPETITION_PENALTY`
- `HF_ENDPOINT`
