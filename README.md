# GLM-OCR Swift

`GlmOCRSwift` is a native Swift port of [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR), built for local on-device OCR with Apple Silicon and MLX.

The package includes a full pipeline:

- Input loading (`CGImage`, image data, PDF data)
- Optional layout detection (`PP-DocLayoutV3` in MLX)
- Region-level recognition (`GLM-OCR` in MLX)
- Markdown/page result formatting
- Model delivery and local cache management

## Requirements

- Swift 6
- iOS 17+ or macOS 14+

## Installation

Add the package:

```swift
dependencies: [
    .package(url: "https://github.com/tansanrao/glm-ocr-swift.git", from: "0.1.0")
]
```

Then add the product to your target:

```swift
target(
    name: "MyApp",
    dependencies: [
        .product(name: "GlmOCRSwift", package: "glm-ocr-swift")
    ]
)
```

## Quick start

```swift
import Foundation
import GlmOCRSwift

let config = GlmOCRConfig()
let pipeline = try await GlmOCRPipeline(config: config)

let imageData = try Data(contentsOf: URL(filePath: "/tmp/page.png"))
let options = ParseOptions()
let result = try await pipeline.parse(.imageData(imageData), options: options)

print(result.markdown)
print(result.diagnostics.timingsMs)
```

`OCRDocumentResult` includes:

- `pages`: structured region results per page
- `markdown`: merged markdown output (if enabled)
- `diagnostics`: warnings, timings, metadata (if enabled)
- `markdownBundle`: optional in-memory bundle (`document.md`, `document.json`, `figures/*.heic`)

## Pipeline behavior

`GlmOCRPipeline.parse` has two execution paths:

- `enableLayout == true`:
  - run layout model
  - crop detected regions
  - recognize by task (`text` / `table` / `formula`)
  - post-format into markdown
  - if `markdownBundle.enabled == true`, build an in-memory markdown bundle and rewrite image markers
- `enableLayout == false`:
  - run whole-page recognition with `noLayoutPrompt`

PDF page limits are resolved as:

- if both `ParseOptions.maxPages` and `GlmOCRConfig.defaultMaxPages` are set, effective cap is `min(a, b)`
- if one is set, that value is used
- if neither is set, all pages are processed

## Configuration defaults

`GlmOCRConfig` default values:

- `recognizerModelID`: `mlx-community/GLM-OCR-bf16`
- `layoutModelID`: `PaddlePaddle/PP-DocLayoutV3_safetensors`
- `maxConcurrentRecognitions`: `1`
- `enableLayout`: `true`
- `markdownBundle.enabled`: `true`
- `markdownBundle.figureFormat`: `heic`
- `markdownBundle.figureNamingScheme`: `pageRegionPadded`
- `markdownBundle.markdownFileName`: `document.md`
- `markdownBundle.jsonFileName`: `document.json`
- `markdownBundle.figuresDirectoryName`: `figures`
- `markdownBundle.heicCompressionQuality`: `0.82`
- `pdfDPI`: `200`
- `pdfMaxRenderedLongSide`: `3500`
- `defaultMaxPages`: `nil`

`maxConcurrentRecognitions` is intentionally capped at `1` to process one page/recognition job at a time by default.

`GlmOCRRecognitionOptions` defaults:

- `maxTokens`: `4096`
- `temperature`: `0.8`
- `prefillStepSize`: `2048`
- `topP`: `0.9`
- `topK`: `50`
- `repetitionPenalty`: `1.1`

`ParseOptions` defaults:

- `includeMarkdown`: `true`
- `includeDiagnostics`: `true`
- `maxPages`: `nil`

## Markdown bundle contract

When all of these are true:

- `GlmOCRConfig.enableLayout == true`
- `GlmOCRConfig.markdownBundle.enabled == true`
- `ParseOptions.includeMarkdown == true`

the pipeline emits `OCRDocumentResult.markdownBundle`:

- `rewrittenMarkdown`: markdown where figure markers point to `figures/*.heic`
- `documentJSON`: JSON sidecar with pages, diagnostics, and figure manifest metadata
- `figures`: HEIC figure binaries + metadata (`relativePath`, `bbox2D`, dimensions, checksum)

The bundle is in-memory so callers can persist it to any filesystem layout.

## Model delivery and caching

On first use, models are resolved by `SandboxModelManager`:

- If `recognizerModelID` / `layoutModelID` is a local directory path, it is used directly.
- Otherwise the manager downloads from Hugging Face (pinned revisions from `model-manifest.json` when available).
- Download/cache root is under Application Support (`GlmOCRSwift/huggingface/hub`).
- Delivery state is persisted at `GlmOCRSwift/ModelDelivery/model-delivery-state.json`.

Optional environment variable:

- `HF_ENDPOINT`: custom Hugging Face endpoint

## Debug environment variables

Pipeline/debug dumps:

- `GLMOCR_DEBUG_PAGE_RENDER_DUMP`: JSON dump path for rendered PDF pages
- `GLMOCR_DEBUG_OCR_PREPROCESS_DUMP`: JSON dump path for crop/preprocess records
- `GLMOCR_DEBUG_OCR_PREPROCESS_ONLY=1`: stop after preprocess/cropping
- `GLMOCR_DEBUG_OCR_POSTPROCESS_INPUT_DUMP`: JSON dump path before formatter
- `GLMOCR_DEBUG_DUMP_CROPS_DIR`: directory to write cropped region PNG files

Recognizer/runtime tracing:

- `GLMOCR_DEBUG_NATIVE_TRACE=1`
- `GLMOCR_DEBUG_NATIVE_TOKENS=1`
- `GLMOCR_DEBUG_PIXEL_TRACE=1`
- `GLMOCR_DEBUG_INTERPOLATION=none|low|medium|high`
- `GLMOCR_DEBUG_VIMAGE_HQ=0|1`
- `GLMOCR_DEBUG_TENSOR_TRACE=1`
- `GLMOCR_DEBUG_POSITION_TRACE=1`

Generation defaults from env (low-level recognizer runtime):

- `GLMOCR_MAX_TOKENS`
- `GLMOCR_TEMPERATURE`
- `GLMOCR_PREFILL_STEP_SIZE`
- `GLMOCR_TOP_P`
- `GLMOCR_TOP_K`
- `GLMOCR_REPETITION_PENALTY`

Attention backend override (advanced/debug):

- `GLMOCR_FORCE_FAST_SDPA=1`: force MLX fast SDPA kernels even when Metal API validation is enabled
- By default, when `MTL_DEBUG_LAYER=1` or `METAL_DEVICE_WRAPPER_TYPE=1`, recognizer attention uses a validation-safe fallback path

## Documentation

- [DocC index](Sources/GlmOCRSwift/GlmOCRSwift.docc/GlmOCRSwift.md)
- [DocC architecture guide](Sources/GlmOCRSwift/GlmOCRSwift.docc/Architecture.md)

## Release

- `0.1.0`: initial public release
