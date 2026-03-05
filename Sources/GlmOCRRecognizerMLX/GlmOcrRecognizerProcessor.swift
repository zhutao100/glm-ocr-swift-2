import CoreGraphics
import Foundation
import MLX
import Tokenizers

internal struct GlmOcrRecognizerProcessor: Sendable {
    private let tokenizer: any Tokenizer
    private let modelConfig: GlmOcrModelConfig
    private let processorConfig: GlmOcrProcessorConfig
    private let traceEnabled = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIPELINE_TRACE"] == "1"

    internal init(
        tokenizer: any Tokenizer,
        modelConfig: GlmOcrModelConfig,
        processorConfig: GlmOcrProcessorConfig
    ) {
        self.tokenizer = tokenizer
        self.modelConfig = modelConfig
        self.processorConfig = processorConfig
    }

    internal func prepare(prompt: String, image: CGImage) throws -> GlmOcrPreparedInput {
        trace("prepare.start source=\(image.width)x\(image.height) promptChars=\(prompt.count)")
        let templatedTokens = try chatTemplateTokens(prompt: prompt)

        let target = try GlmOcrRecognizerImageProcessor.smartResize(
            t: processorConfig.temporalPatchSize,
            h: image.height,
            w: image.width,
            tFactor: processorConfig.temporalPatchSize,
            hFactor: processorConfig.patchSize * processorConfig.mergeSize,
            wFactor: processorConfig.patchSize * processorConfig.mergeSize,
            minPixels: processorConfig.minPixels,
            maxPixels: processorConfig.maxPixels
        )
        trace("prepare.resize target=\(target.width)x\(target.height)")

        let pixels = try GlmOcrRecognizerImageProcessor.normalizedPixelValues(
            from: image,
            targetWidth: target.width,
            targetHeight: target.height,
            mean: processorConfig.imageMean,
            std: processorConfig.imageStd
        )

        let patchified = try GlmOcrRecognizerImageProcessor.patchify(
            imageTensor: pixels,
            mergeSize: processorConfig.mergeSize,
            patchSize: processorConfig.patchSize,
            temporalPatchSize: processorConfig.temporalPatchSize
        )
        trace(
            "prepare.patchify grid=[\(patchified.gridTHW.t),\(patchified.gridTHW.h),\(patchified.gridTHW.w)] pixelValuesShape=\(pixels.shape)"
        )

        let mergeArea = processorConfig.mergeSize * processorConfig.mergeSize
        let imageTokenCount = max(1, patchified.gridTHW.product / mergeArea)
        let imageTokenID = tokenizer.convertTokenToId(processorConfig.imageToken) ?? modelConfig.imageTokenID

        let inputIDs = try expandImagePlaceholders(
            tokens: templatedTokens,
            imageTokenID: imageTokenID,
            replacementCounts: [imageTokenCount]
        )

        let attentionMask = Array(repeating: 1, count: inputIDs.count)
        trace("prepare.done inputTokens=\(inputIDs.count) imageTokenCount=\(imageTokenCount)")

        return GlmOcrPreparedInput(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            pixelValues: patchified.flattenedPatches,
            imageGridTHW: [patchified.gridTHW],
            imageTokenID: imageTokenID
        )
    }

    internal func inputSignature(prompt: String, image: CGImage) throws -> GlmOcrInputSignature {
        let prepared = try prepare(prompt: prompt, image: image)
        let imageTokenCount = prepared.inputIDs.reduce(into: 0) { partialResult, token in
            if token == prepared.imageTokenID {
                partialResult += 1
            }
        }

        return GlmOcrInputSignature(
            tokenCount: prepared.inputIDs.count,
            imageTokenCount: imageTokenCount,
            imageGridTHW: prepared.imageGridTHW.map { [$0.t, $0.h, $0.w] },
            tokens: prepared.inputIDs
        )
    }

    private func chatTemplateTokens(prompt: String) throws -> [Int] {
        let renderedPrompt = "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>\(prompt)<|assistant|>\n"
        return tokenizer.encode(text: renderedPrompt, addSpecialTokens: false)
    }

    private func expandImagePlaceholders(
        tokens: [Int],
        imageTokenID: Int,
        replacementCounts: [Int]
    ) throws -> [Int] {
        var output: [Int] = []
        output.reserveCapacity(tokens.count + replacementCounts.reduce(0, +))

        var replacementIndex = 0
        for token in tokens {
            if token == imageTokenID, replacementIndex < replacementCounts.count {
                output.append(contentsOf: Array(repeating: imageTokenID, count: replacementCounts[replacementIndex]))
                replacementIndex += 1
            } else {
                output.append(token)
            }
        }

        guard replacementIndex == replacementCounts.count else {
            throw GlmOcrRecognizerMLXError.processingFailed(
                "image placeholder count mismatch: expected \(replacementCounts.count), expanded \(replacementIndex)"
            )
        }

        return output
    }

    private func trace(_ message: String) {
        guard traceEnabled else {
            return
        }
        let payload = "[GlmOcrRecognizerProcessor] \(message)\n"
        let data = payload.data(using: .utf8) ?? Data()
        FileHandle.standardError.write(data)
    }
}
