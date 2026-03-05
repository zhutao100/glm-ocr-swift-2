import CoreGraphics
import Foundation
import MLX
import MLXNN
import Tokenizers

public actor GlmOcrRecognizerRuntime {
    public let modelDirectory: URL
    public let modelBundle: GlmOcrModelBundle

    private let tokenizer: any Tokenizer
    private let processor: GlmOcrRecognizerProcessor
    private let model: GlmOcrRecognizerModel
    private let eosTokenIDs: Set<Int>

    private let repetitionContextSize = 20
    private let traceEnabled = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_NATIVE_TRACE"] == "1"
    private let traceAllTokens = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_NATIVE_TOKENS"] == "1"
    private let tracePixels = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIXEL_TRACE"] == "1"

    private static func traceWrite(_ line: String) {
        let payload = "[GlmOcrRecognizerRuntime] \(line)\n"
        let data = payload.data(using: .utf8) ?? Data()
        FileHandle.standardError.write(data)
    }

    private static func imageRGBSummary(_ image: CGImage) -> (count: Int, sum: Float, prefix: [Float])? {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else {
            return nil
        }

        let pixelCount = width * height
        var raw = [UInt8](repeating: 0, count: pixelCount * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard
            let context = CGContext(
                data: &raw,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            )
        else {
            return nil
        }

        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var values = [Float]()
        values.reserveCapacity(pixelCount * 3)
        for index in 0..<pixelCount {
            let base = index * 4
            values.append(Float(raw[base]))
            values.append(Float(raw[base + 1]))
            values.append(Float(raw[base + 2]))
        }

        return (values.count, values.reduce(Float(0), +), Array(values.prefix(8)))
    }

    public init(modelDirectory: URL) async throws {
        if ProcessInfo.processInfo.environment["GLMOCR_DEBUG_NATIVE_TRACE"] == "1" {
            Self.traceWrite("init.start modelDirectory=\(modelDirectory.path)")
        }
        let bundle = try GlmOcrRecognizerConfigLoader.loadBundle(modelDirectory: modelDirectory)

        let tokenizer: any Tokenizer
        do {
            tokenizer = try await AutoTokenizer.from(directory: modelDirectory)
        } catch {
            throw GlmOcrRecognizerMLXError.tokenizerFailed(error.localizedDescription)
        }

        let model: GlmOcrRecognizerModel
        do {
            model = try Self.loadModel(
                modelDirectory: modelDirectory,
                modelConfig: bundle.modelConfig,
                trace: ProcessInfo.processInfo.environment["GLMOCR_DEBUG_NATIVE_TRACE"] == "1"
            )
        } catch let error as GlmOcrRecognizerMLXError {
            throw error
        } catch {
            throw GlmOcrRecognizerMLXError.weightLoadingFailed(error.localizedDescription)
        }

        MLXRandom.seed(0)

        self.modelDirectory = modelDirectory
        self.modelBundle = bundle
        self.tokenizer = tokenizer
        self.processor = GlmOcrRecognizerProcessor(
            tokenizer: tokenizer,
            modelConfig: bundle.modelConfig,
            processorConfig: bundle.processorConfig
        )
        self.model = model

        let configuredEOS =
            bundle.generationConfig.eosTokenIDs.isEmpty
            ? bundle.modelConfig.eosTokenIDs
            : bundle.generationConfig.eosTokenIDs
        self.eosTokenIDs = Set(configuredEOS)
        if ProcessInfo.processInfo.environment["GLMOCR_DEBUG_NATIVE_TRACE"] == "1" {
            Self.traceWrite("init.complete")
        }
    }

    public func prepareInput(prompt: String, image: CGImage) throws -> GlmOcrPreparedInput {
        try processor.prepare(prompt: prompt, image: image)
    }

    public func inputSignature(prompt: String, image: CGImage) throws -> GlmOcrInputSignature {
        try processor.inputSignature(prompt: prompt, image: image)
    }

    public func recognize(
        prompt: String,
        image: CGImage,
        options: GlmOcrGenerationOptions = .fromEnvironment()
    ) async throws -> String {
        let outputs = try await recognizeBatch(
            requests: [
                GlmOcrRecognizerBatchRequest(prompt: prompt, image: image)
            ],
            options: options
        )
        return outputs.first ?? ""
    }

    public func recognizeBatch(
        requests: [GlmOcrRecognizerBatchRequest],
        options: GlmOcrGenerationOptions = .fromEnvironment()
    ) async throws -> [String] {
        try Task.checkCancellation()
        guard !requests.isEmpty else {
            return []
        }

        trace("recognize.batch.start count=\(requests.count)")
        var preparedInputs: [GlmOcrPreparedInput] = []
        preparedInputs.reserveCapacity(requests.count)

        for (index, request) in requests.enumerated() {
            if tracePixels, let source = Self.imageRGBSummary(request.image) {
                trace(
                    "recognize.batch[\(index)].sourceRGB count=\(source.count) sum=\(source.sum) prefix=\(source.prefix)"
                )
            }
            let prepared = try processor.prepare(prompt: request.prompt, image: request.image)
            trace(
                "recognize.batch[\(index)].prepared tokens=\(prepared.inputIDs.count) grid=\(prepared.imageGridTHW.map { "[\($0.t),\($0.h),\($0.w)]" }.joined(separator: ","))"
            )
            if tracePixels {
                let values = prepared.pixelValues.asArray(Float.self)
                let prefix = Array(values.prefix(8))
                let checksum = values.reduce(Float(0), +)
                trace("recognize.batch[\(index)].pixels count=\(values.count) sum=\(checksum) prefix=\(prefix)")
            }
            preparedInputs.append(prepared)
        }

        let generatedTokenIDsBySample = try generateBatch(
            preparedInputs: preparedInputs,
            options: options
        )
        var outputs: [String] = []
        outputs.reserveCapacity(generatedTokenIDsBySample.count)

        for generatedTokenIDs in generatedTokenIDsBySample {
            if traceAllTokens {
                trace("recognize.tokens ids=\(generatedTokenIDs)")
            }
            trace("recognize.generated tokenCount=\(generatedTokenIDs.count)")
            let decoded = tokenizer.decode(tokens: generatedTokenIDs)
            if traceAllTokens {
                trace("recognize.decoded text=\(decoded)")
            }
            outputs.append(decoded.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        return outputs
    }

    private static func loadModel(
        modelDirectory: URL,
        modelConfig: GlmOcrModelConfig,
        trace: Bool
    ) throws -> GlmOcrRecognizerModel {
        if trace {
            traceWrite("loadModel.start")
        }
        let model = GlmOcrRecognizerModel(config: modelConfig)

        var safetensorFiles: [URL] = []
        if let enumerator = FileManager.default.enumerator(
            at: modelDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            for case let url as URL in enumerator {
                if url.pathExtension == "safetensors" {
                    safetensorFiles.append(url)
                }
            }
        }

        guard !safetensorFiles.isEmpty else {
            throw GlmOcrRecognizerMLXError.missingRequiredFile(
                "No .safetensors files found in \(modelDirectory.path)"
            )
        }
        if trace {
            traceWrite("loadModel.shards count=\(safetensorFiles.count)")
        }

        safetensorFiles.sort { $0.path < $1.path }

        var rawWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            if trace {
                traceWrite("loadModel.read shard=\(file.lastPathComponent)")
            }
            let shard = try loadArrays(url: file)
            for (key, value) in shard {
                rawWeights[key] = value
            }
        }
        if trace {
            traceWrite("loadModel.rawWeights count=\(rawWeights.count)")
        }

        let sanitized = model.sanitize(weights: rawWeights)
        if trace {
            traceWrite("loadModel.sanitized count=\(sanitized.count)")
        }
        let parameters = ModuleParameters.unflattened(sanitized)

        do {
            try model.update(parameters: parameters, verify: [.noUnusedKeys, .shapeMismatch])
        } catch {
            throw GlmOcrRecognizerMLXError.weightLoadingFailed(
                "Model parameter update failed: \(error.localizedDescription)"
            )
        }
        if trace {
            traceWrite("loadModel.update complete")
        }

        eval(model)
        if trace {
            traceWrite("loadModel.eval complete")
        }
        return model
    }

    private func generate(
        prepared: GlmOcrPreparedInput,
        options: GlmOcrGenerationOptions
    ) throws -> [Int] {
        try generateBatch(preparedInputs: [prepared], options: options).first ?? []
    }

    private func generateBatch(
        preparedInputs: [GlmOcrPreparedInput],
        options: GlmOcrGenerationOptions
    ) throws -> [[Int]] {
        guard !preparedInputs.isEmpty else {
            return []
        }

        let batchSize = preparedInputs.count
        let maxPromptLength = preparedInputs.map(\.inputIDs.count).max() ?? 0
        let padTokenID = Int32(modelBundle.generationConfig.padTokenID ?? modelBundle.modelConfig.padTokenID)

        var flattenedInputIDs: [Int32] = []
        flattenedInputIDs.reserveCapacity(batchSize * maxPromptLength)
        var flattenedAttentionMask: [Int32] = []
        flattenedAttentionMask.reserveCapacity(batchSize * maxPromptLength)

        for prepared in preparedInputs {
            let paddingCount = maxPromptLength - prepared.inputIDs.count
            if paddingCount > 0 {
                flattenedInputIDs.append(contentsOf: repeatElement(padTokenID, count: paddingCount))
                flattenedAttentionMask.append(contentsOf: repeatElement(Int32(0), count: paddingCount))
            }
            flattenedInputIDs.append(contentsOf: prepared.inputIDs.map(Int32.init))
            flattenedAttentionMask.append(contentsOf: repeatElement(Int32(1), count: prepared.inputIDs.count))
        }

        var inputIDs = MLXArray(flattenedInputIDs).reshaped(batchSize, maxPromptLength).asType(.int32)
        var attentionMask = MLXArray(flattenedAttentionMask).reshaped(batchSize, maxPromptLength).asType(.int32)

        let concatenatedPixelValues = concatenated(preparedInputs.map(\.pixelValues), axis: 0)
        let flattenedGridValues = preparedInputs.flatMap { prepared in
            prepared.imageGridTHW.flatMap { thw in
                [Int32(thw.t), Int32(thw.h), Int32(thw.w)]
            }
        }
        let gridCount = flattenedGridValues.count / 3
        let imageGridTHW: MLXArray
        if gridCount > 0 {
            imageGridTHW = MLXArray(flattenedGridValues).reshaped(gridCount, 3).asType(.int32)
        } else {
            imageGridTHW = MLXArray.zeros([0, 3], dtype: .int32)
        }

        let cache: [GlmOcrKVCache?] = (0..<modelBundle.modelConfig.textConfig.numHiddenLayers).map { _ in
            GlmOcrSimpleKVCache()
        }

        var embeddings = model.getInputEmbeddings(
            inputIDs: inputIDs,
            pixelValues: concatenatedPixelValues,
            attentionMask: attentionMask,
            imageGridTHW: imageGridTHW
        ).inputEmbeddings
        trace("generate.batch.embeddings shape=\(embeddings.shape)")

        if options.prefillStepSize > 0, embeddings.dim(1) > options.prefillStepSize {
            trace("generate.batch.prefill chunking step=\(options.prefillStepSize)")
            while embeddings.dim(1) > 1 {
                let nToProcess = min(options.prefillStepSize, embeddings.dim(1) - 1)
                trace("generate.batch.prefill chunk=\(nToProcess) remaining=\(embeddings.dim(1))")

                let chunkIDs = inputIDs[0..., ..<nToProcess]
                let chunkEmbeddings = embeddings[0..., ..<nToProcess, 0...]
                let chunkAttentionMask = attentionMask[0..., ..<nToProcess]

                let chunkLogits = model.logits(
                    inputIDs: chunkIDs,
                    inputEmbeddings: chunkEmbeddings,
                    attentionMask: chunkAttentionMask,
                    cache: cache,
                    pixelValues: nil,
                    imageGridTHW: nil
                )
                eval(chunkLogits)
                trace("generate.batch.prefill chunk complete")

                embeddings = embeddings[0..., nToProcess..., 0...]
                inputIDs = inputIDs[0..., nToProcess...]
                attentionMask = attentionMask[0..., nToProcess...]

                Memory.clearCache()
            }

            let lastIndex = inputIDs.dim(1) - 1
            inputIDs = inputIDs[0..., lastIndex...]
            attentionMask = attentionMask[0..., lastIndex...]
            trace("generate.batch.prefill done promptReduced=\(inputIDs.shape)")
        }

        var historyTokensBySample = Array(repeating: [Int](), count: batchSize)

        var currentTokens = try generationStepBatch(
            inputIDs: inputIDs,
            inputEmbeddings: embeddings,
            cache: cache,
            attentionMask: attentionMask,
            historyTokensBySample: &historyTokensBySample,
            options: options
        )
        trace("generate.batch.first tokens=\(currentTokens)")

        var generatedBySample = Array(repeating: [Int](), count: batchSize)
        for idx in generatedBySample.indices {
            generatedBySample[idx].reserveCapacity(options.maxTokens)
        }

        var finished = currentTokens.map { eosTokenIDs.contains($0) }

        for index in 0..<options.maxTokens {
            var activeCount = 0
            for sampleIndex in 0..<batchSize {
                if finished[sampleIndex] {
                    continue
                }

                let token = currentTokens[sampleIndex]
                if eosTokenIDs.contains(token) {
                    finished[sampleIndex] = true
                    continue
                }

                generatedBySample[sampleIndex].append(token)
                activeCount += 1
            }

            if finished.allSatisfy({ $0 }) || activeCount == 0 {
                break
            }

            if index % 256 == 0 {
                Memory.clearCache()
            }

            if index == options.maxTokens - 1 {
                break
            }

            let nextInputValues: [Int32] = (0..<batchSize).map { sampleIndex in
                if finished[sampleIndex] {
                    return padTokenID
                }
                return Int32(currentTokens[sampleIndex])
            }
            let nextInput = MLXArray(nextInputValues).reshaped(batchSize, 1).asType(.int32)
            currentTokens = try generationStepBatch(
                inputIDs: nextInput,
                inputEmbeddings: nil,
                cache: cache,
                attentionMask: nil,
                historyTokensBySample: &historyTokensBySample,
                options: options
            )

            if traceAllTokens || index < 4 {
                trace("generate.batch.step index=\(index + 1) tokens=\(currentTokens)")
            }
        }

        return generatedBySample
    }

    private func generationStepBatch(
        inputIDs: MLXArray,
        inputEmbeddings: MLXArray?,
        cache: [GlmOcrKVCache?],
        attentionMask: MLXArray?,
        historyTokensBySample: inout [[Int]],
        options: GlmOcrGenerationOptions
    ) throws -> [Int] {
        let logits3D: MLXArray

        if let inputEmbeddings {
            trace("generationStepBatch.prefillDecode inputShape=\(inputIDs.shape) embedShape=\(inputEmbeddings.shape)")
            logits3D = model.logits(
                inputIDs: inputIDs,
                inputEmbeddings: inputEmbeddings,
                attentionMask: attentionMask,
                cache: cache,
                pixelValues: nil,
                imageGridTHW: nil
            )
        } else {
            trace("generationStepBatch.decodeOnly inputShape=\(inputIDs.shape)")
            logits3D = model.decodeLogits(
                inputIDs: inputIDs,
                attentionMask: attentionMask,
                cache: cache,
                positionIDs: nil
            )
        }

        var logits = logits3D[0..., -1, 0...]
        trace("generationStepBatch.logits shape=\(logits.shape)")

        appendHistoryTokens(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            historyTokensBySample: &historyTokensBySample
        )

        if options.repetitionPenalty != 1 {
            logits = applyRepetitionPenaltyBatch(
                logits: logits,
                penalty: options.repetitionPenalty,
                historyTokensBySample: historyTokensBySample
            )
        }

        let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
        let sampled = sample(logprobs: logprobs, options: options)

        eval(sampled)
        trace("generationStepBatch.sampled")
        return sampled.asType(.int32).asArray(Int32.self).map(Int.init)
    }

    private func appendHistoryTokens(
        inputIDs: MLXArray,
        attentionMask: MLXArray?,
        historyTokensBySample: inout [[Int]]
    ) {
        let batchSize = inputIDs.dim(0)
        let sequenceLength = inputIDs.dim(1)

        if historyTokensBySample.count < batchSize {
            historyTokensBySample.append(
                contentsOf: repeatElement([Int](), count: batchSize - historyTokensBySample.count)
            )
        }

        let tokenValues = inputIDs.asArray(Int32.self).map(Int.init)
        let maskValues = attentionMask?.asArray(Int32.self)

        for batchIndex in 0..<batchSize {
            let rowStart = batchIndex * sequenceLength
            for column in 0..<sequenceLength {
                let offset = rowStart + column
                if let maskValues, maskValues[offset] == 0 {
                    continue
                }
                historyTokensBySample[batchIndex].append(tokenValues[offset])
            }
        }
    }

    private func applyRepetitionPenaltyBatch(
        logits: MLXArray,
        penalty: Float,
        historyTokensBySample: [[Int]]
    ) -> MLXArray {
        guard penalty > 0, !historyTokensBySample.isEmpty else {
            return logits
        }

        let adjusted = logits
        let batchSize = adjusted.dim(0)

        for batchIndex in 0..<batchSize {
            guard batchIndex < historyTokensBySample.count else {
                continue
            }
            let context = Array(historyTokensBySample[batchIndex].suffix(repetitionContextSize))
            guard !context.isEmpty else {
                continue
            }

            let contextArray = MLXArray(context).asType(.int32)
            var selected = adjusted[batchIndex, contextArray]
            selected = `where`(
                selected .< 0,
                selected * penalty,
                selected / penalty
            )
            adjusted[batchIndex, contextArray] = selected
        }

        return adjusted
    }

    private func sample(logprobs: MLXArray, options: GlmOcrGenerationOptions) -> MLXArray {
        if options.temperature == 0 {
            return argMax(logprobs, axis: -1)
        }

        var filtered = logprobs

        if options.topP > 0, options.topP < 1 {
            filtered = applyTopP(filtered, topP: options.topP)
        }

        if options.topK > 0 {
            filtered = applyTopK(filtered, topK: options.topK)
        }

        let scaled = filtered * (1 / options.temperature)
        return MLXRandom.categorical(scaled, axis: -1).asType(.int32)
    }

    private func applyTopP(_ logprobs: MLXArray, topP: Float) -> MLXArray {
        let probs = exp(logprobs)
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)

        var cumulative = sortedProbs.cumsum(axis: -1)
        let inverseIndices = putAlong(
            zeros(like: sortedIndices),
            sortedIndices,
            values: MLXArray(0..<sortedIndices.dim(-1)).asType(sortedIndices.dtype),
            axis: -1
        )
        cumulative = takeAlong(cumulative, inverseIndices, axis: -1)

        let negInf = MLXArray(-Float.infinity, dtype: logprobs.dtype)
        return `where`(cumulative .> (1 - topP), logprobs, negInf)
    }

    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabSize = logprobs.dim(-1)
        guard topK > 0, topK < vocabSize else {
            return logprobs
        }

        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        let negInf = MLXArray(-Float.infinity, dtype: logprobs.dtype)
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
    }

    private func trace(_ message: String) {
        if traceEnabled {
            Self.traceWrite(message)
        }
    }
}
