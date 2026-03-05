import Foundation
import MLX
import MLXNN

internal struct GlmOcrModelEmbeddings {
    let inputEmbeddings: MLXArray
}

internal final class GlmOcrRecognizerModel: Module {
    @ModuleInfo(key: "vision_tower") private var visionTower: GlmOcrVisionModel
    @ModuleInfo(key: "language_model") private var languageModel: GlmOcrLanguageModel

    private let config: GlmOcrModelConfig
    private let debugTensorTrace = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_TENSOR_TRACE"] == "1"

    internal init(config: GlmOcrModelConfig) {
        self.config = config
        self._visionTower.wrappedValue = GlmOcrVisionModel(config: config.visionConfig)
        self._languageModel.wrappedValue = GlmOcrLanguageModel(
            config: config.textConfig,
            modelConfig: config,
            vocabularySize: config.vocabSize
        )
    }

    internal func getInputEmbeddings(
        inputIDs: MLXArray,
        pixelValues: MLXArray?,
        attentionMask: MLXArray?,
        imageGridTHW: MLXArray?
    ) -> GlmOcrModelEmbeddings {
        guard let pixelValues else {
            languageModel.resetPositionState()
            return GlmOcrModelEmbeddings(inputEmbeddings: languageModel.embedTokens(inputIDs))
        }

        let visionDType = visionTower.patchEmbedWeightDType
        let visualHiddenStates = visionTower(
            pixelValues.asType(visionDType),
            gridTHW: imageGridTHW ?? MLXArray.zeros([0, 3], dtype: .int32)
        )
        if debugTensorTrace {
            let values = visualHiddenStates.asArray(Float.self)
            let prefix = Array(values.prefix(8))
            let payload =
                "[GlmOcrRecognizerModel] visualHiddenStates shape=\(visualHiddenStates.shape) count=\(values.count) sum=\(values.reduce(Float(0), +)) prefix=\(prefix)\n"
            let data = payload.data(using: .utf8) ?? Data()
            FileHandle.standardError.write(data)
        }

        let textEmbeddings = languageModel.embedTokens(inputIDs)
        if debugTensorTrace {
            let values = textEmbeddings.asArray(Float.self)
            let prefix = Array(values.prefix(8))
            let payload =
                "[GlmOcrRecognizerModel] textEmbeddings shape=\(textEmbeddings.shape) count=\(values.count) sum=\(values.reduce(Float(0), +)) prefix=\(prefix)\n"
            let data = payload.data(using: .utf8) ?? Data()
            FileHandle.standardError.write(data)
        }
        let merged = GlmOcrRecognizerModel.mergeInputIDsWithImageFeatures(
            imageTokenID: config.imageTokenID,
            videoTokenID: config.videoTokenID,
            imageFeatures: visualHiddenStates,
            inputEmbeddings: textEmbeddings,
            inputIDs: inputIDs
        )

        if let imageGridTHW {
            let (positionIDs, ropeDeltas) = languageModel.getRopeIndex(
                inputIDs: inputIDs,
                imageGridTHW: imageGridTHW,
                videoGridTHW: nil,
                attentionMask: attentionMask
            )
            languageModel.cachePositionState(positionIDs: positionIDs, ropeDeltas: ropeDeltas)
        }

        return GlmOcrModelEmbeddings(inputEmbeddings: merged)
    }

    internal func logits(
        inputIDs: MLXArray,
        inputEmbeddings: MLXArray,
        attentionMask: MLXArray?,
        cache: [GlmOcrKVCache?],
        pixelValues: MLXArray?,
        imageGridTHW: MLXArray?
    ) -> MLXArray {
        languageModel(
            inputIDs,
            inputEmbeddings: inputEmbeddings,
            attentionMask: attentionMask,
            cache: cache,
            pixelValues: pixelValues,
            imageGridTHW: imageGridTHW,
            videoGridTHW: nil,
            positionIDs: nil
        )
    }

    internal func decodeLogits(
        inputIDs: MLXArray,
        attentionMask: MLXArray?,
        cache: [GlmOcrKVCache?],
        positionIDs: MLXArray? = nil
    ) -> MLXArray {
        languageModel(
            inputIDs,
            inputEmbeddings: nil,
            attentionMask: attentionMask,
            cache: cache,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil,
            positionIDs: positionIDs
        )
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        func transformKey(_ key: String) -> String {
            var key = key
            if key.contains("visual") {
                if !key.contains("vision_tower") {
                    key = key.replacingOccurrences(of: "model.", with: "")
                    key = key.replacingOccurrences(of: "visual", with: "vision_tower")
                }
            }
            if key.contains("model.language_model") {
                key = key.replacingOccurrences(of: "model.language_model", with: "language_model.model")
            }
            if key.contains("lm_head"), !key.hasPrefix("language_model") {
                key = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }
            return key
        }

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            let newKey = transformKey(key)
            if newKey.contains("layers.16") {
                continue
            }
            sanitized[newKey] = value
        }

        return visionTower.sanitize(weights: sanitized)
    }

    private static func mergeInputIDsWithImageFeatures(
        imageTokenID: Int,
        videoTokenID: Int,
        imageFeatures: MLXArray,
        inputEmbeddings: MLXArray,
        inputIDs: MLXArray
    ) -> MLXArray {
        var imagePositions = inputIDs .== MLXArray(imageTokenID)
        if imagePositions.sum().item(Int.self) == 0 {
            imagePositions = inputIDs .== MLXArray(videoTokenID)
        }

        let batchSize = inputIDs.dim(0)
        var featureStartIndex = 0
        var batchOutputs: [MLXArray] = []
        batchOutputs.reserveCapacity(batchSize)

        for batchIndex in 0..<batchSize {
            let imageMask = imagePositions[batchIndex, 0...]
            let numPositions = imageMask.asType(.int32).sum().item(Int.self)

            if numPositions > 0 {
                let batchFeatures = imageFeatures[featureStartIndex..<(featureStartIndex + numPositions), 0...]
                let cumsum = imageMask.asType(.int32).cumsum(axis: -1)
                let featureIndices = `where`(imageMask, cumsum - MLXArray(1, dtype: .int32), zeros(like: cumsum))
                let gatheredFeatures = batchFeatures[featureIndices]
                let imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
                let batchOutput = `where`(imageMaskExpanded, gatheredFeatures, inputEmbeddings[batchIndex, 0..., 0...])
                batchOutputs.append(batchOutput)
                featureStartIndex += numPositions
            } else {
                batchOutputs.append(inputEmbeddings[batchIndex, 0..., 0...])
            }
        }

        return stacked(batchOutputs, axis: 0)
    }
}

private func glmOcrCheckArrayShape(_ array: MLXArray) -> Bool {
    switch array.ndim {
    case 4:
        let outChannels = array.dim(0)
        let kernelH = array.dim(1)
        let kernelW = array.dim(2)
        return outChannels >= kernelH && outChannels >= kernelW && kernelH == kernelW
    case 5:
        let outChannels = array.dim(0)
        let kernelH = array.dim(2)
        let kernelW = array.dim(3)
        return outChannels >= kernelH && outChannels >= kernelW && kernelH == kernelW
    default:
        return false
    }
}

private func glmOcrRotateHalfVision(_ x: MLXArray) -> MLXArray {
    let split = x.dim(-1) / 2
    let left = x[.ellipsis, ..<split]
    let right = x[.ellipsis, split...]
    return concatenated([-right, left], axis: -1)
}

private func glmOcrApplyVisionRotary(
    q: MLXArray,
    k: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> (MLXArray, MLXArray) {
    let qDType = q.dtype
    let kDType = k.dtype

    var q = q.asType(.float32)
    var k = k.asType(.float32)
    let cos = expandedDimensions(cos, axis: -2).asType(.float32)
    let sin = expandedDimensions(sin, axis: -2).asType(.float32)

    q = (q * cos) + (glmOcrRotateHalfVision(q) * sin)
    k = (k * cos) + (glmOcrRotateHalfVision(k) * sin)

    return (q.asType(qDType), k.asType(kDType))
}

private final class GlmOcrVisionRotaryEmbedding: Module {
    private let dimensions: Int
    private let theta: Float

    init(dimensions: Int, theta: Float = 10_000) {
        self.dimensions = dimensions
        self.theta = theta
    }

    func callAsFunction(sequenceLength: Int) -> MLXArray {
        let invFreq =
            1.0
            / pow(
                MLXArray(theta),
                MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / Float(dimensions)
            )
        let seq = MLXArray(0..<sequenceLength).asType(invFreq.dtype)
        return outer(seq, invFreq)
    }
}

private final class GlmOcrVisionPatchEmbed: Module, UnaryLayer {
    @ModuleInfo(key: "proj") var proj: Conv3d

    private let patchSize: Int
    private let temporalPatchSize: Int
    private let inChannels: Int
    private let embedDim: Int

    init(config: GlmOcrModelConfig.VisionConfig) {
        self.patchSize = config.patchSize
        self.temporalPatchSize = config.temporalPatchSize
        self.inChannels = config.inChannels
        self.embedDim = config.hiddenSize

        let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
        self._proj.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: kernel,
            stride: kernel,
            bias: true
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hiddenStates =
            hiddenStates
            .reshaped(-1, inChannels, temporalPatchSize, patchSize, patchSize)
            .movedAxis(source: 1, destination: 4)

        hiddenStates = proj(hiddenStates)
        return hiddenStates.reshaped(-1, embedDim)
    }
}

private final class GlmOcrVisionPatchMerger: Module, UnaryLayer {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "post_projection_norm") var postProjectionNorm: LayerNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dim: Int, contextDim: Int) {
        self._proj.wrappedValue = Linear(dim, dim, bias: false)
        self._postProjectionNorm.wrappedValue = LayerNorm(dimensions: dim)
        self._gateProj.wrappedValue = Linear(dim, contextDim, bias: false)
        self._upProj.wrappedValue = Linear(dim, contextDim, bias: false)
        self._downProj.wrappedValue = Linear(contextDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = proj(x)
        let normalized = postProjectionNorm(projected)
        let activated = gelu(normalized)
        return downProj(silu(gateProj(activated)) * upProj(activated))
    }
}

private final class GlmOcrVisionAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let numHeads: Int
    private let headDim: Int
    private let scale: Float

    init(config: GlmOcrModelConfig.VisionConfig) {
        self.numHeads = config.numHeads
        self.headDim = config.hiddenSize / config.numHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qkv.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize * 3,
            bias: config.attentionBias
        )
        self._proj.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize,
            bias: config.attentionBias
        )

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        cuSeqlens: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let sequenceLength = hiddenStates.dim(0)

        var qkvStates = qkv(hiddenStates)
        qkvStates = qkvStates.reshaped(sequenceLength, 3, numHeads, -1)
        qkvStates = qkvStates.transposed(1, 0, 2, 3)

        let qkvParts = split(qkvStates, parts: 3, axis: 0)
        var q = qkvParts[0].squeezed(axis: 0)
        var k = qkvParts[1].squeezed(axis: 0)
        var v = qkvParts[2].squeezed(axis: 0)

        q = qNorm(q)
        k = kNorm(k)

        (q, k) = glmOcrApplyVisionRotary(
            q: q,
            k: k,
            cos: positionEmbeddings.cos,
            sin: positionEmbeddings.sin
        )

        q = q.transposed(1, 0, 2)[.newAxis, 0..., 0..., 0...]
        k = k.transposed(1, 0, 2)[.newAxis, 0..., 0..., 0...]
        v = v.transposed(1, 0, 2)[.newAxis, 0..., 0..., 0...]

        let cuValues = cuSeqlens.asArray(Int32.self).map(Int.init)
        var lengths: [Int] = []
        lengths.reserveCapacity(max(cuValues.count - 1, 0))
        if cuValues.count > 1 {
            for index in 0..<(cuValues.count - 1) {
                lengths.append(cuValues[index + 1] - cuValues[index])
            }
        }

        var splitIndices: [Int] = []
        splitIndices.reserveCapacity(max(lengths.count - 1, 0))
        var running = 0
        for length in lengths.dropLast() {
            running += length
            splitIndices.append(running)
        }

        let qSplits = splitIndices.isEmpty ? [q] : split(q, indices: splitIndices, axis: 2)
        let kSplits = splitIndices.isEmpty ? [k] : split(k, indices: splitIndices, axis: 2)
        let vSplits = splitIndices.isEmpty ? [v] : split(v, indices: splitIndices, axis: 2)

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(qSplits.count)

        for index in 0..<qSplits.count {
            let attended = glmOcrScaledDotProductAttention(
                queries: qSplits[index],
                keys: kSplits[index],
                values: vSplits[index],
                scale: scale,
                mask: .none
            )
            outputs.append(attended)
        }

        let combined = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
        let flattened = combined.transposed(0, 2, 1, 3).reshaped(sequenceLength, -1)
        return proj(flattened)
    }
}

private final class GlmOcrVisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: GlmOcrModelConfig.VisionConfig) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.attentionBias
        )
        self._upProj.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.attentionBias
        )
        self._downProj.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.attentionBias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class GlmOcrVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "attn") var attention: GlmOcrVisionAttention
    @ModuleInfo(key: "mlp") var mlp: GlmOcrVisionMLP

    init(config: GlmOcrModelConfig.VisionConfig) {
        self._norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attention.wrappedValue = GlmOcrVisionAttention(config: config)
        self._mlp.wrappedValue = GlmOcrVisionMLP(config: config)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        cuSeqlens: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        var hiddenStates =
            hiddenStates
            + attention(
                norm1(hiddenStates),
                cuSeqlens: cuSeqlens,
                positionEmbeddings: positionEmbeddings
            )
        hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
        return hiddenStates
    }
}

private final class GlmOcrVisionModel: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: GlmOcrVisionPatchEmbed
    @ModuleInfo(key: "rotary_pos_emb") var rotaryPositionEmbedding: GlmOcrVisionRotaryEmbedding
    @ModuleInfo(key: "blocks") var blocks: [GlmOcrVisionBlock]
    @ModuleInfo(key: "merger") var merger: GlmOcrVisionPatchMerger
    @ModuleInfo(key: "downsample") var downsample: Conv2d
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: RMSNorm

    private let config: GlmOcrModelConfig.VisionConfig
    private let spatialMergeSize: Int

    init(config: GlmOcrModelConfig.VisionConfig) {
        self.config = config
        self.spatialMergeSize = config.spatialMergeSize

        self._patchEmbed.wrappedValue = GlmOcrVisionPatchEmbed(config: config)
        self._rotaryPositionEmbedding.wrappedValue = GlmOcrVisionRotaryEmbedding(
            dimensions: (config.hiddenSize / config.numHeads) / 2,
            theta: 10_000
        )
        self._blocks.wrappedValue = (0..<config.depth).map { _ in
            GlmOcrVisionBlock(config: config)
        }
        self._merger.wrappedValue = GlmOcrVisionPatchMerger(
            dim: config.outHiddenSize,
            contextDim: config.outHiddenSize * config.inChannels
        )
        self._downsample.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize,
            outputChannels: config.outHiddenSize,
            kernelSize: IntOrPair(config.spatialMergeSize),
            stride: IntOrPair(config.spatialMergeSize),
            bias: true
        )
        self._postLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    var patchEmbedWeightDType: DType {
        patchEmbed.proj.weight.dtype
    }

    private func rotaryPositionEmbedding(gridTHW: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        var positionIDs: [MLXArray] = []

        for row in gridTHW.asArray(Int32.self).chunked(into: 3) {
            guard row.count == 3 else { continue }
            let t = Int(row[0])
            let h = Int(row[1])
            let w = Int(row[2])

            var hPos = expandedDimensions(MLXArray(0..<h), axis: 1)
            hPos = repeated(hPos, count: w, axis: 1)
            hPos =
                hPos
                .reshaped(h / spatialMergeSize, spatialMergeSize, w / spatialMergeSize, spatialMergeSize)
                .transposed(0, 2, 1, 3)
                .flattened()

            var wPos = expandedDimensions(MLXArray(0..<w), axis: 0)
            wPos = repeated(wPos, count: h, axis: 0)
            wPos =
                wPos
                .reshaped(h / spatialMergeSize, spatialMergeSize, w / spatialMergeSize, spatialMergeSize)
                .transposed(0, 2, 1, 3)
                .flattened()

            let stackedPos = stacked([hPos, wPos], axis: -1)
            positionIDs.append(tiled(stackedPos, repetitions: [t, 1]))
        }

        let indices = positionIDs.count == 1 ? positionIDs[0] : concatenated(positionIDs, axis: 0)

        let gridArray = gridTHW.asArray(Int32.self).map(Int.init)
        var maxGridSize = 0
        var idx = 0
        while idx + 2 < gridArray.count {
            maxGridSize = max(maxGridSize, max(gridArray[idx + 1], gridArray[idx + 2]))
            idx += 3
        }

        let fullRotary = rotaryPositionEmbedding(sequenceLength: max(maxGridSize, 1))
        let rotary = fullRotary[indices].reshaped(indices.dim(0), -1)
        let emb = concatenated([rotary, rotary], axis: -1)
        return (cos(emb), sin(emb))
    }

    private func cumulativeSequenceLengths(gridTHW: MLXArray) -> MLXArray {
        let values = gridTHW.asArray(Int32.self).map(Int.init)
        var repeatedValues: [Int32] = []
        repeatedValues.reserveCapacity(values.count)

        var index = 0
        while index + 2 < values.count {
            let t = values[index]
            let h = values[index + 1]
            let w = values[index + 2]
            let seqLen = h * w
            for _ in 0..<t {
                repeatedValues.append(Int32(seqLen))
            }
            index += 3
        }

        var cu = MLXArray(repeatedValues).cumsum(axis: 0)
        cu = padded(cu, widths: [1, 0], value: MLXArray(0, dtype: cu.dtype))
        return cu
    }

    func callAsFunction(_ hiddenStates: MLXArray, gridTHW: MLXArray) -> MLXArray {
        var hiddenStates = patchEmbed(hiddenStates)
        let positionEmbeddings = rotaryPositionEmbedding(gridTHW: gridTHW)
        let cuSeqlens = cumulativeSequenceLengths(gridTHW: gridTHW)

        for block in blocks {
            hiddenStates = block(
                hiddenStates,
                cuSeqlens: cuSeqlens,
                positionEmbeddings: positionEmbeddings
            )
        }

        hiddenStates = postLayerNorm(hiddenStates)
        hiddenStates = hiddenStates.reshaped(
            -1,
            spatialMergeSize,
            spatialMergeSize,
            hiddenStates.dim(-1)
        )
        hiddenStates = downsample(hiddenStates).reshaped(-1, config.outHiddenSize)
        return merger(hiddenStates)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            if key.contains("position_ids") {
                continue
            }

            if key.contains("patch_embed.proj.weight") || key.contains("downsample.weight") {
                if glmOcrCheckArrayShape(value) {
                    sanitized[key] = value
                } else if value.ndim == 5 {
                    sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                } else if value.ndim == 4 {
                    sanitized[key] = value.transposed(0, 2, 3, 1)
                } else {
                    sanitized[key] = value
                }
                continue
            }

            sanitized[key] = value
        }

        return sanitized
    }
}

private final class GlmOcrRotaryEmbedding: Module {
    private let invFreq: MLXArray
    private let attentionScaling: Float
    private let mropeSection: [Int]

    init(config: GlmOcrModelConfig.TextConfig) {
        let base = config.ropeParameters.ropeTheta
        let dim = max(2, Int(Float(config.headDim) * config.ropeParameters.partialRotaryFactor))

        self.attentionScaling = 1.0
        self.mropeSection = config.ropeParameters.mropeSection

        let invFreq =
            1.0
            / pow(
                MLXArray(base),
                MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim)
            )
        self.invFreq = invFreq.asType(.float32)
    }

    private func applyMRope(_ freqs: MLXArray) -> MLXArray {
        guard !mropeSection.isEmpty else {
            return freqs
        }

        var splitIndices: [Int] = []
        splitIndices.reserveCapacity(max(mropeSection.count - 1, 0))
        var running = 0
        for section in mropeSection.dropLast() {
            running += section
            splitIndices.append(running)
        }

        let chunks = splitIndices.isEmpty ? [freqs] : split(freqs, indices: splitIndices, axis: -1)
        var selected: [MLXArray] = []
        selected.reserveCapacity(chunks.count)

        for (index, chunk) in chunks.enumerated() {
            selected.append(chunk[index % 3, 0..., 0..., 0...])
        }

        return selected.count == 1 ? selected[0] : concatenated(selected, axis: -1)
    }

    func callAsFunction(_ x: MLXArray, positionIDs: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        var invFreqExpanded = invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32)
        invFreqExpanded = broadcast(invFreqExpanded, to: [3, positionIDs.dim(1), invFreq.dim(0), 1])

        let positionExpanded = positionIDs[0..., 0..., .newAxis, 0...].asType(.float32)
        var freqs = matmul(invFreqExpanded, positionExpanded).transposed(0, 1, 3, 2)
        freqs = applyMRope(freqs)

        let emb = concatenated([freqs, freqs], axis: -1)
        let cosValues = cos(emb) * attentionScaling
        let sinValues = sin(emb) * attentionScaling
        return (cosValues.asType(x.dtype), sinValues.asType(x.dtype))
    }
}

private func glmOcrRotateHalfLLM(_ x: MLXArray) -> MLXArray {
    let lastDim = x.dim(-1)
    guard lastDim % 2 == 0 else {
        return x
    }

    let leading = x.size / lastDim
    var paired = x.reshaped(leading, lastDim / 2, 2)
    let even = paired[0..., 0..., 0]
    let odd = paired[0..., 0..., 1]
    paired = stacked([-odd, even], axis: -1).reshaped(leading, lastDim)
    return paired.reshaped(x.shape)
}

private func glmOcrRepeatInterleave(_ x: MLXArray, repeats: Int, axis: Int) -> MLXArray {
    repeated(x, count: repeats, axis: axis)
}

private func glmOcrApplyLanguageRotary(
    q: MLXArray,
    k: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> (MLXArray, MLXArray) {
    var cosValues = expandedDimensions(cos, axis: 1)
    var sinValues = expandedDimensions(sin, axis: 1)

    let half = max(cosValues.dim(-1) / 2, 1)
    cosValues = glmOcrRepeatInterleave(cosValues[0..., 0..., 0..., ..<half], repeats: 2, axis: -1)
    sinValues = glmOcrRepeatInterleave(sinValues[0..., 0..., 0..., ..<half], repeats: 2, axis: -1)

    let rotaryDim = cosValues.dim(-1)
    let qParts = split(q, indices: [rotaryDim], axis: -1)
    let kParts = split(k, indices: [rotaryDim], axis: -1)

    var qRot = (qParts[0] * cosValues) + (glmOcrRotateHalfLLM(qParts[0]) * sinValues)
    var kRot = (kParts[0] * cosValues) + (glmOcrRotateHalfLLM(kParts[0]) * sinValues)

    if qParts.count > 1 {
        qRot = concatenated([qRot, qParts[1]], axis: -1)
    }
    if kParts.count > 1 {
        kRot = concatenated([kRot, kParts[1]], axis: -1)
    }

    return (qRot, kRot)
}

private final class GlmOcrAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    private let numHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let scale: Float

    init(config: GlmOcrModelConfig.TextConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numAttentionHeads * config.headDim,
            bias: config.attentionBias
        )
        self._kProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: config.attentionBias
        )
        self._vProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: config.attentionBias
        )
        self._oProj.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: GlmOcrKVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(batch, seqLen, numKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(batch, seqLen, numKVHeads, headDim).transposed(0, 2, 1, 3)

        (queries, keys) = glmOcrApplyLanguageRotary(
            q: queries,
            k: keys,
            cos: positionEmbeddings.cos,
            sin: positionEmbeddings.sin
        )

        let attended = glmOcrAttentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )

        let output = attended.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
        return oProj(output)
    }
}

private final class GlmOcrMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: GlmOcrModelConfig.TextConfig) {
        self._gateUpProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize * 2, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = gateUpProj(x)
        let chunks = split(projected, parts: 2, axis: -1)
        return downProj(silu(chunks[0]) * chunks[1])
    }
}

private final class GlmOcrDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: GlmOcrAttention
    @ModuleInfo(key: "mlp") var mlp: GlmOcrMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm

    init(config: GlmOcrModelConfig.TextConfig) {
        self._selfAttention.wrappedValue = GlmOcrAttention(config: config)
        self._mlp.wrappedValue = GlmOcrMLP(config: config)

        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postSelfAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postMlpLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: GlmOcrKVCache?
    ) -> MLXArray {
        var residual = x
        var x = selfAttention(
            inputLayerNorm(x),
            positionEmbeddings: positionEmbeddings,
            mask: mask,
            cache: cache
        )

        x = postSelfAttentionLayerNorm(x)
        x = residual + x

        residual = x
        x = postAttentionLayerNorm(x)
        x = mlp(x)
        x = postMlpLayerNorm(x)
        return residual + x
    }
}

private final class GlmOcrTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [GlmOcrDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    private let rotaryEmbedding: GlmOcrRotaryEmbedding

    init(config: GlmOcrModelConfig.TextConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            GlmOcrDecoderLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rotaryEmbedding = GlmOcrRotaryEmbedding(config: config)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        inputEmbeddings: MLXArray?,
        cache: [GlmOcrKVCache?]?,
        positionIDs: MLXArray?
    ) -> MLXArray {
        var hiddenStates: MLXArray
        if let inputEmbeddings {
            hiddenStates = inputEmbeddings.asType(norm.weight.dtype)
        } else {
            hiddenStates = embedTokens(inputs)
        }

        let resolvedPositionIDs: MLXArray
        if let positionIDs {
            resolvedPositionIDs = positionIDs
        } else {
            let offset = cache?.first??.offset ?? 0
            var positions = MLXArray(stride(from: offset, to: offset + hiddenStates.dim(-2), by: 1))
            positions = positions[.newAxis, 0...]
            positions = tiled(positions, repetitions: [3, 1, 1])
            resolvedPositionIDs = positions
        }

        let positionEmbeddings = rotaryEmbedding(hiddenStates, positionIDs: resolvedPositionIDs)
        let mask = glmOcrCreateAttentionMask(hiddenStates: hiddenStates, cache: cache?.first ?? nil)

        let layerCaches: [GlmOcrKVCache?]
        if let cache {
            layerCaches = cache
        } else {
            layerCaches = Array(repeating: nil, count: layers.count)
        }

        var h = hiddenStates
        for (index, layer) in layers.enumerated() {
            h = layer(
                h,
                positionEmbeddings: positionEmbeddings,
                mask: mask,
                cache: layerCaches[index]
            )
        }

        return norm(h)
    }
}

private final class GlmOcrLanguageModel: Module {
    @ModuleInfo(key: "model") private var model: GlmOcrTextModel
    @ModuleInfo(key: "lm_head") private var lmHead: Linear

    private let textConfig: GlmOcrModelConfig.TextConfig
    private let modelConfig: GlmOcrModelConfig

    private var ropeDeltas: MLXArray?
    private var cachedPositionIDs: MLXArray?
    private let debugPositionTrace = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_POSITION_TRACE"] == "1"

    init(config: GlmOcrModelConfig.TextConfig, modelConfig: GlmOcrModelConfig, vocabularySize: Int) {
        self.textConfig = config
        self.modelConfig = modelConfig

        self._model.wrappedValue = GlmOcrTextModel(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, vocabularySize, bias: false)
    }

    func embedTokens(_ inputIDs: MLXArray) -> MLXArray {
        model.embedTokens(inputIDs)
    }

    func resetPositionState() {
        ropeDeltas = nil
        cachedPositionIDs = nil
    }

    func cachePositionState(positionIDs: MLXArray, ropeDeltas: MLXArray) {
        self.cachedPositionIDs = positionIDs
        self.ropeDeltas = ropeDeltas
    }

    private func tracePosition(_ message: String) {
        guard debugPositionTrace else {
            return
        }

        let payload = "[GlmOcrLanguageModel] \(message)\n"
        let data = payload.data(using: .utf8) ?? Data()
        FileHandle.standardError.write(data)
    }

    func getRopeIndex(
        inputIDs: MLXArray,
        imageGridTHW: MLXArray?,
        videoGridTHW: MLXArray?,
        attentionMask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let batchSize = inputIDs.dim(0)
        let sequenceLength = inputIDs.dim(1)

        var positionIDs = MLXArray(0..<sequenceLength).asType(.int32)
        positionIDs = broadcast(positionIDs[.newAxis, 0...], to: [batchSize, sequenceLength])

        let spatialMergeSize = modelConfig.visionConfig.spatialMergeSize
        let imageTokenID = modelConfig.imageTokenID
        let videoTokenID = modelConfig.videoTokenID
        let imageStartTokenID = modelConfig.imageStartTokenID

        if imageGridTHW != nil || videoGridTHW != nil {
            let mask: MLXArray
            if let attentionMask, attentionMask.dim(-1) == inputIDs.dim(-1) {
                mask = attentionMask
            } else {
                mask = ones(like: inputIDs)
            }

            positionIDs = MLXArray.ones([3, batchSize, sequenceLength], dtype: inputIDs.dtype)
            var imageIndex = 0
            var videoIndex = 0
            var mropeDeltas: [Int32] = []
            mropeDeltas.reserveCapacity(batchSize)

            for batchIndex in 0..<batchSize {
                var batchInput = inputIDs[batchIndex, 0...]
                batchInput = `where`(mask[batchIndex, 0...] .== 1, batchInput, zeros(like: batchInput))

                let inputTokens = batchInput.asArray(Int32.self).map(Int.init)

                var visionStartIndicesSum = 0
                for (index, token) in inputTokens.enumerated() where token == imageStartTokenID {
                    visionStartIndicesSum += index
                }

                let visionTokenIndex = min(max(visionStartIndicesSum + 1, 0), max(inputTokens.count - 1, 0))
                let visionToken = inputTokens.isEmpty ? -1 : inputTokens[visionTokenIndex]

                var imageCount = visionToken == imageTokenID ? 1 : 0
                var videoCount = visionToken == videoTokenID ? 1 : 0

                var llmPositionList: [MLXArray] = []
                var start = 0

                for _ in 0..<(imageCount + videoCount) {
                    let endImage: Int
                    if imageCount > 0, let idx = inputTokens[start...].firstIndex(of: imageTokenID) {
                        endImage = idx
                    } else {
                        endImage = inputTokens.count + 1
                    }

                    let endVideo: Int
                    if videoCount > 0, let idx = inputTokens[start...].firstIndex(of: videoTokenID) {
                        endVideo = idx
                    } else {
                        endVideo = inputTokens.count + 1
                    }

                    let frameT: Int
                    let frameH: Int
                    let frameW: Int
                    let end: Int

                    if endImage < endVideo {
                        guard let imageGridTHW else { break }
                        frameT = imageGridTHW[imageIndex, 0].item(Int.self)
                        frameH = imageGridTHW[imageIndex, 1].item(Int.self)
                        frameW = imageGridTHW[imageIndex, 2].item(Int.self)
                        imageIndex += 1
                        imageCount -= 1
                        end = endImage
                    } else {
                        guard let videoGridTHW else { break }
                        frameT = videoGridTHW[videoIndex, 0].item(Int.self)
                        frameH = videoGridTHW[videoIndex, 1].item(Int.self)
                        frameW = videoGridTHW[videoIndex, 2].item(Int.self)
                        videoIndex += 1
                        videoCount -= 1
                        end = endVideo
                    }

                    let gridT = frameT
                    let gridH = frameH / spatialMergeSize
                    let gridW = frameW / spatialMergeSize

                    let textLength = max(end - start, 0)
                    let startIndex = llmPositionList.last.map { $0.max().item(Int.self) + 1 } ?? 0

                    var index = MLXArray(0..<textLength).reshaped(1, textLength)
                    index = broadcast(index, to: [3, textLength])
                    llmPositionList.append(index + MLXArray(startIndex))

                    var tIndex = MLXArray(0..<gridT).reshaped(gridT, 1)
                    tIndex = broadcast(tIndex, to: [gridT, gridH * gridW]).flattened()

                    var hIndex = MLXArray(0..<gridH).reshaped(1, gridH, 1)
                    hIndex = broadcast(hIndex, to: [gridT, gridH, gridW]).flattened()

                    var wIndex = MLXArray(0..<gridW).reshaped(1, 1, gridW)
                    wIndex = broadcast(wIndex, to: [gridT, gridH, gridW]).flattened()

                    let visualPositions = stacked([tIndex, hIndex, wIndex]) + MLXArray(textLength + startIndex)
                    llmPositionList.append(visualPositions)

                    start = end + gridT * gridH * gridW
                }

                if start < inputTokens.count {
                    let startIndex = llmPositionList.last.map { $0.max().item(Int.self) + 1 } ?? 0
                    let textLength = inputTokens.count - start
                    var tail = MLXArray(0..<textLength).reshaped(1, textLength)
                    tail = broadcast(tail, to: [3, textLength])
                    llmPositionList.append(tail + MLXArray(startIndex))
                }

                guard !llmPositionList.isEmpty else {
                    mropeDeltas.append(0)
                    continue
                }

                let llmPositions = concatenated(llmPositionList, axis: 1).reshaped(3, -1)
                let batchMask = mask[batchIndex, 0...] .== 1
                let expandedMask = broadcast(expandedDimensions(batchMask, axis: 0), to: [3, 1, batchMask.dim(0)])
                let expandedPositions = expandedDimensions(llmPositions, axis: 1)

                let newPositions = `where`(
                    expandedMask,
                    expandedPositions,
                    positionIDs[0..., batchIndex..<(batchIndex + 1), 0...]
                )

                positionIDs = concatenated(
                    [
                        positionIDs[0..., ..<batchIndex, 0...],
                        newPositions,
                        positionIDs[0..., (batchIndex + 1)..., 0...],
                    ],
                    axis: 1
                )

                let delta = llmPositions.max().item(Int.self) + 1 - inputTokens.count
                mropeDeltas.append(Int32(delta))
            }

            let deltaArray = mropeDeltas.isEmpty ? MLXArray(0, dtype: .int32) : MLXArray(mropeDeltas)[0]
            return (positionIDs, deltaArray)
        }

        if let attentionMask {
            var positions = attentionMask.asType(.int64).cumsum(axis: -1) - MLXArray(1, dtype: .int64)
            positions = `where`(attentionMask .== 0, ones(like: positions), positions)
            positions = expandedDimensions(positions[0, 0...], axis: 0)
            positions = tiled(positions, repetitions: [3, 1, 1])

            let maxPosition = positions.max(axis: 0)[0].max(axis: -1)[0]
            let deltas =
                maxPosition + MLXArray(1, dtype: maxPosition.dtype)
                - MLXArray(attentionMask.dim(-1), dtype: maxPosition.dtype)
            return (positions, deltas)
        }

        var positions = MLXArray(0..<inputIDs.dim(1)).reshaped(1, -1)
        positions = broadcast(positions, to: [3, inputIDs.dim(0), inputIDs.dim(1)])
        let deltas = MLXArray.zeros([inputIDs.dim(0), 1], dtype: inputIDs.dtype)
        return (positions, deltas)
    }

    func callAsFunction(
        _ inputIDs: MLXArray,
        inputEmbeddings: MLXArray?,
        attentionMask: MLXArray?,
        cache: [GlmOcrKVCache?],
        pixelValues: MLXArray?,
        imageGridTHW: MLXArray?,
        videoGridTHW: MLXArray?,
        positionIDs: MLXArray?
    ) -> MLXArray {
        var positionIDs = positionIDs

        if pixelValues != nil {
            ropeDeltas = nil
        }

        let cacheOffset = cache.first??.offset ?? 0

        let ropeMask: MLXArray?
        if let attentionMask, attentionMask.dim(-1) == inputIDs.dim(-1) {
            ropeMask = attentionMask
        } else {
            ropeMask = nil
        }

        if positionIDs == nil {
            if (cacheOffset == 0) || ropeDeltas == nil {
                if let cachedPositionIDs {
                    let sequenceLength = inputIDs.dim(1)
                    positionIDs = cachedPositionIDs[0..., 0..., cacheOffset..<(cacheOffset + sequenceLength)]
                } else {
                    let (computedPositionIDs, computedDeltas) = getRopeIndex(
                        inputIDs: inputIDs,
                        imageGridTHW: imageGridTHW,
                        videoGridTHW: videoGridTHW,
                        attentionMask: ropeMask
                    )
                    positionIDs = computedPositionIDs
                    ropeDeltas = computedDeltas
                    cachedPositionIDs = computedPositionIDs
                }
            } else {
                let batchSize = inputIDs.dim(0)
                let sequenceLength = inputIDs.dim(1)

                var delta = MLXArray(cacheOffset).asType(.int32)
                if let ropeDeltas {
                    delta = delta + ropeDeltas.asType(.int32)
                }

                var positionBase = MLXArray(0..<sequenceLength).reshaped(1, -1)
                positionBase = broadcast(positionBase, to: [batchSize, sequenceLength])

                if delta.ndim == 0 {
                    delta = expandedDimensions(delta, axis: 0)
                }
                if delta.ndim == 1 {
                    delta = expandedDimensions(delta, axis: 1)
                }

                if delta.dim(0) < batchSize {
                    delta = tiled(delta, repetitions: [batchSize, 1])
                } else if delta.dim(0) > batchSize {
                    delta = delta[..<batchSize, 0...]
                }

                positionBase = positionBase + delta
                positionIDs = expandedDimensions(positionBase, axis: 0)
                positionIDs = broadcast(positionIDs!, to: [3, batchSize, sequenceLength])
            }
        }

        if debugPositionTrace, let positionIDs {
            let sequenceLength = inputIDs.dim(1)
            let axis0 = positionIDs[0, 0, 0...].asArray(Int32.self).map(Int.init)
            let head = Array(axis0.prefix(4))
            let tail = Array(axis0.suffix(4))
            let ropeDeltaValue: String
            if let ropeDeltas {
                ropeDeltaValue = "\(ropeDeltas)"
            } else {
                ropeDeltaValue = "nil"
            }
            tracePosition(
                "cacheOffset=\(cacheOffset) seq=\(sequenceLength) ropeDeltas=\(ropeDeltaValue) axis0Head=\(head) axis0Tail=\(tail)"
            )
        }

        let hidden = model(
            inputIDs,
            inputEmbeddings: inputEmbeddings,
            cache: cache,
            positionIDs: positionIDs
        )

        return lmHead(hidden)
    }
}

extension Array {
    fileprivate func chunked(into size: Int) -> [[Element]] {
        guard size > 0 else { return [] }
        var result: [[Element]] = []
        result.reserveCapacity((count + size - 1) / size)

        var index = 0
        while index < count {
            let end = Swift.min(index + size, count)
            result.append(Array(self[index..<end]))
            index += size
        }

        return result
    }
}
