import Foundation
import MLX

internal struct PPDocLayoutModel: Sendable {
    private let configuration: PPDocLayoutV3Configuration
    private let weights: PPDocLayoutWeightStore

    private let backbone: PPDocLayoutBackbone
    private let encoder: PPDocLayoutEncoder
    private let attention: PPDocLayoutAttention
    private let decoder: PPDocLayoutDecoder

    internal init(snapshot: PPDocLayoutWeightSnapshot) {
        self.configuration = snapshot.configuration
        self.weights = snapshot.weights

        self.backbone = PPDocLayoutBackbone(
            config: snapshot.configuration,
            weights: snapshot.weights
        )
        self.encoder = PPDocLayoutEncoder(
            config: snapshot.configuration,
            weights: snapshot.weights
        )
        self.attention = PPDocLayoutAttention(
            config: snapshot.configuration,
            weights: snapshot.weights
        )
        self.decoder = PPDocLayoutDecoder(
            config: snapshot.configuration,
            weights: snapshot.weights,
            attention: attention
        )
    }

    internal var id2label: [Int: String] {
        configuration.id2label
    }

    internal func predict(pixelValues: MLXArray) throws -> PPDocLayoutMLXPrediction {
        let backboneOutput = try backbone.forward(pixelValues)

        let projected = try encoder.projectBackboneFeatures(backboneOutput.stageFeatures)
        let encoderOutput = try encoder.forward(
            projectedFeatures: projected,
            x4Feature: backboneOutput.x4Feature
        )

        var sources: [MLXArray] = []
        sources.reserveCapacity(configuration.numFeatureLevels)

        for level in 0..<min(configuration.decoderInChannels.count, encoderOutput.panFeatures.count) {
            sources.append(try decoderInputProjection(encoderOutput.panFeatures[level], level: level))
        }

        if configuration.numFeatureLevels > sources.count,
            let tailFeature = encoderOutput.panFeatures.last
        {
            var nextLevel = sources.count
            while nextLevel < configuration.numFeatureLevels {
                sources.append(try decoderInputProjection(tailFeature, level: nextLevel))
                nextLevel += 1
            }
        }

        let flattened = flattenSources(sources)
        let sourceFlatten = flattened.sourceFlatten
        let spatialShapes = flattened.spatialShapes

        let anchors = generateAnchors(spatialShapes: spatialShapes, batchSize: sourceFlatten.shape[0])
        let memory = sourceFlatten * anchors.validMask

        let outputMemory = try encoderOutputProjection(memory)

        let classWeight = try weights.tensor("model.enc_score_head.weight")
        let classBias = try weights.tensor("model.enc_score_head.bias")
        let encOutputsClass = PPDocLayoutMLXTensorOps.linear(outputMemory, weight: classWeight, bias: classBias)

        let bboxLogits = try bboxPredictionHead(outputMemory)
        let encOutputsCoordLogits = bboxLogits + anchors.anchors

        let topKIndices = topKIndices(
            classLogits: encOutputsClass,
            k: configuration.numQueries
        )

        var referencePointsUnact = gatherRows(encOutputsCoordLogits, indices: topKIndices)
        let target = gatherRows(outputMemory, indices: topKIndices)

        if configuration.maskEnhanced {
            let decoderNormWeight = try weights.tensor("model.decoder_norm.weight")
            let decoderNormBias = try weights.tensor("model.decoder_norm.bias")
            let outQuery = PPDocLayoutMLXTensorOps.layerNorm(
                target,
                weight: decoderNormWeight,
                bias: decoderNormBias,
                eps: configuration.layerNormEps
            )

            let maskQueryEmbed = try maskQueryHead(outQuery)
            let maskHeight = encoderOutput.maskFeatures.shape[2]
            let maskWidth = encoderOutput.maskFeatures.shape[3]
            let maskFeaturesFlattened = encoderOutput.maskFeatures.reshaped(
                encoderOutput.maskFeatures.shape[0],
                encoderOutput.maskFeatures.shape[1],
                maskHeight * maskWidth
            )

            var encOutMasks = matmul(maskQueryEmbed, maskFeaturesFlattened)
            encOutMasks = encOutMasks.reshaped(
                encoderOutput.maskFeatures.shape[0],
                maskQueryEmbed.shape[1],
                maskHeight,
                maskWidth
            )

            let maskBoxes = maskToBoxCoordinate(encOutMasks)
            referencePointsUnact = PPDocLayoutMLXTensorOps.inverseSigmoid(maskBoxes)
        }

        return try decoder.decode(
            target: target,
            encoderHiddenStates: sourceFlatten,
            referencePointsUnact: referencePointsUnact,
            spatialShapes: spatialShapes,
            maskFeatures: encoderOutput.maskFeatures
        )
    }

    private func decoderInputProjection(_ source: MLXArray, level: Int) throws -> MLXArray {
        let convWeight = try weights.tensor("model.decoder_input_proj.\(level).0.weight")
        var hidden = PPDocLayoutMLXTensorOps.conv2dNCHW(
            source,
            weight: convWeight,
            stride: 1,
            padding: 0
        )

        let bnWeight = try weights.tensor("model.decoder_input_proj.\(level).1.weight")
        let bnBias = try weights.tensor("model.decoder_input_proj.\(level).1.bias")
        let bnMean = try weights.tensor("model.decoder_input_proj.\(level).1.running_mean")
        let bnVar = try weights.tensor("model.decoder_input_proj.\(level).1.running_var")

        hidden = PPDocLayoutMLXTensorOps.batchNorm2d(
            hidden,
            weight: bnWeight,
            bias: bnBias,
            runningMean: bnMean,
            runningVar: bnVar,
            eps: configuration.batchNormEps
        )

        return hidden
    }

    private func flattenSources(_ sources: [MLXArray]) -> (
        sourceFlatten: MLXArray,
        spatialShapes: [(height: Int, width: Int)]
    ) {
        var flattened: [MLXArray] = []
        var spatialShapes: [(height: Int, width: Int)] = []

        for source in sources {
            let height = source.shape[2]
            let width = source.shape[3]
            spatialShapes.append((height: height, width: width))

            let flat = source.reshaped(source.shape[0], source.shape[1], height * width).transposed(0, 2, 1)
            flattened.append(flat)
        }

        return (
            sourceFlatten: concatenated(flattened, axis: 1),
            spatialShapes: spatialShapes
        )
    }

    private func generateAnchors(
        spatialShapes: [(height: Int, width: Int)],
        batchSize: Int,
        gridSize: Float = 0.05
    ) -> (anchors: MLXArray, validMask: MLXArray) {
        var anchorValues: [Float] = []
        var validValues: [Float] = []

        for (level, shape) in spatialShapes.enumerated() {
            let height = shape.height
            let width = shape.width
            let wh = gridSize * pow(2, Float(level))

            for y in 0..<height {
                for x in 0..<width {
                    let cx = (Float(x) + 0.5) / Float(width)
                    let cy = (Float(y) + 0.5) / Float(height)

                    let isValid = cx > 0.01 && cx < 0.99 && cy > 0.01 && cy < 0.99 && wh > 0.01 && wh < 0.99

                    if isValid {
                        anchorValues.append(log(cx / (1 - cx)))
                        anchorValues.append(log(cy / (1 - cy)))
                        anchorValues.append(log(wh / (1 - wh)))
                        anchorValues.append(log(wh / (1 - wh)))
                        validValues.append(1)
                    } else {
                        let invalid = Float.greatestFiniteMagnitude
                        anchorValues.append(invalid)
                        anchorValues.append(invalid)
                        anchorValues.append(invalid)
                        anchorValues.append(invalid)
                        validValues.append(0)
                    }
                }
            }
        }

        let anchorCount = validValues.count
        var anchors = MLXArray(anchorValues).reshaped(1, anchorCount, 4)
        var validMask = MLXArray(validValues).reshaped(1, anchorCount, 1)

        if batchSize > 1 {
            anchors = tiled(anchors, repetitions: [batchSize, 1, 1])
            validMask = tiled(validMask, repetitions: [batchSize, 1, 1])
        }

        return (anchors: anchors, validMask: validMask)
    }

    private func encoderOutputProjection(_ memory: MLXArray) throws -> MLXArray {
        let linearWeight = try weights.tensor("model.enc_output.0.weight")
        let linearBias = try weights.tensor("model.enc_output.0.bias")

        var output = PPDocLayoutMLXTensorOps.linear(
            memory,
            weight: linearWeight,
            bias: linearBias
        )

        let normWeight = try weights.tensor("model.enc_output.1.weight")
        let normBias = try weights.tensor("model.enc_output.1.bias")

        output = PPDocLayoutMLXTensorOps.layerNorm(
            output,
            weight: normWeight,
            bias: normBias,
            eps: configuration.layerNormEps
        )

        return output
    }

    private func bboxPredictionHead(_ x: MLXArray) throws -> MLXArray {
        try mlpHead(x, prefix: "model.enc_bbox_head.layers", numLayers: 3)
    }

    private func maskQueryHead(_ x: MLXArray) throws -> MLXArray {
        try mlpHead(x, prefix: "model.mask_query_head.layers", numLayers: 3)
    }

    private func mlpHead(_ x: MLXArray, prefix: String, numLayers: Int) throws -> MLXArray {
        var hidden = x

        for layerIndex in 0..<numLayers {
            let weight = try weights.tensor("\(prefix).\(layerIndex).weight")
            let bias = try weights.tensor("\(prefix).\(layerIndex).bias")
            hidden = PPDocLayoutMLXTensorOps.linear(hidden, weight: weight, bias: bias)
            if layerIndex < numLayers - 1 {
                hidden = PPDocLayoutMLXTensorOps.activation(hidden, name: "relu")
            }
        }

        return hidden
    }

    private func topKIndices(classLogits: MLXArray, k: Int) -> [[Int]] {
        let batch = classLogits.shape[0]
        let sequenceLength = classLogits.shape[1]
        let classCount = classLogits.shape[2]

        let values = classLogits.asArray(Float.self)
        var result: [[Int]] = []
        result.reserveCapacity(batch)

        for batchIndex in 0..<batch {
            var scoreIndexPairs: [(score: Float, index: Int)] = []
            scoreIndexPairs.reserveCapacity(sequenceLength)

            for tokenIndex in 0..<sequenceLength {
                var best = -Float.greatestFiniteMagnitude
                for classIndex in 0..<classCount {
                    let flatIndex = (((batchIndex * sequenceLength) + tokenIndex) * classCount) + classIndex
                    let value = values[flatIndex]
                    if value > best {
                        best = value
                    }
                }
                scoreIndexPairs.append((score: best, index: tokenIndex))
            }

            scoreIndexPairs.sort { lhs, rhs in
                if lhs.score != rhs.score {
                    return lhs.score > rhs.score
                }
                return lhs.index < rhs.index
            }

            let top = scoreIndexPairs.prefix(min(k, sequenceLength)).map(\.index)
            result.append(top)
        }

        return result
    }

    private func gatherRows(_ tensor: MLXArray, indices: [[Int]]) -> MLXArray {
        let batch = tensor.shape[0]
        let sequenceLength = tensor.shape[1]
        let featureDim = tensor.shape[2]

        let tensorValues = tensor.asArray(Float.self)

        let gatheredCount = indices.first?.count ?? 0
        var gathered = [Float](repeating: 0, count: batch * gatheredCount * featureDim)

        for batchIndex in 0..<batch {
            for (gatherIndex, sourceIndex) in indices[batchIndex].enumerated() {
                let boundedSource = min(max(0, sourceIndex), sequenceLength - 1)

                for featureIndex in 0..<featureDim {
                    let sourceFlat = (((batchIndex * sequenceLength) + boundedSource) * featureDim) + featureIndex
                    let targetFlat = (((batchIndex * gatheredCount) + gatherIndex) * featureDim) + featureIndex
                    gathered[targetFlat] = tensorValues[sourceFlat]
                }
            }
        }

        return MLXArray(gathered).reshaped(batch, gatheredCount, featureDim)
    }

    private func maskToBoxCoordinate(_ masks: MLXArray) -> MLXArray {
        let batch = masks.shape[0]
        let queryCount = masks.shape[1]
        let height = masks.shape[2]
        let width = masks.shape[3]

        let values = masks.asArray(Float.self)

        var boxes = [Float](repeating: 0, count: batch * queryCount * 4)

        for batchIndex in 0..<batch {
            for queryIndex in 0..<queryCount {
                var minX = width
                var minY = height
                var maxX = -1
                var maxY = -1

                for y in 0..<height {
                    for x in 0..<width {
                        let flat = (((batchIndex * queryCount + queryIndex) * height + y) * width) + x
                        if values[flat] > 0 {
                            minX = min(minX, x)
                            minY = min(minY, y)
                            maxX = max(maxX, x)
                            maxY = max(maxY, y)
                        }
                    }
                }

                let base = (batchIndex * queryCount + queryIndex) * 4

                if maxX >= minX, maxY >= minY {
                    let xMinNorm = Float(minX) / Float(width)
                    let yMinNorm = Float(minY) / Float(height)
                    let xMaxNorm = Float(maxX + 1) / Float(width)
                    let yMaxNorm = Float(maxY + 1) / Float(height)

                    boxes[base + 0] = (xMinNorm + xMaxNorm) * 0.5
                    boxes[base + 1] = (yMinNorm + yMaxNorm) * 0.5
                    boxes[base + 2] = xMaxNorm - xMinNorm
                    boxes[base + 3] = yMaxNorm - yMinNorm
                } else {
                    // fall back to a tiny centered box when mask is empty
                    boxes[base + 0] = 0.5
                    boxes[base + 1] = 0.5
                    boxes[base + 2] = 1e-3
                    boxes[base + 3] = 1e-3
                }
            }
        }

        return MLXArray(boxes).reshaped(batch, queryCount, 4)
    }
}
