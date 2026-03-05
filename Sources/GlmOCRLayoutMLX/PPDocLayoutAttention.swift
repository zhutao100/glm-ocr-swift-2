import Foundation
import MLX

internal struct PPDocLayoutAttention: Sendable {
    private let config: PPDocLayoutV3Configuration
    private let weights: PPDocLayoutWeightStore

    internal init(
        config: PPDocLayoutV3Configuration,
        weights: PPDocLayoutWeightStore
    ) {
        self.config = config
        self.weights = weights
    }

    internal func deformableCrossAttention(
        hiddenStates: MLXArray,
        positionEmbedding: MLXArray,
        encoderHiddenStates: MLXArray,
        referencePoints: MLXArray,
        spatialShapes: [(height: Int, width: Int)],
        prefix: String
    ) throws -> MLXArray {
        let hiddenWithPosition = hiddenStates + positionEmbedding

        let valueProjWeight = try weights.tensor("\(prefix).value_proj.weight")
        let valueProjBias = try weights.tensor("\(prefix).value_proj.bias")
        let samplingOffsetsWeight = try weights.tensor("\(prefix).sampling_offsets.weight")
        let samplingOffsetsBias = try weights.tensor("\(prefix).sampling_offsets.bias")
        let attentionWeightsWeight = try weights.tensor("\(prefix).attention_weights.weight")
        let attentionWeightsBias = try weights.tensor("\(prefix).attention_weights.bias")
        let outputProjWeight = try weights.tensor("\(prefix).output_proj.weight")
        let outputProjBias = try weights.tensor("\(prefix).output_proj.bias")

        let batch = hiddenStates.shape[0]
        let queryCount = hiddenStates.shape[1]
        let numHeads = config.decoderAttentionHeads
        let headDim = config.dModel / max(1, numHeads)
        let numLevels = spatialShapes.count
        let numPoints = config.decoderNPoints

        var value = PPDocLayoutMLXTensorOps.linear(
            encoderHiddenStates,
            weight: valueProjWeight,
            bias: valueProjBias
        )
        let sequenceLength = value.shape[1]
        value = value.reshaped(batch, sequenceLength, numHeads, headDim)

        var samplingOffsets = PPDocLayoutMLXTensorOps.linear(
            hiddenWithPosition,
            weight: samplingOffsetsWeight,
            bias: samplingOffsetsBias
        )
        samplingOffsets = samplingOffsets.reshaped(batch, queryCount, numHeads, numLevels, numPoints, 2)

        var attentionWeights = PPDocLayoutMLXTensorOps.linear(
            hiddenWithPosition,
            weight: attentionWeightsWeight,
            bias: attentionWeightsBias
        )
        attentionWeights = attentionWeights.reshaped(batch, queryCount, numHeads, numLevels * numPoints)
        attentionWeights = softmax(attentionWeights, axis: -1)
        attentionWeights = attentionWeights.reshaped(batch, queryCount, numHeads, numLevels, numPoints)

        let valueArray = value.asArray(Float.self)
        let samplingOffsetsArray = samplingOffsets.asArray(Float.self)
        let attentionWeightsArray = attentionWeights.asArray(Float.self)
        let referencePointsArray = referencePoints.asArray(Float.self)

        let numReferenceLevels = referencePoints.shape[2]
        let numCoordinates = referencePoints.shape[3]

        var levelStartIndex = [Int](repeating: 0, count: numLevels)
        var cursor = 0
        for level in 0..<numLevels {
            levelStartIndex[level] = cursor
            cursor += spatialShapes[level].height * spatialShapes[level].width
        }

        var output = [Float](
            repeating: 0,
            count: batch * queryCount * numHeads * headDim
        )

        for batchIndex in 0..<batch {
            for queryIndex in 0..<queryCount {
                for headIndex in 0..<numHeads {
                    for levelIndex in 0..<numLevels {
                        let levelHeight = spatialShapes[levelIndex].height
                        let levelWidth = spatialShapes[levelIndex].width
                        let levelStart = levelStartIndex[levelIndex]

                        for pointIndex in 0..<numPoints {
                            let attnWeight = attentionWeightsArray[
                                attentionWeightIndex(
                                    batch: batchIndex,
                                    query: queryIndex,
                                    head: headIndex,
                                    level: levelIndex,
                                    point: pointIndex,
                                    queryCount: queryCount,
                                    numHeads: numHeads,
                                    numLevels: numLevels,
                                    numPoints: numPoints
                                )
                            ]

                            if attnWeight == 0 {
                                continue
                            }

                            let offsetIndex = samplingOffsetIndex(
                                batch: batchIndex,
                                query: queryIndex,
                                head: headIndex,
                                level: levelIndex,
                                point: pointIndex,
                                coord: 0,
                                queryCount: queryCount,
                                numHeads: numHeads,
                                numLevels: numLevels,
                                numPoints: numPoints
                            )

                            let offsetX = samplingOffsetsArray[offsetIndex]
                            let offsetY = samplingOffsetsArray[offsetIndex + 1]

                            let referenceLevel = min(levelIndex, max(0, numReferenceLevels - 1))
                            let referenceIndex = referencePointIndex(
                                batch: batchIndex,
                                query: queryIndex,
                                referenceLevel: referenceLevel,
                                coord: 0,
                                queryCount: queryCount,
                                numReferenceLevels: numReferenceLevels,
                                numCoordinates: numCoordinates
                            )

                            let refX = referencePointsArray[referenceIndex]
                            let refY = referencePointsArray[referenceIndex + 1]

                            let samplingX: Float
                            let samplingY: Float

                            if numCoordinates == 2 {
                                samplingX = refX + (offsetX / Float(levelWidth))
                                samplingY = refY + (offsetY / Float(levelHeight))
                            } else {
                                let refW = referencePointsArray[referenceIndex + 2]
                                let refH = referencePointsArray[referenceIndex + 3]
                                samplingX = refX + (offsetX / Float(numPoints)) * refW * 0.5
                                samplingY = refY + (offsetY / Float(numPoints)) * refH * 0.5
                            }

                            let pixelX = (samplingX * Float(levelWidth)) - 0.5
                            let pixelY = (samplingY * Float(levelHeight)) - 0.5

                            let x0 = Int(floor(pixelX))
                            let y0 = Int(floor(pixelY))
                            let x1 = x0 + 1
                            let y1 = y0 + 1

                            let wx1 = pixelX - Float(x0)
                            let wy1 = pixelY - Float(y0)
                            let wx0 = 1 - wx1
                            let wy0 = 1 - wy1

                            for dimension in 0..<headDim {
                                let outputIndex = outputTensorIndex(
                                    batch: batchIndex,
                                    query: queryIndex,
                                    head: headIndex,
                                    dimension: dimension,
                                    queryCount: queryCount,
                                    numHeads: numHeads,
                                    headDim: headDim
                                )

                                var sampledValue: Float = 0

                                sampledValue += bilinearContribution(
                                    valueArray: valueArray,
                                    batch: batchIndex,
                                    sequenceIndex: levelStart + (y0 * levelWidth) + x0,
                                    head: headIndex,
                                    dimension: dimension,
                                    sequenceLength: sequenceLength,
                                    numHeads: numHeads,
                                    headDim: headDim,
                                    x: x0,
                                    y: y0,
                                    width: levelWidth,
                                    height: levelHeight,
                                    weight: wx0 * wy0
                                )

                                sampledValue += bilinearContribution(
                                    valueArray: valueArray,
                                    batch: batchIndex,
                                    sequenceIndex: levelStart + (y0 * levelWidth) + x1,
                                    head: headIndex,
                                    dimension: dimension,
                                    sequenceLength: sequenceLength,
                                    numHeads: numHeads,
                                    headDim: headDim,
                                    x: x1,
                                    y: y0,
                                    width: levelWidth,
                                    height: levelHeight,
                                    weight: wx1 * wy0
                                )

                                sampledValue += bilinearContribution(
                                    valueArray: valueArray,
                                    batch: batchIndex,
                                    sequenceIndex: levelStart + (y1 * levelWidth) + x0,
                                    head: headIndex,
                                    dimension: dimension,
                                    sequenceLength: sequenceLength,
                                    numHeads: numHeads,
                                    headDim: headDim,
                                    x: x0,
                                    y: y1,
                                    width: levelWidth,
                                    height: levelHeight,
                                    weight: wx0 * wy1
                                )

                                sampledValue += bilinearContribution(
                                    valueArray: valueArray,
                                    batch: batchIndex,
                                    sequenceIndex: levelStart + (y1 * levelWidth) + x1,
                                    head: headIndex,
                                    dimension: dimension,
                                    sequenceLength: sequenceLength,
                                    numHeads: numHeads,
                                    headDim: headDim,
                                    x: x1,
                                    y: y1,
                                    width: levelWidth,
                                    height: levelHeight,
                                    weight: wx1 * wy1
                                )

                                output[outputIndex] += attnWeight * sampledValue
                            }
                        }
                    }
                }
            }
        }

        let outputTensor = MLXArray(output).reshaped(batch, queryCount, numHeads * headDim)

        return PPDocLayoutMLXTensorOps.linear(
            outputTensor,
            weight: outputProjWeight,
            bias: outputProjBias
        )
    }

    internal func globalPointer(_ inputs: MLXArray) throws -> MLXArray {
        let denseWeight = try weights.tensor("model.decoder_global_pointer.dense.weight")
        let denseBias = try weights.tensor("model.decoder_global_pointer.dense.bias")

        let batch = inputs.shape[0]
        let sequenceLength = inputs.shape[1]

        var projection = PPDocLayoutMLXTensorOps.linear(
            inputs,
            weight: denseWeight,
            bias: denseBias
        )
        projection = projection.reshaped(batch, sequenceLength, 2, config.globalPointerHeadSize)

        let parts = projection.split(indices: [1], axis: 2)
        let queries = parts[0].squeezed(axis: 2)
        let keys = parts[1].squeezed(axis: 2)

        var logits = matmul(queries, keys.transposed(0, 2, 1))
        logits = logits * (1.0 / sqrt(Float(config.globalPointerHeadSize)))

        let lowerMask = tril(MLXArray.ones([sequenceLength, sequenceLength])).asType(.bool)
        let expandedMask = lowerMask.expandedDimensions(axis: 0)

        return which(expandedMask, MLXArray(-10_000.0), logits)
    }
}

@inline(__always)
private func valueTensorIndex(
    batch: Int,
    sequenceIndex: Int,
    head: Int,
    dimension: Int,
    sequenceLength: Int,
    numHeads: Int,
    headDim: Int
) -> Int {
    (((batch * sequenceLength + sequenceIndex) * numHeads + head) * headDim) + dimension
}

@inline(__always)
private func outputTensorIndex(
    batch: Int,
    query: Int,
    head: Int,
    dimension: Int,
    queryCount: Int,
    numHeads: Int,
    headDim: Int
) -> Int {
    (((batch * queryCount + query) * numHeads + head) * headDim) + dimension
}

@inline(__always)
private func samplingOffsetIndex(
    batch: Int,
    query: Int,
    head: Int,
    level: Int,
    point: Int,
    coord: Int,
    queryCount: Int,
    numHeads: Int,
    numLevels: Int,
    numPoints: Int
) -> Int {
    ((((((batch * queryCount + query) * numHeads + head) * numLevels + level) * numPoints + point) * 2) + coord)
}

@inline(__always)
private func attentionWeightIndex(
    batch: Int,
    query: Int,
    head: Int,
    level: Int,
    point: Int,
    queryCount: Int,
    numHeads: Int,
    numLevels: Int,
    numPoints: Int
) -> Int {
    (((((batch * queryCount + query) * numHeads + head) * numLevels + level) * numPoints) + point)
}

@inline(__always)
private func referencePointIndex(
    batch: Int,
    query: Int,
    referenceLevel: Int,
    coord: Int,
    queryCount: Int,
    numReferenceLevels: Int,
    numCoordinates: Int
) -> Int {
    ((((batch * queryCount + query) * numReferenceLevels + referenceLevel) * numCoordinates) + coord)
}

@inline(__always)
private func bilinearContribution(
    valueArray: [Float],
    batch: Int,
    sequenceIndex: Int,
    head: Int,
    dimension: Int,
    sequenceLength: Int,
    numHeads: Int,
    headDim: Int,
    x: Int,
    y: Int,
    width: Int,
    height: Int,
    weight: Float
) -> Float {
    if x < 0 || x >= width || y < 0 || y >= height || weight == 0 {
        return 0
    }

    let index = valueTensorIndex(
        batch: batch,
        sequenceIndex: sequenceIndex,
        head: head,
        dimension: dimension,
        sequenceLength: sequenceLength,
        numHeads: numHeads,
        headDim: headDim
    )

    return valueArray[index] * weight
}
