import Foundation
import MLX

internal struct PPDocLayoutDecoder: Sendable {
    private let config: PPDocLayoutV3Configuration
    private let weights: PPDocLayoutWeightStore
    private let attention: PPDocLayoutAttention

    internal init(
        config: PPDocLayoutV3Configuration,
        weights: PPDocLayoutWeightStore,
        attention: PPDocLayoutAttention
    ) {
        self.config = config
        self.weights = weights
        self.attention = attention
    }

    internal func decode(
        target: MLXArray,
        encoderHiddenStates: MLXArray,
        referencePointsUnact: MLXArray,
        spatialShapes: [(height: Int, width: Int)],
        maskFeatures: MLXArray
    ) throws -> PPDocLayoutMLXPrediction {
        let batch = target.shape[0]
        let queryCount = target.shape[1]

        var hiddenStates = target
        var referencePoints = PPDocLayoutMLXTensorOps.sigmoid(referencePointsUnact)

        let maskHeight = maskFeatures.shape[2]
        let maskWidth = maskFeatures.shape[3]
        let maskFeaturesFlattened = maskFeatures.reshaped(batch, maskFeatures.shape[1], maskHeight * maskWidth)

        var finalLogits: MLXArray?
        var finalBoxes: MLXArray?
        var finalOrderLogits: MLXArray?
        var finalMasks: MLXArray?

        for layerIndex in 0..<config.decoderLayers {
            let layerPrefix = "model.decoder.layers.\(layerIndex)"
            let referencePointsInput = referencePoints.expandedDimensions(axis: 2)
            let queryPositionEmbeddings = try queryPositionHead(referencePoints)

            hiddenStates = try decoderLayer(
                hiddenStates,
                positionEmbedding: queryPositionEmbeddings,
                encoderHiddenStates: encoderHiddenStates,
                referencePoints: referencePointsInput,
                spatialShapes: spatialShapes,
                prefix: layerPrefix
            )

            let predictedCorners = try bboxPredictionHead(hiddenStates)
            let inverseReference = PPDocLayoutMLXTensorOps.inverseSigmoid(referencePoints)
            referencePoints = PPDocLayoutMLXTensorOps.sigmoid(predictedCorners + inverseReference)

            let decoderNormWeight = try weights.tensor("model.decoder_norm.weight")
            let decoderNormBias = try weights.tensor("model.decoder_norm.bias")
            let outQuery = PPDocLayoutMLXTensorOps.layerNorm(
                hiddenStates,
                weight: decoderNormWeight,
                bias: decoderNormBias,
                eps: config.layerNormEps
            )

            let maskQueryEmbed = try maskQueryHead(outQuery)
            var outMask = matmul(maskQueryEmbed, maskFeaturesFlattened)
            outMask = outMask.reshaped(batch, queryCount, maskHeight, maskWidth)

            let logits = try classPredictionHead(outQuery)
            let orderProjectionWeight = try weights.tensor("model.decoder_order_head.\(layerIndex).weight")
            let orderProjectionBias = try weights.tensor("model.decoder_order_head.\(layerIndex).bias")

            let orderProjection = PPDocLayoutMLXTensorOps.linear(
                outQuery,
                weight: orderProjectionWeight,
                bias: orderProjectionBias
            )
            let orderLogits = try attention.globalPointer(orderProjection)

            finalLogits = logits
            finalBoxes = referencePoints
            finalOrderLogits = orderLogits
            finalMasks = outMask
        }

        guard let logits = finalLogits,
            let predBoxes = finalBoxes,
            let orderLogits = finalOrderLogits,
            let outMasks = finalMasks
        else {
            throw PPDocLayoutMLXError.modelInitializationFailed(
                "Decoder produced no outputs"
            )
        }

        return PPDocLayoutMLXPrediction(
            logits: logits,
            predBoxes: predBoxes,
            orderLogits: orderLogits,
            outMasks: outMasks
        )
    }

    private func decoderLayer(
        _ hiddenStates: MLXArray,
        positionEmbedding: MLXArray,
        encoderHiddenStates: MLXArray,
        referencePoints: MLXArray,
        spatialShapes: [(height: Int, width: Int)],
        prefix: String
    ) throws -> MLXArray {
        let residual = hiddenStates

        var hidden = try selfAttention(
            hiddenStates,
            positionEmbedding: positionEmbedding,
            prefix: "\(prefix).self_attn"
        )

        hidden = residual + hidden

        let selfAttnNormWeight = try weights.tensor("\(prefix).self_attn_layer_norm.weight")
        let selfAttnNormBias = try weights.tensor("\(prefix).self_attn_layer_norm.bias")
        hidden = PPDocLayoutMLXTensorOps.layerNorm(
            hidden,
            weight: selfAttnNormWeight,
            bias: selfAttnNormBias,
            eps: config.layerNormEps
        )

        let crossResidual = hidden
        hidden = try attention.deformableCrossAttention(
            hiddenStates: hidden,
            positionEmbedding: positionEmbedding,
            encoderHiddenStates: encoderHiddenStates,
            referencePoints: referencePoints,
            spatialShapes: spatialShapes,
            prefix: "\(prefix).encoder_attn"
        )

        hidden = crossResidual + hidden

        let crossNormWeight = try weights.tensor("\(prefix).encoder_attn_layer_norm.weight")
        let crossNormBias = try weights.tensor("\(prefix).encoder_attn_layer_norm.bias")
        hidden = PPDocLayoutMLXTensorOps.layerNorm(
            hidden,
            weight: crossNormWeight,
            bias: crossNormBias,
            eps: config.layerNormEps
        )

        let mlpResidual = hidden
        let fc1Weight = try weights.tensor("\(prefix).fc1.weight")
        let fc1Bias = try weights.tensor("\(prefix).fc1.bias")
        hidden = PPDocLayoutMLXTensorOps.linear(hidden, weight: fc1Weight, bias: fc1Bias)
        hidden = PPDocLayoutMLXTensorOps.activation(hidden, name: config.decoderActivationFunction)

        let fc2Weight = try weights.tensor("\(prefix).fc2.weight")
        let fc2Bias = try weights.tensor("\(prefix).fc2.bias")
        hidden = PPDocLayoutMLXTensorOps.linear(hidden, weight: fc2Weight, bias: fc2Bias)

        hidden = mlpResidual + hidden

        let finalNormWeight = try weights.tensor("\(prefix).final_layer_norm.weight")
        let finalNormBias = try weights.tensor("\(prefix).final_layer_norm.bias")

        return PPDocLayoutMLXTensorOps.layerNorm(
            hidden,
            weight: finalNormWeight,
            bias: finalNormBias,
            eps: config.layerNormEps
        )
    }

    private func selfAttention(
        _ hiddenStates: MLXArray,
        positionEmbedding: MLXArray,
        prefix: String
    ) throws -> MLXArray {
        let batch = hiddenStates.shape[0]
        let sequenceLength = hiddenStates.shape[1]
        let hiddenSize = hiddenStates.shape[2]
        let numHeads = config.decoderAttentionHeads
        let headDim = hiddenSize / max(1, numHeads)

        let queryInput = hiddenStates + positionEmbedding

        let qWeight = try weights.tensor("\(prefix).q_proj.weight")
        let qBias = try weights.tensor("\(prefix).q_proj.bias")
        let kWeight = try weights.tensor("\(prefix).k_proj.weight")
        let kBias = try weights.tensor("\(prefix).k_proj.bias")
        let vWeight = try weights.tensor("\(prefix).v_proj.weight")
        let vBias = try weights.tensor("\(prefix).v_proj.bias")

        var q = PPDocLayoutMLXTensorOps.linear(queryInput, weight: qWeight, bias: qBias)
        var k = PPDocLayoutMLXTensorOps.linear(queryInput, weight: kWeight, bias: kBias)
        var v = PPDocLayoutMLXTensorOps.linear(hiddenStates, weight: vWeight, bias: vBias)

        q = q.reshaped(batch, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

        var attentionWeights = matmul(q, k.transposed(0, 1, 3, 2))
        attentionWeights = attentionWeights * (1.0 / sqrt(Float(headDim)))
        attentionWeights = softmax(attentionWeights, axis: -1)

        var attentionOutput = matmul(attentionWeights, v)
        attentionOutput = attentionOutput.transposed(0, 2, 1, 3).reshaped(batch, sequenceLength, hiddenSize)

        let outWeight = try weights.tensor("\(prefix).out_proj.weight")
        let outBias = try weights.tensor("\(prefix).out_proj.bias")
        return PPDocLayoutMLXTensorOps.linear(attentionOutput, weight: outWeight, bias: outBias)
    }

    private func queryPositionHead(_ x: MLXArray) throws -> MLXArray {
        try mlpHead(
            x,
            prefix: "model.decoder.query_pos_head.layers",
            numLayers: 2
        )
    }

    private func bboxPredictionHead(_ x: MLXArray) throws -> MLXArray {
        try mlpHead(
            x,
            prefix: "model.enc_bbox_head.layers",
            numLayers: 3
        )
    }

    private func maskQueryHead(_ x: MLXArray) throws -> MLXArray {
        try mlpHead(
            x,
            prefix: "model.mask_query_head.layers",
            numLayers: 3
        )
    }

    private func classPredictionHead(_ x: MLXArray) throws -> MLXArray {
        let classWeight = try weights.tensor("model.enc_score_head.weight")
        let classBias = try weights.tensor("model.enc_score_head.bias")
        return PPDocLayoutMLXTensorOps.linear(x, weight: classWeight, bias: classBias)
    }

    private func mlpHead(
        _ x: MLXArray,
        prefix: String,
        numLayers: Int
    ) throws -> MLXArray {
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
}
