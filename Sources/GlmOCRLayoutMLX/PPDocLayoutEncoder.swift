import Foundation
import MLX

internal struct PPDocLayoutEncoderOutput: @unchecked Sendable {
    internal let panFeatures: [MLXArray]
    internal let maskFeatures: MLXArray
}

internal struct PPDocLayoutEncoder: Sendable {
    private let config: PPDocLayoutV3Configuration
    private let weights: PPDocLayoutWeightStore

    internal init(
        config: PPDocLayoutV3Configuration,
        weights: PPDocLayoutWeightStore
    ) {
        self.config = config
        self.weights = weights
    }

    internal func projectBackboneFeatures(_ stageFeatures: [MLXArray]) throws -> [MLXArray] {
        guard stageFeatures.count >= 4 else {
            throw PPDocLayoutMLXError.modelInitializationFailed(
                "Expected 4 backbone stage features, got \(stageFeatures.count)"
            )
        }

        var projected: [MLXArray] = []
        projected.reserveCapacity(3)

        for index in 0..<3 {
            let source = stageFeatures[index + 1]
            let convWeight = try weights.tensor("model.encoder_input_proj.\(index).0.weight")
            var hidden = PPDocLayoutMLXTensorOps.conv2dNCHW(
                source,
                weight: convWeight,
                stride: 1,
                padding: 0
            )

            let bnWeight = try weights.tensor("model.encoder_input_proj.\(index).1.weight")
            let bnBias = try weights.tensor("model.encoder_input_proj.\(index).1.bias")
            let bnMean = try weights.tensor("model.encoder_input_proj.\(index).1.running_mean")
            let bnVar = try weights.tensor("model.encoder_input_proj.\(index).1.running_var")

            hidden = PPDocLayoutMLXTensorOps.batchNorm2d(
                hidden,
                weight: bnWeight,
                bias: bnBias,
                runningMean: bnMean,
                runningVar: bnVar,
                eps: config.batchNormEps
            )
            projected.append(hidden)
        }

        return projected
    }

    internal func forward(
        projectedFeatures: [MLXArray],
        x4Feature: MLXArray
    ) throws -> PPDocLayoutEncoderOutput {
        guard projectedFeatures.count == 3 else {
            throw PPDocLayoutMLXError.modelInitializationFailed(
                "Expected 3 projected features, got \(projectedFeatures.count)"
            )
        }

        var featureMaps = projectedFeatures

        // AIFI (single layer for this config) on specified projected level(s).
        if config.encoderLayers > 0 {
            for (aifiIndex, featureIndex) in config.encodeProjLayers.enumerated() {
                featureMaps[featureIndex] = try aifi(
                    featureMaps[featureIndex],
                    aifiIndex: aifiIndex
                )
            }
        }

        // top-down FPN
        let numFpnStages = featureMaps.count - 1
        var fpnFeatureMaps: [MLXArray] = [featureMaps[numFpnStages]]

        for idx in 0..<numFpnStages {
            let backboneFeatureMap = featureMaps[numFpnStages - idx - 1]
            var topFeature = fpnFeatureMaps[fpnFeatureMaps.count - 1]

            topFeature = try PPDocLayoutMLXTensorOps.convNormAct(
                topFeature,
                store: weights,
                prefix: "model.encoder.lateral_convs.\(idx)",
                activation: config.activationFunction,
                eps: config.batchNormEps
            )
            fpnFeatureMaps[fpnFeatureMaps.count - 1] = topFeature

            let upsampled = PPDocLayoutMLXTensorOps.upsampleNearestNCHW(topFeature, scale: 2)
            let fused = concatenated([upsampled, backboneFeatureMap], axis: 1)
            let newFeature = try cspRepLayer(
                fused,
                prefix: "model.encoder.fpn_blocks.\(idx)"
            )
            fpnFeatureMaps.append(newFeature)
        }

        fpnFeatureMaps.reverse()

        // bottom-up PAN
        let numPanStages = fpnFeatureMaps.count - 1
        var panFeatureMaps: [MLXArray] = [fpnFeatureMaps[0]]

        for idx in 0..<numPanStages {
            let topPan = panFeatureMaps[panFeatureMaps.count - 1]
            let fpnFeature = fpnFeatureMaps[idx + 1]

            let downsampled = try PPDocLayoutMLXTensorOps.convNormAct(
                topPan,
                store: weights,
                prefix: "model.encoder.downsample_convs.\(idx)",
                stride: 2,
                activation: config.activationFunction,
                eps: config.batchNormEps
            )

            let fused = concatenated([downsampled, fpnFeature], axis: 1)
            let newPan = try cspRepLayer(
                fused,
                prefix: "model.encoder.pan_blocks.\(idx)"
            )
            panFeatureMaps.append(newPan)
        }

        var maskFeatures = try maskFeatureHead(panFeatureMaps)
        maskFeatures = PPDocLayoutMLXTensorOps.upsampleBilinearNCHW(maskFeatures, scale: 2)

        let x4Lateral = try PPDocLayoutMLXTensorOps.convLayer(
            x4Feature,
            store: weights,
            prefix: "model.encoder.encoder_mask_lateral",
            activation: "silu",
            eps: config.batchNormEps
        )

        maskFeatures = maskFeatures + x4Lateral

        maskFeatures = try PPDocLayoutMLXTensorOps.convLayer(
            maskFeatures,
            store: weights,
            prefix: "model.encoder.encoder_mask_output.base_conv",
            activation: "silu",
            eps: config.batchNormEps
        )

        let outputConvWeight = try weights.tensor("model.encoder.encoder_mask_output.conv.weight")
        let outputConvBias = try weights.tensor("model.encoder.encoder_mask_output.conv.bias")
        maskFeatures = PPDocLayoutMLXTensorOps.conv2dNCHW(
            maskFeatures,
            weight: outputConvWeight,
            bias: outputConvBias,
            stride: 1,
            padding: 0
        )

        return PPDocLayoutEncoderOutput(
            panFeatures: panFeatureMaps,
            maskFeatures: maskFeatures
        )
    }

    private func aifi(_ x: MLXArray, aifiIndex: Int) throws -> MLXArray {
        let batch = x.shape[0]
        let channels = x.shape[1]
        let height = x.shape[2]
        let width = x.shape[3]

        var hidden = x.reshaped(batch, channels, height * width).transposed(0, 2, 1)
        let positionEmbedding = sinePositionEmbedding(
            width: width,
            height: height,
            embedDim: config.encoderHiddenDim
        )

        for layerIndex in 0..<config.encoderLayers {
            let prefix = "model.encoder.encoder.\(aifiIndex).layers.\(layerIndex)"
            hidden = try encoderLayer(
                hidden,
                positionEmbedding: positionEmbedding,
                prefix: prefix
            )
        }

        return hidden.transposed(0, 2, 1).reshaped(batch, channels, height, width)
    }

    private func encoderLayer(
        _ hiddenStates: MLXArray,
        positionEmbedding: MLXArray,
        prefix: String
    ) throws -> MLXArray {
        let residual = hiddenStates

        var hidden = try selfAttention(
            hiddenStates,
            positionEmbedding: positionEmbedding,
            prefix: "\(prefix).self_attn",
            numHeads: config.encoderAttentionHeads
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

        let mlpResidual = hidden

        let fc1Weight = try weights.tensor("\(prefix).fc1.weight")
        let fc1Bias = try weights.tensor("\(prefix).fc1.bias")
        hidden = PPDocLayoutMLXTensorOps.linear(hidden, weight: fc1Weight, bias: fc1Bias)
        hidden = PPDocLayoutMLXTensorOps.activation(hidden, name: config.encoderActivationFunction)

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
        positionEmbedding: MLXArray?,
        prefix: String,
        numHeads: Int
    ) throws -> MLXArray {
        let batch = hiddenStates.shape[0]
        let sequenceLength = hiddenStates.shape[1]
        let hiddenSize = hiddenStates.shape[2]
        let headDim = hiddenSize / max(1, numHeads)

        let queryInput: MLXArray
        if let positionEmbedding {
            queryInput = hiddenStates + positionEmbedding
        } else {
            queryInput = hiddenStates
        }

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

    private func cspRepLayer(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var hidden1 = try PPDocLayoutMLXTensorOps.convNormAct(
            x,
            store: weights,
            prefix: "\(prefix).conv1",
            activation: config.activationFunction,
            eps: config.batchNormEps
        )

        let hidden2 = try PPDocLayoutMLXTensorOps.convNormAct(
            x,
            store: weights,
            prefix: "\(prefix).conv2",
            activation: config.activationFunction,
            eps: config.batchNormEps
        )

        for bottleneckIndex in 0..<3 {
            hidden1 = try repVggBlock(
                hidden1,
                prefix: "\(prefix).bottlenecks.\(bottleneckIndex)"
            )
        }

        return hidden1 + hidden2
    }

    private func repVggBlock(_ x: MLXArray, prefix: String) throws -> MLXArray {
        let conv1 = try PPDocLayoutMLXTensorOps.convNormAct(
            x,
            store: weights,
            prefix: "\(prefix).conv1",
            activation: nil,
            eps: config.batchNormEps
        )

        let conv2 = try PPDocLayoutMLXTensorOps.convNormAct(
            x,
            store: weights,
            prefix: "\(prefix).conv2",
            activation: nil,
            eps: config.batchNormEps
        )

        return PPDocLayoutMLXTensorOps.activation(conv1 + conv2, name: config.activationFunction)
    }

    private func maskFeatureHead(_ panFeatureMaps: [MLXArray]) throws -> MLXArray {
        let strides = config.featStrides
        let reorderIndex = strides.enumerated().sorted { $0.element < $1.element }.map(\.offset)

        let reordered = reorderIndex.map { panFeatureMaps[$0] }
        guard !reordered.isEmpty else {
            throw PPDocLayoutMLXError.modelInitializationFailed("mask feature head received no feature maps")
        }

        var output = try scaleHead(
            reordered[0],
            scaleIndex: 0,
            fpnStride: strides[reorderIndex[0]],
            baseStride: strides[reorderIndex[0]]
        )

        for i in 1..<reordered.count {
            let stride = strides[reorderIndex[i]]
            var scaled = try scaleHead(
                reordered[i],
                scaleIndex: i,
                fpnStride: stride,
                baseStride: strides[reorderIndex[0]]
            )

            scaled = PPDocLayoutMLXTensorOps.upsampleBilinearNCHW(
                scaled,
                toHeight: output.shape[2],
                toWidth: output.shape[3],
                alignCorners: false
            )

            output = output + scaled
        }

        output = try PPDocLayoutMLXTensorOps.convLayer(
            output,
            store: weights,
            prefix: "model.encoder.mask_feature_head.output_conv",
            activation: "silu",
            eps: config.batchNormEps
        )

        return output
    }

    private func scaleHead(
        _ x: MLXArray,
        scaleIndex: Int,
        fpnStride: Int,
        baseStride: Int
    ) throws -> MLXArray {
        let headLength = max(1, Int(log2(Double(fpnStride)) - log2(Double(baseStride))))
        var hidden = x

        for layerIndex in 0..<headLength {
            let convLayerIndex = layerIndex * 2
            hidden = try PPDocLayoutMLXTensorOps.convLayer(
                hidden,
                store: weights,
                prefix: "model.encoder.mask_feature_head.scale_heads.\(scaleIndex).layers.\(convLayerIndex)",
                activation: "silu",
                eps: config.batchNormEps
            )

            if fpnStride != baseStride {
                hidden = PPDocLayoutMLXTensorOps.upsampleBilinearNCHW(
                    hidden,
                    scale: 2,
                    alignCorners: false
                )
            }
        }

        return hidden
    }

    private func sinePositionEmbedding(width: Int, height: Int, embedDim: Int) -> MLXArray {
        precondition(embedDim % 4 == 0, "embedDim must be divisible by 4")

        let posDim = embedDim / 4
        var omega = [Float](repeating: 0, count: posDim)
        for i in 0..<posDim {
            let ratio = Float(i) / Float(posDim)
            omega[i] = 1.0 / pow(Float(config.positionalEncodingTemperature), ratio)
        }

        let tokenCount = height * width
        var values = [Float](repeating: 0, count: tokenCount * embedDim)

        for y in 0..<height {
            for x in 0..<width {
                let token = (y * width) + x
                let base = token * embedDim

                for i in 0..<posDim {
                    let hValue = Float(y) * omega[i]
                    let wValue = Float(x) * omega[i]

                    values[base + i] = sin(hValue)
                    values[base + posDim + i] = cos(hValue)
                    values[base + (2 * posDim) + i] = sin(wValue)
                    values[base + (3 * posDim) + i] = cos(wValue)
                }
            }
        }

        return MLXArray(values).reshaped(1, tokenCount, embedDim)
    }
}
