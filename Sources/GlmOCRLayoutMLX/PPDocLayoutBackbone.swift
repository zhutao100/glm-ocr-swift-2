import Foundation
import MLX

internal struct PPDocLayoutBackboneOutput: @unchecked Sendable {
    internal let x4Feature: MLXArray
    internal let stageFeatures: [MLXArray]
}

internal struct PPDocLayoutBackbone: Sendable {
    private static let stageInChannels = [48, 128, 512, 1024]
    private static let stageMidChannels = [48, 96, 192, 384]
    private static let stageOutChannels = [128, 512, 1024, 2048]
    private static let stageNumBlocks = [1, 1, 3, 1]
    private static let stageDownsample = [false, true, true, true]
    private static let stageLightBlock = [false, false, true, true]
    private static let stageKernelSize = [3, 3, 5, 5]
    private static let stageNumLayers = [6, 6, 6, 6]

    private let config: PPDocLayoutV3Configuration
    private let weights: PPDocLayoutWeightStore

    internal init(
        config: PPDocLayoutV3Configuration,
        weights: PPDocLayoutWeightStore
    ) {
        self.config = config
        self.weights = weights
    }

    internal func forward(_ pixelValues: MLXArray) throws -> PPDocLayoutBackboneOutput {
        var hidden = try stem(pixelValues)
        var stageFeatures: [MLXArray] = []
        stageFeatures.reserveCapacity(4)

        for stageIndex in 0..<4 {
            hidden = try stageForward(hidden, stageIndex: stageIndex)
            stageFeatures.append(hidden)
        }

        guard let x4Feature = stageFeatures.first else {
            throw PPDocLayoutMLXError.modelInitializationFailed("Backbone produced no stage features")
        }

        return PPDocLayoutBackboneOutput(
            x4Feature: x4Feature,
            stageFeatures: stageFeatures
        )
    }

    private func stem(_ x: MLXArray) throws -> MLXArray {
        var hidden = try PPDocLayoutMLXTensorOps.convLayer(
            x,
            store: weights,
            prefix: "model.backbone.model.embedder.stem1",
            stride: 2,
            activation: "relu",
            eps: config.batchNormEps
        )

        hidden = PPDocLayoutMLXTensorOps.paddedNCHW(
            hidden,
            left: 0,
            right: 1,
            top: 0,
            bottom: 1
        )

        var stem2 = try PPDocLayoutMLXTensorOps.convLayer(
            hidden,
            store: weights,
            prefix: "model.backbone.model.embedder.stem2a",
            stride: 1,
            padding: 0,
            activation: "relu",
            eps: config.batchNormEps
        )

        stem2 = PPDocLayoutMLXTensorOps.paddedNCHW(
            stem2,
            left: 0,
            right: 1,
            top: 0,
            bottom: 1
        )

        stem2 = try PPDocLayoutMLXTensorOps.convLayer(
            stem2,
            store: weights,
            prefix: "model.backbone.model.embedder.stem2b",
            stride: 1,
            padding: 0,
            activation: "relu",
            eps: config.batchNormEps
        )

        let pooled = PPDocLayoutMLXTensorOps.maxPool2dNCHW(hidden, kernel: 2, stride: 1)
        hidden = concatenated([pooled, stem2], axis: 1)

        hidden = try PPDocLayoutMLXTensorOps.convLayer(
            hidden,
            store: weights,
            prefix: "model.backbone.model.embedder.stem3",
            stride: 2,
            activation: "relu",
            eps: config.batchNormEps
        )

        hidden = try PPDocLayoutMLXTensorOps.convLayer(
            hidden,
            store: weights,
            prefix: "model.backbone.model.embedder.stem4",
            stride: 1,
            activation: "relu",
            eps: config.batchNormEps
        )

        return hidden
    }

    private func stageForward(_ x: MLXArray, stageIndex: Int) throws -> MLXArray {
        var hidden = x

        if Self.stageDownsample[stageIndex] {
            hidden = try PPDocLayoutMLXTensorOps.convLayer(
                hidden,
                store: weights,
                prefix: "model.backbone.model.encoder.stages.\(stageIndex).downsample",
                stride: 2,
                groups: Self.stageInChannels[stageIndex],
                activation: nil,
                eps: config.batchNormEps
            )
        }

        for blockIndex in 0..<Self.stageNumBlocks[stageIndex] {
            hidden = try basicLayer(
                hidden,
                stageIndex: stageIndex,
                blockIndex: blockIndex,
                middleChannels: Self.stageMidChannels[stageIndex],
                numLayers: Self.stageNumLayers[stageIndex],
                lightBlock: Self.stageLightBlock[stageIndex],
                kernelSize: Self.stageKernelSize[stageIndex],
                residual: blockIndex != 0
            )
        }

        return hidden
    }

    private func basicLayer(
        _ x: MLXArray,
        stageIndex: Int,
        blockIndex: Int,
        middleChannels: Int,
        numLayers: Int,
        lightBlock: Bool,
        kernelSize: Int,
        residual: Bool
    ) throws -> MLXArray {
        let identity = x
        var features: [MLXArray] = [x]

        var hidden = x
        for layerIndex in 0..<numLayers {
            if lightBlock {
                hidden = try lightConvBlock(
                    hidden,
                    stageIndex: stageIndex,
                    blockIndex: blockIndex,
                    layerIndex: layerIndex,
                    kernelSize: kernelSize
                )
            } else {
                hidden = try PPDocLayoutMLXTensorOps.convLayer(
                    hidden,
                    store: weights,
                    prefix:
                        "model.backbone.model.encoder.stages.\(stageIndex).blocks.\(blockIndex).layers.\(layerIndex)",
                    stride: 1,
                    activation: "relu",
                    eps: config.batchNormEps
                )
            }

            features.append(hidden)
        }

        hidden = concatenated(features, axis: 1)

        hidden = try PPDocLayoutMLXTensorOps.convLayer(
            hidden,
            store: weights,
            prefix: "model.backbone.model.encoder.stages.\(stageIndex).blocks.\(blockIndex).aggregation.0",
            stride: 1,
            activation: "relu",
            eps: config.batchNormEps
        )

        hidden = try PPDocLayoutMLXTensorOps.convLayer(
            hidden,
            store: weights,
            prefix: "model.backbone.model.encoder.stages.\(stageIndex).blocks.\(blockIndex).aggregation.1",
            stride: 1,
            activation: "relu",
            eps: config.batchNormEps
        )

        if residual {
            hidden = hidden + identity
        }

        return hidden
    }

    private func lightConvBlock(
        _ x: MLXArray,
        stageIndex: Int,
        blockIndex: Int,
        layerIndex: Int,
        kernelSize: Int
    ) throws -> MLXArray {
        var hidden = try PPDocLayoutMLXTensorOps.convLayer(
            x,
            store: weights,
            prefix: "model.backbone.model.encoder.stages.\(stageIndex).blocks.\(blockIndex).layers.\(layerIndex).conv1",
            stride: 1,
            padding: 0,
            activation: nil,
            eps: config.batchNormEps
        )

        hidden = try PPDocLayoutMLXTensorOps.convLayer(
            hidden,
            store: weights,
            prefix: "model.backbone.model.encoder.stages.\(stageIndex).blocks.\(blockIndex).layers.\(layerIndex).conv2",
            stride: 1,
            padding: (kernelSize - 1) / 2,
            groups: middleChannelCount(hidden),
            activation: "relu",
            eps: config.batchNormEps
        )

        return hidden
    }

    private func middleChannelCount(_ x: MLXArray) -> Int {
        x.shape[1]
    }
}
