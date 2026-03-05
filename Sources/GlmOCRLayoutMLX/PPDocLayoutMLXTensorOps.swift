import Foundation
import MLX
import MLXNN

internal final class PPDocLayoutWeightStore: @unchecked Sendable {
    private let tensors: [String: MLXArray]

    internal init(tensors: [String: MLXArray]) {
        self.tensors = tensors
    }

    internal func tensor(_ key: String) throws -> MLXArray {
        guard let value = tensors[key] else {
            throw PPDocLayoutMLXError.missingModelFile("Missing weight tensor '\(key)'")
        }
        return value
    }

    internal func has(_ key: String) -> Bool {
        tensors[key] != nil
    }

    internal var count: Int {
        tensors.count
    }
}

internal enum PPDocLayoutMLXTensorOps {
    internal static func activation(_ x: MLXArray, name: String) -> MLXArray {
        switch name.lowercased() {
        case "relu":
            return maximum(x, 0)
        case "silu":
            return x * sigmoid(x)
        case "gelu":
            // tanh approximation used by many inference runtimes.
            let cubic = x * x * x
            let inner = x + (0.044_715 * cubic)
            let scaled = 0.797_884_56 * inner
            return 0.5 * x * (1.0 + tanh(scaled))
        default:
            return x
        }
    }

    internal static func sigmoid(_ x: MLXArray) -> MLXArray {
        1.0 / (1.0 + exp(-x))
    }

    internal static func inverseSigmoid(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
        let clipped = clip(x, min: eps, max: 1 - eps)
        return log(clipped / (1 - clipped))
    }

    internal static func linear(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray?
    ) -> MLXArray {
        var y = matmul(x, weight.T)
        if let bias {
            y = y + bias
        }
        return y
    }

    internal static func layerNorm(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        eps: Float = 1e-5
    ) -> MLXArray {
        let mu = mean(x, axis: -1, keepDims: true)
        let sigma2 = variance(x, axis: -1, keepDims: true)
        let normalized = (x - mu) * rsqrt(sigma2 + eps)
        return (normalized * weight) + bias
    }

    internal static func batchNorm2d(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        runningMean: MLXArray,
        runningVar: MLXArray,
        eps: Float = 1e-5
    ) -> MLXArray {
        let gamma = weight.reshaped(1, weight.shape[0], 1, 1)
        let beta = bias.reshaped(1, bias.shape[0], 1, 1)
        let meanTensor = runningMean.reshaped(1, runningMean.shape[0], 1, 1)
        let varTensor = runningVar.reshaped(1, runningVar.shape[0], 1, 1)

        let normalized = (x - meanTensor) * rsqrt(varTensor + eps)
        return (normalized * gamma) + beta
    }

    internal static func conv2dNCHW(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray? = nil,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1
    ) -> MLXArray {
        let xNHWC = x.transposed(0, 2, 3, 1)
        let weightHWIO = weight.transposed(0, 2, 3, 1)

        var y = conv2d(
            xNHWC,
            weightHWIO,
            stride: .init((stride, stride)),
            padding: .init((padding, padding)),
            groups: groups
        )

        if let bias {
            y = y + bias.reshaped(1, 1, 1, bias.shape[0])
        }

        return y.transposed(0, 3, 1, 2)
    }

    internal static func convNormAct(
        _ x: MLXArray,
        store: PPDocLayoutWeightStore,
        prefix: String,
        stride: Int = 1,
        padding: Int? = nil,
        groups: Int = 1,
        activation: String? = nil,
        eps: Float = 1e-5
    ) throws -> MLXArray {
        let convWeight = try store.tensor("\(prefix).conv.weight")
        let kernelHeight = convWeight.shape[2]
        let inferredPadding = (kernelHeight - 1) / 2

        var y = conv2dNCHW(
            x,
            weight: convWeight,
            stride: stride,
            padding: padding ?? inferredPadding,
            groups: groups
        )

        let bnWeight = try store.tensor("\(prefix).norm.weight")
        let bnBias = try store.tensor("\(prefix).norm.bias")
        let bnMean = try store.tensor("\(prefix).norm.running_mean")
        let bnVar = try store.tensor("\(prefix).norm.running_var")

        y = batchNorm2d(
            y,
            weight: bnWeight,
            bias: bnBias,
            runningMean: bnMean,
            runningVar: bnVar,
            eps: eps
        )

        if let activation {
            y = PPDocLayoutMLXTensorOps.activation(y, name: activation)
        }

        return y
    }

    internal static func convLayer(
        _ x: MLXArray,
        store: PPDocLayoutWeightStore,
        prefix: String,
        stride: Int = 1,
        padding: Int? = nil,
        groups: Int? = nil,
        activation: String? = nil,
        eps: Float = 1e-5
    ) throws -> MLXArray {
        let convWeight = try store.tensor("\(prefix).convolution.weight")
        let kernelHeight = convWeight.shape[2]
        let inferredPadding = (kernelHeight - 1) / 2
        let inputChannels = x.shape[1]
        let inferredGroups = max(1, inputChannels / max(1, convWeight.shape[1]))

        var y = conv2dNCHW(
            x,
            weight: convWeight,
            stride: stride,
            padding: padding ?? inferredPadding,
            groups: groups ?? inferredGroups
        )

        let bnWeight = try store.tensor("\(prefix).normalization.weight")
        let bnBias = try store.tensor("\(prefix).normalization.bias")
        let bnMean = try store.tensor("\(prefix).normalization.running_mean")
        let bnVar = try store.tensor("\(prefix).normalization.running_var")

        y = batchNorm2d(
            y,
            weight: bnWeight,
            bias: bnBias,
            runningMean: bnMean,
            runningVar: bnVar,
            eps: eps
        )

        if let activation {
            y = PPDocLayoutMLXTensorOps.activation(y, name: activation)
        }

        return y
    }

    internal static func paddedNCHW(
        _ x: MLXArray,
        left: Int,
        right: Int,
        top: Int,
        bottom: Int,
        value: Float = 0
    ) -> MLXArray {
        padded(
            x,
            widths: [
                0,
                0,
                .init((top, bottom)),
                .init((left, right)),
            ],
            mode: .constant,
            value: MLXArray(value)
        )
    }

    internal static func maxPool2dNCHW(
        _ x: MLXArray,
        kernel: Int,
        stride: Int
    ) -> MLXArray {
        let pool = MaxPool2d(kernelSize: .init(kernel), stride: .init(stride))
        let nhwc = x.transposed(0, 2, 3, 1)
        let pooled = pool(nhwc)
        return pooled.transposed(0, 3, 1, 2)
    }

    internal static func upsampleNearestNCHW(_ x: MLXArray, scale: Float) -> MLXArray {
        let upsample = Upsample(scaleFactor: [scale, scale], mode: .nearest)
        let nhwc = x.transposed(0, 2, 3, 1)
        let upsampled = upsample(nhwc)
        return upsampled.transposed(0, 3, 1, 2)
    }

    internal static func upsampleBilinearNCHW(
        _ x: MLXArray,
        scale: Float,
        alignCorners: Bool = false
    ) -> MLXArray {
        let upsample = Upsample(scaleFactor: [scale, scale], mode: .linear(alignCorners: alignCorners))
        let nhwc = x.transposed(0, 2, 3, 1)
        let upsampled = upsample(nhwc)
        return upsampled.transposed(0, 3, 1, 2)
    }

    internal static func upsampleBilinearNCHW(
        _ x: MLXArray,
        toHeight: Int,
        toWidth: Int,
        alignCorners: Bool = false
    ) -> MLXArray {
        let currentHeight = x.shape[2]
        let currentWidth = x.shape[3]

        if currentHeight == toHeight, currentWidth == toWidth {
            return x
        }

        let scaleH = Float(toHeight) / Float(currentHeight)
        let scaleW = Float(toWidth) / Float(currentWidth)

        let upsample = Upsample(scaleFactor: [scaleH, scaleW], mode: .linear(alignCorners: alignCorners))
        let nhwc = x.transposed(0, 2, 3, 1)
        let upsampled = upsample(nhwc)
        return upsampled.transposed(0, 3, 1, 2)
    }

    internal static func flattenHW(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let channels = x.shape[1]
        let height = x.shape[2]
        let width = x.shape[3]

        return x.reshaped(batch, channels, height * width).transposed(0, 2, 1)
    }

    internal static func ensureFloat32(_ x: MLXArray) -> MLXArray {
        x.asType(.float32)
    }
}
