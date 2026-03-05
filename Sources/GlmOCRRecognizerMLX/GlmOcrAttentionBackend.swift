import Foundation
import MLX

internal nonisolated enum GlmOcrAttentionBackend: Sendable, Equatable {
    case fast
    case fallback
}

internal nonisolated enum GlmOcrAttentionRuntime {
    internal nonisolated static func resolveBackend(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> GlmOcrAttentionBackend {
        if isTruthy(environment["GLMOCR_FORCE_FAST_SDPA"]) {
            return .fast
        }
        if isMetalValidationEnabled(environment: environment) {
            return .fallback
        }
        return .fast
    }

    internal nonisolated static func isMetalValidationEnabled(environment: [String: String]) -> Bool {
        isTruthy(environment["MTL_DEBUG_LAYER"]) || isTruthy(environment["METAL_DEVICE_WRAPPER_TYPE"])
    }

    private nonisolated static func isTruthy(_ value: String?) -> Bool {
        guard let value else {
            return false
        }
        switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "y", "on":
            return true
        default:
            return false
        }
    }
}

private let defaultAttentionBackend = GlmOcrAttentionRuntime.resolveBackend()

internal func glmOcrScaledDotProductAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    backend: GlmOcrAttentionBackend? = nil
) -> MLXArray {
    let queryLength = queries.dim(2)
    let keyLength = keys.dim(2)
    if queryLength == 0 || keyLength == 0 {
        let outputShape = [queries.dim(0), queries.dim(1), queryLength, values.dim(3)]
        return MLXArray.zeros(outputShape, dtype: values.dtype)
    }

    let resolvedBackend = backend ?? defaultAttentionBackend
    switch resolvedBackend {
    case .fast:
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    case .fallback:
        return glmOcrScaledDotProductAttentionFallback(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
}

private func glmOcrScaledDotProductAttentionFallback(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let numQueryHeads = queries.dim(1)
    let numKVHeads = keys.dim(1)
    let gqaFactor = max(numQueryHeads / max(numKVHeads, 1), 1)

    var queryWork = queries * scale
    var keyWork = keys
    var valueWork = values

    if gqaFactor > 1 {
        queryWork = unflatten(queryWork, axis: 1, shape: [numKVHeads, gqaFactor])
        keyWork = expandedDimensions(keyWork, axis: 2)
        valueWork = expandedDimensions(valueWork, axis: 2)
    }

    var scores = matmul(
        queryWork,
        keyWork.swappedAxes(-1, -2)
    )

    if var resolvedMask = resolvedMaskArray(
        mode: mask,
        queryLength: queryWork.dim(-2),
        keyLength: keyWork.dim(-2)
    ) {
        if gqaFactor > 1, resolvedMask.ndim >= 3 {
            if resolvedMask.dim(-3) == 1 {
                resolvedMask = expandedDimensions(resolvedMask, axis: -3)
            } else {
                resolvedMask = unflatten(
                    resolvedMask,
                    axis: -3,
                    shape: [numKVHeads, gqaFactor]
                )
            }
        }

        if resolvedMask.dtype == .bool {
            let negativeInfinity = MLXArray(-Float.infinity, dtype: scores.dtype)
            scores = `where`(resolvedMask, scores, negativeInfinity)
        } else {
            scores = scores + resolvedMask.asType(scores.dtype)
        }
    }

    scores = softmax(scores, axis: -1, precise: true)
    var output = matmul(scores, valueWork)
    if gqaFactor > 1 {
        output = flatten(output, startAxis: 1, endAxis: 2)
    }

    return output
}

private func resolvedMaskArray(
    mode: MLXFast.ScaledDotProductAttentionMaskMode,
    queryLength: Int,
    keyLength: Int
) -> MLXArray? {
    switch mode {
    case .none:
        return nil
    case .causal:
        return causalMask(queryLength: queryLength, keyLength: keyLength)
    case .array(let array):
        return array
    case .arrays(let arrays):
        return arrays.first
    }
}

private func causalMask(queryLength: Int, keyLength: Int) -> MLXArray {
    let offset = keyLength - queryLength
    let queryPositions = expandedDimensions(MLXArray(offset..<(offset + queryLength)), axis: 1)
    let keyPositions = expandedDimensions(MLXArray(0..<keyLength), axis: 0)
    return queryPositions .>= keyPositions
}
