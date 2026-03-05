import Foundation
import MLX
import MLXNN

internal protocol GlmOcrKVCache: AnyObject {
    var offset: Int { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
}

internal final class GlmOcrSimpleKVCache: GlmOcrKVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset: Int = 0
    private let step: Int

    internal init(step: Int = 256) {
        self.step = max(step, 1)
    }

    internal func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        let newLength = keys.dim(2)

        let shouldGrow: Bool
        if let currentKeys = self.keys {
            shouldGrow = previous + newLength > currentKeys.dim(2)
        } else {
            shouldGrow = true
        }

        if shouldGrow {
            let batch = keys.dim(0)
            let kvHeads = keys.dim(1)
            let keyHeadDim = keys.dim(3)
            let valueHeadDim = values.dim(3)
            let nSteps = (step + newLength - 1) / step

            let newKeys = MLXArray.zeros([batch, kvHeads, nSteps * step, keyHeadDim], dtype: keys.dtype)
            let newValues = MLXArray.zeros([batch, kvHeads, nSteps * step, valueHeadDim], dtype: values.dtype)

            if var existingKeys = self.keys, var existingValues = self.values {
                if previous % step != 0 {
                    existingKeys = existingKeys[.ellipsis, ..<previous, 0...]
                    existingValues = existingValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([existingKeys, newKeys], axis: 2)
                self.values = concatenated([existingValues, newValues], axis: 2)
            } else {
                self.keys = newKeys
                self.values = newValues
            }
        }

        offset += newLength
        self.keys?[.ellipsis, previous..<offset, 0...] = keys
        self.values?[.ellipsis, previous..<offset, 0...] = values

        let cachedKeys = self.keys![.ellipsis, ..<offset, 0...]
        let cachedValues = self.values![.ellipsis, ..<offset, 0...]
        return (cachedKeys, cachedValues)
    }
}

internal func glmOcrCreateCausalMask(n: Int, offset: Int) -> MLXArray {
    var right = MLXArray(Int32(0)..<Int32(offset + n))
    var left = offset == 0 ? right : MLXArray(Int32(offset)..<Int32(offset + n))
    left = left[0..., .newAxis]
    right = right[.newAxis]
    return left .>= right
}

internal func glmOcrCreateAttentionMask(
    hiddenStates: MLXArray,
    cache: GlmOcrKVCache?
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let tokenCount = hiddenStates.dim(1)
    guard tokenCount > 1 else {
        return .none
    }

    let offset = cache?.offset ?? 0
    if offset == 0 {
        return .causal
    }

    return .array(glmOcrCreateCausalMask(n: tokenCount, offset: offset))
}

internal func glmOcrAttentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: GlmOcrKVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    if let cache {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return glmOcrScaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }

    return glmOcrScaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: mask
    )
}
