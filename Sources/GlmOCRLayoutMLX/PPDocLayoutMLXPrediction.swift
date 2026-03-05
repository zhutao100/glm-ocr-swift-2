import Foundation
import MLX

// MLXArray is not Sendable; prediction values are actor-confined in runtime usage.
internal struct PPDocLayoutMLXPrediction: @unchecked Sendable {
    internal let logits: MLXArray
    internal let predBoxes: MLXArray
    internal let orderLogits: MLXArray
    internal let outMasks: MLXArray

    internal func outputShapes() -> [String: [Int]] {
        [
            PPDocLayoutMLXContract.logitsOutputName: logits.shape,
            PPDocLayoutMLXContract.predBoxesOutputName: predBoxes.shape,
            PPDocLayoutMLXContract.orderLogitsOutputName: orderLogits.shape,
            PPDocLayoutMLXContract.outMasksOutputName: outMasks.shape,
        ]
    }
}
