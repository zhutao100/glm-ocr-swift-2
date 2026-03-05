import CoreGraphics
import Foundation
import MLX

internal enum PPDocLayoutMLXImageProcessor {
    private nonisolated static let traceEnabled =
        ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIPELINE_TRACE"] == "1"

    internal static func pixelValues(from image: CGImage) throws -> MLXArray {
        let width = PPDocLayoutMLXContract.inputShape[3]
        let height = PPDocLayoutMLXContract.inputShape[2]
        trace("pixelValues.start source=\(image.width)x\(image.height) target=\(width)x\(height)")

        var rgba = [UInt8](repeating: 0, count: width * height * 4)

        let success = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let baseAddress = bytes.baseAddress else {
                return false
            }

            let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
            guard
                let context = CGContext(
                    data: baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: width * 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: bitmapInfo
                )
            else {
                return false
            }

            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }

        guard success else {
            throw PPDocLayoutMLXError.preprocessFailed("Unable to create RGB context for layout preprocessing")
        }

        let hw = width * height
        var chw = [Float](repeating: 0, count: hw * 3)

        for offset in 0..<hw {
            let rgbaOffset = offset * 4
            chw[offset] = Float(rgba[rgbaOffset]) / 255.0
            chw[hw + offset] = Float(rgba[rgbaOffset + 1]) / 255.0
            chw[(2 * hw) + offset] = Float(rgba[rgbaOffset + 2]) / 255.0
        }

        let pixelValues = MLXArray(chw).reshaped(
            PPDocLayoutMLXContract.inputShape[0],
            PPDocLayoutMLXContract.inputShape[1],
            PPDocLayoutMLXContract.inputShape[2],
            PPDocLayoutMLXContract.inputShape[3]
        )
        trace("pixelValues.done shape=\(pixelValues.shape)")
        return pixelValues
    }

    private nonisolated static func trace(_ message: String) {
        guard traceEnabled else {
            return
        }
        let payload = "[PPDocLayoutMLXImageProcessor] \(message)\n"
        let data = payload.data(using: .utf8) ?? Data()
        FileHandle.standardError.write(data)
    }
}
