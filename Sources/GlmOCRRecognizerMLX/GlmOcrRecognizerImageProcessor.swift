import Accelerate
import CoreGraphics
import CoreImage
import Foundation
import MLX

internal struct GlmOcrPatchifyOutput: @unchecked Sendable {
    let flattenedPatches: MLXArray
    let gridTHW: GlmOcrTHW
}

internal enum GlmOcrRecognizerImageProcessor {
    private static func interpolationQualityFromEnvironment() -> CGInterpolationQuality {
        let raw = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_INTERPOLATION"]?.lowercased()
        switch raw {
        case "none":
            return .none
        case "low":
            return .low
        case "medium":
            return .medium
        case "high":
            return .high
        default:
            return .high
        }
    }

    internal static func smartResize(
        t: Int,
        h: Int,
        w: Int,
        tFactor: Int = 2,
        hFactor: Int = 28,
        wFactor: Int = 28,
        minPixels: Int = 112 * 112,
        maxPixels: Int = 14 * 14 * 2 * 2 * 2 * 6_144
    ) throws -> (height: Int, width: Int) {
        guard t >= tFactor else {
            throw GlmOcrRecognizerMLXError.processingFailed(
                "Temporal dimension \(t) must be >= factor \(tFactor)"
            )
        }

        var height = h
        var width = w

        if height < hFactor || width < wFactor {
            let scale = max(Double(hFactor) / Double(height), Double(wFactor) / Double(width))
            height = Int(Double(height) * scale)
            width = Int(Double(width) * scale)
        }

        if Double(max(height, width)) / Double(min(height, width)) > 200 {
            throw GlmOcrRecognizerMLXError.processingFailed(
                "absolute aspect ratio must be <= 200"
            )
        }

        var hBar = Int((Double(height) / Double(hFactor)).rounded()) * hFactor
        var wBar = Int((Double(width) / Double(wFactor)).rounded()) * wFactor
        let tBar = Int((Double(t) / Double(tFactor)).rounded()) * tFactor

        if tBar * hBar * wBar > maxPixels {
            let beta = sqrt(Double(t * height * width) / Double(maxPixels))
            hBar = max(hFactor, Int(floor(Double(height) / beta / Double(hFactor))) * hFactor)
            wBar = max(wFactor, Int(floor(Double(width) / beta / Double(wFactor))) * wFactor)
        } else if tBar * hBar * wBar < minPixels {
            let beta = sqrt(Double(minPixels) / Double(t * height * width))
            hBar = Int(ceil(Double(height) * beta / Double(hFactor))) * hFactor
            wBar = Int(ceil(Double(width) * beta / Double(wFactor))) * wFactor
        }

        return (hBar, wBar)
    }

    internal static func normalizedPixelValues(
        from image: CGImage,
        targetWidth: Int,
        targetHeight: Int,
        mean: [Float],
        std: [Float]
    ) throws -> MLXArray {
        guard mean.count >= 3, std.count >= 3 else {
            throw GlmOcrRecognizerMLXError.processingFailed(
                "image_mean/image_std must have at least 3 channels"
            )
        }

        let pixelCount = targetWidth * targetHeight
        let raw = try resizedRGBABuffer(
            from: image,
            targetWidth: targetWidth,
            targetHeight: targetHeight
        )

        if ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIXEL_TRACE"] == "1" {
            let rawRGB = stride(from: 0, to: raw.count, by: 4).flatMap { offset in
                [Float(raw[offset]), Float(raw[offset + 1]), Float(raw[offset + 2])]
            }
            let prefix = Array(rawRGB.prefix(8))
            let sum = rawRGB.reduce(Float(0), +)
            let payload =
                "[GlmOcrRecognizerImageProcessor] resizedRGB count=\(rawRGB.count) sum=\(sum) prefix=\(prefix)\n"
            let data = payload.data(using: .utf8) ?? Data()
            FileHandle.standardError.write(data)
        }

        var chw = [Float](repeating: 0, count: pixelCount * 3)
        let channelStride = pixelCount

        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                let pixelIndex = (y * targetWidth + x)
                let base = pixelIndex * 4

                let r = Float(raw[base]) / 255
                let g = Float(raw[base + 1]) / 255
                let b = Float(raw[base + 2]) / 255

                chw[pixelIndex] = (r - mean[0]) / std[0]
                chw[channelStride + pixelIndex] = (g - mean[1]) / std[1]
                chw[(2 * channelStride) + pixelIndex] = (b - mean[2]) / std[2]
            }
        }

        return MLXArray(chw).reshaped(1, 3, targetHeight, targetWidth)
    }

    private static func resizedRGBABuffer(
        from image: CGImage,
        targetWidth: Int,
        targetHeight: Int
    ) throws -> [UInt8] {
        let sourceWidth = image.width
        let sourceHeight = image.height
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw GlmOcrRecognizerMLXError.processingFailed("Input image has invalid dimensions")
        }

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        var source = [UInt8](repeating: 0, count: sourceWidth * sourceHeight * 4)
        guard
            let sourceContext = CGContext(
                data: &source,
                width: sourceWidth,
                height: sourceHeight,
                bitsPerComponent: 8,
                bytesPerRow: sourceWidth * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        else {
            throw GlmOcrRecognizerMLXError.processingFailed("Unable to create source bitmap context")
        }

        sourceContext.interpolationQuality = .none
        sourceContext.draw(image, in: CGRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight))

        if sourceWidth == targetWidth, sourceHeight == targetHeight {
            return source
        }

        let aspectRatio = Double(sourceWidth) / Double(max(sourceHeight, 1))
        let preferVImage = sourceHeight <= 24 || aspectRatio >= 8.0

        if !preferVImage,
            let coreImageResized = resizedWithCoreImageBicubic(
                image: image,
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                targetWidth: targetWidth,
                targetHeight: targetHeight,
                colorSpace: colorSpace
            )
        {
            return coreImageResized
        }

        let useHighQuality = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_VIMAGE_HQ"] != "0"
        let flags = useHighQuality ? vImage_Flags(kvImageHighQualityResampling) : vImage_Flags(0)
        var destination = [UInt8](repeating: 0, count: targetWidth * targetHeight * 4)
        let status: vImage_Error = source.withUnsafeMutableBytes { sourceBytes in
            destination.withUnsafeMutableBytes { destinationBytes in
                var srcBuffer = vImage_Buffer(
                    data: sourceBytes.baseAddress,
                    height: vImagePixelCount(sourceHeight),
                    width: vImagePixelCount(sourceWidth),
                    rowBytes: sourceWidth * 4
                )
                var dstBuffer = vImage_Buffer(
                    data: destinationBytes.baseAddress,
                    height: vImagePixelCount(targetHeight),
                    width: vImagePixelCount(targetWidth),
                    rowBytes: targetWidth * 4
                )
                return vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, flags)
            }
        }
        guard status == kvImageNoError else {
            throw GlmOcrRecognizerMLXError.processingFailed(
                "vImageScale_ARGB8888 failed with status \(status)"
            )
        }

        if ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIXEL_TRACE"] == "1" {
            let payload = "[GlmOcrRecognizerImageProcessor] resizeBackend=vimage\n"
            let data = payload.data(using: .utf8) ?? Data()
            FileHandle.standardError.write(data)
        }

        return destination
    }

    private static func resizedWithCoreImageBicubic(
        image: CGImage,
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int,
        colorSpace: CGColorSpace
    ) -> [UInt8]? {
        guard let filter = CIFilter(name: "CILanczosScaleTransform") else {
            return nil
        }

        let ciImage = CIImage(cgImage: image)
        let scaleY = CGFloat(targetHeight) / CGFloat(sourceHeight)
        let scaleX = CGFloat(targetWidth) / CGFloat(sourceWidth)
        let aspectRatio = scaleX / scaleY

        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(scaleY, forKey: kCIInputScaleKey)
        filter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)

        guard let output = filter.outputImage else {
            return nil
        }

        let context = CIContext(options: [
            CIContextOption.workingColorSpace: colorSpace,
            CIContextOption.outputColorSpace: colorSpace,
        ])

        let outputExtent = output.extent.integral
        guard let resizedImage = context.createCGImage(output, from: outputExtent) else {
            return nil
        }

        var raw = [UInt8](repeating: 0, count: targetWidth * targetHeight * 4)
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard
            let bitmapContext = CGContext(
                data: &raw,
                width: targetWidth,
                height: targetHeight,
                bitsPerComponent: 8,
                bytesPerRow: targetWidth * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        else {
            return nil
        }

        let rect = CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight)
        bitmapContext.interpolationQuality = .none
        bitmapContext.draw(resizedImage, in: rect)
        if ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIXEL_TRACE"] == "1" {
            let payload = "[GlmOcrRecognizerImageProcessor] resizeBackend=coreimage_lanczos\n"
            let data = payload.data(using: .utf8) ?? Data()
            FileHandle.standardError.write(data)
        }
        return raw
    }

    internal static func patchify(
        imageTensor: MLXArray,
        mergeSize: Int,
        patchSize: Int,
        temporalPatchSize: Int
    ) throws -> GlmOcrPatchifyOutput {
        guard imageTensor.ndim == 4 else {
            throw GlmOcrRecognizerMLXError.processingFailed(
                "Expected image tensor rank 4 [N,C,H,W], got rank \(imageTensor.ndim)"
            )
        }

        var patches = imageTensor
        let remainder = patches.dim(0) % temporalPatchSize

        if remainder != 0 {
            let last = patches[-1, .ellipsis]
            let repeatedLast = tiled(
                last[.newAxis, 0..., 0..., 0...],
                repetitions: [temporalPatchSize - remainder, 1, 1, 1]
            )
            patches = concatenated([patches, repeatedLast], axis: 0)
        }

        let channels = patches.dim(1)
        let resizedHeight = patches.dim(2)
        let resizedWidth = patches.dim(3)

        let gridT = patches.dim(0) / temporalPatchSize
        let gridH = resizedHeight / patchSize
        let gridW = resizedWidth / patchSize

        patches = patches.reshaped(
            gridT,
            temporalPatchSize,
            channels,
            gridH / mergeSize,
            mergeSize,
            patchSize,
            gridW / mergeSize,
            mergeSize,
            patchSize
        )

        patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)

        let flattened = patches.reshaped(
            gridT * gridH * gridW,
            channels * temporalPatchSize * patchSize * patchSize
        )

        return GlmOcrPatchifyOutput(
            flattenedPatches: flattened,
            gridTHW: GlmOcrTHW(t: gridT, h: gridH, w: gridW)
        )
    }

}
