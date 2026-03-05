import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

internal protocol PipelineRegionCropping: Sendable {
    func cropRegion(
        page: CGImage,
        bbox2D: [Int],
        polygon2D: [[Int]],
        pageIndex: Int,
        regionIndex: Int
    ) throws -> PipelineRegionCropResult
}

internal enum PipelineRegionCropperError: Error, Sendable, Equatable {
    case invalidBoundingBox([Int])
    case cropFailure
    case maskFailure
}

internal struct PipelineRegionCropper: PipelineRegionCropping {
    internal init() {}

    internal func cropRegion(
        page: CGImage,
        bbox2D: [Int],
        polygon2D: [[Int]],
        pageIndex: Int,
        regionIndex: Int
    ) throws -> PipelineRegionCropResult {
        let width = page.width
        let height = page.height

        let coordinates = try pixelCoordinates(
            from: bbox2D,
            width: width,
            height: height
        )

        let cropRect = CGRect(
            x: coordinates.x1,
            y: coordinates.y1,
            width: coordinates.x2 - coordinates.x1,
            height: coordinates.y2 - coordinates.y1
        )

        guard let cropped = page.cropping(to: cropRect) else {
            throw PipelineRegionCropperError.cropFailure
        }

        guard polygon2D.count >= 3 else {
            dumpIfRequested(image: cropped, pageIndex: pageIndex, regionIndex: regionIndex)
            return PipelineRegionCropResult(image: cropped, warning: nil)
        }

        guard
            let masked = mask(
                cropped: cropped,
                originalWidth: width,
                originalHeight: height,
                cropOriginX: coordinates.x1,
                cropOriginY: coordinates.y1,
                polygon2D: polygon2D
            )
        else {
            dumpIfRequested(image: cropped, pageIndex: pageIndex, regionIndex: regionIndex)
            return PipelineRegionCropResult(image: cropped, warning: nil)
        }

        dumpIfRequested(image: masked, pageIndex: pageIndex, regionIndex: regionIndex)
        return PipelineRegionCropResult(image: masked, warning: nil)
    }

    private func pixelCoordinates(
        from bbox2D: [Int],
        width: Int,
        height: Int
    ) throws -> (x1: Int, y1: Int, x2: Int, y2: Int) {
        guard bbox2D.count == 4 else {
            throw PipelineRegionCropperError.invalidBoundingBox(bbox2D)
        }

        let x1 = clamp(
            denormalize(value: bbox2D[0], size: width),
            minimum: 0,
            maximum: width
        )
        let y1 = clamp(
            denormalize(value: bbox2D[1], size: height),
            minimum: 0,
            maximum: height
        )
        let x2 = clamp(
            denormalize(value: bbox2D[2], size: width),
            minimum: 0,
            maximum: width
        )
        let y2 = clamp(
            denormalize(value: bbox2D[3], size: height),
            minimum: 0,
            maximum: height
        )

        guard x1 < x2, y1 < y2 else {
            throw PipelineRegionCropperError.invalidBoundingBox(bbox2D)
        }

        return (x1, y1, x2, y2)
    }

    private func mask(
        cropped: CGImage,
        originalWidth: Int,
        originalHeight: Int,
        cropOriginX: Int,
        cropOriginY: Int,
        polygon2D: [[Int]]
    ) -> CGImage? {
        let cropWidth = cropped.width
        let cropHeight = cropped.height
        guard cropWidth > 0, cropHeight > 0 else {
            return nil
        }

        guard let croppedRGB = rgbBytes(from: cropped) else {
            return nil
        }

        let points = polygon2D.compactMap { point -> CGPoint? in
            guard point.count >= 2 else {
                return nil
            }

            let x = denormalize(value: point[0], size: originalWidth) - cropOriginX
            let y = denormalize(value: point[1], size: originalHeight) - cropOriginY
            return CGPoint(x: x, y: y)
        }

        guard points.count >= 3 else {
            return nil
        }

        let maskBuffer = buildPolygonMask(
            width: cropWidth,
            height: cropHeight,
            polygon: points
        )
        guard !maskBuffer.isEmpty else {
            return nil
        }

        var outputRGB = [UInt8](repeating: 255, count: cropWidth * cropHeight * 3)
        for pixelIndex in 0..<(cropWidth * cropHeight) where maskBuffer[pixelIndex] != 0 {
            let base = pixelIndex * 3
            outputRGB[base] = croppedRGB[base]
            outputRGB[base + 1] = croppedRGB[base + 1]
            outputRGB[base + 2] = croppedRGB[base + 2]
        }

        return makeImageFromRGB(bytes: outputRGB, width: cropWidth, height: cropHeight)
    }

    private func buildPolygonMask(
        width: Int,
        height: Int,
        polygon: [CGPoint]
    ) -> [UInt8] {
        guard polygon.count >= 3 else {
            return []
        }

        var mask = [UInt8](repeating: 0, count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        guard
            let context = CGContext(
                data: &mask,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        else {
            return []
        }

        context.setShouldAntialias(false)
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        let path = CGMutablePath()
        path.addLines(between: polygon)
        path.closeSubpath()

        context.addPath(path)
        context.setFillColor(gray: 1, alpha: 1)
        context.fillPath()

        for idx in mask.indices where mask[idx] != 0 {
            mask[idx] = 1
        }

        return mask
    }

    private func rgbBytes(from image: CGImage) -> [UInt8]? {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else {
            return nil
        }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        guard
            let context = CGContext(
                data: &rgba,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        else {
            return nil
        }

        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for idx in 0..<(width * height) {
            let rgbaOffset = idx * 4
            let rgbOffset = idx * 3
            rgb[rgbOffset] = rgba[rgbaOffset]
            rgb[rgbOffset + 1] = rgba[rgbaOffset + 1]
            rgb[rgbOffset + 2] = rgba[rgbaOffset + 2]
        }

        return rgb
    }

    private func makeImageFromRGB(bytes: [UInt8], width: Int, height: Int) -> CGImage? {
        guard bytes.count == width * height * 3 else {
            return nil
        }

        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for idx in 0..<(width * height) {
            let rgbOffset = idx * 3
            let rgbaOffset = idx * 4
            rgba[rgbaOffset] = bytes[rgbOffset]
            rgba[rgbaOffset + 1] = bytes[rgbOffset + 1]
            rgba[rgbaOffset + 2] = bytes[rgbOffset + 2]
        }

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        guard
            let context = CGContext(
                data: &rgba,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        else {
            return nil
        }

        return context.makeImage()
    }

    private func denormalize(value: Int, size: Int) -> Int {
        Int((Double(value) * Double(size)) / 1000.0)
    }

    private func clamp(_ value: Int, minimum: Int, maximum: Int) -> Int {
        min(max(value, minimum), maximum)
    }

    private func dumpIfRequested(
        image: CGImage,
        pageIndex: Int,
        regionIndex: Int
    ) {
        guard let directory = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_DUMP_CROPS_DIR"],
            !directory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return
        }

        let dirURL = URL(fileURLWithPath: directory, isDirectory: true)
        try? FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
        let millis = Int(Date().timeIntervalSince1970 * 1000)
        let fileURL = dirURL.appending(
            path: String(format: "t_%013d_page_%03d_region_%03d.png", millis, pageIndex, regionIndex))

        guard
            let destination = CGImageDestinationCreateWithURL(
                fileURL as CFURL,
                UTType.png.identifier as CFString,
                1,
                nil
            )
        else {
            return
        }

        CGImageDestinationAddImage(destination, image, nil)
        _ = CGImageDestinationFinalize(destination)
    }
}
