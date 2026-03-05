import CPDFium
import CoreGraphics
import Foundation

public struct PDFiumPageRasterizer: Sendable {
    public init() {}

    public func render(
        page: PDFiumPage,
        dpi: Double,
        maxRenderedLongSide: Double,
        renderFlags: Int32 = Int32(FPDF_ANNOT)
    ) throws -> CGImage {
        let widthPoints = max(1.0, page.widthPoints)
        let heightPoints = max(1.0, page.heightPoints)
        let longSidePoints = max(widthPoints, heightPoints)

        var scale = dpi / 72.0
        let scaledLongSide = longSidePoints * scale
        if scaledLongSide > maxRenderedLongSide {
            scale = maxRenderedLongSide / longSidePoints
        }

        let srcWidth = max(1, Int(ceil(widthPoints * scale)))
        let srcHeight = max(1, Int(ceil(heightPoints * scale)))
        let srcWidthC = Int32(srcWidth)
        let srcHeightC = Int32(srcHeight)

        guard let bitmap = FPDFBitmap_CreateEx(srcWidthC, srcHeightC, Int32(FPDFBitmap_BGR), nil, 0) else {
            throw PDFiumError.bitmapCreateFailed(srcWidth, srcHeight)
        }

        defer {
            FPDFBitmap_Destroy(bitmap)
        }

        FPDFBitmap_FillRect(bitmap, 0, 0, srcWidthC, srcHeightC, 0xFFFF_FFFF)
        FPDF_RenderPageBitmap(
            bitmap,
            page.handle,
            0,
            0,
            srcWidthC,
            srcHeightC,
            0,
            renderFlags
        )

        guard let buffer = FPDFBitmap_GetBuffer(bitmap) else {
            throw PDFiumError.bitmapBufferUnavailable
        }
        let stride = Int(FPDFBitmap_GetStride(bitmap))

        let rgb = convertBGRBufferToRGB(
            source: buffer.assumingMemoryBound(to: UInt8.self),
            width: srcWidth,
            height: srcHeight,
            stride: stride
        )

        guard let image = makeImageFromRGB(bytes: rgb, width: srcWidth, height: srcHeight) else {
            throw PDFiumError.cgImageCreateFailed(srcWidth, srcHeight)
        }

        return image
    }

    private func convertBGRBufferToRGB(
        source: UnsafePointer<UInt8>,
        width: Int,
        height: Int,
        stride: Int
    ) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)

        for y in 0..<height {
            let rowBase = source.advanced(by: y * stride)
            for x in 0..<width {
                let srcOffset = x * 3
                let dstOffset = (y * width + x) * 3
                rgb[dstOffset] = rowBase[srcOffset + 2]
                rgb[dstOffset + 1] = rowBase[srcOffset + 1]
                rgb[dstOffset + 2] = rowBase[srcOffset]
            }
        }

        return rgb
    }

    private func makeImageFromRGB(
        bytes: [UInt8],
        width: Int,
        height: Int
    ) -> CGImage? {
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
}
