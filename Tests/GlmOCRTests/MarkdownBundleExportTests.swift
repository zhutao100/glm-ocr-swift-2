import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import GlmOCRSwift

final class MarkdownBundleExportTests: XCTestCase {
    func testLayoutAndBundleEnabledProducesHEICFiguresAndRewrittenMarkdown() async throws {
        let pipeline = try makePipeline(enableLayout: true, bundleEnabled: true)
        let result = try await pipeline.parse(.pdfData(Data("fake-pdf".utf8)), options: .init())

        let bundle = try XCTUnwrap(result.markdownBundle)
        XCTAssertEqual(bundle.markdownFileName, "document.md")
        XCTAssertEqual(bundle.jsonFileName, "document.json")
        XCTAssertEqual(bundle.figuresDirectoryName, "figures")
        XCTAssertEqual(bundle.figures.count, 2)

        XCTAssertEqual(result.markdown, bundle.rewrittenMarkdown)
        XCTAssertFalse(result.markdown.localizedStandardContains("bbox="))
        XCTAssertTrue(result.markdown.localizedStandardContains("![Image 0-0](figures/page_0001_region_0001.heic)"))
        XCTAssertTrue(result.markdown.localizedStandardContains("![Image 0-1](figures/page_0001_region_0002.heic)"))

        for figure in bundle.figures {
            XCTAssertEqual(figure.mimeType, "image/heic")
            XCTAssertTrue(figure.relativePath.hasPrefix("figures/"))
            XCTAssertTrue(figure.fileName.hasSuffix(".heic"))
            XCTAssertFalse(figure.sha256.isEmpty)
            XCTAssertGreaterThan(figure.widthPX, 0)
            XCTAssertGreaterThan(figure.heightPX, 0)

            let source = try XCTUnwrap(CGImageSourceCreateWithData(figure.data as CFData, nil))
            let type = try XCTUnwrap(CGImageSourceGetType(source) as String?)
            XCTAssertEqual(type, UTType.heic.identifier)
        }

        let sidecarData = Data(bundle.documentJSON.utf8)
        let sidecarObject = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: sidecarData) as? [String: Any]
        )
        XCTAssertEqual(sidecarObject["schemaVersion"] as? Int, 1)
        XCTAssertEqual(sidecarObject["markdownPath"] as? String, "document.md")
        XCTAssertEqual(sidecarObject["figuresDirectory"] as? String, "figures")
        XCTAssertNotNil(sidecarObject["pages"] as? [[String: Any]])
        XCTAssertNotNil(sidecarObject["diagnostics"] as? [String: Any])
        XCTAssertEqual((sidecarObject["figures"] as? [[String: Any]])?.count, 2)
    }

    func testBundleDisabledKeepsLegacyImageMarkers() async throws {
        let pipeline = try makePipeline(enableLayout: true, bundleEnabled: false)
        let result = try await pipeline.parse(.pdfData(Data("fake-pdf".utf8)), options: .init())

        XCTAssertNil(result.markdownBundle)
        XCTAssertTrue(result.markdown.localizedStandardContains("bbox=["))
        XCTAssertTrue(result.markdown.localizedStandardContains("page=0"))
        XCTAssertTrue(result.markdown.localizedStandardContains("![Image 0-0]("))
    }

    func testNoLayoutDoesNotProduceBundleEvenWhenEnabled() async throws {
        let pipeline = try makePipeline(enableLayout: false, bundleEnabled: true)
        let result = try await pipeline.parse(.pdfData(Data("fake-pdf".utf8)), options: .init())

        XCTAssertNil(result.markdownBundle)
        XCTAssertFalse(result.markdown.isEmpty)
    }

    private func makePipeline(enableLayout: Bool, bundleEnabled: Bool) throws -> GlmOCRPipeline {
        var config = GlmOCRConfig()
        config.enableLayout = enableLayout
        config.markdownBundle.enabled = bundleEnabled

        let page = Self.makeSolidTestImage(width: 640, height: 480)
        let pageLoader = StubPageLoader(pages: [page])
        let layoutDetector = StubLayoutDetector(
            pageDetections: [
                [
                    PipelineLayoutRegion(
                        index: 0,
                        label: "text",
                        task: .text,
                        score: 0.9,
                        bbox2D: [0, 0, 1000, 260],
                        polygon2D: [],
                        order: 0
                    ),
                    PipelineLayoutRegion(
                        index: 1,
                        label: "image",
                        task: .skip,
                        score: 0.9,
                        bbox2D: [80, 300, 420, 900],
                        polygon2D: [],
                        order: 1
                    ),
                    PipelineLayoutRegion(
                        index: 2,
                        label: "chart",
                        task: .skip,
                        score: 0.9,
                        bbox2D: [520, 280, 920, 940],
                        polygon2D: [],
                        order: 2
                    ),
                ]
            ]
        )
        let recognizer = StubRegionRecognizer()

        return try GlmOCRPipeline(
            config: config,
            pageLoader: pageLoader,
            layoutDetector: layoutDetector,
            regionRecognizer: recognizer
        )
    }

    private static func makeSolidTestImage(width: Int, height: Int) -> CGImage {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * bytesPerRow) + (x * 4)
                pixels[offset] = UInt8((x * 255) / max(width - 1, 1))
                pixels[offset + 1] = UInt8((y * 255) / max(height - 1, 1))
                pixels[offset + 2] = 180
                pixels[offset + 3] = 255
            }
        }

        guard
            let context = CGContext(
                data: &pixels,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        else {
            fatalError("Failed to build test CGContext")
        }

        guard let image = context.makeImage() else {
            fatalError("Failed to build test CGImage")
        }
        return image
    }
}

private struct StubPageLoader: PipelinePageLoading {
    let pages: [CGImage]

    func loadPages(from _: InputDocument, maxPages _: Int?) throws -> [CGImage] {
        pages
    }
}

private actor StubLayoutDetector: PipelineLayoutDetecting {
    let pageDetections: [[PipelineLayoutRegion]]

    init(pageDetections: [[PipelineLayoutRegion]]) {
        self.pageDetections = pageDetections
    }

    func detectDetailed(pages _: [CGImage], options _: ParseOptions) async throws -> [[PipelineLayoutRegion]] {
        pageDetections
    }
}

private actor StubRegionRecognizer: RegionRecognizer {
    func recognize(_ region: CGImage, task: OCRTask) async throws -> String {
        switch task {
        case .text:
            return "Recognized text \(region.width)x\(region.height)"
        case .table:
            return "| a | b |\n| - | - |\n| 1 | 2 |"
        case .formula:
            return "x^2 + y^2 = z^2"
        }
    }
}
