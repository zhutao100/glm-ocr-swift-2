import Foundation
import XCTest

@testable import GlmOCRSwift

final class ReproPipelineTests: XCTestCase {
    private func reproPDFData() throws -> Data {
        guard let path = ProcessInfo.processInfo.environment["GLMOCR_REPRO_PDF_PATH"] else {
            throw XCTSkip("Set GLMOCR_REPRO_PDF_PATH to run repro pipeline tests.")
        }
        return try Data(contentsOf: URL(fileURLWithPath: path))
    }

    func testParseWithoutLayoutReproPDF() async throws {
        let pdfData = try reproPDFData()
        var config = GlmOCRConfig()
        config.enableLayout = false

        let pipeline = try await GlmOCRPipeline(config: config)
        let result = try await pipeline.parse(.pdfData(pdfData), options: .init(maxPages: 1))

        XCTAssertFalse(result.pages.isEmpty)
    }

    func testParseWithLayoutReproPDF() async throws {
        let pdfData = try reproPDFData()
        let pipeline = try await GlmOCRPipeline(config: .init())
        let result = try await pipeline.parse(.pdfData(pdfData), options: .init(maxPages: 1))

        XCTAssertFalse(result.pages.isEmpty)
    }
}
