import XCTest

@testable import GlmOCRSwift

final class GlmOCRConfigTests: XCTestCase {
    func testDefaultValuesMatchExpected() {
        let config = GlmOCRConfig()

        XCTAssertEqual(config.recognizerModelID, "mlx-community/GLM-OCR-bf16")
        XCTAssertEqual(config.layoutModelID, "PaddlePaddle/PP-DocLayoutV3_safetensors")
        XCTAssertEqual(config.maxConcurrentRecognitions, 1)
        XCTAssertTrue(config.enableLayout)
        XCTAssertEqual(config.performance.inferenceBatchSize, 4)
        XCTAssertEqual(config.performance.inferenceBatchMaxWaitMs, 8)
        XCTAssertEqual(config.performance.inferenceMaxInflightJobs, 64)
        XCTAssertEqual(config.performance.pdfRenderConcurrency, 2)
        XCTAssertEqual(config.performance.ocrPreprocessConcurrency, 4)
        XCTAssertEqual(config.performance.bundleEncodeConcurrency, 2)
        XCTAssertTrue(config.performance.layoutPostprocessFastPath)
        XCTAssertTrue(config.markdownBundle.enabled)
        XCTAssertEqual(config.markdownBundle.figureFormat, .heic)
        XCTAssertEqual(config.markdownBundle.markdownFileName, "document.md")
        XCTAssertEqual(config.markdownBundle.jsonFileName, "document.json")
        XCTAssertEqual(config.markdownBundle.figuresDirectoryName, "figures")
        XCTAssertEqual(config.markdownBundle.heicCompressionQuality, 0.82, accuracy: 0.0001)
        XCTAssertNil(config.defaultMaxPages)
        XCTAssertEqual(config.recognitionOptions.maxTokens, 4_096)
        XCTAssertEqual(config.recognitionOptions.temperature, 0.8)
        XCTAssertEqual(config.recognitionOptions.prefillStepSize, 2_048)
        XCTAssertEqual(config.recognitionOptions.topP, 0.9)
        XCTAssertEqual(config.recognitionOptions.topK, 50)
        XCTAssertEqual(config.recognitionOptions.repetitionPenalty, 1.1)
    }

    func testDefaultConfigPassesValidation() throws {
        let config = GlmOCRConfig()
        try config.validate()
    }

    func testInvalidConfigurationRejectsEmptyModelIDs() {
        XCTAssertThrowsError(
            try GlmOCRConfig(
                recognizerModelID: "   ",
                layoutModelID: "    "
            ).validate())
    }

    func testRecognitionOptionsRoundTrip() throws {
        let config = GlmOCRConfig()
        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(GlmOCRConfig.self, from: encoded)

        XCTAssertEqual(decoded, config)
    }

    func testLegacyMaxConcurrentMapsToEffectivePerformanceWhenPerformanceDefaults() {
        var config = GlmOCRConfig()
        config.maxConcurrentRecognitions = 3

        XCTAssertEqual(config.effectivePerformanceConfig.inferenceMaxInflightJobs, 3)
    }

    func testExplicitPerformanceOverridesLegacyDefaultConcurrency() {
        var config = GlmOCRConfig()
        config.performance.inferenceBatchSize = 6
        config.performance.inferenceMaxInflightJobs = 40

        XCTAssertEqual(config.maxConcurrentRecognitions, 1)
        XCTAssertEqual(config.effectivePerformanceConfig.inferenceBatchSize, 6)
        XCTAssertEqual(config.effectivePerformanceConfig.inferenceMaxInflightJobs, 40)
    }
}
