import GlmOCRRecognizerMLX
import XCTest

@testable import GlmOCRSwift

final class DefaultsParityTests: XCTestCase {
    func testGenerationOptionsEnvironmentFallbacksMatchExpectedDefaults() {
        let options = GlmOcrGenerationOptions.fromEnvironment([:])

        XCTAssertEqual(options.maxTokens, 4_096)
        XCTAssertEqual(options.temperature, 0.8, accuracy: 0.0001)
        XCTAssertEqual(options.prefillStepSize, 2_048)
        XCTAssertEqual(options.topP, 0.9, accuracy: 0.0001)
        XCTAssertEqual(options.topK, 50)
        XCTAssertEqual(options.repetitionPenalty, 1.1, accuracy: 0.0001)
    }

    func testDefaultFormulaTaskMappingMatchesUpstream() {
        let formulaLabels = GlmOCRLayoutConfig.defaultLabelTaskMapping["formula"]

        XCTAssertEqual(formulaLabels, ["display_formula", "inline_formula"])
        XCTAssertFalse(formulaLabels?.contains("formula") ?? false)
    }
}
