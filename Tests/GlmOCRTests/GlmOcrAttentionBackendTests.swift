import MLX
import XCTest

@testable import GlmOCRRecognizerMLX

final class GlmOcrAttentionBackendTests: XCTestCase {
    func testResolveBackendFallsBackWhenMetalValidationIsEnabled() {
        let backend = GlmOcrAttentionRuntime.resolveBackend(
            environment: ["MTL_DEBUG_LAYER": "1"]
        )
        XCTAssertEqual(backend, .fallback)
    }

    func testResolveBackendHonorsForceFastOverride() {
        let backend = GlmOcrAttentionRuntime.resolveBackend(
            environment: [
                "MTL_DEBUG_LAYER": "1",
                "GLMOCR_FORCE_FAST_SDPA": "1",
            ]
        )
        XCTAssertEqual(backend, .fast)
    }

    func testFallbackAttentionReturnsExpectedShapeWithCausalMask() {
        let queries = MLXRandom.normal([1, 2, 1, 64]).asType(.float32)
        let keys = MLXRandom.normal([1, 1, 3, 64]).asType(.float32)
        let values = MLXRandom.normal([1, 1, 3, 64]).asType(.float32)

        let output = glmOcrScaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(64),
            mask: .causal,
            backend: .fallback
        )
        eval(output)

        XCTAssertEqual(output.shape, [1, 2, 1, 64])
    }

    func testFallbackAttentionSupportsExplicitArrayMask() {
        let queries = MLXRandom.normal([1, 2, 1, 64]).asType(.float32)
        let keys = MLXRandom.normal([1, 1, 3, 64]).asType(.float32)
        let values = MLXRandom.normal([1, 1, 3, 64]).asType(.float32)
        let mask = MLXArray([1, 0, 1]).reshaped(1, 1, 1, 3) .== MLXArray(1)

        let output = glmOcrScaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(64),
            mask: .array(mask),
            backend: .fallback
        )
        eval(output)

        XCTAssertEqual(output.shape, [1, 2, 1, 64])
    }

    func testFallbackAttentionHandlesZeroLengthQuery() throws{
        try ensureMLXMetalLibraryColocated(for: Self.self)

        let queries = MLXArray.zeros([1, 2, 0, 64], dtype: .float32)
        let keys = MLXArray.zeros([1, 1, 0, 64], dtype: .float32)
        let values = MLXArray.zeros([1, 1, 0, 64], dtype: .float32)

        let output = glmOcrScaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(64),
            mask: .none,
            backend: .fallback
        )
        eval(output)

        XCTAssertEqual(output.shape, [1, 2, 0, 64])
    }
}
