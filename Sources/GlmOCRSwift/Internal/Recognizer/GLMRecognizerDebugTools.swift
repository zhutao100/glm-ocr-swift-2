import CoreGraphics
import Foundation
import GlmOCRRecognizerMLX

internal struct GLMInputSignature: Sendable, Equatable {
    internal let tokenCount: Int
    internal let imageTokenCount: Int
    internal let imageGridTHW: [[Int]]
    internal let tokens: [Int]

    internal init(
        tokenCount: Int,
        imageTokenCount: Int,
        imageGridTHW: [[Int]],
        tokens: [Int]
    ) {
        self.tokenCount = tokenCount
        self.imageTokenCount = imageTokenCount
        self.imageGridTHW = imageGridTHW
        self.tokens = tokens
    }
}

internal enum GLMRecognizerDebugTools {
    internal static func inputSignature(
        modelID: String,
        prompt: String,
        image: CGImage
    ) async throws -> GLMInputSignature {
        let modelDirectory = try resolveModelDirectory(modelID: modelID)
        let runtime = try await GlmOcrRecognizerRuntime(modelDirectory: modelDirectory)
        let signature = try await runtime.inputSignature(prompt: prompt, image: image)

        return GLMInputSignature(
            tokenCount: signature.tokenCount,
            imageTokenCount: signature.imageTokenCount,
            imageGridTHW: signature.imageGridTHW,
            tokens: signature.tokens
        )
    }

    private static func resolveModelDirectory(modelID: String) throws -> URL {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if let direct = GLMModelDirectoryLookup.resolveDirectoryIfPresent(path: trimmed) {
            return direct
        }

        if let cached = GLMModelDirectoryLookup.cachedSnapshotDirectory(for: trimmed) {
            return cached
        }

        throw GLMInferenceClientError.unresolvedModelDirectory(trimmed)
    }
}
