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
        let direct = URL(fileURLWithPath: modelID)
        var isDirectory = ObjCBool(false)

        if FileManager.default.fileExists(atPath: direct.path, isDirectory: &isDirectory), isDirectory.boolValue {
            return direct
        }

        let sanitized = modelID.replacingOccurrences(of: "/", with: "--")
        guard
            let appSupportDirectory = FileManager.default.urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask
            ).first
        else {
            throw GLMInferenceClientError.unresolvedModelDirectory(modelID)
        }

        let snapshotsRoot =
            appSupportDirectory
            .appending(path: "GlmOCRSwift")
            .appending(path: "huggingface")
            .appending(path: "hub")
            .appending(path: "models--\(sanitized)")
            .appending(path: "snapshots")

        guard
            let candidates = try? FileManager.default.contentsOfDirectory(
                at: snapshotsRoot,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            ), !candidates.isEmpty
        else {
            throw GLMInferenceClientError.unresolvedModelDirectory(modelID)
        }

        let ranked = candidates.sorted { lhs, rhs in
            let lhsDate =
                (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            let rhsDate =
                (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            return lhsDate > rhsDate
        }

        return ranked[0]
    }
}
