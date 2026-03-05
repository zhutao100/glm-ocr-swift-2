import CoreGraphics
import Foundation
import GlmOCRRecognizerMLX

internal protocol GLMInferenceContainer: Sendable {}

internal struct GLMInferenceRequest: @unchecked Sendable {
    internal let prompt: String
    internal let image: CGImage

    internal init(prompt: String, image: CGImage) {
        self.prompt = prompt
        self.image = image
    }
}

internal protocol GLMInferenceClient: Sendable {
    func loadContainer(modelID: String) async throws -> any GLMInferenceContainer
    func recognize(
        container: any GLMInferenceContainer,
        request: GLMInferenceRequest,
        generationOptions: GlmOcrGenerationOptions
    ) async throws -> String
    func recognizeBatch(
        container: any GLMInferenceContainer,
        requests: [GLMInferenceRequest],
        generationOptions: GlmOcrGenerationOptions
    ) async throws -> [String]
}

internal enum GLMInferenceClientError: Error, Equatable, Sendable {
    case invalidContainerType
    case unresolvedModelDirectory(String)
}

internal struct MLXContainerHandle: GLMInferenceContainer {
    internal let runtime: GlmOcrRecognizerRuntime

    internal init(runtime: GlmOcrRecognizerRuntime) {
        self.runtime = runtime
    }
}

internal struct MLXGLMInferenceClient: GLMInferenceClient {
    internal init() {}

    internal func loadContainer(modelID: String) async throws -> any GLMInferenceContainer {
        let resolvedDirectory: URL
        let candidateDirectory = URL(fileURLWithPath: modelID)
        if FileManager.default.fileExists(atPath: candidateDirectory.path) {
            resolvedDirectory = candidateDirectory
        } else if let cached = cachedSnapshotDirectory(for: modelID) {
            resolvedDirectory = cached
        } else {
            throw GLMInferenceClientError.unresolvedModelDirectory(modelID)
        }

        let runtime = try await GlmOcrRecognizerRuntime(modelDirectory: resolvedDirectory)
        return MLXContainerHandle(runtime: runtime)
    }

    internal func recognize(
        container: any GLMInferenceContainer,
        request: GLMInferenceRequest,
        generationOptions: GlmOcrGenerationOptions
    ) async throws -> String {
        let outputs = try await recognizeBatch(
            container: container,
            requests: [request],
            generationOptions: generationOptions
        )
        return outputs.first ?? ""
    }

    internal func recognizeBatch(
        container: any GLMInferenceContainer,
        requests: [GLMInferenceRequest],
        generationOptions: GlmOcrGenerationOptions
    ) async throws -> [String] {
        guard let handle = container as? MLXContainerHandle else {
            throw GLMInferenceClientError.invalidContainerType
        }

        guard !requests.isEmpty else {
            return []
        }

        let runtimeRequests = requests.map {
            GlmOcrRecognizerBatchRequest(
                prompt: $0.prompt,
                image: $0.image
            )
        }

        return try await handle.runtime.recognizeBatch(
            requests: runtimeRequests,
            options: generationOptions
        )
    }

    private func cachedSnapshotDirectory(for modelID: String) -> URL? {
        let sanitized = modelID.replacingOccurrences(of: "/", with: "--")
        guard
            let appSupportDirectory = FileManager.default.urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask
            ).first
        else {
            return nil
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
            )
        else {
            return nil
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

        return ranked.first
    }
}
