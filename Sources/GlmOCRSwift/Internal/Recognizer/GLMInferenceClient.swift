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
        let trimmedModelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedDirectory: URL

        if let direct = GLMModelDirectoryLookup.resolveDirectoryIfPresent(path: trimmedModelID) {
            resolvedDirectory = direct
        } else if let cached = GLMModelDirectoryLookup.cachedSnapshotDirectory(for: trimmedModelID) {
            resolvedDirectory = cached
        } else {
            throw GLMInferenceClientError.unresolvedModelDirectory(trimmedModelID)
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
}
