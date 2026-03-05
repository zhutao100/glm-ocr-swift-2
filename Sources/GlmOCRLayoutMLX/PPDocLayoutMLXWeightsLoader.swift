import Foundation
import MLX

internal struct PPDocLayoutWeightSnapshot: @unchecked Sendable {
    internal let modelDirectory: URL
    internal let configuration: PPDocLayoutV3Configuration
    internal let weights: PPDocLayoutWeightStore
}

internal actor PPDocLayoutMLXWeightsLoader {
    internal init() {}

    internal func loadSnapshot(modelID: String) async throws -> PPDocLayoutWeightSnapshot {
        let trimmedID = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedID.isEmpty else {
            throw PPDocLayoutMLXError.invalidModelID("layout model id must not be empty")
        }

        let modelDirectory = try await resolveModelDirectory(modelID: trimmedID)
        let configuration = try PPDocLayoutV3Configuration.load(from: modelDirectory)
        let weightFiles = try safetensorFiles(in: modelDirectory)

        guard !weightFiles.isEmpty else {
            throw PPDocLayoutMLXError.missingModelFile(
                "No .safetensors files found in \(modelDirectory.path)"
            )
        }

        var mergedWeights: [String: MLXArray] = [:]
        for fileURL in weightFiles {
            let tensors = try loadArrays(url: fileURL, stream: .cpu)
            for (key, value) in tensors {
                mergedWeights[key] = PPDocLayoutMLXTensorOps.ensureFloat32(value)
            }
        }

        if mergedWeights.isEmpty {
            throw PPDocLayoutMLXError.missingModelFile(
                "Parsed safetensors but found zero tensors in \(modelDirectory.path)"
            )
        }

        let store = PPDocLayoutWeightStore(tensors: mergedWeights)

        return PPDocLayoutWeightSnapshot(
            modelDirectory: modelDirectory,
            configuration: configuration,
            weights: store
        )
    }

    private func resolveModelDirectory(modelID: String) async throws -> URL {
        let candidateURL = URL(fileURLWithPath: modelID)
        var isDirectory = ObjCBool(false)
        if FileManager.default.fileExists(atPath: candidateURL.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            return candidateURL
        }

        throw PPDocLayoutMLXError.modelDownloadFailed(
            "Unable to resolve local layout model directory '\(modelID)'"
        )
    }

    private func safetensorFiles(in modelDirectory: URL) throws -> [URL] {
        let contents = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )

        return
            contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
