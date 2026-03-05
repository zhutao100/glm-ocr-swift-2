import CryptoKit
import Foundation
import HuggingFace

private struct SnapshotValidation {
    let integrityPaths: [String]
}

public actor SandboxModelManager: ModelDeliveryManaging {
    private let manifest: ModelDeliveryManifest
    private let downloader: any HubDownloading
    private let fileManager: FileManager
    private let stateFileURL: URL
    private let maxConcurrentDownloads: Int

    private var state: ModelDeliveryState

    public init() throws {
        let fileManager = FileManager.default

        guard let appSupportDirectory = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        else {
            throw ModelDeliveryError.invalidConfiguration("Unable to resolve Application Support directory")
        }

        let cacheDirectory =
            appSupportDirectory
            .appending(path: "GlmOCRSwift")
            .appending(path: "huggingface")
            .appending(path: "hub")

        let hubCache = HubCache(cacheDirectory: cacheDirectory)

        let endpoint =
            ProcessInfo.processInfo.environment["HF_ENDPOINT"]
            .flatMap(URL.init(string:)) ?? HubClient.defaultHost

        let client = HubClient(
            host: endpoint,
            tokenProvider: .environment,
            cache: hubCache
        )

        try self.init(
            manifest: try ModelDeliveryManifest.bundled(),
            downloader: HuggingFaceHubDownloader(client: client),
            appSupportDirectory: appSupportDirectory,
            fileManager: fileManager,
            maxConcurrentDownloads: 8
        )
    }

    internal init(
        manifest: ModelDeliveryManifest,
        downloader: any HubDownloading,
        appSupportDirectory: URL,
        fileManager: FileManager = .default,
        maxConcurrentDownloads: Int = 8,
        stateFileURL: URL? = nil
    ) throws {
        guard maxConcurrentDownloads > 0 else {
            throw ModelDeliveryError.invalidConfiguration("maxConcurrentDownloads must be greater than zero")
        }

        self.manifest = manifest
        self.downloader = downloader
        self.fileManager = fileManager
        self.maxConcurrentDownloads = maxConcurrentDownloads

        let defaultStateFileURL =
            appSupportDirectory
            .appending(path: "GlmOCRSwift")
            .appending(path: "ModelDelivery")
            .appending(path: "model-delivery-state.json")

        self.stateFileURL = stateFileURL ?? defaultStateFileURL
        self.state = Self.loadState(
            stateFileURL: self.stateFileURL,
            fileManager: fileManager
        )
    }

    public func ensureReady(config: ModelDeliveryRequest) async throws -> ModelDeliveryResolvedPaths {
        let recognizerPath = try await ensureModelReady(
            modelID: config.recognizerModelID,
            localFilesOnly: false
        )

        let layoutPath = try await ensureModelReady(
            modelID: config.layoutModelID,
            localFilesOnly: false
        )

        return ModelDeliveryResolvedPaths(
            recognizerModelDirectory: recognizerPath,
            layoutModelDirectory: layoutPath
        )
    }

    public func verifyOfflineReadiness(config: ModelDeliveryRequest) async throws -> ModelDeliveryResolvedPaths {
        let recognizerPath = try await ensureModelReady(
            modelID: config.recognizerModelID,
            localFilesOnly: true
        )

        let layoutPath = try await ensureModelReady(
            modelID: config.layoutModelID,
            localFilesOnly: true
        )

        return ModelDeliveryResolvedPaths(
            recognizerModelDirectory: recognizerPath,
            layoutModelDirectory: layoutPath
        )
    }

    private func ensureModelReady(
        modelID: String,
        localFilesOnly: Bool
    ) async throws -> URL {
        let trimmedID = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedID.isEmpty else {
            throw ModelDeliveryError.invalidConfiguration("modelID must not be empty")
        }

        if let localDirectory = existingDirectory(at: trimmedID) {
            let validation = try validateSnapshotDirectory(
                localDirectory,
                modelID: trimmedID,
                requiredFiles: ["config.json"],
                requireAnySafetensors: true
            )

            if localFilesOnly, let persisted = state.models[trimmedID] {
                try verifyPersistedState(
                    persisted,
                    modelID: trimmedID,
                    requiredPaths: Set(validation.integrityPaths)
                )
                try verifyChecksumsIfNeeded(
                    directory: localDirectory,
                    modelID: trimmedID,
                    files: persisted.files,
                    requiredPaths: Set(validation.integrityPaths)
                )
            }

            return localDirectory
        }

        let spec = try modelSpec(for: trimmedID)

        guard let repoID = spec.repoID else {
            throw ModelDeliveryError.unsupportedModelID(trimmedID)
        }

        let snapshotPath: URL
        do {
            snapshotPath = try await downloader.downloadSnapshot(
                of: repoID,
                kind: spec.repoKind,
                revision: spec.revision,
                matching: spec.downloadGlobs,
                localFilesOnly: localFilesOnly,
                maxConcurrentDownloads: maxConcurrentDownloads
            )
        } catch {
            throw ModelDeliveryError.hubFailure(error.localizedDescription)
        }

        let validation = try validateSnapshotDirectory(
            snapshotPath,
            modelID: spec.modelID,
            requiredFiles: spec.requiredFiles,
            requireAnySafetensors: spec.requireAnySafetensors
        )

        let requiredIntegrityPaths = Set(validation.integrityPaths)

        if localFilesOnly {
            guard let persisted = state.models[spec.modelID] else {
                throw ModelDeliveryError.missingPersistedState(modelID: spec.modelID)
            }

            try verifyPersistedState(
                persisted,
                modelID: spec.modelID,
                requiredPaths: requiredIntegrityPaths
            )
            try verifyChecksumsIfNeeded(
                directory: snapshotPath,
                modelID: spec.modelID,
                files: persisted.files,
                requiredPaths: requiredIntegrityPaths
            )

            return snapshotPath
        }

        var updatedFiles: [ModelDeliveryFileState] = []
        updatedFiles.reserveCapacity(validation.integrityPaths.count)

        for relativePath in validation.integrityPaths {
            let fileInfo: HubRemoteFileInfo
            do {
                fileInfo = try await downloader.getFileInfo(
                    at: relativePath,
                    in: repoID,
                    kind: spec.repoKind,
                    revision: spec.revision
                )
            } catch {
                throw ModelDeliveryError.hubFailure(
                    "Unable to fetch metadata for '\(spec.modelID)/\(relativePath)': \(error.localizedDescription)"
                )
            }

            guard let etag = fileInfo.etag, !etag.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ModelDeliveryError.missingFileMetadata(modelID: spec.modelID, path: relativePath)
            }

            let normalizedETag = normalizeETag(etag)
            let localFileURL = snapshotPath.appending(path: relativePath)

            let verifiedETag = try resolvedIntegrityETag(
                fileURL: localFileURL,
                snapshotDirectory: snapshotPath,
                modelID: spec.modelID,
                relativePath: relativePath,
                remoteETag: normalizedETag
            )

            updatedFiles.append(
                ModelDeliveryFileState(
                    relativePath: relativePath,
                    etag: verifiedETag,
                    commitHash: fileInfo.revision
                )
            )
        }

        state.models[spec.modelID] = ModelDeliveryModelState(
            modelID: spec.modelID,
            revision: spec.revision,
            snapshotPath: snapshotPath.path,
            updatedAtUTC: Date(),
            files: updatedFiles.sorted { $0.relativePath < $1.relativePath }
        )

        try persistState()

        return snapshotPath
    }

    private func modelSpec(for modelID: String) throws -> ModelDeliveryModelSpec {
        if let explicit = manifest.modelSpec(for: modelID) {
            return explicit
        }

        return try ModelDeliveryManifest.fallbackSpec(for: modelID)
    }

    private func existingDirectory(at path: String) -> URL? {
        let url = URL(fileURLWithPath: path)
        var isDirectory = ObjCBool(false)

        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return nil
        }

        return url
    }

    private func validateSnapshotDirectory(
        _ directory: URL,
        modelID: String,
        requiredFiles: [String],
        requireAnySafetensors: Bool
    ) throws -> SnapshotValidation {
        var allPaths: Set<String> = []

        guard
            let enumerator = fileManager.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
                options: [.skipsHiddenFiles]
            )
        else {
            throw ModelDeliveryError.cacheUnavailable("Unable to enumerate snapshot at \(directory.path)")
        }

        while let fileURL = enumerator.nextObject() as? URL {
            let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard values?.isRegularFile == true || values?.isSymbolicLink == true else {
                continue
            }

            allPaths.insert(relativePath(from: fileURL, base: directory))
        }

        var missing: [String] = []
        for file in requiredFiles {
            if !allPaths.contains(file) {
                missing.append(file)
            }
        }

        if let firstMissing = missing.sorted().first {
            throw ModelDeliveryError.missingRequiredFile(modelID: modelID, path: firstMissing)
        }

        let safetensors =
            allPaths
            .filter { $0.lowercased().hasSuffix(".safetensors") }
            .sorted()

        if requireAnySafetensors, safetensors.isEmpty {
            throw ModelDeliveryError.missingSafetensor(modelID: modelID)
        }

        let integrityPaths = Set(requiredFiles + safetensors)
        return SnapshotValidation(integrityPaths: integrityPaths.sorted())
    }

    private func verifyPersistedState(
        _ persisted: ModelDeliveryModelState,
        modelID: String,
        requiredPaths: Set<String>
    ) throws {
        let persistedPaths = Set(persisted.files.map(\.relativePath))

        for path in requiredPaths where !persistedPaths.contains(path) {
            throw ModelDeliveryError.missingFileMetadata(modelID: modelID, path: path)
        }
    }

    private func verifyChecksumsIfNeeded(
        directory: URL,
        modelID: String,
        files: [ModelDeliveryFileState],
        requiredPaths: Set<String>
    ) throws {
        for file in files {
            guard requiredPaths.contains(file.relativePath) else {
                continue
            }

            let fileURL = directory.appending(path: file.relativePath)
            try verifyChecksumIfNeeded(
                fileURL: fileURL,
                modelID: modelID,
                relativePath: file.relativePath,
                etag: file.etag
            )
        }
    }

    private func verifyChecksumIfNeeded(
        fileURL: URL,
        modelID: String,
        relativePath: String,
        etag: String
    ) throws {
        guard isSHA256(etag) else {
            return
        }

        guard fileManager.fileExists(atPath: fileURL.path) else {
            throw ModelDeliveryError.missingRequiredFile(modelID: modelID, path: relativePath)
        }

        let actualHash = try computeSHA256(fileURL: fileURL)
        if actualHash != etag.lowercased() {
            throw ModelDeliveryError.checksumMismatch(
                modelID: modelID,
                path: relativePath,
                expected: etag.lowercased(),
                actual: actualHash
            )
        }
    }

    private func resolvedIntegrityETag(
        fileURL: URL,
        snapshotDirectory: URL,
        modelID: String,
        relativePath: String,
        remoteETag: String
    ) throws -> String {
        guard isSHA256(remoteETag) else {
            return remoteETag
        }

        guard fileManager.fileExists(atPath: fileURL.path) else {
            throw ModelDeliveryError.missingRequiredFile(modelID: modelID, path: relativePath)
        }

        let normalizedRemote = remoteETag.lowercased()
        let actualHash = try computeSHA256(fileURL: fileURL)
        if actualHash == normalizedRemote {
            return normalizedRemote
        }

        if let metadataHash = loadMetadataETag(
            snapshotDirectory: snapshotDirectory,
            relativePath: relativePath
        )?.lowercased() {
            if isSHA256(metadataHash), metadataHash == actualHash {
                return metadataHash
            }

            throw ModelDeliveryError.checksumMismatch(
                modelID: modelID,
                path: relativePath,
                expected: metadataHash,
                actual: actualHash
            )
        }

        // Some Hub backends expose a transport-level content hash in ETag while
        // the downloaded file checksum is represented separately (e.g. X-Linked-Etag).
        // Persist the verified on-disk checksum to keep offline checks deterministic.
        return actualHash
    }

    private func loadMetadataETag(
        snapshotDirectory: URL,
        relativePath: String
    ) -> String? {
        let metadataURL =
            snapshotDirectory
            .appending(path: ".cache")
            .appending(path: "huggingface")
            .appending(path: "download")
            .appending(path: "\(relativePath).metadata")

        guard fileManager.fileExists(atPath: metadataURL.path),
            let content = try? String(contentsOf: metadataURL, encoding: .utf8)
        else {
            return nil
        }

        let lines =
            content
            .split(whereSeparator: \.isNewline)
            .map(String.init)
        guard lines.count >= 2 else {
            return nil
        }
        return lines[1].trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func relativePath(from fileURL: URL, base: URL) -> String {
        let absolute = fileURL.standardizedFileURL.path
        let basePath = base.standardizedFileURL.path
        let prefix = basePath.hasSuffix("/") ? basePath : basePath + "/"

        if absolute.hasPrefix(prefix) {
            return String(absolute.dropFirst(prefix.count))
        }

        return fileURL.lastPathComponent
    }

    private func normalizeETag(_ etag: String) -> String {
        var normalized = etag.trimmingCharacters(in: .whitespacesAndNewlines)

        if normalized.hasPrefix("W/") {
            normalized = String(normalized.dropFirst(2))
        }

        if normalized.hasPrefix("\"") && normalized.hasSuffix("\"") && normalized.count >= 2 {
            normalized = String(normalized.dropFirst().dropLast())
        }

        return normalized.lowercased()
    }

    private func isSHA256(_ value: String) -> Bool {
        let lower = value.lowercased()
        guard lower.count == 64 else {
            return false
        }

        return lower.unicodeScalars.allSatisfy { scalar in
            switch scalar.value {
            case 48...57, 97...102:
                return true
            default:
                return false
            }
        }
    }

    private func computeSHA256(fileURL: URL) throws -> String {
        guard let handle = try? FileHandle(forReadingFrom: fileURL) else {
            throw ModelDeliveryError.ioFailure("Unable to open file for hashing: \(fileURL.path)")
        }

        defer {
            try? handle.close()
        }

        var hasher = SHA256()
        let chunkSize = 1_048_576

        while true {
            let chunk = try handle.read(upToCount: chunkSize) ?? Data()
            if chunk.isEmpty {
                break
            }
            hasher.update(data: chunk)
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func persistState() throws {
        do {
            try fileManager.createDirectory(
                at: stateFileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            let data = try JSONEncoder().encode(state)
            try data.write(to: stateFileURL, options: .atomic)
        } catch {
            throw ModelDeliveryError.ioFailure(
                "Unable to persist model delivery state to \(stateFileURL.path): \(error.localizedDescription)"
            )
        }
    }

    private static func loadState(
        stateFileURL: URL,
        fileManager: FileManager
    ) -> ModelDeliveryState {
        guard fileManager.fileExists(atPath: stateFileURL.path),
            let data = try? Data(contentsOf: stateFileURL),
            let decoded = try? JSONDecoder().decode(ModelDeliveryState.self, from: data)
        else {
            return ModelDeliveryState()
        }

        return decoded
    }
}
