import Foundation

internal enum GLMModelDirectoryLookup {
    internal static func resolveDirectoryIfPresent(
        path: String,
        fileManager: FileManager = .default
    ) -> URL? {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return nil
        }

        let candidate = URL(fileURLWithPath: trimmed).resolvingSymlinksInPath()
        var isDirectory = ObjCBool(false)
        guard fileManager.fileExists(atPath: candidate.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return nil
        }

        return candidate
    }

    internal static func cachedSnapshotDirectory(
        for modelID: String,
        fileManager: FileManager = .default,
        appSupportDirectory: URL? = nil
    ) -> URL? {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return nil
        }

        let sanitized = trimmed.replacingOccurrences(of: "/", with: "--")
        let effectiveAppSupportDirectory =
            appSupportDirectory
            ?? fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first

        guard let effectiveAppSupportDirectory else {
            return nil
        }

        let snapshotsRoot =
            effectiveAppSupportDirectory
            .appending(path: "GlmOCRSwift")
            .appending(path: "huggingface")
            .appending(path: "hub")
            .appending(path: "models--\(sanitized)")
            .appending(path: "snapshots")

        guard
            let candidates = try? fileManager.contentsOfDirectory(
                at: snapshotsRoot,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            ),
            !candidates.isEmpty
        else {
            return nil
        }

        let resolvedCandidates: [(url: URL, modifiedAt: Date)] = candidates.compactMap { candidate in
            let resolved = candidate.resolvingSymlinksInPath()
            var isDirectory = ObjCBool(false)
            guard fileManager.fileExists(atPath: resolved.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                return nil
            }

            let modifiedAt =
                (try? resolved.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            return (resolved, modifiedAt)
        }

        let ranked = resolvedCandidates.sorted { lhs, rhs in
            if lhs.modifiedAt != rhs.modifiedAt {
                return lhs.modifiedAt > rhs.modifiedAt
            }
            return lhs.url.lastPathComponent > rhs.url.lastPathComponent
        }

        return ranked.first?.url
    }
}
