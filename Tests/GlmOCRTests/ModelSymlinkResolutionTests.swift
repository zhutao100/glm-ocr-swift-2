import Foundation
import GlmOCRModelDelivery
@testable import GlmOCRSwift
import XCTest

final class ModelSymlinkResolutionTests: XCTestCase {
    func testResolveDirectoryIfPresentFollowsSymlink() throws {
        let fileManager = FileManager.default
        let tempRoot = try makeTemporaryDirectory()
        defer { try? fileManager.removeItem(at: tempRoot) }

        let realDirectory = tempRoot.appending(path: "real-model")
        try fileManager.createDirectory(at: realDirectory, withIntermediateDirectories: true)

        let symlinkDirectory = tempRoot.appending(path: "model-link")
        try fileManager.createSymbolicLink(
            atPath: symlinkDirectory.path,
            withDestinationPath: realDirectory.path
        )

        let resolved = try XCTUnwrap(GLMModelDirectoryLookup.resolveDirectoryIfPresent(path: symlinkDirectory.path))

        XCTAssertEqual(resolved.path, realDirectory.resolvingSymlinksInPath().path)
        XCTAssertNotEqual(resolved.path, symlinkDirectory.path)
    }

    func testCachedSnapshotDirectoryResolvesSymlinkCandidates() throws {
        let fileManager = FileManager.default
        let appSupportDirectory = try makeTemporaryDirectory()
        defer { try? fileManager.removeItem(at: appSupportDirectory) }

        let modelID = "acme/glm-ocr-test"
        let sanitized = modelID.replacingOccurrences(of: "/", with: "--")
        let snapshotsRoot =
            appSupportDirectory
            .appending(path: "GlmOCRSwift")
            .appending(path: "huggingface")
            .appending(path: "hub")
            .appending(path: "models--\(sanitized)")
            .appending(path: "snapshots")

        try fileManager.createDirectory(at: snapshotsRoot, withIntermediateDirectories: true)

        let localSnapshot = snapshotsRoot.appending(path: "local")
        try fileManager.createDirectory(at: localSnapshot, withIntermediateDirectories: true)

        let foreignSnapshot = appSupportDirectory.appending(path: "foreign-snapshot")
        try fileManager.createDirectory(at: foreignSnapshot, withIntermediateDirectories: true)

        try fileManager.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 1)],
            ofItemAtPath: localSnapshot.path
        )
        try fileManager.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 2)],
            ofItemAtPath: foreignSnapshot.path
        )

        let symlinkSnapshot = snapshotsRoot.appending(path: "linked")
        try fileManager.createSymbolicLink(
            atPath: symlinkSnapshot.path,
            withDestinationPath: foreignSnapshot.path
        )

        let resolved = try XCTUnwrap(
            GLMModelDirectoryLookup.cachedSnapshotDirectory(
                for: modelID,
                fileManager: fileManager,
                appSupportDirectory: appSupportDirectory
            )
        )

        XCTAssertEqual(resolved.path, foreignSnapshot.resolvingSymlinksInPath().path)
        XCTAssertNotEqual(resolved.path, symlinkSnapshot.path)
    }

    func testSandboxModelManagerEnsureReadyResolvesSymlinkDirectories() async throws {
        let fileManager = FileManager.default
        let tempRoot = try makeTemporaryDirectory()
        defer { try? fileManager.removeItem(at: tempRoot) }

        let realDirectory = tempRoot.appending(path: "real-model")
        try fileManager.createDirectory(at: realDirectory, withIntermediateDirectories: true)

        let configURL = realDirectory.appending(path: "config.json")
        try Data("{\"id2label\":{}}".utf8).write(to: configURL)

        let weightsURL = realDirectory.appending(path: "weights.safetensors")
        try Data().write(to: weightsURL)

        let symlinkDirectory = tempRoot.appending(path: "model-link")
        try fileManager.createSymbolicLink(
            atPath: symlinkDirectory.path,
            withDestinationPath: realDirectory.path
        )

        let manager = try SandboxModelManager()
        let resolved = try await manager.ensureReady(
            config: ModelDeliveryRequest(
                recognizerModelID: symlinkDirectory.path,
                layoutModelID: symlinkDirectory.path
            )
        )

        let expectedRealPath = realDirectory.resolvingSymlinksInPath().path
        XCTAssertEqual(resolved.recognizerModelDirectory.path, expectedRealPath)
        XCTAssertEqual(resolved.layoutModelDirectory.path, expectedRealPath)
        XCTAssertNotEqual(resolved.recognizerModelDirectory.path, symlinkDirectory.path)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
