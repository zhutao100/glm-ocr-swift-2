import CoreGraphics
import Foundation
import GlmOCRLayoutMLX

internal struct LayoutDetectionDetailedOutput: Sendable {
    internal let regions: [PipelineLayoutRegion]
    internal let timings: PipelineLayoutStageTimings

    internal init(
        regions: [PipelineLayoutRegion],
        timings: PipelineLayoutStageTimings
    ) {
        self.regions = regions
        self.timings = timings
    }
}

internal protocol LayoutInferenceClient: Sendable {
    func detectLayout(image: CGImage) async throws -> [PipelineLayoutRegion]
    func detectLayoutDetailed(image: CGImage) async throws -> LayoutDetectionDetailedOutput
}

internal actor MLXLayoutInferenceClient: LayoutInferenceClient {
    private let runner: PPDocLayoutMLXRunner

    internal init(config: GlmOCRConfig) {
        self.runner = PPDocLayoutMLXRunner(
            modelID: config.layoutModelID,
            options: Self.makeRuntimeOptions(from: config)
        )
    }

    internal func detectLayout(image: CGImage) async throws -> [PipelineLayoutRegion] {
        let detailed = try await detectLayoutDetailed(image: image)
        return detailed.regions
    }

    internal func detectLayoutDetailed(image: CGImage) async throws -> LayoutDetectionDetailedOutput {
        let detailedDetections = try await runner.detectDetailed(image: image)
        let detections = detailedDetections.detections
        let imageWidth = max(1, image.width)
        let imageHeight = max(1, image.height)

        var regions: [PipelineLayoutRegion] = []
        regions.reserveCapacity(detections.count)

        var validIndex = 0
        for detection in detections {
            let task = mapTask(detection.task)
            if task == .abandon {
                continue
            }

            regions.append(
                PipelineLayoutRegion(
                    index: validIndex,
                    label: detection.label,
                    task: task,
                    score: Double(detection.score),
                    bbox2D: detection.bbox2D,
                    polygon2D: normalizePolygon(
                        detection.polygon,
                        imageWidth: imageWidth,
                        imageHeight: imageHeight
                    ),
                    order: detection.order
                )
            )
            validIndex += 1
        }

        return LayoutDetectionDetailedOutput(
            regions: regions,
            timings: PipelineLayoutStageTimings(
                preprocessMs: detailedDetections.preprocessMs,
                inferenceMs: detailedDetections.inferenceMs,
                postprocessMs: detailedDetections.postprocessMs
            )
        )
    }

    private func mapTask(_ task: PPDocLayoutTask) -> PipelineTask {
        switch task {
        case .text:
            return .text
        case .table:
            return .table
        case .formula:
            return .formula
        case .skip:
            return .skip
        case .abandon:
            return .abandon
        }
    }

    private func normalizePolygon(
        _ polygon: [[Double]],
        imageWidth: Int,
        imageHeight: Int
    ) -> [[Int]] {
        guard !polygon.isEmpty else {
            return []
        }

        return polygon.compactMap { point in
            guard point.count >= 2 else {
                return nil
            }

            let x = max(0, min(1000, Int((point[0] / Double(imageWidth)) * 1000.0)))
            let y = max(0, min(1000, Int((point[1] / Double(imageHeight)) * 1000.0)))
            return [x, y]
        }
    }

    private static func makeRuntimeOptions(from config: GlmOCRConfig) -> PPDocLayoutRuntimeOptions {
        let layoutConfig = config.layout
        let mergeModes = layoutConfig.layoutMergeBBoxesMode.reduce(into: [Int: LayoutMergeMode]()) { partial, pair in
            guard let mode = LayoutMergeMode(rawValue: pair.value) else {
                return
            }
            partial[pair.key] = mode
        }

        let taskMapping = layoutConfig.labelTaskMapping.reduce(into: [String: Set<String>]()) { partial, pair in
            partial[pair.key] = Set(pair.value)
        }

        return PPDocLayoutRuntimeOptions(
            threshold: layoutConfig.threshold,
            thresholdByClass: layoutConfig.thresholdByClass,
            layoutNMS: layoutConfig.layoutNMS,
            layoutUnclipRatio: layoutConfig.layoutUnclipRatio,
            layoutMergeBBoxesMode: mergeModes.isEmpty
                ? GlmOCRLayoutConfig.defaultLayoutMergeBBoxesMode.reduce(into: [Int: LayoutMergeMode]()) {
                    partial, pair in
                    if let mode = LayoutMergeMode(rawValue: pair.value) {
                        partial[pair.key] = mode
                    }
                }
                : mergeModes,
            labelTaskMapping: taskMapping.isEmpty
                ? GlmOCRLayoutConfig.defaultLabelTaskMapping.reduce(into: [String: Set<String>]()) { partial, pair in
                    partial[pair.key] = Set(pair.value)
                }
                : taskMapping,
            id2label: layoutConfig.id2label,
            layoutPostprocessFastPath: config.performance.layoutPostprocessFastPath
        )
    }
}
