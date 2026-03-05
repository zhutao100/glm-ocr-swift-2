import CoreGraphics
import Foundation

internal actor PPDocLayoutMLXDetector: LayoutDetector, PipelineLayoutDetecting, PipelineLayoutDetectingWithMetrics {
    private let inferenceClient: any LayoutInferenceClient
    private nonisolated static let traceEnabled =
        ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIPELINE_TRACE"] == "1"

    internal init(
        config: GlmOCRConfig,
        inferenceClient: (any LayoutInferenceClient)? = nil
    ) {
        if let inferenceClient {
            self.inferenceClient = inferenceClient
        } else {
            self.inferenceClient = MLXLayoutInferenceClient(config: config)
        }
    }

    internal func detect(pages: [CGImage], options: ParseOptions) async throws -> [[LayoutRegion]] {
        let detailed = try await detectDetailed(pages: pages, options: options)
        return detailed.map { page in
            page.map { region in
                LayoutRegion(
                    index: region.index,
                    label: region.label,
                    score: region.score,
                    bbox2D: region.bbox2D
                )
            }
        }
    }

    internal func detectDetailed(
        pages: [CGImage],
        options: ParseOptions
    ) async throws -> [[PipelineLayoutRegion]] {
        let output = try await detectDetailedWithMetrics(pages: pages, options: options)
        return output.pageRegions
    }

    internal func detectDetailedWithMetrics(
        pages: [CGImage],
        options: ParseOptions
    ) async throws -> PipelineLayoutDetectionOutput {
        _ = options

        guard !pages.isEmpty else {
            return PipelineLayoutDetectionOutput(pageRegions: [], detectionCount: 0)
        }

        var pageRegions: [[PipelineLayoutRegion]] = []
        pageRegions.reserveCapacity(pages.count)
        var aggregatedTimings = PipelineLayoutStageTimings()
        var detectionCount = 0

        for (pageIndex, page) in pages.enumerated() {
            try Task.checkCancellation()
            Self.trace("detectDetailed.page.start index=\(pageIndex) size=\(page.width)x\(page.height)")
            let detailed = try await inferenceClient.detectLayoutDetailed(image: page)
            let regions = detailed.regions
            Self.trace("detectDetailed.page.done index=\(pageIndex) regions=\(regions.count)")
            pageRegions.append(regions)
            detectionCount += regions.count
            aggregatedTimings.preprocessMs += detailed.timings.preprocessMs
            aggregatedTimings.inferenceMs += detailed.timings.inferenceMs
            aggregatedTimings.postprocessMs += detailed.timings.postprocessMs
        }

        return PipelineLayoutDetectionOutput(
            pageRegions: pageRegions,
            timings: aggregatedTimings,
            detectionCount: detectionCount
        )
    }

    private nonisolated static func trace(_ message: String) {
        guard traceEnabled else {
            return
        }
        let payload = "[PPDocLayoutMLXDetector] \(message)\n"
        let data = payload.data(using: .utf8) ?? Data()
        FileHandle.standardError.write(data)
    }
}
