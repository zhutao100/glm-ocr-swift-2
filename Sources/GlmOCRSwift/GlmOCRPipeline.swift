import CoreGraphics
import CryptoKit
import Foundation
import GlmOCRModelDelivery
import ImageIO
import UniformTypeIdentifiers

public actor GlmOCRPipeline {
    public nonisolated let config: GlmOCRConfig
    internal nonisolated let runtimeConfig: GlmOCRConfig
    private nonisolated let performanceConfig: GlmOCRPerformanceConfig
    private let pageLoader: any PipelinePageLoading
    private let layoutDetector: any PipelineLayoutDetecting
    private let regionRecognizer: any RegionRecognizer
    private let regionCropper: any PipelineRegionCropping
    private let formatter: PipelineFormatter
    private nonisolated static let pipelineTraceEnabled =
        ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIPELINE_TRACE"] == "1"

    public init(config: GlmOCRConfig) async throws {
        let modelManager: SandboxModelManager
        do {
            modelManager = try SandboxModelManager()
        } catch {
            throw GlmOCRError.modelDeliveryFailed(error.localizedDescription)
        }

        try await self.init(
            config: config,
            modelDeliveryManager: modelManager
        )
    }

    internal init(
        config: GlmOCRConfig,
        modelDeliveryManager: any ModelDeliveryManaging
    ) async throws {
        try config.validate()

        let resolved: ModelDeliveryResolvedPaths
        do {
            resolved = try await modelDeliveryManager.ensureReady(
                config: ModelDeliveryRequest(
                    recognizerModelID: config.recognizerModelID,
                    layoutModelID: config.layoutModelID
                )
            )
        } catch {
            throw GlmOCRError.modelDeliveryFailed(error.localizedDescription)
        }

        var runtimeConfig = config
        runtimeConfig.recognizerModelID = resolved.recognizerModelDirectory.path
        runtimeConfig.layoutModelID = resolved.layoutModelDirectory.path

        let performanceConfig = config.effectivePerformanceConfig
        self.config = config
        self.runtimeConfig = runtimeConfig
        self.performanceConfig = performanceConfig
        self.pageLoader = PipelinePageLoader(
            pdfDPI: runtimeConfig.pdfDPI,
            maxRenderedLongSide: runtimeConfig.pdfMaxRenderedLongSide,
            defaultMaxPages: runtimeConfig.defaultMaxPages,
            renderConcurrency: performanceConfig.pdfRenderConcurrency
        )
        self.layoutDetector = PPDocLayoutMLXDetector(config: runtimeConfig)
        self.regionRecognizer = GLMRegionRecognizer(config: runtimeConfig)
        self.regionCropper = PipelineRegionCropper()
        self.formatter = PipelineFormatter()
    }

    internal init(
        config: GlmOCRConfig,
        pageLoader: any PipelinePageLoading,
        layoutDetector: any PipelineLayoutDetecting,
        regionRecognizer: any RegionRecognizer,
        regionCropper: any PipelineRegionCropping = PipelineRegionCropper(),
        formatter: PipelineFormatter = PipelineFormatter()
    ) throws {
        try config.validate()
        let performanceConfig = config.effectivePerformanceConfig
        self.config = config
        self.runtimeConfig = config
        self.performanceConfig = performanceConfig
        self.pageLoader = pageLoader
        self.layoutDetector = layoutDetector
        self.regionRecognizer = regionRecognizer
        self.regionCropper = regionCropper
        self.formatter = formatter
    }

    public func parse(_ input: InputDocument, options: ParseOptions) async throws -> OCRDocumentResult {
        Self.trace(
            "parse.start layout=\(config.enableLayout) maxConcurrent=\(config.maxConcurrentRecognitions) batchSize=\(performanceConfig.inferenceBatchSize) inflight=\(performanceConfig.inferenceMaxInflightJobs) maxPagesOption=\(options.maxPages.map(String.init) ?? "nil") defaultMaxPages=\(config.defaultMaxPages.map(String.init) ?? "nil")"
        )
        try options.validate()
        try Task.checkCancellation()

        var warnings: [String] = []
        var timingsMs: [String: Double] = [:]
        var ocrPreprocessStats = OCRPreprocessStats()
        var inferenceSchedulerStats = OCRInferenceSchedulerStats()
        var layoutDetectionCount = 0
        let totalStart = Date()
        let effectiveMaxPages = resolvedEffectiveMaxPages(
            optionsMaxPages: options.maxPages,
            defaultMaxPages: config.defaultMaxPages
        )

        let pageLoadStart = Date()
        let pages = try pageLoader.loadPages(
            from: input,
            maxPages: options.maxPages
        )
        timingsMs["page_load"] = elapsedMilliseconds(since: pageLoadStart)
        let pageSizeSummary = pages.enumerated().map { index, page in
            "p\(index + 1)=\(page.width)x\(page.height)"
        }.joined(separator: ",")
        Self.trace("parse.pageLoad pages=\(pages.count) sizes=[\(pageSizeSummary)]")
        let pageRenderDebug = dumpPageRenderDiagnosticsIfRequested(
            input: input,
            pages: pages
        )

        guard !pages.isEmpty else {
            throw GlmOCRError.invalidConfiguration("Input document produced zero pages")
        }

        let pagesAndMarkdown: (pages: [OCRPageResult], markdown: String)
        if config.enableLayout {
            Self.trace("parse.layout.detect.start pageCount=\(pages.count)")
            let detailedDetections: [[PipelineLayoutRegion]]
            if let metricsDetector = layoutDetector as? any PipelineLayoutDetectingWithMetrics {
                let output = try await metricsDetector.detectDetailedWithMetrics(
                    pages: pages,
                    options: options
                )
                detailedDetections = output.pageRegions
                layoutDetectionCount = output.detectionCount
                timingsMs["layout_preprocess"] = output.timings.preprocessMs
                timingsMs["layout_inference"] = output.timings.inferenceMs
                timingsMs["layout_postprocess"] = output.timings.postprocessMs
            } else {
                let layoutStart = Date()
                detailedDetections = try await layoutDetector.detectDetailed(
                    pages: pages,
                    options: options
                )
                let layoutDuration = elapsedMilliseconds(since: layoutStart)
                timingsMs["layout_preprocess"] = layoutDuration
                timingsMs["layout_inference"] = layoutDuration
                timingsMs["layout_postprocess"] = layoutDuration
                layoutDetectionCount = detailedDetections.reduce(0) { $0 + $1.count }
            }

            let detectionSummary = detailedDetections.enumerated().map { index, regions in
                "p\(index + 1)=\(regions.count)"
            }.joined(separator: ",")
            Self.trace(
                "parse.layout.detect.done pageRegions=[\(detectionSummary)] preprocessMs=\(timingsMs["layout_preprocess"] ?? -1) inferenceMs=\(timingsMs["layout_inference"] ?? -1) postprocessMs=\(timingsMs["layout_postprocess"] ?? -1)"
            )

            guard detailedDetections.count == pages.count else {
                throw GlmOCRError.invalidConfiguration(
                    "Layout detector produced \(detailedDetections.count) pages for \(pages.count) input pages"
                )
            }

            let ocrPreprocessStart = Date()
            let ocrPreprocessed = try await ocrPreprocessLayoutRegions(
                pages: pages,
                detections: detailedDetections
            )
            ocrPreprocessStats = ocrPreprocessed.stats
            warnings.append(contentsOf: ocrPreprocessed.warnings)
            timingsMs["ocr_preprocess"] = elapsedMilliseconds(since: ocrPreprocessStart)
            Self.trace(
                "parse.ocrPreprocess.done jobs=\(ocrPreprocessed.recognitionJobs.count) warnings=\(ocrPreprocessed.warnings.count)"
            )

            let ocrPreprocessOnly = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_OCR_PREPROCESS_ONLY"] == "1"
            let merged: (pageRegions: [[PipelineRegionRecord]], warnings: [String])
            if ocrPreprocessOnly {
                timingsMs["ocr_inference"] = 0
                merged = (ocrPreprocessed.pageRegions, [])
                Self.trace("parse.ocrInference.skipped preprocessOnly=1")
            } else {
                Self.trace("parse.ocrInference.start queuedJobs=\(ocrPreprocessed.recognitionJobs.count)")
                let ocrInferenceStart = Date()
                let ocrInferred = try await ocrInferenceRecognizeQueuedRegions(
                    jobs: ocrPreprocessed.recognitionJobs
                )
                inferenceSchedulerStats = ocrInferred.stats
                timingsMs["ocr_inference"] = elapsedMilliseconds(since: ocrInferenceStart)
                Self.trace(
                    "parse.ocrInference.done results=\(ocrInferred.results.count) batches=\(ocrInferred.stats.batchCount) elapsedMs=\(timingsMs["ocr_inference"] ?? -1)"
                )
                merged = ocrInferenceMergeResults(
                    pageRegions: ocrPreprocessed.pageRegions,
                    inferenceResults: ocrInferred
                )
            }
            warnings.append(contentsOf: merged.warnings)
            dumpOCRPostprocessInputIfRequested(pageRegions: merged.pageRegions)

            let ocrPostprocessStart = Date()
            pagesAndMarkdown = ocrPostprocessFormatLayout(pageRegions: merged.pageRegions)
            timingsMs["ocr_postprocess"] = elapsedMilliseconds(since: ocrPostprocessStart)
            Self.trace(
                "parse.ocrPostprocess.layout.done outputPages=\(pagesAndMarkdown.pages.count) markdownChars=\(pagesAndMarkdown.markdown.count)"
            )
        } else {
            timingsMs["ocr_preprocess"] = 0
            ocrPreprocessStats = OCRPreprocessStats(
                detectionCount: 0,
                recognitionJobCount: pages.count,
                recognitionTextJobCount: pages.count,
                recognitionTableJobCount: 0,
                recognitionFormulaJobCount: 0,
                skipRegionCount: 0
            )
            Self.trace("parse.noLayout.ocrInference.start pageCount=\(pages.count)")
            let ocrInferenceStart = Date()
            let recognizedContents = try await recognizeWholePages(
                pages: pages
            )
            warnings.append(contentsOf: recognizedContents.warnings)
            inferenceSchedulerStats = recognizedContents.stats
            timingsMs["ocr_inference"] = elapsedMilliseconds(since: ocrInferenceStart)
            Self.trace(
                "parse.noLayout.ocrInference.done outputs=\(recognizedContents.contents.count) warnings=\(recognizedContents.warnings.count) batches=\(recognizedContents.stats.batchCount) elapsedMs=\(timingsMs["ocr_inference"] ?? -1)"
            )

            let ocrPostprocessStart = Date()
            pagesAndMarkdown = ocrPostprocessFormatNoLayout(contents: recognizedContents.contents)
            timingsMs["ocr_postprocess"] = elapsedMilliseconds(since: ocrPostprocessStart)
            Self.trace(
                "parse.ocrPostprocess.noLayout.done outputPages=\(pagesAndMarkdown.pages.count) markdownChars=\(pagesAndMarkdown.markdown.count)"
            )
        }

        timingsMs["total"] = elapsedMilliseconds(since: totalStart)
        var markdownOutput = options.includeMarkdown ? pagesAndMarkdown.markdown : ""

        let bundleBuild: MarkdownBundleBuildOutput?
        if config.enableLayout, config.markdownBundle.enabled, options.includeMarkdown {
            let built = await buildMarkdownBundle(
                pages: pages,
                pageResults: pagesAndMarkdown.pages,
                markdown: markdownOutput
            )
            markdownOutput = built.rewrittenMarkdown
            warnings.append(contentsOf: built.warnings)
            bundleBuild = built
        } else {
            bundleBuild = nil
        }

        let metadata: [String: String]
        if options.includeDiagnostics {
            metadata = [
                "layoutEnabled": config.enableLayout ? "true" : "false",
                "markdownBundleEnabled": config.markdownBundle.enabled ? "true" : "false",
                "markdownBundleGenerated": bundleBuild == nil ? "false" : "true",
                "markdownBundleFigureCount": String(bundleBuild?.figures.count ?? 0),
                "pageCount": String(pages.count),
                "layoutDetectionCount": String(layoutDetectionCount),
                "maxConcurrentRecognitions": String(config.maxConcurrentRecognitions),
                "maxPagesOption": options.maxPages.map(String.init) ?? "nil",
                "defaultMaxPages": config.defaultMaxPages.map(String.init) ?? "nil",
                "effectiveMaxPages": effectiveMaxPages.map(String.init) ?? "nil",
                "pdfDPI": String(config.pdfDPI),
                "pdfMaxRenderedLongSide": String(config.pdfMaxRenderedLongSide),
                "inferenceBatchSize": String(performanceConfig.inferenceBatchSize),
                "inferenceBatchMaxWaitMs": String(performanceConfig.inferenceBatchMaxWaitMs),
                "inferenceMaxInflightJobs": String(performanceConfig.inferenceMaxInflightJobs),
                "pdfRenderConcurrency": String(performanceConfig.pdfRenderConcurrency),
                "ocrPreprocessConcurrency": String(performanceConfig.ocrPreprocessConcurrency),
                "bundleEncodeConcurrency": String(performanceConfig.bundleEncodeConcurrency),
                "layoutPostprocessFastPath": performanceConfig.layoutPostprocessFastPath ? "true" : "false",
                "recognitionJobCount": String(ocrPreprocessStats.recognitionJobCount),
                "recognitionTextJobCount": String(ocrPreprocessStats.recognitionTextJobCount),
                "recognitionTableJobCount": String(ocrPreprocessStats.recognitionTableJobCount),
                "recognitionFormulaJobCount": String(ocrPreprocessStats.recognitionFormulaJobCount),
                "skipRegionCount": String(ocrPreprocessStats.skipRegionCount),
                "detectionCount": String(ocrPreprocessStats.detectionCount),
                "inferenceBucketCount": String(inferenceSchedulerStats.bucketCount),
                "inferenceBatchCount": String(inferenceSchedulerStats.batchCount),
                "inferenceMaxBatchSize": String(inferenceSchedulerStats.maxBatchSize),
                "inferenceAverageBatchSize": String(format: "%.2f", inferenceSchedulerStats.averageBatchSize),
                "inferenceQueuedJobCount": String(inferenceSchedulerStats.queuedJobCount),
                "noLayoutPromptHash": truncatedSHA256(config.prompts.noLayoutPrompt),
                "textPromptHash": truncatedSHA256(config.prompts.textPrompt),
                "tablePromptHash": truncatedSHA256(config.prompts.tablePrompt),
                "formulaPromptHash": truncatedSHA256(config.prompts.formulaPrompt),
                "pageRenderDebugDump": pageRenderDebug.path ?? "nil",
                "pageRenderDebugCount": pageRenderDebug.path == nil ? "0" : String(pageRenderDebug.count),
            ]
        } else {
            metadata = [:]
        }

        let diagnostics: ParseDiagnostics
        if options.includeDiagnostics {
            diagnostics = ParseDiagnostics(
                warnings: warnings,
                timingsMs: timingsMs,
                metadata: metadata
            )
        } else {
            diagnostics = ParseDiagnostics()
        }

        let markdownBundle: OCRMarkdownBundle?
        if let bundleBuild {
            let documentJSON = makeMarkdownBundleDocumentJSON(
                pages: pagesAndMarkdown.pages,
                diagnostics: diagnostics,
                figures: bundleBuild.figures
            )
            markdownBundle = OCRMarkdownBundle(
                rewrittenMarkdown: markdownOutput,
                documentJSON: documentJSON,
                markdownFileName: config.markdownBundle.markdownFileName,
                jsonFileName: config.markdownBundle.jsonFileName,
                figuresDirectoryName: config.markdownBundle.figuresDirectoryName,
                figures: bundleBuild.figures
            )
        } else {
            markdownBundle = nil
        }

        return OCRDocumentResult(
            pages: pagesAndMarkdown.pages,
            markdown: markdownOutput,
            diagnostics: diagnostics,
            markdownBundle: markdownBundle
        )
    }

    private nonisolated static func trace(_ message: String) {
        guard pipelineTraceEnabled else {
            return
        }
        let payload = "[GlmOCRPipeline] \(message)\n"
        let data = payload.data(using: .utf8) ?? Data()
        FileHandle.standardError.write(data)
    }

    private func recognizeWholePages(
        pages: [CGImage]
    ) async throws -> (contents: [String], warnings: [String], stats: OCRInferenceSchedulerStats) {
        guard !pages.isEmpty else {
            return ([], [], OCRInferenceSchedulerStats())
        }

        var results = Array(repeating: "", count: pages.count)
        var warnings: [String] = []
        let jobs = pages.enumerated().map { index, image in
            PipelineRecognitionJob(
                key: PipelineRecognitionJobKey(pageIndex: index, regionPosition: 0),
                image: image,
                task: .text,
                promptOverride: config.prompts.noLayoutPrompt
            )
        }

        let inference = try await ocrInferenceRecognizeQueuedRegions(jobs: jobs)
        for job in jobs {
            switch inference.results[job.key] {
            case .success(let text):
                results[job.key.pageIndex] = text
            case .failure(let error):
                results[job.key.pageIndex] = ""
                warnings.append("page[\(job.key.pageIndex)] recognition failed: \(error)")
            case nil:
                results[job.key.pageIndex] = ""
                warnings.append("page[\(job.key.pageIndex)] recognition failed: missing result")
            }
        }

        return (results, warnings, inference.stats)
    }

    private func ocrPreprocessLayoutRegions(
        pages: [CGImage],
        detections: [[PipelineLayoutRegion]]
    ) async throws -> OCRPreprocessOutput {
        let pageUnits = ocrPreprocessBuildPageUnits(pages: pages, detections: detections)
        return try await ocrPreprocessCropAndQueueRegions(units: pageUnits)
    }

    private func ocrPreprocessBuildPageUnits(
        pages: [CGImage],
        detections: [[PipelineLayoutRegion]]
    ) -> [OCRPreprocessPageUnit] {
        pages.indices.map { pageIndex in
            return OCRPreprocessPageUnit(
                pageIndex: pageIndex,
                image: pages[pageIndex],
                detections: detections[pageIndex]
            )
        }
    }

    private func ocrPreprocessCropAndQueueRegions(
        units: [OCRPreprocessPageUnit]
    ) async throws -> OCRPreprocessOutput {
        guard !units.isEmpty else {
            return OCRPreprocessOutput(
                pageRegions: [],
                recognitionJobs: [],
                warnings: [],
                stats: OCRPreprocessStats()
            )
        }

        var pageRegions: [[PipelineRegionRecord]] = Array(repeating: [], count: units.count)
        var recognitionJobs: [PipelineRecognitionJob] = []
        var warnings: [String] = []
        var stats = OCRPreprocessStats()
        var ocrPreprocessDebugEntries: [[String: Any]] = []
        let ocrPreprocessDumpPath = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_OCR_PREPROCESS_DUMP"]
        let shouldDumpOCRPreprocess =
            ocrPreprocessDumpPath?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty == false

        let limiter = AsyncLimiter(limit: performanceConfig.ocrPreprocessConcurrency)
        let cropper = regionCropper
        let pageOutputs: [OCRPreprocessPageOutput] = try await withThrowingTaskGroup(of: OCRPreprocessPageOutput.self) {
            group in
            for unit in units {
                group.addTask {
                    try await limiter.withPermit {
                        try Task.checkCancellation()
                        return try Self.preprocessPage(
                            unit: unit,
                            cropper: cropper,
                            shouldDumpDebugEntries: shouldDumpOCRPreprocess
                        )
                    }
                }
            }

            var collected: [OCRPreprocessPageOutput] = []
            collected.reserveCapacity(units.count)
            for try await output in group {
                collected.append(output)
            }
            return collected.sorted { lhs, rhs in
                lhs.pageIndex < rhs.pageIndex
            }
        }

        for output in pageOutputs {
            pageRegions[output.pageIndex] = output.pageRegions
            recognitionJobs.append(contentsOf: output.recognitionJobs)
            warnings.append(contentsOf: output.warnings)
            ocrPreprocessDebugEntries.append(contentsOf: output.debugEntries)
            stats.detectionCount += output.detectionCount
            stats.skipRegionCount += output.skipRegionCount
            stats.recognitionTextJobCount += output.recognitionTextJobCount
            stats.recognitionTableJobCount += output.recognitionTableJobCount
            stats.recognitionFormulaJobCount += output.recognitionFormulaJobCount
        }

        stats.recognitionJobCount =
            stats.recognitionTextJobCount
            + stats.recognitionTableJobCount
            + stats.recognitionFormulaJobCount

        if let ocrPreprocessDumpPath,
            !ocrPreprocessDumpPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            let data = try? JSONSerialization.data(withJSONObject: ocrPreprocessDebugEntries, options: [.prettyPrinted])
        {
            try? data.write(to: URL(fileURLWithPath: ocrPreprocessDumpPath), options: .atomic)
        }

        return OCRPreprocessOutput(
            pageRegions: pageRegions,
            recognitionJobs: recognitionJobs,
            warnings: warnings,
            stats: stats
        )
    }

    private nonisolated static func preprocessPage(
        unit: OCRPreprocessPageUnit,
        cropper: any PipelineRegionCropping,
        shouldDumpDebugEntries: Bool
    ) throws -> OCRPreprocessPageOutput {
        var pageRegions: [PipelineRegionRecord] = []
        pageRegions.reserveCapacity(unit.detections.count)
        var recognitionJobs: [PipelineRecognitionJob] = []
        recognitionJobs.reserveCapacity(unit.detections.count)
        var warnings: [String] = []
        var debugEntries: [[String: Any]] = []
        var output = OCRPreprocessPageOutput(pageIndex: unit.pageIndex)

        for detection in unit.detections {
            if detection.task == .abandon {
                continue
            }

            output.detectionCount += 1
            if detection.task == .skip {
                output.skipRegionCount += 1
            }

            let regionPosition = pageRegions.count
            var record = PipelineRegionRecord(
                index: detection.index,
                nativeLabel: detection.label,
                task: detection.task,
                bbox2D: detection.bbox2D,
                content: nil
            )

            let debugEntry: [String: Any]? =
                shouldDumpDebugEntries
                ? [
                    "pageIndex": unit.pageIndex,
                    "detectionIndex": detection.index,
                    "order": detection.order,
                    "regionPosition": regionPosition,
                    "task": detection.task.rawValue,
                    "nativeLabel": detection.label,
                    "bbox2D": detection.bbox2D,
                    "polygon2D": detection.polygon2D,
                ] : nil

            if let ocrTask = detection.task.ocrTask {
                do {
                    let cropResult = try cropper.cropRegion(
                        page: unit.image,
                        bbox2D: detection.bbox2D,
                        polygon2D: detection.polygon2D,
                        pageIndex: unit.pageIndex,
                        regionIndex: detection.index
                    )

                    if let warning = cropResult.warning {
                        warnings.append(warning)
                    }

                    recognitionJobs.append(
                        PipelineRecognitionJob(
                            key: PipelineRecognitionJobKey(
                                pageIndex: unit.pageIndex,
                                regionPosition: regionPosition
                            ),
                            image: cropResult.image,
                            task: ocrTask
                        )
                    )

                    switch ocrTask {
                    case .text:
                        output.recognitionTextJobCount += 1
                    case .table:
                        output.recognitionTableJobCount += 1
                    case .formula:
                        output.recognitionFormulaJobCount += 1
                    }

                    if var debugEntry {
                        if let cropMetadata = Self.cropDebugMetadata(for: cropResult.image) {
                            debugEntry["cropWidth"] = cropMetadata.width
                            debugEntry["cropHeight"] = cropMetadata.height
                            debugEntry["cropChannels"] = cropMetadata.channels
                            debugEntry["cropPixelSHA256"] = cropMetadata.sha256
                        } else {
                            debugEntry["cropWidth"] = NSNull()
                            debugEntry["cropHeight"] = NSNull()
                            debugEntry["cropChannels"] = NSNull()
                            debugEntry["cropPixelSHA256"] = NSNull()
                        }
                        debugEntries.append(debugEntry)
                    }
                } catch {
                    record.content = ""
                    warnings.append(
                        "page[\(unit.pageIndex)] region[\(detection.index)] crop failed: \(error)"
                    )
                    if var debugEntry {
                        debugEntry["cropWidth"] = NSNull()
                        debugEntry["cropHeight"] = NSNull()
                        debugEntry["cropChannels"] = NSNull()
                        debugEntry["cropPixelSHA256"] = NSNull()
                        debugEntries.append(debugEntry)
                    }
                }
            } else if var debugEntry {
                debugEntry["cropWidth"] = NSNull()
                debugEntry["cropHeight"] = NSNull()
                debugEntry["cropChannels"] = NSNull()
                debugEntry["cropPixelSHA256"] = NSNull()
                debugEntries.append(debugEntry)
            }

            pageRegions.append(record)
        }

        output.pageRegions = pageRegions
        output.recognitionJobs = recognitionJobs
        output.warnings = warnings
        output.debugEntries = debugEntries
        return output
    }

    private func ocrInferenceRecognizeQueuedRegions(
        jobs: [PipelineRecognitionJob]
    ) async throws -> OCRInferenceOutput {
        guard !jobs.isEmpty else {
            return OCRInferenceOutput(
                results: [:],
                stats: OCRInferenceSchedulerStats(
                    requestedBatchSize: performanceConfig.inferenceBatchSize,
                    maxInflightJobs: performanceConfig.inferenceMaxInflightJobs,
                    batchMaxWaitMs: performanceConfig.inferenceBatchMaxWaitMs
                )
            )
        }

        let batchedJobs = jobs.map { job in
            OCRInferenceBatchJob(
                key: job.key,
                image: job.image,
                task: job.task,
                prompt: resolvedPrompt(for: job)
            )
        }

        let scheduler = OCRInferenceScheduler(
            requestedBatchSize: performanceConfig.inferenceBatchSize,
            maxInflightJobs: performanceConfig.inferenceMaxInflightJobs,
            batchMaxWaitMs: performanceConfig.inferenceBatchMaxWaitMs
        )
        let schedulerResult = try await scheduler.run(
            jobs: batchedJobs,
            executor: { [self] batch in
                try await executeInferenceBatch(batch)
            }
        )

        return OCRInferenceOutput(
            results: schedulerResult.results,
            stats: schedulerResult.stats
        )
    }

    private func ocrInferenceMergeResults(
        pageRegions: [[PipelineRegionRecord]],
        inferenceResults: OCRInferenceOutput
    ) -> (pageRegions: [[PipelineRegionRecord]], warnings: [String]) {
        var mergedPageRegions = pageRegions
        var warnings: [String] = []

        for (key, result) in inferenceResults.results {
            guard key.pageIndex < mergedPageRegions.count,
                key.regionPosition < mergedPageRegions[key.pageIndex].count
            else {
                continue
            }

            switch result {
            case .success(let text):
                mergedPageRegions[key.pageIndex][key.regionPosition].content = text
            case .failure(let error):
                mergedPageRegions[key.pageIndex][key.regionPosition].content = ""
                let regionIndex = mergedPageRegions[key.pageIndex][key.regionPosition].index
                warnings.append(
                    "page[\(key.pageIndex)] region[\(regionIndex)] recognition failed: \(error)"
                )
            }
        }

        return (mergedPageRegions, warnings)
    }

    private func ocrPostprocessFormatLayout(
        pageRegions: [[PipelineRegionRecord]]
    ) -> (pages: [OCRPageResult], markdown: String) {
        formatter.formatLayout(pageRegions: pageRegions)
    }

    private func ocrPostprocessFormatNoLayout(
        contents: [String]
    ) -> (pages: [OCRPageResult], markdown: String) {
        formatter.formatNoLayout(contents: contents)
    }

    private func recognize(
        job: PipelineRecognitionJob,
        recognizer: any RegionRecognizer,
        promptRecognizer: (any PromptRegionRecognizing)?
    ) async throws -> String {
        if let promptRecognizer,
            let prompt = job.promptOverride?.trimmingCharacters(in: .whitespacesAndNewlines),
            !prompt.isEmpty
        {
            return try await promptRecognizer.recognize(job.image, prompt: prompt)
        }

        return try await recognizer.recognize(job.image, task: job.task)
    }

    private func resolvedPrompt(for job: PipelineRecognitionJob) -> String {
        if let prompt = job.promptOverride?.trimmingCharacters(in: .whitespacesAndNewlines),
            !prompt.isEmpty
        {
            return prompt
        }
        return RecognitionPromptMapper.prompt(for: job.task, prompts: config.prompts)
    }

    private func executeInferenceBatch(
        _ batch: [OCRInferenceBatchJob]
    ) async throws -> [Result<String, Error>] {
        try Task.checkCancellation()
        guard !batch.isEmpty else {
            return []
        }

        let recognizer = regionRecognizer
        let promptRecognizer = recognizer as? PromptRegionRecognizing
        let batchPromptRecognizer = recognizer as? BatchPromptRegionRecognizing

        if let batchPromptRecognizer {
            do {
                let outputs = try await batchPromptRecognizer.recognizeBatch(
                    batch.map {
                        PromptRecognitionRequest(
                            image: $0.image,
                            prompt: $0.prompt
                        )
                    }
                )
                guard outputs.count == batch.count else {
                    let error = GlmOCRError.invalidConfiguration(
                        "Batch recognizer returned \(outputs.count) results for \(batch.count) requests"
                    )
                    return batch.map { _ in
                        Result<String, Error>.failure(error)
                    }
                }
                return outputs.map { Result<String, Error>.success($0) }
            } catch {
                if error is CancellationError {
                    throw error
                }
                return batch.map { _ in
                    Result<String, Error>.failure(error)
                }
            }
        }

        var fallbackResults: [Result<String, Error>] = []
        fallbackResults.reserveCapacity(batch.count)
        for entry in batch {
            let job = PipelineRecognitionJob(
                key: entry.key,
                image: entry.image,
                task: entry.task,
                promptOverride: entry.prompt
            )
            do {
                let output = try await recognize(
                    job: job,
                    recognizer: recognizer,
                    promptRecognizer: promptRecognizer
                )
                fallbackResults.append(.success(output))
            } catch {
                if error is CancellationError {
                    throw error
                }
                fallbackResults.append(.failure(error))
            }
        }
        return fallbackResults
    }

    private func buildMarkdownBundle(
        pages: [CGImage],
        pageResults: [OCRPageResult],
        markdown: String
    ) async -> MarkdownBundleBuildOutput {
        let figureCandidates = collectFigureCandidates(from: pageResults)
        guard !figureCandidates.isEmpty else {
            return MarkdownBundleBuildOutput(
                rewrittenMarkdown: markdown,
                figures: [],
                warnings: []
            )
        }

        let limiter = AsyncLimiter(limit: performanceConfig.bundleEncodeConcurrency)
        let figureFormat = config.markdownBundle.figureFormat
        let figureNamingScheme = config.markdownBundle.figureNamingScheme
        let figuresDirectoryName = config.markdownBundle.figuresDirectoryName
        let heicCompressionQuality = config.markdownBundle.heicCompressionQuality

        let outputs: [MarkdownBundleFigureOutput] = await withTaskGroup(of: MarkdownBundleFigureOutput.self) { group in
            for (candidateIndex, candidate) in figureCandidates.enumerated() {
                group.addTask {
                    (try? await limiter.withPermit {
                        Self.buildFigureOutput(
                            candidateIndex: candidateIndex,
                            candidate: candidate,
                            pages: pages,
                            figureFormat: figureFormat,
                            figureNamingScheme: figureNamingScheme,
                            figuresDirectoryName: figuresDirectoryName,
                            heicCompressionQuality: heicCompressionQuality
                        )
                    })
                        ?? MarkdownBundleFigureOutput(
                            candidateIndex: candidateIndex,
                            figure: nil,
                            warning:
                                "bundle.figure encode failed: page=\(candidate.pageIndex) region=\(candidate.regionIndex)"
                        )
                }
            }

            var collected: [MarkdownBundleFigureOutput] = []
            collected.reserveCapacity(figureCandidates.count)
            for await output in group {
                collected.append(output)
            }
            return collected.sorted { lhs, rhs in
                lhs.candidateIndex < rhs.candidateIndex
            }
        }

        var figures: [OCRFigureAsset] = []
        figures.reserveCapacity(outputs.count)
        var figurePathsByCandidate: [String?] = Array(repeating: nil, count: figureCandidates.count)
        var warnings: [String] = []

        for output in outputs {
            if let warning = output.warning {
                warnings.append(warning)
            }
            if let figure = output.figure {
                figures.append(figure)
                figurePathsByCandidate[output.candidateIndex] = figure.relativePath
            }
        }

        let rewritten = rewriteMarkdownFigureMarkers(
            markdown: markdown,
            figurePathsByCandidate: figurePathsByCandidate
        )
        warnings.append(contentsOf: rewritten.warnings)

        return MarkdownBundleBuildOutput(
            rewrittenMarkdown: rewritten.markdown,
            figures: figures,
            warnings: warnings
        )
    }

    private func collectFigureCandidates(from pageResults: [OCRPageResult]) -> [FigureCandidate] {
        var candidates: [FigureCandidate] = []
        for (pageIndex, page) in pageResults.enumerated() {
            let sortedRegions = page.regions.sorted { lhs, rhs in
                lhs.index < rhs.index
            }
            var pageImageIndex = 0
            for region in sortedRegions {
                guard region.label == "image", let bbox = region.bbox2D, bbox.count == 4 else {
                    continue
                }
                candidates.append(
                    FigureCandidate(
                        pageIndex: pageIndex,
                        pageImageIndex: pageImageIndex,
                        regionIndex: region.index,
                        label: region.label,
                        bbox2D: bbox
                    )
                )
                pageImageIndex += 1
            }
        }
        return candidates
    }

    private nonisolated static func buildFigureOutput(
        candidateIndex: Int,
        candidate: FigureCandidate,
        pages: [CGImage],
        figureFormat: GlmOCRFigureFormat,
        figureNamingScheme: GlmOCRFigureNamingScheme,
        figuresDirectoryName: String,
        heicCompressionQuality: Double
    ) -> MarkdownBundleFigureOutput {
        guard candidate.pageIndex >= 0, candidate.pageIndex < pages.count else {
            return MarkdownBundleFigureOutput(
                candidateIndex: candidateIndex,
                figure: nil,
                warning:
                    "bundle.figure page index out of range: page=\(candidate.pageIndex) region=\(candidate.regionIndex)"
            )
        }

        guard
            let cropped = Self.cropFigure(
                from: pages[candidate.pageIndex],
                bbox2D: candidate.bbox2D
            )
        else {
            return MarkdownBundleFigureOutput(
                candidateIndex: candidateIndex,
                figure: nil,
                warning: "bundle.figure crop failed: page=\(candidate.pageIndex) region=\(candidate.regionIndex)"
            )
        }

        guard
            let encoded = Self.encodeFigure(
                image: cropped,
                format: figureFormat,
                heicCompressionQuality: heicCompressionQuality
            )
        else {
            return MarkdownBundleFigureOutput(
                candidateIndex: candidateIndex,
                figure: nil,
                warning: "bundle.figure encode failed: page=\(candidate.pageIndex) region=\(candidate.regionIndex)"
            )
        }

        let fileName: String
        switch figureNamingScheme {
        case .pageRegionPadded:
            fileName =
                "page_\(Self.padded(candidate.pageIndex + 1, width: 4))_region_\(Self.padded(candidate.regionIndex, width: 4)).\(figureFormat.fileExtension)"
        case .upstreamCropped:
            fileName = "cropped_page\(candidate.pageIndex)_idx\(candidate.pageImageIndex).\(figureFormat.fileExtension)"
        }
        let relativePath = "\(figuresDirectoryName)/\(fileName)"
        let figure = OCRFigureAsset(
            pageIndex: candidate.pageIndex,
            regionIndex: candidate.regionIndex,
            label: candidate.label,
            bbox2D: candidate.bbox2D,
            altText: "",
            fileName: fileName,
            relativePath: relativePath,
            widthPX: cropped.width,
            heightPX: cropped.height,
            mimeType: figureFormat.mimeType,
            sha256: Self.fullSHA256Hex(encoded),
            data: encoded
        )
        return MarkdownBundleFigureOutput(
            candidateIndex: candidateIndex,
            figure: figure,
            warning: nil
        )
    }

    private nonisolated static func cropFigure(from page: CGImage, bbox2D: [Int]) -> CGImage? {
        guard bbox2D.count == 4 else {
            return nil
        }

        let width = page.width
        let height = page.height
        guard width > 0, height > 0 else {
            return nil
        }

        let x1 = Self.clamp(Self.denormalize(value: bbox2D[0], size: width), minimum: 0, maximum: width)
        let y1 = Self.clamp(Self.denormalize(value: bbox2D[1], size: height), minimum: 0, maximum: height)
        let x2 = Self.clamp(Self.denormalize(value: bbox2D[2], size: width), minimum: 0, maximum: width)
        let y2 = Self.clamp(Self.denormalize(value: bbox2D[3], size: height), minimum: 0, maximum: height)
        guard x1 < x2, y1 < y2 else {
            return nil
        }

        let rect = CGRect(
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1
        )
        return page.cropping(to: rect)
    }

    private nonisolated static func encodeFigure(
        image: CGImage,
        format: GlmOCRFigureFormat,
        heicCompressionQuality: Double
    ) -> Data? {
        switch format {
        case .heic:
            let data = NSMutableData()
            guard
                let destination = CGImageDestinationCreateWithData(
                    data,
                    UTType.heic.identifier as CFString,
                    1,
                    nil
                )
            else {
                return nil
            }

            let options: [CFString: Any] = [
                kCGImageDestinationLossyCompressionQuality: heicCompressionQuality
            ]
            CGImageDestinationAddImage(destination, image, options as CFDictionary)
            guard CGImageDestinationFinalize(destination) else {
                return nil
            }
            return data as Data
        case .jpeg:
            let data = NSMutableData()
            guard
                let destination = CGImageDestinationCreateWithData(
                    data,
                    UTType.jpeg.identifier as CFString,
                    1,
                    nil
                )
            else {
                return nil
            }

            let options: [CFString: Any] = [
                kCGImageDestinationLossyCompressionQuality: heicCompressionQuality
            ]
            CGImageDestinationAddImage(destination, image, options as CFDictionary)
            guard CGImageDestinationFinalize(destination) else {
                return nil
            }
            return data as Data
        }
    }

    private func rewriteMarkdownFigureMarkers(
        markdown: String,
        figurePathsByCandidate: [String?]
    ) -> (markdown: String, warnings: [String]) {
        guard
            let regex = try? NSRegularExpression(
                pattern: #"\!\[([^\]]*)]\(page\s*=\s*\d+\s*,\s*bbox\s*=\s*\[\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\]\)"#,
                options: []
            )
        else {
            return (markdown, ["bundle.figure regex compile failed"])
        }

        let source = markdown as NSString
        let fullRange = NSRange(location: 0, length: source.length)
        let matches = regex.matches(in: markdown, options: [], range: fullRange)

        var warnings: [String] = []
        if matches.count != figurePathsByCandidate.count {
            warnings.append(
                "bundle.figure marker count mismatch: markers=\(matches.count) figures=\(figurePathsByCandidate.count)"
            )
        }

        let mutable = NSMutableString(string: markdown)
        if matches.isEmpty {
            return (markdown, warnings)
        }
        for index in stride(from: matches.count - 1, through: 0, by: -1) {
            let match = matches[index]
            guard index < figurePathsByCandidate.count else {
                continue
            }
            guard let relativePath = figurePathsByCandidate[index] else {
                continue
            }
            let altText: String
            let altRange = match.range(at: 1)
            if altRange.location != NSNotFound, altRange.length > 0 {
                altText = source.substring(with: altRange)
            } else {
                altText = ""
            }
            mutable.replaceCharacters(in: match.range, with: "![\(altText)](\(relativePath))")
        }

        return (mutable as String, warnings)
    }

    private func makeMarkdownBundleDocumentJSON(
        pages: [OCRPageResult],
        diagnostics: ParseDiagnostics,
        figures: [OCRFigureAsset]
    ) -> String {
        let sidecar = MarkdownBundleSidecar(
            schemaVersion: 1,
            markdownPath: config.markdownBundle.markdownFileName,
            figuresDirectory: config.markdownBundle.figuresDirectoryName,
            pages: pages,
            diagnostics: diagnostics,
            figures: figures.map {
                MarkdownBundleFigureRecord(
                    pageIndex: $0.pageIndex,
                    regionIndex: $0.regionIndex,
                    label: $0.label,
                    bbox2D: $0.bbox2D,
                    altText: $0.altText,
                    fileName: $0.fileName,
                    relativePath: $0.relativePath,
                    widthPX: $0.widthPX,
                    heightPX: $0.heightPX,
                    mimeType: $0.mimeType,
                    sha256: $0.sha256
                )
            }
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(sidecar),
            let json = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }

        return json
    }

    private func denormalize(value: Int, size: Int) -> Int {
        Self.denormalize(value: value, size: size)
    }

    private nonisolated static func denormalize(value: Int, size: Int) -> Int {
        Int((Double(value) * Double(size)) / 1000.0)
    }

    private func clamp(_ value: Int, minimum: Int, maximum: Int) -> Int {
        Self.clamp(value, minimum: minimum, maximum: maximum)
    }

    private nonisolated static func clamp(_ value: Int, minimum: Int, maximum: Int) -> Int {
        min(max(value, minimum), maximum)
    }

    private func padded(_ value: Int, width: Int) -> String {
        Self.padded(value, width: width)
    }

    private nonisolated static func padded(_ value: Int, width: Int) -> String {
        let text = String(value)
        guard text.count < width else {
            return text
        }
        return String(repeating: "0", count: width - text.count) + text
    }

    private func fullSHA256Hex(_ data: Data) -> String {
        Self.fullSHA256Hex(data)
    }

    private nonisolated static func fullSHA256Hex(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        let hexDigits = Array("0123456789abcdef")
        return digest.map { byte in
            let high = Int(byte / 16)
            let low = Int(byte % 16)
            return String([hexDigits[high], hexDigits[low]])
        }.joined()
    }

    private func elapsedMilliseconds(since start: Date) -> Double {
        Date().timeIntervalSince(start) * 1000.0
    }

    private func resolvedEffectiveMaxPages(
        optionsMaxPages: Int?,
        defaultMaxPages: Int?
    ) -> Int? {
        // Page cap semantics: if both are present, the smaller cap wins.
        if let optionsMaxPages, let defaultMaxPages {
            return min(optionsMaxPages, defaultMaxPages)
        }
        return optionsMaxPages ?? defaultMaxPages
    }

    private func dumpPageRenderDiagnosticsIfRequested(
        input: InputDocument,
        pages: [CGImage]
    ) -> (path: String?, count: Int) {
        guard case .pdfData = input else {
            return (nil, 0)
        }
        guard
            let dumpPath = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PAGE_RENDER_DUMP"]?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !dumpPath.isEmpty
        else {
            return (nil, 0)
        }

        let entries: [[String: Any]] = pages.enumerated().map { index, image in
            var payload: [String: Any] = [
                "pageIndex": index,
                "width": image.width,
                "height": image.height,
            ]
            if let metadata = Self.cropDebugMetadata(for: image) {
                payload["rgbSHA256"] = metadata.sha256
            } else {
                payload["rgbSHA256"] = NSNull()
            }
            return payload
        }

        if JSONSerialization.isValidJSONObject(entries),
            let data = try? JSONSerialization.data(withJSONObject: entries, options: [.prettyPrinted])
        {
            try? data.write(to: URL(fileURLWithPath: dumpPath), options: .atomic)
        }

        return (dumpPath, entries.count)
    }

    private func truncatedSHA256(_ value: String, length: Int = 16) -> String {
        let digest = SHA256.hash(data: Data(value.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(max(1, length)))
    }

    private nonisolated static func cropDebugMetadata(
        for image: CGImage
    ) -> (width: Int, height: Int, channels: Int, sha256: String)? {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else {
            return nil
        }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        guard
            let context = CGContext(
                data: &rgba,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        else {
            return nil
        }

        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            let rgbaOffset = pixel * 4
            let rgbOffset = pixel * 3
            rgb[rgbOffset] = rgba[rgbaOffset]
            rgb[rgbOffset + 1] = rgba[rgbaOffset + 1]
            rgb[rgbOffset + 2] = rgba[rgbaOffset + 2]
        }

        let digest = SHA256.hash(data: Data(rgb))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return (width, height, 3, hex)
    }

    private func dumpOCRPostprocessInputIfRequested(pageRegions: [[PipelineRegionRecord]]) {
        guard
            let dumpPath = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_OCR_POSTPROCESS_INPUT_DUMP"]?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !dumpPath.isEmpty
        else {
            return
        }

        let payload: [[[String: Any]]] = pageRegions.map { page in
            page.map { region in
                var entry: [String: Any] = [
                    "index": region.index,
                    "nativeLabel": region.nativeLabel,
                    "task": region.task.rawValue,
                ]
                if let bbox2D = region.bbox2D {
                    entry["bbox2D"] = bbox2D
                } else {
                    entry["bbox2D"] = NSNull()
                }
                if let content = region.content {
                    entry["content"] = content
                } else {
                    entry["content"] = NSNull()
                }
                return entry
            }
        }

        guard JSONSerialization.isValidJSONObject(payload),
            let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        else {
            return
        }

        try? data.write(to: URL(fileURLWithPath: dumpPath), options: .atomic)
    }
}

private struct FigureCandidate: Sendable {
    let pageIndex: Int
    let pageImageIndex: Int
    let regionIndex: Int
    let label: String
    let bbox2D: [Int]
}

private struct MarkdownBundleBuildOutput: Sendable {
    let rewrittenMarkdown: String
    let figures: [OCRFigureAsset]
    let warnings: [String]
}

private struct MarkdownBundleSidecar: Codable {
    let schemaVersion: Int
    let markdownPath: String
    let figuresDirectory: String
    let pages: [OCRPageResult]
    let diagnostics: ParseDiagnostics
    let figures: [MarkdownBundleFigureRecord]
}

private struct MarkdownBundleFigureRecord: Codable {
    let pageIndex: Int
    let regionIndex: Int
    let label: String
    let bbox2D: [Int]
    let altText: String
    let fileName: String
    let relativePath: String
    let widthPX: Int
    let heightPX: Int
    let mimeType: String
    let sha256: String
}

private struct OCRPreprocessPageUnit: @unchecked Sendable {
    let pageIndex: Int
    let image: CGImage
    let detections: [PipelineLayoutRegion]
}

private struct OCRPreprocessOutput: @unchecked Sendable {
    let pageRegions: [[PipelineRegionRecord]]
    let recognitionJobs: [PipelineRecognitionJob]
    let warnings: [String]
    let stats: OCRPreprocessStats
}

private struct OCRInferenceOutput: @unchecked Sendable {
    let results: [PipelineRecognitionJobKey: Result<String, Error>]
    let stats: OCRInferenceSchedulerStats
}

private struct OCRPreprocessStats: Sendable {
    var detectionCount: Int = 0
    var recognitionJobCount: Int = 0
    var recognitionTextJobCount: Int = 0
    var recognitionTableJobCount: Int = 0
    var recognitionFormulaJobCount: Int = 0
    var skipRegionCount: Int = 0
}

private struct OCRPreprocessPageOutput: @unchecked Sendable {
    let pageIndex: Int
    var pageRegions: [PipelineRegionRecord] = []
    var recognitionJobs: [PipelineRecognitionJob] = []
    var warnings: [String] = []
    var debugEntries: [[String: Any]] = []
    var detectionCount: Int = 0
    var recognitionTextJobCount: Int = 0
    var recognitionTableJobCount: Int = 0
    var recognitionFormulaJobCount: Int = 0
    var skipRegionCount: Int = 0
}

private struct MarkdownBundleFigureOutput: Sendable {
    let candidateIndex: Int
    let figure: OCRFigureAsset?
    let warning: String?
}
