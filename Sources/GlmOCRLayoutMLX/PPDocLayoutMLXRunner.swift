import CoreGraphics
import Foundation
import MLX

package struct PPDocLayoutDetailedDetections: Sendable {
    package let detections: [PPDocLayoutLayoutDetection]
    package let preprocessMs: Double
    package let inferenceMs: Double
    package let postprocessMs: Double

    package init(
        detections: [PPDocLayoutLayoutDetection],
        preprocessMs: Double,
        inferenceMs: Double,
        postprocessMs: Double
    ) {
        self.detections = detections
        self.preprocessMs = preprocessMs
        self.inferenceMs = inferenceMs
        self.postprocessMs = postprocessMs
    }
}

package actor PPDocLayoutMLXRunner {
    private let modelID: String
    private let options: PPDocLayoutRuntimeOptions
    private let weightsLoader: PPDocLayoutMLXWeightsLoader

    private var modelTask: Task<PPDocLayoutModel, Error>?
    private nonisolated static let traceEnabled =
        ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIPELINE_TRACE"] == "1"

    package init(
        modelID: String,
        options: PPDocLayoutRuntimeOptions = .init()
    ) {
        self.modelID = modelID
        self.options = options
        self.weightsLoader = PPDocLayoutMLXWeightsLoader()
    }

    internal func forward(image: CGImage) async throws -> PPDocLayoutMLXPrediction {
        try Task.checkCancellation()
        Self.trace("forward.start image=\(image.width)x\(image.height)")
        let pixelValues = try layoutPreprocess(image: image)
        Self.trace("forward.preprocess shape=\(pixelValues.shape)")
        let model = try await loadedModel()
        let prediction = try layoutInference(pixelValues: pixelValues, model: model)
        Self.trace("forward.inference.done logits=\(prediction.logits.shape) boxes=\(prediction.predBoxes.shape)")
        return prediction
    }

    package func detect(
        image: CGImage,
        threshold: Float? = nil
    ) async throws -> [PPDocLayoutLayoutDetection] {
        let detailed = try await detectDetailed(
            image: image,
            threshold: threshold
        )
        return detailed.detections
    }

    package func detectDetailed(
        image: CGImage,
        threshold: Float? = nil
    ) async throws -> PPDocLayoutDetailedDetections {
        Self.trace("detect.start image=\(image.width)x\(image.height)")
        let preprocessStart = Date()
        let pixelValues = try layoutPreprocess(image: image)
        let preprocessMs = Date().timeIntervalSince(preprocessStart) * 1_000.0
        Self.trace("detect.preprocess shape=\(pixelValues.shape)")

        let inferenceStart = Date()
        let model = try await loadedModel()
        let prediction = try layoutInference(pixelValues: pixelValues, model: model)
        let inferenceMs = Date().timeIntervalSince(inferenceStart) * 1_000.0
        Self.trace(
            "detect.inference.done logits=\(prediction.logits.shape) boxes=\(prediction.predBoxes.shape) order=\(prediction.orderLogits.shape) masks=\(prediction.outMasks.shape)"
        )

        let postprocessStart = Date()
        let targetSize = CGSize(width: image.width, height: image.height)
        let effectiveThreshold = threshold ?? options.threshold
        let id2label = resolvedID2Label(model: model)
        let detections = try layoutPostprocess(
            prediction: prediction,
            targetSize: targetSize,
            threshold: effectiveThreshold,
            id2label: id2label,
            useFastBoundaryPath: options.layoutPostprocessFastPath
        )
        let postprocessMs = Date().timeIntervalSince(postprocessStart) * 1_000.0
        Self.trace("detect.postprocess.done detections=\(detections.count)")
        return PPDocLayoutDetailedDetections(
            detections: detections,
            preprocessMs: preprocessMs,
            inferenceMs: inferenceMs,
            postprocessMs: postprocessMs
        )
    }

    private func layoutPreprocess(image: CGImage) throws -> MLXArray {
        try PPDocLayoutMLXImageProcessor.pixelValues(from: image)
    }

    private func layoutInference(
        pixelValues: MLXArray,
        model: PPDocLayoutModel
    ) throws -> PPDocLayoutMLXPrediction {
        try PPDocLayoutMLXContractValidator.validateInputShape(pixelValues.shape)

        let prediction = try model.predict(pixelValues: pixelValues)

        try PPDocLayoutMLXContractValidator.validatePrediction(prediction)
        return prediction
    }

    private func layoutPostprocess(
        prediction: PPDocLayoutMLXPrediction,
        targetSize: CGSize,
        threshold: Float,
        id2label: [Int: String],
        useFastBoundaryPath: Bool
    ) throws -> [PPDocLayoutLayoutDetection] {
        let preThreshold = minimumPostprocessThreshold(
            threshold: threshold,
            thresholdByClass: options.thresholdByClass
        )

        var rawDetections = try PPDocLayoutHFPostprocess.decode(
            prediction: prediction,
            targetSize: targetSize,
            threshold: preThreshold,
            id2label: id2label,
            useFastBoundaryPath: useFastBoundaryPath
        )

        rawDetections = applyPerClassThresholdIfNeeded(
            detections: rawDetections,
            threshold: threshold,
            thresholdByClass: options.thresholdByClass,
            id2label: id2label
        )

        return PPDocLayoutLayoutPostprocess.postprocess(
            rawDetections: rawDetections,
            targetSize: targetSize,
            layoutNMS: options.layoutNMS,
            unclipRatio: options.layoutUnclipRatio,
            mergeModes: options.layoutMergeBBoxesMode,
            labelTaskMapping: options.labelTaskMapping
        )
    }

    private func resolvedID2Label(model: PPDocLayoutModel) -> [Int: String] {
        if let configured = options.id2label, !configured.isEmpty {
            return configured
        }
        if !model.id2label.isEmpty {
            return model.id2label
        }
        return PPDocLayoutMLXContract.id2label
    }

    private func minimumPostprocessThreshold(
        threshold: Float,
        thresholdByClass: [String: Float]?
    ) -> Float {
        guard let thresholdByClass, !thresholdByClass.isEmpty else {
            return threshold
        }
        return min(threshold, thresholdByClass.values.min() ?? threshold)
    }

    private func applyPerClassThresholdIfNeeded(
        detections: [PPDocLayoutRawDetection],
        threshold: Float,
        thresholdByClass: [String: Float]?,
        id2label: [Int: String]
    ) -> [PPDocLayoutRawDetection] {
        guard let thresholdByClass, !thresholdByClass.isEmpty else {
            return detections
        }

        var classThresholds: [Int: Float] = [:]
        let label2id = Dictionary(uniqueKeysWithValues: id2label.map { ($1, $0) })

        for (key, value) in thresholdByClass {
            if let classID = Int(key) {
                classThresholds[classID] = value
                continue
            }

            if let classID = label2id[key] {
                classThresholds[classID] = value
            }
        }

        return detections.filter { detection in
            let classThreshold = classThresholds[detection.clsID] ?? threshold
            return detection.score >= classThreshold
        }
    }

    private func loadedModel() async throws -> PPDocLayoutModel {
        if let modelTask {
            return try await modelTask.value
        }

        let loader = weightsLoader
        let id = modelID
        let loadTask = Task {
            let snapshot = try await loader.loadSnapshot(modelID: id)
            return PPDocLayoutModel(snapshot: snapshot)
        }

        modelTask = loadTask

        do {
            let model = try await loadTask.value
            Self.trace("loadedModel.ready modelID=\(modelID)")
            return model
        } catch {
            modelTask = nil
            Self.trace("loadedModel.failed modelID=\(modelID) error=\(error)")
            throw error
        }
    }

    private nonisolated static func trace(_ message: String) {
        guard traceEnabled else {
            return
        }
        let payload = "[PPDocLayoutMLXRunner] \(message)\n"
        let data = payload.data(using: .utf8) ?? Data()
        FileHandle.standardError.write(data)
    }
}
