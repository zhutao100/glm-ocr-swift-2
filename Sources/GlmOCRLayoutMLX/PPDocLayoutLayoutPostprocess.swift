import CoreGraphics
import Foundation

package enum PPDocLayoutTask: String, Sendable, Equatable {
    case text
    case table
    case formula
    case skip
    case abandon
}

package struct PPDocLayoutLayoutDetection: Sendable, Equatable {
    package let index: Int
    package let clsID: Int
    package let label: String
    package let task: PPDocLayoutTask
    package let score: Float
    package let bbox: [Double]
    package let bbox2D: [Int]
    package let polygon: [[Double]]
    package let order: Int

    package init(
        index: Int,
        clsID: Int,
        label: String,
        task: PPDocLayoutTask,
        score: Float,
        bbox: [Double],
        bbox2D: [Int],
        polygon: [[Double]],
        order: Int
    ) {
        self.index = index
        self.clsID = clsID
        self.label = label
        self.task = task
        self.score = score
        self.bbox = bbox
        self.bbox2D = bbox2D
        self.polygon = polygon
        self.order = order
    }
}

internal enum PPDocLayoutLayoutPostprocess {
    internal static func postprocess(
        rawDetections: [PPDocLayoutRawDetection],
        targetSize: CGSize,
        layoutNMS: Bool,
        unclipRatio: (Double, Double),
        mergeModes: [Int: LayoutMergeMode],
        labelTaskMapping: [String: Set<String>]
    ) -> [PPDocLayoutLayoutDetection] {
        var filtered = rawDetections

        if layoutNMS {
            filtered = applyNMS(
                filtered,
                sameClassIoUThreshold: 0.6,
                differentClassIoUThreshold: 0.98
            )
        }

        filtered = applyLargeImageFilter(filtered, imageSize: targetSize)
        filtered = applyContainmentFilter(
            filtered,
            threshold: 0.8,
            mergeModes: mergeModes,
            preservedLabels: PPDocLayoutMLXContract.preservedContainmentLabels
        )

        let sorted =
            filtered
            .enumerated()
            .sorted { lhs, rhs in
                if lhs.element.order != rhs.element.order {
                    return lhs.element.order < rhs.element.order
                }
                return lhs.offset < rhs.offset
            }
            .map(\.element)

        let unclippedDetections = sorted.map {
            unclipped(
                detection: $0,
                ratio: unclipRatio,
                canvasSize: targetSize
            )
        }

        var finalized: [PPDocLayoutRawDetection] = []
        finalized.reserveCapacity(unclippedDetections.count)

        let imageWidth = max(1, Double(targetSize.width))
        let imageHeight = max(1, Double(targetSize.height))

        for detection in unclippedDetections {
            let clamped = clampedBBox(
                detection.bbox,
                imageWidth: imageWidth,
                imageHeight: imageHeight
            )

            guard clamped[0] < clamped[2], clamped[1] < clamped[3] else {
                continue
            }

            let intBBox = truncatedBBox(clamped)
            finalized.append(
                PPDocLayoutRawDetection(
                    clsID: detection.clsID,
                    label: detection.label,
                    score: detection.score,
                    bbox: intBBox,
                    order: detection.order,
                    polygon: finalizedPolygon(
                        detection.polygon,
                        fallbackBBox: intBBox,
                        imageWidth: imageWidth,
                        imageHeight: imageHeight
                    )
                )
            )
        }

        return finalized.enumerated().map { index, detection in
            PPDocLayoutLayoutDetection(
                index: index,
                clsID: detection.clsID,
                label: detection.label,
                task: mapTask(label: detection.label, labelTaskMapping: labelTaskMapping),
                score: detection.score,
                bbox: detection.bbox,
                bbox2D: normalizedBBox(
                    detection.bbox,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight
                ),
                polygon: detection.polygon,
                order: detection.order
            )
        }
    }

    internal static func mapTask(
        label: String,
        labelTaskMapping: [String: Set<String>]
    ) -> PPDocLayoutTask {
        for (taskName, labels) in labelTaskMapping where labels.contains(label) {
            switch taskName {
            case "text":
                return .text
            case "table":
                return .table
            case "formula":
                return .formula
            case "skip":
                return .skip
            case "abandon":
                return .abandon
            default:
                break
            }
        }

        return .text
    }

    internal static func applyNMS(
        _ detections: [PPDocLayoutRawDetection],
        sameClassIoUThreshold: Double,
        differentClassIoUThreshold: Double
    ) -> [PPDocLayoutRawDetection] {
        guard !detections.isEmpty else {
            return []
        }

        let sorted = detections.sorted { lhs, rhs in
            lhs.score > rhs.score
        }

        var kept: [PPDocLayoutRawDetection] = []
        kept.reserveCapacity(sorted.count)

        for candidate in sorted {
            var suppressed = false

            for existing in kept {
                let threshold =
                    candidate.clsID == existing.clsID
                    ? sameClassIoUThreshold
                    : differentClassIoUThreshold

                if iou(candidate.bbox, existing.bbox) >= threshold {
                    suppressed = true
                    break
                }
            }

            if !suppressed {
                kept.append(candidate)
            }
        }

        return kept
    }

    internal static func applyLargeImageFilter(
        _ detections: [PPDocLayoutRawDetection],
        imageSize: CGSize
    ) -> [PPDocLayoutRawDetection] {
        guard detections.count > 1 else {
            return detections
        }

        let imageWidth = max(1, Double(imageSize.width))
        let imageHeight = max(1, Double(imageSize.height))
        let imageArea = imageWidth * imageHeight
        let areaThreshold = imageWidth > imageHeight ? 0.82 : 0.93

        var filtered: [PPDocLayoutRawDetection] = []
        filtered.reserveCapacity(detections.count)

        for detection in detections {
            if detection.label == "image" {
                let clamped = clampedBBox(
                    detection.bbox,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight
                )
                let boxArea = max(0, clamped[2] - clamped[0]) * max(0, clamped[3] - clamped[1])
                if boxArea <= areaThreshold * imageArea {
                    filtered.append(detection)
                }
            } else {
                filtered.append(detection)
            }
        }

        return filtered.isEmpty ? detections : filtered
    }

    internal static func applyContainmentFilter(
        _ detections: [PPDocLayoutRawDetection],
        threshold: Double,
        mergeModes: [Int: LayoutMergeMode],
        preservedLabels: Set<String>
    ) -> [PPDocLayoutRawDetection] {
        guard detections.count > 1 else {
            return detections
        }
        guard !mergeModes.isEmpty else {
            return detections
        }

        var keepMask = [Bool](repeating: true, count: detections.count)

        for classID in mergeModes.keys.sorted() {
            guard let mode = mergeModes[classID] else {
                continue
            }

            switch mode {
            case .union:
                continue
            case .large:
                let containment = checkContainment(
                    detections,
                    threshold: threshold,
                    preserveLabels: preservedLabels,
                    categoryIndex: classID,
                    mode: mode
                )

                for idx in detections.indices where containment.containedByOther[idx] == 1 {
                    keepMask[idx] = false
                }
            case .small:
                let containment = checkContainment(
                    detections,
                    threshold: threshold,
                    preserveLabels: preservedLabels,
                    categoryIndex: classID,
                    mode: mode
                )

                for idx in detections.indices {
                    let keep = containment.containsOther[idx] == 0 || containment.containedByOther[idx] == 1
                    keepMask[idx] = keepMask[idx] && keep
                }
            }
        }

        return detections.enumerated().compactMap { index, detection in
            keepMask[index] ? detection : nil
        }
    }

    internal static func unclipped(
        detection: PPDocLayoutRawDetection,
        ratio: (Double, Double),
        canvasSize: CGSize
    ) -> PPDocLayoutRawDetection {
        let xRatio = max(1, ratio.0)
        let yRatio = max(1, ratio.1)

        let x1 = detection.bbox[0]
        let y1 = detection.bbox[1]
        let x2 = detection.bbox[2]
        let y2 = detection.bbox[3]

        let centerX = (x1 + x2) * 0.5
        let centerY = (y1 + y2) * 0.5

        let width = (x2 - x1) * xRatio
        let height = (y2 - y1) * yRatio

        let imageWidth = max(1, Double(canvasSize.width))
        let imageHeight = max(1, Double(canvasSize.height))

        let expandedX1 = max(0, min(imageWidth, centerX - (0.5 * width)))
        let expandedY1 = max(0, min(imageHeight, centerY - (0.5 * height)))
        let expandedX2 = max(0, min(imageWidth, centerX + (0.5 * width)))
        let expandedY2 = max(0, min(imageHeight, centerY + (0.5 * height)))

        return PPDocLayoutRawDetection(
            clsID: detection.clsID,
            label: detection.label,
            score: detection.score,
            bbox: [expandedX1, expandedY1, expandedX2, expandedY2],
            order: detection.order,
            polygon: detection.polygon
        )
    }

    private static func finalizedPolygon(
        _ polygon: [[Double]],
        fallbackBBox: [Double],
        imageWidth: Double,
        imageHeight: Double
    ) -> [[Double]] {
        if polygon.count >= 3 {
            return polygon.map { point in
                guard point.count >= 2 else {
                    return [0.0, 0.0]
                }
                return [
                    max(0, min(imageWidth, point[0])),
                    max(0, min(imageHeight, point[1])),
                ]
            }
        }

        let x1 = fallbackBBox[0]
        let y1 = fallbackBBox[1]
        let x2 = fallbackBBox[2]
        let y2 = fallbackBBox[3]
        return [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]
    }

    private static func checkContainment(
        _ detections: [PPDocLayoutRawDetection],
        threshold: Double,
        preserveLabels: Set<String>,
        categoryIndex: Int? = nil,
        mode: LayoutMergeMode? = nil
    ) -> (containsOther: [Int], containedByOther: [Int]) {
        let count = detections.count
        var containsOther = [Int](repeating: 0, count: count)
        var containedByOther = [Int](repeating: 0, count: count)

        for left in detections.indices {
            if preserveLabels.contains(detections[left].label) {
                continue
            }

            for right in detections.indices where left != right {
                if let categoryIndex, let mode {
                    switch mode {
                    case .large:
                        guard detections[right].clsID == categoryIndex else {
                            continue
                        }
                    case .small:
                        guard detections[left].clsID == categoryIndex else {
                            continue
                        }
                    case .union:
                        continue
                    }
                }

                if isContained(
                    detections[left].bbox,
                    in: detections[right].bbox,
                    threshold: threshold
                ) {
                    containedByOther[left] = 1
                    containsOther[right] = 1
                }
            }
        }

        return (containsOther: containsOther, containedByOther: containedByOther)
    }

    private static func isContained(
        _ lhs: [Double],
        in rhs: [Double],
        threshold: Double
    ) -> Bool {
        let lhsArea = max(0, lhs[2] - lhs[0]) * max(0, lhs[3] - lhs[1])
        guard lhsArea > 0 else {
            return false
        }

        let x1 = max(lhs[0], rhs[0])
        let y1 = max(lhs[1], rhs[1])
        let x2 = min(lhs[2], rhs[2])
        let y2 = min(lhs[3], rhs[3])

        let interWidth = max(0, x2 - x1)
        let interHeight = max(0, y2 - y1)
        let intersection = interWidth * interHeight

        return (intersection / lhsArea) >= threshold
    }

    private static func clampedBBox(
        _ bbox: [Double],
        imageWidth: Double,
        imageHeight: Double
    ) -> [Double] {
        [
            max(0, min(imageWidth, bbox[0])),
            max(0, min(imageHeight, bbox[1])),
            max(0, min(imageWidth, bbox[2])),
            max(0, min(imageHeight, bbox[3])),
        ]
    }

    private static func truncatedBBox(_ bbox: [Double]) -> [Double] {
        bbox.map { Double(Int($0)) }
    }

    private static func normalizedBBox(
        _ bbox: [Double],
        imageWidth: Double,
        imageHeight: Double
    ) -> [Int] {
        let xScale = 1000.0 / imageWidth
        let yScale = 1000.0 / imageHeight

        let x1 = Int(bbox[0] * xScale)
        let y1 = Int(bbox[1] * yScale)
        let x2 = Int(bbox[2] * xScale)
        let y2 = Int(bbox[3] * yScale)

        return [
            clamp(x1),
            clamp(y1),
            clamp(x2),
            clamp(y2),
        ]
    }

    private static func clamp(_ value: Int) -> Int {
        min(1000, max(0, value))
    }

    private static func iou(_ lhs: [Double], _ rhs: [Double]) -> Double {
        let intersection = intersectionAreaInclusive(lhs, rhs)
        let unionArea = areaInclusive(lhs) + areaInclusive(rhs) - intersection
        guard unionArea > 0 else {
            return 0
        }
        return intersection / unionArea
    }

    private static func intersectionAreaInclusive(_ lhs: [Double], _ rhs: [Double]) -> Double {
        let x1 = max(lhs[0], rhs[0])
        let y1 = max(lhs[1], rhs[1])
        let x2 = min(lhs[2], rhs[2])
        let y2 = min(lhs[3], rhs[3])

        let width = max(0, x2 - x1 + 1)
        let height = max(0, y2 - y1 + 1)
        return width * height
    }

    private static func areaInclusive(_ box: [Double]) -> Double {
        let width = max(0, box[2] - box[0] + 1)
        let height = max(0, box[3] - box[1] + 1)
        return width * height
    }
}
