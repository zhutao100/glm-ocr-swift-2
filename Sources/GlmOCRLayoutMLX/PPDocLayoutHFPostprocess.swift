import CoreGraphics
import Foundation
import MLX

internal struct PPDocLayoutRawDetection: Sendable, Equatable {
    internal let clsID: Int
    internal let label: String
    internal let score: Float
    internal let bbox: [Double]  // absolute [x1, y1, x2, y2]
    internal let order: Int
    internal let polygon: [[Double]]  // absolute polygon points
}

internal enum PPDocLayoutHFPostprocess {
    private struct CandidateScore {
        let score: Float
        let flatIndex: Int
    }

    private struct GatheredDetection {
        let query: Int
        let classIndex: Int
        let score: Float
        let order: Int
        let bbox: [Double]
        let mask: [UInt8]  // flattened [maskHeight * maskWidth]
    }

    private struct IntPoint: Hashable {
        let x: Int
        let y: Int
    }

    internal static func decode(
        prediction: PPDocLayoutMLXPrediction,
        targetSize: CGSize,
        threshold: Float,
        id2label: [Int: String],
        useFastBoundaryPath: Bool = true
    ) throws -> [PPDocLayoutRawDetection] {
        try validateOutputShapes(prediction)

        let batchSize = prediction.logits.shape[0]
        let queryCount = prediction.logits.shape[1]
        let classCount = prediction.logits.shape[2]

        guard batchSize == 1 else {
            throw PPDocLayoutMLXError.modelInitializationFailed(
                "PPDocLayoutHFPostprocess currently expects batch size 1, got \(batchSize)"
            )
        }

        let logits = prediction.logits.asArray(Float.self)
        let predBoxes = prediction.predBoxes.asArray(Float.self)
        let orderLogits = prediction.orderLogits.asArray(Float.self)
        let outMasks = prediction.outMasks.asArray(Float.self)
        let maskHeight = prediction.outMasks.shape[2]
        let maskWidth = prediction.outMasks.shape[3]

        let orderSeq = computeOrderSequence(orderLogits: orderLogits, queryCount: queryCount)
        let topQueryCount = queryCount

        let minNormWidth = Float(1.0 / Double(max(1, maskWidth)))
        let minNormHeight = Float(1.0 / Double(max(1, maskHeight)))

        var flattenedScores: [CandidateScore] = []
        flattenedScores.reserveCapacity(queryCount * classCount)

        for query in 0..<queryCount {
            let boxStart = query * 4
            let boxWidth = predBoxes[boxStart + 2]
            let boxHeight = predBoxes[boxStart + 3]
            let validQuery = boxWidth > minNormWidth && boxHeight > minNormHeight

            for classIndex in 0..<classCount {
                let flatIndex = (query * classCount) + classIndex
                let score = sigmoid(validQuery ? logits[flatIndex] : -100.0)
                flattenedScores.append(CandidateScore(score: score, flatIndex: flatIndex))
            }
        }

        flattenedScores.sort { lhs, rhs in
            if lhs.score != rhs.score {
                return lhs.score > rhs.score
            }
            return lhs.flatIndex < rhs.flatIndex
        }

        let imageWidth = max(1.0, Double(targetSize.width))
        let imageHeight = max(1.0, Double(targetSize.height))
        let maskElementCount = maskHeight * maskWidth

        var gathered: [GatheredDetection] = []
        gathered.reserveCapacity(topQueryCount)

        for candidate in flattenedScores.prefix(topQueryCount) {
            let query = candidate.flatIndex / classCount
            let classIndex = candidate.flatIndex % classCount
            let boxStart = query * 4

            let centerX = Double(predBoxes[boxStart + 0]) * imageWidth
            let centerY = Double(predBoxes[boxStart + 1]) * imageHeight
            let boxWidth = Double(predBoxes[boxStart + 2]) * imageWidth
            let boxHeight = Double(predBoxes[boxStart + 3]) * imageHeight

            let x1 = centerX - (0.5 * boxWidth)
            let y1 = centerY - (0.5 * boxHeight)
            let x2 = centerX + (0.5 * boxWidth)
            let y2 = centerY + (0.5 * boxHeight)

            let maskStart = query * maskElementCount
            guard maskStart >= 0, maskStart + maskElementCount <= outMasks.count else {
                continue
            }

            var binarizedMask = [UInt8](repeating: 0, count: maskElementCount)
            for idx in 0..<maskElementCount {
                binarizedMask[idx] = sigmoid(outMasks[maskStart + idx]) > threshold ? 1 : 0
            }

            gathered.append(
                GatheredDetection(
                    query: query,
                    classIndex: classIndex,
                    score: candidate.score,
                    order: orderSeq[query],
                    bbox: [x1, y1, x2, y2],
                    mask: binarizedMask
                )
            )
        }

        let scoreFiltered = gathered.filter { $0.score >= threshold }
        let ordered =
            scoreFiltered
            .enumerated()
            .sorted { lhs, rhs in
                if lhs.element.order != rhs.element.order {
                    return lhs.element.order < rhs.element.order
                }
                return lhs.offset < rhs.offset
            }
            .map(\.element)

        var detections: [PPDocLayoutRawDetection] = []
        detections.reserveCapacity(ordered.count)

        for candidate in ordered {
            let label = id2label[candidate.classIndex] ?? "class_\(candidate.classIndex)"
            let polygon =
                extractPolygonPointsByMask(
                    box: candidate.bbox,
                    mask: candidate.mask,
                    maskWidth: maskWidth,
                    maskHeight: maskHeight,
                    targetSize: targetSize,
                    useFastBoundaryPath: useFastBoundaryPath
                ) ?? fallbackRectangle(for: candidate.bbox)

            detections.append(
                PPDocLayoutRawDetection(
                    clsID: candidate.classIndex,
                    label: label,
                    score: candidate.score,
                    bbox: candidate.bbox,
                    order: candidate.order,
                    polygon: polygon
                )
            )
        }

        return detections
    }

    internal static func computeOrderSequence(
        orderLogits: [Float],
        queryCount: Int
    ) -> [Int] {
        guard queryCount > 0 else {
            return []
        }

        var votes = [Float](repeating: 0, count: queryCount)
        for pointer in 0..<queryCount {
            var vote: Float = 0
            for idx in 0..<queryCount {
                let upperIndex = (idx * queryCount) + pointer
                let lowerIndex = (pointer * queryCount) + idx

                if idx < pointer {
                    vote += sigmoid(orderLogits[upperIndex])
                } else if idx > pointer {
                    vote += 1 - sigmoid(orderLogits[lowerIndex])
                }
            }
            votes[pointer] = vote
        }

        let sortedPointers = votes.enumerated().sorted { lhs, rhs in
            if lhs.element != rhs.element {
                return lhs.element < rhs.element
            }
            return lhs.offset < rhs.offset
        }.map(\.offset)

        var orderSeq = [Int](repeating: 0, count: queryCount)
        for (rank, pointer) in sortedPointers.enumerated() {
            orderSeq[pointer] = rank
        }
        return orderSeq
    }

    internal static func extractPolygonPointsByMask(
        box: [Double],
        mask: [UInt8],
        maskWidth: Int,
        maskHeight: Int,
        targetSize: CGSize,
        useFastBoundaryPath: Bool = true
    ) -> [[Double]]? {
        let xMin = Int(box[0])
        let yMin = Int(box[1])
        let xMax = Int(box[2])
        let yMax = Int(box[3])
        let boxW = xMax - xMin
        let boxH = yMax - yMin

        let rect = fallbackRectangle(for: box)
        guard boxW > 0, boxH > 0 else {
            return rect
        }
        guard mask.count == maskWidth * maskHeight else {
            return rect
        }

        let processorWidth = Double(PPDocLayoutMLXContract.inputShape[3])
        let processorHeight = Double(PPDocLayoutMLXContract.inputShape[2])
        let targetWidth = max(1.0, Double(targetSize.width))
        let targetHeight = max(1.0, Double(targetSize.height))

        let scaleWidth = (processorWidth / targetWidth) / 4.0
        let scaleHeight = (processorHeight / targetHeight) / 4.0

        let xCoordinates = [
            Int((Double(xMin) * scaleWidth).rounded()),
            Int((Double(xMax) * scaleWidth).rounded()),
        ]
        let yCoordinates = [
            Int((Double(yMin) * scaleHeight).rounded()),
            Int((Double(yMax) * scaleHeight).rounded()),
        ]

        let xStart = clip(xCoordinates[0], minValue: 0, maxValue: maskWidth)
        let xEnd = clip(xCoordinates[1], minValue: 0, maxValue: maskWidth)
        let yStart = clip(yCoordinates[0], minValue: 0, maxValue: maskHeight)
        let yEnd = clip(yCoordinates[1], minValue: 0, maxValue: maskHeight)

        let croppedWidth = max(0, xEnd - xStart)
        let croppedHeight = max(0, yEnd - yStart)
        guard croppedWidth > 0, croppedHeight > 0 else {
            return rect
        }

        let croppedMask = cropMask(
            mask: mask,
            maskWidth: maskWidth,
            xStart: xStart,
            xEnd: xEnd,
            yStart: yStart,
            yEnd: yEnd
        )
        guard !croppedMask.isEmpty else {
            return rect
        }

        let resizedMask = resizeMaskNearest(
            source: croppedMask,
            sourceWidth: croppedWidth,
            sourceHeight: croppedHeight,
            targetWidth: boxW,
            targetHeight: boxH
        )
        guard !resizedMask.isEmpty else {
            return rect
        }

        guard
            var polygon = maskToPolygon(
                mask: resizedMask,
                width: boxW,
                height: boxH,
                useFastBoundaryPath: useFastBoundaryPath
            )
        else {
            return rect
        }

        if polygon.count < 4 {
            return rect
        }

        for idx in polygon.indices {
            polygon[idx][0] += Double(xMin)
            polygon[idx][1] += Double(yMin)
        }

        return polygon
    }

    private static func maskToPolygon(
        mask: [UInt8],
        width: Int,
        height: Int,
        useFastBoundaryPath: Bool
    ) -> [[Double]]? {
        guard
            let contour = largestExternalContour(
                mask: mask,
                width: width,
                height: height,
                useFastBoundaryPath: useFastBoundaryPath
            ), contour.count >= 3
        else {
            return nil
        }

        let epsilon = maskPolygonEpsilonRatio() * arcLength(contour)
        let approximated = approxPolyDP(contour: contour, epsilon: epsilon)
        guard !approximated.isEmpty else {
            return nil
        }

        let polygon = approximated.map { [Double($0.x), Double($0.y)] }
        return extractCustomVertices(polygon)
    }

    private static func largestExternalContour(
        mask: [UInt8],
        width: Int,
        height: Int,
        useFastBoundaryPath: Bool
    ) -> [IntPoint]? {
        guard width > 0, height > 0 else {
            return nil
        }

        var visited = [Bool](repeating: false, count: width * height)
        let neighbors = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1),
        ]

        var bestComponentIndices: [Int] = []
        var bestComponentMembership = [Bool](repeating: false, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                guard !visited[index], mask[index] != 0 else {
                    continue
                }

                var queue: [Int] = [index]
                visited[index] = true
                var componentIndices: [Int] = []
                componentIndices.reserveCapacity(128)

                var queueIndex = 0
                while queueIndex < queue.count {
                    let pointIndex = queue[queueIndex]
                    queueIndex += 1
                    componentIndices.append(pointIndex)
                    let pointX = pointIndex % width
                    let pointY = pointIndex / width

                    for (dx, dy) in neighbors {
                        let nx = pointX + dx
                        let ny = pointY + dy
                        guard nx >= 0, nx < width, ny >= 0, ny < height else {
                            continue
                        }

                        let nIndex = ny * width + nx
                        if !visited[nIndex], mask[nIndex] != 0 {
                            visited[nIndex] = true
                            queue.append(nIndex)
                        }
                    }
                }

                if componentIndices.count > bestComponentIndices.count {
                    for existingIndex in bestComponentIndices {
                        bestComponentMembership[existingIndex] = false
                    }
                    for newIndex in componentIndices {
                        bestComponentMembership[newIndex] = true
                    }
                    bestComponentIndices = componentIndices
                }
            }
        }

        guard !bestComponentIndices.isEmpty else {
            return nil
        }

        let boundary: [IntPoint]
        if useFastBoundaryPath {
            boundary = boundaryPointsFast(
                componentIndices: bestComponentIndices,
                membership: bestComponentMembership,
                width: width,
                height: height
            )
        } else {
            let componentSet = Set(bestComponentIndices.map { IntPoint(x: $0 % width, y: $0 / width) })
            boundary = boundaryPoints(component: componentSet, width: width, height: height)
        }
        guard boundary.count >= 3 else {
            return nil
        }

        let hull = convexHull(points: boundary)
        guard hull.count >= 3 else {
            return nil
        }
        return ensureNegativeSignedArea(hull)
    }

    private static func boundaryPointsFast(
        componentIndices: [Int],
        membership: [Bool],
        width: Int,
        height: Int
    ) -> [IntPoint] {
        var boundary: [IntPoint] = []
        boundary.reserveCapacity(componentIndices.count)

        for index in componentIndices {
            let x = index % width
            let y = index / width

            if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                boundary.append(IntPoint(x: x, y: y))
                continue
            }

            let neighbors = [
                index - width,  // up
                index - 1,  // left
                index + 1,  // right
                index + width,  // down
            ]
            if neighbors.contains(where: { !membership[$0] }) {
                boundary.append(IntPoint(x: x, y: y))
            }
        }

        return boundary
    }

    private static func boundaryPoints(
        component: Set<IntPoint>,
        width: Int,
        height: Int
    ) -> [IntPoint] {
        let neighbors = [
            (0, -1),
            (-1, 0), (1, 0),
            (0, 1),
        ]

        return component.filter { point in
            for (dx, dy) in neighbors {
                let nx = point.x + dx
                let ny = point.y + dy
                if nx < 0 || nx >= width || ny < 0 || ny >= height {
                    return true
                }
                if !component.contains(IntPoint(x: nx, y: ny)) {
                    return true
                }
            }
            return false
        }
    }

    private static func convexHull(points: [IntPoint]) -> [IntPoint] {
        guard !points.isEmpty else {
            return []
        }

        let sorted = points.sorted { lhs, rhs in
            if lhs.x != rhs.x {
                return lhs.x < rhs.x
            }
            return lhs.y < rhs.y
        }

        var uniqueSorted: [IntPoint] = []
        uniqueSorted.reserveCapacity(sorted.count)
        var previous: IntPoint?
        for point in sorted {
            if point != previous {
                uniqueSorted.append(point)
                previous = point
            }
        }

        guard uniqueSorted.count >= 3 else {
            return uniqueSorted
        }

        var lower: [IntPoint] = []
        lower.reserveCapacity(uniqueSorted.count)
        for point in uniqueSorted {
            while lower.count >= 2,
                cross(lower[lower.count - 2], lower[lower.count - 1], point) <= 0
            {
                lower.removeLast()
            }
            lower.append(point)
        }

        var upper: [IntPoint] = []
        upper.reserveCapacity(uniqueSorted.count)
        for point in uniqueSorted.reversed() {
            while upper.count >= 2,
                cross(upper[upper.count - 2], upper[upper.count - 1], point) <= 0
            {
                upper.removeLast()
            }
            upper.append(point)
        }

        lower.removeLast()
        upper.removeLast()
        return lower + upper
    }

    private static func cross(_ origin: IntPoint, _ left: IntPoint, _ right: IntPoint) -> Int64 {
        let leftX = Int64(left.x - origin.x)
        let leftY = Int64(left.y - origin.y)
        let rightX = Int64(right.x - origin.x)
        let rightY = Int64(right.y - origin.y)
        return (leftX * rightY) - (leftY * rightX)
    }

    private static func ensureNegativeSignedArea(_ contour: [IntPoint]) -> [IntPoint] {
        guard contour.count >= 3 else {
            return contour
        }
        return signedArea(contour) > 0 ? Array(contour.reversed()) : contour
    }

    private static func signedArea(_ contour: [IntPoint]) -> Double {
        guard contour.count >= 3 else {
            return 0
        }

        var area = 0.0
        for idx in contour.indices {
            let current = contour[idx]
            let next = contour[(idx + 1) % contour.count]
            area += Double(current.x * next.y - next.x * current.y)
        }
        return area / 2.0
    }

    private static func arcLength(_ contour: [IntPoint]) -> Double {
        guard contour.count >= 2 else {
            return 0
        }

        var length = 0.0
        for idx in contour.indices {
            let current = contour[idx]
            let next = contour[(idx + 1) % contour.count]
            let dx = Double(next.x - current.x)
            let dy = Double(next.y - current.y)
            length += sqrt((dx * dx) + (dy * dy))
        }
        return length
    }

    private static func approxPolyDP(contour: [IntPoint], epsilon: Double) -> [IntPoint] {
        guard contour.count > 3 else {
            return contour
        }

        var closed = contour
        if closed.first != closed.last, let first = closed.first {
            closed.append(first)
        }

        var approximated = ramerDouglasPeucker(points: closed, epsilon: epsilon)
        if approximated.first == approximated.last {
            approximated.removeLast()
        }

        return approximated.count >= 3 ? approximated : contour
    }

    private static func ramerDouglasPeucker(points: [IntPoint], epsilon: Double) -> [IntPoint] {
        guard points.count >= 3 else {
            return points
        }

        var maxDistance = 0.0
        var index = 0

        let first = points[0]
        let last = points[points.count - 1]

        for idx in 1..<(points.count - 1) {
            let distance = perpendicularDistance(point: points[idx], lineStart: first, lineEnd: last)
            if distance > maxDistance {
                maxDistance = distance
                index = idx
            }
        }

        if maxDistance > epsilon {
            let left = ramerDouglasPeucker(points: Array(points[0...index]), epsilon: epsilon)
            let right = ramerDouglasPeucker(points: Array(points[index...(points.count - 1)]), epsilon: epsilon)
            return Array(left.dropLast()) + right
        }

        return [first, last]
    }

    private static func perpendicularDistance(point: IntPoint, lineStart: IntPoint, lineEnd: IntPoint) -> Double {
        let x = Double(point.x)
        let y = Double(point.y)
        let x1 = Double(lineStart.x)
        let y1 = Double(lineStart.y)
        let x2 = Double(lineEnd.x)
        let y2 = Double(lineEnd.y)

        let dx = x2 - x1
        let dy = y2 - y1

        if dx == 0, dy == 0 {
            let px = x - x1
            let py = y - y1
            return sqrt((px * px) + (py * py))
        }

        let numerator = abs((dy * x) - (dx * y) + (x2 * y1) - (y2 * x1))
        let denominator = sqrt((dx * dx) + (dy * dy))
        return numerator / denominator
    }

    private static func extractCustomVertices(
        _ polygon: [[Double]],
        sharpAngleThreshold: Double = 45.0
    ) -> [[Double]] {
        let n = polygon.count
        guard n > 0 else {
            return []
        }

        var result: [[Double]] = []
        result.reserveCapacity(n)

        for i in 0..<n {
            let previous = polygon[(i - 1 + n) % n]
            let current = polygon[i]
            let next = polygon[(i + 1) % n]

            let vector1 = [previous[0] - current[0], previous[1] - current[1]]
            let vector2 = [next[0] - current[0], next[1] - current[1]]

            let crossValue = (vector1[1] * vector2[0]) - (vector1[0] * vector2[1])
            if crossValue < 0 {
                let norm1 = hypot(vector1[0], vector1[1])
                let norm2 = hypot(vector2[0], vector2[1])
                guard norm1 > 0, norm2 > 0 else {
                    continue
                }

                let cosine = clip(
                    (vector1[0] * vector2[0] + vector1[1] * vector2[1]) / (norm1 * norm2),
                    minValue: -1.0,
                    maxValue: 1.0
                )
                let angle = acos(cosine) * (180.0 / Double.pi)

                if abs(angle - sharpAngleThreshold) < 1.0 {
                    var direction = [
                        (vector1[0] / norm1) + (vector2[0] / norm2),
                        (vector1[1] / norm1) + (vector2[1] / norm2),
                    ]
                    let directionNorm = hypot(direction[0], direction[1])
                    guard directionNorm > 0 else {
                        result.append(current)
                        continue
                    }
                    direction[0] /= directionNorm
                    direction[1] /= directionNorm

                    let step = (norm1 + norm2) * 0.5
                    result.append([
                        current[0] + (direction[0] * step),
                        current[1] + (direction[1] * step),
                    ])
                } else {
                    result.append(current)
                }
            }
        }

        return result
    }

    private static func cropMask(
        mask: [UInt8],
        maskWidth: Int,
        xStart: Int,
        xEnd: Int,
        yStart: Int,
        yEnd: Int
    ) -> [UInt8] {
        let croppedWidth = xEnd - xStart
        let croppedHeight = yEnd - yStart
        guard croppedWidth > 0, croppedHeight > 0 else {
            return []
        }

        var cropped = [UInt8](repeating: 0, count: croppedWidth * croppedHeight)
        for y in 0..<croppedHeight {
            let sourceOffset = (yStart + y) * maskWidth + xStart
            let targetOffset = y * croppedWidth
            cropped[targetOffset..<(targetOffset + croppedWidth)] = mask[sourceOffset..<(sourceOffset + croppedWidth)]
        }
        return cropped
    }

    private static func resizeMaskNearest(
        source: [UInt8],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [UInt8] {
        guard sourceWidth > 0, sourceHeight > 0, targetWidth > 0, targetHeight > 0 else {
            return []
        }

        var resized = [UInt8](repeating: 0, count: targetWidth * targetHeight)
        for y in 0..<targetHeight {
            let mappedY = min(sourceHeight - 1, Int((Double(y) * Double(sourceHeight)) / Double(targetHeight)))
            for x in 0..<targetWidth {
                let mappedX = min(sourceWidth - 1, Int((Double(x) * Double(sourceWidth)) / Double(targetWidth)))
                resized[y * targetWidth + x] = source[mappedY * sourceWidth + mappedX]
            }
        }
        return resized
    }

    private static func fallbackRectangle(for box: [Double]) -> [[Double]] {
        guard box.count == 4 else {
            return []
        }

        let x1 = Double(Int(box[0]))
        let y1 = Double(Int(box[1]))
        let x2 = Double(Int(box[2]))
        let y2 = Double(Int(box[3]))
        return [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]
    }

    private static func sigmoid(_ value: Float) -> Float {
        1.0 / (1.0 + exp(-value))
    }

    private static func clip<T: Comparable>(_ value: T, minValue: T, maxValue: T) -> T {
        min(max(value, minValue), maxValue)
    }

    private static func maskPolygonEpsilonRatio() -> Double { 0.004 }

    private static func validateOutputShapes(_ prediction: PPDocLayoutMLXPrediction) throws {
        try PPDocLayoutMLXContractValidator.validatePrediction(prediction)
    }
}
