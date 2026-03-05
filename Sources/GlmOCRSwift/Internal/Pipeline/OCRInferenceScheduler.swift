import CoreGraphics
import Foundation

internal struct OCRInferenceBatchJob: @unchecked Sendable {
    internal let key: PipelineRecognitionJobKey
    internal let image: CGImage
    internal let task: OCRTask
    internal let prompt: String

    internal init(
        key: PipelineRecognitionJobKey,
        image: CGImage,
        task: OCRTask,
        prompt: String
    ) {
        self.key = key
        self.image = image
        self.task = task
        self.prompt = prompt
    }
}

internal struct OCRInferenceSchedulerStats: Sendable, Equatable {
    internal let bucketCount: Int
    internal let batchCount: Int
    internal let maxBatchSize: Int
    internal let averageBatchSize: Double
    internal let queuedJobCount: Int
    internal let requestedBatchSize: Int
    internal let maxInflightJobs: Int
    internal let batchMaxWaitMs: Int

    internal init(
        bucketCount: Int = 0,
        batchCount: Int = 0,
        maxBatchSize: Int = 0,
        averageBatchSize: Double = 0,
        queuedJobCount: Int = 0,
        requestedBatchSize: Int = 0,
        maxInflightJobs: Int = 0,
        batchMaxWaitMs: Int = 0
    ) {
        self.bucketCount = bucketCount
        self.batchCount = batchCount
        self.maxBatchSize = maxBatchSize
        self.averageBatchSize = averageBatchSize
        self.queuedJobCount = queuedJobCount
        self.requestedBatchSize = requestedBatchSize
        self.maxInflightJobs = maxInflightJobs
        self.batchMaxWaitMs = batchMaxWaitMs
    }
}

internal struct OCRInferenceSchedulerResult: @unchecked Sendable {
    internal let results: [PipelineRecognitionJobKey: Result<String, Error>]
    internal let stats: OCRInferenceSchedulerStats

    internal init(
        results: [PipelineRecognitionJobKey: Result<String, Error>],
        stats: OCRInferenceSchedulerStats
    ) {
        self.results = results
        self.stats = stats
    }
}

internal struct OCRInferenceScheduler {
    internal typealias BatchExecutor = @Sendable ([OCRInferenceBatchJob]) async throws -> [Result<String, Error>]

    private struct BucketKey: Hashable {
        let prompt: String
        let task: OCRTask
        let width: Int
        let height: Int
    }

    private struct BatchCompletion: @unchecked Sendable {
        let pairs: [(PipelineRecognitionJobKey, Result<String, Error>)]
    }

    private enum SchedulerError: Error {
        case batchResultCountMismatch(expected: Int, actual: Int)
    }

    private let requestedBatchSize: Int
    private let maxInflightJobs: Int
    private let batchMaxWaitMs: Int

    internal init(
        requestedBatchSize: Int,
        maxInflightJobs: Int,
        batchMaxWaitMs: Int
    ) {
        self.requestedBatchSize = max(1, requestedBatchSize)
        self.maxInflightJobs = max(1, maxInflightJobs)
        self.batchMaxWaitMs = max(0, batchMaxWaitMs)
    }

    internal func run(
        jobs: [OCRInferenceBatchJob],
        executor: @escaping BatchExecutor
    ) async throws -> OCRInferenceSchedulerResult {
        guard !jobs.isEmpty else {
            return OCRInferenceSchedulerResult(
                results: [:],
                stats: OCRInferenceSchedulerStats(
                    requestedBatchSize: requestedBatchSize,
                    maxInflightJobs: maxInflightJobs,
                    batchMaxWaitMs: batchMaxWaitMs
                )
            )
        }

        var buckets: [BucketKey: [OCRInferenceBatchJob]] = [:]
        var bucketOrder: [BucketKey] = []
        bucketOrder.reserveCapacity(jobs.count)

        for job in jobs {
            let key = BucketKey(
                prompt: job.prompt,
                task: job.task,
                width: job.image.width,
                height: job.image.height
            )
            if buckets[key] == nil {
                bucketOrder.append(key)
            }
            buckets[key, default: []].append(job)
        }

        var queuedBatches: [[OCRInferenceBatchJob]] = []
        queuedBatches.reserveCapacity(max(1, jobs.count / requestedBatchSize))
        for key in bucketOrder {
            guard let bucketJobs = buckets[key] else {
                continue
            }

            var start = 0
            while start < bucketJobs.count {
                let end = min(start + requestedBatchSize, bucketJobs.count)
                queuedBatches.append(Array(bucketJobs[start..<end]))
                start = end
            }
        }

        let maxConcurrentBatches = max(1, maxInflightJobs / requestedBatchSize)
        var results: [PipelineRecognitionJobKey: Result<String, Error>] = [:]
        results.reserveCapacity(jobs.count)

        var nextBatchIndex = 0
        var activeBatchCount = 0

        try await withThrowingTaskGroup(of: BatchCompletion.self) { group in
            while nextBatchIndex < queuedBatches.count || activeBatchCount > 0 {
                while nextBatchIndex < queuedBatches.count, activeBatchCount < maxConcurrentBatches {
                    let batch = queuedBatches[nextBatchIndex]
                    nextBatchIndex += 1
                    activeBatchCount += 1

                    group.addTask {
                        do {
                            let batchResults = try await executor(batch)
                            guard batchResults.count == batch.count else {
                                throw SchedulerError.batchResultCountMismatch(
                                    expected: batch.count,
                                    actual: batchResults.count
                                )
                            }

                            let pairs = zip(batch, batchResults).map { job, result in
                                (job.key, result)
                            }
                            return BatchCompletion(pairs: pairs)
                        } catch {
                            if error is CancellationError {
                                throw error
                            }

                            let failurePairs = batch.map { job in
                                (job.key, Result<String, Error>.failure(error))
                            }
                            return BatchCompletion(pairs: failurePairs)
                        }
                    }
                }

                guard let completion = try await group.next() else {
                    continue
                }
                activeBatchCount -= 1
                for (key, result) in completion.pairs {
                    results[key] = result
                }
            }
        }

        let batchSizes = queuedBatches.map(\.count)
        let maxBatchSize = batchSizes.max() ?? 0
        let totalBatchJobs = batchSizes.reduce(0, +)
        let averageBatchSize = queuedBatches.isEmpty ? 0 : Double(totalBatchJobs) / Double(queuedBatches.count)

        let stats = OCRInferenceSchedulerStats(
            bucketCount: bucketOrder.count,
            batchCount: queuedBatches.count,
            maxBatchSize: maxBatchSize,
            averageBatchSize: averageBatchSize,
            queuedJobCount: jobs.count,
            requestedBatchSize: requestedBatchSize,
            maxInflightJobs: maxInflightJobs,
            batchMaxWaitMs: batchMaxWaitMs
        )
        return OCRInferenceSchedulerResult(results: results, stats: stats)
    }
}
