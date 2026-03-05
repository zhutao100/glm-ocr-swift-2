import CoreGraphics
import XCTest

@testable import GlmOCRSwift

final class OCRInferenceSchedulerTests: XCTestCase {
    func testSchedulerMapsOutOfOrderBatchCompletionToCorrectKeys() async throws {
        let image = Self.makeTestImage(width: 16, height: 16)
        let jobs: [OCRInferenceBatchJob] = [
            OCRInferenceBatchJob(
                key: PipelineRecognitionJobKey(pageIndex: 0, regionPosition: 0),
                image: image,
                task: .text,
                prompt: "A"
            ),
            OCRInferenceBatchJob(
                key: PipelineRecognitionJobKey(pageIndex: 0, regionPosition: 1),
                image: image,
                task: .text,
                prompt: "A"
            ),
            OCRInferenceBatchJob(
                key: PipelineRecognitionJobKey(pageIndex: 1, regionPosition: 0),
                image: image,
                task: .table,
                prompt: "B"
            ),
        ]

        let scheduler = OCRInferenceScheduler(
            requestedBatchSize: 2,
            maxInflightJobs: 8,
            batchMaxWaitMs: 8
        )
        let result = try await scheduler.run(jobs: jobs) { batch in
            if batch.first?.prompt == "A" {
                try await Task.sleep(nanoseconds: 40_000_000)
            } else {
                try await Task.sleep(nanoseconds: 5_000_000)
            }
            return batch.map { job in
                .success("p\(job.key.pageIndex)-r\(job.key.regionPosition)-\(job.prompt)")
            }
        }

        XCTAssertEqual(try result.results[jobs[0].key]?.get(), "p0-r0-A")
        XCTAssertEqual(try result.results[jobs[1].key]?.get(), "p0-r1-A")
        XCTAssertEqual(try result.results[jobs[2].key]?.get(), "p1-r0-B")
        XCTAssertEqual(result.stats.bucketCount, 2)
        XCTAssertEqual(result.stats.batchCount, 2)
        XCTAssertEqual(result.stats.maxBatchSize, 2)
    }

    func testSchedulerBucketsByPromptTaskAndImageShape() async throws {
        let imageA = Self.makeTestImage(width: 16, height: 16)
        let imageB = Self.makeTestImage(width: 24, height: 16)
        let jobs: [OCRInferenceBatchJob] = [
            OCRInferenceBatchJob(
                key: PipelineRecognitionJobKey(pageIndex: 0, regionPosition: 0),
                image: imageA,
                task: .text,
                prompt: "same"
            ),
            OCRInferenceBatchJob(
                key: PipelineRecognitionJobKey(pageIndex: 0, regionPosition: 1),
                image: imageA,
                task: .table,
                prompt: "same"
            ),
            OCRInferenceBatchJob(
                key: PipelineRecognitionJobKey(pageIndex: 0, regionPosition: 2),
                image: imageB,
                task: .text,
                prompt: "same"
            ),
        ]

        let scheduler = OCRInferenceScheduler(
            requestedBatchSize: 8,
            maxInflightJobs: 32,
            batchMaxWaitMs: 8
        )
        let result = try await scheduler.run(jobs: jobs) { batch in
            batch.map { _ in .success("ok") }
        }

        XCTAssertEqual(result.stats.bucketCount, 3)
        XCTAssertEqual(result.stats.batchCount, 3)
    }

    private static func makeTestImage(width: Int, height: Int) -> CGImage {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
        for offset in stride(from: 0, to: pixels.count, by: 4) {
            pixels[offset] = 120
            pixels[offset + 1] = 160
            pixels[offset + 2] = 200
            pixels[offset + 3] = 255
        }

        guard
            let context = CGContext(
                data: &pixels,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        else {
            fatalError("Unable to create CGContext")
        }
        guard let image = context.makeImage() else {
            fatalError("Unable to create CGImage")
        }
        return image
    }
}
