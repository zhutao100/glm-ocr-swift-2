import Foundation
import GlmOCRSwift

private struct BenchmarkOutput: Codable {
    let timestampUTC: String
    let pdfPath: String
    let pageCount: Int
    let markdownLength: Int
    let parseWallMs: Double
    let diagnosticsTimingsMs: [String: Double]
    let diagnosticsMetadata: [String: String]
    let warningsCount: Int
    let ocrInferenceJobsPerSecond: Double?
}

@main
struct GlmOCRBenchmarkCLI {
    static func main() async {
        do {
            let cli = try parseArguments(CommandLine.arguments)
            let pdfData = try Data(contentsOf: URL(fileURLWithPath: cli.pdfPath))

            var config = GlmOCRConfig()
            if let maxConcurrentRecognitions = cli.maxConcurrentRecognitions {
                config.maxConcurrentRecognitions = maxConcurrentRecognitions
            }
            if let inferenceBatchSize = cli.inferenceBatchSize {
                config.performance.inferenceBatchSize = inferenceBatchSize
            }
            if let inferenceMaxInflightJobs = cli.inferenceMaxInflightJobs {
                config.performance.inferenceMaxInflightJobs = inferenceMaxInflightJobs
            }
            if cli.disableLayout {
                config.enableLayout = false
            }

            let pipeline = try await GlmOCRPipeline(config: config)
            let options = ParseOptions(
                includeMarkdown: true,
                includeDiagnostics: true,
                maxPages: cli.maxPages
            )

            let start = Date()
            let result = try await pipeline.parse(.pdfData(pdfData), options: options)
            let wallMs = Date().timeIntervalSince(start) * 1_000.0

            let ocrInferenceMs = result.diagnostics.timingsMs["ocr_inference"] ?? 0
            let recognitionJobs = Int(result.diagnostics.metadata["recognitionJobCount"] ?? "") ?? 0
            let jobsPerSecond: Double?
            if ocrInferenceMs > 0, recognitionJobs > 0 {
                jobsPerSecond = Double(recognitionJobs) / (ocrInferenceMs / 1_000.0)
            } else {
                jobsPerSecond = nil
            }

            let output = BenchmarkOutput(
                timestampUTC: ISO8601DateFormatter().string(from: Date()),
                pdfPath: cli.pdfPath,
                pageCount: result.pages.count,
                markdownLength: result.markdown.count,
                parseWallMs: wallMs,
                diagnosticsTimingsMs: result.diagnostics.timingsMs,
                diagnosticsMetadata: result.diagnostics.metadata,
                warningsCount: result.diagnostics.warnings.count,
                ocrInferenceJobsPerSecond: jobsPerSecond
            )

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(output)
            if let json = String(data: data, encoding: .utf8) {
                print(json)
            } else {
                throw CLIError.invalidUTF8Encoding
            }
        } catch {
            fputs("benchmark failed: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func parseArguments(_ args: [String]) throws -> CLIArguments {
        var pdfPath: String = ""
        var maxPages: Int?
        var disableLayout = false
        var maxConcurrentRecognitions: Int?
        var inferenceBatchSize: Int?
        var inferenceMaxInflightJobs: Int?

        var index = 1
        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--pdf":
                index += 1
                guard index < args.count else { throw CLIError.missingValue("--pdf") }
                pdfPath = args[index]
            case "--max-pages":
                index += 1
                guard index < args.count else { throw CLIError.missingValue("--max-pages") }
                guard let parsed = Int(args[index]) else {
                    throw CLIError.invalidValue(arg: "--max-pages", value: args[index])
                }
                maxPages = parsed
            case "--disable-layout":
                disableLayout = true
            case "--max-concurrent":
                index += 1
                guard index < args.count else { throw CLIError.missingValue("--max-concurrent") }
                guard let parsed = Int(args[index]) else {
                    throw CLIError.invalidValue(arg: "--max-concurrent", value: args[index])
                }
                maxConcurrentRecognitions = parsed
            case "--batch-size":
                index += 1
                guard index < args.count else { throw CLIError.missingValue("--batch-size") }
                guard let parsed = Int(args[index]) else {
                    throw CLIError.invalidValue(arg: "--batch-size", value: args[index])
                }
                inferenceBatchSize = parsed
            case "--inflight":
                index += 1
                guard index < args.count else { throw CLIError.missingValue("--inflight") }
                guard let parsed = Int(args[index]) else {
                    throw CLIError.invalidValue(arg: "--inflight", value: args[index])
                }
                inferenceMaxInflightJobs = parsed
            case "--help", "-h":
                printUsageAndExit()
            default:
                throw CLIError.unknownArgument(arg)
            }
            index += 1
        }

        return CLIArguments(
            pdfPath: pdfPath,
            maxPages: maxPages,
            disableLayout: disableLayout,
            maxConcurrentRecognitions: maxConcurrentRecognitions,
            inferenceBatchSize: inferenceBatchSize,
            inferenceMaxInflightJobs: inferenceMaxInflightJobs
        )
    }

    private static func printUsageAndExit() -> Never {
        let usage = """
            Usage: swift run GlmOCRBenchmark [options]
              --pdf <path>             PDF input path
              --max-pages <n>          Optional page cap
              --disable-layout         Run OCR without layout detection
              --max-concurrent <n>     Set GlmOCRConfig.maxConcurrentRecognitions
              --batch-size <n>         Set performance.inferenceBatchSize
              --inflight <n>           Set performance.inferenceMaxInflightJobs
            """
        print(usage)
        exit(0)
    }
}

private struct CLIArguments {
    let pdfPath: String
    let maxPages: Int?
    let disableLayout: Bool
    let maxConcurrentRecognitions: Int?
    let inferenceBatchSize: Int?
    let inferenceMaxInflightJobs: Int?
}

private enum CLIError: Error, CustomStringConvertible {
    case unknownArgument(String)
    case missingValue(String)
    case invalidValue(arg: String, value: String)
    case invalidUTF8Encoding

    var description: String {
        switch self {
        case .unknownArgument(let arg):
            return "Unknown argument: \(arg)"
        case .missingValue(let arg):
            return "Missing value for argument: \(arg)"
        case .invalidValue(let arg, let value):
            return "Invalid value for \(arg): \(value)"
        case .invalidUTF8Encoding:
            return "Failed to encode benchmark output as UTF-8"
        }
    }
}
