import Foundation
import GlmOCRSwift

@main
struct GlmOCRCLI {
    static func main() async {
        do {
            let cli = try CLIArguments.parse(CommandLine.arguments)
            try await runExamples(cli)
        } catch let error as CLIError {
            fputs("glm-ocr: \(error.description)\n", stderr)
            exit(2)
        } catch {
            fputs("glm-ocr: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func runExamples(_ cli: CLIArguments) async throws {
        let fileManager = FileManager.default
        let sourceDir = URL(fileURLWithPath: cli.sourceDir)
        let outputDir = URL(fileURLWithPath: cli.outputDir)

        var config = GlmOCRConfig()
        config.enableLayout = !cli.disableLayout
        config.markdownBundle.enabled = true
        config.markdownBundle.figureFormat = cli.figureFormat
        config.markdownBundle.figureNamingScheme = cli.figureNamingScheme
        config.markdownBundle.figuresDirectoryName = cli.figuresDirectoryName

        let pipeline = try await GlmOCRPipeline(config: config)
        let options = ParseOptions(
            includeMarkdown: true,
            includeDiagnostics: cli.includeDiagnostics,
            maxPages: cli.maxPages
        )

        try fileManager.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let urls = try fileManager.contentsOfDirectory(
            at: sourceDir,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )

        let inputs = urls
            .filter { isSupportedInputFile($0) }
            .sorted { lhs, rhs in
                lhs.lastPathComponent.localizedStandardCompare(rhs.lastPathComponent) == .orderedAscending
            }

        if inputs.isEmpty {
            throw CLIError.noInputs(sourceDir: sourceDir.path)
        }

        var failures: [String] = []
        for input in inputs {
            let stem = input.deletingPathExtension().lastPathComponent
            let exampleOutDir = outputDir.appendingPathComponent(stem, isDirectory: true)

            do {
                if fileManager.fileExists(atPath: exampleOutDir.path) {
                    try fileManager.removeItem(at: exampleOutDir)
                }
                try fileManager.createDirectory(at: exampleOutDir, withIntermediateDirectories: true)

                let data = try Data(contentsOf: input)
                let doc: InputDocument
                if input.pathExtension.lowercased() == "pdf" {
                    doc = .pdfData(data)
                } else {
                    doc = .imageData(data)
                }

                let result = try await pipeline.parse(doc, options: options)

                let markdownPath = exampleOutDir.appendingPathComponent("\(stem).md")
                var markdownText = result.markdown
                if !markdownText.hasSuffix("\n") {
                    markdownText += "\n"
                }
                try markdownText.write(to: markdownPath, atomically: true, encoding: .utf8)

                let jsonPath = exampleOutDir.appendingPathComponent("\(stem).json")
                let jsonData = try makeReferenceJSON(pages: result.pages)
                try jsonData.write(to: jsonPath, options: .atomic)

                if let bundle = result.markdownBundle {
                    for figure in bundle.figures {
                        let figurePath = exampleOutDir.appendingPathComponent(figure.relativePath)
                        try fileManager.createDirectory(
                            at: figurePath.deletingLastPathComponent(),
                            withIntermediateDirectories: true
                        )
                        try figure.data.write(to: figurePath, options: .atomic)
                    }
                }

                fputs("[examples] OK: \(stem)\n", stderr)
            } catch {
                failures.append(stem)
                fputs("[examples] FAIL: \(stem) (\(error))\n", stderr)
                continue
            }
        }

        if !failures.isEmpty {
            fputs("[examples] completed with failures: \(failures.joined(separator: ", "))\n", stderr)
            exit(1)
        }
    }

    private static func isSupportedInputFile(_ url: URL) -> Bool {
        let allowedExtensions = Set(["png", "jpg", "jpeg", "heic", "pdf"])
        guard allowedExtensions.contains(url.pathExtension.lowercased()) else {
            return false
        }

        guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey]) else {
            return false
        }
        return values.isRegularFile == true
    }

    private static func makeReferenceJSON(pages: [OCRPageResult]) throws -> Data {
        let payload: [[[String: Any]]] = pages.map { page in
            let sorted = page.regions.sorted { lhs, rhs in
                lhs.index < rhs.index
            }
            return sorted.map { region in
                var entry: [String: Any] = [
                    "index": region.index,
                    "label": region.label,
                    "content": region.content ?? "",
                ]
                if let bbox = region.bbox2D {
                    entry["bbox_2d"] = bbox
                } else {
                    entry["bbox_2d"] = NSNull()
                }
                return entry
            }
        }

        guard JSONSerialization.isValidJSONObject(payload) else {
            throw CLIError.invalidJSONPayload
        }
        return try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    }
}

private struct CLIArguments {
    var sourceDir: String
    var outputDir: String
    var maxPages: Int?
    var disableLayout: Bool
    var includeDiagnostics: Bool
    var figuresDirectoryName: String
    var figureFormat: GlmOCRFigureFormat
    var figureNamingScheme: GlmOCRFigureNamingScheme

    static func parse(_ argv: [String]) throws -> CLIArguments {
        var sourceDir = "examples/source"
        var outputDir = "examples/result"
        var maxPages: Int?
        var disableLayout = false
        var includeDiagnostics = false
        var figuresDirectoryName = "imgs"
        var figureFormat: GlmOCRFigureFormat = .jpeg
        var figureNamingScheme: GlmOCRFigureNamingScheme = .upstreamCropped

        var index = 1
        while index < argv.count {
            let arg = argv[index]
            switch arg {
            case "--source-dir":
                index += 1
                guard index < argv.count else { throw CLIError.missingValue("--source-dir") }
                sourceDir = argv[index]
            case "--output-dir":
                index += 1
                guard index < argv.count else { throw CLIError.missingValue("--output-dir") }
                outputDir = argv[index]
            case "--max-pages":
                index += 1
                guard index < argv.count else { throw CLIError.missingValue("--max-pages") }
                guard let parsed = Int(argv[index]) else {
                    throw CLIError.invalidValue(arg: "--max-pages", value: argv[index])
                }
                maxPages = parsed
            case "--disable-layout":
                disableLayout = true
            case "--include-diagnostics":
                includeDiagnostics = true
            case "--figures-dir-name":
                index += 1
                guard index < argv.count else { throw CLIError.missingValue("--figures-dir-name") }
                figuresDirectoryName = argv[index]
            case "--figure-format":
                index += 1
                guard index < argv.count else { throw CLIError.missingValue("--figure-format") }
                let value = argv[index].lowercased()
                switch value {
                case "heic":
                    figureFormat = .heic
                case "jpeg", "jpg":
                    figureFormat = .jpeg
                default:
                    throw CLIError.invalidValue(arg: "--figure-format", value: argv[index])
                }
            case "--figure-naming":
                index += 1
                guard index < argv.count else { throw CLIError.missingValue("--figure-naming") }
                let value = argv[index].lowercased()
                switch value {
                case "page-region-padded":
                    figureNamingScheme = .pageRegionPadded
                case "upstream-cropped":
                    figureNamingScheme = .upstreamCropped
                default:
                    throw CLIError.invalidValue(arg: "--figure-naming", value: argv[index])
                }
            case "--help", "-h":
                printUsageAndExit()
            default:
                throw CLIError.unknownArgument(arg)
            }
            index += 1
        }

        return CLIArguments(
            sourceDir: sourceDir,
            outputDir: outputDir,
            maxPages: maxPages,
            disableLayout: disableLayout,
            includeDiagnostics: includeDiagnostics,
            figuresDirectoryName: figuresDirectoryName,
            figureFormat: figureFormat,
            figureNamingScheme: figureNamingScheme
        )
    }

    private static func printUsageAndExit() -> Never {
        let usage = """
            Usage: GlmOCRCLI [options]

            Batch OCR over a directory and write results in the `examples/reference_result`-compatible layout.

            Options:
              --source-dir <path>        Input directory (default: examples/source)
              --output-dir <path>        Output directory (default: examples/result)
              --max-pages <n>            Optional PDF page cap
              --disable-layout           Disable layout detection
              --include-diagnostics      Include diagnostics in pipeline (does not affect output files)
              --figures-dir-name <name>  Output subdir for cropped figures (default: imgs)
              --figure-format heic|jpeg  Figure image format (default: jpeg)
              --figure-naming <mode>     page-region-padded | upstream-cropped (default: upstream-cropped)
            """
        print(usage)
        exit(0)
    }
}

private enum CLIError: Error, CustomStringConvertible {
    case unknownArgument(String)
    case missingValue(String)
    case invalidValue(arg: String, value: String)
    case noInputs(sourceDir: String)
    case invalidJSONPayload

    var description: String {
        switch self {
        case .unknownArgument(let arg):
            return "unknown argument: \(arg)"
        case .missingValue(let arg):
            return "missing value for argument: \(arg)"
        case .invalidValue(let arg, let value):
            return "invalid value for \(arg): \(value)"
        case .noInputs(let sourceDir):
            return "no supported input files found in: \(sourceDir)"
        case .invalidJSONPayload:
            return "internal error: invalid JSON payload"
        }
    }
}
