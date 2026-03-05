import CoreGraphics
import Foundation

public struct GlmOCRRecognitionOptions: Sendable, Codable, Equatable {
    public var maxTokens: Int
    public var temperature: Float
    public var prefillStepSize: Int
    public var topP: Float
    public var topK: Int
    public var repetitionPenalty: Float

    public init(
        maxTokens: Int = 4_096,
        temperature: Float = 0.8,
        prefillStepSize: Int = 2_048,
        topP: Float = 0.9,
        topK: Int = 50,
        repetitionPenalty: Float = 1.1
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.prefillStepSize = prefillStepSize
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
    }

    func validate() throws {
        guard maxTokens > 0 else {
            throw GlmOCRError.invalidConfiguration("recognitionOptions.maxTokens must be greater than zero")
        }
        guard prefillStepSize > 0 else {
            throw GlmOCRError.invalidConfiguration("recognitionOptions.prefillStepSize must be greater than zero")
        }
        guard temperature >= 0 else {
            throw GlmOCRError.invalidConfiguration("recognitionOptions.temperature must be >= 0")
        }
        guard topP >= 0 else {
            throw GlmOCRError.invalidConfiguration("recognitionOptions.topP must be >= 0")
        }
        guard topK > 0 else {
            throw GlmOCRError.invalidConfiguration("recognitionOptions.topK must be greater than zero")
        }
        guard repetitionPenalty > 0 else {
            throw GlmOCRError.invalidConfiguration("recognitionOptions.repetitionPenalty must be greater than zero")
        }
    }

    private enum CodingKeys: String, CodingKey {
        case maxTokens
        case temperature
        case prefillStepSize
        case topP
        case topK
        case repetitionPenalty
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self()
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? defaults.maxTokens
        self.temperature = try container.decodeIfPresent(Float.self, forKey: .temperature) ?? defaults.temperature
        self.prefillStepSize =
            try container.decodeIfPresent(Int.self, forKey: .prefillStepSize) ?? defaults.prefillStepSize
        self.topP = try container.decodeIfPresent(Float.self, forKey: .topP) ?? defaults.topP
        self.topK = try container.decodeIfPresent(Int.self, forKey: .topK) ?? defaults.topK
        self.repetitionPenalty =
            try container.decodeIfPresent(Float.self, forKey: .repetitionPenalty)
            ?? defaults.repetitionPenalty
    }
}

public struct GlmOCRPromptConfig: Sendable, Codable, Equatable {
    public var noLayoutPrompt: String
    public var textPrompt: String
    public var tablePrompt: String
    public var formulaPrompt: String

    public init(
        noLayoutPrompt: String =
            "Recognize the text in the image and output in Markdown format. Preserve the original layout (headings/paragraphs/tables/formulas). Do not fabricate content that does not exist in the image.",
        textPrompt: String = "Text Recognition:",
        tablePrompt: String = "Table Recognition:",
        formulaPrompt: String = "Formula Recognition:"
    ) {
        self.noLayoutPrompt = noLayoutPrompt
        self.textPrompt = textPrompt
        self.tablePrompt = tablePrompt
        self.formulaPrompt = formulaPrompt
    }

    func validate() throws {
        if noLayoutPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw GlmOCRError.invalidConfiguration("prompts.noLayoutPrompt must not be empty")
        }
        if textPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw GlmOCRError.invalidConfiguration("prompts.textPrompt must not be empty")
        }
        if tablePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw GlmOCRError.invalidConfiguration("prompts.tablePrompt must not be empty")
        }
        if formulaPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw GlmOCRError.invalidConfiguration("prompts.formulaPrompt must not be empty")
        }
    }

    private enum CodingKeys: String, CodingKey {
        case noLayoutPrompt
        case textPrompt
        case tablePrompt
        case formulaPrompt
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self()
        self.noLayoutPrompt =
            try container.decodeIfPresent(String.self, forKey: .noLayoutPrompt) ?? defaults.noLayoutPrompt
        self.textPrompt = try container.decodeIfPresent(String.self, forKey: .textPrompt) ?? defaults.textPrompt
        self.tablePrompt = try container.decodeIfPresent(String.self, forKey: .tablePrompt) ?? defaults.tablePrompt
        self.formulaPrompt =
            try container.decodeIfPresent(String.self, forKey: .formulaPrompt) ?? defaults.formulaPrompt
    }
}

public struct GlmOCRLayoutConfig: Sendable, Codable, Equatable {
    public var threshold: Float
    public var thresholdByClass: [String: Float]?
    public var layoutNMS: Bool
    public var layoutUnclipRatio: (Double, Double)
    public var layoutMergeBBoxesMode: [Int: String]
    public var labelTaskMapping: [String: [String]]
    public var id2label: [Int: String]?

    public init(
        threshold: Float = 0.3,
        thresholdByClass: [String: Float]? = nil,
        layoutNMS: Bool = true,
        layoutUnclipRatio: (Double, Double) = (1.0, 1.0),
        layoutMergeBBoxesMode: [Int: String] = Self.defaultLayoutMergeBBoxesMode,
        labelTaskMapping: [String: [String]] = Self.defaultLabelTaskMapping,
        id2label: [Int: String]? = nil
    ) {
        self.threshold = threshold
        self.thresholdByClass = thresholdByClass
        self.layoutNMS = layoutNMS
        self.layoutUnclipRatio = layoutUnclipRatio
        self.layoutMergeBBoxesMode = layoutMergeBBoxesMode
        self.labelTaskMapping = labelTaskMapping
        self.id2label = id2label
    }

    func validate() throws {
        guard threshold >= 0 else {
            throw GlmOCRError.invalidConfiguration("layout.threshold must be >= 0")
        }
        if let thresholdByClass {
            for (key, value) in thresholdByClass {
                if key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    throw GlmOCRError.invalidConfiguration("layout.thresholdByClass must not contain empty keys")
                }
                if value < 0 {
                    throw GlmOCRError.invalidConfiguration("layout.thresholdByClass[\(key)] must be >= 0")
                }
            }
        }
        if layoutUnclipRatio.0 <= 0 || layoutUnclipRatio.1 <= 0 {
            throw GlmOCRError.invalidConfiguration("layout.layoutUnclipRatio values must be > 0")
        }
        for (classID, mode) in layoutMergeBBoxesMode {
            if !Self.validMergeModes.contains(mode) {
                throw GlmOCRError.invalidConfiguration(
                    "layout.layoutMergeBBoxesMode[\(classID)] must be one of union|large|small"
                )
            }
        }
    }

    private enum CodingKeys: String, CodingKey {
        case threshold
        case thresholdByClass
        case layoutNMS
        case layoutUnclipRatio
        case layoutMergeBBoxesMode
        case labelTaskMapping
        case id2label
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self()

        self.threshold = try container.decodeIfPresent(Float.self, forKey: .threshold) ?? defaults.threshold
        self.thresholdByClass = try container.decodeIfPresent([String: Float].self, forKey: .thresholdByClass)
        self.layoutNMS = try container.decodeIfPresent(Bool.self, forKey: .layoutNMS) ?? defaults.layoutNMS
        self.layoutUnclipRatio = try Self.decodeUnclipRatio(
            from: container,
            key: .layoutUnclipRatio,
            fallback: defaults.layoutUnclipRatio
        )
        self.layoutMergeBBoxesMode =
            try container.decodeIfPresent([Int: String].self, forKey: .layoutMergeBBoxesMode)
            ?? defaults.layoutMergeBBoxesMode
        self.labelTaskMapping =
            try container.decodeIfPresent([String: [String]].self, forKey: .labelTaskMapping)
            ?? defaults.labelTaskMapping
        self.id2label = try container.decodeIfPresent([Int: String].self, forKey: .id2label)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(threshold, forKey: .threshold)
        try container.encodeIfPresent(thresholdByClass, forKey: .thresholdByClass)
        try container.encode(layoutNMS, forKey: .layoutNMS)
        try container.encode([layoutUnclipRatio.0, layoutUnclipRatio.1], forKey: .layoutUnclipRatio)
        try container.encode(layoutMergeBBoxesMode, forKey: .layoutMergeBBoxesMode)
        try container.encode(labelTaskMapping, forKey: .labelTaskMapping)
        try container.encodeIfPresent(id2label, forKey: .id2label)
    }

    private static func decodeUnclipRatio(
        from container: KeyedDecodingContainer<CodingKeys>,
        key: CodingKeys,
        fallback: (Double, Double)
    ) throws -> (Double, Double) {
        guard container.contains(key) else {
            return fallback
        }

        if let values = try container.decodeIfPresent([Double].self, forKey: key), values.count == 2 {
            return (values[0], values[1])
        }

        if let value = try container.decodeIfPresent(Double.self, forKey: key) {
            return (value, value)
        }

        if let tuple = try? container.decode(UnclipRatioCodable.self, forKey: key) {
            return (tuple.x, tuple.y)
        }

        return fallback
    }

    private struct UnclipRatioCodable: Decodable {
        let x: Double
        let y: Double

        private enum CodingKeys: String, CodingKey {
            case x
            case y
            case width
            case height
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.x =
                try container.decodeIfPresent(Double.self, forKey: .x)
                ?? container.decodeIfPresent(Double.self, forKey: .width)
                ?? 1.0
            self.y =
                try container.decodeIfPresent(Double.self, forKey: .y)
                ?? container.decodeIfPresent(Double.self, forKey: .height)
                ?? 1.0
        }
    }

    private static let validMergeModes: Set<String> = ["union", "large", "small"]

    public static let defaultLayoutMergeBBoxesMode: [Int: String] = [
        0: "large",
        1: "large",
        2: "large",
        3: "large",
        4: "large",
        5: "large",
        6: "large",
        7: "large",
        8: "large",
        9: "large",
        10: "large",
        11: "large",
        12: "large",
        13: "large",
        14: "large",
        15: "large",
        16: "large",
        17: "large",
        18: "small",
        19: "large",
        20: "large",
        21: "large",
        22: "large",
        23: "large",
        24: "large",
    ]

    public static let defaultLabelTaskMapping: [String: [String]] = [
        "text": [
            "abstract", "algorithm", "content", "doc_title", "figure_title",
            "paragraph_title", "reference_content", "text", "vertical_text",
            "vision_footnote", "seal", "formula_number",
        ],
        "table": ["table"],
        "formula": ["display_formula", "inline_formula"],
        "skip": ["chart", "image"],
        "abandon": [
            "header", "footer", "number", "footnote", "aside_text", "reference",
            "footer_image", "header_image",
        ],
    ]

    public static func == (lhs: GlmOCRLayoutConfig, rhs: GlmOCRLayoutConfig) -> Bool {
        lhs.threshold == rhs.threshold
            && lhs.thresholdByClass == rhs.thresholdByClass
            && lhs.layoutNMS == rhs.layoutNMS
            && lhs.layoutUnclipRatio.0 == rhs.layoutUnclipRatio.0
            && lhs.layoutUnclipRatio.1 == rhs.layoutUnclipRatio.1
            && lhs.layoutMergeBBoxesMode == rhs.layoutMergeBBoxesMode
            && lhs.labelTaskMapping == rhs.labelTaskMapping
            && lhs.id2label == rhs.id2label
    }
}

public enum GlmOCRFigureFormat: String, Sendable, Codable, Equatable {
    case heic

    internal var fileExtension: String {
        switch self {
        case .heic:
            return "heic"
        }
    }

    internal var mimeType: String {
        switch self {
        case .heic:
            return "image/heic"
        }
    }
}

public struct GlmOCRMarkdownBundleConfig: Sendable, Codable, Equatable {
    public var enabled: Bool
    public var figureFormat: GlmOCRFigureFormat
    public var markdownFileName: String
    public var jsonFileName: String
    public var figuresDirectoryName: String
    public var heicCompressionQuality: Double

    public init(
        enabled: Bool = true,
        figureFormat: GlmOCRFigureFormat = .heic,
        markdownFileName: String = "document.md",
        jsonFileName: String = "document.json",
        figuresDirectoryName: String = "figures",
        heicCompressionQuality: Double = 0.82
    ) {
        self.enabled = enabled
        self.figureFormat = figureFormat
        self.markdownFileName = markdownFileName
        self.jsonFileName = jsonFileName
        self.figuresDirectoryName = figuresDirectoryName
        self.heicCompressionQuality = heicCompressionQuality
    }

    func validate() throws {
        try validatePathComponent(markdownFileName, field: "markdownBundle.markdownFileName")
        try validatePathComponent(jsonFileName, field: "markdownBundle.jsonFileName")
        try validatePathComponent(figuresDirectoryName, field: "markdownBundle.figuresDirectoryName")
        guard heicCompressionQuality >= 0, heicCompressionQuality <= 1 else {
            throw GlmOCRError.invalidConfiguration(
                "markdownBundle.heicCompressionQuality must be between 0 and 1"
            )
        }
    }

    private static let invalidPathCharacters = CharacterSet(charactersIn: "/\\")

    private func validatePathComponent(_ value: String, field: String) throws {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw GlmOCRError.invalidConfiguration("\(field) must not be empty")
        }
        guard trimmed.rangeOfCharacter(from: Self.invalidPathCharacters) == nil else {
            throw GlmOCRError.invalidConfiguration("\(field) must not contain path separators")
        }
        guard trimmed != "." && trimmed != ".." else {
            throw GlmOCRError.invalidConfiguration("\(field) must be a regular file/directory name")
        }
    }

    private enum CodingKeys: String, CodingKey {
        case enabled
        case figureFormat
        case markdownFileName
        case jsonFileName
        case figuresDirectoryName
        case heicCompressionQuality
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self()
        self.enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? defaults.enabled
        self.figureFormat =
            try container.decodeIfPresent(GlmOCRFigureFormat.self, forKey: .figureFormat)
            ?? defaults.figureFormat
        self.markdownFileName =
            try container.decodeIfPresent(String.self, forKey: .markdownFileName)
            ?? defaults.markdownFileName
        self.jsonFileName = try container.decodeIfPresent(String.self, forKey: .jsonFileName) ?? defaults.jsonFileName
        self.figuresDirectoryName =
            try container.decodeIfPresent(String.self, forKey: .figuresDirectoryName)
            ?? defaults.figuresDirectoryName
        self.heicCompressionQuality =
            try container.decodeIfPresent(Double.self, forKey: .heicCompressionQuality)
            ?? defaults.heicCompressionQuality
    }
}

public struct GlmOCRPerformanceConfig: Sendable, Codable, Equatable {
    public var inferenceBatchSize: Int
    public var inferenceBatchMaxWaitMs: Int
    public var inferenceMaxInflightJobs: Int
    public var pdfRenderConcurrency: Int
    public var ocrPreprocessConcurrency: Int
    public var bundleEncodeConcurrency: Int
    public var layoutPostprocessFastPath: Bool

    public init(
        inferenceBatchSize: Int = 4,
        inferenceBatchMaxWaitMs: Int = 8,
        inferenceMaxInflightJobs: Int = 64,
        pdfRenderConcurrency: Int = 2,
        ocrPreprocessConcurrency: Int = 4,
        bundleEncodeConcurrency: Int = 2,
        layoutPostprocessFastPath: Bool = true
    ) {
        self.inferenceBatchSize = inferenceBatchSize
        self.inferenceBatchMaxWaitMs = inferenceBatchMaxWaitMs
        self.inferenceMaxInflightJobs = inferenceMaxInflightJobs
        self.pdfRenderConcurrency = pdfRenderConcurrency
        self.ocrPreprocessConcurrency = ocrPreprocessConcurrency
        self.bundleEncodeConcurrency = bundleEncodeConcurrency
        self.layoutPostprocessFastPath = layoutPostprocessFastPath
    }

    func validate() throws {
        guard inferenceBatchSize > 0 else {
            throw GlmOCRError.invalidConfiguration("performance.inferenceBatchSize must be greater than zero")
        }
        guard inferenceBatchMaxWaitMs >= 0 else {
            throw GlmOCRError.invalidConfiguration("performance.inferenceBatchMaxWaitMs must be >= 0")
        }
        guard inferenceMaxInflightJobs > 0 else {
            throw GlmOCRError.invalidConfiguration("performance.inferenceMaxInflightJobs must be greater than zero")
        }
        guard pdfRenderConcurrency > 0 else {
            throw GlmOCRError.invalidConfiguration("performance.pdfRenderConcurrency must be greater than zero")
        }
        guard ocrPreprocessConcurrency > 0 else {
            throw GlmOCRError.invalidConfiguration("performance.ocrPreprocessConcurrency must be greater than zero")
        }
        guard bundleEncodeConcurrency > 0 else {
            throw GlmOCRError.invalidConfiguration("performance.bundleEncodeConcurrency must be greater than zero")
        }
    }

    private enum CodingKeys: String, CodingKey {
        case inferenceBatchSize
        case inferenceBatchMaxWaitMs
        case inferenceMaxInflightJobs
        case pdfRenderConcurrency
        case ocrPreprocessConcurrency
        case bundleEncodeConcurrency
        case layoutPostprocessFastPath
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self()
        self.inferenceBatchSize =
            try container.decodeIfPresent(Int.self, forKey: .inferenceBatchSize)
            ?? defaults.inferenceBatchSize
        self.inferenceBatchMaxWaitMs =
            try container.decodeIfPresent(Int.self, forKey: .inferenceBatchMaxWaitMs)
            ?? defaults.inferenceBatchMaxWaitMs
        self.inferenceMaxInflightJobs =
            try container.decodeIfPresent(Int.self, forKey: .inferenceMaxInflightJobs)
            ?? defaults.inferenceMaxInflightJobs
        self.pdfRenderConcurrency =
            try container.decodeIfPresent(Int.self, forKey: .pdfRenderConcurrency)
            ?? defaults.pdfRenderConcurrency
        self.ocrPreprocessConcurrency =
            try container.decodeIfPresent(Int.self, forKey: .ocrPreprocessConcurrency)
            ?? defaults.ocrPreprocessConcurrency
        self.bundleEncodeConcurrency =
            try container.decodeIfPresent(Int.self, forKey: .bundleEncodeConcurrency)
            ?? defaults.bundleEncodeConcurrency
        self.layoutPostprocessFastPath =
            try container.decodeIfPresent(Bool.self, forKey: .layoutPostprocessFastPath)
            ?? defaults.layoutPostprocessFastPath
    }
}

public struct GlmOCRConfig: Sendable, Codable, Equatable {
    public var recognizerModelID: String
    public var layoutModelID: String
    public var maxConcurrentRecognitions: Int
    public var enableLayout: Bool
    public var performance: GlmOCRPerformanceConfig
    public var markdownBundle: GlmOCRMarkdownBundleConfig
    public var recognitionOptions: GlmOCRRecognitionOptions
    public var prompts: GlmOCRPromptConfig
    public var layout: GlmOCRLayoutConfig
    public var pdfDPI: Double
    public var pdfMaxRenderedLongSide: Double
    /// Default PDF page cap used when `ParseOptions.maxPages` is not provided.
    /// If both are provided, effective cap is `min(ParseOptions.maxPages, defaultMaxPages)`.
    /// This cap applies to PDF inputs only.
    public var defaultMaxPages: Int?

    public init(
        recognizerModelID: String = "mlx-community/GLM-OCR-bf16",
        layoutModelID: String = "PaddlePaddle/PP-DocLayoutV3_safetensors",
        maxConcurrentRecognitions: Int = 1,
        enableLayout: Bool = true,
        performance: GlmOCRPerformanceConfig = .init(),
        markdownBundle: GlmOCRMarkdownBundleConfig = .init(),
        recognitionOptions: GlmOCRRecognitionOptions = .init(),
        prompts: GlmOCRPromptConfig = .init(),
        layout: GlmOCRLayoutConfig = .init(),
        pdfDPI: Double = 200,
        pdfMaxRenderedLongSide: Double = 3_500,
        defaultMaxPages: Int? = nil
    ) {
        self.recognizerModelID = recognizerModelID
        self.layoutModelID = layoutModelID
        self.maxConcurrentRecognitions = maxConcurrentRecognitions
        self.enableLayout = enableLayout
        self.performance = performance
        self.markdownBundle = markdownBundle
        self.recognitionOptions = recognitionOptions
        self.prompts = prompts
        self.layout = layout
        self.pdfDPI = pdfDPI
        self.pdfMaxRenderedLongSide = pdfMaxRenderedLongSide
        self.defaultMaxPages = defaultMaxPages
    }

    func validate() throws {
        let recognizer = recognizerModelID.trimmingCharacters(in: .whitespacesAndNewlines)
        let layoutModel = layoutModelID.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !recognizer.isEmpty else {
            throw GlmOCRError.invalidConfiguration("recognizerModelID must not be empty")
        }
        guard !layoutModel.isEmpty else {
            throw GlmOCRError.invalidConfiguration("layoutModelID must not be empty")
        }
        guard maxConcurrentRecognitions > 0 else {
            throw GlmOCRError.invalidConfiguration("maxConcurrentRecognitions must be greater than zero")
        }
        guard pdfDPI > 0 else {
            throw GlmOCRError.invalidConfiguration("pdfDPI must be greater than zero")
        }
        guard pdfMaxRenderedLongSide > 0 else {
            throw GlmOCRError.invalidConfiguration("pdfMaxRenderedLongSide must be greater than zero")
        }
        if let defaultMaxPages, defaultMaxPages <= 0 {
            throw GlmOCRError.invalidConfiguration("defaultMaxPages must be greater than zero when provided")
        }

        try performance.validate()
        try recognitionOptions.validate()
        try prompts.validate()
        try layout.validate()
        try markdownBundle.validate()
    }

    private enum CodingKeys: String, CodingKey {
        case recognizerModelID
        case layoutModelID
        case maxConcurrentRecognitions
        case enableLayout
        case performance
        case markdownBundle
        case recognitionOptions
        case prompts
        case layout
        case pdfDPI
        case pdfMaxRenderedLongSide
        case defaultMaxPages
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self()

        self.recognizerModelID =
            try container.decodeIfPresent(String.self, forKey: .recognizerModelID)
            ?? defaults.recognizerModelID
        self.layoutModelID =
            try container.decodeIfPresent(String.self, forKey: .layoutModelID)
            ?? defaults.layoutModelID
        self.maxConcurrentRecognitions =
            try container.decodeIfPresent(Int.self, forKey: .maxConcurrentRecognitions)
            ?? defaults.maxConcurrentRecognitions
        self.enableLayout = try container.decodeIfPresent(Bool.self, forKey: .enableLayout) ?? defaults.enableLayout
        self.performance =
            try container.decodeIfPresent(GlmOCRPerformanceConfig.self, forKey: .performance)
            ?? defaults.performance
        self.markdownBundle =
            try container.decodeIfPresent(GlmOCRMarkdownBundleConfig.self, forKey: .markdownBundle)
            ?? defaults.markdownBundle
        self.recognitionOptions =
            try container.decodeIfPresent(GlmOCRRecognitionOptions.self, forKey: .recognitionOptions)
            ?? defaults.recognitionOptions
        self.prompts = try container.decodeIfPresent(GlmOCRPromptConfig.self, forKey: .prompts) ?? defaults.prompts
        self.layout = try container.decodeIfPresent(GlmOCRLayoutConfig.self, forKey: .layout) ?? defaults.layout
        self.pdfDPI = try container.decodeIfPresent(Double.self, forKey: .pdfDPI) ?? defaults.pdfDPI
        self.pdfMaxRenderedLongSide =
            try container.decodeIfPresent(Double.self, forKey: .pdfMaxRenderedLongSide)
            ?? defaults.pdfMaxRenderedLongSide
        self.defaultMaxPages = try container.decodeIfPresent(Int.self, forKey: .defaultMaxPages)
    }
}

extension GlmOCRConfig {
    private static let legacyMaxConcurrentRecognitionsDefault = 1

    var effectivePerformanceConfig: GlmOCRPerformanceConfig {
        var resolved = performance
        let performanceIsDefault = performance == GlmOCRPerformanceConfig()
        let legacyConcurrencyOverridden = maxConcurrentRecognitions != Self.legacyMaxConcurrentRecognitionsDefault

        if performanceIsDefault || legacyConcurrencyOverridden {
            resolved.inferenceMaxInflightJobs = maxConcurrentRecognitions
        }

        return resolved
    }
}

public struct ParseOptions: Sendable, Codable, Equatable {
    public var includeMarkdown: Bool
    public var includeDiagnostics: Bool
    public var maxPages: Int?

    public init(
        includeMarkdown: Bool = true,
        includeDiagnostics: Bool = true,
        maxPages: Int? = nil
    ) {
        self.includeMarkdown = includeMarkdown
        self.includeDiagnostics = includeDiagnostics
        self.maxPages = maxPages
    }

    func validate() throws {
        if let maxPages, maxPages <= 0 {
            throw GlmOCRError.invalidConfiguration("maxPages must be greater than zero when provided")
        }
    }

    private enum CodingKeys: String, CodingKey {
        case includeMarkdown
        case includeDiagnostics
        case maxPages
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self()
        self.includeMarkdown =
            try container.decodeIfPresent(Bool.self, forKey: .includeMarkdown) ?? defaults.includeMarkdown
        self.includeDiagnostics =
            try container.decodeIfPresent(Bool.self, forKey: .includeDiagnostics)
            ?? defaults.includeDiagnostics
        self.maxPages = try container.decodeIfPresent(Int.self, forKey: .maxPages)
    }
}

public enum InputDocument: Sendable {
    case image(CGImage)
    case imageData(Data)
    case pdfData(Data)
}

public enum OCRTask: String, Sendable, Codable, Equatable, CaseIterable {
    case text
    case table
    case formula
}

public struct LayoutRegion: Sendable, Codable, Equatable {
    public let index: Int
    public let label: String
    public let score: Double
    public let bbox2D: [Int]

    public init(index: Int, label: String, score: Double, bbox2D: [Int]) {
        self.index = index
        self.label = label
        self.score = score
        self.bbox2D = bbox2D
    }
}

public struct OCRRegion: Sendable, Codable, Equatable {
    public let index: Int
    public let label: String
    public let content: String?
    public let bbox2D: [Int]?

    public init(index: Int, label: String, content: String?, bbox2D: [Int]?) {
        self.index = index
        self.label = label
        self.content = content
        self.bbox2D = bbox2D
    }
}

public struct OCRPageResult: Sendable, Codable, Equatable {
    public let regions: [OCRRegion]

    public init(regions: [OCRRegion]) {
        self.regions = regions
    }
}

public struct ParseDiagnostics: Sendable, Codable, Equatable {
    public let warnings: [String]
    public let timingsMs: [String: Double]
    public let metadata: [String: String]

    public init(
        warnings: [String] = [],
        timingsMs: [String: Double] = [:],
        metadata: [String: String] = [:]
    ) {
        self.warnings = warnings
        self.timingsMs = timingsMs
        self.metadata = metadata
    }
}

public struct OCRFigureAsset: Sendable, Codable, Equatable {
    public let pageIndex: Int
    public let regionIndex: Int
    public let label: String
    public let bbox2D: [Int]
    public let altText: String
    public let fileName: String
    public let relativePath: String
    public let widthPX: Int
    public let heightPX: Int
    public let mimeType: String
    public let sha256: String
    public let data: Data

    public init(
        pageIndex: Int,
        regionIndex: Int,
        label: String,
        bbox2D: [Int],
        altText: String,
        fileName: String,
        relativePath: String,
        widthPX: Int,
        heightPX: Int,
        mimeType: String,
        sha256: String,
        data: Data
    ) {
        self.pageIndex = pageIndex
        self.regionIndex = regionIndex
        self.label = label
        self.bbox2D = bbox2D
        self.altText = altText
        self.fileName = fileName
        self.relativePath = relativePath
        self.widthPX = widthPX
        self.heightPX = heightPX
        self.mimeType = mimeType
        self.sha256 = sha256
        self.data = data
    }
}

public struct OCRMarkdownBundle: Sendable, Codable, Equatable {
    public let rewrittenMarkdown: String
    public let documentJSON: String
    public let markdownFileName: String
    public let jsonFileName: String
    public let figuresDirectoryName: String
    public let figures: [OCRFigureAsset]

    public init(
        rewrittenMarkdown: String,
        documentJSON: String,
        markdownFileName: String,
        jsonFileName: String,
        figuresDirectoryName: String,
        figures: [OCRFigureAsset]
    ) {
        self.rewrittenMarkdown = rewrittenMarkdown
        self.documentJSON = documentJSON
        self.markdownFileName = markdownFileName
        self.jsonFileName = jsonFileName
        self.figuresDirectoryName = figuresDirectoryName
        self.figures = figures
    }
}

public struct OCRDocumentResult: Sendable, Codable, Equatable {
    public let pages: [OCRPageResult]
    public let markdown: String
    public let diagnostics: ParseDiagnostics
    public let markdownBundle: OCRMarkdownBundle?

    public init(
        pages: [OCRPageResult],
        markdown: String,
        diagnostics: ParseDiagnostics,
        markdownBundle: OCRMarkdownBundle? = nil
    ) {
        self.pages = pages
        self.markdown = markdown
        self.diagnostics = diagnostics
        self.markdownBundle = markdownBundle
    }
}

public protocol LayoutDetector: Sendable {
    func detect(pages: [CGImage], options: ParseOptions) async throws -> [[LayoutRegion]]
}

public protocol RegionRecognizer: Sendable {
    func recognize(_ region: CGImage, task: OCRTask) async throws -> String
}

public enum GlmOCRError: Error, Sendable, Equatable {
    case invalidConfiguration(String)
    case pdfRenderingFailed(String)
    case modelDeliveryFailed(String)
    case notImplemented(String)
}
