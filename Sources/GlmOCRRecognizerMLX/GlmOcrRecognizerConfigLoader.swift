import Foundation

public enum GlmOcrRecognizerConfigLoader {
    public static func loadBundle(modelDirectory: URL) throws -> GlmOcrModelBundle {
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: modelDirectory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw GlmOcrRecognizerMLXError.invalidModelDirectory(modelDirectory.path)
        }

        let modelConfig = try loadModelConfig(modelDirectory: modelDirectory)
        let processorConfig = try loadProcessorConfig(modelDirectory: modelDirectory)
        let generationConfig = try loadGenerationConfig(
            modelDirectory: modelDirectory,
            fallbackEos: modelConfig.eosTokenIDs,
            fallbackPad: modelConfig.padTokenID
        )

        return GlmOcrModelBundle(
            modelDirectory: modelDirectory,
            modelConfig: modelConfig,
            processorConfig: processorConfig,
            generationConfig: generationConfig
        )
    }

    private static func loadModelConfig(modelDirectory: URL) throws -> GlmOcrModelConfig {
        let url = modelDirectory.appending(path: "config.json")
        let data = try readFile(at: url)
        let raw = try JSONDecoder().decode(RawModelConfig.self, from: data)

        let rope = GlmOcrModelConfig.RopeParameters(
            mropeSection: raw.textConfig.ropeParameters.mropeSection ?? [16, 24, 24],
            ropeTheta: raw.textConfig.ropeParameters.ropeTheta ?? 10_000,
            partialRotaryFactor: raw.textConfig.ropeParameters.partialRotaryFactor ?? 1
        )

        let text = GlmOcrModelConfig.TextConfig(
            vocabSize: raw.textConfig.vocabSize,
            hiddenSize: raw.textConfig.hiddenSize,
            intermediateSize: raw.textConfig.intermediateSize,
            numAttentionHeads: raw.textConfig.numAttentionHeads,
            numKeyValueHeads: raw.textConfig.numKeyValueHeads ?? raw.textConfig.numAttentionHeads,
            numHiddenLayers: raw.textConfig.numHiddenLayers,
            headDim: raw.textConfig.headDim ?? (raw.textConfig.hiddenSize / raw.textConfig.numAttentionHeads),
            rmsNormEps: raw.textConfig.rmsNormEps ?? 1e-5,
            attentionBias: raw.textConfig.attentionBias ?? false,
            maxPositionEmbeddings: raw.textConfig.maxPositionEmbeddings ?? 131_072,
            ropeParameters: rope
        )

        let vision = GlmOcrModelConfig.VisionConfig(
            depth: raw.visionConfig.depth,
            hiddenSize: raw.visionConfig.hiddenSize,
            intermediateSize: raw.visionConfig.intermediateSize,
            numHeads: raw.visionConfig.numHeads,
            patchSize: raw.visionConfig.patchSize,
            spatialMergeSize: raw.visionConfig.spatialMergeSize,
            temporalPatchSize: raw.visionConfig.temporalPatchSize ?? 2,
            inChannels: raw.visionConfig.inChannels ?? 3,
            outHiddenSize: raw.visionConfig.outHiddenSize ?? raw.textConfig.hiddenSize,
            rmsNormEps: raw.visionConfig.rmsNormEps ?? 1e-5,
            attentionBias: raw.visionConfig.attentionBias ?? true
        )

        return GlmOcrModelConfig(
            modelType: raw.modelType ?? "glm_ocr",
            vocabSize: raw.vocabSize ?? raw.textConfig.vocabSize,
            imageTokenID: raw.imageTokenID ?? 59_280,
            videoTokenID: raw.videoTokenID ?? 59_281,
            imageStartTokenID: raw.imageStartTokenID ?? 59_256,
            imageEndTokenID: raw.imageEndTokenID ?? 59_257,
            eosTokenIDs: raw.eosTokenID?.values ?? [59_246, 59_253],
            padTokenID: raw.textConfig.padTokenID,
            tieWordEmbeddings: raw.tieWordEmbeddings ?? false,
            textConfig: text,
            visionConfig: vision
        )
    }

    private static func loadProcessorConfig(modelDirectory: URL) throws -> GlmOcrProcessorConfig {
        let processorURL = modelDirectory.appending(path: "processor_config.json")
        let preprocessorURL = modelDirectory.appending(path: "preprocessor_config.json")

        if FileManager.default.fileExists(atPath: processorURL.path) {
            let data = try readFile(at: processorURL)
            let raw = try JSONDecoder().decode(RawProcessorRoot.self, from: data)
            if let imageProcessor = raw.imageProcessor {
                return materializeProcessorConfig(from: imageProcessor)
            }
        }

        if FileManager.default.fileExists(atPath: preprocessorURL.path) {
            let data = try readFile(at: preprocessorURL)
            let raw = try JSONDecoder().decode(RawImageProcessor.self, from: data)
            return materializeProcessorConfig(from: raw)
        }

        throw GlmOcrRecognizerMLXError.missingRequiredFile(
            "Missing processor_config.json and preprocessor_config.json in \(modelDirectory.path)"
        )
    }

    private static func loadGenerationConfig(
        modelDirectory: URL,
        fallbackEos: [Int],
        fallbackPad: Int
    ) throws -> GlmOcrGenerationConfig {
        let url = modelDirectory.appending(path: "generation_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            return GlmOcrGenerationConfig(eosTokenIDs: fallbackEos, padTokenID: fallbackPad)
        }

        let data = try readFile(at: url)
        let raw = try JSONDecoder().decode(RawGenerationConfig.self, from: data)
        return GlmOcrGenerationConfig(
            eosTokenIDs: raw.eosTokenID?.values ?? fallbackEos,
            padTokenID: raw.padTokenID ?? fallbackPad
        )
    }

    private static func materializeProcessorConfig(from raw: RawImageProcessor) -> GlmOcrProcessorConfig {
        let minPixels = raw.minPixels ?? raw.size?.shortestEdge ?? 112 * 112
        let maxPixels = raw.maxPixels ?? raw.size?.longestEdge ?? (14 * 14 * 4 * 1_280)

        return GlmOcrProcessorConfig(
            imageMean: raw.imageMean ?? [0.48145466, 0.4578275, 0.40821073],
            imageStd: raw.imageStd ?? [0.26862954, 0.26130258, 0.27577711],
            patchSize: raw.patchSize ?? 14,
            mergeSize: raw.spatialMergeSize ?? raw.mergeSize ?? 2,
            temporalPatchSize: raw.temporalPatchSize ?? 2,
            minPixels: minPixels,
            maxPixels: maxPixels,
            imageToken: raw.imageToken ?? "<|image|>"
        )
    }

    private static func readFile(at url: URL) throws -> Data {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GlmOcrRecognizerMLXError.missingRequiredFile(url.path)
        }

        do {
            return try Data(contentsOf: url)
        } catch {
            throw GlmOcrRecognizerMLXError.invalidConfiguration(
                "Unable to read \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
    }
}

private struct RawIntOrArray: Decodable {
    let values: [Int]

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            values = [value]
        } else if let list = try? container.decode([Int].self) {
            values = list
        } else {
            values = []
        }
    }
}

private struct RawGenerationConfig: Decodable {
    let eosTokenID: RawIntOrArray?
    let padTokenID: Int?

    enum CodingKeys: String, CodingKey {
        case eosTokenID = "eos_token_id"
        case padTokenID = "pad_token_id"
    }
}

private struct RawModelConfig: Decodable {
    struct RawTextConfig: Decodable {
        struct RawRope: Decodable {
            let mropeSection: [Int]?
            let ropeTheta: Float?
            let partialRotaryFactor: Float?

            enum CodingKeys: String, CodingKey {
                case mropeSection = "mrope_section"
                case ropeTheta = "rope_theta"
                case partialRotaryFactor = "partial_rotary_factor"
            }
        }

        let vocabSize: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let numAttentionHeads: Int
        let numKeyValueHeads: Int?
        let numHiddenLayers: Int
        let headDim: Int?
        let rmsNormEps: Float?
        let attentionBias: Bool?
        let maxPositionEmbeddings: Int?
        let padTokenID: Int
        let ropeParameters: RawRope

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case numHiddenLayers = "num_hidden_layers"
            case headDim = "head_dim"
            case rmsNormEps = "rms_norm_eps"
            case attentionBias = "attention_bias"
            case maxPositionEmbeddings = "max_position_embeddings"
            case padTokenID = "pad_token_id"
            case ropeParameters = "rope_parameters"
        }
    }

    struct RawVisionConfig: Decodable {
        let depth: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let numHeads: Int
        let patchSize: Int
        let spatialMergeSize: Int
        let temporalPatchSize: Int?
        let inChannels: Int?
        let outHiddenSize: Int?
        let rmsNormEps: Float?
        let attentionBias: Bool?

        enum CodingKeys: String, CodingKey {
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case inChannels = "in_channels"
            case outHiddenSize = "out_hidden_size"
            case rmsNormEps = "rms_norm_eps"
            case attentionBias = "attention_bias"
        }
    }

    let modelType: String?
    let vocabSize: Int?
    let imageTokenID: Int?
    let videoTokenID: Int?
    let imageStartTokenID: Int?
    let imageEndTokenID: Int?
    let eosTokenID: RawIntOrArray?
    let tieWordEmbeddings: Bool?
    let textConfig: RawTextConfig
    let visionConfig: RawVisionConfig

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case imageStartTokenID = "image_start_token_id"
        case imageEndTokenID = "image_end_token_id"
        case eosTokenID = "eos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }
}

private struct RawProcessorRoot: Decodable {
    let imageProcessor: RawImageProcessor?

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
    }
}

private struct RawImageProcessor: Decodable {
    struct RawSize: Decodable {
        let shortestEdge: Int?
        let longestEdge: Int?

        enum CodingKeys: String, CodingKey {
            case shortestEdge = "shortest_edge"
            case longestEdge = "longest_edge"
        }
    }

    let imageMean: [Float]?
    let imageStd: [Float]?
    let patchSize: Int?
    let mergeSize: Int?
    let spatialMergeSize: Int?
    let temporalPatchSize: Int?
    let size: RawSize?
    let minPixels: Int?
    let maxPixels: Int?
    let imageToken: String?

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case patchSize = "patch_size"
        case mergeSize = "merge_size"
        case spatialMergeSize = "spatial_merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case size
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case imageToken = "image_token"
    }
}
