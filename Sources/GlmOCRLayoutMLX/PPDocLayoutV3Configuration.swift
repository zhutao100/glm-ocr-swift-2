import Foundation

internal struct PPDocLayoutV3Configuration: Sendable, Equatable {
    internal let modelType: String
    internal let numLabels: Int
    internal let numQueries: Int
    internal let id2label: [Int: String]

    internal let dModel: Int
    internal let encoderHiddenDim: Int
    internal let encoderInChannels: [Int]
    internal let decoderInChannels: [Int]
    internal let featStrides: [Int]
    internal let numFeatureLevels: Int

    internal let encoderLayers: Int
    internal let decoderLayers: Int
    internal let encoderAttentionHeads: Int
    internal let decoderAttentionHeads: Int
    internal let encoderFFNDim: Int
    internal let decoderFFNDim: Int
    internal let decoderNPoints: Int

    internal let encodeProjLayers: [Int]
    internal let activationFunction: String
    internal let encoderActivationFunction: String
    internal let decoderActivationFunction: String

    internal let layerNormEps: Float
    internal let batchNormEps: Float
    internal let hiddenExpansion: Float

    internal let maskFeatureChannels: [Int]
    internal let x4FeatDim: Int
    internal let numPrototypes: Int
    internal let maskEnhanced: Bool

    internal let globalPointerHeadSize: Int
    internal let positionalEncodingTemperature: Int

    internal static let fallback = PPDocLayoutV3Configuration(
        modelType: "pp_doclayout_v3",
        numLabels: PPDocLayoutMLXContract.logitsShape[2],
        numQueries: PPDocLayoutMLXContract.logitsShape[1],
        id2label: PPDocLayoutMLXContract.id2label,
        dModel: 256,
        encoderHiddenDim: 256,
        encoderInChannels: [512, 1024, 2048],
        decoderInChannels: [256, 256, 256],
        featStrides: [8, 16, 32],
        numFeatureLevels: 3,
        encoderLayers: 1,
        decoderLayers: 6,
        encoderAttentionHeads: 8,
        decoderAttentionHeads: 8,
        encoderFFNDim: 1024,
        decoderFFNDim: 1024,
        decoderNPoints: 4,
        encodeProjLayers: [2],
        activationFunction: "silu",
        encoderActivationFunction: "gelu",
        decoderActivationFunction: "relu",
        layerNormEps: 1e-5,
        batchNormEps: 1e-5,
        hiddenExpansion: 1.0,
        maskFeatureChannels: [64, 64],
        x4FeatDim: 128,
        numPrototypes: 32,
        maskEnhanced: true,
        globalPointerHeadSize: 64,
        positionalEncodingTemperature: 10_000
    )

    internal static func load(from modelDirectory: URL) throws -> PPDocLayoutV3Configuration {
        let configURL = modelDirectory.appending(path: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw PPDocLayoutMLXError.missingModelFile("Missing config.json in \(modelDirectory.path)")
        }

        let data = try Data(contentsOf: configURL)

        do {
            let decoded = try JSONDecoder().decode(RawConfiguration.self, from: data)
            let mappedLabels = decoded.id2label.reduce(into: [Int: String]()) { partialResult, pair in
                if let key = Int(pair.key) {
                    partialResult[key] = pair.value
                }
            }

            let fallback = Self.fallback

            return PPDocLayoutV3Configuration(
                modelType: decoded.modelType ?? fallback.modelType,
                numLabels: decoded.numLabels ?? mappedLabels.count.nonZeroOr(fallback.numLabels),
                numQueries: decoded.numQueries ?? fallback.numQueries,
                id2label: mappedLabels.isEmpty ? fallback.id2label : mappedLabels,
                dModel: decoded.dModel ?? fallback.dModel,
                encoderHiddenDim: decoded.encoderHiddenDim ?? fallback.encoderHiddenDim,
                encoderInChannels: decoded.encoderInChannels ?? fallback.encoderInChannels,
                decoderInChannels: decoded.decoderInChannels ?? fallback.decoderInChannels,
                featStrides: decoded.featureStrides ?? fallback.featStrides,
                numFeatureLevels: decoded.numFeatureLevels ?? fallback.numFeatureLevels,
                encoderLayers: decoded.encoderLayers ?? fallback.encoderLayers,
                decoderLayers: decoded.decoderLayers ?? fallback.decoderLayers,
                encoderAttentionHeads: decoded.encoderAttentionHeads ?? fallback.encoderAttentionHeads,
                decoderAttentionHeads: decoded.decoderAttentionHeads ?? fallback.decoderAttentionHeads,
                encoderFFNDim: decoded.encoderFFNDim ?? fallback.encoderFFNDim,
                decoderFFNDim: decoded.decoderFFNDim ?? fallback.decoderFFNDim,
                decoderNPoints: decoded.decoderNPoints ?? fallback.decoderNPoints,
                encodeProjLayers: decoded.encodeProjLayers ?? fallback.encodeProjLayers,
                activationFunction: decoded.activationFunction ?? fallback.activationFunction,
                encoderActivationFunction: decoded.encoderActivationFunction ?? fallback.encoderActivationFunction,
                decoderActivationFunction: decoded.decoderActivationFunction ?? fallback.decoderActivationFunction,
                layerNormEps: decoded.layerNormEps ?? fallback.layerNormEps,
                batchNormEps: decoded.batchNormEps ?? fallback.batchNormEps,
                hiddenExpansion: decoded.hiddenExpansion ?? fallback.hiddenExpansion,
                maskFeatureChannels: decoded.maskFeatureChannels ?? fallback.maskFeatureChannels,
                x4FeatDim: decoded.x4FeatDim ?? fallback.x4FeatDim,
                numPrototypes: decoded.numPrototypes ?? fallback.numPrototypes,
                maskEnhanced: decoded.maskEnhanced ?? fallback.maskEnhanced,
                globalPointerHeadSize: decoded.globalPointerHeadSize ?? fallback.globalPointerHeadSize,
                positionalEncodingTemperature: decoded.positionalEncodingTemperature
                    ?? fallback.positionalEncodingTemperature
            )
        } catch {
            throw PPDocLayoutMLXError.modelConfigurationDecodeFailed(
                "Failed to decode config.json at \(configURL.path): \(error.localizedDescription)"
            )
        }
    }
}

private struct RawConfiguration: Decodable {
    let modelType: String?
    let numLabels: Int?
    let numQueries: Int?
    let id2label: [String: String]

    let dModel: Int?
    let encoderHiddenDim: Int?
    let encoderInChannels: [Int]?
    let decoderInChannels: [Int]?
    let featureStrides: [Int]?
    let numFeatureLevels: Int?

    let encoderLayers: Int?
    let decoderLayers: Int?
    let encoderAttentionHeads: Int?
    let decoderAttentionHeads: Int?
    let encoderFFNDim: Int?
    let decoderFFNDim: Int?
    let decoderNPoints: Int?

    let encodeProjLayers: [Int]?
    let activationFunction: String?
    let encoderActivationFunction: String?
    let decoderActivationFunction: String?

    let layerNormEps: Float?
    let batchNormEps: Float?
    let hiddenExpansion: Float?

    let maskFeatureChannels: [Int]?
    let x4FeatDim: Int?
    let numPrototypes: Int?
    let maskEnhanced: Bool?

    let globalPointerHeadSize: Int?
    let positionalEncodingTemperature: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numLabels = "num_labels"
        case numQueries = "num_queries"
        case id2label

        case dModel = "d_model"
        case encoderHiddenDim = "encoder_hidden_dim"
        case encoderInChannels = "encoder_in_channels"
        case decoderInChannels = "decoder_in_channels"
        case featureStrides = "feature_strides"
        case numFeatureLevels = "num_feature_levels"

        case encoderLayers = "encoder_layers"
        case decoderLayers = "decoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case decoderAttentionHeads = "decoder_attention_heads"
        case encoderFFNDim = "encoder_ffn_dim"
        case decoderFFNDim = "decoder_ffn_dim"
        case decoderNPoints = "decoder_n_points"

        case encodeProjLayers = "encode_proj_layers"
        case activationFunction = "activation_function"
        case encoderActivationFunction = "encoder_activation_function"
        case decoderActivationFunction = "decoder_activation_function"

        case layerNormEps = "layer_norm_eps"
        case batchNormEps = "batch_norm_eps"
        case hiddenExpansion = "hidden_expansion"

        case maskFeatureChannels = "mask_feature_channels"
        case x4FeatDim = "x4_feat_dim"
        case numPrototypes = "num_prototypes"
        case maskEnhanced = "mask_enhanced"

        case globalPointerHeadSize = "global_pointer_head_size"
        case positionalEncodingTemperature = "positional_encoding_temperature"
    }
}

extension Int {
    fileprivate func nonZeroOr(_ fallback: Int) -> Int {
        self > 0 ? self : fallback
    }
}
