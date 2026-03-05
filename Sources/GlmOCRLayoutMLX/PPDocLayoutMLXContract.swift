import Foundation

package enum LayoutMergeMode: String, Sendable {
    case union
    case large
    case small
}

internal enum PPDocLayoutMLXContract {
    internal static let inputName = "pixel_values"
    internal static let inputShape = [1, 3, 800, 800]

    internal static let logitsOutputName = "logits"
    internal static let predBoxesOutputName = "pred_boxes"
    internal static let orderLogitsOutputName = "order_logits"
    internal static let outMasksOutputName = "out_masks"

    internal static let logitsShape = [1, 300, 25]
    internal static let predBoxesShape = [1, 300, 4]
    internal static let orderLogitsShape = [1, 300, 300]
    internal static let outMasksShape = [1, 300, 200, 200]

    internal static let requiredOutputs: [(name: String, shape: [Int])] = [
        (name: logitsOutputName, shape: logitsShape),
        (name: predBoxesOutputName, shape: predBoxesShape),
        (name: orderLogitsOutputName, shape: orderLogitsShape),
        (name: outMasksOutputName, shape: outMasksShape),
    ]

    internal static let defaultDetectionThreshold: Float = 0.3
    internal static let nmsSameClassIoUThreshold = 0.6
    internal static let nmsDifferentClassIoUThreshold = 0.98
    internal static let containmentThreshold = 0.8
    internal static let defaultUnclipRatio: (Double, Double) = (1.0, 1.0)

    internal static let id2label: [Int: String] = [
        0: "abstract",
        1: "algorithm",
        2: "aside_text",
        3: "chart",
        4: "content",
        5: "display_formula",
        6: "doc_title",
        7: "figure_title",
        8: "footer",
        9: "footer_image",
        10: "footnote",
        11: "formula_number",
        12: "header",
        13: "header_image",
        14: "image",
        15: "inline_formula",
        16: "number",
        17: "paragraph_title",
        18: "reference",
        19: "reference_content",
        20: "seal",
        21: "table",
        22: "text",
        23: "vertical_text",
        24: "vision_footnote",
    ]

    internal static let labelTaskMapping: [String: Set<String>] = [
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

    internal static let layoutMergeBBoxesMode: [Int: LayoutMergeMode] = [
        0: .large,
        1: .large,
        2: .large,
        3: .large,
        4: .large,
        5: .large,
        6: .large,
        7: .large,
        8: .large,
        9: .large,
        10: .large,
        11: .large,
        12: .large,
        13: .large,
        14: .large,
        15: .large,
        16: .large,
        17: .large,
        18: .small,
        19: .large,
        20: .large,
        21: .large,
        22: .large,
        23: .large,
        24: .large,
    ]

    internal static let preservedContainmentLabels: Set<String> = ["image", "seal", "chart"]
}

internal enum PPDocLayoutMLXContractValidator {
    internal static func validateInputShape(_ actual: [Int]) throws {
        guard actual == PPDocLayoutMLXContract.inputShape else {
            throw PPDocLayoutMLXError.invalidInputShape(
                expected: PPDocLayoutMLXContract.inputShape,
                actual: actual
            )
        }
    }

    internal static func validateOutputShapes(_ outputShapes: [String: [Int]]) throws {
        for required in PPDocLayoutMLXContract.requiredOutputs {
            guard let actual = outputShapes[required.name] else {
                throw PPDocLayoutMLXError.missingOutput(required.name)
            }

            guard actual == required.shape else {
                throw PPDocLayoutMLXError.invalidOutputShape(
                    name: required.name,
                    expected: required.shape,
                    actual: actual
                )
            }
        }
    }

    internal static func validatePrediction(_ prediction: PPDocLayoutMLXPrediction) throws {
        try validateOutputShapes(prediction.outputShapes())
    }
}
