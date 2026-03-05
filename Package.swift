// swift-tools-version: 6.0
import PackageDescription

let strictConcurrency: [SwiftSetting] = [
    .unsafeFlags(["-strict-concurrency=complete"])
]

let package = Package(
    name: "GlmOCRSwift",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "GlmOCRSwift",
            targets: ["GlmOCRSwift"]
        ),
        .executable(
            name: "GlmOCRBenchmark",
            targets: ["GlmOCRBenchmark"]
        ),
        .executable(
            name: "GlmOCRCLI",
            targets: ["GlmOCRCLI"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/huggingface/swift-huggingface.git",
            from: "0.8.0"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            .upToNextMinor(from: "0.30.6")
        ),
        .package(
            url: "https://github.com/DePasqualeOrg/swift-tokenizers.git",
            revision: "dbf45b169dbdfadd0c24ea7a9c5c47e1f311f280"
        ),
    ],
    targets: [
        .target(
            name: "GlmOCRSwift",
            dependencies: [
                "GlmOCRCore",
                "GlmOCRLayoutMLX",
                "GlmOCRModelDelivery",
                "GlmOCRFormatting",
                "GlmOCRRecognizerMLX",
                .target(name: "GlmOCRPDFium", condition: .when(platforms: [.macOS])),
            ],
            swiftSettings: strictConcurrency
        ),
        .target(
            name: "GlmOCRCore",
            swiftSettings: strictConcurrency
        ),
        .target(
            name: "GlmOCRLayoutMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            swiftSettings: strictConcurrency
        ),
        .target(
            name: "GlmOCRModelDelivery",
            dependencies: [
                .product(name: "HuggingFace", package: "swift-huggingface")
            ],
            resources: [
                .process("Resources")
            ],
            swiftSettings: strictConcurrency
        ),
        .target(
            name: "GlmOCRFormatting",
            swiftSettings: strictConcurrency
        ),
        .target(
            name: "GlmOCRRecognizerMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-tokenizers"),
            ],
            swiftSettings: strictConcurrency
        ),
        .target(
            name: "GlmOCRPDFium",
            dependencies: [
                "CPDFium",
                .target(name: "PDFiumBinary", condition: .when(platforms: [.macOS])),
            ],
            swiftSettings: strictConcurrency
        ),
        .target(
            name: "CPDFium",
            publicHeadersPath: "include"
        ),
        .binaryTarget(
            name: "PDFiumBinary",
            path: "Vendor/PDFium/PDFium.xcframework"
        ),
        .testTarget(
            name: "GlmOCRTests",
            dependencies: [
                "GlmOCRSwift",
                "GlmOCRRecognizerMLX",
            ],
            swiftSettings: strictConcurrency
        ),
        .executableTarget(
            name: "GlmOCRBenchmark",
            dependencies: [
                "GlmOCRSwift"
            ],
            swiftSettings: strictConcurrency
        ),
        .executableTarget(
            name: "GlmOCRCLI",
            dependencies: [
                "GlmOCRSwift"
            ],
            swiftSettings: strictConcurrency
        ),
    ],
    swiftLanguageModes: [.v6]
)
