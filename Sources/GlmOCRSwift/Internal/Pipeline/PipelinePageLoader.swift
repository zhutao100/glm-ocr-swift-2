import CoreGraphics
import Dispatch
import Foundation
import ImageIO

#if os(macOS)
    import GlmOCRPDFium
#endif

internal protocol PipelinePageLoading: Sendable {
    func loadPages(
        from input: InputDocument,
        maxPages: Int?
    ) throws -> [CGImage]
}

internal struct PipelinePageLoader: PipelinePageLoading {
    private final class RenderState: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var firstError: GlmOCRError?
        private var renderedPages: [CGImage?]

        init(pageCount: Int) {
            self.renderedPages = Array(repeating: nil, count: pageCount)
        }

        func shouldSkip() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return firstError != nil
        }

        func storeRenderedPage(_ image: CGImage, at index: Int) {
            lock.lock()
            renderedPages[index] = image
            lock.unlock()
        }

        func storeError(_ error: GlmOCRError) {
            lock.lock()
            if firstError == nil {
                firstError = error
            }
            lock.unlock()
        }

        func finish() throws -> [CGImage] {
            lock.lock()
            defer { lock.unlock() }
            if let firstError {
                throw firstError
            }
            return renderedPages.compactMap { $0 }
        }
    }

    private let pdfDPI: Double
    private let maxRenderedLongSide: Double
    private let defaultMaxPages: Int?
    private let renderConcurrency: Int

    internal init(
        pdfDPI: Double = 200.0,
        maxRenderedLongSide: Double = 3500.0,
        defaultMaxPages: Int? = nil,
        renderConcurrency: Int = 2
    ) {
        self.pdfDPI = pdfDPI
        self.maxRenderedLongSide = maxRenderedLongSide
        self.defaultMaxPages = defaultMaxPages
        self.renderConcurrency = max(1, renderConcurrency)
    }

    internal func loadPages(
        from input: InputDocument,
        maxPages: Int?
    ) throws -> [CGImage] {
        switch input {
        case .image(let image):
            return [image]
        case .imageData(let data):
            return [try decodeImageData(data)]
        case .pdfData(let data):
            return try decodePDFData(data, maxPages: maxPages)
        }
    }

    private func decodeImageData(_ data: Data) throws -> CGImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw GlmOCRError.invalidConfiguration("Unable to decode imageData input")
        }

        return image
    }

    private func decodePDFData(
        _ data: Data,
        maxPages: Int?
    ) throws -> [CGImage] {
        #if os(macOS)
            let document: PDFiumDocument
            do {
                document = try PDFiumDocument(data: data)
            } catch let error as PDFiumError {
                throw GlmOCRError.pdfRenderingFailed(describePDFiumError(error))
            } catch {
                throw GlmOCRError.pdfRenderingFailed(
                    "Unable to initialize PDFium document: \(error.localizedDescription)")
            }

            let pageCount = document.pageCount
            guard pageCount > 0 else {
                throw GlmOCRError.invalidConfiguration("pdfData input has no pages")
            }

            let requestedPageCount = try resolveRequestedPageCount(
                pageCount: pageCount,
                maxPages: maxPages
            )
            if renderConcurrency <= 1 {
                return try renderSequentialPDFiumPages(
                    document: document,
                    requestedPageCount: requestedPageCount
                )
            }

            let state = RenderState(pageCount: requestedPageCount)
            let group = DispatchGroup()
            let semaphore = DispatchSemaphore(value: renderConcurrency)

            for pageIndex in 0..<requestedPageCount {
                semaphore.wait()
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    defer {
                        semaphore.signal()
                        group.leave()
                    }

                    if state.shouldSkip() {
                        return
                    }

                    do {
                        let rendered = try renderSinglePDFiumPage(
                            data: data,
                            pageIndex: pageIndex
                        )
                        state.storeRenderedPage(rendered, at: pageIndex)
                    } catch let error as GlmOCRError {
                        state.storeError(error)
                    } catch {
                        state.storeError(
                            GlmOCRError.pdfRenderingFailed(
                                "Unable to render PDF page \(pageIndex + 1): \(error.localizedDescription)"
                            )
                        )
                    }
                }
            }

            group.wait()
            return try state.finish()
        #else
            guard let provider = CGDataProvider(data: data as CFData),
                let document = CGPDFDocument(provider)
            else {
                throw GlmOCRError.invalidConfiguration("Unable to decode pdfData input")
            }

            let pageCount = document.numberOfPages
            guard pageCount > 0 else {
                throw GlmOCRError.invalidConfiguration("pdfData input has no pages")
            }

            let requestedPageCount = try resolveRequestedPageCount(pageCount: pageCount, maxPages: maxPages)
            if renderConcurrency <= 1 {
                return try renderSequentialCGPDFPages(
                    document: document,
                    requestedPageCount: requestedPageCount
                )
            }

            let state = RenderState(pageCount: requestedPageCount)
            let group = DispatchGroup()
            let semaphore = DispatchSemaphore(value: renderConcurrency)

            for pageIndex in 0..<requestedPageCount {
                semaphore.wait()
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    defer {
                        semaphore.signal()
                        group.leave()
                    }

                    if state.shouldSkip() {
                        return
                    }

                    do {
                        let rendered = try renderSingleCGPDFPage(
                            data: data,
                            pageNumber: pageIndex + 1
                        )
                        state.storeRenderedPage(rendered, at: pageIndex)
                    } catch let error as GlmOCRError {
                        state.storeError(error)
                    } catch {
                        state.storeError(
                            GlmOCRError.pdfRenderingFailed(
                                "Unable to render PDF page \(pageIndex + 1): \(error.localizedDescription)"
                            )
                        )
                    }
                }
            }

            group.wait()
            return try state.finish()
        #endif
    }

    private func resolveRequestedPageCount(
        pageCount: Int,
        maxPages: Int?
    ) throws -> Int {
        let effectiveLimit: Int?
        // `defaultMaxPages` applies to PDF only, with `maxPages` taking precedence only when explicitly set.
        if let maxPages, let defaultMaxPages {
            effectiveLimit = min(maxPages, defaultMaxPages)
        } else if let maxPages {
            effectiveLimit = maxPages
        } else {
            effectiveLimit = defaultMaxPages
        }

        let requestedPageCount = effectiveLimit.map { min($0, pageCount) } ?? pageCount
        guard requestedPageCount > 0 else {
            throw GlmOCRError.invalidConfiguration("No PDF pages selected after maxPages filtering")
        }

        return requestedPageCount
    }

    #if os(macOS)
        private func renderSequentialPDFiumPages(
            document: PDFiumDocument,
            requestedPageCount: Int
        ) throws -> [CGImage] {
            let rasterizer = PDFiumPageRasterizer()
            var renderedPages: [CGImage] = []
            renderedPages.reserveCapacity(requestedPageCount)

            for pageIndex in 0..<requestedPageCount {
                let page: PDFiumPage
                do {
                    page = try document.page(at: pageIndex)
                } catch let error as PDFiumError {
                    throw GlmOCRError.pdfRenderingFailed(describePDFiumError(error))
                } catch {
                    throw GlmOCRError.pdfRenderingFailed(
                        "Unable to load PDF page \(pageIndex + 1): \(error.localizedDescription)")
                }

                do {
                    let rendered = try rasterizer.render(
                        page: page,
                        dpi: pdfDPI,
                        maxRenderedLongSide: maxRenderedLongSide
                    )
                    renderedPages.append(rendered)
                } catch let error as PDFiumError {
                    throw GlmOCRError.pdfRenderingFailed(describePDFiumError(error))
                } catch {
                    throw GlmOCRError.pdfRenderingFailed(
                        "Unable to render PDF page \(pageIndex + 1): \(error.localizedDescription)"
                    )
                }
            }
            return renderedPages
        }

        private func renderSinglePDFiumPage(data: Data, pageIndex: Int) throws -> CGImage {
            let document: PDFiumDocument
            do {
                document = try PDFiumDocument(data: data)
            } catch let error as PDFiumError {
                throw GlmOCRError.pdfRenderingFailed(describePDFiumError(error))
            } catch {
                throw GlmOCRError.pdfRenderingFailed(
                    "Unable to initialize PDFium document: \(error.localizedDescription)")
            }

            let page: PDFiumPage
            do {
                page = try document.page(at: pageIndex)
            } catch let error as PDFiumError {
                throw GlmOCRError.pdfRenderingFailed(describePDFiumError(error))
            } catch {
                throw GlmOCRError.pdfRenderingFailed(
                    "Unable to load PDF page \(pageIndex + 1): \(error.localizedDescription)")
            }

            do {
                let rasterizer = PDFiumPageRasterizer()
                return try rasterizer.render(
                    page: page,
                    dpi: pdfDPI,
                    maxRenderedLongSide: maxRenderedLongSide
                )
            } catch let error as PDFiumError {
                throw GlmOCRError.pdfRenderingFailed(describePDFiumError(error))
            } catch {
                throw GlmOCRError.pdfRenderingFailed(
                    "Unable to render PDF page \(pageIndex + 1): \(error.localizedDescription)")
            }
        }

        private func describePDFiumError(_ error: PDFiumError) -> String {
            switch error {
            case .libraryLoadFailed(let message):
                return "PDFium runtime initialization failed: \(message)"
            case .documentLoadFailed(let code):
                return "PDFium failed to load PDF document (error code: \(code))"
            case .invalidPageIndex(let index):
                return "PDFium page index out of range: \(index)"
            case .pageLoadFailed(let index, let code):
                return "PDFium failed to load page \(index + 1) (error code: \(code))"
            case .bitmapCreateFailed(let width, let height):
                return "PDFium failed to allocate bitmap \(width)x\(height)"
            case .bitmapBufferUnavailable:
                return "PDFium bitmap buffer unavailable"
            case .cgImageCreateFailed(let width, let height):
                return "Unable to create CGImage from PDFium bitmap \(width)x\(height)"
            }
        }
    #endif

    #if !os(macOS)
        private func renderSequentialCGPDFPages(
            document: CGPDFDocument,
            requestedPageCount: Int
        ) throws -> [CGImage] {
            var renderedPages: [CGImage] = []
            renderedPages.reserveCapacity(requestedPageCount)

            for pageNumber in 1...requestedPageCount {
                guard let page = document.page(at: pageNumber) else {
                    throw GlmOCRError.invalidConfiguration("Unable to access PDF page \(pageNumber)")
                }

                renderedPages.append(try render(page: page))
            }
            return renderedPages
        }

        private func renderSingleCGPDFPage(data: Data, pageNumber: Int) throws -> CGImage {
            guard let provider = CGDataProvider(data: data as CFData),
                let document = CGPDFDocument(provider)
            else {
                throw GlmOCRError.invalidConfiguration("Unable to decode pdfData input")
            }
            guard let page = document.page(at: pageNumber) else {
                throw GlmOCRError.invalidConfiguration("Unable to access PDF page \(pageNumber)")
            }
            return try render(page: page)
        }
    #endif

    private func render(page: CGPDFPage) throws -> CGImage {
        let mediaBox = page.getBoxRect(.mediaBox)
        let widthPoints = max(1.0, Double(mediaBox.width))
        let heightPoints = max(1.0, Double(mediaBox.height))

        let baseScale = pdfDPI / 72.0
        let longSidePoints = max(widthPoints, heightPoints)
        let capScale = maxRenderedLongSide / max(1.0, longSidePoints)
        let scale = CGFloat(min(baseScale, capScale))

        let widthPixels = max(1, Int((CGFloat(widthPoints) * scale).rounded(.toNearestOrAwayFromZero)))
        let heightPixels = max(1, Int((CGFloat(heightPoints) * scale).rounded(.toNearestOrAwayFromZero)))

        guard
            let context = CGContext(
                data: nil,
                width: widthPixels,
                height: heightPixels,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        else {
            throw GlmOCRError.invalidConfiguration("Unable to allocate PDF rendering context")
        }

        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: widthPixels, height: heightPixels))

        let targetRect = CGRect(x: 0, y: 0, width: widthPixels, height: heightPixels)
        let transform = page.getDrawingTransform(
            .mediaBox,
            rect: targetRect,
            rotate: 0,
            preserveAspectRatio: true
        )
        context.concatenate(transform)
        context.drawPDFPage(page)

        guard let rendered = context.makeImage() else {
            throw GlmOCRError.invalidConfiguration("Unable to render PDF page")
        }

        return rendered
    }
}
