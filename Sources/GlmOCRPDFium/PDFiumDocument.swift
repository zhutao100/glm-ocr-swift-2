import CPDFium
import Foundation

public enum PDFiumError: Error, Sendable, Equatable {
    case libraryLoadFailed(String)
    case documentLoadFailed(Int)
    case invalidPageIndex(Int)
    case pageLoadFailed(Int, Int)
    case bitmapCreateFailed(Int, Int)
    case bitmapBufferUnavailable
    case cgImageCreateFailed(Int, Int)
}

public final class PDFiumDocument: @unchecked Sendable {
    private let handle: FPDF_DOCUMENT
    private let memory: UnsafeMutableRawPointer
    private let memoryCount: Int

    public init(data: Data) throws {
        PDFiumRuntime.ensureInitialized()

        self.memoryCount = max(1, data.count)
        self.memory = UnsafeMutableRawPointer.allocate(
            byteCount: self.memoryCount, alignment: MemoryLayout<UInt8>.alignment)

        if data.isEmpty {
            self.memory.storeBytes(of: 0 as UInt8, as: UInt8.self)
        } else {
            data.copyBytes(to: self.memory.assumingMemoryBound(to: UInt8.self), count: data.count)
        }

        let loaded = FPDF_LoadMemDocument64(self.memory, self.memoryCount, nil)
        guard let loaded else {
            let code = Int(FPDF_GetLastError())
            self.memory.deallocate()
            throw PDFiumError.documentLoadFailed(code)
        }

        self.handle = loaded
    }

    deinit {
        FPDF_CloseDocument(handle)
        memory.deallocate()
    }

    public var pageCount: Int {
        Int(FPDF_GetPageCount(handle))
    }

    public func page(at index: Int) throws -> PDFiumPage {
        guard index >= 0, index < pageCount else {
            throw PDFiumError.invalidPageIndex(index)
        }

        guard let pageHandle = FPDF_LoadPage(handle, Int32(index)) else {
            throw PDFiumError.pageLoadFailed(index, Int(FPDF_GetLastError()))
        }

        return PDFiumPage(handle: pageHandle)
    }
}

public final class PDFiumPage: @unchecked Sendable {
    let handle: FPDF_PAGE

    fileprivate init(handle: FPDF_PAGE) {
        self.handle = handle
    }

    deinit {
        FPDF_ClosePage(handle)
    }

    public var widthPoints: Double {
        Double(FPDF_GetPageWidthF(handle))
    }

    public var heightPoints: Double {
        Double(FPDF_GetPageHeightF(handle))
    }
}
