import CPDFium
import Foundation

@_cdecl("glmocr_pdfium_destroy_at_exit")
private func glmocrPDFiumDestroyAtExit() {
    FPDF_DestroyLibrary()
}

public enum PDFiumRuntime {
    private static let initialized: Bool = {
        FPDF_InitLibrary()
        atexit(glmocrPDFiumDestroyAtExit)
        return true
    }()

    public static func ensureInitialized() {
        _ = initialized
    }
}
