import Foundation

internal struct PipelineFormatter: Sendable {
    private static let imageLabels: Set<String> = [
        "chart", "image",
    ]

    private static let tableLabels: Set<String> = [
        "table"
    ]

    private static let formulaLabels: Set<String> = [
        "display_formula", "inline_formula", "formula",
    ]

    private static let textLabels: Set<String> = [
        "abstract", "algorithm", "content", "doc_title", "figure_title",
        "paragraph_title", "reference_content", "text", "vertical_text",
        "vision_footnote", "seal", "formula_number",
    ]

    internal init() {}

    internal func formatNoLayout(
        contents: [String]
    ) -> (pages: [OCRPageResult], markdown: String) {
        let pageResults: [OCRPageResult] = contents.map { content in
            let cleaned = cleanContent(content)
            return OCRPageResult(regions: [
                OCRRegion(
                    index: 0,
                    label: "text",
                    content: cleaned,
                    bbox2D: nil
                )
            ])
        }

        let markdown =
            contents
            .map(cleanContent)
            .joined(separator: "\n\n---\n\n")

        return (pageResults, markdown)
    }

    internal func formatLayout(
        pageRegions: [[PipelineRegionRecord]]
    ) -> (pages: [OCRPageResult], markdown: String) {
        var pages: [OCRPageResult] = []
        pages.reserveCapacity(pageRegions.count)

        var markdownPages: [String] = []
        markdownPages.reserveCapacity(pageRegions.count)

        for (pageIndex, regions) in pageRegions.enumerated() {
            let normalized = normalizePageRegions(regions)
            let mergedFormulaNumbers = mergeFormulaNumbers(normalized)
            let mergedTextBlocks = mergeTextBlocks(mergedFormulaNumbers)
            let bulletFormatted = formatBulletPoints(mergedTextBlocks)

            let reindexed = bulletFormatted.enumerated().map { newIndex, region in
                var updated = region
                updated.index = newIndex
                return updated
            }

            let pageOCRRegions = reindexed.map { region in
                OCRRegion(
                    index: region.index,
                    label: region.label,
                    content: region.content,
                    bbox2D: region.bbox2D
                )
            }

            pages.append(OCRPageResult(regions: pageOCRRegions))
            markdownPages.append(markdownText(for: reindexed, pageIndex: pageIndex))
        }

        return (pages, markdownPages.joined(separator: "\n\n"))
    }

    private func normalizePageRegions(
        _ regions: [PipelineRegionRecord]
    ) -> [FormattedRegion] {
        let sorted = regions.sorted { lhs, rhs in
            lhs.index < rhs.index
        }

        return sorted.enumerated().map { index, region in
            let mappedLabel = mapLabel(region.nativeLabel)
            let formattedContent = formatContent(
                region.content,
                mappedLabel: mappedLabel,
                nativeLabel: region.nativeLabel
            )

            return FormattedRegion(
                index: index,
                nativeLabel: region.nativeLabel,
                label: mappedLabel,
                bbox2D: region.bbox2D,
                content: formattedContent
            )
        }
    }

    private func markdownText(
        for regions: [FormattedRegion],
        pageIndex: Int
    ) -> String {
        var blocks: [String] = []
        blocks.reserveCapacity(regions.count)

        var imageIndex = 0
        for region in regions {
            if region.label == "image", let bbox = region.bbox2D {
                blocks.append("![Image \(pageIndex)-\(imageIndex)](page=\(pageIndex),bbox=\(bbox))")
                imageIndex += 1
                continue
            }

            if let content = region.content {
                blocks.append(content)
            }
        }

        return blocks.joined(separator: "\n\n")
    }

    private func mapLabel(_ label: String) -> String {
        if Self.imageLabels.contains(label) {
            return "image"
        }
        if Self.tableLabels.contains(label) {
            return "table"
        }
        if Self.formulaLabels.contains(label) {
            return "formula"
        }
        if Self.textLabels.contains(label) {
            return "text"
        }
        return label
    }

    private func formatContent(
        _ content: String?,
        mappedLabel: String,
        nativeLabel: String
    ) -> String? {
        guard let content else {
            return nil
        }

        var formatted = cleanContent(content)

        if nativeLabel == "doc_title" {
            formatted = "# " + stripLeadingMarkdownDecorations(formatted)
        } else if nativeLabel == "paragraph_title" {
            formatted = "## " + stripLeadingMarkdownDecorations(formatted)
        }

        if mappedLabel == "formula" {
            formatted = normalizeFormulaBlock(formatted)
        }

        if mappedLabel == "text" {
            formatted = normalizeTextBlock(formatted)
        }

        return formatted
    }

    private func cleanContent(_ content: String) -> String {
        var value = content.trimmingCharacters(in: .whitespacesAndNewlines)

        while value.hasPrefix("\\t") {
            value.removeFirst(2)
        }
        while value.hasSuffix("\\t") {
            value.removeLast(2)
        }

        value = value.replacingOccurrences(
            of: "(\\.)\\1{2,}",
            with: "...",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: "(·)\\1{2,}",
            with: "···",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: "(_)\\1{2,}",
            with: "___",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: "(\\\\_)\\1{2,}",
            with: "\\\\_\\\\_\\\\_",
            options: .regularExpression
        )

        return value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func stripLeadingMarkdownDecorations(_ value: String) -> String {
        var output = value
        while output.hasPrefix("#") {
            output.removeFirst()
        }

        output = output.trimmingCharacters(in: .whitespacesAndNewlines)

        if output.hasPrefix("- ") || output.hasPrefix("* ") {
            output.removeFirst(2)
        }

        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func normalizeFormulaBlock(_ value: String) -> String {
        var formula = value.trimmingCharacters(in: .whitespacesAndNewlines)

        if formula.hasPrefix("$$"), formula.hasSuffix("$$"), formula.count >= 4 {
            formula = String(formula.dropFirst(2).dropLast(2))
        } else if formula.hasPrefix("\\["), formula.hasSuffix("\\]"), formula.count >= 4 {
            formula = String(formula.dropFirst(2).dropLast(2))
        } else if formula.hasPrefix("\\("), formula.hasSuffix("\\)"), formula.count >= 4 {
            formula = String(formula.dropFirst(2).dropLast(2))
        }

        formula = formula.trimmingCharacters(in: .whitespacesAndNewlines)
        return "$$\n\(formula)\n$$"
    }

    private func normalizeTextBlock(_ value: String) -> String {
        var text = value

        if text.hasPrefix("·") || text.hasPrefix("•") {
            text = "- " + text.dropFirst().trimmingCharacters(in: .whitespaces)
        }

        if text.hasPrefix("* ") {
            text = "- " + text.dropFirst(2).trimmingCharacters(in: .whitespaces)
        }

        text = normalizeBulletNumbering(text)
        return doubleSingleNewlines(text)
    }

    private func normalizeBulletNumbering(_ value: String) -> String {
        let fullRange = NSRange(value.startIndex..<value.endIndex, in: value)

        if let enclosed = firstMatch(
            pattern: #"^(\(|（)(\d+|[A-Za-z])(\)|）)(.*)$"#,
            in: value,
            range: fullRange
        ) {
            let symbol = enclosed[2]
            let rest = enclosed[4].trimmingCharacters(in: .whitespaces)
            return "(\(symbol))" + (rest.isEmpty ? "" : " \(rest)")
        }

        if let suffix = firstMatch(
            pattern: #"^(\d+|[A-Za-z])(\.|\)|）)(.*)$"#,
            in: value,
            range: fullRange
        ) {
            let symbol = suffix[1]
            let separator = suffix[2] == "）" ? ")" : suffix[2]
            let rest = suffix[3].trimmingCharacters(in: .whitespaces)
            return "\(symbol)\(separator)" + (rest.isEmpty ? "" : " \(rest)")
        }

        return value
    }

    private func firstMatch(
        pattern: String,
        in text: String,
        range: NSRange
    ) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
            let match = regex.firstMatch(in: text, range: range)
        else {
            return nil
        }

        var groups: [String] = []
        groups.reserveCapacity(match.numberOfRanges)

        for index in 0..<match.numberOfRanges {
            let matchRange = match.range(at: index)
            if matchRange.location == NSNotFound {
                groups.append("")
                continue
            }

            guard let range = Range(matchRange, in: text) else {
                groups.append("")
                continue
            }

            groups.append(String(text[range]))
        }

        return groups
    }

    private func doubleSingleNewlines(_ value: String) -> String {
        let lines = value.components(separatedBy: "\n")
        if lines.count <= 1 {
            return value
        }

        var output: [String] = []
        output.reserveCapacity(lines.count * 2)

        for (index, line) in lines.enumerated() {
            output.append(line)

            guard index < lines.count - 1 else {
                continue
            }

            let currentIsEmpty = line.isEmpty
            let nextIsEmpty = lines[index + 1].isEmpty
            if !currentIsEmpty && !nextIsEmpty {
                output.append("")
            }
        }

        return output.joined(separator: "\n")
    }

    private func mergeFormulaNumbers(
        _ regions: [FormattedRegion]
    ) -> [FormattedRegion] {
        guard !regions.isEmpty else {
            return regions
        }

        var merged: [FormattedRegion] = []
        merged.reserveCapacity(regions.count)

        var index = 0
        while index < regions.count {
            let region = regions[index]

            if region.nativeLabel == "formula_number" {
                if index + 1 < regions.count, regions[index + 1].label == "formula" {
                    var nextFormula = regions[index + 1]
                    if let formula = nextFormula.content, let number = region.content {
                        nextFormula.content = mergeFormulaTag(
                            formulaContent: formula,
                            numberContent: number
                        )
                    }
                    merged.append(nextFormula)
                    index += 2
                    continue
                }

                index += 1
                continue
            }

            if region.label == "formula",
                index + 1 < regions.count,
                regions[index + 1].nativeLabel == "formula_number"
            {
                var updatedFormula = region
                if let formula = region.content, let number = regions[index + 1].content {
                    updatedFormula.content = mergeFormulaTag(
                        formulaContent: formula,
                        numberContent: number
                    )
                }
                merged.append(updatedFormula)
                index += 2
                continue
            }

            merged.append(region)
            index += 1
        }

        return merged
    }

    private func mergeFormulaTag(
        formulaContent: String,
        numberContent: String
    ) -> String {
        let cleanedNumber = cleanFormulaNumber(numberContent)
        guard formulaContent.hasSuffix("\n$$"), formulaContent.count >= 3 else {
            return formulaContent
        }

        let trimmed = formulaContent.dropLast(3)
        return "\(trimmed) \\tag{\(cleanedNumber)}\n$$"
    }

    private func cleanFormulaNumber(_ value: String) -> String {
        var cleaned = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.hasPrefix("("), cleaned.hasSuffix(")"), cleaned.count >= 2 {
            cleaned = String(cleaned.dropFirst().dropLast())
        } else if cleaned.hasPrefix("（"), cleaned.hasSuffix("）"), cleaned.count >= 2 {
            cleaned = String(cleaned.dropFirst().dropLast())
        }
        return cleaned
    }

    private func mergeTextBlocks(
        _ regions: [FormattedRegion]
    ) -> [FormattedRegion] {
        guard !regions.isEmpty else {
            return regions
        }

        var output: [FormattedRegion] = []
        output.reserveCapacity(regions.count)

        var index = 0
        while index < regions.count {
            let region = regions[index]

            guard region.label == "text",
                let content = region.content,
                content.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("-"),
                index + 1 < regions.count,
                regions[index + 1].label == "text",
                let nextContent = regions[index + 1].content,
                let first = nextContent.trimmingCharacters(in: .whitespacesAndNewlines).first,
                first.isLowercase
            else {
                output.append(region)
                index += 1
                continue
            }

            var merged = region
            let left = content.trimmingCharacters(in: .whitespacesAndNewlines)
            let right = nextContent.trimmingCharacters(in: .whitespacesAndNewlines)
            merged.content = String(left.dropLast()) + right
            output.append(merged)
            index += 2
        }

        return output
    }

    private func formatBulletPoints(
        _ regions: [FormattedRegion],
        leftAlignThreshold: Int = 10
    ) -> [FormattedRegion] {
        guard regions.count >= 3 else {
            return regions
        }

        var output = regions

        for index in 1..<(output.count - 1) {
            guard output[index].nativeLabel == "text",
                output[index - 1].nativeLabel == "text",
                output[index + 1].nativeLabel == "text",
                let current = output[index].content,
                let previous = output[index - 1].content,
                let next = output[index + 1].content,
                !current.hasPrefix("- "),
                previous.hasPrefix("- "),
                next.hasPrefix("- "),
                let currentLeft = output[index].bbox2D?.first,
                let previousLeft = output[index - 1].bbox2D?.first,
                let nextLeft = output[index + 1].bbox2D?.first
            else {
                continue
            }

            let alignedPrev = abs(currentLeft - previousLeft) <= leftAlignThreshold
            let alignedNext = abs(currentLeft - nextLeft) <= leftAlignThreshold

            if alignedPrev && alignedNext {
                output[index].content = "- " + current
            }
        }

        return output
    }
}

private struct FormattedRegion: Sendable, Equatable {
    internal var index: Int
    internal var nativeLabel: String
    internal var label: String
    internal var bbox2D: [Int]?
    internal var content: String?

    internal init(
        index: Int,
        nativeLabel: String,
        label: String,
        bbox2D: [Int]?,
        content: String?
    ) {
        self.index = index
        self.nativeLabel = nativeLabel
        self.label = label
        self.bbox2D = bbox2D
        self.content = content
    }
}
