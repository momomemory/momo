use super::ExtractedContent;
use crate::error::{MomoError, Result};
use crate::models::DocumentType;

/// Extractor for CSV files that converts them to markdown table format
pub struct CsvExtractor;

impl CsvExtractor {
    /// Extract content from CSV bytes and convert to markdown table
    pub fn extract(bytes: &[u8]) -> Result<ExtractedContent> {
        // Strip BOM if present
        let bytes = strip_bom(bytes);

        if bytes.is_empty() {
            return Err(MomoError::Processing("Empty CSV file".to_string()));
        }

        // Auto-detect delimiter by analyzing the content
        let delimiter = detect_delimiter(bytes);

        // Parse CSV
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .flexible(true)
            .from_reader(bytes);

        // Collect all records
        let headers = reader
            .headers()
            .map_err(|e| MomoError::Processing(format!("Failed to read CSV headers: {}", e)))?
            .iter()
            .map(|h| h.to_string())
            .collect::<Vec<_>>();

        if headers.is_empty() {
            return Err(MomoError::Processing("CSV has no headers".to_string()));
        }

        let mut records: Vec<Vec<String>> = Vec::new();
        for result in reader.records() {
            let record = result
                .map_err(|e| MomoError::Processing(format!("Failed to read CSV record: {}", e)))?;
            records.push(record.iter().map(|f| f.to_string()).collect());
        }

        // Convert to markdown table
        let markdown = convert_to_markdown(&headers, &records);
        let word_count = count_words(&markdown);

        Ok(ExtractedContent {
            text: markdown,
            title: None,
            doc_type: DocumentType::Csv,
            url: None,
            word_count,
            source_path: None,
        })
    }
}

/// Strip UTF-8 BOM if present
fn strip_bom(bytes: &[u8]) -> &[u8] {
    if bytes.len() >= 3 && bytes[0..3] == [0xEF, 0xBB, 0xBF] {
        &bytes[3..]
    } else {
        bytes
    }
}

/// Auto-detect delimiter by trying common delimiters and picking the one
/// that produces the most consistent number of columns
fn detect_delimiter(bytes: &[u8]) -> u8 {
    let candidates = [b',', b';', b'\t'];
    let mut best_delimiter = b',';
    let mut best_score = 0;

    for &delimiter in &candidates {
        let score = evaluate_delimiter(bytes, delimiter);
        if score > best_score {
            best_score = score;
            best_delimiter = delimiter;
        }
    }

    best_delimiter
}

/// Evaluate how well a delimiter works for the given CSV content
/// Returns a score - higher is better
fn evaluate_delimiter(bytes: &[u8], delimiter: u8) -> usize {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .from_reader(bytes);

    let mut column_counts: Vec<usize> = Vec::new();

    // Check headers
    if let Ok(headers) = reader.headers() {
        column_counts.push(headers.len());
    }

    // Check first few records
    for (i, result) in reader.records().enumerate() {
        if i >= 5 {
            break;
        }
        if let Ok(record) = result {
            column_counts.push(record.len());
        }
    }

    if column_counts.is_empty() {
        return 0;
    }

    // Score based on consistency and number of columns
    let first_count = column_counts[0];
    let consistent = column_counts.iter().all(|&c| c == first_count);
    let has_multiple_columns = first_count > 1;

    if consistent && has_multiple_columns {
        // Bonus for consistency and having multiple columns
        first_count * 10
    } else if has_multiple_columns {
        first_count
    } else {
        0
    }
}

/// Convert CSV headers and records to markdown table format
fn convert_to_markdown(headers: &[String], records: &[Vec<String>]) -> String {
    let mut result = String::new();

    // Header row
    result.push_str("| ");
    result.push_str(&headers.join(" | "));
    result.push_str(" |\n");

    // Separator row - use minimum 3 dashes per column (markdown standard)
    result.push_str("|");
    for header in headers {
        let width = header.len().max(3);
        result.push_str(&"-".repeat(width + 2));
        result.push('|');
    }
    result.push('\n');

    // Data rows
    for record in records {
        result.push_str("| ");
        let row: Vec<String> = record.iter().map(|field| field.to_string()).collect();
        result.push_str(&row.join(" | "));
        result.push_str(" |\n");
    }

    result
}

/// Count words in text
fn count_words(text: &str) -> i32 {
    text.split_whitespace().count() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_bom() {
        let with_bom = vec![0xEF, 0xBB, 0xBF, b'h', b'e', b'l', b'l', b'o'];
        let without_bom = super::strip_bom(&with_bom);
        assert_eq!(without_bom, b"hello");

        let no_bom = b"hello";
        assert_eq!(super::strip_bom(no_bom), b"hello");
    }

    #[test]
    fn test_detect_delimiter_comma() {
        let csv = b"Name,Age,City\nAlice,30,NYC";
        assert_eq!(super::detect_delimiter(csv), b',');
    }

    #[test]
    fn test_detect_delimiter_semicolon() {
        let csv = b"Name;Age;City\nAlice;30;NYC";
        assert_eq!(super::detect_delimiter(csv), b';');
    }

    #[test]
    fn test_detect_delimiter_tab() {
        let csv = b"Name\tAge\tCity\nAlice\t30\tNYC";
        assert_eq!(super::detect_delimiter(csv), b'\t');
    }

    #[test]
    fn test_convert_to_markdown() {
        let headers = vec!["Name".to_string(), "Age".to_string()];
        let records = vec![vec!["Alice".to_string(), "30".to_string()]];
        let markdown = super::convert_to_markdown(&headers, &records);

        assert!(markdown.contains("| Name | Age |"));
        assert!(markdown.contains("|------|-----|"));
        assert!(markdown.contains("| Alice | 30 |"));
    }
}
