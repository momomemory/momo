use momo::models::DocumentType;
use momo::processing::extractors::csv::CsvExtractor;

mod common;
use common::ensure_fixtures;

#[test]
fn test_csv_with_headers() {
    ensure_fixtures();

    let csv_content = b"Name,Age,City\nAlice,30,New York\nBob,25,Los Angeles";
    let result = CsvExtractor::extract(csv_content);

    assert!(result.is_ok());
    let extracted = result.unwrap();

    assert_eq!(extracted.doc_type, DocumentType::Csv);
    assert!(extracted.text.contains("| Name | Age | City |"));
    assert!(extracted.text.contains("| Alice | 30 | New York |"));
    assert!(extracted.text.contains("| Bob | 25 | Los Angeles |"));
    assert!(extracted.title.is_none());
}

#[test]
fn test_csv_auto_delimiter_comma() {
    let csv_content = b"Name,Age,City\nAlice,30,New York";
    let result = CsvExtractor::extract(csv_content);

    assert!(result.is_ok());
    let extracted = result.unwrap();
    assert!(extracted.text.contains("| Name | Age | City |"));
}

#[test]
fn test_csv_auto_delimiter_semicolon() {
    let csv_content = b"Name;Age;City\nAlice;30;New York\nBob;25;Los Angeles";
    let result = CsvExtractor::extract(csv_content);

    assert!(result.is_ok());
    let extracted = result.unwrap();
    assert!(extracted.text.contains("| Name | Age | City |"));
    assert!(extracted.text.contains("| Alice | 30 | New York |"));
}

#[test]
fn test_csv_auto_delimiter_tab() {
    let csv_content = b"Name\tAge\tCity\nAlice\t30\tNew York\nBob\t25\tLos Angeles";
    let result = CsvExtractor::extract(csv_content);

    assert!(result.is_ok());
    let extracted = result.unwrap();
    assert!(extracted.text.contains("| Name | Age | City |"));
    assert!(extracted.text.contains("| Alice | 30 | New York |"));
}

#[test]
fn test_csv_empty() {
    let csv_content = b"";
    let result = CsvExtractor::extract(csv_content);

    // Empty CSV should return an error
    assert!(result.is_err());
}

#[test]
fn test_csv_empty_with_headers_only() {
    let csv_content = b"Name,Age,City\n";
    let result = CsvExtractor::extract(csv_content);

    assert!(result.is_ok());
    let extracted = result.unwrap();
    assert!(extracted.text.contains("| Name | Age | City |"));
    assert!(extracted.text.contains("|------|-----|------|"));
}

#[test]
fn test_csv_malformed() {
    // CSV with inconsistent column counts
    let csv_content = b"Name,Age,City\nAlice,30\nBob,25,Los Angeles,Extra";
    let result = CsvExtractor::extract(csv_content);

    // Should still parse but may have empty cells
    assert!(result.is_ok());
    let extracted = result.unwrap();
    assert!(extracted.text.contains("| Name | Age | City |"));
}

#[test]
fn test_csv_bom_stripping() {
    // UTF-8 BOM followed by CSV content
    let mut csv_content = vec![0xEF, 0xBB, 0xBF];
    csv_content.extend_from_slice(b"Name,Age\nAlice,30");

    let result = CsvExtractor::extract(&csv_content);

    assert!(result.is_ok());
    let extracted = result.unwrap();
    // Should not contain BOM in output
    assert!(!extracted.text.contains('\u{FEFF}'));
    assert!(extracted.text.contains("| Name | Age |"));
    assert!(extracted.text.contains("| Alice | 30 |"));
}

#[test]
fn test_csv_with_quoted_fields() {
    let csv_content =
        b"Name,Description\nAlice,\"A software engineer\"\nBob,\"Works in \"\"design\"\"\"";
    let result = CsvExtractor::extract(csv_content);

    assert!(result.is_ok());
    let extracted = result.unwrap();
    assert!(extracted.text.contains("| Name | Description |"));
    assert!(extracted.text.contains("| Alice | A software engineer |"));
    // Should handle escaped quotes
    assert!(extracted.text.contains("Works in"));
}

#[test]
fn test_csv_word_count() {
    let csv_content = b"Name,Age\nAlice,30\nBob,25";
    let result = CsvExtractor::extract(csv_content);

    assert!(result.is_ok());
    let extracted = result.unwrap();
    // Word count should be positive
    assert!(extracted.word_count > 0);
}

#[test]
fn test_csv_from_fixture() {
    ensure_fixtures();
    let bytes = common::load_fixture("sample.csv");
    let result = CsvExtractor::extract(&bytes);

    assert!(result.is_ok());
    let extracted = result.unwrap();
    assert_eq!(extracted.doc_type, DocumentType::Csv);
    assert!(extracted
        .text
        .contains("| Name | Age | City | Occupation |"));
    assert!(extracted
        .text
        .contains("| Alice | 30 | New York | Engineer |"));
    assert!(extracted
        .text
        .contains("| Diana | 28 | Seattle | Developer |"));
}
