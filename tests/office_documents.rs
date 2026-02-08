//! Integration tests for Office document processing pipeline

mod common;

use momo::models::DocumentType;
use momo::processing::extractors::csv::CsvExtractor;
use momo::processing::extractors::docx::DocxExtractor;
use momo::processing::extractors::pptx::PptxExtractor;
use momo::processing::extractors::xlsx::XlsxExtractor;

use common::{ensure_fixtures, load_fixture};

/// Test that CSV files are correctly processed through the extraction pipeline
#[test]
fn test_integration_csv_pipeline() {
    ensure_fixtures();
    let bytes = load_fixture("sample.csv");

    let result = CsvExtractor::extract(&bytes);

    assert!(result.is_ok());
    let content = result.unwrap();
    assert_eq!(content.doc_type, DocumentType::Csv);
    assert!(content.text.contains('|')); // Markdown table
    assert!(content.word_count > 0);
}

/// Test that DOCX files are correctly processed
#[test]
fn test_integration_docx_pipeline() {
    ensure_fixtures();
    let bytes = load_fixture("sample.docx");

    let result = DocxExtractor::extract(&bytes);

    assert!(result.is_ok());
    let content = result.unwrap();
    assert_eq!(content.doc_type, DocumentType::Docx);
    assert!(!content.text.is_empty());
}

/// Test that XLSX files are correctly processed
#[test]
fn test_integration_xlsx_pipeline() {
    ensure_fixtures();
    let bytes = load_fixture("sample.xlsx");

    let result = XlsxExtractor::extract(&bytes);

    assert!(result.is_ok());
    let content = result.unwrap();
    assert_eq!(content.doc_type, DocumentType::Xlsx);
    assert!(content.text.contains("## Sheet:")); // Sheet headers
}

/// Test that PPTX files are correctly processed
#[test]
fn test_integration_pptx_pipeline() {
    ensure_fixtures();
    let bytes = load_fixture("sample.pptx");

    let result = PptxExtractor::extract(&bytes);

    assert!(result.is_ok());
    let content = result.unwrap();
    assert_eq!(content.doc_type, DocumentType::Pptx);
    assert!(content.text.contains("## Slide") || content.text.contains("# Slide"));
    // Slide headers
}

/// Test that extractors auto-detect and process file types correctly via the ExtractedContent type
#[test]
fn test_integration_extracted_content_structure() {
    ensure_fixtures();

    // Test CSV extraction produces proper ExtractedContent
    let csv_bytes = load_fixture("sample.csv");
    let csv_result = CsvExtractor::extract(&csv_bytes);
    assert!(csv_result.is_ok());
    let csv_content = csv_result.unwrap();
    assert_eq!(csv_content.doc_type, DocumentType::Csv);
    assert!(csv_content.word_count > 0);
    assert!(csv_content.title.is_none()); // CSV files don't have titles

    // Test DOCX extraction produces proper ExtractedContent
    let docx_bytes = load_fixture("sample.docx");
    let docx_result = DocxExtractor::extract(&docx_bytes);
    assert!(docx_result.is_ok());
    let docx_content = docx_result.unwrap();
    assert_eq!(docx_content.doc_type, DocumentType::Docx);
    assert!(docx_content.word_count > 0);

    // Test XLSX extraction produces proper ExtractedContent
    let xlsx_bytes = load_fixture("sample.xlsx");
    let xlsx_result = XlsxExtractor::extract(&xlsx_bytes);
    assert!(xlsx_result.is_ok());
    let xlsx_content = xlsx_result.unwrap();
    assert_eq!(xlsx_content.doc_type, DocumentType::Xlsx);
    assert!(xlsx_content.word_count > 0);
    assert!(xlsx_content.title.is_none()); // XLSX files don't have titles

    // Test PPTX extraction produces proper ExtractedContent
    let pptx_bytes = load_fixture("sample.pptx");
    let pptx_result = PptxExtractor::extract(&pptx_bytes);
    assert!(pptx_result.is_ok());
    let pptx_content = pptx_result.unwrap();
    assert_eq!(pptx_content.doc_type, DocumentType::Pptx);
    assert!(pptx_content.word_count > 0);
}

/// Test that corrupt files return appropriate errors
#[test]
fn test_integration_corrupt_files() {
    // Random bytes should fail for each format
    let garbage = vec![0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE];

    let csv_result = CsvExtractor::extract(&garbage);
    assert!(csv_result.is_err());

    let docx_result = DocxExtractor::extract(&garbage);
    assert!(docx_result.is_err());

    let xlsx_result = XlsxExtractor::extract(&garbage);
    assert!(xlsx_result.is_err());

    let pptx_result = PptxExtractor::extract(&garbage);
    assert!(pptx_result.is_err());
}

/// Test all formats process without errors
#[test]
fn test_integration_all_formats() {
    ensure_fixtures();

    let formats: Vec<(&str, DocumentType)> = vec![
        ("sample.csv", DocumentType::Csv),
        ("sample.docx", DocumentType::Docx),
        ("sample.xlsx", DocumentType::Xlsx),
        ("sample.pptx", DocumentType::Pptx),
    ];

    for (filename, expected_type) in formats {
        let bytes = load_fixture(filename);

        let result = match expected_type {
            DocumentType::Csv => CsvExtractor::extract(&bytes),
            DocumentType::Docx => DocxExtractor::extract(&bytes),
            DocumentType::Xlsx => XlsxExtractor::extract(&bytes),
            DocumentType::Pptx => PptxExtractor::extract(&bytes),
            _ => panic!("Unexpected document type in test"),
        };

        assert!(
            result.is_ok(),
            "Failed for {}: {:?}",
            filename,
            result.err()
        );
        assert_eq!(
            result.unwrap().doc_type,
            expected_type,
            "Wrong type for {filename}"
        );
    }
}

/// Test that empty files are handled gracefully
#[test]
fn test_integration_empty_files() {
    // Empty CSV should error
    let empty_csv: Vec<u8> = vec![];
    let csv_result = CsvExtractor::extract(&empty_csv);
    assert!(csv_result.is_err(), "Empty CSV should return error");

    // Empty bytes for DOCX/XLSX/PPTX should error
    let empty: Vec<u8> = vec![];
    assert!(DocxExtractor::extract(&empty).is_err());
    assert!(XlsxExtractor::extract(&empty).is_err());
    assert!(PptxExtractor::extract(&empty).is_err());
}

/// Test word count accuracy across different formats
#[test]
fn test_integration_word_counts() {
    ensure_fixtures();

    // All fixtures should have positive word counts
    let csv_content = CsvExtractor::extract(&load_fixture("sample.csv")).unwrap();
    assert!(csv_content.word_count > 0, "CSV should have word count");

    let docx_content = DocxExtractor::extract(&load_fixture("sample.docx")).unwrap();
    assert!(docx_content.word_count > 0, "DOCX should have word count");

    let xlsx_content = XlsxExtractor::extract(&load_fixture("sample.xlsx")).unwrap();
    assert!(xlsx_content.word_count > 0, "XLSX should have word count");

    let pptx_content = PptxExtractor::extract(&load_fixture("sample.pptx")).unwrap();
    assert!(pptx_content.word_count > 0, "PPTX should have word count");
}

/// Test that extracted text contains expected content patterns
#[test]
fn test_integration_content_patterns() {
    ensure_fixtures();

    // CSV should have markdown table format
    let csv = CsvExtractor::extract(&load_fixture("sample.csv")).unwrap();
    assert!(
        csv.text.contains("| Name |"),
        "CSV should have table header"
    );
    assert!(
        csv.text.contains("|------|"),
        "CSV should have table separator"
    );
    assert!(csv.text.contains("| Alice |"), "CSV should have data rows");

    // DOCX should have text content
    let docx = DocxExtractor::extract(&load_fixture("sample.docx")).unwrap();
    assert!(
        docx.text.contains("Hello World"),
        "DOCX should contain fixture text"
    );

    // XLSX should have sheet headers and table
    let xlsx = XlsxExtractor::extract(&load_fixture("sample.xlsx")).unwrap();
    assert!(
        xlsx.text.contains("## Sheet: Sheet1"),
        "XLSX should have sheet header"
    );
    assert!(
        xlsx.text.contains("| Product |"),
        "XLSX should have table header"
    );

    // PPTX should have slide markers
    let pptx = PptxExtractor::extract(&load_fixture("sample.pptx")).unwrap();
    assert!(
        pptx.text.contains("Test Presentation"),
        "PPTX should contain slide content"
    );
}
