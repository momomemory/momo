use std::io::Cursor;

mod common;
use common::ensure_fixtures;

use momo::models::DocumentType;
use momo::processing::extractors::docx::DocxExtractor;

fn create_test_docx<F>(builder_fn: F) -> Vec<u8>
where
    F: FnOnce(docx_rs::Docx) -> docx_rs::Docx,
{
    use docx_rs::*;

    let docx = builder_fn(Docx::new());
    let mut buffer = Cursor::new(Vec::new());
    docx.build().pack(&mut buffer).expect("Failed to pack DOCX");
    buffer.into_inner()
}

#[test]
fn test_docx_basic_text() {
    use docx_rs::*;

    let bytes = create_test_docx(|docx| {
        docx.add_paragraph(Paragraph::new().add_run(Run::new().add_text("Hello World")))
            .add_paragraph(
                Paragraph::new().add_run(Run::new().add_text("This is a test document.")),
            )
    });

    let result = DocxExtractor::extract(&bytes);
    assert!(result.is_ok(), "Should successfully extract DOCX content");

    let extracted = result.unwrap();
    assert_eq!(extracted.doc_type, DocumentType::Docx);
    assert!(
        extracted.text.contains("Hello World"),
        "Should contain first paragraph"
    );
    assert!(
        extracted.text.contains("This is a test document."),
        "Should contain second paragraph"
    );
    assert!(extracted.word_count > 0, "Should have word count");
}

#[test]
fn test_docx_headings() {
    use docx_rs::*;

    let bytes = create_test_docx(|docx| {
        docx.add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("Main Title"))
                .style("Heading1"),
        )
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("Section One"))
                .style("Heading2"),
        )
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("Subsection"))
                .style("Heading3"),
        )
        .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Regular paragraph text.")))
    });

    let result = DocxExtractor::extract(&bytes);
    assert!(result.is_ok());

    let extracted = result.unwrap();
    assert!(
        extracted.text.contains("# Main Title"),
        "H1 should become # Heading"
    );
    assert!(
        extracted.text.contains("## Section One"),
        "H2 should become ## Heading"
    );
    assert!(
        extracted.text.contains("### Subsection"),
        "H3 should become ### Heading"
    );
    assert!(
        extracted.text.contains("Regular paragraph text."),
        "Regular text should remain unchanged"
    );
}

#[test]
fn test_docx_bullet_lists() {
    use docx_rs::*;

    // Use even num_id (2) for bullet lists
    let bytes = create_test_docx(|docx| {
        docx.add_abstract_numbering(AbstractNumbering::new(2).add_level(Level::new(
            0,
            Start::new(1),
            NumberFormat::new("bullet"),
            LevelText::new("\u{2022}"),
            LevelJc::new("left"),
        )))
        .add_numbering(Numbering::new(2, 2))
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("First bullet"))
                .numbering(NumberingId::new(2), IndentLevel::new(0)),
        )
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("Second bullet"))
                .numbering(NumberingId::new(2), IndentLevel::new(0)),
        )
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("Third bullet"))
                .numbering(NumberingId::new(2), IndentLevel::new(0)),
        )
    });

    let result = DocxExtractor::extract(&bytes);
    assert!(result.is_ok());

    let extracted = result.unwrap();
    assert!(
        extracted.text.contains("- First bullet"),
        "Bullet items should start with -"
    );
    assert!(
        extracted.text.contains("- Second bullet"),
        "Bullet items should start with -"
    );
    assert!(
        extracted.text.contains("- Third bullet"),
        "Bullet items should start with -"
    );
}

#[test]
fn test_docx_numbered_lists() {
    use docx_rs::*;

    // Use odd num_id (1) for numbered lists
    let bytes = create_test_docx(|docx| {
        docx.add_abstract_numbering(AbstractNumbering::new(1).add_level(Level::new(
            0,
            Start::new(1),
            NumberFormat::new("decimal"),
            LevelText::new("%1."),
            LevelJc::new("left"),
        )))
        .add_numbering(Numbering::new(1, 1))
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("First item"))
                .numbering(NumberingId::new(1), IndentLevel::new(0)),
        )
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("Second item"))
                .numbering(NumberingId::new(1), IndentLevel::new(0)),
        )
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("Third item"))
                .numbering(NumberingId::new(1), IndentLevel::new(0)),
        )
    });

    let result = DocxExtractor::extract(&bytes);
    assert!(result.is_ok());

    let extracted = result.unwrap();
    assert!(
        extracted.text.contains("1. First item"),
        "Numbered items should have numbering prefix"
    );
    assert!(
        extracted.text.contains("1. Second item"),
        "Numbered items should have numbering prefix"
    );
    assert!(
        extracted.text.contains("1. Third item"),
        "Numbered items should have numbering prefix"
    );
}

#[test]
fn test_docx_tables() {
    use docx_rs::*;

    let table = Table::new(vec![
        TableRow::new(vec![
            TableCell::new().add_paragraph(Paragraph::new().add_run(Run::new().add_text("Name"))),
            TableCell::new().add_paragraph(Paragraph::new().add_run(Run::new().add_text("Age"))),
        ]),
        TableRow::new(vec![
            TableCell::new().add_paragraph(Paragraph::new().add_run(Run::new().add_text("Alice"))),
            TableCell::new().add_paragraph(Paragraph::new().add_run(Run::new().add_text("30"))),
        ]),
        TableRow::new(vec![
            TableCell::new().add_paragraph(Paragraph::new().add_run(Run::new().add_text("Bob"))),
            TableCell::new().add_paragraph(Paragraph::new().add_run(Run::new().add_text("25"))),
        ]),
    ]);

    let bytes = create_test_docx(|docx| docx.add_table(table));

    let result = DocxExtractor::extract(&bytes);
    assert!(result.is_ok());

    let extracted = result.unwrap();
    assert!(
        extracted.text.contains("| Name | Age |"),
        "Table header should be pipe-separated"
    );
    assert!(
        extracted.text.contains("|------|------|"),
        "Table separator should be present"
    );
    assert!(
        extracted.text.contains("| Alice | 30 |"),
        "Table rows should be pipe-separated"
    );
    assert!(
        extracted.text.contains("| Bob | 25 |"),
        "Table rows should be pipe-separated"
    );
}

#[test]
fn test_docx_empty() {
    let bytes = create_test_docx(|docx| docx);

    let result = DocxExtractor::extract(&bytes);
    assert!(result.is_ok(), "Should handle empty document");

    let extracted = result.unwrap();
    assert_eq!(
        extracted.word_count, 0,
        "Empty document should have 0 words"
    );
    assert!(
        extracted.text.is_empty() || extracted.text.trim().is_empty(),
        "Empty document should have empty text"
    );
}

#[test]
fn test_docx_corrupt() {
    let corrupt_bytes = vec![0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE];

    let result = DocxExtractor::extract(&corrupt_bytes);
    assert!(result.is_err(), "Should fail on corrupt DOCX data");

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("DOCX") || err_msg.contains("parse"),
        "Error should mention DOCX parsing"
    );
}

#[test]
fn test_docx_with_fixture() {
    ensure_fixtures();
    let bytes = common::load_fixture("sample.docx");

    let result = DocxExtractor::extract(&bytes);
    assert!(result.is_ok(), "Should extract from fixture DOCX");

    let extracted = result.unwrap();
    assert!(
        extracted.text.contains("Hello World"),
        "Should contain fixture content"
    );
    assert_eq!(extracted.doc_type, DocumentType::Docx);
}
