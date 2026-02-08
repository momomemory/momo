use std::io::{Cursor, Write};
use zip::write::SimpleFileOptions;

mod common;
use common::ensure_fixtures;

use momo::models::DocumentType;
use momo::processing::extractors::pptx::PptxExtractor;

fn create_test_pptx(slides: Vec<SlideContent>) -> Vec<u8> {
    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);
        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);

        zip.start_file("[Content_Types].xml", options).unwrap();
        zip.write_all(generate_content_types(&slides).as_bytes())
            .unwrap();

        zip.add_directory("_rels", options).unwrap();
        zip.start_file("_rels/.rels", options).unwrap();
        zip.write_all(RELS_XML.as_bytes()).unwrap();

        zip.add_directory("ppt", options).unwrap();
        zip.start_file("ppt/presentation.xml", options).unwrap();
        zip.write_all(generate_presentation_xml(&slides).as_bytes())
            .unwrap();

        zip.add_directory("ppt/_rels", options).unwrap();
        zip.start_file("ppt/_rels/presentation.xml.rels", options)
            .unwrap();
        zip.write_all(generate_presentation_rels(&slides).as_bytes())
            .unwrap();

        zip.add_directory("ppt/slides", options).unwrap();
        for (i, slide) in slides.iter().enumerate() {
            let filename = format!("ppt/slides/slide{}.xml", i + 1);
            zip.start_file(&filename, options).unwrap();
            zip.write_all(generate_slide_xml(&slide.title, &slide.content).as_bytes())
                .unwrap();
        }

        let has_notes = slides.iter().any(|s| s.notes.is_some());
        if has_notes {
            zip.add_directory("ppt/slides/_rels", options).unwrap();
            for (i, slide) in slides.iter().enumerate() {
                if slide.notes.is_some() {
                    let filename = format!("ppt/slides/_rels/slide{}.xml.rels", i + 1);
                    zip.start_file(&filename, options).unwrap();
                    zip.write_all(generate_slide_rels(i + 1).as_bytes())
                        .unwrap();
                }
            }
        }

        let notes_slides: Vec<_> = slides
            .iter()
            .enumerate()
            .filter(|(_, s)| s.notes.is_some())
            .collect();
        if !notes_slides.is_empty() {
            zip.add_directory("ppt/notesSlides", options).unwrap();
            for (i, slide) in &notes_slides {
                let filename = format!("ppt/notesSlides/notesSlide{}.xml", i + 1);
                zip.start_file(&filename, options).unwrap();
                zip.write_all(generate_notes_xml(slide.notes.as_ref().unwrap()).as_bytes())
                    .unwrap();
            }
        }

        zip.finish().unwrap();
    }
    buffer.into_inner()
}

struct SlideContent {
    title: String,
    content: String,
    notes: Option<String>,
}

impl SlideContent {
    fn new(title: &str, content: &str) -> Self {
        Self {
            title: title.to_string(),
            content: content.to_string(),
            notes: None,
        }
    }

    fn with_notes(mut self, notes: &str) -> Self {
        self.notes = Some(notes.to_string());
        self
    }
}

const RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"#;

fn generate_content_types(slides: &[SlideContent]) -> String {
    let mut slide_overrides = String::new();
    let mut notes_overrides = String::new();

    for (i, slide) in slides.iter().enumerate() {
        slide_overrides.push_str(&format!(
            r#"<Override PartName="/ppt/slides/slide{}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>"#,
            i + 1
        ));
        if slide.notes.is_some() {
            notes_overrides.push_str(&format!(
                r#"<Override PartName="/ppt/notesSlides/notesSlide{}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>"#,
                i + 1
            ));
        }
    }

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
{slide_overrides}{notes_overrides}</Types>"#
    )
}

fn generate_presentation_xml(slides: &[SlideContent]) -> String {
    let mut slide_ids = String::new();
    for (i, _) in slides.iter().enumerate() {
        slide_ids.push_str(&format!(
            r#"<p:sldId id="{}" r:id="rId{}"/>"#,
            256 + i,
            i + 1
        ));
    }

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldIdLst>{slide_ids}</p:sldIdLst>
</p:presentation>"#
    )
}

fn generate_presentation_rels(slides: &[SlideContent]) -> String {
    let mut relationships = String::new();
    for (i, _) in slides.iter().enumerate() {
        relationships.push_str(&format!(
            r#"<Relationship Id="rId{}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{}.xml"/>"#,
            i + 1, i + 1
        ));
    }

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{relationships}</Relationships>"#
    )
}

fn generate_slide_xml(title: &str, content: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree>
<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr/>
<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>
<p:sp><p:nvSpPr><p:cNvPr id="3" name="Content"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:p><a:r><a:t>{content}</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld>
</p:sld>"#
    )
}

fn generate_slide_rels(slide_num: usize) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{}.xml"/>
</Relationships>"#,
        slide_num
    )
}

fn generate_notes_xml(notes_text: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree>
<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr/>
<p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:p><a:r><a:t>{notes_text}</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld>
</p:notes>"#
    )
}

#[test]
fn test_pptx_basic_text() {
    let bytes = create_test_pptx(vec![SlideContent::new(
        "Welcome Slide",
        "This is the introduction content.",
    )]);

    let result = PptxExtractor::extract(&bytes);
    assert!(result.is_ok(), "Should successfully extract PPTX content");

    let extracted = result.unwrap();
    assert_eq!(extracted.doc_type, DocumentType::Pptx);
    assert!(
        extracted.text.contains("Welcome Slide"),
        "Should contain slide title"
    );
    assert!(
        extracted.text.contains("This is the introduction content."),
        "Should contain slide content"
    );
    assert!(extracted.word_count > 0, "Should have word count");
}

#[test]
fn test_pptx_slide_ordering() {
    let bytes = create_test_pptx(vec![
        SlideContent::new("Slide 1 Title", "First slide content"),
        SlideContent::new("Slide 2 Title", "Second slide content"),
        SlideContent::new("Slide 3 Title", "Third slide content"),
    ]);

    let result = PptxExtractor::extract(&bytes);
    assert!(result.is_ok());

    let extracted = result.unwrap();
    let text = &extracted.text;

    let pos1 = text.find("Slide 1").expect("Should find Slide 1");
    let pos2 = text.find("Slide 2").expect("Should find Slide 2");
    let pos3 = text.find("Slide 3").expect("Should find Slide 3");

    assert!(pos1 < pos2, "Slide 1 should appear before Slide 2");
    assert!(pos2 < pos3, "Slide 2 should appear before Slide 3");

    assert!(
        text.contains("## Slide 1") || text.contains("# Slide 1"),
        "Should have slide 1 header"
    );
    assert!(
        text.contains("## Slide 2") || text.contains("# Slide 2"),
        "Should have slide 2 header"
    );
    assert!(
        text.contains("## Slide 3") || text.contains("# Slide 3"),
        "Should have slide 3 header"
    );
}

#[test]
fn test_pptx_speaker_notes() {
    let bytes = create_test_pptx(vec![
        SlideContent::new("Intro", "Welcome to the presentation")
            .with_notes("Remember to greet the audience."),
        SlideContent::new("Main Point", "Key information here"),
        SlideContent::new("Conclusion", "Thank you for listening")
            .with_notes("Take questions from the audience."),
    ]);

    let result = PptxExtractor::extract(&bytes);
    assert!(result.is_ok());

    let extracted = result.unwrap();
    let text = &extracted.text;

    assert!(
        text.contains("Remember to greet the audience"),
        "Should contain speaker notes for slide 1"
    );
    assert!(
        text.contains("Take questions from the audience"),
        "Should contain speaker notes for slide 3"
    );
    assert!(text.contains("[Notes]"), "Notes should be marked in output");
}

#[test]
fn test_pptx_empty() {
    let bytes = create_test_pptx(vec![]);

    let result = PptxExtractor::extract(&bytes);
    assert!(result.is_ok(), "Should handle empty presentation");

    let extracted = result.unwrap();
    assert_eq!(
        extracted.word_count, 0,
        "Empty presentation should have 0 words"
    );
    assert!(
        extracted.text.is_empty() || extracted.text.trim().is_empty(),
        "Empty presentation should have empty text"
    );
}

#[test]
fn test_pptx_corrupt() {
    let corrupt_bytes = vec![0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE];

    let result = PptxExtractor::extract(&corrupt_bytes);
    assert!(result.is_err(), "Should fail on corrupt PPTX data");

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("PPTX") || err_msg.contains("parse") || err_msg.contains("zip"),
        "Error should mention PPTX/parsing: {}",
        err_msg
    );
}

#[test]
fn test_pptx_with_fixture() {
    ensure_fixtures();
    let bytes = common::load_fixture("sample.pptx");

    let result = PptxExtractor::extract(&bytes);
    assert!(result.is_ok(), "Should extract from fixture PPTX");

    let extracted = result.unwrap();
    assert!(
        extracted.text.contains("Test Presentation"),
        "Should contain fixture content"
    );
    assert_eq!(extracted.doc_type, DocumentType::Pptx);
}
