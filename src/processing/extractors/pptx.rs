//! PPTX extractor using zip + quick-xml

use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;
use std::io::{Cursor, Read};
use zip::ZipArchive;

use super::ExtractedContent;
use crate::error::{MomoError, Result};
use crate::models::DocumentType;

pub struct PptxExtractor;

impl PptxExtractor {
    pub fn extract(bytes: &[u8]) -> Result<ExtractedContent> {
        let cursor = Cursor::new(bytes);
        let mut archive = ZipArchive::new(cursor)
            .map_err(|e| MomoError::Processing(format!("PPTX parse error: {e}")))?;

        let slide_order = Self::get_slide_order(&mut archive)?;

        if slide_order.is_empty() {
            return Ok(ExtractedContent {
                text: String::new(),
                title: None,
                doc_type: DocumentType::Pptx,
                url: None,
                word_count: 0,
                source_path: None,
            });
        }

        let slide_mapping = Self::get_slide_mapping(&mut archive)?;

        let mut text = String::new();
        let mut title = None;

        for (slide_num, r_id) in slide_order.iter().enumerate() {
            let slide_number = slide_num + 1;

            let slide_filename = match slide_mapping.get(r_id) {
                Some(filename) => filename.clone(),
                None => format!("ppt/slides/slide{slide_number}.xml"),
            };

            let slide_text = Self::extract_slide_content(&mut archive, &slide_filename)?;

            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&format!("## Slide {slide_number}\n\n"));

            if !slide_text.is_empty() {
                text.push_str(&slide_text);

                if title.is_none() && !slide_text.trim().is_empty() {
                    title = slide_text.lines().next().map(|s| s.trim().to_string());
                }
            }

            let notes_text = Self::extract_notes(&mut archive, slide_number)?;
            if let Some(notes) = notes_text {
                text.push_str("\n\n[Notes]: ");
                text.push_str(&notes);
            }
        }

        let word_count = Self::count_words(&text);

        Ok(ExtractedContent {
            text,
            title,
            doc_type: DocumentType::Pptx,
            url: None,
            word_count,
            source_path: None,
        })
    }

    fn get_slide_order(archive: &mut ZipArchive<Cursor<&[u8]>>) -> Result<Vec<String>> {
        let xml = Self::read_file_from_archive(archive, "ppt/presentation.xml")?;

        let mut reader = Reader::from_str(&xml);
        reader.config_mut().trim_text(true);

        let mut slide_ids = Vec::new();
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Empty(e)) | Ok(Event::Start(e)) => {
                    if e.name().as_ref() == b"p:sldId" {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"r:id" {
                                if let Ok(val) = std::str::from_utf8(&attr.value) {
                                    slide_ids.push(val.to_string());
                                }
                            }
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    return Err(MomoError::Processing(format!(
                        "Error parsing presentation.xml: {e}"
                    )))
                }
                _ => {}
            }
            buf.clear();
        }

        Ok(slide_ids)
    }

    fn get_slide_mapping(
        archive: &mut ZipArchive<Cursor<&[u8]>>,
    ) -> Result<HashMap<String, String>> {
        let xml = Self::read_file_from_archive(archive, "ppt/_rels/presentation.xml.rels")?;

        let mut reader = Reader::from_str(&xml);
        reader.config_mut().trim_text(true);

        let mut mapping = HashMap::new();
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Empty(e)) | Ok(Event::Start(e)) => {
                    if e.name().as_ref() == b"Relationship" {
                        let mut id = None;
                        let mut target = None;
                        let mut rel_type = None;

                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"Id" => {
                                    id = std::str::from_utf8(&attr.value).ok().map(String::from);
                                }
                                b"Target" => {
                                    target =
                                        std::str::from_utf8(&attr.value).ok().map(String::from);
                                }
                                b"Type" => {
                                    rel_type =
                                        std::str::from_utf8(&attr.value).ok().map(String::from);
                                }
                                _ => {}
                            }
                        }

                        if let (Some(id), Some(target), Some(rel_type)) = (id, target, rel_type) {
                            if rel_type.contains("slide") && !rel_type.contains("slideLayout") {
                                let full_path = format!("ppt/{target}");
                                mapping.insert(id, full_path);
                            }
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
            buf.clear();
        }

        Ok(mapping)
    }

    fn extract_slide_content(
        archive: &mut ZipArchive<Cursor<&[u8]>>,
        slide_path: &str,
    ) -> Result<String> {
        let xml = match Self::read_file_from_archive(archive, slide_path) {
            Ok(content) => content,
            Err(_) => return Ok(String::new()),
        };

        Ok(Self::extract_text_from_xml(&xml))
    }

    fn extract_notes(
        archive: &mut ZipArchive<Cursor<&[u8]>>,
        slide_number: usize,
    ) -> Result<Option<String>> {
        let notes_path = format!("ppt/notesSlides/notesSlide{slide_number}.xml");

        let xml = match Self::read_file_from_archive(archive, &notes_path) {
            Ok(content) => content,
            Err(_) => return Ok(None),
        };

        let text = Self::extract_text_from_xml(&xml);
        if text.trim().is_empty() {
            Ok(None)
        } else {
            Ok(Some(text))
        }
    }

    fn extract_text_from_xml(xml: &str) -> String {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut text_parts = Vec::new();
        let mut current_paragraph = String::new();
        let mut in_text_element = false;
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    if e.name().as_ref() == b"a:t" {
                        in_text_element = true;
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_text_element {
                        if let Ok(text) = std::str::from_utf8(e.as_ref()) {
                            let unescaped = Self::unescape_xml(text);
                            current_paragraph.push_str(&unescaped);
                        }
                    }
                }
                Ok(Event::End(e)) => {
                    if e.name().as_ref() == b"a:t" {
                        in_text_element = false;
                    } else if e.name().as_ref() == b"a:p" {
                        let trimmed = current_paragraph.trim().to_string();
                        if !trimmed.is_empty() {
                            text_parts.push(trimmed);
                        }
                        current_paragraph.clear();
                    }
                }
                Ok(Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
            buf.clear();
        }

        let trimmed = current_paragraph.trim().to_string();
        if !trimmed.is_empty() {
            text_parts.push(trimmed);
        }

        text_parts.join("\n\n")
    }

    fn unescape_xml(text: &str) -> String {
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&apos;", "'")
            .replace("&quot;", "\"")
    }

    fn read_file_from_archive(
        archive: &mut ZipArchive<Cursor<&[u8]>>,
        path: &str,
    ) -> Result<String> {
        let mut file = archive
            .by_name(path)
            .map_err(|e| MomoError::Processing(format!("Failed to read {path} from PPTX: {e}")))?;

        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| MomoError::Processing(format!("Failed to read {path} content: {e}")))?;

        Ok(content)
    }

    fn count_words(text: &str) -> i32 {
        text.split_whitespace().count() as i32
    }
}
