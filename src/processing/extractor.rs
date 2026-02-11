use scraper::{Html, Selector};
use url::Url;

use crate::error::{MomoError, Result};
use crate::models::DocumentType;
use crate::processing::extractors::{self, ExtractedContent};
use crate::processing::language::detect_language;

/// Image format detected from magic bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    WebP,
    Tiff,
    Bmp,
}

/// Audio format detected from magic bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Mp3,
    Wav,
    Flac,
    Ogg,
    M4a,
}

/// Video format detected from magic bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoFormat {
    Mp4,
    Webm,
    Avi,
    Mkv,
}

/// Detect image format from magic bytes
pub fn detect_image_format(bytes: &[u8]) -> Option<ImageFormat> {
    // JPEG: FF D8 FF
    if bytes.len() >= 3 && bytes[0..3] == [0xFF, 0xD8, 0xFF] {
        return Some(ImageFormat::Jpeg);
    }
    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if bytes.len() >= 8 && bytes[0..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some(ImageFormat::Png);
    }
    // WebP: RIFF....WEBP
    if bytes.len() >= 12
        && bytes[0..4] == [0x52, 0x49, 0x46, 0x46]
        && bytes[8..12] == [0x57, 0x45, 0x42, 0x50]
    {
        return Some(ImageFormat::WebP);
    }
    // TIFF Little Endian: 49 49 2A 00
    if bytes.len() >= 4 && bytes[0..4] == [0x49, 0x49, 0x2A, 0x00] {
        return Some(ImageFormat::Tiff);
    }
    // TIFF Big Endian: 4D 4D 00 2A
    if bytes.len() >= 4 && bytes[0..4] == [0x4D, 0x4D, 0x00, 0x2A] {
        return Some(ImageFormat::Tiff);
    }
    // BMP: 42 4D
    if bytes.len() >= 2 && bytes[0..2] == [0x42, 0x4D] {
        return Some(ImageFormat::Bmp);
    }
    None
}

/// Detect audio format from magic bytes
pub fn detect_audio_format(bytes: &[u8]) -> Option<AudioFormat> {
    // MP3 frame sync: FF FB / FF F3 / FF F2
    if bytes.len() >= 2
        && bytes[0] == 0xFF
        && (bytes[1] == 0xFB || bytes[1] == 0xF3 || bytes[1] == 0xF2)
    {
        return Some(AudioFormat::Mp3);
    }

    // MP3 ID3 tag: "ID3"
    if bytes.len() >= 3 && bytes[0..3] == [0x49, 0x44, 0x33] {
        return Some(AudioFormat::Mp3);
    }

    // WAV: RIFF....WAVE
    if bytes.len() >= 12
        && bytes[0..4] == [0x52, 0x49, 0x46, 0x46]
        && bytes[8..12] == [0x57, 0x41, 0x56, 0x45]
    {
        return Some(AudioFormat::Wav);
    }

    // FLAC: fLaC
    if bytes.len() >= 4 && bytes[0..4] == [0x66, 0x4C, 0x61, 0x43] {
        return Some(AudioFormat::Flac);
    }

    // OGG: OggS
    if bytes.len() >= 4 && bytes[0..4] == [0x4F, 0x67, 0x67, 0x53] {
        return Some(AudioFormat::Ogg);
    }

    // M4A container: ftyp with audio-specific brands
    if bytes.len() >= 12 && bytes[4..8] == [0x66, 0x74, 0x79, 0x70] {
        let brand = &bytes[8..12];
        if brand == b"M4A " || brand == b"M4B " {
            return Some(AudioFormat::M4a);
        }
    }

    None
}

/// Detect video format from magic bytes
pub fn detect_video_format(bytes: &[u8]) -> Option<VideoFormat> {
    // MP4/WebM/etc. often have ISO BMFF ftyp box at offset 4
    if bytes.len() >= 12 && bytes[4..8] == [0x66, 0x74, 0x79, 0x70] {
        let brand = &bytes[8..12];
        if brand == b"isom" || brand == b"iso2" || brand == b"mp41" || brand == b"mp42" {
            return Some(VideoFormat::Mp4);
        }
    }

    // AVI: RIFF....AVI
    if bytes.len() >= 12
        && bytes[0..4] == [0x52, 0x49, 0x46, 0x46]
        && bytes[8..12] == [0x41, 0x56, 0x49, 0x20]
    {
        return Some(VideoFormat::Avi);
    }

    // WebM/Matroska: 1A 45 DF A3
    if bytes.len() >= 4 && bytes[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        if bytes.windows(4).any(|w| w == b"webm") {
            return Some(VideoFormat::Webm);
        }
        return Some(VideoFormat::Mkv);
    }

    None
}

pub struct ContentExtractor {
    http_client: reqwest::Client,
}

impl ContentExtractor {
    pub fn new() -> Self {
        Self {
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .user_agent("NovaMemory/1.0")
                .build()
                .unwrap_or_default(),
        }
    }

    pub async fn extract(&self, content: &str) -> Result<ExtractedContent> {
        if content.starts_with("http://") || content.starts_with("https://") {
            self.extract_from_url(content).await
        } else if Self::looks_like_html(content) {
            self.extract_from_html(content)
        } else {
            let doc_type = if Self::looks_like_code(content) {
                DocumentType::Code
            } else {
                DocumentType::Text
            };
            Ok(ExtractedContent {
                text: content.to_string(),
                title: None,
                doc_type,
                url: None,
                word_count: Self::count_words(content),
                source_path: None,
            })
        }
    }

    pub async fn extract_from_url(&self, url_str: &str) -> Result<ExtractedContent> {
        let url = Url::parse(url_str)?;
        let source_path = Self::extract_source_path_from_url(&url);
        let response = self.http_client.get(url.clone()).send().await?;

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/plain");

        let doc_type = Self::detect_type_from_content_type(content_type, url_str);

        match doc_type {
            DocumentType::Pdf => {
                let bytes = response.bytes().await?;
                let mut extracted = self.extract_from_pdf(&bytes, Some(url_str))?;
                extracted.source_path = source_path;
                Ok(extracted)
            }
            DocumentType::Docx => {
                let bytes = response.bytes().await?;
                let mut extracted = self.extract_from_docx(&bytes)?;
                extracted.url = Some(url_str.to_string());
                extracted.source_path = source_path;
                Ok(extracted)
            }
            DocumentType::Xlsx => {
                let bytes = response.bytes().await?;
                let mut extracted = self.extract_from_xlsx(&bytes)?;
                extracted.url = Some(url_str.to_string());
                extracted.source_path = source_path;
                Ok(extracted)
            }
            DocumentType::Pptx => {
                let bytes = response.bytes().await?;
                let mut extracted = self.extract_from_pptx(&bytes)?;
                extracted.url = Some(url_str.to_string());
                extracted.source_path = source_path;
                Ok(extracted)
            }
            DocumentType::Csv => {
                let bytes = response.bytes().await?;
                let mut extracted = self.extract_from_csv(&bytes)?;
                extracted.url = Some(url_str.to_string());
                extracted.source_path = source_path;
                Ok(extracted)
            }
            _ => {
                let text = response.text().await?;

                // Check if URL points to a code file before treating as webpage
                if let Some(ref path) = source_path {
                    if detect_language(path).is_some() {
                        let word_count = Self::count_words(&text);
                        return Ok(ExtractedContent {
                            text,
                            title: None,
                            doc_type: DocumentType::Code,
                            url: Some(url_str.to_string()),
                            word_count,
                            source_path,
                        });
                    }
                }

                let mut extracted = self.extract_from_html(&text)?;
                extracted.url = Some(url_str.to_string());
                extracted.doc_type = DocumentType::Webpage;
                extracted.source_path = source_path;
                Ok(extracted)
            }
        }
    }

    fn extract_source_path_from_url(url: &Url) -> Option<String> {
        let path = url.path();
        if path.is_empty() || path == "/" {
            return None;
        }
        let filename = path.rsplit('/').next()?;
        if filename.is_empty() || !filename.contains('.') {
            return None;
        }
        Some(filename.to_string())
    }

    pub fn extract_from_html(&self, html: &str) -> Result<ExtractedContent> {
        let document = Html::parse_document(html);

        let title = Self::extract_title(&document);
        let text = Self::extract_text(&document);
        let word_count = Self::count_words(&text);

        Ok(ExtractedContent {
            text,
            title,
            doc_type: DocumentType::Webpage,
            url: None,
            word_count,
            source_path: None,
        })
    }

    pub fn extract_from_pdf(&self, bytes: &[u8], url: Option<&str>) -> Result<ExtractedContent> {
        let text = pdf_extract::extract_text_from_mem(bytes)
            .map_err(|e| MomoError::Processing(format!("PDF extraction failed: {e}")))?;

        let word_count = Self::count_words(&text);

        Ok(ExtractedContent {
            text,
            title: None,
            doc_type: DocumentType::Pdf,
            url: url.map(String::from),
            word_count,
            source_path: None,
        })
    }

    pub fn extract_from_csv(&self, bytes: &[u8]) -> Result<ExtractedContent> {
        extractors::CsvExtractor::extract(bytes)
    }

    pub fn extract_from_docx(&self, bytes: &[u8]) -> Result<ExtractedContent> {
        extractors::DocxExtractor::extract(bytes)
    }

    pub fn extract_from_xlsx(&self, bytes: &[u8]) -> Result<ExtractedContent> {
        extractors::XlsxExtractor::extract(bytes)
    }

    pub fn extract_from_pptx(&self, bytes: &[u8]) -> Result<ExtractedContent> {
        extractors::PptxExtractor::extract(bytes)
    }

    fn extract_title(document: &Html) -> Option<String> {
        let title_selector = Selector::parse("title").ok()?;
        document
            .select(&title_selector)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_string())
            .filter(|s| !s.is_empty())
    }

    fn extract_text(document: &Html) -> String {
        let selectors_to_remove = [
            "script", "style", "noscript", "nav", "header", "footer", "aside", "iframe", "svg",
        ];

        let body_selector = Selector::parse("body").unwrap();
        let article_selector = Selector::parse("article, main, .content, #content").unwrap();

        let content_root = document
            .select(&article_selector)
            .next()
            .or_else(|| document.select(&body_selector).next());

        let Some(root) = content_root else {
            return document.root_element().text().collect::<String>();
        };

        let mut text = String::new();

        for node in root.descendants() {
            if let Some(element) = node.value().as_element() {
                let tag_name = element.name();
                if selectors_to_remove.contains(&tag_name) {
                    continue;
                }
            }

            if let Some(text_node) = node.value().as_text() {
                let content = text_node.trim();
                if !content.is_empty() {
                    if !text.is_empty() && !text.ends_with(' ') && !text.ends_with('\n') {
                        text.push(' ');
                    }
                    text.push_str(content);
                }
            }
        }

        Self::clean_text(&text)
    }

    fn clean_text(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut prev_was_whitespace = false;
        let mut consecutive_newlines = 0;

        for c in text.chars() {
            if c == '\n' {
                consecutive_newlines += 1;
                if consecutive_newlines <= 2 {
                    result.push(c);
                }
                prev_was_whitespace = true;
            } else if c.is_whitespace() {
                consecutive_newlines = 0;
                if !prev_was_whitespace {
                    result.push(' ');
                    prev_was_whitespace = true;
                }
            } else {
                consecutive_newlines = 0;
                result.push(c);
                prev_was_whitespace = false;
            }
        }

        result.trim().to_string()
    }

    /// Heuristic to detect if plain text content looks like source code.
    /// Checks for common code patterns: function definitions, import statements,
    /// braces/semicolons density, and language-specific keywords.
    fn looks_like_code(content: &str) -> bool {
        let trimmed = content.trim();
        if trimmed.len() < 20 {
            return false;
        }

        let lines: Vec<&str> = trimmed.lines().take(50).collect();
        if lines.is_empty() {
            return false;
        }

        let mut score: i32 = 0;

        // Check for shebang line
        if trimmed.starts_with("#!")
            && (trimmed.starts_with("#!/usr/bin") || trimmed.starts_with("#!/bin"))
        {
            score += 3;
        }

        // Language-specific patterns (check first ~50 lines)
        let sample = lines.join("\n");

        // Rust patterns
        if sample.contains("fn ") && (sample.contains("-> ") || sample.contains("pub ")) {
            score += 3;
        }
        if sample.contains("use std::") || sample.contains("use crate::") {
            score += 3;
        }

        // Python patterns
        if sample.contains("def ") && sample.contains(':') {
            score += 2;
        }
        if sample.contains("import ") || sample.contains("from ") && sample.contains(" import ") {
            score += 2;
        }

        // JS/TS patterns
        if sample.contains("function ") || sample.contains("const ") || sample.contains("let ") {
            score += 1;
        }
        if sample.contains("require(") || sample.contains("module.exports") {
            score += 3;
        }
        if sample.contains("import ") && sample.contains(" from ") {
            score += 2;
        }
        if sample.contains("export ")
            && (sample.contains("default")
                || sample.contains("function")
                || sample.contains("class"))
        {
            score += 2;
        }

        // Go patterns
        if sample.contains("func ") && sample.contains("package ") {
            score += 3;
        }

        // Java/C/C++ patterns
        if sample.contains("public class ")
            || sample.contains("private ")
            || sample.contains("protected ")
        {
            score += 2;
        }
        if sample.contains("#include") {
            score += 3;
        }
        if sample.contains("int main(") || sample.contains("void main(") {
            score += 3;
        }

        // Generic code indicators: semicolons + braces density
        let semicolons = sample.chars().filter(|c| *c == ';').count();
        let braces = sample.chars().filter(|c| *c == '{' || *c == '}').count();
        let total_chars = sample.len().max(1);

        // High density of semicolons or braces suggests code
        if semicolons > 3 && (semicolons as f64 / total_chars as f64) > 0.005 {
            score += 2;
        }
        if braces > 3 && (braces as f64 / total_chars as f64) > 0.005 {
            score += 2;
        }

        // Indentation patterns (consistent 2/4 space or tab indentation)
        let indented_lines = lines
            .iter()
            .filter(|l| l.starts_with("    ") || l.starts_with('\t'))
            .count();
        if lines.len() > 3 && indented_lines as f64 / lines.len() as f64 > 0.3 {
            score += 1;
        }

        score >= 4
    }

    fn looks_like_html(content: &str) -> bool {
        let trimmed = content.trim_start();
        trimmed.starts_with("<!DOCTYPE")
            || trimmed.starts_with("<!doctype")
            || trimmed.starts_with("<html")
            || trimmed.starts_with("<HTML")
    }

    fn detect_type_from_content_type(content_type: &str, url: &str) -> DocumentType {
        if content_type.contains("application/pdf") || url.ends_with(".pdf") {
            DocumentType::Pdf
        } else if content_type.contains("text/html") {
            DocumentType::Webpage
        } else if content_type.contains("text/markdown") || url.ends_with(".md") {
            DocumentType::Markdown
        } else if content_type.contains("image/") {
            DocumentType::Image
        } else if content_type.contains("video/") {
            DocumentType::Video
        } else if content_type.contains("audio/") {
            DocumentType::Audio
        } else if content_type
            .contains("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            || url.ends_with(".docx")
        {
            DocumentType::Docx
        } else if content_type
            .contains("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            || url.ends_with(".xlsx")
        {
            DocumentType::Xlsx
        } else if content_type
            .contains("application/vnd.openxmlformats-officedocument.presentationml.presentation")
            || url.ends_with(".pptx")
        {
            DocumentType::Pptx
        } else if content_type.contains("text/csv")
            || content_type.contains("application/csv")
            || url.ends_with(".csv")
        {
            DocumentType::Csv
        } else if detect_language(url).is_some() {
            DocumentType::Code
        } else {
            DocumentType::Text
        }
    }

    pub fn detect_type_from_bytes(bytes: &[u8]) -> DocumentType {
        if detect_image_format(bytes).is_some() {
            return DocumentType::Image;
        }

        if detect_audio_format(bytes).is_some() {
            return DocumentType::Audio;
        }

        if detect_video_format(bytes).is_some() {
            return DocumentType::Video;
        }

        if bytes.starts_with(&[0x25, 0x50, 0x44, 0x46]) {
            return DocumentType::Pdf;
        }

        if bytes.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
            if let Ok(mut archive) = zip::ZipArchive::new(std::io::Cursor::new(bytes)) {
                if archive.by_name("[Content_Types].xml").is_ok() {
                    if archive.by_name("word/document.xml").is_ok() {
                        return DocumentType::Docx;
                    }
                    if archive.by_name("xl/workbook.xml").is_ok() {
                        return DocumentType::Xlsx;
                    }
                    if archive.by_name("ppt/presentation.xml").is_ok() {
                        return DocumentType::Pptx;
                    }
                }
            }
            return DocumentType::Unknown;
        }

        if Self::looks_like_csv(bytes) {
            return DocumentType::Csv;
        }

        // Try to detect text-based formats
        if let Ok(text) = std::str::from_utf8(bytes) {
            let trimmed = text.trim();

            // Check for HTML
            let lower = trimmed.to_lowercase();
            if lower.starts_with("<!doctype")
                || lower.starts_with("<html")
                || lower.starts_with("<head")
            {
                return DocumentType::Webpage;
            }

            // Check for Markdown (basic heuristics)
            if trimmed.starts_with('#')
                || trimmed.contains("\n# ")
                || trimmed.starts_with("- ")
                || trimmed.starts_with("* ")
                || trimmed.contains("\n- ")
                || trimmed.contains("\n* ")
            {
                return DocumentType::Markdown;
            }

            // Valid UTF-8 text
            return DocumentType::Text;
        }

        DocumentType::Unknown
    }

    pub fn detect_type_from_upload(
        bytes: &[u8],
        file_name: Option<&str>,
        content_type: Option<&str>,
    ) -> DocumentType {
        let by_bytes = Self::detect_type_from_bytes(bytes);
        if !matches!(by_bytes, DocumentType::Unknown) {
            return by_bytes;
        }

        if let Some(ct) = content_type {
            let ct_lower = ct.to_lowercase();
            if ct_lower.starts_with("image/") {
                return DocumentType::Image;
            }
            if ct_lower.starts_with("audio/") {
                return DocumentType::Audio;
            }
            if ct_lower.starts_with("video/") {
                return DocumentType::Video;
            }
        }

        if let Some(name) = file_name {
            let lower = name.to_lowercase();
            if lower.ends_with(".mp3")
                || lower.ends_with(".wav")
                || lower.ends_with(".m4a")
                || lower.ends_with(".ogg")
                || lower.ends_with(".flac")
            {
                return DocumentType::Audio;
            }
            if lower.ends_with(".mp4")
                || lower.ends_with(".webm")
                || lower.ends_with(".avi")
                || lower.ends_with(".mkv")
                || lower.ends_with(".mov")
            {
                return DocumentType::Video;
            }
        }

        DocumentType::Unknown
    }

    fn looks_like_csv(bytes: &[u8]) -> bool {
        let text = match std::str::from_utf8(bytes) {
            Ok(t) => t,
            Err(_) => return false,
        };

        let lines: Vec<&str> = text.lines().take(5).collect();
        if lines.len() < 2 {
            return false;
        }

        let delimiters = [',', ';', '\t'];
        for delimiter in &delimiters {
            let first_line_cols = lines[0].split(*delimiter).count();
            if first_line_cols >= 2 {
                let consistent = lines.iter().all(|line| {
                    let cols = line.split(*delimiter).count();
                    cols == first_line_cols || cols == 1
                });
                if consistent {
                    return true;
                }
            }
        }

        false
    }

    fn count_words(text: &str) -> i32 {
        text.split_whitespace().count() as i32
    }
}

impl Default for ContentExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_image_type_jpeg() {
        let jpeg_bytes = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&jpeg_bytes),
            DocumentType::Image
        );
    }

    #[test]
    fn test_detect_image_type_png() {
        let png_bytes = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&png_bytes),
            DocumentType::Image
        );
    }

    #[test]
    fn test_detect_image_type_webp() {
        let mut webp_bytes = vec![0x52, 0x49, 0x46, 0x46];
        webp_bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        webp_bytes.extend_from_slice(&[0x57, 0x45, 0x42, 0x50]);
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&webp_bytes),
            DocumentType::Image
        );
    }

    #[test]
    fn test_detect_image_type_tiff_le() {
        let tiff_le = [0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00];
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&tiff_le),
            DocumentType::Image
        );
    }

    #[test]
    fn test_detect_image_type_tiff_be() {
        let tiff_be = [0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08];
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&tiff_be),
            DocumentType::Image
        );
    }

    #[test]
    fn test_detect_image_type_bmp() {
        let bmp_bytes = [0x42, 0x4D, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&bmp_bytes),
            DocumentType::Image
        );
    }

    #[test]
    fn test_detect_image_format_returns_correct_type() {
        assert_eq!(
            detect_image_format(&[0xFF, 0xD8, 0xFF]),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(
            detect_image_format(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
            Some(ImageFormat::Png)
        );
        assert_eq!(
            detect_image_format(&[0x42, 0x4D, 0x00, 0x00]),
            Some(ImageFormat::Bmp)
        );
        assert_eq!(detect_image_format(&[0x00, 0x00, 0x00, 0x00]), None);
    }

    #[test]
    fn test_detect_pdf_still_works() {
        let pdf_bytes = [0x25, 0x50, 0x44, 0x46, 0x2D, 0x31, 0x2E];
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&pdf_bytes),
            DocumentType::Pdf
        );
    }

    #[test]
    fn test_detect_audio_types_from_bytes() {
        let wav_bytes = [
            0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
        ];
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&wav_bytes),
            DocumentType::Audio
        );

        let mp3_id3 = [0x49, 0x44, 0x33, 0x04, 0x00, 0x00];
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&mp3_id3),
            DocumentType::Audio
        );
    }

    #[test]
    fn test_detect_video_types_from_bytes() {
        let mp4_bytes = [
            0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6F, 0x6D,
        ];
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&mp4_bytes),
            DocumentType::Video
        );

        let mkv_bytes = [0x1A, 0x45, 0xDF, 0xA3, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            ContentExtractor::detect_type_from_bytes(&mkv_bytes),
            DocumentType::Video
        );
    }

    #[test]
    fn test_detect_type_from_upload_fallbacks() {
        let unknown = [0x00, 0x01, 0x02, 0x03];

        assert_eq!(
            ContentExtractor::detect_type_from_upload(&unknown, Some("voice-note.mp3"), None),
            DocumentType::Audio
        );

        assert_eq!(
            ContentExtractor::detect_type_from_upload(
                &unknown,
                Some("clip.bin"),
                Some("video/webm")
            ),
            DocumentType::Video
        );
    }

    #[test]
    fn test_looks_like_code_rust() {
        let rust_code = r#"
use std::collections::HashMap;

pub fn process(data: &str) -> Result<(), Error> {
    let map = HashMap::new();
    println!("{:?}", map);
    Ok(())
}
"#;
        assert!(ContentExtractor::looks_like_code(rust_code));
    }

    #[test]
    fn test_looks_like_code_python() {
        let python_code = r#"
import os
from pathlib import Path

def process_file(path: str) -> None:
    with open(path) as f:
        data = f.read()
    print(data)
"#;
        assert!(ContentExtractor::looks_like_code(python_code));
    }

    #[test]
    fn test_looks_like_code_javascript() {
        let js_code = r#"
const express = require('express');
const app = express();

function handleRequest(req, res) {
    res.json({ status: 'ok' });
}

module.exports = { handleRequest };
"#;
        assert!(ContentExtractor::looks_like_code(js_code));
    }

    #[test]
    fn test_looks_like_code_c_cpp() {
        let c_code = r#"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    printf("Hello, world!\n");
    return 0;
}
"#;
        assert!(ContentExtractor::looks_like_code(c_code));
    }

    #[test]
    fn test_looks_like_code_rejects_prose() {
        let prose = "This is a normal paragraph of text about programming. \
            It mentions functions and variables but is clearly just prose \
            written in natural language without any code structure.";
        assert!(!ContentExtractor::looks_like_code(prose));
    }

    #[test]
    fn test_looks_like_code_rejects_short_text() {
        assert!(!ContentExtractor::looks_like_code("fn x"));
        assert!(!ContentExtractor::looks_like_code(""));
    }

    #[test]
    fn test_extract_plain_code_returns_code_type() {
        let extractor = ContentExtractor::new();
        let code = r#"
use std::io;

fn main() {
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    println!("You said: {}", input.trim());
}
"#;
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(extractor.extract(code));
        assert!(result.is_ok());
        let extracted = result.unwrap();
        assert_eq!(extracted.doc_type, DocumentType::Code);
    }

    #[test]
    fn test_extract_plain_text_returns_text_type() {
        let extractor = ContentExtractor::new();
        let text = "This is a simple paragraph of text without any code patterns. \
            It's just regular prose about everyday topics.";
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(extractor.extract(text));
        assert!(result.is_ok());
        let extracted = result.unwrap();
        assert_eq!(extracted.doc_type, DocumentType::Text);
    }

    #[test]
    fn test_detect_type_from_content_type_code_extensions() {
        assert_eq!(
            ContentExtractor::detect_type_from_content_type(
                "text/plain",
                "https://example.com/main.rs"
            ),
            DocumentType::Code
        );
        assert_eq!(
            ContentExtractor::detect_type_from_content_type(
                "text/plain",
                "https://example.com/app.py"
            ),
            DocumentType::Code
        );
        assert_eq!(
            ContentExtractor::detect_type_from_content_type(
                "text/plain",
                "https://example.com/index.ts"
            ),
            DocumentType::Code
        );
        assert_eq!(
            ContentExtractor::detect_type_from_content_type(
                "text/plain",
                "https://example.com/readme.txt"
            ),
            DocumentType::Text
        );
    }

    #[test]
    fn test_extract_source_path_from_url() {
        let url = Url::parse("https://example.com/repo/src/main.rs").unwrap();
        assert_eq!(
            ContentExtractor::extract_source_path_from_url(&url),
            Some("main.rs".to_string())
        );

        let url = Url::parse("https://example.com/").unwrap();
        assert_eq!(ContentExtractor::extract_source_path_from_url(&url), None);

        let url = Url::parse("https://example.com/no-extension").unwrap();
        assert_eq!(ContentExtractor::extract_source_path_from_url(&url), None);
    }
}
