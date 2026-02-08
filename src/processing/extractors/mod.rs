use crate::models::DocumentType;

#[derive(Debug)]
pub struct ExtractedContent {
    pub text: String,
    pub title: Option<String>,
    pub doc_type: DocumentType,
    pub url: Option<String>,
    pub word_count: i32,
    pub source_path: Option<String>,
}

pub mod audio;
pub mod csv;
pub mod docx;
pub mod image;
pub mod pptx;
pub mod video;
pub mod xlsx;

pub use audio::AudioExtractor;
pub use csv::CsvExtractor;
pub use docx::DocxExtractor;
pub use image::ImageExtractor;
pub use pptx::PptxExtractor;
pub use video::VideoExtractor;
pub use xlsx::XlsxExtractor;
