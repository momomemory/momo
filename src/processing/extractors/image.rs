use super::ExtractedContent;
use crate::config::OcrConfig;
use crate::error::Result;
use crate::models::DocumentType;
use crate::ocr::{preprocess_image, OcrProvider};

pub struct ImageExtractor;

impl ImageExtractor {
    /// Extract text from an image using OCR
    ///
    /// # Arguments
    /// * `bytes` - Raw image bytes (PNG, JPEG, etc.)
    /// * `ocr_provider` - OCR provider instance for text extraction
    /// * `config` - OCR configuration for preprocessing
    ///
    /// # Returns
    /// ExtractedContent with extracted text and metadata
    pub async fn extract(
        bytes: &[u8],
        ocr_provider: &OcrProvider,
        config: &OcrConfig,
    ) -> Result<ExtractedContent> {
        let processed = preprocess_image(bytes, config)?;
        let text = ocr_provider.ocr(&processed).await?;
        let word_count = text.split_whitespace().count() as i32;

        Ok(ExtractedContent {
            text,
            title: None,
            doc_type: DocumentType::Image,
            url: None,
            word_count,
            source_path: None,
        })
    }

    /// Extract text from an image without preprocessing
    ///
    /// This is useful when the image has already been preprocessed
    /// or when preprocessing is not desired.
    ///
    /// # Arguments
    /// * `bytes` - Preprocessed image bytes
    /// * `ocr_provider` - OCR provider instance for text extraction
    ///
    /// # Returns
    /// ExtractedContent with extracted text and metadata
    pub async fn extract_raw(bytes: &[u8], ocr_provider: &OcrProvider) -> Result<ExtractedContent> {
        let text = ocr_provider.ocr(bytes).await?;
        let word_count = text.split_whitespace().count() as i32;

        Ok(ExtractedContent {
            text,
            title: None,
            doc_type: DocumentType::Image,
            url: None,
            word_count,
            source_path: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OcrConfig;
    use crate::ocr::OcrProvider;

    fn create_test_config() -> OcrConfig {
        OcrConfig {
            model: "local/tesseract".to_string(),
            api_key: None,
            base_url: None,
            languages: "eng".to_string(),
            timeout_secs: 60,
            max_image_dimension: 4096,
            min_image_dimension: 50,
        }
    }

    fn create_test_png(width: u32, height: u32) -> Vec<u8> {
        use image::{DynamicImage, ImageFormat};
        let img = DynamicImage::new_rgb8(width, height);
        let mut output = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut output), ImageFormat::Png)
            .unwrap();
        output
    }

    #[test]
    fn test_image_extractor_struct_exists() {
        let _ = ImageExtractor;
    }

    #[tokio::test]
    async fn test_extract_handles_zero_text_image() {
        let image_data = create_test_png(100, 100);
        let config = create_test_config();
        let ocr_provider = OcrProvider::new(&config).expect("Failed to create OCR provider");

        let result = ImageExtractor::extract(&image_data, &ocr_provider, &config).await;

        assert!(result.is_ok(), "Should handle zero-text images gracefully");
        let extracted = result.unwrap();
        assert_eq!(extracted.doc_type, DocumentType::Image);
        assert_eq!(extracted.word_count, 0, "Blank image should have 0 words");
        assert!(extracted.title.is_none());
        assert!(extracted.url.is_none());
    }

    #[tokio::test]
    async fn test_extract_raw_skips_preprocessing() {
        let image_data = create_test_png(100, 100);
        let config = create_test_config();
        let ocr_provider = OcrProvider::new(&config).expect("Failed to create OCR provider");

        let result = ImageExtractor::extract_raw(&image_data, &ocr_provider).await;

        assert!(result.is_ok(), "Should handle raw extraction");
        let extracted = result.unwrap();
        assert_eq!(extracted.doc_type, DocumentType::Image);
    }

    #[tokio::test]
    async fn test_extract_returns_error_for_invalid_image() {
        let invalid_data = vec![0u8, 1, 2, 3, 4, 5];
        let config = create_test_config();
        let ocr_provider = OcrProvider::new(&config).expect("Failed to create OCR provider");

        let result = ImageExtractor::extract(&invalid_data, &ocr_provider, &config).await;

        assert!(result.is_err(), "Should reject invalid image data");
    }

    #[tokio::test]
    async fn test_extract_returns_error_for_tiny_image() {
        let tiny_image = create_test_png(10, 10);
        let config = create_test_config();
        let ocr_provider = OcrProvider::new(&config).expect("Failed to create OCR provider");

        let result = ImageExtractor::extract(&tiny_image, &ocr_provider, &config).await;

        assert!(result.is_err(), "Should reject tiny images");
        let err_string = result.unwrap_err().to_string();
        assert!(
            err_string.contains("too small"),
            "Error should indicate image is too small: {err_string}"
        );
    }

    #[tokio::test]
    async fn test_extracted_content_structure() {
        let image_data = create_test_png(100, 100);
        let config = create_test_config();
        let ocr_provider = OcrProvider::new(&config).expect("Failed to create OCR provider");

        let result = ImageExtractor::extract(&image_data, &ocr_provider, &config).await;
        assert!(result.is_ok());

        let extracted = result.unwrap();
        assert_eq!(extracted.doc_type, DocumentType::Image);
        assert!(extracted.title.is_none());
        assert!(extracted.url.is_none());
        assert!(extracted.source_path.is_none());
        assert!(
            extracted.word_count >= 0,
            "Word count should be non-negative"
        );
    }
}
