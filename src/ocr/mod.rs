//! OCR (Optical Character Recognition) Module
//!
//! This module provides image text extraction capabilities for the Momo memory system.
//! It supports both local OCR engines (like Tesseract) and cloud-based OCR services.
//!
//! # Architecture
//!
//! The OCR module follows a provider pattern similar to the embeddings module:
//! - `OcrProvider` trait defines the interface
//! - `TesseractProvider` implements local OCR via leptess
//! - `ApiProvider` implements cloud OCR via HTTP APIs
//!
//! # Configuration
//!
//! OCR behavior is controlled via `OcrConfig` (see `config.rs`):
//! - `model`: Provider/model selection (e.g., "local/tesseract", "openai/gpt-4o")
//! - `api_key`: Authentication for cloud providers
//! - `base_url`: Custom endpoint for self-hosted or proxy setups
//! - `languages`: Comma-separated ISO 639-2 language codes
//! - `timeout_secs`: Request timeout for API calls
//! - `max/min_image_dimension`: Size limits for input validation
//!
//! # Usage
//!
//! ```rust,ignore
//! let ocr = OcrProvider::new(&config.ocr)?;
//! let text = ocr.ocr(image_bytes).await?;
//! ```

mod api;
mod preprocessing;
mod provider;

pub use preprocessing::preprocess_image;
pub use provider::OcrProvider;
