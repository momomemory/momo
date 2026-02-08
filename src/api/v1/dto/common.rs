//! Shared DTO types used across multiple v1 API endpoints.
//!
//! These types define the canonical wire format for common concepts like
//! ingestion status, document types, memory types, and metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::models::{DocumentType, MemoryType, ProcessingStatus};

/// Metadata type alias for v1 API — arbitrary key-value pairs.
///
/// Wire format: `{ "key": <any JSON value>, ... }`
pub type Metadata = HashMap<String, serde_json::Value>;

/// Simplified ingestion status for v1 API.
///
/// Maps from the internal `ProcessingStatus` (which has many intermediate
/// states) to four user-facing states.
///
/// Wire format: `"queued"`, `"processing"`, `"completed"`, or `"failed"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub enum IngestionStatus {
    /// The document is queued for processing.
    Queued,
    /// The document is actively being processed (extracting, chunking, embedding, or indexing).
    Processing,
    /// The document has been fully processed and indexed.
    Completed,
    /// Processing failed. Check `errorMessage` for details.
    Failed,
}

impl From<ProcessingStatus> for IngestionStatus {
    fn from(status: ProcessingStatus) -> Self {
        match status {
            ProcessingStatus::Unknown | ProcessingStatus::Queued => IngestionStatus::Queued,
            ProcessingStatus::Extracting
            | ProcessingStatus::Chunking
            | ProcessingStatus::Embedding
            | ProcessingStatus::Indexing => IngestionStatus::Processing,
            ProcessingStatus::Done => IngestionStatus::Completed,
            ProcessingStatus::Failed => IngestionStatus::Failed,
        }
    }
}

/// Document type classification for v1 API.
///
/// Wire format: lowercase string (e.g. `"text"`, `"pdf"`, `"webpage"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub enum V1DocumentType {
    Text,
    Pdf,
    Webpage,
    Image,
    Video,
    Audio,
    Markdown,
    Code,
    Csv,
    Docx,
    Pptx,
    Xlsx,
    Unknown,
}

impl From<DocumentType> for V1DocumentType {
    fn from(dt: DocumentType) -> Self {
        match dt {
            DocumentType::Text => V1DocumentType::Text,
            DocumentType::Pdf => V1DocumentType::Pdf,
            DocumentType::Webpage | DocumentType::Tweet => V1DocumentType::Webpage,
            DocumentType::GoogleDoc
            | DocumentType::GoogleSlide
            | DocumentType::GoogleSheet
            | DocumentType::NotionDoc
            | DocumentType::Onedrive => V1DocumentType::Text,
            DocumentType::Image => V1DocumentType::Image,
            DocumentType::Video => V1DocumentType::Video,
            DocumentType::Audio => V1DocumentType::Audio,
            DocumentType::Markdown => V1DocumentType::Markdown,
            DocumentType::Code => V1DocumentType::Code,
            DocumentType::Csv => V1DocumentType::Csv,
            DocumentType::Docx => V1DocumentType::Docx,
            DocumentType::Pptx => V1DocumentType::Pptx,
            DocumentType::Xlsx => V1DocumentType::Xlsx,
            DocumentType::Unknown => V1DocumentType::Unknown,
        }
    }
}

/// Memory type classification for v1 API.
///
/// Wire format: lowercase string (`"fact"`, `"preference"`, `"episode"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub enum V1MemoryType {
    /// Factual information about the user or topic.
    Fact,
    /// User preference or choice.
    Preference,
    /// Event or experience.
    Episode,
}

impl From<MemoryType> for V1MemoryType {
    fn from(mt: MemoryType) -> Self {
        match mt {
            MemoryType::Fact => V1MemoryType::Fact,
            MemoryType::Preference => V1MemoryType::Preference,
            MemoryType::Episode => V1MemoryType::Episode,
        }
    }
}

impl From<V1MemoryType> for MemoryType {
    fn from(mt: V1MemoryType) -> Self {
        match mt {
            V1MemoryType::Fact => MemoryType::Fact,
            V1MemoryType::Preference => MemoryType::Preference,
            V1MemoryType::Episode => MemoryType::Episode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingestion_status_unknown_maps_to_queued() {
        assert_eq!(
            IngestionStatus::from(ProcessingStatus::Unknown),
            IngestionStatus::Queued
        );
    }

    #[test]
    fn ingestion_status_intermediate_maps_to_processing() {
        assert_eq!(
            IngestionStatus::from(ProcessingStatus::Extracting),
            IngestionStatus::Processing
        );
        assert_eq!(
            IngestionStatus::from(ProcessingStatus::Chunking),
            IngestionStatus::Processing
        );
        assert_eq!(
            IngestionStatus::from(ProcessingStatus::Embedding),
            IngestionStatus::Processing
        );
        assert_eq!(
            IngestionStatus::from(ProcessingStatus::Indexing),
            IngestionStatus::Processing
        );
    }

    #[test]
    fn ingestion_status_done_maps_to_completed() {
        assert_eq!(
            IngestionStatus::from(ProcessingStatus::Done),
            IngestionStatus::Completed
        );
    }

    #[test]
    fn ingestion_status_failed_maps_to_failed() {
        assert_eq!(
            IngestionStatus::from(ProcessingStatus::Failed),
            IngestionStatus::Failed
        );
    }

    #[test]
    fn ingestion_status_serializes_camel_case() {
        assert_eq!(
            serde_json::to_value(IngestionStatus::Queued).unwrap(),
            serde_json::json!("queued")
        );
        assert_eq!(
            serde_json::to_value(IngestionStatus::Processing).unwrap(),
            serde_json::json!("processing")
        );
        assert_eq!(
            serde_json::to_value(IngestionStatus::Completed).unwrap(),
            serde_json::json!("completed")
        );
        assert_eq!(
            serde_json::to_value(IngestionStatus::Failed).unwrap(),
            serde_json::json!("failed")
        );
    }

    #[test]
    fn v1_memory_type_roundtrip() {
        let mt = MemoryType::Preference;
        let v1: V1MemoryType = mt.into();
        let back: MemoryType = v1.into();
        assert_eq!(mt, back);
    }
}
