//! Document request/response DTOs for the v1 API.
//!
//! These types define the wire format for document creation, retrieval,
//! listing, updating, batch operations, and ingestion status tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::common::{IngestionStatus, Metadata, V1DocumentType};
use crate::models;

// ---------------------------------------------------------------------------
// Request DTOs
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/documents`.
///
/// Creates a new document and queues it for async ingestion.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct CreateDocumentRequest {
    /// The document content (text, URL, or base64-encoded file).
    pub content: String,
    /// Optional container tag for multi-tenant isolation.
    pub container_tag: Option<String>,
    /// Client-provided identifier for idempotency or external linking.
    pub custom_id: Option<String>,
    /// Arbitrary key-value metadata attached to the document.
    #[schema(value_type = Object)]
    pub metadata: Option<Metadata>,
    /// Content type hint for base64-encoded files (e.g. `"pdf"`, `"docx"`).
    pub content_type: Option<String>,
    /// When `true`, extract memories from document content after processing.
    #[serde(default)]
    pub extract_memories: Option<bool>,
}

/// Request body for `POST /v1/documents/batch`.
///
/// Creates multiple documents in a single request.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct BatchCreateDocumentRequest {
    /// List of documents to create (1–600 items).
    pub documents: Vec<BatchDocumentItem>,
    /// Default container tag applied to all documents in the batch.
    pub container_tag: Option<String>,
    /// Default metadata applied to all documents in the batch.
    #[schema(value_type = Object)]
    pub metadata: Option<Metadata>,
}

/// A single document within a batch create request.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct BatchDocumentItem {
    /// The document content.
    pub content: String,
    /// Client-provided identifier for this document.
    pub custom_id: Option<String>,
    /// Metadata specific to this document (overrides batch-level metadata).
    #[schema(value_type = Object)]
    pub metadata: Option<Metadata>,
    /// When `true`, extract memories from this document after processing.
    #[serde(default)]
    pub extract_memories: Option<bool>,
}

/// Request body for `PATCH /v1/documents/{documentId}`.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct UpdateDocumentRequest {
    /// Updated document title.
    pub title: Option<String>,
    /// Updated metadata (replaces existing metadata).
    #[schema(value_type = Object)]
    pub metadata: Option<Metadata>,
    /// Updated container tags.
    pub container_tags: Option<Vec<String>>,
}

/// Query parameters for `GET /v1/documents` (list endpoint).
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema, utoipa::IntoParams)]
#[serde(rename_all = "camelCase")]
pub struct ListDocumentsQuery {
    /// Filter by container tags.
    pub container_tags: Option<Vec<String>>,
    /// Maximum results per page (default 20, max 100).
    pub limit: Option<u32>,
    /// Opaque cursor for pagination.
    pub cursor: Option<String>,
}

// ---------------------------------------------------------------------------
// Response DTOs
// ---------------------------------------------------------------------------

/// Response body for `POST /v1/documents` — async ingestion accepted.
///
/// Wire format:
/// ```json
/// { "documentId": "V1StGXR8_Z5jdHi6B-myT", "ingestionId": "550e8400-..." }
/// ```
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct CreateDocumentResponse {
    /// The assigned document ID (nanoid, 21 chars).
    pub document_id: String,
    /// The ingestion tracking ID (UUID v4).
    pub ingestion_id: String,
}

/// Response body for `POST /v1/documents/batch`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct BatchCreateDocumentResponse {
    /// Results for each document in the batch, in order.
    pub documents: Vec<CreateDocumentResponse>,
}

/// Full document response for `GET /v1/documents/{documentId}`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct DocumentResponse {
    /// Unique document ID (nanoid, 21 chars).
    pub document_id: String,
    /// Client-provided identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_id: Option<String>,
    /// Document title (may be auto-extracted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Document content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Auto-generated summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Source URL (if document was fetched from a URL).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Document type classification.
    pub doc_type: V1DocumentType,
    /// Current ingestion status.
    pub ingestion_status: IngestionStatus,
    /// Attached metadata.
    #[schema(value_type = Object)]
    pub metadata: Metadata,
    /// Container tags for multi-tenant isolation.
    pub container_tags: Vec<String>,
    /// Number of chunks created from this document.
    pub chunk_count: i32,
    /// Error message if ingestion failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// When the document was created.
    #[schema(value_type = String)]
    pub created_at: DateTime<Utc>,
    /// When the document was last updated.
    #[schema(value_type = String)]
    pub updated_at: DateTime<Utc>,
}

impl From<models::Document> for DocumentResponse {
    fn from(doc: models::Document) -> Self {
        Self {
            document_id: doc.id,
            custom_id: doc.custom_id,
            title: doc.title,
            content: doc.content,
            summary: doc.summary,
            url: doc.url,
            doc_type: doc.doc_type.into(),
            ingestion_status: doc.status.into(),
            metadata: doc.metadata,
            container_tags: doc.container_tags,
            chunk_count: doc.chunk_count,
            error_message: doc.error_message,
            created_at: doc.created_at,
            updated_at: doc.updated_at,
        }
    }
}

/// Summary document for list responses — lighter than full `DocumentResponse`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSummaryResponse {
    /// Unique document ID.
    pub document_id: String,
    /// Client-provided identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_id: Option<String>,
    /// Document title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Document type.
    pub doc_type: V1DocumentType,
    /// Current ingestion status.
    pub ingestion_status: IngestionStatus,
    /// Attached metadata.
    #[schema(value_type = Object)]
    pub metadata: Metadata,
    /// Container tags.
    pub container_tags: Vec<String>,
    /// When the document was created.
    #[schema(value_type = String)]
    pub created_at: DateTime<Utc>,
    /// When the document was last updated.
    #[schema(value_type = String)]
    pub updated_at: DateTime<Utc>,
}

impl From<models::DocumentSummary> for DocumentSummaryResponse {
    fn from(doc: models::DocumentSummary) -> Self {
        Self {
            document_id: doc.id,
            custom_id: doc.custom_id,
            title: doc.title,
            doc_type: doc.doc_type.into(),
            ingestion_status: doc.status.into(),
            metadata: doc.metadata,
            container_tags: doc.container_tags,
            created_at: doc.created_at,
            updated_at: doc.updated_at,
        }
    }
}

impl From<models::Document> for DocumentSummaryResponse {
    fn from(doc: models::Document) -> Self {
        Self {
            document_id: doc.id,
            custom_id: doc.custom_id,
            title: doc.title,
            doc_type: doc.doc_type.into(),
            ingestion_status: doc.status.into(),
            metadata: doc.metadata,
            container_tags: doc.container_tags,
            created_at: doc.created_at,
            updated_at: doc.updated_at,
        }
    }
}

/// Response wrapper for document list endpoints.
///
/// Pagination is handled by the envelope's `meta.nextCursor` / `meta.total`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ListDocumentsResponse {
    /// Documents matching the query.
    pub documents: Vec<DocumentSummaryResponse>,
}

/// Ingestion status for a single document (used in status polling).
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct IngestionStatusResponse {
    /// Document ID.
    pub document_id: String,
    /// Current ingestion status.
    pub status: IngestionStatus,
    /// Document title (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// When the document was created.
    #[schema(value_type = String)]
    pub created_at: DateTime<Utc>,
}

impl From<models::ProcessingDocument> for IngestionStatusResponse {
    fn from(doc: models::ProcessingDocument) -> Self {
        Self {
            document_id: doc.id,
            status: doc.status.into(),
            title: doc.title,
            created_at: doc.created_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Document, DocumentType, ProcessingStatus};

    #[test]
    fn document_response_from_domain() {
        let doc = Document {
            id: "abc123def456ghi78_j".to_string(),
            custom_id: Some("ext-1".to_string()),
            connection_id: None,
            title: Some("Test Doc".to_string()),
            content: Some("Hello world".to_string()),
            summary: None,
            url: None,
            source: None,
            doc_type: DocumentType::Pdf,
            status: ProcessingStatus::Extracting,
            metadata: std::collections::HashMap::new(),
            container_tags: vec!["user_1".to_string()],
            chunk_count: 5,
            token_count: Some(100),
            word_count: Some(50),
            error_message: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let resp: DocumentResponse = doc.into();
        assert_eq!(resp.document_id, "abc123def456ghi78_j");
        assert_eq!(resp.custom_id, Some("ext-1".to_string()));
        assert_eq!(resp.doc_type, V1DocumentType::Pdf);
        assert_eq!(resp.ingestion_status, IngestionStatus::Processing);
        assert_eq!(resp.chunk_count, 5);
    }

    #[test]
    fn document_response_serializes_camel_case() {
        let resp = DocumentResponse {
            document_id: "test_id".to_string(),
            custom_id: None,
            title: Some("Title".to_string()),
            content: None,
            summary: None,
            url: None,
            doc_type: V1DocumentType::Text,
            ingestion_status: IngestionStatus::Completed,
            metadata: std::collections::HashMap::new(),
            container_tags: vec![],
            chunk_count: 0,
            error_message: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let json = serde_json::to_value(&resp).expect("serialize");
        assert!(json.get("documentId").is_some());
        assert!(json.get("document_id").is_none());
        assert!(json.get("docType").is_some());
        assert!(json.get("ingestionStatus").is_some());
        assert!(json.get("containerTags").is_some());
        assert!(json.get("chunkCount").is_some());
        assert!(json.get("createdAt").is_some());
        // Optional None fields should be absent
        assert!(json.get("customId").is_none());
        assert!(json.get("content").is_none());
    }
}
