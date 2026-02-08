use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{DocumentType, Metadata, ProcessingStatus};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub custom_id: Option<String>,
    pub connection_id: Option<String>,
    pub title: Option<String>,
    pub content: Option<String>,
    pub summary: Option<String>,
    pub url: Option<String>,
    pub source: Option<String>,
    #[serde(rename = "type")]
    pub doc_type: DocumentType,
    pub status: ProcessingStatus,
    pub metadata: Metadata,
    pub container_tags: Vec<String>,
    pub chunk_count: i32,
    pub token_count: Option<i32>,
    pub word_count: Option<i32>,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Document {
    pub fn new(id: String) -> Self {
        let now = Utc::now();
        Self {
            id,
            custom_id: None,
            connection_id: None,
            title: None,
            content: None,
            summary: None,
            url: None,
            source: None,
            doc_type: DocumentType::default(),
            status: ProcessingStatus::default(),
            metadata: Metadata::new(),
            container_tags: Vec::new(),
            chunk_count: 0,
            token_count: None,
            word_count: None,
            error_message: None,
            created_at: now,
            updated_at: now,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CreateDocumentRequest {
    #[validate(length(min = 1, max = 10_000_000))]
    pub content: String,
    pub container_tag: Option<String>,
    pub custom_id: Option<String>,
    #[validate(length(max = 1500))]
    pub entity_context: Option<String>,
    pub metadata: Option<Metadata>,
    pub source_path: Option<String>,
    /// Content type hint for base64-encoded files: "docx", "xlsx", "pptx", "csv", "pdf"
    pub content_type: Option<String>,
    /// When true, extract memories from document content after processing
    #[serde(default)]
    pub extract_memories: Option<bool>,
    /// When true, apply LLM filtering to document content
    #[serde(default)]
    pub should_llm_filter: Option<bool>,
    /// Custom LLM prompt for filtering this document (max 1000 chars)
    #[validate(length(max = 1000))]
    pub filter_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct BatchCreateDocumentRequest {
    #[validate(length(min = 1, max = 600))]
    pub documents: Vec<BatchDocumentItem>,
    pub container_tag: Option<String>,
    pub metadata: Option<Metadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct BatchDocumentItem {
    #[validate(length(min = 1))]
    pub content: String,
    pub custom_id: Option<String>,
    pub metadata: Option<Metadata>,
    pub source_path: Option<String>,
    /// When true, extract memories from document content after processing
    #[serde(default)]
    pub extract_memories: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateDocumentResponse {
    pub id: String,
    pub status: ProcessingStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateDocumentRequest {
    pub title: Option<String>,
    pub metadata: Option<Metadata>,
    pub container_tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListDocumentsRequest {
    pub container_tags: Option<Vec<String>>,
    pub filters: Option<String>,
    pub limit: Option<u32>,
    pub page: Option<u32>,
    pub order: Option<String>,
    pub sort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListDocumentsResponse {
    pub documents: Vec<DocumentSummary>,
    pub pagination: super::Pagination,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCreateDocumentResponse {
    pub documents: Vec<CreateDocumentResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSummary {
    pub id: String,
    pub custom_id: Option<String>,
    pub title: Option<String>,
    #[serde(rename = "type")]
    pub doc_type: DocumentType,
    pub status: ProcessingStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: Metadata,
    pub container_tags: Vec<String>,
}

impl From<Document> for DocumentSummary {
    fn from(doc: Document) -> Self {
        Self {
            id: doc.id,
            custom_id: doc.custom_id,
            title: doc.title,
            doc_type: doc.doc_type,
            status: doc.status,
            created_at: doc.created_at,
            updated_at: doc.updated_at,
            metadata: doc.metadata,
            container_tags: doc.container_tags,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingDocument {
    pub id: String,
    pub status: ProcessingStatus,
    pub title: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_document_request_with_extract_memories_true() {
        let json = r#"{"content": "test content", "extract_memories": true}"#;
        let request: CreateDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.extract_memories, Some(true));
        assert_eq!(request.content, "test content");
    }

    #[test]
    fn test_create_document_request_with_extract_memories_false() {
        let json = r#"{"content": "test content", "extract_memories": false}"#;
        let request: CreateDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.extract_memories, Some(false));
    }

    #[test]
    fn test_create_document_request_without_extract_memories() {
        let json = r#"{"content": "test content"}"#;
        let request: CreateDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.extract_memories, None);
    }

    #[test]
    fn test_batch_document_item_with_extract_memories_true() {
        let json = r#"{"content": "batch test", "extract_memories": true}"#;
        let item: BatchDocumentItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.extract_memories, Some(true));
    }

    #[test]
    fn test_batch_document_item_without_extract_memories() {
        let json = r#"{"content": "batch test"}"#;
        let item: BatchDocumentItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.extract_memories, None);
    }

    #[test]
    fn test_create_document_request_with_filter_settings() {
        let json =
            r#"{"content": "test", "should_llm_filter": true, "filter_prompt": "only tech docs"}"#;
        let request: CreateDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.should_llm_filter, Some(true));
        assert_eq!(request.filter_prompt, Some("only tech docs".to_string()));
    }

    #[test]
    fn test_create_document_request_without_filter_settings() {
        let json = r#"{"content": "test"}"#;
        let request: CreateDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.should_llm_filter, None);
        assert_eq!(request.filter_prompt, None);
    }

    #[test]
    fn test_create_document_request_filter_prompt_max_length() {
        let long_prompt = "a".repeat(1000);
        let json = format!(
            r#"{{"content": "test", "filter_prompt": "{long_prompt}"}}"#
        );
        let request: CreateDocumentRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request.filter_prompt, Some(long_prompt));
    }

    #[test]
    fn test_create_document_request_filter_disabled_with_prompt() {
        let json = r#"{"content": "test", "should_llm_filter": false, "filter_prompt": "prompt"}"#;
        let request: CreateDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.should_llm_filter, Some(false));
        assert_eq!(request.filter_prompt, Some("prompt".to_string()));
    }

    #[test]
    fn test_create_document_request_filter_prompt_exceeds_max_length() {
        let long_prompt = "a".repeat(1001);
        let json = format!(
            r#"{{"content": "test", "filter_prompt": "{long_prompt}"}}"#
        );
        let request: CreateDocumentRequest = serde_json::from_str(&json).unwrap();

        use validator::Validate;
        let validation_result = request.validate();
        assert!(validation_result.is_err());
    }

    #[test]
    fn test_create_document_request_filter_only_enabled() {
        let json = r#"{"content": "test", "should_llm_filter": true}"#;
        let request: CreateDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.should_llm_filter, Some(true));
        assert_eq!(request.filter_prompt, None);
    }

    #[test]
    fn test_create_document_request_filter_only_prompt() {
        let json = r#"{"content": "test", "filter_prompt": "technical docs only"}"#;
        let request: CreateDocumentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.should_llm_filter, None);
        assert_eq!(
            request.filter_prompt,
            Some("technical docs only".to_string())
        );
    }
}
