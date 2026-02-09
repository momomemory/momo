//! Search request/response DTOs for the v1 API.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::common::{Metadata, V1DocumentType};
use crate::models;

/// Search scope determines which indices to query.
///
/// Wire format: `"documents"`, `"memories"`, or `"hybrid"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub enum SearchScope {
    /// Search document chunks only.
    Documents,
    /// Search memories only.
    Memories,
    /// Search both documents and memories (default).
    Hybrid,
}

impl Default for SearchScope {
    fn default() -> Self {
        Self::Hybrid
    }
}

impl From<SearchScope> for models::SearchMode {
    fn from(scope: SearchScope) -> Self {
        match scope {
            SearchScope::Documents => models::SearchMode::Documents,
            SearchScope::Memories => models::SearchMode::Memories,
            SearchScope::Hybrid => models::SearchMode::Hybrid,
        }
    }
}

impl From<models::SearchMode> for SearchScope {
    fn from(mode: models::SearchMode) -> Self {
        match mode {
            models::SearchMode::Documents => SearchScope::Documents,
            models::SearchMode::Memories => SearchScope::Memories,
            models::SearchMode::Hybrid => SearchScope::Hybrid,
        }
    }
}

/// Flags controlling which data to include in search results.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct SearchIncludeFlags {
    /// Include document data in results (default: true).
    #[serde(default = "default_true")]
    pub documents: bool,
    /// Include individual chunks in document results (default: false).
    #[serde(default)]
    pub chunks: bool,
}

fn default_true() -> bool {
    true
}

impl Default for SearchIncludeFlags {
    fn default() -> Self {
        Self {
            documents: true,
            chunks: false,
        }
    }
}

/// Request body for `POST /v1/search`.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct SearchRequest {
    /// The search query string.
    pub q: String,
    /// Which indices to search.
    #[serde(default)]
    pub scope: SearchScope,
    /// Filter by container tags.
    pub container_tags: Option<Vec<String>>,
    /// Minimum similarity threshold (0.0–1.0).
    pub threshold: Option<f32>,
    /// Maximum number of results to return.
    pub limit: Option<u32>,
    /// Which data to include in results.
    #[serde(default)]
    pub include: SearchIncludeFlags,
    /// Enable cross-encoder reranking.
    pub rerank: Option<bool>,
}

/// Unified search response for `POST /v1/search`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct SearchResponse {
    /// Combined search results (documents and/or memories based on scope).
    pub results: Vec<SearchResultItem>,
    /// Total number of matching results.
    pub total: u32,
    /// Query execution time in milliseconds.
    pub timing_ms: u64,
}

/// A single item in the search results — can be a document or memory hit.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "type")]
pub enum SearchResultItem {
    /// A document search result.
    #[serde(rename = "document")]
    Document(DocumentSearchResult),
    /// A memory search result.
    #[serde(rename = "memory")]
    Memory(MemorySearchResult),
}

/// Document match within search results.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSearchResult {
    /// The matched document ID.
    pub document_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_type: Option<V1DocumentType>,
    /// Vector similarity score.
    pub score: f32,
    /// Reranking score (if reranking was applied).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f32>,
    /// Matched chunks (if `include.chunks` was true).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<ChunkResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[schema(value_type = Object)]
    pub metadata: Metadata,
    #[schema(value_type = String)]
    pub created_at: DateTime<Utc>,
    #[schema(value_type = String)]
    pub updated_at: DateTime<Utc>,
}

/// A chunk within a document search result.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ChunkResult {
    pub content: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f32>,
}

impl From<models::ChunkSearchResult> for ChunkResult {
    fn from(chunk: models::ChunkSearchResult) -> Self {
        Self {
            content: chunk.content,
            score: chunk.score,
            rerank_score: chunk.rerank_score,
        }
    }
}

impl From<models::DocumentSearchResult> for DocumentSearchResult {
    fn from(doc: models::DocumentSearchResult) -> Self {
        Self {
            document_id: doc.document_id,
            title: doc.title,
            doc_type: doc.doc_type.map(Into::into),
            score: doc.score,
            rerank_score: doc.rerank_score,
            chunks: doc.chunks.into_iter().map(Into::into).collect(),
            summary: doc.summary,
            content: doc.content,
            metadata: doc.metadata,
            created_at: doc.created_at,
            updated_at: doc.updated_at,
        }
    }
}

/// Memory match within search results.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemorySearchResult {
    /// The matched memory ID.
    pub memory_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Vector similarity score.
    pub similarity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<i32>,
    #[schema(value_type = Object)]
    pub metadata: Metadata,
    #[schema(value_type = String)]
    pub updated_at: DateTime<Utc>,
}

impl From<models::MemorySearchResult> for MemorySearchResult {
    fn from(mem: models::MemorySearchResult) -> Self {
        Self {
            memory_id: mem.id,
            content: mem.memory,
            similarity: mem.similarity,
            rerank_score: mem.rerank_score,
            version: mem.version,
            metadata: mem.metadata,
            updated_at: mem.updated_at,
        }
    }
}

/// Hybrid search result — can represent either a memory or a chunk.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct HybridSearchResultResponse {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,
    pub similarity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f32>,
    #[schema(value_type = Object)]
    pub metadata: Metadata,
    #[schema(value_type = String)]
    pub updated_at: DateTime<Utc>,
}

impl From<models::HybridSearchResult> for HybridSearchResultResponse {
    fn from(result: models::HybridSearchResult) -> Self {
        Self {
            id: result.id,
            memory: result.memory,
            chunk: result.chunk,
            document_id: result.document_id,
            similarity: result.similarity,
            rerank_score: result.rerank_score,
            metadata: result.metadata,
            updated_at: result.updated_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_scope_default_is_hybrid() {
        assert_eq!(SearchScope::default(), SearchScope::Hybrid);
    }

    #[test]
    fn search_scope_roundtrip() {
        let mode: models::SearchMode = SearchScope::Documents.into();
        let scope: SearchScope = mode.into();
        assert_eq!(scope, SearchScope::Documents);
    }

    #[test]
    fn search_request_deserializes_with_defaults() {
        let json = r#"{"q": "test query"}"#;
        let req: SearchRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.q, "test query");
        assert_eq!(req.scope, SearchScope::Hybrid);
        assert!(req.include.documents);
        assert!(!req.include.chunks);
    }

    #[test]
    fn search_result_item_tagged_correctly() {
        let item = SearchResultItem::Memory(MemorySearchResult {
            memory_id: "mem_1".to_string(),
            content: Some("test".to_string()),
            similarity: 0.9,
            rerank_score: None,
            version: Some(1),
            metadata: std::collections::HashMap::new(),
            updated_at: chrono::Utc::now(),
        });

        let json = serde_json::to_value(&item).expect("serialize");
        assert_eq!(json.get("type").expect("type field"), "memory");
        assert!(json.get("memoryId").is_some());
    }
}
