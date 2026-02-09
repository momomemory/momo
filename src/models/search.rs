use chrono::{DateTime, Utc};
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};

use super::{ChunkSearchResult, DocumentType, MemoryContext, Metadata};

/// Mode for hybrid search to determine which indices to query
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Search both documents and memories (default)
    #[default]
    Hybrid,
    /// Search documents only
    Documents,
    /// Search memories only
    Memories,
}

impl<'de> Deserialize<'de> for SearchMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        match raw.to_lowercase().as_str() {
            "hybrid" => Ok(SearchMode::Hybrid),
            "documents" => Ok(SearchMode::Documents),
            "memories" => Ok(SearchMode::Memories),
            _ => Err(de::Error::custom(format!(
                "Invalid searchMode '{raw}'. Valid modes: hybrid, documents, memories"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchDocumentsRequest {
    pub q: String,
    pub container_tags: Option<Vec<String>>,
    pub chunk_threshold: Option<f32>,
    pub document_threshold: Option<f32>,
    pub doc_id: Option<String>,
    pub filters: Option<SearchFilters>,
    pub include_full_docs: Option<bool>,
    pub include_summary: Option<bool>,
    pub limit: Option<u32>,
    pub only_matching_chunks: Option<bool>,
    pub rerank: Option<bool>,
    pub rerank_level: Option<String>,
    pub rerank_top_k: Option<usize>,
    #[serde(rename = "rewriteQuery")]
    pub rewrite_query: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilters {
    #[serde(rename = "AND")]
    pub and: Option<Vec<FilterCondition>>,
    #[serde(rename = "OR")]
    pub or: Option<Vec<FilterCondition>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub key: String,
    pub value: String,
    pub negate: Option<bool>,
    pub filter_type: Option<String>,
    pub numeric_operator: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchDocumentsResponse {
    pub results: Vec<DocumentSearchResult>,
    pub total: u32,
    pub timing: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rewritten_query: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSearchResult {
    pub document_id: String,
    pub title: Option<String>,
    #[serde(rename = "type")]
    pub doc_type: Option<DocumentType>,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f32>,
    pub chunks: Vec<ChunkSearchResult>,
    pub summary: Option<String>,
    pub content: Option<String>,
    pub metadata: Metadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchMemoriesRequest {
    pub q: String,
    pub container_tag: Option<String>,
    pub threshold: Option<f32>,
    pub filters: Option<SearchFilters>,
    pub include: Option<SearchIncludeOptions>,
    pub limit: Option<u32>,
    pub rerank: Option<bool>,
    #[serde(rename = "rewriteQuery")]
    pub rewrite_query: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HybridSearchRequest {
    pub q: String,
    pub container_tag: Option<String>,
    pub threshold: Option<f32>,
    pub filters: Option<SearchFilters>,
    pub include: Option<SearchIncludeOptions>,
    pub limit: Option<u32>,
    pub rerank: Option<bool>,
    #[serde(rename = "rewriteQuery")]
    pub rewrite_query: Option<bool>,
    #[serde(default)]
    #[serde(rename = "searchMode", alias = "search_mode")]
    pub search_mode: SearchMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchIncludeOptions {
    /// Include documents related to the memory in the response.
    pub documents: Option<bool>,
    /// Include summaries for matched memories.
    pub summaries: Option<bool>,
    /// Include related memories in the response.
    pub related_memories: Option<bool>,
    /// Include forgotten memories (soft-deleted/forgotten).
    ///
    /// JSON name: `forgottenMemories`.
    /// When omitted or false, forgotten memories are excluded. Treat `None` as `false`.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "forgottenMemories")]
    pub forgotten_memories: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::{
        DocumentSearchResult, HybridSearchResult, MemorySearchResult, SearchIncludeOptions,
        SearchMode,
    };
    use chrono::Utc;
    use serde_json::{from_value, json, to_value};
    use std::collections::HashMap;

    #[test]
    fn test_search_mode_serialization_lowercase() {
        let hybrid = SearchMode::Hybrid;
        let documents = SearchMode::Documents;
        let memories = SearchMode::Memories;

        assert_eq!(to_value(hybrid).unwrap(), json!("hybrid"));
        assert_eq!(to_value(documents).unwrap(), json!("documents"));
        assert_eq!(to_value(memories).unwrap(), json!("memories"));
    }

    #[test]
    fn test_search_mode_deserialization_case_insensitive() {
        assert_eq!(
            from_value::<SearchMode>(json!("hybrid")).unwrap(),
            SearchMode::Hybrid
        );
        assert_eq!(
            from_value::<SearchMode>(json!("DOCUMENTS")).unwrap(),
            SearchMode::Documents
        );
        assert_eq!(
            from_value::<SearchMode>(json!("Memories")).unwrap(),
            SearchMode::Memories
        );
    }

    #[test]
    fn test_search_mode_deserialization_rejects_invalid() {
        let err = from_value::<SearchMode>(json!("invalid"))
            .expect_err("expected invalid search mode to fail");
        let message = err.to_string();
        assert!(message.contains("Invalid searchMode"));
        assert!(message.contains("hybrid"));
        assert!(message.contains("documents"));
        assert!(message.contains("memories"));
    }

    #[test]
    fn test_search_mode_default_is_hybrid() {
        let default_mode = SearchMode::default();
        assert_eq!(default_mode, SearchMode::Hybrid);
    }

    #[test]
    fn test_search_include_options_serializes_with_forgotten_memories() {
        let opts = SearchIncludeOptions {
            documents: Some(true),
            summaries: Some(false),
            related_memories: Some(true),
            forgotten_memories: Some(true),
        };

        let v = to_value(&opts).expect("serialize");
        assert!(v.get("forgottenMemories").is_some());
        assert_eq!(v.get("forgottenMemories").unwrap(), &json!(true));
    }

    #[test]
    fn test_search_include_options_defaults_forgotten_to_false() {
        let opts = SearchIncludeOptions {
            documents: None,
            summaries: None,
            related_memories: None,
            forgotten_memories: None,
        };

        let v = to_value(&opts).expect("serialize");
        assert!(v.get("forgottenMemories").is_none());

        let from_false: SearchIncludeOptions =
            from_value(json!({ "forgottenMemories": false })).expect("deserialize");
        assert_eq!(from_false.forgotten_memories, Some(false));

        let from_absent: SearchIncludeOptions = from_value(json!({})).expect("deserialize");
        assert_eq!(from_absent.forgotten_memories, None);
    }

    #[test]
    fn test_document_search_result_with_rerank_score() {
        let result = DocumentSearchResult {
            document_id: "doc_123".to_string(),
            title: Some("Test Document".to_string()),
            doc_type: None,
            score: 0.85,
            rerank_score: Some(0.92),
            chunks: vec![],
            summary: None,
            content: None,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let v = to_value(&result).expect("serialize");
        let score = v.get("score").unwrap().as_f64().unwrap();
        assert!(
            (score - 0.85).abs() < 0.001,
            "score should be approximately 0.85, got {score}"
        );
        let rerank_score = v.get("rerank_score").unwrap().as_f64().unwrap();
        assert!(
            (rerank_score - 0.92).abs() < 0.001,
            "rerank_score should be approximately 0.92, got {rerank_score}"
        );
    }

    #[test]
    fn test_document_search_result_without_rerank_score() {
        let result = DocumentSearchResult {
            document_id: "doc_456".to_string(),
            title: Some("Another Document".to_string()),
            doc_type: None,
            score: 0.75,
            rerank_score: None,
            chunks: vec![],
            summary: None,
            content: None,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let v = to_value(&result).expect("serialize");
        let score = v.get("score").unwrap().as_f64().unwrap();
        assert!(
            (score - 0.75).abs() < 0.001,
            "score should be approximately 0.75, got {score}"
        );
        assert!(v.get("rerank_score").is_none());
    }

    #[test]
    fn test_memory_search_result_with_rerank_score() {
        let result = MemorySearchResult {
            id: "mem_123".to_string(),
            memory: Some("Test memory".to_string()),
            chunk: None,
            metadata: HashMap::new(),
            similarity: 0.88,
            rerank_score: Some(0.95),
            version: Some(1),
            updated_at: Utc::now(),
            context: None,
            documents: None,
        };

        let v = to_value(&result).expect("serialize");
        let similarity = v.get("similarity").unwrap().as_f64().unwrap();
        assert!(
            (similarity - 0.88).abs() < 0.001,
            "similarity should be approximately 0.88, got {similarity}"
        );
        let rerank_score = v.get("rerank_score").unwrap().as_f64().unwrap();
        assert!(
            (rerank_score - 0.95).abs() < 0.001,
            "rerank_score should be approximately 0.95, got {rerank_score}"
        );
    }

    #[test]
    fn test_memory_search_result_without_rerank_score() {
        let result = MemorySearchResult {
            id: "mem_456".to_string(),
            memory: Some("Another memory".to_string()),
            chunk: None,
            metadata: HashMap::new(),
            similarity: 0.72,
            rerank_score: None,
            version: Some(2),
            updated_at: Utc::now(),
            context: None,
            documents: None,
        };

        let v = to_value(&result).expect("serialize");
        let similarity = v.get("similarity").unwrap().as_f64().unwrap();
        assert!(
            (similarity - 0.72).abs() < 0.001,
            "similarity should be approximately 0.72, got {similarity}"
        );
        assert!(v.get("rerank_score").is_none());
    }

    #[test]
    fn test_hybrid_result_serialization_with_memory_only() {
        let now = Utc::now();
        let result = HybridSearchResult {
            id: "mem_123".to_string(),
            memory: Some("This is a memory".to_string()),
            chunk: None,
            document_id: None,
            similarity: 0.85,
            rerank_score: Some(0.92),
            metadata: HashMap::new(),
            updated_at: now,
        };

        let v = to_value(&result).expect("serialize");

        assert_eq!(v.get("id").unwrap().as_str().unwrap(), "mem_123");
        assert_eq!(
            v.get("memory").unwrap().as_str().unwrap(),
            "This is a memory"
        );
        assert!(v.get("chunk").is_none());
        assert!(v.get("documentId").is_none());

        let similarity = v.get("similarity").unwrap().as_f64().unwrap();
        assert!((similarity - 0.85).abs() < 0.001);

        let rerank_score = v.get("rerankScore").unwrap().as_f64().unwrap();
        assert!((rerank_score - 0.92).abs() < 0.001);
    }

    #[test]
    fn test_hybrid_result_serialization_with_chunk_only() {
        let now = Utc::now();
        let result = HybridSearchResult {
            id: "chunk_456".to_string(),
            memory: None,
            chunk: Some("This is a chunk".to_string()),
            document_id: Some("doc_789".to_string()),
            similarity: 0.78,
            rerank_score: None,
            metadata: HashMap::new(),
            updated_at: now,
        };

        let v = to_value(&result).expect("serialize");

        assert_eq!(v.get("id").unwrap().as_str().unwrap(), "chunk_456");
        assert!(v.get("memory").is_none());
        assert_eq!(v.get("chunk").unwrap().as_str().unwrap(), "This is a chunk");
        assert_eq!(v.get("documentId").unwrap().as_str().unwrap(), "doc_789");

        let similarity = v.get("similarity").unwrap().as_f64().unwrap();
        assert!((similarity - 0.78).abs() < 0.001);

        assert!(v.get("rerankScore").is_none());
    }

    #[test]
    fn test_hybrid_result_optional_fields_skipped_when_none() {
        let now = Utc::now();
        let result = HybridSearchResult {
            id: "test_001".to_string(),
            memory: Some("test".to_string()),
            chunk: None,
            document_id: None,
            similarity: 0.5,
            rerank_score: None,
            metadata: HashMap::new(),
            updated_at: now,
        };

        let json = serde_json::to_string(&result).expect("serialize to JSON string");

        assert!(!json.contains("\"chunk\""));
        assert!(!json.contains("\"documentId\""));
        assert!(!json.contains("\"rerankScore\""));
        assert!(json.contains("\"memory\""));
        assert!(json.contains("\"similarity\""));
    }

    #[test]
    fn test_hybrid_result_camel_case_field_names() {
        let now = Utc::now();
        let result = HybridSearchResult {
            id: "test_002".to_string(),
            memory: None,
            chunk: Some("content".to_string()),
            document_id: Some("doc_001".to_string()),
            similarity: 0.9,
            rerank_score: Some(0.95),
            metadata: HashMap::new(),
            updated_at: now,
        };

        let v = to_value(&result).expect("serialize");

        assert!(v.get("documentId").is_some());
        assert!(v.get("document_id").is_none());

        assert!(v.get("rerankScore").is_some());
        assert!(v.get("rerank_score").is_none());

        assert!(v.get("updatedAt").is_some());
        assert!(v.get("updated_at").is_none());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMemoriesResponse {
    pub results: Vec<MemorySearchResult>,
    pub total: u32,
    pub timing: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rewritten_query: Option<String>,
}

/// Unified result type for hybrid search that can represent either a memory or a document chunk.
/// Unified result type where `memory` and `chunk` are mutually exclusive.
///
/// - When representing a memory: `memory` is Some, `chunk` is None, `document_id` is None
/// - When representing a chunk: `chunk` is Some, `memory` is None, `document_id` is Some
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult {
    /// Unique identifier (memory ID or chunk ID)
    pub id: String,

    /// Memory content (mutually exclusive with chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<String>,

    /// Chunk content (mutually exclusive with memory)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk: Option<String>,

    /// Document ID (only present when chunk is present)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "documentId")]
    pub document_id: Option<String>,

    /// Vector similarity score
    pub similarity: f32,

    /// Optional reranking score (if reranking was applied)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "rerankScore")]
    pub rerank_score: Option<f32>,

    /// Associated metadata
    pub metadata: Metadata,

    /// Last updated timestamp
    #[serde(rename = "updatedAt")]
    pub updated_at: DateTime<Utc>,
}

/// Enum wrapper for backward compatibility with existing code that expects
/// either a DocumentSearchResult or MemorySearchResult.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum HybridSearchResultVariant {
    Document(DocumentSearchResult),
    Memory(MemorySearchResult),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResponse {
    pub results: Vec<HybridSearchResult>,
    pub total: u32,
    pub timing: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rewritten_query: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult {
    pub id: String,
    pub memory: Option<String>,
    pub chunk: Option<String>,
    pub metadata: Metadata,
    pub similarity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f32>,
    pub version: Option<i32>,
    pub updated_at: DateTime<Utc>,
    pub context: Option<MemoryContext>,
    pub documents: Option<Vec<RelatedDocument>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedDocument {
    pub id: String,
    pub title: Option<String>,
    #[serde(rename = "type")]
    pub doc_type: Option<DocumentType>,
    pub metadata: Metadata,
    pub summary: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
