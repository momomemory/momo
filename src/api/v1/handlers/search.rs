//! v1 Search handler.
//!
//! Implements `POST /api/v1/search` with unified `scope` parameter to
//! search documents, memories, or both (hybrid).

use axum::extract::State;
use std::time::Instant;

use crate::api::v1::dto::{
    DocumentSearchResult as V1DocumentSearchResult, HybridSearchResultResponse,
    MemorySearchResult as V1MemorySearchResult, SearchRequest, SearchResponse, SearchResultItem,
    SearchScope,
};
use crate::api::v1::response::{ApiError, ApiResponse};
use crate::api::AppState;
use crate::models::{
    HybridSearchRequest, SearchDocumentsRequest, SearchMemoriesRequest, SearchMode,
};

/// `POST /api/v1/search`
///
/// Unified search endpoint. Uses `scope` to determine which indices to query:
/// - `documents` → document chunk search
/// - `memories` → memory-only search
/// - `hybrid` (default) → both documents and memories, deduplicated
#[utoipa::path(
    post,
    path = "/api/v1/search",
    tag = "search",
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 400, description = "Invalid request", body = ApiError),
    )
)]
pub async fn search(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<SearchRequest>,
) -> ApiResponse<SearchResponse> {
    // Validate query
    if req.q.trim().is_empty() {
        return ApiResponse::error(
            crate::api::v1::response::ErrorCode::InvalidRequest,
            "Query cannot be empty",
        );
    }

    let start = Instant::now();

    match req.scope {
        SearchScope::Documents => search_documents(&state, &req, start).await,
        SearchScope::Memories => search_memories(&state, &req, start).await,
        SearchScope::Hybrid => search_hybrid(&state, &req, start).await,
    }
}

/// Document-scope search: delegates to `SearchService::search_documents`.
async fn search_documents(
    state: &AppState,
    req: &SearchRequest,
    start: Instant,
) -> ApiResponse<SearchResponse> {
    let internal_req = SearchDocumentsRequest {
        q: req.q.clone(),
        container_tags: req.container_tags.clone(),
        chunk_threshold: req.threshold,
        document_threshold: None,
        doc_id: None,
        filters: None,
        include_full_docs: Some(req.include.documents),
        include_summary: Some(req.include.documents),
        limit: req.limit,
        only_matching_chunks: Some(!req.include.chunks),
        rerank: req.rerank,
        rerank_level: None,
        rerank_top_k: None,
        rewrite_query: None,
    };

    let response = match state.search.search_documents(internal_req).await {
        Ok(resp) => resp,
        Err(e) => return ApiResponse::from(e),
    };

    let results: Vec<SearchResultItem> = response
        .results
        .into_iter()
        .map(|doc| SearchResultItem::Document(V1DocumentSearchResult::from(doc)))
        .collect();

    let total = results.len() as u32;
    let timing_ms = start.elapsed().as_millis() as u64;

    ApiResponse::success(SearchResponse {
        results,
        total,
        timing_ms,
    })
}

/// Memory-scope search: delegates to `SearchService::search_memories`.
async fn search_memories(
    state: &AppState,
    req: &SearchRequest,
    start: Instant,
) -> ApiResponse<SearchResponse> {
    // SearchMemoriesRequest uses singular container_tag; pick the first one.
    let container_tag = req
        .container_tags
        .as_ref()
        .and_then(|tags| tags.first().cloned());

    let internal_req = SearchMemoriesRequest {
        q: req.q.clone(),
        container_tag,
        threshold: req.threshold,
        filters: None,
        include: None,
        limit: req.limit,
        rerank: req.rerank,
        rewrite_query: None,
    };

    let response = match state.search.search_memories(internal_req).await {
        Ok(resp) => resp,
        Err(e) => return ApiResponse::from(e),
    };

    let results: Vec<SearchResultItem> = response
        .results
        .into_iter()
        .map(|mem| SearchResultItem::Memory(V1MemorySearchResult::from(mem)))
        .collect();

    let total = results.len() as u32;
    let timing_ms = start.elapsed().as_millis() as u64;

    ApiResponse::success(SearchResponse {
        results,
        total,
        timing_ms,
    })
}

/// Hybrid-scope search: delegates to `SearchService::search_hybrid`.
///
/// Hybrid results can be either memories or document chunks. We map them
/// into the tagged `SearchResultItem` enum based on which fields are populated.
async fn search_hybrid(
    state: &AppState,
    req: &SearchRequest,
    start: Instant,
) -> ApiResponse<SearchResponse> {
    let container_tag = req
        .container_tags
        .as_ref()
        .and_then(|tags| tags.first().cloned());

    let internal_req = HybridSearchRequest {
        q: req.q.clone(),
        container_tag,
        threshold: req.threshold,
        filters: None,
        include: None,
        limit: req.limit,
        rerank: req.rerank,
        rewrite_query: None,
        search_mode: SearchMode::Hybrid,
    };

    let response = match state.search.search_hybrid(internal_req).await {
        Ok(resp) => resp,
        Err(e) => return ApiResponse::from(e),
    };

    // Hybrid results are either memory-backed or chunk-backed.
    // We convert them to v1 DTOs based on which field is populated.
    let results: Vec<SearchResultItem> = response
        .results
        .into_iter()
        .map(|result| {
            let v1_result = HybridSearchResultResponse::from(result);
            // Memory-type results have `memory` set, chunk-type have `chunk` set.
            if v1_result.memory.is_some() {
                SearchResultItem::Memory(V1MemorySearchResult {
                    memory_id: v1_result.id,
                    content: v1_result.memory,
                    similarity: v1_result.similarity,
                    rerank_score: v1_result.rerank_score,
                    version: None,
                    metadata: v1_result.metadata,
                    updated_at: v1_result.updated_at,
                })
            } else {
                SearchResultItem::Document(V1DocumentSearchResult {
                    document_id: v1_result.document_id.unwrap_or_default(),
                    title: None,
                    doc_type: None,
                    score: v1_result.similarity,
                    rerank_score: v1_result.rerank_score,
                    chunks: if v1_result.chunk.is_some() {
                        vec![crate::api::v1::dto::ChunkResult {
                            content: v1_result.chunk.unwrap_or_default(),
                            score: v1_result.similarity,
                            rerank_score: v1_result.rerank_score,
                        }]
                    } else {
                        vec![]
                    },
                    summary: None,
                    content: None,
                    metadata: v1_result.metadata,
                    created_at: v1_result.updated_at,
                    updated_at: v1_result.updated_at,
                })
            }
        })
        .collect();

    let total = results.len() as u32;
    let timing_ms = start.elapsed().as_millis() as u64;

    ApiResponse::success(SearchResponse {
        results,
        total,
        timing_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_request_defaults_to_hybrid_scope() {
        let json = r#"{"q": "test query"}"#;
        let req: SearchRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.scope, SearchScope::Hybrid);
    }

    #[test]
    fn search_request_with_documents_scope() {
        let json = r#"{"q": "test", "scope": "documents"}"#;
        let req: SearchRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.scope, SearchScope::Documents);
    }

    #[test]
    fn search_request_with_memories_scope() {
        let json = r#"{"q": "test", "scope": "memories"}"#;
        let req: SearchRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.scope, SearchScope::Memories);
    }

    #[test]
    fn search_request_include_flags_defaults() {
        let json = r#"{"q": "test"}"#;
        let req: SearchRequest = serde_json::from_str(json).expect("deserialize");
        assert!(req.include.documents);
        assert!(!req.include.chunks);
        assert!(req.include.memories);
        assert!(!req.include.graph_context);
    }

    #[test]
    fn search_request_with_all_fields() {
        let json = r#"{
            "q": "test query",
            "scope": "hybrid",
            "containerTags": ["user_123", "project_abc"],
            "threshold": 0.7,
            "limit": 25,
            "include": {
                "documents": true,
                "chunks": true,
                "memories": false,
                "graphContext": true
            },
            "rerank": true
        }"#;
        let req: SearchRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.q, "test query");
        assert_eq!(req.scope, SearchScope::Hybrid);
        assert_eq!(
            req.container_tags.as_ref().map(|t| t.len()),
            Some(2)
        );
        assert_eq!(req.threshold, Some(0.7));
        assert_eq!(req.limit, Some(25));
        assert!(req.include.documents);
        assert!(req.include.chunks);
        assert!(!req.include.memories);
        assert!(req.include.graph_context);
        assert_eq!(req.rerank, Some(true));
    }

    #[test]
    fn search_scope_converts_to_search_mode() {
        let mode: SearchMode = SearchScope::Documents.into();
        assert_eq!(mode, SearchMode::Documents);

        let mode: SearchMode = SearchScope::Memories.into();
        assert_eq!(mode, SearchMode::Memories);

        let mode: SearchMode = SearchScope::Hybrid.into();
        assert_eq!(mode, SearchMode::Hybrid);
    }
}
