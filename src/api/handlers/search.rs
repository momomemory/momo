use axum::{extract::State, Json};

use crate::api::{AppJson, AppState};
use crate::error::{MomoError, Result};
use crate::models::{
    HybridSearchRequest, HybridSearchResponse, SearchDocumentsRequest, SearchDocumentsResponse,
    SearchMemoriesRequest, SearchMemoriesResponse, SearchMode,
};

pub async fn search_documents(
    State(state): State<AppState>,
    AppJson(req): AppJson<SearchDocumentsRequest>,
) -> Result<Json<SearchDocumentsResponse>> {
    let response = state.search.search_documents(req).await?;
    Ok(Json(response))
}

#[derive(serde::Serialize)]
#[serde(untagged)]
pub(crate) enum SearchResponse {
    Hybrid(HybridSearchResponse),
    Memories(SearchMemoriesResponse),
}

/// Search for memories or raw document chunks (hybrid mode).
///
/// This endpoint supports three search modes:
/// - `hybrid` (default): Searches both extracted memories and raw document chunks.
///   Results are deduplicated so that if a memory exists for a chunk, only the memory is returned.
/// - `memories`: Searches only the refined memory layer.
/// - `documents`: Redirects to document search (returns 400 - use /v3/search for document-only).
pub async fn search_memories(
    State(state): State<AppState>,
    AppJson(req): AppJson<HybridSearchRequest>,
) -> Result<Json<SearchResponse>> {
    match req.search_mode {
        SearchMode::Hybrid => {
            let response = state.search.search_hybrid(req).await?;
            Ok(Json(SearchResponse::Hybrid(response)))
        }
        SearchMode::Memories => {
            let response = state.search.search_memories(convert_request(req)).await?;
            Ok(Json(SearchResponse::Memories(response)))
        }
        SearchMode::Documents => Err(MomoError::Validation(
            "Use /v3/search for document-only search".to_string(),
        )),
    }
}

fn convert_request(req: HybridSearchRequest) -> SearchMemoriesRequest {
    SearchMemoriesRequest {
        q: req.q,
        container_tag: req.container_tag,
        threshold: req.threshold,
        filters: req.filters,
        include: req.include,
        limit: req.limit,
        rerank: req.rerank,
        rewrite_query: req.rewrite_query,
    }
}

// Legacy v3/v4 search handler integration tests removed.
//
// These tests previously hit `/v4/search` and `/v3/search` which were removed
// when the v1 API was introduced (Task 12). The v1 search handler has its own
// unit tests in `api::v1::handlers::search::tests` covering scope parsing,
// include flags, and field mappings. Router-level v1 invariant tests (auth,
// envelope shape, public routes) live in `api::v1::tests`.
#[cfg(test)]
mod tests {}
