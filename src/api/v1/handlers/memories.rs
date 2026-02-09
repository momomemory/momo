//! v1 Memory handlers.

use axum::extract::{Path, State};
use axum_extra::extract::Query;

use crate::api::v1::dto::{
    ContentForgetRequest, CreateMemoryRequest, ForgetMemoryRequest, ForgetMemoryResponse,
    ListMemoriesQuery, ListMemoriesResponse, MemoryResponse, UpdateMemoryRequest,
    UpdateMemoryResponse,
};
use crate::api::v1::response::{ApiError, ApiResponse, ErrorCode, ResponseMeta};
use crate::api::AppState;
use crate::models::MemoryType;

/// `POST /api/v1/memories`
#[utoipa::path(
    post,
    path = "/api/v1/memories",
    tag = "memories",
    operation_id = "memories.create",
    request_body = CreateMemoryRequest,
    responses(
        (status = 201, description = "Memory created", body = MemoryResponse),
        (status = 400, description = "Invalid request", body = ApiError),
    )
)]
pub async fn create_memory(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<CreateMemoryRequest>,
) -> ApiResponse<MemoryResponse> {
    if req.content.trim().is_empty() {
        return ApiResponse::error(ErrorCode::InvalidRequest, "Content cannot be empty");
    }

    if req.container_tag.trim().is_empty() {
        return ApiResponse::error(ErrorCode::InvalidRequest, "Container tag cannot be empty");
    }

    let memory_type: MemoryType = req.memory_type.map(Into::into).unwrap_or(MemoryType::Fact);

    let memory = match state
        .memory
        .create_memory_with_type(&req.content, &req.container_tag, false, memory_type)
        .await
    {
        Ok(mut mem) => {
            if let Some(metadata) = req.metadata {
                for (k, v) in metadata {
                    mem.metadata.insert(k, v);
                }
                if let Err(e) = state
                    .db
                    .update_memory_relations(&mem.id, mem.memory_relations.clone())
                    .await
                {
                    tracing::warn!(error = %e, "Failed to persist metadata on new memory");
                }
            }
            mem
        }
        Err(e) => return e.into(),
    };

    ApiResponse::created(MemoryResponse::from(memory))
}

/// `GET /api/v1/memories/{memoryId}`
#[utoipa::path(
    get,
    path = "/api/v1/memories/{memoryId}",
    tag = "memories",
    operation_id = "memories.get",
    params(("memoryId" = String, Path, description = "Memory ID")),
    responses(
        (status = 200, description = "Memory found", body = MemoryResponse),
        (status = 404, description = "Memory not found", body = ApiError),
    )
)]
pub async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResponse<MemoryResponse> {
    match state.db.get_memory_by_id(&id).await {
        Ok(Some(mem)) => ApiResponse::success(MemoryResponse::from(mem)),
        Ok(None) => ApiResponse::error(ErrorCode::NotFound, format!("Memory {id} not found")),
        Err(e) => e.into(),
    }
}

/// `PATCH /api/v1/memories/{memoryId}`
#[utoipa::path(
    patch,
    path = "/api/v1/memories/{memoryId}",
    tag = "memories",
    operation_id = "memories.update",
    params(("memoryId" = String, Path, description = "Memory ID")),
    request_body = UpdateMemoryRequest,
    responses(
        (status = 200, description = "Memory updated", body = UpdateMemoryResponse),
        (status = 404, description = "Memory not found", body = ApiError),
    )
)]
pub async fn update_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    axum::Json(req): axum::Json<UpdateMemoryRequest>,
) -> ApiResponse<UpdateMemoryResponse> {
    let existing = match state.db.get_memory_by_id(&id).await {
        Ok(Some(mem)) => mem,
        Ok(None) => {
            return ApiResponse::error(ErrorCode::NotFound, format!("Memory {id} not found"))
        }
        Err(e) => return e.into(),
    };

    let container_tag = existing
        .container_tag
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let internal_req = crate::models::UpdateMemoryRequest {
        id: Some(id),
        content: None,
        container_tag,
        new_content: req.content,
        metadata: req.metadata,
        is_static: req.is_static,
    };

    match state.memory.update_memory(internal_req).await {
        Ok(resp) => ApiResponse::success(UpdateMemoryResponse::from(resp)),
        Err(e) => e.into(),
    }
}

/// `DELETE /api/v1/memories/{memoryId}`
#[utoipa::path(
    delete,
    path = "/api/v1/memories/{memoryId}",
    tag = "memories",
    operation_id = "memories.forgetById",
    params(("memoryId" = String, Path, description = "Memory ID")),
    request_body(content = ForgetMemoryRequest, description = "Optional reason for deletion"),
    responses(
        (status = 200, description = "Memory forgotten", body = ForgetMemoryResponse),
        (status = 404, description = "Memory not found", body = ApiError),
    )
)]
pub async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: Option<axum::Json<ForgetMemoryRequest>>,
) -> ApiResponse<ForgetMemoryResponse> {
    let existing = match state.db.get_memory_by_id(&id).await {
        Ok(Some(mem)) => mem,
        Ok(None) => {
            return ApiResponse::error(ErrorCode::NotFound, format!("Memory {id} not found"))
        }
        Err(e) => return e.into(),
    };

    let reason = body.and_then(|b| b.0.reason);

    let container_tag = existing
        .container_tag
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let internal_req = crate::models::ForgetMemoryRequest {
        id: Some(id),
        content: None,
        container_tag,
        reason,
    };

    match state.memory.forget_memory(internal_req).await {
        Ok(resp) => ApiResponse::success(ForgetMemoryResponse::from(resp)),
        Err(e) => e.into(),
    }
}

/// `GET /api/v1/memories`
///
/// Lists memories with cursor-based pagination. Requires `containerTag` query
/// parameter since the DB layer does not support unscoped memory listing.
#[utoipa::path(
    get,
    path = "/api/v1/memories",
    tag = "memories",
    operation_id = "memories.list",
    params(ListMemoriesQuery),
    responses(
        (status = 200, description = "Memories listed", body = ListMemoriesResponse),
        (status = 400, description = "Missing containerTag", body = ApiError),
    )
)]
pub async fn list_memories(
    State(state): State<AppState>,
    Query(query): Query<ListMemoriesQuery>,
) -> ApiResponse<ListMemoriesResponse> {
    let container_tag = match query.container_tag {
        Some(ref tag) if !tag.is_empty() => tag.clone(),
        _ => {
            return ApiResponse::error(
                ErrorCode::InvalidRequest,
                "containerTag query parameter is required",
            );
        }
    };

    let limit = query.limit.unwrap_or(20).clamp(1, 100);

    // Cursor encodes a page number (1-based)
    let page: u32 = query
        .cursor
        .as_ref()
        .and_then(|c| c.parse::<u32>().ok())
        .unwrap_or(1);

    let offset = (page - 1) * limit;

    // Use get_container_graph to fetch actual memory rows for the container.
    // We request a large limit and paginate in-memory, since the DB trait
    // doesn't expose a dedicated list_memories method.
    let fetch_limit = offset + limit + 1; // +1 to detect if there's a next page
    let graph = match state
        .db
        .get_container_graph(&container_tag, fetch_limit)
        .await
    {
        Ok(g) => g,
        Err(e) => return e.into(),
    };

    let all_memories = graph.memories;
    let total = all_memories.len() as u64;
    let start = offset as usize;

    let memories: Vec<MemoryResponse> = all_memories
        .into_iter()
        .skip(start)
        .take(limit as usize)
        .map(MemoryResponse::from)
        .collect();

    let has_more = total > (offset + limit) as u64;
    let next_cursor = if has_more {
        Some((page + 1).to_string())
    } else {
        None
    };

    let meta = ResponseMeta {
        next_cursor,
        total: Some(total),
    };

    ApiResponse::success_with_meta(ListMemoriesResponse { memories }, meta)
}

/// `POST /api/v1/memories:forget`
///
/// Content-based forget: finds a memory by content within a container and
/// marks it as forgotten.
#[utoipa::path(
    post,
    path = "/api/v1/memories:forget",
    tag = "memories",
    operation_id = "memories.forget",
    request_body = ContentForgetRequest,
    responses(
        (status = 200, description = "Memory forgotten", body = ForgetMemoryResponse),
        (status = 400, description = "Invalid request", body = ApiError),
    )
)]
pub async fn forget_memory(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<ContentForgetRequest>,
) -> ApiResponse<ForgetMemoryResponse> {
    if req.content.trim().is_empty() {
        return ApiResponse::error(ErrorCode::InvalidRequest, "Content cannot be empty");
    }

    if req.container_tag.trim().is_empty() {
        return ApiResponse::error(ErrorCode::InvalidRequest, "Container tag cannot be empty");
    }

    let internal_req = crate::models::ForgetMemoryRequest {
        id: None,
        content: Some(req.content),
        container_tag: req.container_tag,
        reason: req.reason,
    };

    match state.memory.forget_memory(internal_req).await {
        Ok(resp) => ApiResponse::success(ForgetMemoryResponse::from(resp)),
        Err(e) => e.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_response_from_domain_model() {
        let mem = crate::models::Memory::new(
            "mem_test".to_string(),
            "User prefers dark mode".to_string(),
            "space1".to_string(),
        );
        let resp = MemoryResponse::from(mem);
        assert_eq!(resp.memory_id, "mem_test");
        assert_eq!(resp.content, "User prefers dark mode");
        assert!(resp.is_latest);
        assert!(!resp.is_forgotten);
    }

    #[test]
    fn update_memory_response_from_domain_model() {
        let internal = crate::models::UpdateMemoryResponse {
            id: "mem_new".to_string(),
            memory: "Updated content".to_string(),
            version: 2,
            parent_memory_id: Some("mem_old".to_string()),
            root_memory_id: Some("mem_root".to_string()),
            created_at: chrono::Utc::now(),
        };
        let resp = UpdateMemoryResponse::from(internal);
        assert_eq!(resp.memory_id, "mem_new");
        assert_eq!(resp.version, 2);
        assert_eq!(resp.parent_memory_id, Some("mem_old".to_string()));
    }

    #[test]
    fn forget_memory_response_from_domain_model() {
        let internal = crate::models::ForgetMemoryResponse {
            id: "mem_123".to_string(),
            forgotten: true,
        };
        let resp = ForgetMemoryResponse::from(internal);
        assert_eq!(resp.memory_id, "mem_123");
        assert!(resp.forgotten);
    }

    #[test]
    fn content_forget_request_deserializes() {
        let json = r#"{"content":"old preference","containerTag":"user_1","reason":"changed"}"#;
        let req: ContentForgetRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.content, "old preference");
        assert_eq!(req.container_tag, "user_1");
        assert_eq!(req.reason, Some("changed".to_string()));
    }

    #[test]
    fn list_memories_query_defaults() {
        let json = r#"{}"#;
        let query: ListMemoriesQuery = serde_json::from_str(json).expect("deserialize");
        assert!(query.container_tag.is_none());
        assert!(query.limit.is_none());
        assert!(query.cursor.is_none());
    }
}
