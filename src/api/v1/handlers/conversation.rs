//! v1 Conversation handlers.

use axum::extract::State;
use nanoid::nanoid;

use crate::api::v1::dto::conversation::{ConversationIngestRequest, ConversationIngestResponse};
use crate::api::v1::response::{ApiError, ApiResponse, ErrorCode};
use crate::api::AppState;
use crate::models::{ConversationMessage, ConversationResponse, MemoryType};

/// `POST /api/v1/conversations:ingest`
///
/// Ingests a conversation, extracts memories via LLM, runs contradiction
/// detection and deduplication, then persists the resulting memories.
#[utoipa::path(
    post,
    path = "/api/v1/conversations:ingest",
    tag = "conversation",
    request_body = ConversationIngestRequest,
    responses(
        (status = 200, description = "Conversation ingested", body = ConversationIngestResponse),
        (status = 400, description = "Invalid request", body = ApiError),
    )
)]
pub async fn ingest_conversation(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<ConversationIngestRequest>,
) -> ApiResponse<ConversationIngestResponse> {
    if req.messages.is_empty() {
        return ApiResponse::error(ErrorCode::InvalidRequest, "Messages cannot be empty");
    }

    if req.container_tag.trim().is_empty() {
        return ApiResponse::error(ErrorCode::InvalidRequest, "Container tag cannot be empty");
    }

    let session_id = req.session_id.clone().unwrap_or_else(|| nanoid!());

    let messages: Vec<ConversationMessage> = req.messages.into_iter().map(Into::into).collect();

    let extraction_result = match state.extractor.extract_from_conversation(&messages).await {
        Ok(r) => r,
        Err(e) => return e.into(),
    };

    let memories = if state
        .config
        .llm
        .as_ref()
        .map_or(false, |l| l.enable_contradiction_detection)
    {
        match state
            .extractor
            .check_contradictions(
                extraction_result.memories,
                &req.container_tag,
                &*state.db,
            )
            .await
        {
            Ok(m) => m,
            Err(e) => return e.into(),
        }
    } else {
        extraction_result.memories
    };

    let deduplicated = match state
        .extractor
        .deduplicate(memories, &req.container_tag, &*state.db)
        .await
    {
        Ok(d) => d,
        Err(e) => return e.into(),
    };

    let request_memory_type: Option<MemoryType> = req.memory_type.map(Into::into);

    let mut memory_ids = Vec::new();
    for memory in &deduplicated {
        let memory_type = if let Some(req_type) = request_memory_type {
            req_type
        } else {
            memory
                .memory_type
                .parse()
                .unwrap_or(MemoryType::Fact)
        };

        match state
            .memory
            .create_memory_with_type(&memory.content, &req.container_tag, false, memory_type)
            .await
        {
            Ok(created) => memory_ids.push(created.id),
            Err(e) => tracing::error!(error = %e, "Failed to create memory from conversation"),
        }
    }

    ApiResponse::success(ConversationIngestResponse::from(ConversationResponse {
        memories_extracted: memory_ids.len() as i32,
        memory_ids,
        session_id,
    }))
}

#[cfg(test)]
mod tests {
    use crate::api::v1::dto::conversation::ConversationIngestRequest;

    #[test]
    fn ingest_request_deserializes_minimal() {
        let json = r#"{
            "messages": [
                {"role": "user", "content": "I like Rust"}
            ],
            "containerTag": "user_1"
        }"#;
        let req: ConversationIngestRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.container_tag, "user_1");
        assert!(req.session_id.is_none());
        assert!(req.memory_type.is_none());
    }

    #[test]
    fn ingest_request_with_session_and_type() {
        let json = r#"{
            "messages": [
                {"role": "user", "content": "Meeting at 3pm"},
                {"role": "assistant", "content": "Got it"}
            ],
            "containerTag": "team_1",
            "sessionId": "sess_123",
            "memoryType": "episode"
        }"#;
        let req: ConversationIngestRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.session_id, Some("sess_123".to_string()));
        assert!(req.memory_type.is_some());
    }
}
