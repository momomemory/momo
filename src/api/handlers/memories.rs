use axum::{extract::State, Json};
use nanoid::nanoid;

use crate::api::{AppJson, AppState};
use crate::error::Result;
use crate::models::{
    ConversationRequest, ConversationResponse, ForgetMemoryRequest, ForgetMemoryResponse,
    GetProfileRequest, ProfileResponse, UpdateMemoryRequest, UpdateMemoryResponse,
};

pub async fn update_memory(
    State(state): State<AppState>,
    AppJson(req): AppJson<UpdateMemoryRequest>,
) -> Result<Json<UpdateMemoryResponse>> {
    let response = state.memory.update_memory(req).await?;
    Ok(Json(response))
}

pub async fn forget_memory(
    State(state): State<AppState>,
    AppJson(req): AppJson<ForgetMemoryRequest>,
) -> Result<Json<ForgetMemoryResponse>> {
    let response = state.memory.forget_memory(req).await?;
    Ok(Json(response))
}

pub async fn get_profile(
    State(state): State<AppState>,
    AppJson(req): AppJson<GetProfileRequest>,
) -> Result<Json<ProfileResponse>> {
    // Pass through new flags compact and generate_narrative if provided
    let response = state.memory.get_profile(req, &state.search).await?;
    Ok(Json(response))
}

pub async fn conversation(
    State(state): State<AppState>,
    AppJson(req): AppJson<ConversationRequest>,
) -> Result<Json<ConversationResponse>> {
    let session_id = req.session_id.clone().unwrap_or_else(|| nanoid!());

    let extraction_result = state
        .extractor
        .extract_from_conversation(&req.messages)
        .await?;

    let memories = if state
        .config
        .llm
        .as_ref()
        .map_or(false, |l| l.enable_contradiction_detection)
    {
        state
            .extractor
            .check_contradictions(extraction_result.memories, &req.container_tag, &*state.db)
            .await?
    } else {
        extraction_result.memories
    };

    let deduplicated = state
        .extractor
        .deduplicate(memories, &req.container_tag, &*state.db)
        .await?;

    let mut memory_ids = Vec::new();
    for memory in &deduplicated {
        let memory_type = if let Some(req_type) = req.memory_type {
            req_type
        } else {
            memory
                .memory_type
                .parse()
                .unwrap_or(crate::models::MemoryType::Fact)
        };

        match state
            .memory
            .create_memory_with_type(&memory.content, &req.container_tag, false, memory_type)
            .await
        {
            Ok(created) => memory_ids.push(created.id),
            Err(e) => tracing::error!(error = %e, "Failed to create inferred memory"),
        }
    }

    Ok(Json(ConversationResponse {
        memories_extracted: memory_ids.len() as i32,
        memory_ids,
        session_id,
    }))
}
