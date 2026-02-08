//! v1 Profile handlers.

use axum::extract::State;

use crate::api::v1::dto::profile::{ComputeProfileRequest, ProfileResponse};
use crate::api::v1::response::{ApiError, ApiResponse, ErrorCode};
use crate::api::AppState;
use crate::models::GetProfileRequest;

/// `POST /api/v1/profile:compute`
///
/// Computes or retrieves a user profile from accumulated memories.
/// Uses the full `MemoryService::get_profile` pipeline (narrative generation,
/// compaction, search) and then maps the resulting `UserProfile` into the
/// v1 `ProfileResponse` DTO.
#[utoipa::path(
    post,
    path = "/api/v1/profile:compute",
    tag = "profile",
    operation_id = "profile.compute",
    request_body = ComputeProfileRequest,
    responses(
        (status = 200, description = "Profile computed", body = ProfileResponse),
        (status = 400, description = "Invalid request", body = ApiError),
    )
)]
pub async fn compute_profile(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<ComputeProfileRequest>,
) -> ApiResponse<ProfileResponse> {
    if req.container_tag.trim().is_empty() {
        return ApiResponse::error(ErrorCode::InvalidRequest, "Container tag cannot be empty");
    }

    // Fetch the full UserProfile (with ProfileFact details: confidence, created_at)
    // directly from the DB, matching the pattern used by list_memories.
    let include_dynamic = req.include_dynamic.unwrap_or(true);
    let limit = req.limit.unwrap_or(50);

    let mut profile = match state
        .db
        .get_user_profile(&req.container_tag, include_dynamic, limit)
        .await
    {
        Ok(p) => p,
        Err(e) => return e.into(),
    };

    // If narrative generation is requested, delegate to the service layer which
    // has access to `ProfileGenerator` and caching.
    if req.generate_narrative.unwrap_or(false) {
        let domain_req = GetProfileRequest {
            container_tag: req.container_tag.clone(),
            q: req.q.clone(),
            threshold: req.threshold,
            include_dynamic: req.include_dynamic,
            limit: req.limit,
            compact: None,
            generate_narrative: Some(true),
        };

        match state.memory.get_profile(domain_req, &state.search).await {
            Ok(domain_resp) => {
                profile.narrative = domain_resp.narrative;
            }
            Err(e) => {
                // Narrative generation is best-effort; log and continue
                // without a narrative rather than failing the whole request.
                tracing::warn!(error = %e, "Failed to generate profile narrative");
            }
        }
    }

    ApiResponse::success(ProfileResponse::from(profile))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_profile_request_deserializes() {
        let json = r#"{
            "containerTag": "user_1",
            "includeDynamic": true,
            "limit": 25,
            "generateNarrative": true
        }"#;
        let req: ComputeProfileRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.container_tag, "user_1");
        assert_eq!(req.include_dynamic, Some(true));
        assert_eq!(req.limit, Some(25));
        assert_eq!(req.generate_narrative, Some(true));
    }

    #[test]
    fn compute_profile_request_minimal() {
        let json = r#"{"containerTag": "user_2"}"#;
        let req: ComputeProfileRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.container_tag, "user_2");
        assert!(req.q.is_none());
        assert!(req.threshold.is_none());
        assert!(req.include_dynamic.is_none());
        assert!(req.limit.is_none());
        assert!(req.generate_narrative.is_none());
    }
}
