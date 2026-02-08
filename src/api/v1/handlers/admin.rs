//! v1 Admin handlers.

use axum::extract::State;
use chrono::Utc;

use crate::api::v1::dto::ForgettingRunResponse;
use crate::api::v1::response::{ApiResponse, ErrorCode};
use crate::api::AppState;
use crate::services::ForgettingManager;

/// `POST /api/v1/admin/forgetting:run`
#[utoipa::path(
    post,
    path = "/api/v1/admin/forgetting:run",
    tag = "admin",
    responses(
        (status = 200, description = "Forgetting cycle completed", body = ForgettingRunResponse),
    ),
    security(("bearer_auth" = []))
)]
pub async fn run_forgetting(State(state): State<AppState>) -> ApiResponse<ForgettingRunResponse> {
    let candidates = match state.db.get_forgetting_candidates(Utc::now()).await {
        Ok(c) => c,
        Err(e) => return e.into(),
    };
    let evaluated = candidates.len() as u32;

    let manager = ForgettingManager::new(
        state.db.clone(),
        state.config.memory.forgetting_check_interval_secs,
    );

    match manager.run_once().await {
        Ok(forgotten_count) => ApiResponse::success(ForgettingRunResponse {
            memories_forgotten: forgotten_count as u32,
            memories_evaluated: evaluated,
        }),
        Err(e) => ApiResponse::error(
            ErrorCode::InternalError,
            format!("Forgetting cycle failed: {e}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use crate::api::v1::dto::ForgettingRunResponse;

    #[test]
    fn forgetting_run_response_serializes_camel_case() {
        let resp = ForgettingRunResponse {
            memories_forgotten: 5,
            memories_evaluated: 42,
        };
        let json = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(json["memoriesForgotten"], 5);
        assert_eq!(json["memoriesEvaluated"], 42);
    }
}
