use axum::{extract::State, Json};
use serde::Serialize;

use crate::api::state::AppState;
use crate::services::ForgettingManager;

#[derive(Serialize)]
pub struct RunForgettingResponse {
    pub success: bool,
    pub forgotten_count: u64,
    pub message: String,
}

pub async fn run_forgetting(State(state): State<AppState>) -> Json<RunForgettingResponse> {
    // Create ForgettingManager from state
    let manager = ForgettingManager::new(
        state.db.clone(),
        state.config.memory.forgetting_check_interval_secs,
    );

    match manager.run_once().await {
        Ok(forgotten_count) => Json(RunForgettingResponse {
            success: true,
            forgotten_count,
            message: format!(
                "Forgetting run completed: {} memories forgotten",
                forgotten_count
            ),
        }),
        Err(e) => Json(RunForgettingResponse {
            success: false,
            forgotten_count: 0,
            message: format!("Forgetting run failed: {}", e),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_serializes() {
        let resp = RunForgettingResponse {
            success: true,
            forgotten_count: 42,
            message: "done".to_string(),
        };

        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains("\"success\""));
        assert!(s.contains("\"forgotten_count\""));
        assert!(s.contains("\"message\""));
    }
}
