//! Admin DTOs for the v1 API.

use serde::Serialize;

/// Response for `POST /v1/admin/run-forgetting`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ForgettingRunResponse {
    /// Number of memories forgotten in this cycle.
    pub memories_forgotten: u32,
    /// Number of memories evaluated for forgetting.
    pub memories_evaluated: u32,
}
