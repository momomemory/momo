//! Memory request/response DTOs for the v1 API.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::common::{Metadata, V1MemoryType};
use crate::models;

// ---------------------------------------------------------------------------
// Request DTOs
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/memories`.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct CreateMemoryRequest {
    /// The memory content text.
    pub content: String,
    /// Container tag for multi-tenant isolation.
    pub container_tag: String,
    /// Memory type classification.
    pub memory_type: Option<V1MemoryType>,
    /// Arbitrary key-value metadata.
    #[schema(value_type = Object)]
    pub metadata: Option<Metadata>,
}

/// Request body for `PATCH /v1/memories/{memoryId}`.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct UpdateMemoryRequest {
    /// The new memory content (replaces existing).
    pub content: String,
    /// Arbitrary metadata update.
    #[schema(value_type = Object)]
    pub metadata: Option<Metadata>,
    /// Pin this memory so it's never forgotten.
    pub is_static: Option<bool>,
}

/// Request body for `DELETE /v1/memories/{memoryId}`.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ForgetMemoryRequest {
    /// Reason for forgetting (stored for audit trail).
    pub reason: Option<String>,
}

/// Request body for `POST /v1/memories:forget` (content-based forget).
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ContentForgetRequest {
    /// The memory content to search for and forget.
    pub content: String,
    /// Container tag to scope the search.
    pub container_tag: String,
    /// Reason for forgetting (stored for audit trail).
    pub reason: Option<String>,
}

/// Query parameters for `GET /v1/memories`.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema, utoipa::IntoParams)]
#[serde(rename_all = "camelCase")]
pub struct ListMemoriesQuery {
    /// Filter by container tag.
    pub container_tag: Option<String>,
    /// Maximum results per page (default 20, max 100).
    pub limit: Option<u32>,
    /// Opaque cursor for pagination.
    pub cursor: Option<String>,
}

// ---------------------------------------------------------------------------
// Response DTOs
// ---------------------------------------------------------------------------

/// Full memory response for `GET /v1/memories/{memoryId}`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryResponse {
    /// Unique memory ID (nanoid, 21 chars).
    pub memory_id: String,
    /// The memory content text.
    pub content: String,
    /// Container tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_tag: Option<String>,
    /// Memory type classification.
    pub memory_type: V1MemoryType,
    /// Version number (incremented on contradiction resolution).
    pub version: i32,
    /// Whether this is the latest version.
    pub is_latest: bool,
    /// Whether this memory was derived by inference.
    pub is_inference: bool,
    /// Whether this memory has been forgotten (soft-deleted).
    pub is_forgotten: bool,
    /// Whether this memory is pinned (never auto-forgotten).
    pub is_static: bool,
    /// Confidence score (0.0–1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Attached metadata.
    #[schema(value_type = Object)]
    pub metadata: Metadata,
    /// When the memory was created.
    #[schema(value_type = String)]
    pub created_at: DateTime<Utc>,
    /// When the memory was last updated.
    #[schema(value_type = String)]
    pub updated_at: DateTime<Utc>,
}

impl From<models::Memory> for MemoryResponse {
    fn from(mem: models::Memory) -> Self {
        Self {
            memory_id: mem.id,
            content: mem.memory,
            container_tag: mem.container_tag,
            memory_type: mem.memory_type.into(),
            version: mem.version,
            is_latest: mem.is_latest,
            is_inference: mem.is_inference,
            is_forgotten: mem.is_forgotten,
            is_static: mem.is_static,
            confidence: mem.confidence,
            metadata: mem.metadata,
            created_at: mem.created_at,
            updated_at: mem.updated_at,
        }
    }
}

/// Response for `PATCH /v1/memories/{memoryId}`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct UpdateMemoryResponse {
    /// The memory ID.
    pub memory_id: String,
    /// The updated content.
    pub content: String,
    /// The new version number.
    pub version: i32,
    /// Parent memory ID (if this is an update to a previous version).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_memory_id: Option<String>,
    /// When the update was created.
    #[schema(value_type = String)]
    pub created_at: DateTime<Utc>,
}

impl From<models::UpdateMemoryResponse> for UpdateMemoryResponse {
    fn from(resp: models::UpdateMemoryResponse) -> Self {
        Self {
            memory_id: resp.id,
            content: resp.memory,
            version: resp.version,
            parent_memory_id: resp.parent_memory_id,
            created_at: resp.created_at,
        }
    }
}

/// Response for `DELETE /v1/memories/{memoryId}`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ForgetMemoryResponse {
    /// The memory ID that was forgotten.
    pub memory_id: String,
    /// Whether the memory was successfully forgotten.
    pub forgotten: bool,
}

impl From<models::ForgetMemoryResponse> for ForgetMemoryResponse {
    fn from(resp: models::ForgetMemoryResponse) -> Self {
        Self {
            memory_id: resp.id,
            forgotten: resp.forgotten,
        }
    }
}

/// Memory list response wrapper.
///
/// Pagination is handled by the envelope's `meta.nextCursor` / `meta.total`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ListMemoriesResponse {
    pub memories: Vec<MemoryResponse>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Memory;

    #[test]
    fn memory_response_from_domain() {
        let mem = Memory::new(
            "mem_test123".to_string(),
            "User likes cats".to_string(),
            "space1".to_string(),
        );
        let resp: MemoryResponse = mem.into();
        assert_eq!(resp.memory_id, "mem_test123");
        assert_eq!(resp.content, "User likes cats");
        assert_eq!(resp.memory_type, V1MemoryType::Fact);
        assert!(resp.is_latest);
        assert!(!resp.is_forgotten);
    }

    #[test]
    fn memory_response_serializes_camel_case() {
        let resp = MemoryResponse {
            memory_id: "mem_1".to_string(),
            content: "test".to_string(),
            container_tag: Some("user_1".to_string()),
            memory_type: V1MemoryType::Preference,
            version: 2,
            is_latest: true,
            is_inference: false,
            is_forgotten: false,
            is_static: false,
            confidence: Some(0.85),
            metadata: std::collections::HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let json = serde_json::to_value(&resp).expect("serialize");
        assert!(json.get("memoryId").is_some());
        assert!(json.get("memory_id").is_none());
        assert!(json.get("memoryType").is_some());
        assert!(json.get("isLatest").is_some());
        assert!(json.get("isInference").is_some());
        assert!(json.get("isForgotten").is_some());
        assert!(json.get("isStatic").is_some());
        assert!(json.get("containerTag").is_some());
    }

    #[test]
    fn forget_memory_request_deserializes() {
        let json = r#"{"reason": "outdated info"}"#;
        let req: ForgetMemoryRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.reason, Some("outdated info".to_string()));
    }
}
