//! Profile request/response DTOs for the v1 API.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::models;

/// Request body for `POST /v1/profile`.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ComputeProfileRequest {
    /// Container tag identifying the user/entity.
    pub container_tag: String,
    /// Optional search query to filter relevant facts.
    pub q: Option<String>,
    /// Minimum similarity threshold for query-based filtering.
    pub threshold: Option<f32>,
    /// Include dynamic (episode-based) facts.
    pub include_dynamic: Option<bool>,
    /// Maximum number of facts to include.
    pub limit: Option<u32>,
    /// Generate a narrative summary.
    pub generate_narrative: Option<bool>,
}

/// Profile response for `POST /v1/profile`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProfileResponse {
    /// Container tag this profile belongs to.
    pub container_tag: String,
    /// AI-generated narrative summary of the user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub narrative: Option<String>,
    /// Long-lived factual information.
    pub static_facts: Vec<ProfileFactResponse>,
    /// Temporal/episode-based information.
    pub dynamic_facts: Vec<ProfileFactResponse>,
    /// Total number of memories used to compute this profile.
    pub total_memories: i32,
    /// When the profile was last computed.
    #[schema(value_type = String)]
    pub last_updated: DateTime<Utc>,
}

/// A single fact within a profile.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProfileFactResponse {
    /// The fact content.
    pub content: String,
    /// Confidence score (0.0–1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// When this fact was first recorded.
    #[schema(value_type = String)]
    pub created_at: DateTime<Utc>,
}

impl From<models::ProfileFact> for ProfileFactResponse {
    fn from(fact: models::ProfileFact) -> Self {
        Self {
            content: fact.memory,
            confidence: fact.confidence,
            created_at: fact.created_at,
        }
    }
}

impl From<models::UserProfile> for ProfileResponse {
    fn from(profile: models::UserProfile) -> Self {
        Self {
            container_tag: profile.container_tag,
            narrative: profile.narrative,
            static_facts: profile.static_facts.into_iter().map(Into::into).collect(),
            dynamic_facts: profile.dynamic_facts.into_iter().map(Into::into).collect(),
            total_memories: profile.total_memories,
            last_updated: profile.last_updated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_response_serializes_camel_case() {
        let resp = ProfileResponse {
            container_tag: "user_1".to_string(),
            narrative: Some("A developer who likes Rust".to_string()),
            static_facts: vec![ProfileFactResponse {
                content: "Likes Rust".to_string(),
                confidence: Some(0.95),
                created_at: chrono::Utc::now(),
            }],
            dynamic_facts: vec![],
            total_memories: 10,
            last_updated: chrono::Utc::now(),
        };

        let json = serde_json::to_value(&resp).expect("serialize");
        assert!(json.get("containerTag").is_some());
        assert!(json.get("staticFacts").is_some());
        assert!(json.get("dynamicFacts").is_some());
        assert!(json.get("totalMemories").is_some());
        assert!(json.get("lastUpdated").is_some());
    }
}
