//! Conversation request/response DTOs for the v1 API.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::common::{Metadata, V1MemoryType};
use crate::models;

/// Request body for `POST /v1/conversation`.
///
/// Ingests a conversation and extracts memories from it.
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConversationIngestRequest {
    /// Conversation messages to process.
    pub messages: Vec<ConversationMessageDto>,
    /// Container tag for multi-tenant isolation.
    pub container_tag: String,
    /// Session ID for grouping related conversations.
    pub session_id: Option<String>,
    /// Arbitrary metadata for the conversation.
    #[schema(value_type = Option<Object>)]
    pub metadata: Option<Metadata>,
    /// Memory type to assign to extracted memories.
    pub memory_type: Option<V1MemoryType>,
}

/// A single message within a conversation.
#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConversationMessageDto {
    /// Message role (e.g. `"user"`, `"assistant"`, `"system"`).
    pub role: String,
    /// Message content text.
    pub content: String,
    /// When this message was sent.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<String>)]
    pub timestamp: Option<DateTime<Utc>>,
}

impl From<ConversationMessageDto> for models::ConversationMessage {
    fn from(msg: ConversationMessageDto) -> Self {
        Self {
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp,
        }
    }
}

/// Response for `POST /v1/conversation`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConversationIngestResponse {
    /// Number of memories extracted from the conversation.
    pub memories_extracted: i32,
    /// IDs of the extracted memories.
    pub memory_ids: Vec<String>,
    /// Session ID (generated if not provided in request).
    pub session_id: String,
}

impl From<models::ConversationResponse> for ConversationIngestResponse {
    fn from(resp: models::ConversationResponse) -> Self {
        Self {
            memories_extracted: resp.memories_extracted,
            memory_ids: resp.memory_ids,
            session_id: resp.session_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_request_deserializes() {
        let json = r#"{
            "messages": [
                {"role": "user", "content": "I like pizza"},
                {"role": "assistant", "content": "Noted!"}
            ],
            "containerTag": "user_1"
        }"#;
        let req: ConversationIngestRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.container_tag, "user_1");
    }

    #[test]
    fn conversation_response_serializes_camel_case() {
        let resp = ConversationIngestResponse {
            memories_extracted: 2,
            memory_ids: vec!["mem_1".to_string(), "mem_2".to_string()],
            session_id: "sess_abc".to_string(),
        };

        let json = serde_json::to_value(&resp).expect("serialize");
        assert!(json.get("memoriesExtracted").is_some());
        assert!(json.get("memoryIds").is_some());
        assert!(json.get("sessionId").is_some());
    }
}
