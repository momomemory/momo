use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::Metadata;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub document_id: String,
    pub content: String,
    pub embedded_content: Option<String>,
    pub position: i32,
    pub token_count: Option<i32>,
    pub created_at: DateTime<Utc>,
}

impl Chunk {
    pub fn new(id: String, document_id: String, content: String, position: i32) -> Self {
        Self {
            id,
            document_id,
            content,
            embedded_content: None,
            position,
            token_count: None,
            created_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSearchResult {
    pub content: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f32>,
    pub is_relevant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkWithDocument {
    pub chunk_id: String,
    pub document_id: String,
    pub chunk_content: String,
    pub document_title: Option<String>,
    pub document_metadata: Metadata,
    pub score: f32,
}
