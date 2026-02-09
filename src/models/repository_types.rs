use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{Document, GraphEdge, Memory};

/// A memory search result with similarity score.
#[derive(Debug, Clone)]
pub struct MemorySearchHit {
    pub memory: Memory,
    pub score: f32,
}

/// Graph data returned from neighborhood/container graph queries.
#[derive(Debug, Clone)]
pub struct GraphData {
    pub memories: Vec<Memory>,
    pub edges: Vec<GraphEdge>,
    pub documents: Vec<Document>,
}

/// A cached user profile entry.
#[derive(Debug, Clone)]
pub struct CachedProfile {
    #[allow(dead_code)] // Part of DB schema mapping
    pub container_tag: String,
    pub narrative: Option<String>,
    pub summary: Option<String>,
    pub cached_at: Option<String>,
}

/// A link between a memory and its source document/chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySource {
    pub id: String,
    pub memory_id: String,
    pub document_id: String,
    pub chunk_id: Option<String>,
    pub created_at: DateTime<Utc>,
}
