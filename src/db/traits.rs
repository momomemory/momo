use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::error::Result;
use crate::models::{
    CachedProfile, Chunk, ChunkWithDocument, ContainerFilter, Document, DocumentSummary, GraphData,
    GraphEdgeType, ListDocumentsRequest, Memory, MemoryRelationType, MemorySearchHit, MemorySource,
    Pagination, ProcessingDocument, ProcessingStatus, UserProfile,
};

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EpisodeDecayCandidate {
    pub id: String,
    pub memory: String,
    pub space_id: String,
    pub last_accessed: Option<String>,
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Individual store traits
// ---------------------------------------------------------------------------

/// CRUD and query operations for documents.
#[async_trait]
pub trait DocumentStore: Send + Sync {
    async fn create_document(&self, doc: &Document) -> Result<()>;
    async fn get_document_by_id(&self, id: &str) -> Result<Option<Document>>;
    async fn get_documents_by_ids(&self, ids: &[String]) -> Result<Vec<Document>>;
    async fn get_document_by_custom_id(&self, custom_id: &str) -> Result<Option<Document>>;
    async fn update_document(&self, doc: &Document) -> Result<()>;
    async fn delete_document(&self, id: &str) -> Result<bool>;
    async fn delete_document_by_custom_id(&self, custom_id: &str) -> Result<bool>;
    async fn list_documents(
        &self,
        req: &ListDocumentsRequest,
    ) -> Result<(Vec<DocumentSummary>, Pagination)>;
    async fn get_processing_documents(&self) -> Result<Vec<ProcessingDocument>>;
    async fn update_document_status(
        &self,
        id: &str,
        status: ProcessingStatus,
        error: Option<&str>,
    ) -> Result<()>;
    async fn queue_all_documents_for_reprocessing(&self) -> Result<u64>;
}

/// CRUD and vector-search operations for chunks.
#[async_trait]
pub trait ChunkStore: Send + Sync {
    async fn create_chunks_batch(&self, chunks: &[Chunk]) -> Result<()>;
    async fn update_chunk_embeddings_batch(&self, updates: &[(String, Vec<f32>)]) -> Result<()>;
    async fn delete_chunks_by_document_id(&self, document_id: &str) -> Result<()>;
    async fn search_similar_chunks(
        &self,
        embedding: &[f32],
        limit: u32,
        threshold: f32,
        container_tags: Option<&[String]>,
    ) -> Result<Vec<ChunkWithDocument>>;

    /// Delete all chunks from the store.
    async fn delete_all_chunks(&self) -> Result<u64>;
}

/// CRUD, search, graph, and profile operations for memories.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn create_memory(&self, memory: &Memory) -> Result<()>;
    async fn get_memory_by_id(&self, id: &str) -> Result<Option<Memory>>;
    async fn get_memories_by_ids(&self, ids: &[String]) -> Result<Vec<Memory>>;
    async fn get_memory_by_content(
        &self,
        content: &str,
        container_tag: &str,
    ) -> Result<Option<Memory>>;
    async fn update_memory_to_not_latest(&self, id: &str) -> Result<()>;
    async fn forget_memory(&self, id: &str, reason: Option<&str>) -> Result<()>;
    async fn update_memory_last_accessed_batch(&self, ids: &[&str]) -> Result<u64>;
    async fn update_memory_source_count(&self, id: &str, new_count: i32) -> Result<()>;
    async fn update_memory_version_chain(
        &self,
        id: &str,
        parent_memory_id: &str,
        root_memory_id: &str,
        version: i32,
    ) -> Result<()>;
    async fn update_memory_embedding(&self, memory_id: &str, embedding: &[f32]) -> Result<()>;
    async fn search_similar_memories(
        &self,
        embedding: &[f32],
        limit: u32,
        threshold: f32,
        container_tag: Option<&str>,
        include_forgotten: bool,
    ) -> Result<Vec<MemorySearchHit>>;
    async fn get_memory_children(&self, parent_id: &str) -> Result<Vec<Memory>>;
    async fn get_memory_parents(&self, root_id: &str) -> Result<Vec<Memory>>;
    async fn get_forgetting_candidates(&self, before: DateTime<Utc>) -> Result<Vec<Memory>>;
    async fn get_seed_memories(&self, limit: usize) -> Result<Vec<Memory>>;
    async fn check_inference_exists(&self, source_ids: &[String]) -> Result<bool>;
    async fn get_user_profile(
        &self,
        container_tag: &str,
        include_dynamic: bool,
        limit: u32,
    ) -> Result<UserProfile>;
    async fn update_memory_relations(
        &self,
        id: &str,
        new_relations: HashMap<String, MemoryRelationType>,
    ) -> Result<()>;
    async fn add_memory_relation(
        &self,
        id: &str,
        related_id: &str,
        relation_type: MemoryRelationType,
    ) -> Result<()>;
    async fn get_graph_neighborhood(
        &self,
        id: &str,
        depth: u32,
        max_nodes: u32,
        relation_types: Option<&[GraphEdgeType]>,
    ) -> Result<GraphData>;
    async fn get_container_graph(&self, container_tag: &str, max_nodes: u32) -> Result<GraphData>;
    async fn get_cached_profile(&self, container_tag: &str) -> Result<Option<CachedProfile>>;
    async fn upsert_cached_profile(
        &self,
        container_tag: &str,
        narrative: Option<&str>,
        summary: Option<&str>,
    ) -> Result<()>;

    // -- Episode decay helpers ------------------------------------------------

    /// Return active episode memories eligible for decay evaluation.
    /// Filters: `is_forgotten = 0`, `is_static = 0`, `memory_type = 'episode'`, `is_latest = 1`.
    async fn get_episode_decay_candidates(&self) -> Result<Vec<EpisodeDecayCandidate>>;

    /// Set `forget_after` on a memory (must not be forgotten or static).
    async fn set_memory_forget_after(&self, id: &str, forget_after: DateTime<Utc>) -> Result<u64>;

    // -- Profile refresh helpers -----------------------------------------------

    /// Return distinct `container_tag` values that have active (latest, not-forgotten) memories.
    async fn get_active_container_tags(&self) -> Result<Vec<String>>;

    /// Return `MAX(updated_at)` for active memories with the given container_tag.
    async fn get_max_memory_updated_at(&self, container_tag: &str)
        -> Result<Option<DateTime<Utc>>>;
}

/// CRUD operations for memory-to-source links.
#[async_trait]
pub trait MemorySourceStore: Send + Sync {
    async fn create_memory_source(
        &self,
        memory_id: &str,
        document_id: &str,
        chunk_id: Option<&str>,
    ) -> Result<MemorySource>;
    async fn get_sources_by_memory(&self, memory_id: &str) -> Result<Vec<MemorySource>>;
}

/// Key-value metadata store (e.g. embedding dimensions).
#[async_trait]
pub trait MetadataStore: Send + Sync {
    async fn get_embedding_dimensions(&self) -> Result<Option<usize>>;
    async fn set_embedding_dimensions(&self, dims: usize) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Unified backend supertrait
// ---------------------------------------------------------------------------

/// A complete database backend that combines all store traits plus lifecycle
/// operations (initialization, sync).
#[async_trait]
pub trait DatabaseBackend:
    DocumentStore + ChunkStore + MemoryStore + MemorySourceStore + MetadataStore
{
    /// Sync with remote (e.g. Turso replication). No-op for local-only backends.
    async fn sync(&self) -> Result<()>;

    /// Get filter configuration for a container tag
    async fn get_container_filter(&self, tag: &str) -> Result<Option<ContainerFilter>>;
}
