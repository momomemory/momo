use std::collections::HashMap;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use crate::error::Result;
use crate::db::connection::Database;
use crate::db::repository::{
    ChunkRepository, DocumentRepository, MemoryRepository, MemorySourcesRepository,
};
use crate::db::MetadataRepository;
use libsql::params;
use crate::db::traits::{
    ChunkStore, DatabaseBackend, DocumentStore, EpisodeDecayCandidate, MemorySourceStore,
    MemoryStore, MetadataStore,
};
use crate::models::{
    CachedProfile, Chunk, ChunkWithDocument, ContainerFilter, Document, DocumentSummary,
    GraphData, GraphEdgeType, ListDocumentsRequest, Memory, MemoryRelationType, MemorySearchHit,
    MemorySource, Pagination, ProcessingDocument, ProcessingStatus, UserProfile,
};

pub struct LibSqlBackend {
    db: Database,
}

impl LibSqlBackend {
    pub fn new(db: Database) -> Self {
        Self { db }
    }
}

#[async_trait]
impl DocumentStore for LibSqlBackend {
    async fn create_document(&self, doc: &Document) -> Result<()> {
        let conn = self.db.connect()?;
        DocumentRepository::create(&conn, doc).await
    }
    async fn get_document_by_id(&self, id: &str) -> Result<Option<Document>> {
        let conn = self.db.connect()?;
        DocumentRepository::get_by_id(&conn, id).await
    }
    async fn get_documents_by_ids(&self, ids: &[String]) -> Result<Vec<Document>> {
        let conn = self.db.connect()?;
        DocumentRepository::get_by_ids(&conn, ids).await
    }
    async fn get_document_by_custom_id(&self, custom_id: &str) -> Result<Option<Document>> {
        let conn = self.db.connect()?;
        DocumentRepository::get_by_custom_id(&conn, custom_id).await
    }
    async fn update_document(&self, doc: &Document) -> Result<()> {
        let conn = self.db.connect()?;
        DocumentRepository::update(&conn, doc).await
    }
    async fn delete_document(&self, id: &str) -> Result<bool> {
        let conn = self.db.connect()?;
        DocumentRepository::delete(&conn, id).await
    }
    async fn delete_document_by_custom_id(&self, custom_id: &str) -> Result<bool> {
        let conn = self.db.connect()?;
        DocumentRepository::delete_by_custom_id(&conn, custom_id).await
    }
    async fn list_documents(&self, req: &ListDocumentsRequest) -> Result<(Vec<DocumentSummary>, Pagination)> {
        let conn = self.db.connect()?;
        DocumentRepository::list(&conn, req).await
    }
    async fn get_processing_documents(&self) -> Result<Vec<ProcessingDocument>> {
        let conn = self.db.connect()?;
        DocumentRepository::get_processing(&conn).await
    }
    async fn update_document_status(&self, id: &str, status: ProcessingStatus, error: Option<&str>) -> Result<()> {
        let conn = self.db.connect()?;
        DocumentRepository::update_status(&conn, id, status, error).await
    }
    async fn queue_all_documents_for_reprocessing(&self) -> Result<u64> {
        let conn = self.db.connect()?;
        let affected = conn
            .execute(
                "UPDATE documents SET status = 'queued' WHERE status = 'done'",
                (),
            )
            .await?;
        Ok(affected)
    }
}

#[async_trait]
impl ChunkStore for LibSqlBackend {
    async fn create_chunk(&self, chunk: &Chunk) -> Result<()> {
        let conn = self.db.connect()?;
        ChunkRepository::create(&conn, chunk).await
    }
    async fn create_chunks_batch(&self, chunks: &[Chunk]) -> Result<()> {
        let conn = self.db.connect()?;
        ChunkRepository::create_batch(&conn, chunks).await
    }
    async fn get_chunks_by_document_id(&self, document_id: &str) -> Result<Vec<Chunk>> {
        let conn = self.db.connect()?;
        ChunkRepository::get_by_document_id(&conn, document_id).await
    }
    async fn update_chunk_embedding(&self, chunk_id: &str, embedding: &[f32]) -> Result<()> {
        let conn = self.db.connect()?;
        ChunkRepository::update_embedding(&conn, chunk_id, embedding).await
    }
    async fn update_chunk_embeddings_batch(&self, updates: &[(String, Vec<f32>)]) -> Result<()> {
        let conn = self.db.connect()?;
        ChunkRepository::update_embeddings_batch(&conn, updates).await
    }
    async fn delete_chunks_by_document_id(&self, document_id: &str) -> Result<()> {
        let conn = self.db.connect()?;
        ChunkRepository::delete_by_document_id(&conn, document_id).await
    }
    async fn search_similar_chunks(&self, embedding: &[f32], limit: u32, threshold: f32, container_tags: Option<&[String]>) -> Result<Vec<ChunkWithDocument>> {
        let conn = self.db.connect()?;
        ChunkRepository::search_similar(&conn, embedding, limit, threshold, container_tags).await
    }
    async fn search_chunks_with_index(&self, embedding: &[f32], limit: u32, container_tags: Option<&[String]>) -> Result<Vec<ChunkWithDocument>> {
        let conn = self.db.connect()?;
        ChunkRepository::search_with_vector_index(&conn, embedding, limit, container_tags).await
    }
    async fn delete_all_chunks(&self) -> Result<u64> {
        let conn = self.db.connect()?;
        let affected = conn.execute("DELETE FROM chunks", ()).await?;
        Ok(affected)
    }
}

#[async_trait]
impl MemoryStore for LibSqlBackend {
    async fn create_memory(&self, memory: &Memory) -> Result<()> {
        let conn = self.db.connect()?;
        MemoryRepository::create(&conn, memory).await
    }
    async fn get_memory_by_id(&self, id: &str) -> Result<Option<Memory>> {
        let conn = self.db.connect()?;
        MemoryRepository::get_by_id(&conn, id).await
    }
    async fn get_memories_by_ids(&self, ids: &[String]) -> Result<Vec<Memory>> {
        let conn = self.db.connect()?;
        MemoryRepository::get_by_ids(&conn, ids).await
    }
    async fn get_memory_by_content(&self, content: &str, container_tag: &str) -> Result<Option<Memory>> {
        let conn = self.db.connect()?;
        MemoryRepository::get_by_content(&conn, content, container_tag).await
    }
    async fn update_memory_to_not_latest(&self, id: &str) -> Result<()> {
        let conn = self.db.connect()?;
        MemoryRepository::update_to_not_latest(&conn, id).await
    }
    async fn forget_memory(&self, id: &str, reason: Option<&str>) -> Result<()> {
        let conn = self.db.connect()?;
        MemoryRepository::forget(&conn, id, reason).await
    }
    async fn update_memory_last_accessed(&self, id: &str) -> Result<()> {
        let conn = self.db.connect()?;
        MemoryRepository::update_last_accessed(&conn, id).await
    }
    async fn update_memory_last_accessed_batch(&self, ids: &[&str]) -> Result<u64> {
        let conn = self.db.connect()?;
        MemoryRepository::update_last_accessed_batch(&conn, ids).await
    }
    async fn update_memory_source_count(&self, id: &str, new_count: i32) -> Result<()> {
        let conn = self.db.connect()?;
        MemoryRepository::update_source_count(&conn, id, new_count).await
    }
    async fn update_memory_version_chain(&self, id: &str, parent_memory_id: &str, root_memory_id: &str, version: i32) -> Result<()> {
        let conn = self.db.connect()?;
        MemoryRepository::update_version_chain(&conn, id, parent_memory_id, root_memory_id, version).await
    }
    async fn update_memory_embedding(&self, memory_id: &str, embedding: &[f32]) -> Result<()> {
        let conn = self.db.connect()?;
        MemoryRepository::update_embedding(&conn, memory_id, embedding).await
    }
    async fn search_similar_memories(&self, embedding: &[f32], limit: u32, threshold: f32, container_tag: Option<&str>, include_forgotten: bool) -> Result<Vec<MemorySearchHit>> {
        let conn = self.db.connect()?;
        MemoryRepository::search_similar(&conn, embedding, limit, threshold, container_tag, include_forgotten).await
    }
    async fn get_memory_children(&self, parent_id: &str) -> Result<Vec<Memory>> {
        let conn = self.db.connect()?;
        MemoryRepository::get_children(&conn, parent_id).await
    }
    async fn get_memory_parents(&self, root_id: &str) -> Result<Vec<Memory>> {
        let conn = self.db.connect()?;
        MemoryRepository::get_parents(&conn, root_id).await
    }
    async fn get_forgetting_candidates(&self, before: DateTime<Utc>) -> Result<Vec<Memory>> {
        let conn = self.db.connect()?;
        MemoryRepository::get_forgetting_candidates(&conn, before).await
    }
    async fn get_seed_memories(&self, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.db.connect()?;
        MemoryRepository::get_seed_memories(&conn, limit).await
    }
    async fn check_inference_exists(&self, source_ids: &[String]) -> Result<bool> {
        let conn = self.db.connect()?;
        MemoryRepository::check_inference_exists(&conn, source_ids).await
    }
    async fn get_user_profile(&self, container_tag: &str, include_dynamic: bool, limit: u32) -> Result<UserProfile> {
        let conn = self.db.connect()?;
        MemoryRepository::get_user_profile(&conn, container_tag, include_dynamic, limit).await
    }
    async fn update_memory_relations(&self, id: &str, new_relations: HashMap<String, MemoryRelationType>) -> Result<()> {
        let conn = self.db.connect()?;
        MemoryRepository::update_relations(&conn, id, new_relations).await
    }
    async fn add_memory_relation(&self, id: &str, related_id: &str, relation_type: MemoryRelationType) -> Result<()> {
        let conn = self.db.connect()?;
        MemoryRepository::add_relation(&conn, id, related_id, relation_type).await
    }
    async fn get_graph_neighborhood(&self, id: &str, depth: u32, max_nodes: u32, relation_types: Option<&[GraphEdgeType]>) -> Result<GraphData> {
        let conn = self.db.connect()?;
        MemoryRepository::get_graph_neighborhood(&conn, id, depth, max_nodes, relation_types).await
    }
    async fn get_container_graph(&self, container_tag: &str, max_nodes: u32) -> Result<GraphData> {
        let conn = self.db.connect()?;
        MemoryRepository::get_container_graph(&conn, container_tag, max_nodes).await
    }
    async fn get_cached_profile(&self, container_tag: &str) -> Result<Option<CachedProfile>> {
        let conn = self.db.connect()?;
        MemoryRepository::get_cached_profile(&conn, container_tag).await
    }
    async fn upsert_cached_profile(
        &self,
        container_tag: &str,
        narrative: Option<&str>,
        summary: Option<&str>,
    ) -> Result<()> {
        let conn = self.db.connect()?;
        MemoryRepository::upsert_cached_profile(&conn, container_tag, narrative, summary).await
    }

    async fn get_episode_decay_candidates(&self) -> Result<Vec<EpisodeDecayCandidate>> {
        let conn = self.db.connect()?;
        let mut rows = conn
            .query(
                "SELECT id, memory, space_id, last_accessed, created_at FROM memories WHERE is_forgotten = 0 AND is_static = 0 AND memory_type = 'episode' AND is_latest = 1",
                (),
            )
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(EpisodeDecayCandidate {
                id: row.get(0)?,
                memory: row.get(1)?,
                space_id: row.get(2)?,
                last_accessed: row.get(3)?,
                created_at: row.get(4)?,
            });
        }
        Ok(results)
    }

    async fn set_memory_forget_after(&self, id: &str, forget_after: DateTime<Utc>) -> Result<u64> {
        let conn = self.db.connect()?;
        let ts = forget_after.to_rfc3339();
        let now = Utc::now().to_rfc3339();

        let res = conn
            .execute(
                "UPDATE memories SET forget_after = ?2, updated_at = ?3 WHERE id = ?1 AND is_forgotten = 0 AND is_static = 0",
                params![id, ts, now],
            )
            .await?;

        Ok(res)
    }

    async fn get_active_container_tags(&self) -> Result<Vec<String>> {
        let conn = self.db.connect()?;
        let mut rows = conn
            .query(
                "SELECT DISTINCT container_tag FROM memories WHERE is_forgotten = 0 AND is_latest = 1 AND container_tag IS NOT NULL",
                (),
            )
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(row.get(0)?);
        }
        Ok(results)
    }

    async fn get_max_memory_updated_at(&self, container_tag: &str) -> Result<Option<DateTime<Utc>>> {
        let conn = self.db.connect()?;
        let row = conn
            .query(
                "SELECT MAX(updated_at) FROM memories WHERE container_tag = ?1 AND is_forgotten = 0 AND is_latest = 1",
                params![container_tag],
            )
            .await?
            .next()
            .await?;

        if let Some(row) = row {
            let val: Option<String> = row.get(0)?;
            if let Some(s) = val {
                if let Ok(dt) = DateTime::parse_from_rfc3339(&s) {
                    return Ok(Some(dt.with_timezone(&Utc)));
                }
            }
        }
        Ok(None)
    }
}

#[async_trait]
impl MemorySourceStore for LibSqlBackend {
    async fn create_memory_source(&self, memory_id: &str, document_id: &str, chunk_id: Option<&str>) -> Result<MemorySource> {
        let conn = self.db.connect()?;
        MemorySourcesRepository::create(&conn, memory_id, document_id, chunk_id).await
    }
    async fn get_sources_by_memory(&self, memory_id: &str) -> Result<Vec<MemorySource>> {
        let conn = self.db.connect()?;
        MemorySourcesRepository::get_by_memory(&conn, memory_id).await
    }
    async fn get_sources_by_document(&self, document_id: &str) -> Result<Vec<MemorySource>> {
        let conn = self.db.connect()?;
        MemorySourcesRepository::get_by_document(&conn, document_id).await
    }
}

#[async_trait]
impl MetadataStore for LibSqlBackend {
    async fn get_metadata(&self, key: &str) -> Result<Option<String>> {
        let conn = self.db.connect()?;
        MetadataRepository::get(&conn, key).await
    }
    async fn set_metadata(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.db.connect()?;
        MetadataRepository::set(&conn, key, value).await
    }
    async fn get_embedding_dimensions(&self) -> Result<Option<usize>> {
        let conn = self.db.connect()?;
        MetadataRepository::get_embedding_dimensions(&conn).await
    }
    async fn set_embedding_dimensions(&self, dims: usize) -> Result<()> {
        let conn = self.db.connect()?;
        MetadataRepository::set_embedding_dimensions(&conn, dims).await
    }
}

#[async_trait]
impl DatabaseBackend for LibSqlBackend {
    async fn initialize(&self) -> Result<()> {
        let conn = self.db.connect()?;
        crate::db::schema::init_schema(&conn).await
    }
    async fn sync(&self) -> Result<()> {
        self.db.sync().await
    }

    async fn get_container_filter(&self, tag: &str) -> Result<Option<ContainerFilter>> {
        let conn = self.db.connect()?;
        let row = conn
            .query(
                "SELECT tag, should_llm_filter, filter_prompt FROM container_tags WHERE tag = ?1",
                params![tag],
            )
            .await?
            .next()
            .await?;

        if let Some(row) = row {
            let should_llm_filter: i64 = row.get(1)?;
            let filter_prompt: Option<String> = row.get(2)?;

            Ok(Some(ContainerFilter {
                tag: tag.to_string(),
                should_llm_filter: should_llm_filter != 0,
                filter_prompt,
            }))
        } else {
            Ok(None)
        }
    }

    async fn set_container_filter(&self, tag: &str, filter: &ContainerFilter) -> Result<()> {
        let conn = self.db.connect()?;
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "UPDATE container_tags SET should_llm_filter = ?2, filter_prompt = ?3, updated_at = ?4 WHERE tag = ?1",
            params![tag, if filter.should_llm_filter { 1 } else { 0 }, filter.filter_prompt.clone(), now],
        )
        .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatabaseConfig;
    use crate::db::connection::Database;

    async fn setup_test_db() -> LibSqlBackend {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let thread_id = std::thread::current().id();
        
        let config = DatabaseConfig {
            url: format!("file:/tmp/momo_test_db_{:?}_{}?mode=memory&cache=shared", thread_id, timestamp),
            auth_token: None,
            local_path: None,
        };
        let db = Database::new(&config).await.expect("Failed to create database");
        let backend = LibSqlBackend::new(db);
        backend
    }

    #[tokio::test]
    async fn test_get_container_filter_non_existent() {
        let backend = setup_test_db().await;
        
        let result = backend.get_container_filter("non_existent_tag").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_set_and_get_container_filter() {
        let backend = setup_test_db().await;
        let tag = "test_container";

        let conn = backend.db.connect().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO container_tags (tag, metadata, document_count, memory_count, created_at, updated_at) VALUES (?1, '{}', 0, 0, ?2, ?2)",
            params![tag, now],
        )
        .await
        .unwrap();

        let filter = ContainerFilter {
            tag: tag.to_string(),
            should_llm_filter: true,
            filter_prompt: Some("technical only".to_string()),
        };

        backend.set_container_filter(tag, &filter).await.unwrap();

        let retrieved = backend.get_container_filter(tag).await.unwrap();
        assert!(retrieved.is_some());
        
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.tag, tag);
        assert_eq!(retrieved.should_llm_filter, true);
        assert_eq!(retrieved.filter_prompt, Some("technical only".to_string()));
    }

    #[tokio::test]
    async fn test_update_existing_filter() {
        let backend = setup_test_db().await;
        let tag = "update_test";

        let conn = backend.db.connect().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO container_tags (tag, metadata, document_count, memory_count, should_llm_filter, filter_prompt, created_at, updated_at) VALUES (?1, '{}', 0, 0, 1, 'old prompt', ?2, ?2)",
            params![tag, now],
        )
        .await
        .unwrap();

        let new_filter = ContainerFilter {
            tag: tag.to_string(),
            should_llm_filter: false,
            filter_prompt: None,
        };

        backend.set_container_filter(tag, &new_filter).await.unwrap();

        let retrieved = backend.get_container_filter(tag).await.unwrap().unwrap();
        assert_eq!(retrieved.should_llm_filter, false);
        assert_eq!(retrieved.filter_prompt, None);
    }

    #[tokio::test]
    async fn test_filter_with_llm_enabled() {
        let backend = setup_test_db().await;
        let tag = "llm_enabled_test";

        let conn = backend.db.connect().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO container_tags (tag, metadata, document_count, memory_count, created_at, updated_at) VALUES (?1, '{}', 0, 0, ?2, ?2)",
            params![tag, now],
        )
        .await
        .unwrap();

        let filter = ContainerFilter {
            tag: tag.to_string(),
            should_llm_filter: true,
            filter_prompt: Some("Only work-related memories".to_string()),
        };

        backend.set_container_filter(tag, &filter).await.unwrap();

        let retrieved = backend.get_container_filter(tag).await.unwrap().unwrap();
        assert!(retrieved.should_llm_filter);
        assert_eq!(retrieved.filter_prompt, Some("Only work-related memories".to_string()));
    }

    #[tokio::test]
    async fn test_filter_with_llm_disabled() {
        let backend = setup_test_db().await;
        let tag = "llm_disabled_test";

        let conn = backend.db.connect().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO container_tags (tag, metadata, document_count, memory_count, created_at, updated_at) VALUES (?1, '{}', 0, 0, ?2, ?2)",
            params![tag, now],
        )
        .await
        .unwrap();

        let filter = ContainerFilter {
            tag: tag.to_string(),
            should_llm_filter: false,
            filter_prompt: None,
        };

        backend.set_container_filter(tag, &filter).await.unwrap();

        let retrieved = backend.get_container_filter(tag).await.unwrap().unwrap();
        assert!(!retrieved.should_llm_filter);
        assert!(retrieved.filter_prompt.is_none());
    }

    #[tokio::test]
    async fn test_queue_reprocess_status() {
        let backend = setup_test_db().await;
        let conn = backend.db.connect().unwrap();
        let now = chrono::Utc::now().to_rfc3339();

        // Insert documents with various statuses
        for (id, status) in &[
            ("doc_done_1", "done"),
            ("doc_done_2", "done"),
            ("doc_queued", "queued"),
            ("doc_extracting", "extracting"),
        ] {
            conn.execute(
                "INSERT INTO documents (id, doc_type, status, created_at, updated_at) VALUES (?1, 'text', ?2, ?3, ?3)",
                params![*id, *status, now.clone()],
            )
            .await
            .unwrap();
        }

        // Queue all done documents for reprocessing
        let affected = backend.queue_all_documents_for_reprocessing().await.unwrap();
        assert_eq!(affected, 2, "Should have updated exactly 2 'done' documents");

        // Verify the done documents are now queued
        for id in &["doc_done_1", "doc_done_2"] {
            let mut rows = conn
                .query("SELECT status FROM documents WHERE id = ?1", params![*id])
                .await
                .unwrap();
            let row = rows.next().await.unwrap().unwrap();
            let status: String = row.get(0).unwrap();
            assert_eq!(status, "queued", "Document {id} should be 'queued'");
        }

        // Verify non-done documents were not affected
        let mut rows = conn
            .query("SELECT status FROM documents WHERE id = 'doc_queued'", ())
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let status: String = row.get(0).unwrap();
        assert_eq!(status, "queued", "Already-queued document should remain 'queued'");

        let mut rows = conn
            .query("SELECT status FROM documents WHERE id = 'doc_extracting'", ())
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let status: String = row.get(0).unwrap();
        assert_eq!(status, "extracting", "Extracting document should remain 'extracting'");
    }
}
