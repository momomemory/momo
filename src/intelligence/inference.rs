use std::collections::HashMap;
use std::sync::Arc;

use nanoid::nanoid;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::config::InferenceConfig;
use crate::db::DatabaseBackend;
use crate::embeddings::EmbeddingProvider;
use crate::error::{MomoError, Result};
use crate::llm::{prompts, LlmProvider};
use crate::models::{Memory, MemoryRelationType, MemoryType};

/// Statistics from a single inference run
#[derive(Debug, Clone, Default)]
pub struct InferenceStats {
    /// Number of seed memories processed
    pub seeds_processed: usize,
    /// Number of inferences successfully created
    pub inferences_created: usize,
    /// Number of inferences skipped due to deduplication
    pub duplicates_skipped: usize,
    /// Number of inferences skipped due to low confidence
    pub low_confidence_skipped: usize,
    /// Number of errors during inference generation
    pub errors: usize,
}

/// LLM response for a generated inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatedInference {
    /// Synthesized content
    pub content: String,
    /// Explanation of how the inference was derived
    pub reasoning: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// IDs of source memories that support this inference
    pub source_ids: Vec<String>,
}

/// Engine that derives new inferred memories from existing ones.
///
/// Runs as a background job, selecting seed memories, finding related memories
/// via vector search, and using an LLM to synthesize new insights.
///
/// Key guardrails:
/// - Never uses `is_inference=true` memories as seeds (prevents feedback loops)
/// - Never uses Episode memories as sources (they decay)
/// - Filters by confidence threshold
/// - Deduplicates by source memory IDs
#[derive(Clone)]
pub struct InferenceEngine {
    db: Arc<dyn DatabaseBackend>,
    llm: LlmProvider,
    embeddings: EmbeddingProvider,
    config: InferenceConfig,
}

impl InferenceEngine {
    /// Create a new InferenceEngine
    pub fn new(
        db: Arc<dyn DatabaseBackend>,
        llm: LlmProvider,
        embeddings: EmbeddingProvider,
        config: InferenceConfig,
    ) -> Self {
        Self {
            db,
            llm,
            embeddings,
            config,
        }
    }

    /// Main entry point: run a single pass of the inference engine.
    ///
    /// 1. Select eligible seed memories
    /// 2. For each seed, find related memories via vector search
    /// 3. Ask the LLM to synthesize an inference
    /// 4. If confidence passes threshold and not a duplicate, store it
    pub async fn run_once(&self) -> Result<InferenceStats> {
        info!("Starting inference engine run");

        if !self.llm.is_available() {
            warn!("LLM unavailable, skipping inference run");
            return Ok(InferenceStats::default());
        }

        let seeds = self.select_seed_memories().await?;
        let seed_count = seeds.len();

        if seed_count == 0 {
            info!("No eligible seed memories for inference");
            return Ok(InferenceStats::default());
        }

        debug!("Found {} seed memories for inference", seed_count);

        let mut stats = InferenceStats::default();

        for seed in seeds {
            if stats.inferences_created >= self.config.max_per_run {
                debug!(
                    "Reached max_per_run limit ({}), stopping",
                    self.config.max_per_run
                );
                break;
            }

            stats.seeds_processed += 1;

            // Embed the seed memory for vector search
            let embedding = match self.embeddings.embed_passage(&seed.memory).await {
                Ok(emb) => emb,
                Err(e) => {
                    error!(seed_id = %seed.id, error = %e, "Failed to embed seed memory");
                    stats.errors += 1;
                    continue;
                }
            };

            // Find related memories
            let related = match self
                .find_related_memories(&seed.id, &embedding, seed.container_tag.as_deref())
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    error!(seed_id = %seed.id, error = %e, "Failed to find related memories");
                    stats.errors += 1;
                    continue;
                }
            };

            if related.is_empty() {
                debug!(seed_id = %seed.id, "No related memories found, skipping");
                continue;
            }

            // Generate inference via LLM
            let inference = match self.generate_inference(&seed, &related).await {
                Ok(Some(inf)) => inf,
                Ok(None) => {
                    debug!(seed_id = %seed.id, "LLM did not produce a valid inference");
                    stats.errors += 1;
                    continue;
                }
                Err(e) => {
                    error!(seed_id = %seed.id, error = %e, "Failed to generate inference");
                    stats.errors += 1;
                    continue;
                }
            };

            // Check confidence threshold
            if inference.confidence < self.config.confidence_threshold {
                debug!(
                    seed_id = %seed.id,
                    confidence = inference.confidence,
                    threshold = self.config.confidence_threshold,
                    "Inference below confidence threshold, skipping"
                );
                stats.low_confidence_skipped += 1;
                continue;
            }

            // Deduplication check: have we already created an inference from these exact sources?
            let mut all_source_ids: Vec<String> = inference.source_ids.clone();
            if !all_source_ids.contains(&seed.id) {
                all_source_ids.push(seed.id.clone());
            }

            if self.check_inference_exists(&all_source_ids).await? {
                debug!(
                    seed_id = %seed.id,
                    "Duplicate inference detected, skipping"
                );
                stats.duplicates_skipped += 1;
                continue;
            }

            // Store the inference
            match self
                .create_inference_memory(&inference, &seed, &all_source_ids)
                .await
            {
                Ok(_memory) => {
                    stats.inferences_created += 1;
                    info!(
                        seed_id = %seed.id,
                        confidence = inference.confidence,
                        sources = all_source_ids.len(),
                        "Created new inference memory"
                    );
                }
                Err(e) => {
                    error!(seed_id = %seed.id, error = %e, "Failed to store inference memory");
                    stats.errors += 1;
                }
            }
        }

        info!(
            seeds_processed = stats.seeds_processed,
            inferences_created = stats.inferences_created,
            duplicates_skipped = stats.duplicates_skipped,
            low_confidence = stats.low_confidence_skipped,
            errors = stats.errors,
            "Inference engine run complete"
        );

        Ok(stats)
    }

    /// Select seed memories eligible for inference.
    ///
    /// Excludes:
    /// - Memories that are already inferences (`is_inference = true`)
    /// - Episode memories (if `exclude_episodes` is enabled)
    /// - Forgotten memories
    /// - Non-latest versions
    async fn select_seed_memories(&self) -> Result<Vec<Memory>> {
        let limit = self.config.seed_limit;

        let mut seeds = self.db.get_seed_memories(limit).await?;

        if self.config.exclude_episodes {
            seeds.retain(|m| m.memory_type != MemoryType::Episode);
        }

        Ok(seeds)
    }

    /// Find memories related to a seed via vector similarity search.
    ///
    /// Excludes:
    /// - The seed memory itself
    /// - Inference memories (prevents circular derivation)
    /// - Episode memories (if `exclude_episodes` is enabled)
    async fn find_related_memories(
        &self,
        seed_id: &str,
        embedding: &[f32],
        container_tag: Option<&str>,
    ) -> Result<Vec<Memory>> {
        let hits = self
            .db
            .search_similar_memories(
                embedding,
                self.config.candidate_count as u32,
                self.config.confidence_threshold,
                container_tag,
                false,
            )
            .await?;

        let exclude_episodes = self.config.exclude_episodes;

        let related: Vec<Memory> = hits
            .into_iter()
            .filter(|hit| hit.memory.id != seed_id)
            .filter(|hit| !hit.memory.is_inference)
            .filter(|hit| {
                if exclude_episodes {
                    hit.memory.memory_type != MemoryType::Episode
                } else {
                    true
                }
            })
            .map(|hit| hit.memory)
            .collect();

        Ok(related)
    }

    /// Use the LLM to generate a synthesized inference from a seed and related memories.
    ///
    /// Returns `None` if the LLM response cannot be parsed.
    async fn generate_inference(
        &self,
        seed: &Memory,
        related: &[Memory],
    ) -> Result<Option<CreatedInference>> {
        let related_pairs: Vec<(&str, &str)> = related
            .iter()
            .map(|m| (m.id.as_str(), m.memory.as_str()))
            .collect();

        let prompt = prompts::inference_generation_prompt(&seed.memory, &related_pairs);

        match self
            .llm
            .complete_structured::<CreatedInference>(&prompt)
            .await
        {
            Ok(inference) => Ok(Some(inference)),
            Err(MomoError::LlmUnavailable(reason)) => {
                warn!(%reason, "LLM unavailable during inference generation");
                Ok(None)
            }
            Err(e) => {
                error!(error = %e, "LLM inference generation failed");
                Ok(None)
            }
        }
    }

    /// Store the inference as a new Memory with `is_inference = true` and
    /// `Derives` relations to all source memories.
    async fn create_inference_memory(
        &self,
        inference: &CreatedInference,
        seed: &Memory,
        source_ids: &[String],
    ) -> Result<Memory> {
        let id = nanoid!();

        // Build Derives relations to all source memories
        let mut relations = HashMap::new();
        for source_id in source_ids {
            relations.insert(source_id.clone(), MemoryRelationType::Derives);
        }

        let memory = Memory {
            id: id.clone(),
            memory: inference.content.clone(),
            space_id: seed.space_id.clone(),
            container_tag: seed.container_tag.clone(),
            version: 1,
            is_latest: true,
            parent_memory_id: None,
            root_memory_id: None,
            memory_relations: relations,
            source_count: source_ids.len() as i32,
            is_inference: true,
            is_forgotten: false,
            is_static: false,
            forget_after: None,
            forget_reason: None,
            memory_type: MemoryType::Fact,
            last_accessed: None,
            confidence: Some(inference.confidence as f64),
            metadata: Default::default(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        self.db.create_memory(&memory).await?;

        // Embed and store the embedding for the new inference
        match self.embeddings.embed_passage(&memory.memory).await {
            Ok(embedding) => {
                self.db.update_memory_embedding(&id, &embedding).await?;
            }
            Err(e) => {
                warn!(memory_id = %id, error = %e, "Failed to embed inference memory");
            }
        }

        Ok(memory)
    }

    /// Check if an inference already exists that was derived from the exact same set of source IDs.
    ///
    /// We normalize the source set (sort + deduplicate) and compare against existing
    /// inference memories' `memory_relations` keys.
    async fn check_inference_exists(&self, source_ids: &[String]) -> Result<bool> {
        self.db.check_inference_exists(source_ids).await
    }

    /// Get the configured interval in seconds
    pub fn interval_secs(&self) -> u64 {
        self.config.interval_secs
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use libsql::Builder;
    use serde_json::json;
    use tempfile::TempDir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::config::{DatabaseConfig, EmbeddingsConfig, LlmConfig};
    use crate::db::repository::MemoryRepository;
    use crate::db::{Database, LibSqlBackend};

    // ── Test helpers ──────────────────────────────────────────────────

    async fn test_embeddings_provider() -> EmbeddingProvider {
        let config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };

        EmbeddingProvider::new(&config).expect("failed to create test embeddings provider")
    }

    fn test_llm_unavailable() -> LlmProvider {
        LlmProvider::unavailable("test")
    }

    fn test_llm_provider(base_url: String) -> LlmProvider {
        let config = LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(base_url),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,
            filter_prompt: None,
        };

        LlmProvider::new(Some(&config))
    }

    fn llm_response(content: &str) -> serde_json::Value {
        json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        })
    }

    fn test_config() -> InferenceConfig {
        InferenceConfig {
            enabled: true,
            interval_secs: 60,
            confidence_threshold: 0.7,
            max_per_run: 50,
            candidate_count: 5,
            seed_limit: 50,
            exclude_episodes: true,
        }
    }

    async fn test_database() -> (libsql::Connection, Arc<dyn DatabaseBackend>, TempDir) {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("inference_test.db");
        let raw_db = Builder::new_local(db_path.display().to_string())
            .build()
            .await
            .expect("failed to create raw database");
        let conn = raw_db.connect().expect("connect should work");
        conn.execute(
            r#"
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                memory TEXT NOT NULL,
                space_id TEXT NOT NULL,
                container_tag TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                is_latest INTEGER NOT NULL DEFAULT 1,
                parent_memory_id TEXT,
                root_memory_id TEXT,
                memory_relations TEXT NOT NULL DEFAULT '{}',
                source_count INTEGER NOT NULL DEFAULT 0,
                is_inference INTEGER NOT NULL DEFAULT 0,
                is_forgotten INTEGER NOT NULL DEFAULT 0,
                is_static INTEGER NOT NULL DEFAULT 0,
                forget_after TEXT,
                forget_reason TEXT,
                memory_type TEXT NOT NULL DEFAULT 'fact',
                last_accessed TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                embedding F32_BLOB(384)
            )
            "#,
            (),
        )
        .await
        .expect("memories table should be created");

        let config = DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };

        let db = Database::new(&config)
            .await
            .expect("failed to create test database");

        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db));
        (conn, backend, temp_dir)
    }

    fn test_memory(id: &str, memory: &str, container_tag: Option<&str>) -> Memory {
        Memory {
            id: id.to_string(),
            memory: memory.to_string(),
            space_id: "default".to_string(),
            container_tag: container_tag.map(str::to_string),
            version: 1,
            is_latest: true,
            parent_memory_id: None,
            root_memory_id: None,
            memory_relations: Default::default(),
            source_count: 1,
            is_inference: false,
            is_forgotten: false,
            is_static: false,
            forget_after: None,
            forget_reason: None,
            memory_type: MemoryType::Fact,
            last_accessed: None,
            confidence: None,
            metadata: Default::default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn test_inference_memory(id: &str, memory: &str, source_ids: &[&str]) -> Memory {
        let mut relations = HashMap::new();
        for src in source_ids {
            relations.insert(src.to_string(), MemoryRelationType::Derives);
        }
        Memory {
            id: id.to_string(),
            memory: memory.to_string(),
            space_id: "default".to_string(),
            container_tag: None,
            version: 1,
            is_latest: true,
            parent_memory_id: None,
            root_memory_id: None,
            memory_relations: relations,
            source_count: source_ids.len() as i32,
            is_inference: true,
            is_forgotten: false,
            is_static: false,
            forget_after: None,
            forget_reason: None,
            memory_type: MemoryType::Fact,
            last_accessed: None,
            confidence: None,
            metadata: Default::default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn test_episode_memory(id: &str, memory: &str) -> Memory {
        Memory {
            id: id.to_string(),
            memory: memory.to_string(),
            space_id: "default".to_string(),
            container_tag: None,
            version: 1,
            is_latest: true,
            parent_memory_id: None,
            root_memory_id: None,
            memory_relations: Default::default(),
            source_count: 1,
            is_inference: false,
            is_forgotten: false,
            is_static: false,
            forget_after: None,
            forget_reason: None,
            memory_type: MemoryType::Episode,
            last_accessed: None,
            confidence: None,
            metadata: Default::default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    // ── Unit tests ────────────────────────────────────────────────────

    #[test]
    fn test_inference_stats_default() {
        let stats = InferenceStats::default();
        assert_eq!(stats.seeds_processed, 0);
        assert_eq!(stats.inferences_created, 0);
        assert_eq!(stats.duplicates_skipped, 0);
        assert_eq!(stats.low_confidence_skipped, 0);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_created_inference_deserializes() {
        let json = r#"{
            "content": "User prefers dark themes",
            "reasoning": "Multiple sources confirm preference",
            "confidence": 0.9,
            "source_ids": ["mem_1", "mem_2"]
        }"#;

        let inf: CreatedInference = serde_json::from_str(json).unwrap();
        assert_eq!(inf.content, "User prefers dark themes");
        assert_eq!(inf.confidence, 0.9);
        assert_eq!(inf.source_ids.len(), 2);
    }

    #[test]
    fn test_created_inference_serializes() {
        let inf = CreatedInference {
            content: "Test inference".to_string(),
            reasoning: "Because reasons".to_string(),
            confidence: 0.85,
            source_ids: vec!["a".to_string(), "b".to_string()],
        };

        let json = serde_json::to_string(&inf).unwrap();
        assert!(json.contains("Test inference"));
        assert!(json.contains("0.85"));
    }

    #[test]
    fn test_clone_impl() {
        let config = test_config();
        let llm = test_llm_unavailable();

        // We can't easily construct EmbeddingProvider or Database synchronously,
        // so just verify InferenceConfig and LlmProvider clone correctly
        let config2 = config.clone();
        assert_eq!(config2.confidence_threshold, config.confidence_threshold);

        let llm2 = llm.clone();
        assert!(!llm2.is_available());
    }

    #[tokio::test]
    async fn test_run_once_returns_empty_when_llm_unavailable() {
        let (_conn, db, _temp_dir) = test_database().await;
        let embeddings = test_embeddings_provider().await;
        let engine = InferenceEngine::new(db, test_llm_unavailable(), embeddings, test_config());

        let stats = engine.run_once().await.expect("run_once should not fail");

        assert_eq!(stats.seeds_processed, 0);
        assert_eq!(stats.inferences_created, 0);
    }

    #[tokio::test]
    async fn test_run_once_returns_empty_when_no_seeds() {
        let llm_server = MockServer::start().await;
        // LLM mock that never gets called (no seeds to process)
        let (_conn, db, _temp_dir) = test_database().await;
        let embeddings = test_embeddings_provider().await;
        let engine = InferenceEngine::new(
            db,
            test_llm_provider(llm_server.uri()),
            embeddings,
            test_config(),
        );

        let stats = engine.run_once().await.expect("run_once should not fail");

        assert_eq!(stats.seeds_processed, 0);
        assert_eq!(stats.inferences_created, 0);
    }

    #[tokio::test]
    async fn test_select_seed_memories_excludes_inferences() {
        let (conn, db, _temp_dir) = test_database().await;

        // Create a regular memory and an inference memory
        let regular = test_memory("mem_reg", "Regular memory", None);
        MemoryRepository::create(&conn, &regular).await.unwrap();
        let embedding = vec![0.1_f32; 384];
        MemoryRepository::update_embedding(&conn, "mem_reg", &embedding)
            .await
            .unwrap();

        let inference = test_inference_memory("mem_inf", "Inferred memory", &["mem_reg"]);
        MemoryRepository::create(&conn, &inference).await.unwrap();
        MemoryRepository::update_embedding(&conn, "mem_inf", &embedding)
            .await
            .unwrap();

        let embeddings = test_embeddings_provider().await;
        let engine = InferenceEngine::new(db, test_llm_unavailable(), embeddings, test_config());

        let seeds = engine.select_seed_memories().await.unwrap();

        // Only the regular memory should be a seed
        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].id, "mem_reg");
    }

    #[tokio::test]
    async fn test_select_seed_memories_excludes_episodes_when_configured() {
        let (conn, db, _temp_dir) = test_database().await;

        let fact = test_memory("mem_fact", "A fact", None);
        MemoryRepository::create(&conn, &fact).await.unwrap();
        let embedding = vec![0.1_f32; 384];
        MemoryRepository::update_embedding(&conn, "mem_fact", &embedding)
            .await
            .unwrap();

        let episode = test_episode_memory("mem_ep", "An episode");
        MemoryRepository::create(&conn, &episode).await.unwrap();
        MemoryRepository::update_embedding(&conn, "mem_ep", &embedding)
            .await
            .unwrap();

        let embeddings = test_embeddings_provider().await;
        let config = test_config(); // exclude_episodes = true
        let engine = InferenceEngine::new(db, test_llm_unavailable(), embeddings, config);

        let seeds = engine.select_seed_memories().await.unwrap();

        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].id, "mem_fact");
    }

    #[tokio::test]
    async fn test_select_seed_memories_includes_episodes_when_not_excluded() {
        let (conn, db, _temp_dir) = test_database().await;

        let fact = test_memory("mem_fact", "A fact", None);
        MemoryRepository::create(&conn, &fact).await.unwrap();
        let embedding = vec![0.1_f32; 384];
        MemoryRepository::update_embedding(&conn, "mem_fact", &embedding)
            .await
            .unwrap();

        let episode = test_episode_memory("mem_ep", "An episode");
        MemoryRepository::create(&conn, &episode).await.unwrap();
        MemoryRepository::update_embedding(&conn, "mem_ep", &embedding)
            .await
            .unwrap();

        let embeddings = test_embeddings_provider().await;
        let mut config = test_config();
        config.exclude_episodes = false;
        let engine = InferenceEngine::new(db, test_llm_unavailable(), embeddings, config);

        let seeds = engine.select_seed_memories().await.unwrap();

        assert_eq!(seeds.len(), 2);
    }

    #[tokio::test]
    async fn test_check_inference_exists_returns_true_for_duplicate() {
        let (conn, db, _temp_dir) = test_database().await;

        // Create an existing inference memory derived from mem_a and mem_b
        let inference = test_inference_memory("inf_1", "Existing inference", &["mem_a", "mem_b"]);
        MemoryRepository::create(&conn, &inference).await.unwrap();

        let embeddings = test_embeddings_provider().await;
        let engine = InferenceEngine::new(db, test_llm_unavailable(), embeddings, test_config());

        // Same source IDs (different order) should be detected as duplicate
        let exists = engine
            .check_inference_exists(&["mem_b".to_string(), "mem_a".to_string()])
            .await
            .unwrap();

        assert!(exists);
    }

    #[tokio::test]
    async fn test_check_inference_exists_returns_false_for_new_sources() {
        let (conn, db, _temp_dir) = test_database().await;

        let inference = test_inference_memory("inf_1", "Existing inference", &["mem_a", "mem_b"]);
        MemoryRepository::create(&conn, &inference).await.unwrap();

        let embeddings = test_embeddings_provider().await;
        let engine = InferenceEngine::new(db, test_llm_unavailable(), embeddings, test_config());

        // Different source IDs should not be a duplicate
        let exists = engine
            .check_inference_exists(&["mem_a".to_string(), "mem_c".to_string()])
            .await
            .unwrap();

        assert!(!exists);
    }

    #[tokio::test]
    async fn test_find_related_memories_excludes_self_and_inferences() {
        let (conn, db, _temp_dir) = test_database().await;

        // Create memories
        let seed = test_memory("seed_1", "Seed memory about Rust", Some("user_1"));
        MemoryRepository::create(&conn, &seed).await.unwrap();
        let embedding = vec![0.1_f32; 384];
        MemoryRepository::update_embedding(&conn, "seed_1", &embedding)
            .await
            .unwrap();

        let related = test_memory("rel_1", "Related fact about programming", Some("user_1"));
        MemoryRepository::create(&conn, &related).await.unwrap();
        MemoryRepository::update_embedding(&conn, "rel_1", &embedding)
            .await
            .unwrap();

        let inf = test_inference_memory("inf_1", "An inference", &["other"]);
        MemoryRepository::create(&conn, &inf).await.unwrap();
        MemoryRepository::update_embedding(&conn, "inf_1", &embedding)
            .await
            .unwrap();

        let embeddings = test_embeddings_provider().await;
        let engine = InferenceEngine::new(db, test_llm_unavailable(), embeddings, test_config());

        let found = engine
            .find_related_memories("seed_1", &embedding, Some("user_1"))
            .await
            .unwrap();

        // Should include rel_1 but NOT seed_1 (self) or inf_1 (inference)
        assert!(found.iter().any(|m| m.id == "rel_1"));
        assert!(!found.iter().any(|m| m.id == "seed_1"));
        assert!(!found.iter().any(|m| m.id == "inf_1"));
    }

    #[tokio::test]
    async fn test_run_once_creates_inference_end_to_end() {
        let llm_server = MockServer::start().await;

        // Mock LLM response with a valid inference
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"content": "User is a Rust developer who prefers dark mode", "reasoning": "Combined facts", "confidence": 0.9, "source_ids": ["mem_2"]}"#,
            )))
            .mount(&llm_server)
            .await;

        let (conn, db, _temp_dir) = test_database().await;
        let embeddings = test_embeddings_provider().await;

        // Create seed and related memories with real embeddings
        let seed = test_memory("mem_1", "User is a developer", Some("user_1"));
        MemoryRepository::create(&conn, &seed).await.unwrap();
        let seed_embedding = embeddings
            .embed_passage("User is a developer")
            .await
            .unwrap();
        MemoryRepository::update_embedding(&conn, "mem_1", &seed_embedding)
            .await
            .unwrap();

        let related = test_memory("mem_2", "User prefers dark mode", Some("user_1"));
        MemoryRepository::create(&conn, &related).await.unwrap();
        let related_embedding = embeddings
            .embed_passage("User prefers dark mode")
            .await
            .unwrap();
        MemoryRepository::update_embedding(&conn, "mem_2", &related_embedding)
            .await
            .unwrap();

        let engine = InferenceEngine::new(
            Arc::clone(&db),
            test_llm_provider(llm_server.uri()),
            embeddings,
            test_config(),
        );

        let stats = engine.run_once().await.expect("run_once should succeed");

        // Should have processed seeds and created at least one inference
        assert!(stats.seeds_processed > 0);
        assert!(stats.inferences_created > 0);
        assert_eq!(stats.errors, 0);

        // Verify inference was stored in DB
        let mut rows = conn
            .query(
                "SELECT id, is_inference FROM memories WHERE is_inference = 1",
                (),
            )
            .await
            .unwrap();

        let mut inference_count = 0;
        while let Some(_row) = rows.next().await.unwrap() {
            inference_count += 1;
        }
        assert!(inference_count > 0);
    }

    #[tokio::test]
    async fn test_run_once_skips_low_confidence() {
        let llm_server = MockServer::start().await;

        // Mock LLM with low confidence response
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"content": "Weak inference", "reasoning": "Not sure", "confidence": 0.3, "source_ids": ["mem_2"]}"#,
            )))
            .mount(&llm_server)
            .await;

        let (conn, db, _temp_dir) = test_database().await;
        let embeddings = test_embeddings_provider().await;

        let seed = test_memory("mem_1", "User mentioned something", Some("user_1"));
        MemoryRepository::create(&conn, &seed).await.unwrap();
        let seed_embedding = embeddings
            .embed_passage("User mentioned something")
            .await
            .unwrap();
        MemoryRepository::update_embedding(&conn, "mem_1", &seed_embedding)
            .await
            .unwrap();

        let related = test_memory("mem_2", "Another thing mentioned", Some("user_1"));
        MemoryRepository::create(&conn, &related).await.unwrap();
        let related_embedding = embeddings
            .embed_passage("Another thing mentioned")
            .await
            .unwrap();
        MemoryRepository::update_embedding(&conn, "mem_2", &related_embedding)
            .await
            .unwrap();

        let engine = InferenceEngine::new(
            db,
            test_llm_provider(llm_server.uri()),
            embeddings,
            test_config(),
        );

        let stats = engine.run_once().await.expect("run_once should succeed");

        assert_eq!(stats.inferences_created, 0);
        assert!(stats.low_confidence_skipped > 0);
    }

    #[tokio::test]
    async fn test_run_once_respects_max_per_run() {
        let config = InferenceConfig {
            max_per_run: 1,
            ..test_config()
        };

        let llm_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"content": "Valid inference", "reasoning": "Good", "confidence": 0.95, "source_ids": ["mem_2"]}"#,
            )))
            .mount(&llm_server)
            .await;

        let (conn, db, _temp_dir) = test_database().await;

        // Create multiple seeds
        for i in 1..=5 {
            let mem = test_memory(
                &format!("mem_{i}"),
                &format!("Fact number {i}"),
                Some("user_1"),
            );
            MemoryRepository::create(&conn, &mem).await.unwrap();
            let embedding = vec![0.1_f32; 384];
            MemoryRepository::update_embedding(&conn, &format!("mem_{i}"), &embedding)
                .await
                .unwrap();
        }

        let embeddings = test_embeddings_provider().await;
        let engine =
            InferenceEngine::new(db, test_llm_provider(llm_server.uri()), embeddings, config);

        let stats = engine.run_once().await.expect("run_once should succeed");

        // max_per_run = 1, so at most 1 inference created
        assert!(stats.inferences_created <= 1);
    }

    #[tokio::test]
    async fn test_interval_secs() {
        let config = InferenceConfig {
            interval_secs: 42,
            ..test_config()
        };
        let (_conn, db, _temp_dir) = test_database().await;
        let embeddings = test_embeddings_provider().await;
        let engine = InferenceEngine::new(db, test_llm_unavailable(), embeddings, config);

        assert_eq!(engine.interval_secs(), 42);
    }
}
