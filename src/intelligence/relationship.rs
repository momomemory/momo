use crate::db::DatabaseBackend;
use crate::embeddings::EmbeddingProvider;
use crate::error::{MomoError, Result};
use crate::llm::{prompts, LlmProvider};

use super::types::{
    DetectionResult, HeuristicContext, RelationshipClassification,
    RelationshipClassificationsResponse,
};

pub struct RelationshipDetector {
    llm: LlmProvider,
    embeddings: EmbeddingProvider,
}

impl Clone for RelationshipDetector {
    fn clone(&self) -> Self {
        Self {
            llm: self.llm.clone(),
            embeddings: self.embeddings.clone(),
        }
    }
}

impl RelationshipDetector {
    pub fn new(llm: LlmProvider, embeddings: EmbeddingProvider) -> Self {
        Self { llm, embeddings }
    }

    pub async fn detect(
        &self,
        new_memory_id: &str,
        new_memory_content: &str,
        container_tag: Option<&str>,
        db: &dyn DatabaseBackend,
        heuristic_context: Option<&HeuristicContext>,
    ) -> Result<DetectionResult> {
        tracing::trace!(
            embedding_dimensions = self.embeddings.dimensions(),
            "Detecting relationships"
        );

        if !self.llm.is_available() {
            tracing::warn!("LLM unavailable, skipping relationship detection");
            return Ok(empty_result());
        }

        let embedding = match self.embeddings.embed_passage(new_memory_content).await {
            Ok(embedding) => embedding,
            Err(error) => {
                tracing::error!(error = %error, "Failed to embed memory for relationship detection");
                return Ok(empty_result());
            }
        };

        let candidates = db
            .search_similar_memories(&embedding, 5, 0.7, container_tag, false)
            .await?
            .into_iter()
            .filter(|hit| hit.memory.id != new_memory_id)
            .map(|hit| hit.memory)
            .collect::<Vec<_>>();

        if candidates.is_empty() {
            return Ok(empty_result());
        }

        let prompt_candidates = candidates
            .iter()
            .map(|memory| (memory.id.as_str(), memory.memory.as_str()))
            .collect::<Vec<_>>();

        let prompt = prompts::relationship_detection_prompt(
            new_memory_content,
            &prompt_candidates,
            heuristic_context,
        );
        match self
            .llm
            .complete_structured::<RelationshipClassificationsResponse>(&prompt)
            .await
        {
            Ok(response) => {
                let classifications = response.into_classifications();
                let filtered: Vec<RelationshipClassification> = classifications
                    .into_iter()
                    .filter(|c| c.confidence >= 0.7)
                    .collect();

                let heuristic_overridden = heuristic_context.map(|ctx| {
                    let flagged_classification = filtered
                        .iter()
                        .find(|c| c.memory_id == ctx.candidate_memory_id);

                    match flagged_classification {
                        Some(c) if c.relation_type == "updates" => {
                            tracing::info!(
                                candidate_memory_id = %ctx.candidate_memory_id,
                                heuristic = %ctx.heuristic_result,
                                confidence = c.confidence,
                                "Heuristic contradiction confirmed by LLM"
                            );
                            true
                        }
                        Some(c) => {
                            tracing::info!(
                                candidate_memory_id = %ctx.candidate_memory_id,
                                heuristic = %ctx.heuristic_result,
                                llm_relation_type = %c.relation_type,
                                reasoning = ?c.reasoning,
                                "Heuristic contradiction overridden by LLM"
                            );
                            false
                        }
                        None => {
                            tracing::info!(
                                candidate_memory_id = %ctx.candidate_memory_id,
                                heuristic = %ctx.heuristic_result,
                                "Heuristic contradiction overridden by LLM (candidate filtered out)"
                            );
                            false
                        }
                    }
                });

                Ok(DetectionResult {
                    classifications: filtered,
                    heuristic_overridden,
                })
            }
            Err(MomoError::LlmUnavailable(reason)) => {
                tracing::warn!(%reason, "LLM unavailable during relationship detection");
                Ok(empty_result())
            }
            Err(error) => {
                tracing::error!(error = %error, "Failed to detect relationships");
                Ok(empty_result())
            }
        }
    }
}

fn empty_result() -> DetectionResult {
    DetectionResult {
        classifications: Vec::new(),
        heuristic_overridden: None,
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
    use crate::models::Memory;

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
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        })
    }

    async fn test_database() -> (Database, TempDir) {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("relationship_test.db");
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

        (db, temp_dir)
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
            memory_type: crate::models::MemoryType::Fact,
            last_accessed: None,
            confidence: None,
            metadata: Default::default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_detect_returns_empty_when_llm_unavailable() {
        let embeddings = test_embeddings_provider().await;
        let detector = RelationshipDetector::new(test_llm_unavailable(), embeddings);
        let (db, _temp_dir) = test_database().await;
        let backend = LibSqlBackend::new(db);

        let result = detector
            .detect("mem_new", "User prefers dark mode", None, &backend, None)
            .await
            .expect("detect should not fail");

        assert!(result.classifications.is_empty());
    }

    #[tokio::test]
    async fn test_detect_returns_classifications_for_valid_llm_json() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"[{"memory_id":"mem_1","relation_type":"updates","confidence":0.9,"reasoning":"Contradiction"}]"#,
            )))
            .mount(&llm_server)
            .await;

        let embeddings = test_embeddings_provider().await;
        let detector =
            RelationshipDetector::new(test_llm_provider(llm_server.uri()), embeddings.clone());
        let (db, _temp_dir) = test_database().await;

        let conn = db.connect().expect("connect should work");
        let memory = test_memory("mem_1", "User prefers light mode", Some("user_123"));
        MemoryRepository::create(&conn, &memory)
            .await
            .expect("memory create should succeed");
        // Use a real embedding so cosine similarity with the query exceeds the 0.7 threshold
        let embedding = embeddings
            .embed_passage("User prefers light mode")
            .await
            .expect("embed should succeed");
        MemoryRepository::update_embedding(&conn, "mem_1", &embedding)
            .await
            .expect("embedding update should succeed");

        let backend = LibSqlBackend::new(db);
        let result = detector
            .detect(
                "mem_new",
                "User prefers dark mode",
                Some("user_123"),
                &backend,
                None,
            )
            .await
            .expect("detect should not fail");

        assert_eq!(result.classifications.len(), 1);
        assert_eq!(result.classifications[0].memory_id, "mem_1");
        assert_eq!(result.classifications[0].relation_type, "updates");
        assert_eq!(result.classifications[0].confidence, 0.9);
        assert_eq!(result.heuristic_overridden, None);
    }

    #[tokio::test]
    async fn test_detect_heuristic_confirmed_by_llm() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"[{"memory_id":"mem_1","relation_type":"updates","confidence":0.95,"reasoning":"Confirmed contradiction"}]"#,
            )))
            .mount(&llm_server)
            .await;

        let embeddings = test_embeddings_provider().await;
        let detector =
            RelationshipDetector::new(test_llm_provider(llm_server.uri()), embeddings.clone());
        let (db, _temp_dir) = test_database().await;

        let conn = db.connect().expect("connect should work");
        let memory = test_memory("mem_1", "User prefers light mode", Some("user_123"));
        MemoryRepository::create(&conn, &memory)
            .await
            .expect("create should succeed");
        let real_embedding = embeddings
            .embed_passage("User prefers light mode")
            .await
            .expect("embed should succeed");
        MemoryRepository::update_embedding(&conn, "mem_1", &real_embedding)
            .await
            .expect("embedding update should succeed");

        let ctx = HeuristicContext {
            candidate_memory_id: "mem_1".to_string(),
            candidate_content: "User prefers light mode".to_string(),
            heuristic_result: crate::intelligence::contradiction::ContradictionCheckResult::Likely,
        };

        let backend = LibSqlBackend::new(db);
        let result = detector
            .detect(
                "mem_new",
                "User prefers dark mode",
                Some("user_123"),
                &backend,
                Some(&ctx),
            )
            .await
            .expect("detect should not fail");

        assert_eq!(result.classifications.len(), 1);
        assert_eq!(result.classifications[0].relation_type, "updates");
        assert_eq!(result.heuristic_overridden, Some(true));
    }

    #[tokio::test]
    async fn test_detect_heuristic_overridden_by_llm() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"[{"memory_id":"mem_1","relation_type":"extends","confidence":0.85,"reasoning":"Not a contradiction, adds detail"}]"#,
            )))
            .mount(&llm_server)
            .await;

        let embeddings = test_embeddings_provider().await;
        let detector =
            RelationshipDetector::new(test_llm_provider(llm_server.uri()), embeddings.clone());
        let (db, _temp_dir) = test_database().await;

        let conn = db.connect().expect("connect should work");
        let memory = test_memory("mem_1", "User is a developer", Some("user_123"));
        MemoryRepository::create(&conn, &memory)
            .await
            .expect("create should succeed");
        let real_embedding = embeddings
            .embed_passage("User is a developer")
            .await
            .expect("embed should succeed");
        MemoryRepository::update_embedding(&conn, "mem_1", &real_embedding)
            .await
            .expect("embedding update should succeed");

        let ctx = HeuristicContext {
            candidate_memory_id: "mem_1".to_string(),
            candidate_content: "User is a developer".to_string(),
            heuristic_result:
                crate::intelligence::contradiction::ContradictionCheckResult::Unlikely,
        };

        let backend = LibSqlBackend::new(db);
        let result = detector
            .detect(
                "mem_new",
                "User is a senior developer",
                Some("user_123"),
                &backend,
                Some(&ctx),
            )
            .await
            .expect("detect should not fail");

        assert_eq!(result.classifications.len(), 1);
        assert_eq!(result.classifications[0].relation_type, "extends");
        assert_eq!(result.heuristic_overridden, Some(false));
    }

    #[tokio::test]
    async fn test_detect_no_heuristic_backward_compat() {
        let embeddings = test_embeddings_provider().await;
        let detector = RelationshipDetector::new(test_llm_unavailable(), embeddings);
        let (db, _temp_dir) = test_database().await;
        let backend = LibSqlBackend::new(db);

        let result = detector
            .detect("mem_new", "User prefers dark mode", None, &backend, None)
            .await
            .expect("detect should not fail");

        assert!(result.classifications.is_empty());
        assert_eq!(result.heuristic_overridden, None);
    }
}
