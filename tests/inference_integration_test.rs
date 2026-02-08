use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use libsql::Builder;
use serde_json::json;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use momo::config::{DatabaseConfig, EmbeddingsConfig, InferenceConfig, LlmConfig};
use momo::db::{Database, DatabaseBackend, LibSqlBackend, MemoryRepository};
use momo::embeddings::EmbeddingProvider;
use momo::intelligence::InferenceEngine;
use momo::llm::LlmProvider;
use momo::models::{Memory, MemoryRelationType, MemoryType};

// ── Test Helpers ──────────────────────────────────────────────────────────

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

async fn test_database() -> (Database, TempDir) {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let db_path = temp_dir.path().join("inference_integ_test.db");
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

async fn test_embeddings_provider(mock_server: &MockServer) -> EmbeddingProvider {
    let config = EmbeddingsConfig {
        model: "openai/text-embedding-3-small".to_string(),
        dimensions: 384,
        batch_size: 8,
        api_key: Some("test-key".to_string()),
        base_url: Some(mock_server.uri()),
        rate_limit: None,
        timeout_secs: 5,
        max_retries: 0,
    };

    EmbeddingProvider::new_async(&config)
        .await
        .expect("failed to create test embeddings provider")
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
        enable_llm_filter: false,
        filter_prompt: None,
    };

    LlmProvider::new(Some(&config))
}

fn test_memory(id: &str, content: &str, container_tag: Option<&str>) -> Memory {
    Memory {
        id: id.to_string(),
        memory: content.to_string(),
        space_id: "default".to_string(),
        container_tag: container_tag.map(str::to_string),
        confidence: None,
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
        metadata: Default::default(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

fn test_episode_memory(id: &str, content: &str) -> Memory {
    Memory {
        id: id.to_string(),
        memory: content.to_string(),
        space_id: "default".to_string(),
        container_tag: None,
        confidence: None,
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
        metadata: Default::default(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

/// Mount mock embeddings endpoint on a mock server
async fn mount_embeddings_mock(mock_server: &MockServer) {
    let embedding = vec![0.1_f32; 384];
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "embedding": embedding }]
        })))
        .mount(mock_server)
        .await;
}

/// Seed the database with Fact memories and their embeddings
async fn seed_fact_memories(
    db: &Database,
    count: usize,
    container_tag: Option<&str>,
) -> Vec<String> {
    let conn = db.connect().expect("connect should work");
    let embedding = vec![0.1_f32; 384];
    let mut ids = Vec::new();

    for i in 1..=count {
        let id = format!("fact_{}", i);
        let content = format!("Important fact number {} about the user", i);
        let mem = test_memory(&id, &content, container_tag);
        MemoryRepository::create(&conn, &mem).await.unwrap();
        MemoryRepository::update_embedding(&conn, &id, &embedding)
            .await
            .unwrap();
        ids.push(id);
    }

    ids
}

/// Query the database for all inference memories
async fn get_inference_memories(db: &Database) -> Vec<Memory> {
    let conn = db.connect().expect("connect should work");
    let mut rows = conn
        .query(
            r#"
            SELECT id, memory, space_id, container_tag, version, is_latest,
                   parent_memory_id, root_memory_id, memory_relations, source_count,
                   is_inference, is_forgotten, is_static, forget_after, forget_reason,
                   memory_type, last_accessed, metadata, created_at, updated_at
            FROM memories
            WHERE is_inference = 1
            "#,
            (),
        )
        .await
        .unwrap();

    let mut memories = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        memories.push(MemoryRepository::row_to_memory(&row).unwrap());
    }
    memories
}

// ── Integration Tests ─────────────────────────────────────────────────────

/// Full pipeline: seed Facts → generate inferences → verify created with Derives relations
#[tokio::test]
async fn inference_e2e_generates_memories() {
    let mock_server = MockServer::start().await;

    // Mount embeddings mock
    mount_embeddings_mock(&mock_server).await;

    // Mount LLM mock with a valid high-confidence inference response
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
            r#"{"content": "User is technically skilled and detail-oriented", "reasoning": "Combined multiple facts", "confidence": 0.92, "source_ids": ["fact_2"]}"#,
        )))
        .mount(&mock_server)
        .await;

    let (db, _temp_dir) = test_database().await;
    let embeddings = test_embeddings_provider(&mock_server).await;
    let llm = test_llm_provider(mock_server.uri());

    // Seed 3 fact memories
    let seed_ids = seed_fact_memories(&db, 3, Some("user_1")).await;
    assert_eq!(seed_ids.len(), 3);

    let engine = InferenceEngine::new(Arc::new(LibSqlBackend::new(db.clone())), llm, embeddings, test_config());
    let stats = engine.run_once().await.expect("run_once should succeed");

    // Should have processed seeds and created inferences
    assert!(
        stats.seeds_processed > 0,
        "Should have processed at least one seed"
    );
    assert!(
        stats.inferences_created > 0,
        "Should have created at least one inference"
    );

    // Verify inference was stored in DB with correct properties
    let inferences = get_inference_memories(&db).await;
    assert!(
        !inferences.is_empty(),
        "Should have stored inference memories in DB"
    );

    // Check that the inference has Derives relations
    let inf = &inferences[0];
    assert!(inf.is_inference, "Memory should be marked as inference");
    assert!(
        !inf.memory_relations.is_empty(),
        "Inference should have Derives relations"
    );

    // All relations should be Derives type
    for (_related_id, relation_type) in &inf.memory_relations {
        assert_eq!(
            *relation_type,
            MemoryRelationType::Derives,
            "All inference relations should be Derives type"
        );
    }
}

/// Run inference twice with the same seeds; verify no duplicates are created on the second run
#[tokio::test]
async fn inference_deduplication_across_runs() {
    let mock_server = MockServer::start().await;

    mount_embeddings_mock(&mock_server).await;

    // LLM always returns the same inference referencing fact_2
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
            r#"{"content": "User has strong programming background", "reasoning": "Facts indicate expertise", "confidence": 0.88, "source_ids": ["fact_2"]}"#,
        )))
        .mount(&mock_server)
        .await;

    let (db, _temp_dir) = test_database().await;
    let embeddings = test_embeddings_provider(&mock_server).await;
    let llm = test_llm_provider(mock_server.uri());

    // Seed memories
    seed_fact_memories(&db, 3, Some("user_1")).await;

    let engine = InferenceEngine::new(Arc::new(LibSqlBackend::new(db.clone())), llm, embeddings, test_config());

    // First run — should create inferences
    let stats1 = engine.run_once().await.expect("first run should succeed");
    let inferences_after_first = get_inference_memories(&db).await;
    let count_after_first = inferences_after_first.len();

    assert!(
        stats1.inferences_created > 0,
        "First run should create inferences"
    );

    // Second run — same seeds, same LLM response → deduplication should prevent new ones
    let stats2 = engine.run_once().await.expect("second run should succeed");
    let inferences_after_second = get_inference_memories(&db).await;

    // No new inferences should be created (all duplicates)
    assert_eq!(
        inferences_after_second.len(),
        count_after_first,
        "Second run should not create duplicate inferences (had {}, now {})",
        count_after_first,
        inferences_after_second.len()
    );
    assert!(
        stats2.duplicates_skipped > 0 || stats2.seeds_processed == 0,
        "Second run should skip duplicates or have no new seeds"
    );
}

/// Empty database should return gracefully with 0 inferences generated
#[tokio::test]
async fn inference_handles_empty_database() {
    let mock_server = MockServer::start().await;

    mount_embeddings_mock(&mock_server).await;

    // Mount LLM mock (should never be called — no seeds)
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
            r#"{"content": "Should not be called", "reasoning": "n/a", "confidence": 1.0, "source_ids": []}"#,
        )))
        .expect(0) // Assert LLM is never called
        .mount(&mock_server)
        .await;

    let (db, _temp_dir) = test_database().await;
    let embeddings = test_embeddings_provider(&mock_server).await;
    let llm = test_llm_provider(mock_server.uri());

    // Don't seed any memories — DB is empty
    let engine = InferenceEngine::new(Arc::new(LibSqlBackend::new(db.clone())), llm, embeddings, test_config());
    let stats = engine
        .run_once()
        .await
        .expect("run_once should not fail on empty DB");

    assert_eq!(stats.seeds_processed, 0, "No seeds to process");
    assert_eq!(stats.inferences_created, 0, "No inferences created");
    assert_eq!(stats.errors, 0, "No errors");
    assert_eq!(stats.duplicates_skipped, 0, "No duplicates");
    assert_eq!(stats.low_confidence_skipped, 0, "No low confidence skips");

    // Verify DB still has no inference memories
    let inferences = get_inference_memories(&db).await;
    assert!(inferences.is_empty(), "No inferences should exist");
}

/// When LLM is unavailable, inference engine should degrade gracefully
#[tokio::test]
async fn inference_degrades_when_llm_unavailable() {
    let mock_server = MockServer::start().await;

    mount_embeddings_mock(&mock_server).await;

    let (db, _temp_dir) = test_database().await;
    let embeddings = test_embeddings_provider(&mock_server).await;

    // Use an unavailable LLM provider (no mock needed)
    let llm = LlmProvider::unavailable("test: LLM not configured");

    // Seed some memories (they exist but LLM can't process them)
    seed_fact_memories(&db, 3, Some("user_1")).await;

    let engine = InferenceEngine::new(Arc::new(LibSqlBackend::new(db.clone())), llm, embeddings, test_config());
    let stats = engine
        .run_once()
        .await
        .expect("run_once should handle unavailable LLM gracefully");

    // With LLM unavailable, engine should bail early
    assert_eq!(
        stats.seeds_processed, 0,
        "Should not process seeds without LLM"
    );
    assert_eq!(stats.inferences_created, 0, "No inferences without LLM");
    assert_eq!(stats.errors, 0, "Graceful handling, not errors");

    // Verify no inference memories were created
    let inferences = get_inference_memories(&db).await;
    assert!(
        inferences.is_empty(),
        "No inferences should be created without LLM"
    );
}

/// Low confidence (0.5) inferences should be skipped, high confidence (0.9) should be created
#[tokio::test]
async fn inference_filters_by_confidence() {
    let mock_server = MockServer::start().await;

    mount_embeddings_mock(&mock_server).await;

    // LLM returns a low-confidence inference (0.5 < threshold 0.7)
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
            r#"{"content": "Weak guess about user", "reasoning": "Tenuous connection", "confidence": 0.5, "source_ids": ["fact_2"]}"#,
        )))
        .mount(&mock_server)
        .await;

    let (db, _temp_dir) = test_database().await;
    let embeddings = test_embeddings_provider(&mock_server).await;
    let llm = test_llm_provider(mock_server.uri());

    seed_fact_memories(&db, 3, Some("user_1")).await;

    let config = test_config(); // confidence_threshold = 0.7
    let engine = InferenceEngine::new(Arc::new(LibSqlBackend::new(db.clone())), llm, embeddings, config);
    let stats = engine.run_once().await.expect("run_once should succeed");

    // Low confidence (0.5) should be skipped
    assert_eq!(
        stats.inferences_created, 0,
        "Low confidence inferences should not be created"
    );
    assert!(
        stats.low_confidence_skipped > 0,
        "Should have skipped low confidence inferences"
    );

    let inferences = get_inference_memories(&db).await;
    assert!(
        inferences.is_empty(),
        "No inferences should be stored for low confidence"
    );

    // Now test with high confidence — need a fresh mock server
    let mock_server2 = MockServer::start().await;
    mount_embeddings_mock(&mock_server2).await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
            r#"{"content": "Strong conclusion about user skills", "reasoning": "Clear evidence", "confidence": 0.9, "source_ids": ["fact_2"]}"#,
        )))
        .mount(&mock_server2)
        .await;

    let (db2, _temp_dir2) = test_database().await;
    let embeddings2 = test_embeddings_provider(&mock_server2).await;
    let llm2 = test_llm_provider(mock_server2.uri());

    seed_fact_memories(&db2, 3, Some("user_1")).await;

    let config2 = test_config(); // confidence_threshold = 0.7
    let engine2 = InferenceEngine::new(Arc::new(LibSqlBackend::new(db2.clone())), llm2, embeddings2, config2);
    let stats2 = engine2.run_once().await.expect("run_once should succeed");

    // High confidence (0.9) should be created
    assert!(
        stats2.inferences_created > 0,
        "High confidence inferences should be created"
    );
    assert_eq!(
        stats2.low_confidence_skipped, 0,
        "No low confidence skips for high confidence"
    );

    let inferences2 = get_inference_memories(&db2).await;
    assert!(
        !inferences2.is_empty(),
        "High confidence inferences should be stored"
    );
}

/// Episodes should never be used as inference sources when exclude_episodes is true
#[tokio::test]
async fn inference_excludes_episodes_from_sources() {
    let mock_server = MockServer::start().await;

    mount_embeddings_mock(&mock_server).await;

    // LLM returns a valid high-confidence inference
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
            r#"{"content": "User cares about code quality", "reasoning": "Based on facts", "confidence": 0.95, "source_ids": []}"#,
        )))
        .mount(&mock_server)
        .await;

    let (db, _temp_dir) = test_database().await;
    let conn = db.connect().expect("connect should work");
    let embedding = vec![0.1_f32; 384];

    // Create ONLY episode memories (no facts)
    for i in 1..=3 {
        let id = format!("ep_{}", i);
        let mem = test_episode_memory(&id, &format!("User had conversation {} about coding", i));
        MemoryRepository::create(&conn, &mem).await.unwrap();
        MemoryRepository::update_embedding(&conn, &id, &embedding)
            .await
            .unwrap();
    }

    let embeddings = test_embeddings_provider(&mock_server).await;
    let llm = test_llm_provider(mock_server.uri());

    // Config with exclude_episodes = true (default)
    let config = test_config();
    assert!(
        config.exclude_episodes,
        "Precondition: exclude_episodes should be true"
    );

    let engine = InferenceEngine::new(Arc::new(LibSqlBackend::new(db.clone())), llm, embeddings, config);
    let stats = engine.run_once().await.expect("run_once should succeed");

    // Episodes should be excluded from seed selection
    assert_eq!(
        stats.seeds_processed, 0,
        "Episodes should not be selected as seeds"
    );
    assert_eq!(
        stats.inferences_created, 0,
        "No inferences from episode-only DB"
    );

    // Verify no inference memories were created
    let inferences = get_inference_memories(&db).await;
    assert!(
        inferences.is_empty(),
        "No inferences should exist when only episodes are present"
    );

    // Now add a single fact memory alongside episodes — only the fact should be used
    let fact = test_memory(
        "fact_solo",
        "User prefers Rust for systems programming",
        Some("user_1"),
    );
    MemoryRepository::create(&conn, &fact).await.unwrap();
    MemoryRepository::update_embedding(&conn, "fact_solo", &embedding)
        .await
        .unwrap();

    // Create a new engine instance (same DB with the added fact)
    let mock_server2 = MockServer::start().await;
    mount_embeddings_mock(&mock_server2).await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
            r#"{"content": "User is a Rust systems programmer", "reasoning": "Derived from fact", "confidence": 0.9, "source_ids": []}"#,
        )))
        .mount(&mock_server2)
        .await;

    let embeddings2 = test_embeddings_provider(&mock_server2).await;
    let llm2 = test_llm_provider(mock_server2.uri());
    let config2 = test_config();

    let engine2 = InferenceEngine::new(Arc::new(LibSqlBackend::new(db.clone())), llm2, embeddings2, config2);
    let stats2 = engine2
        .run_once()
        .await
        .expect("run_once should succeed with fact present");

    // The single fact should be used as a seed
    assert!(
        stats2.seeds_processed > 0,
        "Fact memory should be selected as seed"
    );

    // Any inferences created should NOT have episode IDs in their relations
    let inferences2 = get_inference_memories(&db).await;
    for inf in &inferences2 {
        for (related_id, _) in &inf.memory_relations {
            assert!(
                !related_id.starts_with("ep_"),
                "Inference should not derive from episode memory (found relation to {})",
                related_id
            );
        }
    }
}
