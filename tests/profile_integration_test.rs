use chrono::Utc;
use libsql::Builder;
use serde_json::json;
use std::sync::Arc;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use momo::config::{DatabaseConfig, LlmConfig};
use momo::db::repository::MemoryRepository;
use momo::db::{Database, LibSqlBackend};
use momo::llm::LlmProvider;
use momo::models::{Memory, MemoryType};
use momo::services::ProfileRefreshManager;

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

async fn test_database() -> (Database, TempDir) {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let db_path = temp_dir.path().join("profile_integ_test.db");

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
            confidence REAL,
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

fn test_memory(id: &str, content: &str, container_tag: &str, is_static: bool) -> Memory {
    Memory {
        id: id.to_string(),
        memory: content.to_string(),
        space_id: "default".to_string(),
        container_tag: Some(container_tag.to_string()),
        confidence: None,
        version: 1,
        is_latest: true,
        parent_memory_id: None,
        root_memory_id: None,
        memory_relations: Default::default(),
        source_count: 1,
        is_inference: false,
        is_forgotten: false,
        is_static,
        forget_after: None,
        forget_reason: None,
        memory_type: MemoryType::Fact,
        last_accessed: None,
        metadata: Default::default(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

async fn seed_memories(db: &Database, container_tag: &str, facts: &[(&str, &str, bool)]) {
    let conn = db.connect().expect("connect should work");
    for (id, content, is_static) in facts {
        let mem = test_memory(id, content, container_tag, *is_static);
        MemoryRepository::create(&conn, &mem).await.unwrap();
    }
}

/// Mounts two sequential wiremock responses for `POST /chat/completions`:
/// first returns `narrative` JSON, second returns `compacted` JSON.
async fn mount_profile_llm_mock(mock_server: &MockServer, narrative: &str, compacted: &str) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(narrative)))
        .up_to_n_times(1)
        .expect(1)
        .mount(mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(compacted)))
        .up_to_n_times(1)
        .expect(1)
        .mount(mock_server)
        .await;
}

// ── Integration Tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn profile_refresh_e2e_creates_cached_profile() {
    let mock_server = MockServer::start().await;

    mount_profile_llm_mock(
        &mock_server,
        r#"{"narrative": "The user is a Rust developer who enjoys hiking and prefers dark mode."}"#,
        r#"{"Technical": ["Uses Rust programming language"], "Hobbies": ["Enjoys hiking"], "Preferences": ["Prefers dark mode"]}"#,
    )
    .await;

    let (db, _temp_dir) = test_database().await;

    seed_memories(
        &db,
        "user_1",
        &[
            ("m1", "User programs in Rust", true),
            ("m2", "User enjoys hiking on weekends", false),
            ("m3", "User prefers dark mode in all editors", true),
        ],
    )
    .await;

    let conn = db.connect().unwrap();
    let cached = MemoryRepository::get_cached_profile(&conn, "user_1")
        .await
        .unwrap();
    assert!(cached.is_none(), "No cache should exist before refresh");

    let llm = test_llm_provider(mock_server.uri());
    let manager = ProfileRefreshManager::new(Arc::new(LibSqlBackend::new(db.clone())), llm, 3600);
    let refreshed = manager.run_once().await.expect("run_once should succeed");

    assert_eq!(refreshed, 1, "Should have refreshed exactly 1 profile");

    let cached = MemoryRepository::get_cached_profile(&conn, "user_1")
        .await
        .unwrap();
    assert!(cached.is_some(), "Cache should exist after refresh");

    let profile = cached.unwrap();
    assert_eq!(profile.container_tag, "user_1");
    assert!(profile.narrative.is_some());
    assert!(profile
        .narrative
        .as_ref()
        .unwrap()
        .contains("Rust developer"),);
    assert!(profile.summary.is_some());
    assert!(profile.cached_at.is_some());
}

#[tokio::test]
async fn profile_refresh_skips_fresh_cache() {
    let mock_server = MockServer::start().await;

    mount_profile_llm_mock(
        &mock_server,
        r#"{"narrative": "The user likes Python."}"#,
        r#"{"Technical": ["Likes Python"]}"#,
    )
    .await;

    let (db, _temp_dir) = test_database().await;

    seed_memories(&db, "user_2", &[("m1", "User likes Python", true)]).await;

    let llm = test_llm_provider(mock_server.uri());
    let manager = ProfileRefreshManager::new(Arc::new(LibSqlBackend::new(db.clone())), llm, 3600);

    let refreshed1 = manager.run_once().await.expect("first run should succeed");
    assert_eq!(refreshed1, 1);

    let mock_server2 = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(llm_response(r#"{"narrative": "Should not be called"}"#)),
        )
        .expect(0)
        .mount(&mock_server2)
        .await;

    let llm2 = test_llm_provider(mock_server2.uri());
    let manager2 = ProfileRefreshManager::new(Arc::new(LibSqlBackend::new(db.clone())), llm2, 3600);

    let refreshed2 = manager2
        .run_once()
        .await
        .expect("second run should succeed");
    assert_eq!(refreshed2, 0, "Fresh cache should not be re-generated");
}

#[tokio::test]
async fn profile_refresh_updates_stale_cache() {
    let mock_server1 = MockServer::start().await;
    mount_profile_llm_mock(
        &mock_server1,
        r#"{"narrative": "The user is a Go developer."}"#,
        r#"{"Technical": ["Uses Go"]}"#,
    )
    .await;

    let (db, _temp_dir) = test_database().await;

    seed_memories(&db, "user_3", &[("m1", "User programs in Go", true)]).await;

    let llm1 = test_llm_provider(mock_server1.uri());
    let manager1 = ProfileRefreshManager::new(Arc::new(LibSqlBackend::new(db.clone())), llm1, 3600);
    let refreshed1 = manager1.run_once().await.expect("first run should succeed");
    assert_eq!(refreshed1, 1);

    let conn = db.connect().unwrap();
    let initial = MemoryRepository::get_cached_profile(&conn, "user_3")
        .await
        .unwrap()
        .expect("cache should exist");
    assert!(initial.narrative.as_ref().unwrap().contains("Go developer"));

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    seed_memories(&db, "user_3", &[("m2", "User also programs in Rust", true)]).await;

    let mock_server2 = MockServer::start().await;
    mount_profile_llm_mock(
        &mock_server2,
        r#"{"narrative": "The user is a Go and Rust developer."}"#,
        r#"{"Technical": ["Uses Go", "Uses Rust"]}"#,
    )
    .await;

    let llm2 = test_llm_provider(mock_server2.uri());
    let manager2 = ProfileRefreshManager::new(Arc::new(LibSqlBackend::new(db.clone())), llm2, 3600);
    let refreshed2 = manager2
        .run_once()
        .await
        .expect("second run should succeed");
    assert_eq!(refreshed2, 1, "Stale profile should be re-generated");

    let updated = MemoryRepository::get_cached_profile(&conn, "user_3")
        .await
        .unwrap()
        .expect("cache should still exist");
    assert!(updated.narrative.as_ref().unwrap().contains("Go and Rust"),);
    assert_ne!(initial.cached_at, updated.cached_at);
}

#[tokio::test]
async fn profile_refresh_graceful_when_llm_unavailable() {
    let (db, _temp_dir) = test_database().await;

    seed_memories(&db, "user_4", &[("m1", "User likes TypeScript", true)]).await;

    let llm = LlmProvider::unavailable("test: LLM not configured");
    let manager = ProfileRefreshManager::new(Arc::new(LibSqlBackend::new(db.clone())), llm, 3600);

    let refreshed = manager
        .run_once()
        .await
        .expect("should handle unavailable LLM gracefully");
    assert_eq!(refreshed, 0);

    let conn = db.connect().unwrap();
    let cached = MemoryRepository::get_cached_profile(&conn, "user_4")
        .await
        .unwrap();
    assert!(cached.is_none());
}

#[tokio::test]
async fn profile_refresh_handles_empty_db() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(llm_response(r#"{"narrative": "Should not be called"}"#)),
        )
        .expect(0)
        .mount(&mock_server)
        .await;

    let (db, _temp_dir) = test_database().await;
    let llm = test_llm_provider(mock_server.uri());
    let manager = ProfileRefreshManager::new(Arc::new(LibSqlBackend::new(db)), llm, 3600);

    let refreshed = manager
        .run_once()
        .await
        .expect("should handle empty DB gracefully");
    assert_eq!(refreshed, 0);
}

#[tokio::test]
async fn profile_refresh_multiple_container_tags() {
    let mock_server = MockServer::start().await;

    for _ in 0..2 {
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"narrative": "A user profile narrative."}"#,
            )))
            .up_to_n_times(1)
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(llm_response(r#"{"General": ["Some fact"]}"#)),
            )
            .up_to_n_times(1)
            .mount(&mock_server)
            .await;
    }

    let (db, _temp_dir) = test_database().await;

    seed_memories(&db, "alice", &[("a1", "Alice likes Rust", true)]).await;
    seed_memories(&db, "bob", &[("b1", "Bob likes Python", true)]).await;

    let llm = test_llm_provider(mock_server.uri());
    let manager = ProfileRefreshManager::new(Arc::new(LibSqlBackend::new(db.clone())), llm, 3600);

    let refreshed = manager.run_once().await.expect("run_once should succeed");
    assert_eq!(refreshed, 2);

    let conn = db.connect().unwrap();
    let alice_cache = MemoryRepository::get_cached_profile(&conn, "alice")
        .await
        .unwrap();
    let bob_cache = MemoryRepository::get_cached_profile(&conn, "bob")
        .await
        .unwrap();

    assert!(alice_cache.is_some());
    assert!(bob_cache.is_some());
}

#[tokio::test]
async fn profile_refresh_continues_on_individual_tag_failure() {
    let mock_server = MockServer::start().await;

    // 500 errors for the first tag's narrative + compact calls
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .up_to_n_times(2)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
            r#"{"narrative": "Bob is a Python developer."}"#,
        )))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(llm_response(r#"{"Technical": ["Uses Python"]}"#)),
        )
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    let (db, _temp_dir) = test_database().await;

    seed_memories(&db, "aaa_fail", &[("f1", "User likes Java", true)]).await;
    seed_memories(&db, "zzz_success", &[("s1", "User likes Python", true)]).await;

    let llm = test_llm_provider(mock_server.uri());
    let manager = ProfileRefreshManager::new(Arc::new(LibSqlBackend::new(db.clone())), llm, 3600);

    let refreshed = manager
        .run_once()
        .await
        .expect("should not crash on partial failure");

    let conn = db.connect().unwrap();
    let success_cache = MemoryRepository::get_cached_profile(&conn, "zzz_success")
        .await
        .unwrap();

    assert!(
        refreshed >= 1 || success_cache.is_some(),
        "At least one profile should be refreshed despite errors on another tag"
    );
}

#[tokio::test]
async fn profile_refresh_interval_secs() {
    let (db, _temp_dir) = test_database().await;
    let llm = LlmProvider::unavailable("test");
    let manager = ProfileRefreshManager::new(Arc::new(LibSqlBackend::new(db)), llm, 86400);
    assert_eq!(manager.interval_secs(), 86400);
}
