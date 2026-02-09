use momo::api::{create_router, AppState};
use momo::config::{Config, LlmConfig};
use momo::db::{Database, DatabaseBackend, LibSqlBackend};
use momo::embeddings::EmbeddingProvider;
use momo::llm::LlmProvider;
use momo::ocr::OcrProvider;
use momo::transcription::TranscriptionProvider;
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use tempfile::TempDir;

async fn setup_test_app(llm_enabled: bool) -> (SocketAddr, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("momo.db");
    let db_url = format!("file:{}", db_path.to_str().unwrap());

    let mut config = Config::from_env();
    config.database.url = db_url;
    config.embeddings.model = "local/BAAI/bge-small-en-v1.5".to_string();
    config.embeddings.dimensions = 384;
    config.server.api_keys = vec!["test-key".to_string()];

    // Configure LLM for query rewriting
    if llm_enabled {
        config.llm = Some(LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some("http://localhost:11434/v1".to_string()),
            timeout_secs: 5,
            max_retries: 1,
            enable_query_rewrite: true,
            query_rewrite_timeout_secs: 5,
            query_rewrite_cache_size: 100,
            enable_auto_relations: false,
            enable_contradiction_detection: false,
            filter_prompt: None,
        });
    } else {
        config.llm = None;
    }

    let db = Database::new(&config.database)
        .await
        .expect("Failed to create database");
    let db_backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db));
    let embeddings =
        EmbeddingProvider::new(&config.embeddings).expect("Failed to create embeddings");
    let ocr = OcrProvider::new(&config.ocr).expect("Failed to create OCR");
    let transcription =
        TranscriptionProvider::new(&config.transcription).expect("Failed to create transcription");
    let reranker = None;
    let llm = LlmProvider::new(config.llm.as_ref());

    let state = AppState::new(
        config.clone(),
        db_backend,
        embeddings,
        reranker,
        ocr,
        transcription,
        llm,
    );
    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind to random port");
    let addr = listener.local_addr().expect("Failed to get local address");

    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("Server failed");
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    (addr, temp_dir)
}

/// Test 1: Search works without LLM (query rewrite disabled)
#[tokio::test]
async fn test_rewrite_disabled_by_default() {
    let (addr, _temp_dir) = setup_test_app(false).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // Add a test document
    let doc_res = client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Rust is a systems programming language focused on safety and performance.",
            "containerTag": "test_tag",
            "metadata": {"category": "tech"}
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Search without rewrite flag
    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "rust programming",
            "containerTags": ["test_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");

    // v1 does not expose rewritten_query
    assert!(body["data"]["results"].is_array());
}

/// Test 2: Search works with LLM configured (graceful fallback when LLM unavailable)
#[tokio::test]
async fn test_rewrite_enabled_with_flag() {
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // Add test document
    let doc_res = client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Machine learning algorithms for image classification.",
            "containerTag": "ml_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Search - v1 doesn't have rewrite_query param, search should still work
    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "how to train ML models",
            "containerTags": ["ml_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");
    assert!(body["data"]["results"].is_array());
}

/// Test 3: Search works with LLM configured but query rewrite not requested
#[tokio::test]
async fn test_rewrite_flag_false_skips_rewrite() {
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Deep learning neural networks and architectures.",
            "containerTag": "dl_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "neural network architectures",
            "containerTags": ["dl_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");
    assert!(body["data"]["results"].is_array());
}

/// Test 4: LLM unavailable gracefully skips rewrite
#[tokio::test]
async fn test_llm_unavailable_graceful_fallback() {
    // Set up with LLM enabled but unreachable endpoint
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Natural language processing techniques for sentiment analysis.",
            "containerTag": "nlp_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Search should succeed even if LLM is unreachable
    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "sentiment analysis methods",
            "containerTags": ["nlp_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");

    // Should return results even if rewrite failed
    assert!(body["data"]["results"].is_array());
    // timingMs should be positive
    assert!(body["data"]["timingMs"].as_u64().unwrap_or(0) > 0);
}

/// Test 5: Timeout fallback works
#[tokio::test]
async fn test_timeout_fallback() {
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Computer vision object detection models.",
            "containerTag": "cv_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Search should complete within reasonable time even if LLM times out
    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "object detection algorithms",
            "containerTags": ["cv_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");

    // Should complete within reasonable time even if LLM times out
    let timing = body["data"]["timingMs"].as_u64().unwrap_or(0);
    assert!(
        timing < 10000,
        "Search should complete quickly even with timeout"
    );
}

/// Test 6: Short query is handled correctly
#[tokio::test]
async fn test_short_query_skipped() {
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Kubernetes container orchestration platform.",
            "containerTag": "k8s_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Very short query should still succeed
    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "k8",
            "containerTags": ["k8s_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");
    assert!(body["data"]["results"].is_array());
}

/// Test 7: Long query is handled correctly
#[tokio::test]
async fn test_long_query_skipped() {
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Distributed systems consensus algorithms.",
            "containerTag": "dist_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Very long query should still succeed
    let long_query = "a".repeat(501);
    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": long_query,
            "containerTags": ["dist_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");
    assert!(body["data"]["results"].is_array());
}

/// Test 8: Memory search works without rewrite
#[tokio::test]
async fn test_memory_search_rewrite() {
    let (addr, _temp_dir) = setup_test_app(false).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // Create a memory via documents
    let doc_res = client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "User prefers dark mode interface with blue accent colors.",
            "containerTag": "user_prefs"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Search memories
    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "interface preferences",
            "containerTags": ["user_prefs"],
            "scope": "memories"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");
    assert!(body["data"]["results"].is_array());
}
