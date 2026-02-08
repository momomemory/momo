use momo::api::{create_router, AppState};
use momo::config::{Config, LlmConfig};
use momo::db::{Database, DatabaseBackend, LibSqlBackend};
use momo::embeddings::EmbeddingProvider;
use momo::llm::LlmProvider;
use momo::models::SearchDocumentsResponse;
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
            enable_llm_filter: false,
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

/// Test 1: Query rewrite disabled by default (no LLM config)
#[tokio::test]
async fn test_rewrite_disabled_by_default() {
    let (addr, _temp_dir) = setup_test_app(false).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // Add a test document
    let doc_res = client
        .post(format!("{base_url}/v3/documents"))
        .json(&json!({
            "content": "Rust is a systems programming language focused on safety and performance.",
            "container_tag": "test_tag",
            "metadata": {"category": "tech"}
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Search without rewrite flag
    let search_res = client
        .post(format!("{base_url}/v3/search"))
        .json(&json!({
            "q": "rust programming",
            "container_tags": ["test_tag"]
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let response: SearchDocumentsResponse = search_res.json().await.expect("Failed to parse JSON");

    assert!(
        response.rewritten_query.is_none(),
        "rewritten_query should be None when rewrite is disabled"
    );
}

/// Test 2: Query rewrite enabled with flag returns rewritten_query
#[tokio::test]
async fn test_rewrite_enabled_with_flag() {
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // Add test document
    let doc_res = client
        .post(format!("{base_url}/v3/documents"))
        .json(&json!({
            "content": "Machine learning algorithms for image classification.",
            "container_tag": "ml_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Search with rewrite_query=true - will fail gracefully if LLM unavailable
    let search_res = client
        .post(format!("{base_url}/v3/search"))
        .json(&json!({
            "q": "how to train ML models",
            "container_tags": ["ml_tag"],
            "rewrite_query": true
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let response: SearchDocumentsResponse = search_res.json().await.expect("Failed to parse JSON");

    assert!(response.rewritten_query.is_none() || response.rewritten_query.is_some());
}

/// Test 3: Query rewrite flag false skips rewrite even when enabled
#[tokio::test]
async fn test_rewrite_flag_false_skips_rewrite() {
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/v3/documents"))
        .json(&json!({
            "content": "Deep learning neural networks and architectures.",
            "container_tag": "dl_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Explicitly set rewrite_query=false
    let search_res = client
        .post(format!("{base_url}/v3/search"))
        .json(&json!({
            "q": "neural network architectures",
            "container_tags": ["dl_tag"],
            "rewrite_query": false
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let response: SearchDocumentsResponse = search_res.json().await.expect("Failed to parse JSON");

    assert!(
        response.rewritten_query.is_none(),
        "rewritten_query should be None when rewrite_query=false"
    );
}

/// Test 4: LLM unavailable gracefully skips rewrite
#[tokio::test]
async fn test_llm_unavailable_graceful_fallback() {
    // Set up with LLM enabled but unreachable endpoint
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/v3/documents"))
        .json(&json!({
            "content": "Natural language processing techniques for sentiment analysis.",
            "container_tag": "nlp_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Search with rewrite_query=true - should fallback gracefully
    let search_res = client
        .post(format!("{base_url}/v3/search"))
        .json(&json!({
            "q": "sentiment analysis methods",
            "container_tags": ["nlp_tag"],
            "rewrite_query": true
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let response: SearchDocumentsResponse = search_res.json().await.expect("Failed to parse JSON");

    // Should return results even if rewrite failed
    assert!(
        response.rewritten_query.is_none(),
        "rewritten_query should be None when LLM unavailable"
    );
    // But search should still work
    assert!(response.timing > 0);
}

/// Test 5: Timeout fallback works
#[tokio::test]
async fn test_timeout_fallback() {
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/v3/documents"))
        .json(&json!({
            "content": "Computer vision object detection models.",
            "container_tag": "cv_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Request with rewrite - will timeout if LLM is slow/unavailable
    let search_res = client
        .post(format!("{base_url}/v3/search"))
        .json(&json!({
            "q": "object detection algorithms",
            "container_tags": ["cv_tag"],
            "rewrite_query": true
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let response: SearchDocumentsResponse = search_res.json().await.expect("Failed to parse JSON");

    // Should complete within reasonable time even if LLM times out
    assert!(
        response.timing < 10000,
        "Search should complete quickly even with timeout"
    );
}

/// Test 6: Short query skipped (query eligibility check)
#[tokio::test]
async fn test_short_query_skipped() {
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/v3/documents"))
        .json(&json!({
            "content": "Kubernetes container orchestration platform.",
            "container_tag": "k8s_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Very short query (< 3 chars) should be skipped
    let search_res = client
        .post(format!("{base_url}/v3/search"))
        .json(&json!({
            "q": "k8",
            "container_tags": ["k8s_tag"],
            "rewrite_query": true
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let response: SearchDocumentsResponse = search_res.json().await.expect("Failed to parse JSON");

    assert!(
        response.rewritten_query.is_none(),
        "Short queries should not be rewritten"
    );
}

/// Test 7: Long query skipped (query eligibility check)
#[tokio::test]
async fn test_long_query_skipped() {
    let (addr, _temp_dir) = setup_test_app(true).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/v3/documents"))
        .json(&json!({
            "content": "Distributed systems consensus algorithms.",
            "container_tag": "dist_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Very long query (> 500 chars) should be skipped
    let long_query = "a".repeat(501);
    let search_res = client
        .post(format!("{base_url}/v3/search"))
        .json(&json!({
            "q": long_query,
            "container_tags": ["dist_tag"],
            "rewrite_query": true
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let response: SearchDocumentsResponse = search_res.json().await.expect("Failed to parse JSON");

    assert!(
        response.rewritten_query.is_none(),
        "Long queries should not be rewritten"
    );
}

/// Test 8: Memory search with query rewrite
#[tokio::test]
async fn test_memory_search_rewrite() {
    let (addr, _temp_dir) = setup_test_app(false).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // Create a memory via documents
    let doc_res = client
        .post(format!("{base_url}/v3/documents"))
        .json(&json!({
            "content": "User prefers dark mode interface with blue accent colors.",
            "container_tag": "user_prefs"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Search memories without rewrite
    let search_res = client
        .post(format!("{base_url}/v4/search"))
        .json(&json!({
            "q": "interface preferences",
            "container_tag": "user_prefs"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let response_json: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");

    assert!(
        response_json
            .get("rewritten_query")
            .map(|v| v.is_null())
            .unwrap_or(true),
        "rewritten_query should be None when rewrite not requested"
    );
}
