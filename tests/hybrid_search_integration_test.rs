use momo::api::create_router;
use momo::config::{Config, DatabaseConfig, EmbeddingsConfig, LlmConfig};
use momo::db::{Database, LibSqlBackend};
use momo::embeddings::EmbeddingProvider;
use momo::llm::LlmProvider;
use momo::models::HybridSearchResponse;
use momo::ocr::OcrProvider;
use momo::transcription::TranscriptionProvider;
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn setup_test_app() -> (SocketAddr, TempDir, MockServer) {
    let mock_server = MockServer::start().await;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("momo_test.db");
    let db_url = format!("file:{}", db_path.to_str().unwrap());

    // Mock Embeddings
    let embedding = vec![0.1f32; 384];
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"embedding": embedding}]
        })))
        .mount(&mock_server)
        .await;

    // Mock LLM for memory extraction
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "{\"memories\":[{\"content\":\"Rust is safe\",\"memory_type\":\"fact\",\"confidence\":0.9}]}"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        })))
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.database = DatabaseConfig {
        url: db_url,
        auth_token: None,
        local_path: None,
    };

    config.embeddings = EmbeddingsConfig {
        model: "openai/text-embedding-3-small".to_string(),
        dimensions: 384,
        batch_size: 8,
        api_key: Some("test-key".to_string()),
        base_url: Some(mock_server.uri()),
        rate_limit: None,
        timeout_secs: 5,
        max_retries: 0,
    };

    config.llm = Some(LlmConfig {
        model: "openai/gpt-4o-mini".to_string(),
        api_key: Some("test-key".to_string()),
        base_url: Some(mock_server.uri()),
        timeout_secs: 5,
        max_retries: 0,
        enable_query_rewrite: false,
        query_rewrite_cache_size: 1000,
        query_rewrite_timeout_secs: 2,
        enable_auto_relations: false,
        enable_contradiction_detection: false,
        enable_llm_filter: false,
        filter_prompt: None,
    });

    let db = Database::new(&config.database)
        .await
        .expect("Failed to create database");
    let db_backend: Arc<dyn momo::db::DatabaseBackend> = Arc::new(LibSqlBackend::new(db));
    let embeddings = EmbeddingProvider::new_async(&config.embeddings)
        .await
        .expect("Failed to create embeddings");
    let ocr = OcrProvider::new(&config.ocr).expect("Failed to create OCR");
    let transcription =
        TranscriptionProvider::new(&config.transcription).expect("Failed to create transcription");
    let llm = LlmProvider::new(config.llm.as_ref());

    let state = momo::api::AppState::new(
        config.clone(),
        db_backend,
        embeddings,
        None,
        ocr,
        transcription,
        llm,
    );
    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind");
    let addr = listener.local_addr().expect("Failed to get address");

    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("Server failed");
    });

    (addr, temp_dir, mock_server)
}

#[tokio::test]
async fn test_hybrid_search_workflow() {
    let (addr, _temp_dir, _mock_server) = setup_test_app().await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // 1. Upload document with extract_memories=true
    let doc_res = client
        .post(format!("{base_url}/v3/documents"))
        .json(&json!({
            "content": "Rust is safe and fast. It is a systems programming language.",
            "container_tag": "integration_test",
            "extract_memories": true
        }))
        .send()
        .await
        .expect("Failed to upload document");

    assert!(doc_res.status().is_success());
    let doc_data: serde_json::Value = doc_res.json().await.expect("Failed to parse doc response");
    let doc_id = doc_data["id"].as_str().expect("id missing");

    // 2. Wait for processing (including memory extraction)
    // In a real scenario we'd poll /v3/documents/{id}, but here we just wait a bit
    // since it's an async task.
    let mut processed = false;
    for _ in 0..10 {
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        let check_res = client
            .get(format!("{base_url}/v3/documents/{doc_id}"))
            .send()
            .await
            .expect("Failed to check status");
        let check_data: serde_json::Value = check_res.json().await.expect("Failed to parse status");
        if check_data["status"] == "done" {
            processed = true;
            break;
        }
    }
    assert!(processed, "Document processing timed out");

    // Give it a tiny bit more for memory extraction which happens AFTER status=done
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // 3. Perform hybrid search
    let search_res = client
        .post(format!("{base_url}/v4/search"))
        .json(&json!({
            "q": "rust",
            "container_tag": "integration_test",
            "searchMode": "hybrid"
        }))
        .send()
        .await
        .expect("Failed to search");

    assert!(search_res.status().is_success());
    let response: HybridSearchResponse = search_res
        .json()
        .await
        .expect("Failed to parse search response");

    // 4. Verify memory is returned (deduplicated)
    // Memory from mock LLM: "Rust is safe"
    // Document chunk: "Rust is safe and fast..."
    // Since memory covers the same content, it should be deduplicated (memory returned, chunk excluded)
    // In our case, the deduplication logic in SearchService should handle it if correctly implemented.

    let has_memory = response.results.iter().any(|r| r.memory.is_some());

    assert!(has_memory, "Should return at least one memory");
    // If deduplication works, we might still have chunks that AREN'T duplicates, but in this small text
    // "Rust is safe" might deduplicate the first chunk.

    // 5. Test SearchMode variants

    // Memories only
    let mem_only_res = client
        .post(format!("{base_url}/v4/search"))
        .json(&json!({
            "q": "rust",
            "container_tag": "integration_test",
            "searchMode": "memories"
        }))
        .send()
        .await
        .expect("Failed to search memories");
    let mem_only_data: serde_json::Value = mem_only_res.json().await.expect("Failed to parse");
    assert!(mem_only_data["results"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r.get("memory").is_some()));

    // Hybrid explicit
    let hybrid_res = client
        .post(format!("{base_url}/v4/search"))
        .json(&json!({
            "q": "rust",
            "container_tag": "integration_test",
            "searchMode": "hybrid"
        }))
        .send()
        .await
        .expect("Failed to search hybrid");
    assert!(hybrid_res.status().is_success());

    // Documents (should return 400 for /v4/search as per handler logic)
    let doc_mode_res = client
        .post(format!("{base_url}/v4/search"))
        .json(&json!({
            "q": "rust",
            "container_tag": "integration_test",
            "searchMode": "documents"
        }))
        .send()
        .await
        .expect("Failed to search documents");
    assert_eq!(doc_mode_res.status(), 400);
}

#[tokio::test]
async fn test_hybrid_search_empty_results() {
    let (addr, _temp_dir, _mock_server) = setup_test_app().await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let search_res = client
        .post(format!("{base_url}/v4/search"))
        .json(&json!({
            "q": "nonexistent",
            "container_tag": "empty_test"
        }))
        .send()
        .await
        .expect("Failed to search");

    assert!(search_res.status().is_success());
    let response: HybridSearchResponse = search_res.json().await.expect("Failed to parse response");
    assert_eq!(response.results.len(), 0);
}
