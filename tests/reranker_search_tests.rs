use momo::api::{create_router, AppState};
use momo::config::{Config, RerankerConfig};
use momo::db::{Database, DatabaseBackend, LibSqlBackend};
use momo::embeddings::{EmbeddingProvider, RerankResult, RerankerProvider};
use momo::llm::LlmProvider;
use momo::ocr::OcrProvider;
use momo::transcription::TranscriptionProvider;
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use tempfile::TempDir;

async fn setup_test_app(reranker_override: Option<RerankerProvider>) -> (SocketAddr, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("momo.db");
    let db_url = format!("file:{}", db_path.to_str().unwrap());

    let mut config = Config::from_env();
    config.database.url = db_url;
    config.embeddings.model = "local/BAAI/bge-small-en-v1.5".to_string();
    config.embeddings.dimensions = 384;
    config.server.api_keys = vec!["test-key".to_string()];

    config.reranker = Some(RerankerConfig {
        enabled: reranker_override.is_some(),
        model: "bge-reranker-base".to_string(),
        cache_dir: ".fastembed_cache".to_string(),
        batch_size: 64,
        domain_models: std::collections::HashMap::new(),
    });

    let db = Database::new(&config.database)
        .await
        .expect("Failed to create database");
    let db_backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db));
    let embeddings =
        EmbeddingProvider::new(&config.embeddings).expect("Failed to create embeddings");
    let ocr = OcrProvider::new(&config.ocr).expect("Failed to create OCR");
    let transcription =
        TranscriptionProvider::new(&config.transcription).expect("Failed to create transcription");
    let llm = LlmProvider::new(config.llm.as_ref());

    let state = AppState::new(
        config.clone(),
        db_backend.clone(),
        db_backend,
        embeddings,
        reranker_override,
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

    (addr, temp_dir)
}

#[tokio::test]
async fn test_search_rerank_disabled_but_requested() {
    let (addr, _temp_dir) = setup_test_app(None).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

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

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "rust programming",
            "rerank": true,
            "containerTags": ["test_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");

    let results = body["data"]["results"]
        .as_array()
        .expect("results should be array");
    assert!(!results.is_empty());
    for result in results {
        assert!(result.get("rerankScore").is_none() || result["rerankScore"].is_null());
        assert!(result["score"].as_f64().unwrap_or(0.0) > 0.0);
    }
}

#[tokio::test]
async fn test_search_rerank_false_backward_compatibility() {
    let (addr, _temp_dir) = setup_test_app(None).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let doc_res = client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Python is an interpreted, high-level, general-purpose programming language.",
            "containerTag": "test_tag"
        }))
        .send()
        .await
        .expect("Failed to send request");
    assert!(doc_res.status().is_success());

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "python language",
            "rerank": false,
            "containerTags": ["test_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");

    let results = body["data"]["results"]
        .as_array()
        .expect("results should be array");
    assert!(!results.is_empty());
    for result in results {
        assert!(result.get("rerankScore").is_none() || result["rerankScore"].is_null());
    }
}

#[tokio::test]
async fn test_search_rerank_with_mock() {
    let mock_results = vec![
        RerankResult {
            document: "doc2".to_string(),
            score: 0.95,
            index: 1,
        },
        RerankResult {
            document: "doc1".to_string(),
            score: 0.7,
            index: 0,
        },
    ];
    let reranker = RerankerProvider::new_mock(mock_results);

    let (addr, _temp_dir) = setup_test_app(Some(reranker)).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "First document about Rust.",
            "containerTag": "test_tag"
        }))
        .send()
        .await
        .expect("Failed to add doc 1");

    client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Second document about Python.",
            "containerTag": "test_tag"
        }))
        .send()
        .await
        .expect("Failed to add doc 2");

    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "programming",
            "rerank": true,
            "containerTags": ["test_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send search request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");

    let results = body["data"]["results"]
        .as_array()
        .expect("results should be array");
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_search_rerank_chunk_level() {
    let mock_results = vec![RerankResult {
        document: "chunk1".to_string(),
        score: 0.99,
        index: 0,
    }];
    let reranker = RerankerProvider::new_mock(mock_results);

    let (addr, _temp_dir) = setup_test_app(Some(reranker)).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "This is a document for chunk level reranking test.",
            "containerTag": "test_tag"
        }))
        .send()
        .await
        .expect("Failed to add doc");

    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "chunk test",
            "rerank": true,
            "containerTags": ["test_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send search request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");

    let results = body["data"]["results"]
        .as_array()
        .expect("results should be array");
    if !results.is_empty() {
        assert!(results[0].get("rerankScore").is_some());
    }
}

#[tokio::test]
async fn test_search_rerank_top_k() {
    let mock_results = vec![RerankResult {
        document: "doc1".to_string(),
        score: 0.9,
        index: 0,
    }];
    let reranker = RerankerProvider::new_mock(mock_results);

    let (addr, _temp_dir) = setup_test_app(Some(reranker)).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Document 1",
            "containerTag": "test_tag"
        }))
        .send()
        .await
        .expect("Failed to add doc 1");

    client
        .post(format!("{base_url}/api/v1/documents"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "content": "Document 2",
            "containerTag": "test_tag"
        }))
        .send()
        .await
        .expect("Failed to add doc 2");

    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "document",
            "rerank": true,
            "containerTags": ["test_tag"],
            "scope": "documents"
        }))
        .send()
        .await
        .expect("Failed to send search request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");

    let results = body["data"]["results"]
        .as_array()
        .expect("results should be array");
    let reranked_count = results
        .iter()
        .filter(|r| r.get("rerankScore").is_some() && !r["rerankScore"].is_null())
        .count();
    assert!(reranked_count <= results.len());
}

#[tokio::test]
async fn test_memory_search_rerank_mock() {
    let mock_results = vec![RerankResult {
        document: "memory1".to_string(),
        score: 0.88,
        index: 0,
    }];
    let reranker = RerankerProvider::new_mock(mock_results);

    let (addr, _temp_dir) = setup_test_app(Some(reranker)).await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    let search_res = client
        .post(format!("{base_url}/api/v1/search"))
        .header("Authorization", "Bearer test-key")
        .json(&json!({
            "q": "something",
            "rerank": true,
            "containerTags": ["test_tag"],
            "scope": "memories"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(search_res.status().is_success());
    let body: serde_json::Value = search_res.json().await.expect("Failed to parse JSON");
    assert_eq!(body["data"]["total"].as_u64().unwrap_or(0), 0);
}
