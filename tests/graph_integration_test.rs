use momo::api::create_router;
use momo::config::{Config, DatabaseConfig, EmbeddingsConfig, LlmConfig};
use momo::db::repository::{DocumentRepository, MemoryRepository, MemorySourcesRepository};
use momo::db::{Database, DatabaseBackend, LibSqlBackend};
use momo::embeddings::EmbeddingProvider;
use momo::llm::LlmProvider;
use momo::models::{Document, Memory, MemoryRelationType};
use momo::ocr::OcrProvider;
use momo::transcription::TranscriptionProvider;
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn setup_test_app() -> (SocketAddr, TempDir, MockServer, Database) {
    let mock_server = MockServer::start().await;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("momo_graph_test.db");
    let db_url = format!("file:{}", db_path.to_str().unwrap());

    // Minimal embedding mock (not used heavily by graph tests but required by state)
    let embedding = vec![0.1f32; 384];
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"embedding": embedding}]
        })))
        .mount(&mock_server)
        .await;

    // Minimal LLM mock
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [{"index": 0, "message": {"role":"assistant","content":"{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.database = DatabaseConfig {
        url: db_url.clone(),
        auth_token: None,
        local_path: None,
    };

    config.server.api_keys = vec!["test-key".to_string()];

    config.embeddings = EmbeddingsConfig {
        model: "BAAI/bge-small-en-v1.5".to_string(),
        dimensions: 384,
        batch_size: 8,
    };

    config.llm = Some(LlmConfig {
        model: "openai/gpt-test".to_string(),
        api_key: Some("test-key".to_string()),
        base_url: Some(mock_server.uri()),
        timeout_secs: 5,
        max_retries: 0,
        enable_query_rewrite: false,
        query_rewrite_cache_size: 1000,
        query_rewrite_timeout_secs: 2,
        enable_auto_relations: false,
        enable_contradiction_detection: false,
        filter_prompt: None,
    });

    let db = Database::new(&config.database)
        .await
        .expect("Failed to create database");
    let db_backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db.clone()));
    let embeddings =
        EmbeddingProvider::new(&config.embeddings).expect("Failed to create embeddings");
    let ocr = OcrProvider::new(&config.ocr).expect("Failed to create OCR");
    let transcription =
        TranscriptionProvider::new(&config.transcription).expect("Failed to create transcription");
    let llm = LlmProvider::new(config.llm.as_ref());

    let state = momo::api::AppState::new(
        config.clone(),
        db_backend.clone(),
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

    (addr, temp_dir, mock_server, db)
}

#[tokio::test]
async fn test_graph_endpoints_return_expected_nodes_and_edges() {
    let (addr, _tmp, _mock, db) = setup_test_app().await;
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // Prepare DB: create three memories with relations and one document source
    let conn = db.connect().expect("connect");

    let mut m1 = Memory::new(
        "m1".to_string(),
        "Memory 1".to_string(),
        "space1".to_string(),
    );
    m1.container_tag = Some("graph_test".to_string());
    m1.memory_relations
        .insert("m2".to_string(), MemoryRelationType::Updates);

    let mut m2 = Memory::new(
        "m2".to_string(),
        "Memory 2".to_string(),
        "space1".to_string(),
    );
    m2.container_tag = Some("graph_test".to_string());
    m2.memory_relations
        .insert("m3".to_string(), MemoryRelationType::Extends);

    let mut m3 = Memory::new(
        "m3".to_string(),
        "Memory 3".to_string(),
        "space1".to_string(),
    );
    m3.container_tag = Some("graph_test".to_string());

    MemoryRepository::create(&conn, &m1)
        .await
        .expect("create m1");
    MemoryRepository::create(&conn, &m2)
        .await
        .expect("create m2");
    MemoryRepository::create(&conn, &m3)
        .await
        .expect("create m3");

    // Create a document and link as source for m1
    let mut doc = Document::new("doc1".to_string());
    doc.title = Some("Doc 1".to_string());
    DocumentRepository::create(&conn, &doc)
        .await
        .expect("create doc");
    MemorySourcesRepository::create(&conn, "m1", "doc1", None)
        .await
        .expect("create memory source");

    // Call memory neighborhood graph (v1 API)
    let res = client
        .get(format!(
            "{base_url}/api/v1/memories/m1/graph?depth=2&maxNodes=50"
        ))
        .header("Authorization", "Bearer test-key")
        .send()
        .await
        .expect("request");
    let status = res.status();
    let body: serde_json::Value = res.json().await.expect("parse json");
    assert!(
        status.is_success(),
        "status is success, got {status}: {body}"
    );

    // v1 wraps response in {"data": {...}}
    let data = body.get("data").expect("data envelope");

    // Basic shape checks
    let nodes = data
        .get("nodes")
        .and_then(|v| v.as_array())
        .expect("nodes array");
    let edges = data
        .get("links")
        .and_then(|v| v.as_array())
        .expect("links array");

    // Expect at least three memories and one document
    let ids: Vec<String> = nodes
        .iter()
        .filter_map(|n| n.get("id").and_then(|s| s.as_str()).map(|s| s.to_string()))
        .collect();
    assert!(ids.contains(&"m1".to_string()));
    assert!(ids.contains(&"m2".to_string()));
    assert!(ids.contains(&"m3".to_string()));
    assert!(ids.contains(&"doc1".to_string()));

    // Check edge types and presence
    let mut found_updates = false;
    let mut found_extends = false;
    let mut found_source = false;
    for e in edges {
        let source = e.get("source").and_then(|v| v.as_str()).unwrap_or("");
        let target = e.get("target").and_then(|v| v.as_str()).unwrap_or("");
        let etype = e.get("type").and_then(|v| v.as_str()).unwrap_or("");

        if source == "m1" && target == "m2" && etype == "updates" {
            found_updates = true;
        }
        if source == "m2" && target == "m3" && etype == "relatesTo" {
            found_extends = true;
        }
        if source == "m1" && target == "doc1" && etype == "sources" {
            found_source = true;
        }
    }

    assert!(found_updates, "found updates edge m1->m2");
    assert!(found_extends, "found relates_to edge m2->m3");
    assert!(found_source, "found source edge m1->doc1");

    // Container-level graph (v1 API)
    let res2 = client
        .get(format!(
            "{base_url}/api/v1/containers/graph_test/graph?maxNodes=100"
        ))
        .header("Authorization", "Bearer test-key")
        .send()
        .await
        .expect("request2");
    assert!(res2.status().is_success());
    let body2: serde_json::Value = res2.json().await.expect("parse json2");
    let data2 = body2.get("data").expect("data envelope");
    let nodes2 = data2
        .get("nodes")
        .and_then(|v| v.as_array())
        .expect("nodes2");
    let ids2: Vec<String> = nodes2
        .iter()
        .filter_map(|n| n.get("id").and_then(|s| s.as_str()).map(|s| s.to_string()))
        .collect();
    assert!(ids2.contains(&"m1".to_string()));
    assert!(ids2.contains(&"m2".to_string()));
    assert!(ids2.contains(&"m3".to_string()));
}
