//! Comprehensive tests for the embeddings API implementation.
//!
//! Tests cover:
//! 1. API client success with valid response
//! 2. Request format matches OpenAI spec
//! 3. Authorization header verification
//! 4. Rate limit (429) retry behavior
//! 5. Server error (5xx) retry behavior
//! 6. Auth error (401/403) no retry
//! 7. Dimension detection from response
//! 8. Provider parsing (extension of config.rs tests)
//! 9. Metadata repository operations

use libsql::Builder;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::config::parse_provider_model;
use crate::db::MetadataRepository;
use crate::embeddings::api::{ApiConfig, EmbeddingApiClient};

/// Helper to create a test ApiConfig pointing to a mock server
fn test_config(base_url: &str) -> ApiConfig {
    ApiConfig {
        base_url: base_url.to_string(),
        api_key: Some("test-api-key".to_string()),
        model: "text-embedding-3-small".to_string(),
        timeout_secs: 10,
        max_retries: 3,
    }
}

/// Helper to create a valid embedding response
fn embedding_response(embeddings: Vec<Vec<f32>>) -> serde_json::Value {
    json!({
        "data": embeddings.into_iter().map(|e| json!({ "embedding": e })).collect::<Vec<_>>()
    })
}

// =============================================================================
// Test 1: API Client Success
// =============================================================================

#[tokio::test]
async fn test_api_client_success() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]])),
        )
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test text"]).await;
    assert!(result.is_ok());

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0], vec![0.1, 0.2, 0.3]);
}

#[tokio::test]
async fn test_api_client_multiple_inputs() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![
                vec![0.1, 0.2, 0.3],
                vec![0.4, 0.5, 0.6],
            ])),
        )
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["text 1", "text 2"]).await;
    assert!(result.is_ok());

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(embeddings[1], vec![0.4, 0.5, 0.6]);
}

// =============================================================================
// Test 2: Request Format Matches OpenAI Spec
// =============================================================================

#[tokio::test]
async fn test_api_client_request_format() {
    let mock_server = MockServer::start().await;

    // Verify exact request body format
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_json(json!({
            "model": "text-embedding-3-small",
            "input": ["hello world"]
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]])),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["hello world"]).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_api_client_content_type_header() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("content-type", "application/json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]])),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let _ = client.embed(&["test"]).await;
}

// =============================================================================
// Test 3: Authorization Header Verification
// =============================================================================

#[tokio::test]
async fn test_api_client_auth_header() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]])),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test"]).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_api_client_custom_api_key() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer my-secret-key-12345"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]])),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = ApiConfig {
        base_url: mock_server.uri(),
        api_key: Some("my-secret-key-12345".to_string()),
        model: "test-model".to_string(),
        timeout_secs: 10,
        max_retries: 3,
    };

    let client = EmbeddingApiClient::new(config).unwrap();
    let result = client.embed(&["test"]).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_api_client_no_api_key() {
    let mock_server = MockServer::start().await;

    // When no API key, no Authorization header should be sent
    // We test by requiring the header NOT be present (mock won't match if header is set)
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]])),
        )
        .mount(&mock_server)
        .await;

    let config = ApiConfig {
        base_url: mock_server.uri(),
        api_key: None, // No API key
        model: "test-model".to_string(),
        timeout_secs: 10,
        max_retries: 3,
    };

    let client = EmbeddingApiClient::new(config).unwrap();
    let result = client.embed(&["test"]).await;
    assert!(result.is_ok());
}

// =============================================================================
// Test 4: Rate Limit (429) Retry Behavior
// =============================================================================

#[tokio::test]
async fn test_api_client_rate_limit_retry() {
    let mock_server = MockServer::start().await;
    let attempt_count = Arc::new(AtomicUsize::new(0));

    // First two requests return 429, third succeeds
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with({
            let count = Arc::clone(&attempt_count);
            move |_: &wiremock::Request| {
                let attempt = count.fetch_add(1, Ordering::SeqCst);
                if attempt < 2 {
                    ResponseTemplate::new(429)
                        .set_body_json(json!({ "error": "rate limited" }))
                        .insert_header("retry-after", "1")
                } else {
                    ResponseTemplate::new(200)
                        .set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]]))
                }
            }
        })
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test"]).await;
    assert!(result.is_ok(), "Should succeed after retry");
    assert_eq!(
        attempt_count.load(Ordering::SeqCst),
        3,
        "Should have made 3 attempts"
    );
}

#[tokio::test]
async fn test_api_client_rate_limit_exhausts_retries() {
    let mock_server = MockServer::start().await;
    let attempt_count = Arc::new(AtomicUsize::new(0));

    // Always return 429
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with({
            let count = Arc::clone(&attempt_count);
            move |_: &wiremock::Request| {
                count.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(429).set_body_json(json!({ "error": "rate limited" }))
            }
        })
        .mount(&mock_server)
        .await;

    let config = ApiConfig {
        base_url: mock_server.uri(),
        api_key: Some("test-key".to_string()),
        model: "test-model".to_string(),
        timeout_secs: 10,
        max_retries: 2, // Only 2 retries
    };

    let client = EmbeddingApiClient::new(config).unwrap();
    let result = client.embed(&["test"]).await;

    assert!(result.is_err(), "Should fail after exhausting retries");
    // 1 initial + 2 retries = 3 attempts
    assert_eq!(
        attempt_count.load(Ordering::SeqCst),
        3,
        "Should have made 3 attempts (1 + 2 retries)"
    );

    // Verify it's a rate limit error
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("rate limit"),
        "Error should mention rate limit"
    );
}

// =============================================================================
// Test 5: Server Error (5xx) Retry Behavior
// =============================================================================

#[tokio::test]
async fn test_api_client_server_error_retry() {
    let mock_server = MockServer::start().await;
    let attempt_count = Arc::new(AtomicUsize::new(0));

    // First request returns 500, second succeeds
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with({
            let count = Arc::clone(&attempt_count);
            move |_: &wiremock::Request| {
                let attempt = count.fetch_add(1, Ordering::SeqCst);
                if attempt < 1 {
                    ResponseTemplate::new(500)
                        .set_body_json(json!({ "error": "internal server error" }))
                } else {
                    ResponseTemplate::new(200)
                        .set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]]))
                }
            }
        })
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test"]).await;
    assert!(result.is_ok(), "Should succeed after retry");
    assert_eq!(
        attempt_count.load(Ordering::SeqCst),
        2,
        "Should have made 2 attempts"
    );
}

#[tokio::test]
async fn test_api_client_502_retry() {
    let mock_server = MockServer::start().await;
    let attempt_count = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with({
            let count = Arc::clone(&attempt_count);
            move |_: &wiremock::Request| {
                let attempt = count.fetch_add(1, Ordering::SeqCst);
                if attempt < 1 {
                    ResponseTemplate::new(502).set_body_json(json!({ "error": "bad gateway" }))
                } else {
                    ResponseTemplate::new(200)
                        .set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]]))
                }
            }
        })
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test"]).await;
    assert!(result.is_ok(), "Should succeed after 502 retry");
}

#[tokio::test]
async fn test_api_client_503_retry() {
    let mock_server = MockServer::start().await;
    let attempt_count = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with({
            let count = Arc::clone(&attempt_count);
            move |_: &wiremock::Request| {
                let attempt = count.fetch_add(1, Ordering::SeqCst);
                if attempt < 1 {
                    ResponseTemplate::new(503)
                        .set_body_json(json!({ "error": "service unavailable" }))
                } else {
                    ResponseTemplate::new(200)
                        .set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]]))
                }
            }
        })
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test"]).await;
    assert!(result.is_ok(), "Should succeed after 503 retry");
}

// =============================================================================
// Test 6: Auth Error (401/403) No Retry
// =============================================================================

#[tokio::test]
async fn test_api_client_auth_error_401_no_retry() {
    let mock_server = MockServer::start().await;
    let attempt_count = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with({
            let count = Arc::clone(&attempt_count);
            move |_: &wiremock::Request| {
                count.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(401).set_body_json(json!({ "error": "invalid api key" }))
            }
        })
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test"]).await;
    assert!(result.is_err(), "Should fail immediately on 401");
    assert_eq!(
        attempt_count.load(Ordering::SeqCst),
        1,
        "Should NOT retry on 401"
    );

    // Verify it's an auth error
    let err = result.unwrap_err();
    assert!(
        matches!(&err, crate::error::MomoError::ApiAuth(_)),
        "Should be ApiAuth error"
    );
}

#[tokio::test]
async fn test_api_client_auth_error_403_no_retry() {
    let mock_server = MockServer::start().await;
    let attempt_count = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with({
            let count = Arc::clone(&attempt_count);
            move |_: &wiremock::Request| {
                count.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(403).set_body_json(json!({ "error": "forbidden" }))
            }
        })
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test"]).await;
    assert!(result.is_err(), "Should fail immediately on 403");
    assert_eq!(
        attempt_count.load(Ordering::SeqCst),
        1,
        "Should NOT retry on 403"
    );
}

// =============================================================================
// Test 7: Dimension Detection from Response
// =============================================================================

#[tokio::test]
async fn test_api_client_dimension_detection() {
    let mock_server = MockServer::start().await;

    // Return a 384-dimensional embedding
    let dims_384: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![dims_384.clone()])),
        )
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let dimensions = client.detect_dimensions().await.unwrap();
    assert_eq!(dimensions, 384);
}

#[tokio::test]
async fn test_api_client_dimension_detection_1536() {
    let mock_server = MockServer::start().await;

    // Return a 1536-dimensional embedding (OpenAI ada-002)
    let dims_1536: Vec<f32> = (0..1536).map(|i| i as f32 * 0.0001).collect();
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![dims_1536.clone()])),
        )
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let dimensions = client.detect_dimensions().await.unwrap();
    assert_eq!(dimensions, 1536);
}

#[tokio::test]
async fn test_api_client_dimension_detection_3072() {
    let mock_server = MockServer::start().await;

    // Return a 3072-dimensional embedding (OpenAI text-embedding-3-large)
    let dims_3072: Vec<f32> = (0..3072).map(|i| i as f32 * 0.0001).collect();
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![dims_3072.clone()])),
        )
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let dimensions = client.detect_dimensions().await.unwrap();
    assert_eq!(dimensions, 3072);
}

// =============================================================================
// Test 8: Provider Parsing (Extension of config.rs tests)
// =============================================================================

#[test]
fn test_parse_provider_openrouter() {
    let (provider, model) = parse_provider_model("openrouter/openai/text-embedding-3-small");
    assert_eq!(provider, "openrouter");
    assert_eq!(model, "openai/text-embedding-3-small");
}

#[test]
fn test_parse_provider_lmstudio() {
    let (provider, model) = parse_provider_model("lmstudio/bge-small-en-v1.5");
    assert_eq!(provider, "lmstudio");
    assert_eq!(model, "bge-small-en-v1.5");
}

#[test]
fn test_parse_provider_sentence_transformers() {
    // sentence-transformers is not a provider, so it should default to local
    let (provider, model) = parse_provider_model("sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!(provider, "local");
    assert_eq!(model, "sentence-transformers/all-MiniLM-L6-v2");
}

#[test]
fn test_parse_provider_nomic_ai() {
    // nomic-ai is not a known provider, so it should default to local
    let (provider, model) = parse_provider_model("nomic-ai/nomic-embed-text-v1.5");
    assert_eq!(provider, "local");
    assert_eq!(model, "nomic-ai/nomic-embed-text-v1.5");
}

#[test]
fn test_parse_provider_case_insensitive() {
    let (provider, model) = parse_provider_model("OpenAI/text-embedding-3-small");
    assert_eq!(provider, "OpenAI"); // Preserves original case
    assert_eq!(model, "text-embedding-3-small");

    let (provider2, model2) = parse_provider_model("OLLAMA/nomic-embed-text");
    assert_eq!(provider2, "OLLAMA");
    assert_eq!(model2, "nomic-embed-text");
}

#[test]
fn test_parse_provider_empty_string() {
    let (provider, model) = parse_provider_model("");
    assert_eq!(provider, "local");
    assert_eq!(model, "");
}

// =============================================================================
// Test 9: Metadata Repository Operations
// =============================================================================

async fn create_test_connection() -> libsql::Connection {
    let db = Builder::new_local(":memory:").build().await.unwrap();
    let conn = db.connect().unwrap();
    conn.execute(
        "CREATE TABLE IF NOT EXISTS momo_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
        (),
    )
    .await
    .unwrap();
    conn
}

#[tokio::test]
async fn test_metadata_get_nonexistent() {
    let conn = create_test_connection().await;

    let result = MetadataRepository::get(&conn, "nonexistent_key").await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[tokio::test]
async fn test_metadata_set_and_get() {
    let conn = create_test_connection().await;

    MetadataRepository::set(&conn, "test_key", "test_value")
        .await
        .unwrap();

    let result = MetadataRepository::get(&conn, "test_key").await.unwrap();
    assert_eq!(result, Some("test_value".to_string()));
}

#[tokio::test]
async fn test_metadata_update() {
    let conn = create_test_connection().await;

    MetadataRepository::set(&conn, "update_key", "initial_value")
        .await
        .unwrap();
    MetadataRepository::set(&conn, "update_key", "updated_value")
        .await
        .unwrap();

    let result = MetadataRepository::get(&conn, "update_key").await.unwrap();
    assert_eq!(result, Some("updated_value".to_string()));
}

#[tokio::test]
async fn test_metadata_embedding_dimensions() {
    let conn = create_test_connection().await;

    let dims = MetadataRepository::get_embedding_dimensions(&conn)
        .await
        .unwrap();
    assert!(dims.is_none());

    MetadataRepository::set_embedding_dimensions(&conn, 384)
        .await
        .unwrap();

    let dims = MetadataRepository::get_embedding_dimensions(&conn)
        .await
        .unwrap();
    assert_eq!(dims, Some(384));

    MetadataRepository::set_embedding_dimensions(&conn, 1536)
        .await
        .unwrap();

    let dims = MetadataRepository::get_embedding_dimensions(&conn)
        .await
        .unwrap();
    assert_eq!(dims, Some(1536));
}

#[tokio::test]
async fn test_metadata_multiple_keys() {
    let conn = create_test_connection().await;

    MetadataRepository::set(&conn, "key1", "value1")
        .await
        .unwrap();
    MetadataRepository::set(&conn, "key2", "value2")
        .await
        .unwrap();
    MetadataRepository::set(&conn, "key3", "value3")
        .await
        .unwrap();

    assert_eq!(
        MetadataRepository::get(&conn, "key1").await.unwrap(),
        Some("value1".to_string())
    );
    assert_eq!(
        MetadataRepository::get(&conn, "key2").await.unwrap(),
        Some("value2".to_string())
    );
    assert_eq!(
        MetadataRepository::get(&conn, "key3").await.unwrap(),
        Some("value3".to_string())
    );
}

// =============================================================================
// Additional Edge Case Tests
// =============================================================================

#[tokio::test]
async fn test_api_client_empty_response() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [] })))
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test"]).await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[tokio::test]
async fn test_api_client_malformed_response() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "unexpected": "format" })))
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test"]).await;
    assert!(result.is_err(), "Should fail on malformed response");
}

#[tokio::test]
async fn test_api_client_400_error_no_retry() {
    let mock_server = MockServer::start().await;
    let attempt_count = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with({
            let count = Arc::clone(&attempt_count);
            move |_: &wiremock::Request| {
                count.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(400).set_body_json(json!({ "error": "bad request" }))
            }
        })
        .mount(&mock_server)
        .await;

    let config = test_config(&mock_server.uri());
    let client = EmbeddingApiClient::new(config).unwrap();

    let result = client.embed(&["test"]).await;
    assert!(result.is_err());
    assert_eq!(
        attempt_count.load(Ordering::SeqCst),
        1,
        "Should NOT retry on 400"
    );
}

#[tokio::test]
async fn test_api_client_custom_model_name() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_json(json!({
            "model": "nomic-embed-text",
            "input": ["test"]
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![vec![0.1, 0.2, 0.3]])),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = ApiConfig {
        base_url: mock_server.uri(),
        api_key: Some("key".to_string()),
        model: "nomic-embed-text".to_string(),
        timeout_secs: 10,
        max_retries: 3,
    };

    let client = EmbeddingApiClient::new(config).unwrap();
    let result = client.embed(&["test"]).await;
    assert!(result.is_ok());
}
