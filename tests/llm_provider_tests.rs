use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use momo::config::LlmConfig;
use momo::error::MomoError;
use momo::llm::{LlmApiClient, LlmBackend, LlmProvider};

fn llm_config(model: &str) -> LlmConfig {
    LlmConfig {
        model: model.to_string(),
        api_key: Some("test-key".to_string()),
        base_url: None,
        timeout_secs: 30,
        max_retries: 3,
        enable_query_rewrite: false,
        query_rewrite_cache_size: 1000,
        query_rewrite_timeout_secs: 2,
        enable_auto_relations: true,
        enable_contradiction_detection: false,
        enable_llm_filter: false,
        filter_prompt: None,
    }
}

fn llm_config_with_base_url(model: &str, base_url: String, max_retries: u32) -> LlmConfig {
    LlmConfig {
        model: model.to_string(),
        api_key: Some("test-key".to_string()),
        base_url: Some(base_url),
        timeout_secs: 5,
        max_retries,
        enable_query_rewrite: false,
        query_rewrite_cache_size: 1000,
        query_rewrite_timeout_secs: 2,
        enable_auto_relations: true,
        enable_contradiction_detection: false,
        enable_llm_filter: false,
        filter_prompt: None,
    }
}

fn completion_body(content: &str) -> serde_json::Value {
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
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2
        }
    })
}

fn api_error_body(message: &str, error_type: &str, code: &str) -> serde_json::Value {
    json!({
        "error": {
            "message": message,
            "type": error_type,
            "param": serde_json::Value::Null,
            "code": code
        }
    })
}

#[test]
fn test_openai_provider_detection() {
    let config = llm_config("openai/gpt-4o");
    let provider = LlmProvider::new(Some(&config));

    assert!(matches!(provider.backend(), LlmBackend::OpenAI));
    assert_eq!(provider.base_url(), Some("https://api.openai.com/v1"));
}

#[test]
fn test_openrouter_provider_detection() {
    let config = llm_config("openrouter/openai/gpt-4o");
    let provider = LlmProvider::new(Some(&config));

    assert!(matches!(provider.backend(), LlmBackend::OpenRouter));
    assert_eq!(provider.base_url(), Some("https://openrouter.ai/api/v1"));
}

#[test]
fn test_ollama_provider_detection() {
    let config = llm_config("ollama/llama3.2");
    let provider = LlmProvider::new(Some(&config));

    assert!(matches!(provider.backend(), LlmBackend::Ollama));
    assert_eq!(provider.base_url(), Some("http://localhost:11434/v1"));
}

#[test]
fn test_unavailable_provider() {
    let provider = LlmProvider::new(None);

    assert!(matches!(provider.backend(), LlmBackend::Unavailable { .. }));
}

#[test]
fn test_is_available_true() {
    let config = llm_config("openai/gpt-4o");
    let provider = LlmProvider::new(Some(&config));

    assert!(provider.is_available());
}

#[test]
fn test_is_available_false() {
    let provider = LlmProvider::new(None);

    assert!(!provider.is_available());
}

#[test]
fn test_provider_clone() {
    let config = llm_config("openrouter/openai/gpt-4o-mini");
    let provider = LlmProvider::new(Some(&config));
    let cloned = provider.clone();

    assert!(matches!(provider.backend(), LlmBackend::OpenRouter));
    assert!(matches!(cloned.backend(), LlmBackend::OpenRouter));
    assert!(cloned.is_available());
    assert_eq!(
        cloned.config().map(|c| c.model.as_str()),
        Some(config.model.as_str())
    );
}

#[test]
fn test_api_client_uses_provider_default_base_url() {
    let config = llm_config("openrouter/openai/gpt-4o-mini");
    let client = LlmApiClient::new(&config);

    match client {
        Ok(value) => assert_eq!(value.base_url(), "https://openrouter.ai/api/v1"),
        Err(error) => panic!("Expected API client creation to succeed, got: {error}"),
    }
}

#[tokio::test]
async fn test_complete_returns_response_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("Hello from mock")))
        .expect(1)
        .mount(&server)
        .await;

    let config = llm_config_with_base_url("openai/gpt-4o-mini", format!("{}/v1", server.uri()), 1);
    let provider = LlmProvider::new(Some(&config));

    let result = provider.complete("Hello", None).await;

    match result {
        Ok(value) => assert_eq!(value, "Hello from mock"),
        Err(error) => panic!("Expected completion to succeed, got: {error}"),
    }
}

#[tokio::test]
async fn test_retry_on_server_error() {
    let server = MockServer::start().await;
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_mock = Arc::clone(&attempts);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_request: &Request| {
            if attempts_for_mock.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(500).set_body_string("upstream temporary failure")
            } else {
                ResponseTemplate::new(200).set_body_json(completion_body("Recovered response"))
            }
        })
        .mount(&server)
        .await;

    let config = llm_config_with_base_url("openai/gpt-4o-mini", format!("{}/v1", server.uri()), 2);
    let provider = LlmProvider::new(Some(&config));

    let result = provider.complete("Retry test", None).await;

    match result {
        Ok(value) => assert_eq!(value, "Recovered response"),
        Err(error) => panic!("Expected retry completion to succeed, got: {error}"),
    }
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_rate_limit_handling() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "7")
                .set_body_json(api_error_body(
                    "Rate limit exceeded",
                    "insufficient_quota",
                    "insufficient_quota",
                )),
        )
        .mount(&server)
        .await;

    let config = llm_config_with_base_url("openai/gpt-4o-mini", format!("{}/v1", server.uri()), 1);
    let provider = LlmProvider::new(Some(&config));

    let result = provider.complete("Rate limit test", None).await;

    assert!(matches!(
        result,
        Err(MomoError::LlmRateLimit { retry_after: None })
    ));
}

#[tokio::test]
async fn test_auth_error_returns_llm_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_json(api_error_body(
            "Invalid API key",
            "invalid_request_error",
            "invalid_api_key",
        )))
        .mount(&server)
        .await;

    let config = llm_config_with_base_url("openai/gpt-4o-mini", format!("{}/v1", server.uri()), 1);
    let provider = LlmProvider::new(Some(&config));

    let result = provider.complete("Auth test", None).await;

    match result {
        Err(MomoError::Llm(message)) => {
            assert!(message.to_lowercase().contains("authentication failed"));
        }
        other => panic!("Expected Llm auth error, got: {other:?}"),
    }
}

#[tokio::test]
async fn test_empty_prompt_validation() {
    let config = llm_config("openai/gpt-4o-mini");
    let provider = LlmProvider::new(Some(&config));

    let result = provider.complete("   ", None).await;

    match result {
        Err(MomoError::Validation(message)) => {
            assert!(message.contains("Prompt cannot be empty"));
        }
        other => panic!("Expected Validation error, got: {other:?}"),
    }
}

#[tokio::test]
async fn test_complete_stream_unavailable_provider_fails_fast() {
    let provider = LlmProvider::new(None);
    let stream = provider.complete_stream("Hello");
    futures::pin_mut!(stream);

    let first = stream.next().await;

    assert!(matches!(first, Some(Err(MomoError::LlmUnavailable(_)))));

    let second = stream.next().await;
    assert!(second.is_none());
}

#[tokio::test]
async fn test_complete_stream_no_api_key_fails() {
    let mut config = llm_config("openai/gpt-4o");
    config.api_key = None;
    let provider = LlmProvider::new(Some(&config));
    let stream = provider.complete_stream("Hello");
    futures::pin_mut!(stream);

    let first = stream.next().await;

    assert!(matches!(first, Some(Err(MomoError::Llm(_)))));
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
struct TestResponse {
    message: String,
    count: i32,
}

#[tokio::test]
async fn test_complete_json_unavailable_provider() {
    let provider = LlmProvider::new(None);
    let result = provider.complete_json("test prompt", None).await;
    assert!(matches!(result, Err(MomoError::LlmUnavailable(_))));
}

#[tokio::test]
async fn test_complete_structured_unavailable_provider() {
    let provider = LlmProvider::new(None);
    let result: Result<TestResponse, MomoError> = provider.complete_structured("test prompt").await;
    assert!(matches!(result, Err(MomoError::LlmUnavailable(_))));
}

#[tokio::test]
async fn test_complete_json_no_api_key() {
    let mut config = llm_config("openai/gpt-4o");
    config.api_key = None;
    let provider = LlmProvider::new(Some(&config));
    let result = provider.complete_json("test prompt", None).await;
    assert!(matches!(result, Err(MomoError::Llm(_))));
}
