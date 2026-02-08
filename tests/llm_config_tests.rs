use axum::{http::StatusCode, response::IntoResponse};
use std::env;

use momo::config::{parse_llm_provider_model, Config, LlmConfig, KNOWN_LLM_PROVIDERS};
use momo::error::MomoError;

#[test]
fn test_llm_config_openai() {
    let (provider, model) = parse_llm_provider_model("openai/gpt-4o");
    assert_eq!(provider, "openai");
    assert_eq!(model, "gpt-4o");
}

#[test]
fn test_llm_config_ollama() {
    let (provider, model) = parse_llm_provider_model("ollama/llama3.2");
    assert_eq!(provider, "ollama");
    assert_eq!(model, "llama3.2");
}

#[test]
fn test_llm_config_openrouter() {
    let (provider, model) = parse_llm_provider_model("openrouter/anthropic/claude-3.5-sonnet");
    assert_eq!(provider, "openrouter");
    assert_eq!(model, "anthropic/claude-3.5-sonnet");
}

#[test]
fn test_llm_config_unknown_provider_defaults_to_local() {
    let (provider, model) = parse_llm_provider_model("some-custom-model");
    assert_eq!(provider, "local");
    assert_eq!(model, "some-custom-model");
}

#[test]
fn test_llm_config_unknown_prefix_defaults_to_local() {
    let (provider, model) = parse_llm_provider_model("unknown/model-name");
    assert_eq!(provider, "local");
    assert_eq!(model, "unknown/model-name");
}

#[test]
fn test_known_llm_providers_constant() {
    assert!(KNOWN_LLM_PROVIDERS.contains(&"openai"));
    assert!(KNOWN_LLM_PROVIDERS.contains(&"openrouter"));
    assert!(KNOWN_LLM_PROVIDERS.contains(&"ollama"));
    assert!(KNOWN_LLM_PROVIDERS.contains(&"lmstudio"));
    assert_eq!(KNOWN_LLM_PROVIDERS.len(), 4);
}

#[test]
fn test_llm_config_none_when_no_env() {
    env::remove_var("LLM_MODEL");

    let config = Config::default();

    assert!(
        config.llm.is_none(),
        "LlmConfig should be None when LLM_MODEL is not set"
    );
}

#[test]
fn test_llm_config_some_when_model_env_set() {
    env::set_var("LLM_MODEL", "openai/gpt-4o-mini");
    env::remove_var("LLM_API_KEY");
    env::remove_var("LLM_BASE_URL");
    env::remove_var("LLM_TIMEOUT");
    env::remove_var("LLM_MAX_RETRIES");

    let config = Config::default();

    assert!(
        config.llm.is_some(),
        "LlmConfig should be Some when LLM_MODEL is set"
    );

    let llm = config.llm.unwrap();
    assert_eq!(llm.model, "openai/gpt-4o-mini");
    assert!(llm.api_key.is_none());
    assert!(llm.base_url.is_none());
    assert_eq!(llm.timeout_secs, 30);
    assert_eq!(llm.max_retries, 3);

    env::remove_var("LLM_MODEL");
}

#[test]
fn test_llm_config_with_all_env_vars() {
    env::set_var("LLM_MODEL", "openrouter/anthropic/claude-3.5-sonnet");
    env::set_var("LLM_API_KEY", "sk-test-key");
    env::set_var("LLM_BASE_URL", "https://api.custom.com/v1");
    env::set_var("LLM_TIMEOUT", "60");
    env::set_var("LLM_MAX_RETRIES", "5");

    let config = Config::default();

    let llm = config.llm.expect("LlmConfig should exist");
    assert_eq!(llm.model, "openrouter/anthropic/claude-3.5-sonnet");
    assert_eq!(llm.api_key, Some("sk-test-key".to_string()));
    assert_eq!(llm.base_url, Some("https://api.custom.com/v1".to_string()));
    assert_eq!(llm.timeout_secs, 60);
    assert_eq!(llm.max_retries, 5);

    env::remove_var("LLM_MODEL");
    env::remove_var("LLM_API_KEY");
    env::remove_var("LLM_BASE_URL");
    env::remove_var("LLM_TIMEOUT");
    env::remove_var("LLM_MAX_RETRIES");
}

#[test]
fn test_llm_error_status_code_mapping() {
    let error = MomoError::Llm("Test error".to_string());
    let response = error.into_response();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[test]
fn test_llm_unavailable_error_status_code_mapping() {
    let error = MomoError::LlmUnavailable("Service down".to_string());
    let response = error.into_response();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn test_llm_rate_limit_error_status_code_mapping() {
    let error = MomoError::LlmRateLimit {
        retry_after: Some(60),
    };
    let response = error.into_response();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[test]
fn test_llm_rate_limit_error_without_retry_after() {
    let error = MomoError::LlmRateLimit { retry_after: None };
    let response = error.into_response();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[test]
fn test_llm_config_clone() {
    let config = LlmConfig {
        model: "openai/gpt-4o".to_string(),
        api_key: Some("secret".to_string()),
        base_url: Some("https://api.openai.com".to_string()),
        timeout_secs: 30,
        max_retries: 3,
        enable_query_rewrite: false,
        query_rewrite_cache_size: 1000,
        query_rewrite_timeout_secs: 2,
        enable_auto_relations: true,
        enable_contradiction_detection: false,
        enable_llm_filter: false,
        filter_prompt: None,
    };

    let cloned = config.clone();

    assert_eq!(cloned.model, config.model);
    assert_eq!(cloned.api_key, config.api_key);
    assert_eq!(cloned.base_url, config.base_url);
    assert_eq!(cloned.timeout_secs, config.timeout_secs);
    assert_eq!(cloned.max_retries, config.max_retries);
    assert_eq!(cloned.enable_query_rewrite, config.enable_query_rewrite);
    assert_eq!(
        cloned.query_rewrite_cache_size,
        config.query_rewrite_cache_size
    );
    assert_eq!(
        cloned.query_rewrite_timeout_secs,
        config.query_rewrite_timeout_secs
    );
    assert_eq!(cloned.enable_auto_relations, config.enable_auto_relations);
}
