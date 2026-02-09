use std::time::Duration;

use serde_json::Value;

use async_openai::{
    config::OpenAIConfig,
    error::{ApiError, OpenAIError},
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequest, CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
        Stop,
    },
    Client,
};

use crate::{
    config::{parse_llm_provider_model, LlmConfig},
    error::{MomoError, Result},
    llm::provider::CompletionOptions,
};

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";

#[derive(Debug, Clone)]
struct ApiConfig {
    base_url: String,
    api_key: Option<String>,
    model: String,
    timeout_secs: u64,
    max_retries: u32,
}

#[derive(Clone)]
pub struct LlmApiClient {
    client: Client<OpenAIConfig>,
    config: ApiConfig,
}

impl LlmApiClient {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        let api_config = ApiConfig::from_llm_config(config);

        let (provider, _) = parse_llm_provider_model(&config.model);
        let needs_api_key = !matches!(
            provider.to_lowercase().as_str(),
            "ollama" | "local" | "lmstudio"
        );

        if needs_api_key && api_config.api_key.is_none() {
            return Err(MomoError::Llm(
                "API key required for this provider".to_string(),
            ));
        }

        let openai_config = OpenAIConfig::new()
            .with_api_base(api_config.base_url.clone())
            .with_api_key(api_config.api_key.clone().unwrap_or_default());

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(api_config.timeout_secs))
            .build()
            .map_err(|error| {
                MomoError::Llm(format!("Failed to create LLM HTTP client: {error}"))
            })?;

        // Configure async-openai's internal backoff to respect our timeout.
        // Without this, async-openai retries 500 errors with exponential backoff
        // for up to 15 minutes (the default max_elapsed_time), independent of
        // momo's own retry logic in complete()/complete_json().
        let backoff = backoff::ExponentialBackoff {
            max_elapsed_time: Some(Duration::from_secs(api_config.timeout_secs)),
            ..Default::default()
        };

        let client = Client::with_config(openai_config)
            .with_http_client(http_client)
            .with_backoff(backoff);

        Ok(Self {
            client,
            config: api_config,
        })
    }

    pub async fn complete(
        &self,
        prompt: &str,
        system_prompt: Option<&str>,
        options: Option<&CompletionOptions>,
    ) -> Result<String> {
        if prompt.trim().is_empty() {
            return Err(MomoError::Validation("Prompt cannot be empty".to_string()));
        }

        let mut last_error: Option<MomoError> = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay_ms = 100 * 2_u64.pow(attempt - 1);
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }

            let request = self.build_request(prompt, system_prompt, options)?;

            match self.client.chat().create(request).await {
                Ok(response) => return Self::extract_content(response),
                Err(error) => {
                    if let Some(rate_limit_error) = Self::rate_limit_error(&error) {
                        return Err(rate_limit_error);
                    }

                    if let Some(auth_error) = Self::auth_error(&error) {
                        return Err(auth_error);
                    }

                    let retryable = Self::is_retryable(&error);
                    let mapped_error = Self::map_openai_error(error);

                    if retryable && attempt < self.config.max_retries {
                        last_error = Some(mapped_error);
                        continue;
                    }

                    return Err(mapped_error);
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| MomoError::Llm("LLM completion failed after retries".to_string())))
    }

    pub async fn complete_json(
        &self,
        prompt: &str,
        options: Option<&CompletionOptions>,
    ) -> Result<Value> {
        if prompt.trim().is_empty() {
            return Err(MomoError::Validation("Prompt cannot be empty".to_string()));
        }

        let mut last_error: Option<MomoError> = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay_ms = 100 * 2_u64.pow(attempt - 1);
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }

            let request = self.build_json_request(prompt, options)?;

            match self.client.chat().create(request).await {
                Ok(response) => {
                    let content = Self::extract_content(response)?;
                    tracing::debug!(response_len = content.len(), "LLM JSON response received");
                    return serde_json::from_str(&content).map_err(|e| {
                        tracing::error!(response_len = content.len(), response_preview = %&content.chars().take(100).collect::<String>(), error = %e, "Failed to parse JSON response");
                        MomoError::Llm(format!("Failed to parse JSON response: {e}"))
                    });
                }
                Err(error) => {
                    if let Some(rate_limit_error) = Self::rate_limit_error(&error) {
                        return Err(rate_limit_error);
                    }

                    if let Some(auth_error) = Self::auth_error(&error) {
                        return Err(auth_error);
                    }

                    let retryable = Self::is_retryable(&error);
                    let mapped_error = Self::map_openai_error(error);

                    if retryable && attempt < self.config.max_retries {
                        last_error = Some(mapped_error);
                        continue;
                    }

                    return Err(mapped_error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            MomoError::Llm("LLM JSON completion failed after retries".to_string())
        }))
    }

    fn build_request(
        &self,
        prompt: &str,
        system_prompt: Option<&str>,
        options: Option<&CompletionOptions>,
    ) -> Result<CreateChatCompletionRequest> {
        let mut messages = Vec::new();

        if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt)
                    .build()
                    .map_err(|error| {
                        MomoError::Validation(format!("Invalid system prompt: {error}"))
                    })?
                    .into(),
            );
        }

        messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()
                .map_err(|error| MomoError::Validation(format!("Invalid user prompt: {error}")))?
                .into(),
        );

        let mut request = CreateChatCompletionRequestArgs::default();
        request.model(self.config.model.clone()).messages(messages);
        Self::apply_completion_options(&mut request, options);

        request.build().map_err(|error| {
            MomoError::Validation(format!("Invalid LLM completion request: {error}"))
        })
    }

    fn build_json_request(
        &self,
        prompt: &str,
        options: Option<&CompletionOptions>,
    ) -> Result<CreateChatCompletionRequest> {
        let messages = vec![ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()
            .map_err(|error| MomoError::Validation(format!("Invalid user prompt: {error}")))?
            .into()];

        let mut request = CreateChatCompletionRequestArgs::default();
        request.model(self.config.model.clone()).messages(messages);
        Self::apply_completion_options(&mut request, options);

        request
            .build()
            .map_err(|error| MomoError::Validation(format!("Invalid LLM JSON request: {error}")))
    }

    fn apply_completion_options(
        request: &mut CreateChatCompletionRequestArgs,
        options: Option<&CompletionOptions>,
    ) {
        let Some(options) = options else {
            return;
        };

        if let Some(temperature) = options.temperature {
            request.temperature(temperature);
        }

        if let Some(max_tokens) = options.max_tokens {
            request.max_tokens(max_tokens);
        }

        if let Some(top_p) = options.top_p {
            request.top_p(top_p);
        }

        if let Some(stop) = options.stop.as_ref().filter(|values| !values.is_empty()) {
            request.stop(Stop::StringArray(stop.clone()));
        }
    }

    fn extract_content(response: CreateChatCompletionResponse) -> Result<String> {
        let message = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| MomoError::Llm("LLM response contained no choices".to_string()))?
            .message
            .content
            .unwrap_or_default();

        if message.trim().is_empty() {
            return Err(MomoError::Llm(
                "LLM response contained empty content".to_string(),
            ));
        }

        Ok(message)
    }

    fn is_retryable(error: &OpenAIError) -> bool {
        match error {
            OpenAIError::ApiError(api_error) => {
                api_error.r#type.is_none() && api_error.code.is_none()
            }
            OpenAIError::Reqwest(reqwest_error) => reqwest_error
                .status()
                .map(|status| status.is_server_error())
                .unwrap_or(true),
            _ => false,
        }
    }

    fn rate_limit_error(error: &OpenAIError) -> Option<MomoError> {
        match error {
            OpenAIError::Reqwest(reqwest_error)
                if reqwest_error.status() == Some(reqwest::StatusCode::TOO_MANY_REQUESTS) =>
            {
                Some(MomoError::LlmRateLimit { retry_after: None })
            }
            OpenAIError::ApiError(api_error) if Self::is_rate_limit_api_error(api_error) => {
                Some(MomoError::LlmRateLimit { retry_after: None })
            }
            _ => None,
        }
    }

    fn auth_error(error: &OpenAIError) -> Option<MomoError> {
        match error {
            OpenAIError::Reqwest(reqwest_error)
                if reqwest_error.status() == Some(reqwest::StatusCode::UNAUTHORIZED)
                    || reqwest_error.status() == Some(reqwest::StatusCode::FORBIDDEN) =>
            {
                Some(MomoError::Llm(format!(
                    "LLM authentication failed: {reqwest_error}"
                )))
            }
            OpenAIError::ApiError(api_error) if Self::is_auth_api_error(api_error) => Some(
                MomoError::Llm(format!("LLM authentication failed: {api_error}")),
            ),
            _ => None,
        }
    }

    fn is_rate_limit_api_error(api_error: &ApiError) -> bool {
        let message = api_error.message.to_lowercase();
        let error_type = api_error.r#type.clone().unwrap_or_default().to_lowercase();
        let code = api_error.code.clone().unwrap_or_default().to_lowercase();

        message.contains("rate limit")
            || message.contains("too many requests")
            || error_type.contains("rate_limit")
            || code.contains("rate_limit")
            || code == "insufficient_quota"
    }

    fn is_auth_api_error(api_error: &ApiError) -> bool {
        let message = api_error.message.to_lowercase();
        let error_type = api_error.r#type.clone().unwrap_or_default().to_lowercase();
        let code = api_error.code.clone().unwrap_or_default().to_lowercase();

        message.contains("unauthorized")
            || message.contains("forbidden")
            || message.contains("authentication")
            || message.contains("invalid api key")
            || code.contains("invalid_api_key")
            || code.contains("authentication")
            || error_type.contains("authentication")
    }

    fn map_openai_error(error: OpenAIError) -> MomoError {
        match error {
            OpenAIError::Reqwest(reqwest_error) => {
                MomoError::Llm(format!("LLM request failed: {reqwest_error}"))
            }
            OpenAIError::ApiError(api_error) => {
                MomoError::Llm(format!("LLM API error: {api_error}"))
            }
            OpenAIError::JSONDeserialize(err) => {
                MomoError::Llm(format!("Failed to parse LLM response: {err}"))
            }
            OpenAIError::InvalidArgument(message) => MomoError::Validation(message),
            other => MomoError::Llm(other.to_string()),
        }
    }
}

impl ApiConfig {
    fn from_llm_config(config: &LlmConfig) -> Self {
        let (provider, model) = parse_llm_provider_model(&config.model);

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| default_base_url(provider).to_string());

        let normalized_model = if provider.eq_ignore_ascii_case("local") {
            config.model.clone()
        } else {
            model.to_string()
        };

        Self {
            base_url,
            api_key: config.api_key.clone(),
            model: normalized_model,
            timeout_secs: config.timeout_secs,
            max_retries: config.max_retries,
        }
    }
}

fn default_base_url(provider: &str) -> &'static str {
    match provider.to_lowercase().as_str() {
        "openai" => OPENAI_BASE_URL,
        "openrouter" => OPENROUTER_BASE_URL,
        "ollama" => OLLAMA_BASE_URL,
        "lmstudio" => "http://localhost:1234/v1",
        _ => OPENAI_BASE_URL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlmConfig;

    fn test_llm_config() -> LlmConfig {
        LlmConfig {
            model: "ollama/llama3".to_string(),
            api_key: None,
            base_url: None,
            timeout_secs: 30,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 0,
            query_rewrite_timeout_secs: 5,
            enable_auto_relations: false,
            enable_contradiction_detection: false,
            filter_prompt: None,
        }
    }

    #[test]
    fn test_llm_array_response_parsing() {
        let array_response = r#"[
            {"content": "User prefers dark mode", "memory_type": "preference", "confidence": 0.9},
            {"content": "User is a software engineer", "memory_type": "fact", "confidence": 0.85}
        ]"#;

        let parsed: serde_json::Result<Value> = serde_json::from_str(array_response);
        assert!(parsed.is_ok(), "Array JSON should parse successfully");

        let value = parsed.unwrap();
        assert!(value.is_array(), "Parsed value should be an array");
        assert_eq!(value.as_array().unwrap().len(), 2);
        assert_eq!(
            value[0]["content"].as_str().unwrap(),
            "User prefers dark mode"
        );
    }

    #[test]
    fn test_llm_object_response_parsing() {
        let object_response = r#"{
            "content": "User prefers low-light UI and modal-less editors",
            "reasoning": "Derived from preferences",
            "confidence": 0.87,
            "source_ids": ["mem_1", "mem_2"]
        }"#;

        let parsed: serde_json::Result<Value> = serde_json::from_str(object_response);
        assert!(parsed.is_ok(), "Object JSON should parse successfully");

        let value = parsed.unwrap();
        assert!(value.is_object(), "Parsed value should be an object");
        assert_eq!(
            value["content"].as_str().unwrap(),
            "User prefers low-light UI and modal-less editors"
        );
    }

    #[test]
    fn test_build_json_request_does_not_force_json_object_format() {
        let config = test_llm_config();
        let client = LlmApiClient::new(&config).expect("client should be created");

        let request = client
            .build_json_request("test prompt", None)
            .expect("request should build");

        assert!(
            request.response_format.is_none(),
            "build_json_request should NOT set response_format so array responses work"
        );
    }

    #[test]
    fn test_llm_empty_array_response_parsing() {
        let empty_array = "[]";
        let parsed: serde_json::Result<Value> = serde_json::from_str(empty_array);
        assert!(parsed.is_ok(), "Empty array should parse successfully");
        assert!(parsed.unwrap().is_array());
    }

    #[test]
    fn test_llm_nested_array_response_parsing() {
        let nested_response = r#"[
            {"memory_id": "mem_123", "relation_type": "updates", "confidence": 0.95, "reasoning": "Contradicts preference"},
            {"memory_id": "mem_456", "relation_type": "none", "confidence": 0.9, "reasoning": "Unrelated"}
        ]"#;

        let parsed: serde_json::Result<Value> = serde_json::from_str(nested_response);
        assert!(parsed.is_ok());

        let value = parsed.unwrap();
        assert!(value.is_array());
        assert_eq!(value[0]["relation_type"].as_str().unwrap(), "updates");
        assert_eq!(value[1]["relation_type"].as_str().unwrap(), "none");
    }
}
