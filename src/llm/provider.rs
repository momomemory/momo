use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::config::{parse_llm_provider_model, LlmConfig};
use crate::error::{MomoError, Result};
use crate::llm::api::LlmApiClient;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmBackend {
    OpenAI,
    OpenRouter,
    Ollama,
    LmStudio,
    OpenAICompatible { base_url: String },
    Unavailable { reason: String },
}

#[derive(Debug, Clone, Default)]
pub struct CompletionOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct LlmProvider {
    backend: LlmBackend,
    config: Option<Arc<LlmConfig>>,
}

impl LlmProvider {
    pub fn new(config: Option<&LlmConfig>) -> Self {
        let Some(config) = config else {
            return Self::unavailable("No LLM configuration provided");
        };

        let (provider, _model) = parse_llm_provider_model(&config.model);

        let backend = match provider.to_lowercase().as_str() {
            "openai" => LlmBackend::OpenAI,
            "openrouter" => LlmBackend::OpenRouter,
            "ollama" => LlmBackend::Ollama,
            "lmstudio" => LlmBackend::LmStudio,
            _ => {
                if let Some(base_url) = &config.base_url {
                    LlmBackend::OpenAICompatible {
                        base_url: base_url.clone(),
                    }
                } else {
                    LlmBackend::Unavailable {
                        reason: format!("Unknown provider in model: {}", config.model),
                    }
                }
            }
        };

        Self {
            backend,
            config: Some(Arc::new(config.clone())),
        }
    }

    pub fn unavailable(reason: &str) -> Self {
        Self {
            backend: LlmBackend::Unavailable {
                reason: reason.to_string(),
            },
            config: None,
        }
    }

    pub fn is_available(&self) -> bool {
        !matches!(self.backend, LlmBackend::Unavailable { .. })
    }

    pub fn backend(&self) -> &LlmBackend {
        &self.backend
    }

    pub fn config(&self) -> Option<&LlmConfig> {
        self.config.as_deref()
    }

    pub async fn complete(
        &self,
        prompt: &str,
        options: Option<&CompletionOptions>,
    ) -> Result<String> {
        if !self.is_available() {
            return Err(MomoError::LlmUnavailable(self.unavailable_reason()));
        }

        let config = self
            .config()
            .ok_or_else(|| MomoError::LlmUnavailable("No config available".to_string()))?;

        let client = LlmApiClient::new(config)?;
        client.complete(prompt, None, options).await
    }

    pub async fn complete_json(
        &self,
        prompt: &str,
        options: Option<&CompletionOptions>,
    ) -> Result<Value> {
        if !self.is_available() {
            return Err(MomoError::LlmUnavailable(self.unavailable_reason()));
        }

        let config = self
            .config()
            .ok_or_else(|| MomoError::LlmUnavailable("No config available".to_string()))?;

        let client = LlmApiClient::new(config)?;
        client.complete_json(prompt, options).await
    }

    pub async fn complete_structured<T: DeserializeOwned>(&self, prompt: &str) -> Result<T> {
        let json_value = self.complete_json(prompt, None).await?;

        serde_json::from_value(json_value)
            .map_err(|e| MomoError::Llm(format!("Failed to deserialize response: {e}")))
    }

    fn unavailable_reason(&self) -> String {
        match &self.backend {
            LlmBackend::Unavailable { reason } => reason.clone(),
            _ => "LLM completion is not implemented yet".to_string(),
        }
    }
}
