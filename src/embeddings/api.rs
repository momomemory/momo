use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::error::{MomoError, Result};

/// Provider-specific default base URLs
pub fn default_base_url(provider: &str) -> &'static str {
    match provider.to_lowercase().as_str() {
        "openai" => "https://api.openai.com/v1",
        "openrouter" => "https://openrouter.ai/api/v1",
        "ollama" => "http://localhost:11434/v1",
        "lmstudio" => "http://localhost:1234/v1",
        _ => "https://api.openai.com/v1", // default fallback
    }
}

#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Clone)]
pub struct EmbeddingApiClient {
    client: Client,
    config: ApiConfig,
}

impl EmbeddingApiClient {
    pub fn new(config: ApiConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| MomoError::Embedding(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self { client, config })
    }

    pub async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let request = EmbeddingRequest {
            model: &self.config.model,
            input: texts.to_vec(),
        };

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(ref api_key) = self.config.api_key {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {api_key}"))
                    .map_err(|e| MomoError::Embedding(format!("Invalid API key header: {e}")))?,
            );
        }

        let url = format!("{}/embeddings", self.config.base_url);

        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(100 * 2_u64.pow(attempt - 1));
                tokio::time::sleep(delay).await;
            }

            let response = self
                .client
                .post(&url)
                .headers(headers.clone())
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();

                    if status.is_success() {
                        let body: EmbeddingResponse = resp.json().await.map_err(|e| {
                            MomoError::Embedding(format!("Failed to parse response: {e}"))
                        })?;
                        return Ok(body.data.into_iter().map(|d| d.embedding).collect());
                    }

                    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                        let retry_after = resp
                            .headers()
                            .get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.parse().ok());
                        last_error = Some(MomoError::ApiRateLimit { retry_after });
                        continue;
                    }

                    if status == reqwest::StatusCode::UNAUTHORIZED
                        || status == reqwest::StatusCode::FORBIDDEN
                    {
                        let body = resp.text().await.unwrap_or_default();
                        return Err(MomoError::ApiAuth(body));
                    }

                    if status.is_server_error() {
                        let body = resp.text().await.unwrap_or_default();
                        last_error = Some(MomoError::Embedding(format!(
                            "Server error {status}: {body}"
                        )));
                        continue;
                    }

                    let body = resp.text().await.unwrap_or_default();
                    return Err(MomoError::Embedding(format!(
                        "API error {status}: {body}"
                    )));
                }
                Err(e) => {
                    last_error = Some(MomoError::Embedding(format!("Request failed: {e}")));
                    continue;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| MomoError::Embedding("Unknown error".to_string())))
    }

    pub async fn detect_dimensions(&self) -> Result<usize> {
        let embeddings = self.embed(&["test"]).await?;
        embeddings
            .first()
            .map(|e| e.len())
            .ok_or_else(|| MomoError::Embedding("No embedding returned".to_string()))
    }
}
