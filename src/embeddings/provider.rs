use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::{parse_provider_model, EmbeddingsConfig};
use crate::embeddings::api::{default_base_url, ApiConfig, EmbeddingApiClient};
use crate::error::{MomoError, Result};

enum EmbeddingBackend {
    Local {
        model: Arc<Mutex<TextEmbedding>>,
        batch_size: usize,
    },
    Api {
        client: EmbeddingApiClient,
    },
}

pub struct EmbeddingProvider {
    backend: EmbeddingBackend,
    dimensions: usize,
}

impl EmbeddingProvider {
    /// Sync constructor for local models only.
    /// Returns an error if an API provider is specified.
    pub fn new(config: &EmbeddingsConfig) -> Result<Self> {
        let (provider, model_name) = parse_provider_model(&config.model);

        if provider != "local" {
            return Err(MomoError::Embedding(
                "Use new_async() for API providers".to_string(),
            ));
        }

        Self::new_local(config, model_name)
    }

    /// Async constructor that supports both local and API providers.
    pub async fn new_async(config: &EmbeddingsConfig) -> Result<Self> {
        let (provider, model_name) = parse_provider_model(&config.model);

        if provider == "local" {
            return Self::new_local(config, model_name);
        }

        // API provider
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| default_base_url(provider).to_string());

        let api_config = ApiConfig {
            base_url,
            api_key: config.api_key.clone(),
            model: model_name.to_string(),
            timeout_secs: config.timeout_secs,
            max_retries: config.max_retries,
        };

        let client = EmbeddingApiClient::new(api_config)?;
        let dimensions = client.detect_dimensions().await?;

        Ok(Self {
            backend: EmbeddingBackend::Api { client },
            dimensions,
        })
    }

    fn new_local(config: &EmbeddingsConfig, model_name: &str) -> Result<Self> {
        let embedding_model = match model_name {
            "BAAI/bge-small-en-v1.5" | "bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
            "BAAI/bge-base-en-v1.5" | "bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
            "BAAI/bge-large-en-v1.5" | "bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
            "all-MiniLM-L6-v2" | "sentence-transformers/all-MiniLM-L6-v2" => {
                EmbeddingModel::AllMiniLML6V2
            }
            "all-MiniLM-L12-v2" | "sentence-transformers/all-MiniLM-L12-v2" => {
                EmbeddingModel::AllMiniLML12V2
            }
            "nomic-embed-text-v1" | "nomic-ai/nomic-embed-text-v1" => {
                EmbeddingModel::NomicEmbedTextV1
            }
            "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" => {
                EmbeddingModel::NomicEmbedTextV15
            }
            _ => EmbeddingModel::BGESmallENV15,
        };

        let model = TextEmbedding::try_new(
            InitOptions::new(embedding_model).with_show_download_progress(true),
        )
        .map_err(|e| MomoError::Embedding(e.to_string()))?;

        Ok(Self {
            backend: EmbeddingBackend::Local {
                model: Arc::new(Mutex::new(model)),
                batch_size: config.batch_size,
            },
            dimensions: config.dimensions,
        })
    }

    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        match &self.backend {
            EmbeddingBackend::Local { model, batch_size } => {
                let mut model = model.lock().await;
                model
                    .embed(texts, Some(*batch_size))
                    .map_err(|e| MomoError::Embedding(e.to_string()))
            }
            EmbeddingBackend::Api { client } => {
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                client.embed(&refs).await
            }
        }
    }

    pub async fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed(vec![text.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| MomoError::Embedding("No embedding generated".to_string()))
    }

    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        match &self.backend {
            EmbeddingBackend::Local { .. } => {
                // Local models use query: prefix
                let prefixed = format!("query: {}", query);
                self.embed_single(&prefixed).await
            }
            EmbeddingBackend::Api { .. } => {
                // API providers don't use prefix
                self.embed_single(query).await
            }
        }
    }

    pub async fn embed_passage(&self, passage: &str) -> Result<Vec<f32>> {
        match &self.backend {
            EmbeddingBackend::Local { .. } => {
                // Local models use passage: prefix
                let prefixed = format!("passage: {}", passage);
                self.embed_single(&prefixed).await
            }
            EmbeddingBackend::Api { .. } => {
                // API providers don't use prefix
                self.embed_single(passage).await
            }
        }
    }

    pub async fn embed_passages(&self, passages: Vec<String>) -> Result<Vec<Vec<f32>>> {
        match &self.backend {
            EmbeddingBackend::Local { .. } => {
                let prefixed: Vec<String> = passages
                    .into_iter()
                    .map(|p| format!("passage: {}", p))
                    .collect();
                self.embed(prefixed).await
            }
            EmbeddingBackend::Api { .. } => {
                // API providers don't use prefix
                self.embed(passages).await
            }
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl Clone for EmbeddingProvider {
    fn clone(&self) -> Self {
        match &self.backend {
            EmbeddingBackend::Local { model, batch_size } => Self {
                backend: EmbeddingBackend::Local {
                    model: Arc::clone(model),
                    batch_size: *batch_size,
                },
                dimensions: self.dimensions,
            },
            EmbeddingBackend::Api { client } => Self {
                backend: EmbeddingBackend::Api {
                    client: client.clone(),
                },
                dimensions: self.dimensions,
            },
        }
    }
}
