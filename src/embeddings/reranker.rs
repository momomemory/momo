use fastembed::{
    RerankInitOptions, RerankResult as FastEmbedRerankResult, RerankerModel, TextRerank,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::RerankerConfig;
use crate::error::{MomoError, Result};

/// Result from reranking operation
#[derive(Debug, Clone)]
pub struct RerankResult {
    pub document: String,
    pub score: f32,
    pub index: usize,
}

#[derive(Clone)]
enum RerankerBackend {
    Local(Arc<Mutex<TextRerank>>),
    #[allow(dead_code)]
    Mock(Arc<Mutex<Vec<RerankResult>>>),
}

/// Thread-safe reranker provider wrapping FastEmbed's TextRerank
#[derive(Clone)]
pub struct RerankerProvider {
    backend: Option<RerankerBackend>,
    batch_size: usize,
    loaded_model: String,
    domain_models: HashMap<String, String>,
}

impl From<FastEmbedRerankResult> for RerankResult {
    fn from(result: FastEmbedRerankResult) -> Self {
        Self {
            document: result.document.unwrap_or_default(),
            score: result.score,
            index: result.index,
        }
    }
}

impl RerankerProvider {
    pub async fn new_async(config: &RerankerConfig) -> Result<Self> {
        if !config.enabled {
            return Ok(Self {
                backend: None,
                batch_size: config.batch_size,
                loaded_model: config.model.clone(),
                domain_models: config.domain_models.clone(),
            });
        }

        let reranker_model = Self::parse_model(&config.model)?;

        let model = TextRerank::try_new(
            RerankInitOptions::new(reranker_model)
                .with_cache_dir(PathBuf::from(&config.cache_dir))
                .with_show_download_progress(true),
        )
        .map_err(|e| MomoError::Reranker(format!("Failed to initialize reranker: {}", e)))?;

        Ok(Self {
            backend: Some(RerankerBackend::Local(Arc::new(Mutex::new(model)))),
            batch_size: config.batch_size,
            loaded_model: config.model.clone(),
            domain_models: config.domain_models.clone(),
        })
    }

    pub fn new_mock(results: Vec<RerankResult>) -> Self {
        Self {
            backend: Some(RerankerBackend::Mock(Arc::new(Mutex::new(results)))),
            batch_size: 1,
            loaded_model: String::new(),
            domain_models: HashMap::new(),
        }
    }

    fn parse_model(model_name: &str) -> Result<RerankerModel> {
        match model_name {
            "bge-reranker-base" | "BAAI/bge-reranker-base" => Ok(RerankerModel::BGERerankerBase),
            "bge-reranker-v2-m3" | "rozgo/bge-reranker-v2-m3" => {
                Ok(RerankerModel::BGERerankerV2M3)
            }
            "jina-reranker-v1-turbo-en" | "jinaai/jina-reranker-v1-turbo-en" => {
                Ok(RerankerModel::JINARerankerV1TurboEn)
            }
            "jina-reranker-v2-base-multilingual"
            | "jinaai/jina-reranker-v2-base-multilingual" => {
                Ok(RerankerModel::JINARerankerV2BaseMultiligual)
            }
            _ => Err(MomoError::Reranker(format!(
                "Unsupported reranker model: {}. Supported models: bge-reranker-base, bge-reranker-v2-m3, jina-reranker-v1-turbo-en, jina-reranker-v2-base-multilingual",
                model_name
            ))),
        }
    }

    pub fn is_supported_model(model_name: &str) -> bool {
        Self::parse_model(model_name).is_ok()
    }

    pub fn is_enabled(&self) -> bool {
        self.backend.is_some()
    }

    /// Return the reranker model name configured for a given domain,
    /// falling back to the loaded (default) model if no override exists.
    pub fn model_for_domain(&self, domain: &str) -> &str {
        self.domain_models
            .get(domain)
            .map(|s| s.as_str())
            .unwrap_or(&self.loaded_model)
    }

    /// Rerank documents using the model appropriate for `domain`.
    ///
    /// If the domain has an explicit model override that matches the currently
    /// loaded model, or if no override is configured for the domain, this
    /// delegates directly to [`Self::rerank`].  When the domain maps to a
    /// *different* model than the one loaded in memory, a warning is logged
    /// and the currently loaded model is used (hot-swapping is not yet
    /// supported).
    pub async fn rerank_for_domain(
        &self,
        domain: &str,
        query: &str,
        documents: Vec<String>,
        top_k: usize,
    ) -> Result<Vec<RerankResult>> {
        let target_model = self.model_for_domain(domain);

        if target_model != self.loaded_model && !target_model.is_empty() {
            tracing::warn!(
                domain = %domain,
                target_model = %target_model,
                loaded_model = %self.loaded_model,
                "Domain model differs from loaded model; using loaded model \
                 (hot-swapping not yet supported)"
            );
        }

        self.rerank(query, documents, top_k).await
    }

    pub async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_k: usize,
    ) -> Result<Vec<RerankResult>> {
        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| MomoError::Reranker("Reranker is not enabled".to_string()))?;

        if documents.is_empty() {
            return Ok(Vec::new());
        }

        match backend {
            RerankerBackend::Local(model) => {
                let mut model = model.lock().await;
                let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
                let results = model
                    .rerank(query, &doc_refs, true, Some(self.batch_size))
                    .map_err(|e| MomoError::Reranker(format!("Reranking failed: {}", e)))?;

                Ok(results
                    .into_iter()
                    .take(top_k)
                    .map(RerankResult::from)
                    .collect())
            }
            RerankerBackend::Mock(results) => {
                let results = results.lock().await;
                Ok(results.clone().into_iter().take(top_k).collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_parse_model_bge_base() {
        let result = RerankerProvider::parse_model("bge-reranker-base");
        assert!(result.is_ok());

        let result = RerankerProvider::parse_model("BAAI/bge-reranker-base");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_model_bge_v2_m3() {
        let result = RerankerProvider::parse_model("bge-reranker-v2-m3");
        assert!(result.is_ok());

        let result = RerankerProvider::parse_model("rozgo/bge-reranker-v2-m3");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_model_jina_turbo() {
        let result = RerankerProvider::parse_model("jina-reranker-v1-turbo-en");
        assert!(result.is_ok());

        let result = RerankerProvider::parse_model("jinaai/jina-reranker-v1-turbo-en");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_model_jina_multilingual() {
        let result = RerankerProvider::parse_model("jina-reranker-v2-base-multilingual");
        assert!(result.is_ok());

        let result = RerankerProvider::parse_model("jinaai/jina-reranker-v2-base-multilingual");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_model_unsupported() {
        let result = RerankerProvider::parse_model("unknown-model");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported reranker model"));
    }

    #[tokio::test]
    async fn test_disabled_reranker() {
        let config = RerankerConfig {
            enabled: false,
            model: "bge-reranker-base".to_string(),
            top_k: 10,
            cache_dir: ".fastembed_cache".to_string(),
            batch_size: 64,
            domain_models: HashMap::new(),
        };

        let provider = RerankerProvider::new_async(&config).await.unwrap();
        assert!(!provider.is_enabled());
    }

    #[tokio::test]
    async fn test_rerank_disabled_error() {
        let config = RerankerConfig {
            enabled: false,
            model: "bge-reranker-base".to_string(),
            top_k: 10,
            cache_dir: ".fastembed_cache".to_string(),
            batch_size: 64,
            domain_models: HashMap::new(),
        };

        let provider = RerankerProvider::new_async(&config).await.unwrap();
        let result = provider
            .rerank("query", vec!["doc1".to_string(), "doc2".to_string()], 10)
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enabled"));
    }

    #[tokio::test]
    async fn test_rerank_empty_documents() {
        let config = RerankerConfig {
            enabled: false,
            model: "bge-reranker-base".to_string(),
            top_k: 10,
            cache_dir: ".fastembed_cache".to_string(),
            batch_size: 64,
            domain_models: HashMap::new(),
        };

        let provider = RerankerProvider::new_async(&config).await.unwrap();

        // Even with disabled reranker, empty docs should return empty results
        // But since we check enabled first, this will error
        let result = provider.rerank("query", vec![], 10).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_rerank_result_conversion() {
        let fastembed_result = FastEmbedRerankResult {
            document: Some("test document".to_string()),
            score: 0.95,
            index: 0,
        };

        let result: RerankResult = fastembed_result.into();
        assert_eq!(result.document, "test document");
        assert_eq!(result.score, 0.95);
        assert_eq!(result.index, 0);
    }

    #[test]
    fn test_rerank_result_conversion_no_document() {
        let fastembed_result = FastEmbedRerankResult {
            document: None,
            score: 0.85,
            index: 1,
        };

        let result: RerankResult = fastembed_result.into();
        assert_eq!(result.document, "");
        assert_eq!(result.score, 0.85);
        assert_eq!(result.index, 1);
    }

    #[test]
    fn test_model_for_domain_with_override() {
        let mut domain_models = HashMap::new();
        domain_models.insert("code".to_string(), "jina-reranker-v1-turbo-en".to_string());

        let provider = RerankerProvider {
            backend: None,
            batch_size: 64,
            loaded_model: "bge-reranker-base".to_string(),
            domain_models,
        };

        assert_eq!(provider.model_for_domain("code"), "jina-reranker-v1-turbo-en");
    }

    #[test]
    fn test_model_for_domain_falls_back_to_loaded() {
        let provider = RerankerProvider {
            backend: None,
            batch_size: 64,
            loaded_model: "bge-reranker-base".to_string(),
            domain_models: HashMap::new(),
        };

        assert_eq!(provider.model_for_domain("unknown"), "bge-reranker-base");
    }

    #[tokio::test]
    async fn test_rerank_for_domain_delegates_to_rerank() {
        let mock_results = vec![
            RerankResult { document: "doc1".to_string(), score: 0.9, index: 0 },
            RerankResult { document: "doc2".to_string(), score: 0.7, index: 1 },
        ];
        let provider = RerankerProvider::new_mock(mock_results);

        let results = provider
            .rerank_for_domain("code", "query", vec!["doc1".to_string(), "doc2".to_string()], 10)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document, "doc1");
    }

    #[tokio::test]
    async fn test_rerank_for_domain_with_mismatched_model() {
        let mut domain_models = HashMap::new();
        domain_models.insert("code".to_string(), "jina-reranker-v1-turbo-en".to_string());

        let mock_results = vec![
            RerankResult { document: "doc1".to_string(), score: 0.9, index: 0 },
        ];
        let provider = RerankerProvider {
            backend: Some(RerankerBackend::Mock(Arc::new(Mutex::new(mock_results)))),
            batch_size: 1,
            loaded_model: "bge-reranker-base".to_string(),
            domain_models,
        };

        let results = provider
            .rerank_for_domain("code", "query", vec!["doc1".to_string()], 10)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document, "doc1");
    }
}
