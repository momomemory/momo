use fastembed::{
    RerankInitOptions, RerankResult as FastEmbedRerankResult, RerankerModel, TextRerank,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::RerankerConfig;
use crate::error::{MomoError, Result};

/// Result from reranking operation
#[derive(Debug, Clone)]
pub struct RerankResult {
    #[allow(dead_code)]
    pub document: String,
    pub score: f32,
    pub index: usize,
}

#[derive(Clone)]
enum RerankerBackend {
    Local(Arc<Mutex<TextRerank>>),
    #[allow(dead_code)]
    Mock(Arc<Vec<RerankResult>>),
}

/// Thread-safe reranker provider wrapping FastEmbed's TextRerank
#[derive(Clone)]
pub struct RerankerProvider {
    backend: Option<RerankerBackend>,
    batch_size: usize,
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
            });
        }

        let reranker_model = Self::parse_model(&config.model)?;

        let model = TextRerank::try_new(
            RerankInitOptions::new(reranker_model)
                .with_cache_dir(PathBuf::from(&config.cache_dir))
                .with_show_download_progress(true),
        )
        .map_err(|e| MomoError::Reranker(format!("Failed to initialize reranker: {e}")))?;

        Ok(Self {
            backend: Some(RerankerBackend::Local(Arc::new(Mutex::new(model)))),
            batch_size: config.batch_size,
        })
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
                "Unsupported reranker model: {model_name}. Supported models: bge-reranker-base, bge-reranker-v2-m3, jina-reranker-v1-turbo-en, jina-reranker-v2-base-multilingual"
            ))),
        }
    }

    #[allow(dead_code)]
    pub fn is_supported_model(model_name: &str) -> bool {
        Self::parse_model(model_name).is_ok()
    }

    pub fn is_enabled(&self) -> bool {
        self.backend.is_some()
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
                    .map_err(|e| MomoError::Reranker(format!("Reranking failed: {e}")))?;

                Ok(results
                    .into_iter()
                    .take(top_k)
                    .map(RerankResult::from)
                    .collect())
            }
            RerankerBackend::Mock(results) => Ok(results.iter().take(top_k).cloned().collect()),
        }
    }

    #[allow(dead_code)]
    pub fn new_mock(results: Vec<RerankResult>) -> Self {
        Self {
            backend: Some(RerankerBackend::Mock(Arc::new(results))),
            batch_size: 64,
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
}
