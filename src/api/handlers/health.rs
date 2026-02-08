use axum::{extract::State, Json};
use serde::Serialize;

use crate::api::state::AppState;
use crate::config::RerankerConfig;
use crate::embeddings::RerankerProvider as RealRerankerProvider;
use crate::llm::LlmBackend;

#[derive(Serialize)]
pub struct LlmStatus {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Serialize)]
pub struct DatabaseStatus {
    pub status: String,
}

#[derive(Serialize)]
pub struct EmbeddingsStatus {
    pub status: String,
    pub model: String,
    pub dimensions: usize,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub database: DatabaseStatus,
    pub embeddings: EmbeddingsStatus,
    pub llm: LlmStatus,
    pub reranker: RerankerStatus,
}

pub async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let llm_status = if state.llm.is_available() {
        let provider = match state.llm.backend() {
            LlmBackend::OpenAI => "openai",
            LlmBackend::OpenRouter => "openrouter",
            LlmBackend::Ollama => "ollama",
            LlmBackend::LmStudio => "lmstudio",
            LlmBackend::OpenAICompatible { .. } => "openai-compatible",
            LlmBackend::Unavailable { .. } => "unavailable",
        };

        let model = state.llm.config().map(|c| c.model.clone());

        LlmStatus {
            status: "available".to_string(),
            provider: Some(provider.to_string()),
            model,
        }
    } else {
        LlmStatus {
            status: "unavailable".to_string(),
            provider: None,
            model: None,
        }
    };

    let db_status = match state.db.sync().await {
        Ok(_) => DatabaseStatus {
            status: "ok".to_string(),
        },
        Err(_) => DatabaseStatus {
            status: "error".to_string(),
        },
    };

    let embeddings_status = EmbeddingsStatus {
        status: "ok".to_string(),
        model: state.config.embeddings.model.clone(),
        dimensions: state.embeddings.dimensions(),
    };

    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        database: db_status,
        embeddings: embeddings_status,
        llm: llm_status,
        reranker: compute_reranker_status(&state.reranker, &state.config.reranker),
    })
}

#[derive(Serialize)]
pub struct RerankerStatus {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub status: String,
}

// Separated helper so tests can exercise logic without running the full handler
pub trait RerankerChecker {
    fn is_enabled(&self) -> bool;
}

impl RerankerChecker for RealRerankerProvider {
    fn is_enabled(&self) -> bool {
        // call the existing method
        self.is_enabled()
    }
}

pub fn compute_reranker_status<T: RerankerChecker>(
    provider: &Option<T>,
    cfg: &Option<RerankerConfig>,
) -> RerankerStatus {
    match cfg {
        None => RerankerStatus {
            enabled: false,
            model: None,
            status: "disabled".to_string(),
        },
        Some(cfg) => match provider {
            None => RerankerStatus {
                enabled: false,
                model: Some(cfg.model.clone()),
                status: "error".to_string(),
            },
            Some(p) => {
                if p.is_enabled() {
                    RerankerStatus {
                        enabled: true,
                        model: Some(cfg.model.clone()),
                        status: "ready".to_string(),
                    }
                } else {
                    RerankerStatus {
                        enabled: false,
                        model: Some(cfg.model.clone()),
                        status: "disabled".to_string(),
                    }
                }
            }
        },
    }
}

// Provide a trivial impl for unit type so tests can call compute_reranker_status::<()> when provider is None
impl RerankerChecker for () {
    fn is_enabled(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RerankerConfig;

    #[test]
    fn test_reranker_disabled_when_not_configured() {
        // Use a dummy type for provider since cfg is None
        let status = compute_reranker_status::<()>(&None, &None);
        assert!(!status.enabled);
        assert_eq!(status.model, None);
        assert_eq!(status.status, "disabled");
    }

    #[test]
    fn test_reranker_configured_but_not_loaded_reports_error() {
        let cfg = Some(RerankerConfig {
            enabled: true,
            model: "bge-reranker-base".to_string(),
            top_k: 10,
            cache_dir: ".fastembed_cache".to_string(),
            batch_size: 64,
            domain_models: std::collections::HashMap::new(),
        });

        let status = compute_reranker_status::<()>(&None, &cfg);
        assert!(!status.enabled);
        assert_eq!(status.model, Some("bge-reranker-base".to_string()));
        assert_eq!(status.status, "error");
    }

    struct MockReranker {
        enabled: bool,
    }
    impl super::RerankerChecker for MockReranker {
        fn is_enabled(&self) -> bool {
            self.enabled
        }
    }

    #[test]
    fn test_reranker_provider_ready() {
        let cfg = Some(RerankerConfig::default());
        let provider = Some(MockReranker { enabled: true });
        let status = compute_reranker_status(&provider, &cfg);
        assert!(status.enabled);
        assert_eq!(status.model, Some(RerankerConfig::default().model));
        assert_eq!(status.status, "ready");
    }

    #[test]
    fn test_reranker_provider_present_but_disabled() {
        let cfg = Some(RerankerConfig::default());
        let provider = Some(MockReranker { enabled: false });
        let status = compute_reranker_status(&provider, &cfg);
        assert!(!status.enabled);
        assert_eq!(status.model, Some(RerankerConfig::default().model));
        assert_eq!(status.status, "disabled");
    }

    #[test]
    fn test_database_status_serializes_ok() {
        let status = DatabaseStatus {
            status: "ok".to_string(),
        };
        let json = serde_json::to_value(&status).expect("serialize");
        assert_eq!(json["status"], "ok");
    }

    #[test]
    fn test_database_status_serializes_error() {
        let status = DatabaseStatus {
            status: "error".to_string(),
        };
        let json = serde_json::to_value(&status).expect("serialize");
        assert_eq!(json["status"], "error");
    }

    #[test]
    fn test_embeddings_status_serializes_all_fields() {
        let status = EmbeddingsStatus {
            status: "ok".to_string(),
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
        };
        let json = serde_json::to_value(&status).expect("serialize");
        assert_eq!(json["status"], "ok");
        assert_eq!(json["model"], "BAAI/bge-small-en-v1.5");
        assert_eq!(json["dimensions"], 384);
    }

    #[test]
    fn test_health_response_includes_database_and_embeddings() {
        let response = HealthResponse {
            status: "ok".to_string(),
            version: "0.1.0".to_string(),
            database: DatabaseStatus {
                status: "ok".to_string(),
            },
            embeddings: EmbeddingsStatus {
                status: "ok".to_string(),
                model: "all-MiniLM-L6-v2".to_string(),
                dimensions: 384,
            },
            llm: LlmStatus {
                status: "unavailable".to_string(),
                provider: None,
                model: None,
            },
            reranker: RerankerStatus {
                enabled: false,
                model: None,
                status: "disabled".to_string(),
            },
        };
        let json = serde_json::to_value(&response).expect("serialize");
        assert_eq!(json["database"]["status"], "ok");
        assert_eq!(json["embeddings"]["status"], "ok");
        assert_eq!(json["embeddings"]["model"], "all-MiniLM-L6-v2");
        assert_eq!(json["embeddings"]["dimensions"], 384);
        assert_eq!(json["llm"]["status"], "unavailable");
        assert_eq!(json["reranker"]["status"], "disabled");
    }
}
