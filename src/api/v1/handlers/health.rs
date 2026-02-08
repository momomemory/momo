use axum::extract::State;
use serde::Serialize;

use crate::api::state::AppState;
use crate::api::v1::response::ApiResponse;
use crate::llm::LlmBackend;

/// Health data returned inside the v1 envelope.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct HealthData {
    pub status: String,
    pub version: String,
    pub database: DatabaseStatus,
    pub embeddings: EmbeddingsStatus,
    pub llm: LlmStatus,
    pub reranker: RerankerStatus,
}

#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct DatabaseStatus {
    pub status: String,
}

#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct EmbeddingsStatus {
    pub status: String,
    pub model: String,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct LlmStatus {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct RerankerStatus {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub status: String,
}

/// `GET /api/v1/health`
#[utoipa::path(
    get,
    path = "/api/v1/health",
    tag = "health",
    responses(
        (status = 200, description = "Service health status", body = HealthData),
    )
)]
pub async fn health_check(State(state): State<AppState>) -> ApiResponse<HealthData> {
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

    let reranker_status = match &state.config.reranker {
        None => RerankerStatus {
            enabled: false,
            model: None,
            status: "disabled".to_string(),
        },
        Some(cfg) => match &state.reranker {
            None => RerankerStatus {
                enabled: false,
                model: Some(cfg.model.clone()),
                status: "error".to_string(),
            },
            Some(r) => {
                if r.is_enabled() {
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
    };

    ApiResponse::success(HealthData {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        database: db_status,
        embeddings: embeddings_status,
        llm: llm_status,
        reranker: reranker_status,
    })
}
