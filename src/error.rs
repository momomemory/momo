use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MomoError {
    #[error("Database error: {0}")]
    Database(#[from] libsql::Error),

    #[error("Database backend error: {0}")]
    DatabaseBackend(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),

    #[error("Authentication required")]
    Unauthorized,

    #[error("Rate limit exceeded")]
    RateLimited,

    #[error("API rate limit exceeded, retry after {retry_after:?} seconds")]
    ApiRateLimit { retry_after: Option<u64> },

    #[error("API authentication error: {0}")]
    ApiAuth(String),

    #[error("Embedding dimension mismatch: database has {db_dimensions} dimensions, model produces {model_dimensions} dimensions")]
    DimensionMismatch {
        db_dimensions: usize,
        model_dimensions: usize,
    },

    #[error("Migration required: database has {db_dimensions} dimensions, model produces {model_dimensions} dimensions")]
    MigrationRequired {
        db_dimensions: usize,
        model_dimensions: usize,
    },

    #[error("Internal server error: {0}")]
    Internal(String),

    #[error("OCR error: {0}")]
    Ocr(String),

    #[error("OCR unavailable: {0}")]
    OcrUnavailable(String),

    #[error("Transcription error: {0}")]
    Transcription(String),

    #[error("Transcription unavailable: {0}")]
    TranscriptionUnavailable(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("LLM unavailable: {0}")]
    LlmUnavailable(String),

    #[error("LLM rate limit exceeded, retry after {retry_after:?} seconds")]
    LlmRateLimit { retry_after: Option<u64> },

    #[error("Reranker error: {0}")]
    Reranker(String),
}

impl IntoResponse for MomoError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            MomoError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            MomoError::Validation(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            MomoError::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized".to_string()),
            MomoError::RateLimited => (StatusCode::TOO_MANY_REQUESTS, "Rate limited".to_string()),
            MomoError::Database(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            MomoError::DatabaseBackend(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            MomoError::Embedding(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            MomoError::Processing(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            MomoError::Http(e) => (StatusCode::BAD_GATEWAY, e.to_string()),
            MomoError::Json(e) => (StatusCode::BAD_REQUEST, e.to_string()),
            MomoError::Io(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            MomoError::UrlParse(e) => (StatusCode::BAD_REQUEST, e.to_string()),
            MomoError::ApiRateLimit { .. } => (StatusCode::TOO_MANY_REQUESTS, self.to_string()),
            MomoError::ApiAuth(msg) => (StatusCode::UNAUTHORIZED, msg.clone()),
            MomoError::DimensionMismatch { .. } => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            MomoError::MigrationRequired { .. } => {
                (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
            }
            MomoError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            MomoError::Ocr(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            MomoError::OcrUnavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg.clone()),
            MomoError::Transcription(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            MomoError::TranscriptionUnavailable(msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, msg.clone())
            }
            MomoError::Llm(msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
            MomoError::LlmUnavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg.clone()),
            MomoError::LlmRateLimit { retry_after } => (
                StatusCode::TOO_MANY_REQUESTS,
                format!(
                    "LLM rate limit exceeded, retry after {:?} seconds",
                    retry_after
                ),
            ),
            MomoError::Reranker(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };

        let body = Json(json!({
            "error": message,
            "code": status.as_u16()
        }));

        (status, body).into_response()
    }
}

pub type Result<T> = std::result::Result<T, MomoError>;
