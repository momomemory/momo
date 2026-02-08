//! # V1 API Key Authentication Middleware
//!
//! Protects all v1 API routes (except explicitly public ones like `/health`)
//! with Bearer token authentication. Validates the token against the
//! `MOMO_API_KEYS` configuration.
//!
//! Unlike the admin middleware (`src/api/middleware.rs`) which returns raw
//! `StatusCode`, this middleware returns the v1 `ApiResponse` JSON envelope
//! so auth errors conform to the v1 contract.

use axum::{
    body::Body,
    extract::State,
    http::Request,
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::api::state::AppState;

use super::response::{ApiResponse, ErrorCode};

/// Axum middleware that enforces Bearer token authentication for v1 API routes.
///
/// # Behavior
///
/// - If `MOMO_API_KEYS` is empty/unset → returns 401 with JSON error envelope.
///   The server still starts, but protected routes are locked down.
/// - If the `Authorization: Bearer <token>` header is missing or malformed → 401.
/// - If the token is not in the configured key list → 401.
/// - If the token is valid → passes the request through to the next handler.
///
/// # Error format
///
/// All errors are returned as `ApiResponse<()>` JSON envelopes:
/// ```json
/// { "error": { "code": "unauthorized", "message": "..." } }
/// ```
pub async fn v1_auth_middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if state.config.server.api_keys.is_empty() {
        return ApiResponse::<()>::error(
            ErrorCode::Unauthorized,
            "API keys not configured. Set MOMO_API_KEYS to enable access.",
        )
        .into_response();
    }

    let auth_header = request
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok());

    let token = match auth_header {
        Some(h) if h.starts_with("Bearer ") => &h[7..],
        Some(_) => {
            return ApiResponse::<()>::error(
                ErrorCode::Unauthorized,
                "Invalid authorization header format. Expected: Bearer <token>",
            )
            .into_response();
        }
        None => {
            return ApiResponse::<()>::error(
                ErrorCode::Unauthorized,
                "Missing authorization header",
            )
            .into_response();
        }
    };

    if state.config.server.api_keys.contains(&token.to_string()) {
        next.run(request).await
    } else {
        ApiResponse::<()>::error(ErrorCode::Unauthorized, "Invalid API key").into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::state::AppState;
    use crate::config::{
        Config, DatabaseConfig, EmbeddingsConfig, InferenceConfig, MemoryConfig, OcrConfig,
        ProcessingConfig, ServerConfig, TranscriptionConfig,
    };
    use axum::http::StatusCode;
    use axum::{middleware, routing::get, Router};
    use axum::http::Request;
    use axum::body::Body;
    use tower::ServiceExt;

    fn make_config(api_keys: Vec<String>) -> Config {
        Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 3000,
                api_keys,
            },
            database: DatabaseConfig {
                url: "file::memory:".to_string(),
                auth_token: None,
                local_path: None,
            },
            embeddings: EmbeddingsConfig {
                model: "BAAI/bge-small-en-v1.5".to_string(),
                dimensions: 384,
                batch_size: 256,
                api_key: None,
                base_url: None,
                rate_limit: None,
                timeout_secs: 30,
                max_retries: 3,
            },
            processing: ProcessingConfig {
                chunk_size: 512,
                chunk_overlap: 50,
                max_content_length: 10_000_000,
            },
            memory: MemoryConfig {
                episode_decay_days: 30.0,
                episode_decay_factor: 0.9,
                episode_decay_threshold: 0.3,
                episode_forget_grace_days: 7,
                forgetting_check_interval_secs: 3600,
                profile_refresh_interval_secs: 86400,
                inference: InferenceConfig {
                    enabled: false,
                    interval_secs: 86400,
                    confidence_threshold: 0.7,
                    max_per_run: 50,
                    candidate_count: 5,
                    seed_limit: 50,
                    exclude_episodes: true,
                },
            },
            ocr: OcrConfig {
                model: "local/tesseract".to_string(),
                api_key: None,
                base_url: None,
                languages: "eng".to_string(),
                timeout_secs: 60,
                max_image_dimension: 4096,
                min_image_dimension: 50,
            },
            transcription: TranscriptionConfig::default(),
            llm: None,
            reranker: None,
        }
    }

    async fn build_test_app(api_keys: Vec<String>) -> Router {
        let config = make_config(api_keys);

        let raw_db =
            crate::db::Database::new(&config.database).await.unwrap();
        let db_backend = crate::db::LibSqlBackend::new(raw_db);
        let db: std::sync::Arc<dyn crate::db::DatabaseBackend> =
            std::sync::Arc::new(db_backend);

        let embeddings =
            crate::embeddings::EmbeddingProvider::new(&config.embeddings).unwrap();
        let ocr = crate::ocr::OcrProvider::new(&config.ocr).unwrap();
        let transcription =
            crate::transcription::TranscriptionProvider::new(&config.transcription).unwrap();
        let llm = crate::llm::LlmProvider::new(config.llm.as_ref());

        let state = AppState::new(config, db, embeddings, None, ocr, transcription, llm);

        async fn protected_handler() -> &'static str {
            "protected"
        }

        async fn health_handler() -> &'static str {
            "healthy"
        }

        let public_routes = Router::new().route("/health", get(health_handler));

        let protected_routes = Router::new()
            .route("/protected", get(protected_handler))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                v1_auth_middleware,
            ));

        Router::new()
            .merge(public_routes)
            .merge(protected_routes)
            .with_state(state)
    }

    /// Parses JSON error envelope from response body.
    async fn parse_error_body(response: Response) -> (StatusCode, serde_json::Value) {
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        (status, json)
    }

    #[tokio::test]
    async fn test_v1_auth_rejects_when_no_keys_configured() {
        let app = build_test_app(vec![]).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let (status, json) = parse_error_body(response).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(json["error"]["code"], "unauthorized");
        assert!(json["error"]["message"].as_str().unwrap().contains("API keys not configured"));
        assert!(json.get("data").is_none());
    }

    #[tokio::test]
    async fn test_v1_auth_allows_with_valid_key() {
        let app = build_test_app(vec!["test-key-v1".to_string()]).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .header("Authorization", "Bearer test-key-v1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_v1_auth_rejects_invalid_key() {
        let app = build_test_app(vec!["test-key-v1".to_string()]).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .header("Authorization", "Bearer wrong-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let (status, json) = parse_error_body(response).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(json["error"]["code"], "unauthorized");
        assert_eq!(json["error"]["message"], "Invalid API key");
    }

    #[tokio::test]
    async fn test_v1_auth_rejects_missing_header() {
        let app = build_test_app(vec!["test-key-v1".to_string()]).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let (status, json) = parse_error_body(response).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(json["error"]["code"], "unauthorized");
        assert_eq!(json["error"]["message"], "Missing authorization header");
    }

    #[tokio::test]
    async fn test_v1_health_bypasses_auth() {
        let app = build_test_app(vec![]).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_v1_health_accessible_without_key_when_keys_configured() {
        let app = build_test_app(vec!["secret-key".to_string()]).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_v1_auth_error_response_is_json_envelope() {
        let app = build_test_app(vec!["key".to_string()]).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .header("Authorization", "Bearer bad")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let content_type = response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(content_type.contains("application/json"));

        let (status, json) = parse_error_body(response).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(json.get("data").is_none());
        assert!(json.get("meta").is_none());
        assert!(json.get("error").is_some());
        assert_eq!(json["error"]["code"], "unauthorized");
        assert!(json["error"]["message"].is_string());
    }
}
