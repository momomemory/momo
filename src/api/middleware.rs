use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};

use super::AppState;

pub async fn auth_middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // When no API keys are configured, admin endpoints are locked down.
    // Operators must set MOMO_API_KEYS to access admin routes.
    if state.config.server.api_keys.is_empty() {
        return Err(StatusCode::UNAUTHORIZED);
    }

    let auth_header = request
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok());

    let token = match auth_header {
        Some(h) if h.starts_with("Bearer ") => &h[7..],
        _ => return Err(StatusCode::UNAUTHORIZED),
    };

    if state.config.server.api_keys.contains(&token.to_string()) {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::AppState;
    use crate::config::{
        Config, DatabaseConfig, EmbeddingsConfig, MemoryConfig, OcrConfig, ProcessingConfig,
        ServerConfig, TranscriptionConfig, InferenceConfig,
    };
    use axum::{middleware, routing::get, Router};
    use axum::http::Request;
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

        async fn handler() -> &'static str {
            "ok"
        }

        Router::new()
            .route("/admin/test", get(handler))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                auth_middleware,
            ))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_admin_auth_rejects_when_no_keys_configured() {
        let app = build_test_app(vec![]).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/admin/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_admin_auth_allows_with_valid_key() {
        let app = build_test_app(vec!["test-key".to_string()]).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/admin/test")
                    .header("Authorization", "Bearer test-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_admin_auth_rejects_invalid_key() {
        let app = build_test_app(vec!["test-key".to_string()]).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/admin/test")
                    .header("Authorization", "Bearer wrong-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
