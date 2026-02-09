pub mod dto;
pub mod handlers;
pub mod middleware;
pub mod openapi;
pub mod response;
pub mod router;

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use crate::api::routes::create_router;
    use crate::api::state::AppState;
    use crate::config::{
        Config, DatabaseConfig, EmbeddingsConfig, InferenceConfig, MemoryConfig, OcrConfig,
        ProcessingConfig, ServerConfig, TranscriptionConfig,
    };

    async fn test_state(api_keys: Vec<String>) -> AppState {
        let config = Config {
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
            },
            processing: ProcessingConfig {
                chunk_size: 512,
                chunk_overlap: 50,
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
        };

        let raw_db = crate::db::Database::new(&config.database).await.unwrap();
        let db_backend = crate::db::LibSqlBackend::new(raw_db);
        let db: std::sync::Arc<dyn crate::db::DatabaseBackend> = std::sync::Arc::new(db_backend);

        let embeddings = crate::embeddings::EmbeddingProvider::new(&config.embeddings).unwrap();
        let ocr = crate::ocr::OcrProvider::new(&config.ocr).unwrap();
        let transcription =
            crate::transcription::TranscriptionProvider::new(&config.transcription).unwrap();
        let llm = crate::llm::LlmProvider::new(config.llm.as_ref());

        AppState::new(config, db, embeddings, None, ocr, transcription, llm)
    }

    async fn body_json(response: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn protected_route_requires_auth() {
        let app = create_router(test_state(vec!["test-key".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"q":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let json = body_json(response).await;
        assert_eq!(json["error"]["code"], "unauthorized");
    }

    #[tokio::test]
    async fn health_is_public() {
        let app = create_router(test_state(vec!["secret".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn openapi_json_is_public_and_valid() {
        let app = create_router(test_state(vec!["secret".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        let version = json["openapi"]
            .as_str()
            .expect("openapi field should be a string");
        assert!(
            version.starts_with("3"),
            "OpenAPI version should start with 3, got: {version}"
        );
    }

    #[tokio::test]
    async fn success_envelope_has_data_no_error() {
        let app = create_router(test_state(vec!["k".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert!(json.get("data").is_some(), "success should have 'data' key");
        assert!(
            json.get("error").is_none(),
            "success should NOT have 'error' key"
        );
    }

    #[tokio::test]
    async fn error_envelope_has_error_no_data() {
        let app = create_router(test_state(vec!["key".to_string()]).await);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"q":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let json = body_json(response).await;
        assert!(
            json.get("error").is_some(),
            "error response should have 'error' key"
        );
        assert!(
            json.get("data").is_none(),
            "error response should NOT have 'data' key"
        );
        assert!(
            json["error"]["code"].is_string(),
            "error.code should be a string"
        );
        assert!(
            json["error"]["message"].is_string(),
            "error.message should be a string"
        );
    }
}
