#!/usr/bin/env rust
// Integration tests for episode decay behavior using the HTTP handlers/router

#[cfg(test)]
mod integration {
    use crate::api::create_router;
    use crate::api::state::AppState;
    use crate::config::Config;
    use crate::db::Database;
    use crate::embeddings::EmbeddingProvider;
    use crate::llm::LlmProvider;
    use crate::ocr::OcrProvider;
    use crate::transcription::TranscriptionProvider;
    use crate::models::{Memory, MemoryType};
    use crate::services::EpisodeDecayManager;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::response::Response;
    use chrono::{Duration, Utc};
    use libsql::Connection;
    use serde_json::{json, Value};
    use std::sync::Arc;
    use tempfile::tempdir;
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn test_state() -> AppState {
        let mut config = Config::default();
        // in-memory DB for tests
        config.database.url = ":memory:".to_string();

        // create a mock embeddings server that returns a deterministic embedding
        let mock_server = MockServer::start().await;
        let mut embedding = vec![0.0f32; 384];
        embedding[0] = 1.0;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": embedding}]
            })))
            .mount(&mock_server)
            .await;

        config.embeddings.model = "openai/text-embedding-3-small".to_string();
        config.embeddings.dimensions = 384;
        config.embeddings.api_key = Some("test-key".to_string());
        config.embeddings.base_url = Some(mock_server.uri());

        let db = Database::new(&config.database).await.unwrap();
        let embeddings = EmbeddingProvider::new_async(&config.embeddings).await.unwrap();
        let ocr = OcrProvider::new(&config.ocr).unwrap();
        let transcription = TranscriptionProvider::new(&config.transcription).unwrap();
        let llm = LlmProvider::unavailable("tests");

        AppState::new(config, db, embeddings, None, ocr, transcription, llm)
    }

    async fn response_json(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        serde_json::from_slice(&bytes).expect("parse response json")
    }

    async fn post_search(app: axum::Router, path: &str, body: Value) -> Response {
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        app.oneshot(request).await.unwrap()
    }

    async fn insert_episode_memory(conn: &Connection, id: &str, last_accessed: Option<String>, is_static: bool) {
        let mut m = Memory::new(id.to_string(), format!("Memory {}", id), "space1".to_string());
        m.memory_type = MemoryType::Episode;
        m.last_accessed = last_accessed.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok()).map(|dt| dt.with_timezone(&Utc));

        crate::db::repository::memories::MemoryRepository::create(conn, &m).await.unwrap();

        // give it an embedding identical to query embedding so searches will find it
        let mut embedding = vec![0.0f32; 384];
        embedding[0] = 1.0;
        crate::db::repository::memories::MemoryRepository::update_embedding(conn, &m.id, &embedding).await.unwrap();

        if is_static {
            conn.execute("UPDATE memories SET is_static = 1 WHERE id = ?1", (id,)).await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_search_updates_last_accessed() {
        let state = test_state().await;
        let app = create_router(state.clone());

        let conn = state.db.connect().unwrap();

        // create an old episode memory
        let past = (Utc::now() - Duration::days(365)).to_rfc3339();
        insert_episode_memory(&conn, "ep1", Some(past.clone()), false).await;

        // ensure last_accessed is the past value
        let row = conn
            .query("SELECT last_accessed FROM memories WHERE id = 'ep1'", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();
        let before: Option<String> = row.get(0).unwrap();
        assert!(before.is_some());

        // perform a memories-only search which should update last_accessed
        let response = post_search(app, "/v4/search", json!({"q": "hello", "searchMode": "memories"})).await;
        let (status, _body) = (response.status(), response_json(response).await);
        assert_ne!(status, StatusCode::BAD_REQUEST);

        // check last_accessed updated to a recent timestamp
        let row_after = conn
            .query("SELECT last_accessed FROM memories WHERE id = 'ep1'", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();
        let after: Option<String> = row_after.get(0).unwrap();
        assert!(after.is_some());

        let before_dt = chrono::DateTime::parse_from_rfc3339(&before.unwrap()).unwrap().with_timezone(&Utc);
        let after_dt = chrono::DateTime::parse_from_rfc3339(&after.unwrap()).unwrap().with_timezone(&Utc);

        assert!(after_dt > before_dt);
    }

    #[tokio::test]
    async fn test_repeated_search_delays_decay() {
        let state = test_state().await;
        let app = create_router(state.clone());

        let conn = state.db.connect().unwrap();

        // create an old episode memory
        let past = (Utc::now() - Duration::days(365)).to_rfc3339();
        insert_episode_memory(&conn, "ep2", Some(past.clone()), false).await;

        // run multiple searches to bump last_accessed several times
        for _ in 0..3 {
            let response = post_search(app.clone(), "/v4/search", json!({"q": "hello", "searchMode": "memories"})).await;
            assert_ne!(response.status(), StatusCode::BAD_REQUEST);
        }

        // run episode decay manager - it should NOT schedule the recently-accessed episode
        let mgr = EpisodeDecayManager::new(state.db.clone(), 0.9, 7, 30.0, 0.9);
        let scheduled = mgr.run_once().await.unwrap();
        assert_eq!(scheduled, 0);
    }

    #[tokio::test]
    async fn test_static_episodes_not_decayed() {
        let state = test_state().await;
        let app = create_router(state.clone());

        let conn = state.db.connect().unwrap();

        // create an old episode memory but mark it static
        let past = (Utc::now() - Duration::days(365)).to_rfc3339();
        insert_episode_memory(&conn, "ep3", Some(past.clone()), true).await;

        // run a search to exercise code paths (should not alter staticness)
        let response = post_search(app, "/v4/search", json!({"q": "hello", "searchMode": "memories"})).await;
        assert_ne!(response.status(), StatusCode::BAD_REQUEST);

        // run episode decay manager - static episodes must be excluded from scheduling
        let mgr = EpisodeDecayManager::new(state.db.clone(), 0.5, 7, 30.0, 0.9);
        let scheduled = mgr.run_once().await.unwrap();

        // scheduled may be 0; ensure ep3 did not get forget_after set
        let row = conn
            .query("SELECT forget_after FROM memories WHERE id = 'ep3'", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();
        let f: Option<String> = row.get(0).unwrap();
        assert!(f.is_none());

        // also ensure we didn't schedule it (manager should not report >0 because static excluded)
        assert_eq!(scheduled, 0);
    }
}
