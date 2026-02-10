use axum::http::StatusCode;
use axum::routing::{any, get};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::mcp;

use super::frontend;
use super::v1;
use super::AppState;

async fn api_not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // legacy v3/v4/admin routers removed — only v1 remains mounted
    let v1 = v1::router::v1_router(state.clone());
    let mcp = mcp::mcp_router(state.clone());

    Router::new()
        .merge(mcp)
        .nest("/api/v1", v1)
        .route("/api", any(api_not_found))
        .route("/api/{*path}", any(api_not_found))
        .route("/", get(frontend::serve_root))
        .route("/{*path}", get(frontend::serve_path))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
