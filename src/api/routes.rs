use axum::Router;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use super::v1;
use super::AppState;

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // legacy v3/v4/admin routers removed — only v1 remains mounted
    let v1 = v1::router::v1_router(state.clone());

    Router::new()
        .nest("/api/v1", v1)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
