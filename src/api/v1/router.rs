use axum::{
    middleware,
    routing::{get, post},
    Router,
};

use crate::api::state::AppState;

use super::handlers;
use super::middleware::v1_auth_middleware;

pub fn v1_router(state: AppState) -> Router<AppState> {
    let documents = Router::new()
        .route(
            "/",
            get(handlers::documents::list_documents).post(handlers::documents::create_document),
        )
        .route(
            "/{documentId}",
            get(handlers::documents::get_document)
                .patch(handlers::documents::update_document)
                .delete(handlers::documents::delete_document),
        );

    let ingestions = Router::new().route(
        "/{ingestionId}",
        get(handlers::documents::get_ingestion_status),
    );

    let memories = Router::new()
        .route(
            "/",
            get(handlers::memories::list_memories).post(handlers::memories::create_memory),
        )
        .route(
            "/{memoryId}",
            get(handlers::memories::get_memory)
                .patch(handlers::memories::update_memory)
                .delete(handlers::memories::delete_memory),
        )
        .route("/{memoryId}/graph", get(handlers::graph::get_memory_graph));
    let search = Router::new().route("/", post(handlers::search::search));
    let containers = Router::new().route("/{tag}/graph", get(handlers::graph::get_container_graph));
    let public_routes = Router::new()
        .route("/health", get(handlers::health_check))
        .route("/openapi.json", get(super::openapi::openapi_json))
        .merge(super::openapi::redoc_router());

    let protected_routes = Router::new()
        .nest("/documents", documents)
        .route(
            "/documents:batch",
            post(handlers::documents::batch_create_documents),
        )
        .route(
            "/documents:upload",
            post(handlers::documents::upload_document),
        )
        .route("/memories:forget", post(handlers::memories::forget_memory))
        .route("/profile:compute", post(handlers::profile::compute_profile))
        .route(
            "/conversations:ingest",
            post(handlers::conversation::ingest_conversation),
        )
        .route(
            "/admin/forgetting:run",
            post(handlers::admin::run_forgetting),
        )
        .nest("/ingestions", ingestions)
        .nest("/memories", memories)
        .nest("/search", search)
        .nest("/containers", containers)
        .route_layer(middleware::from_fn_with_state(state, v1_auth_middleware));

    Router::new().merge(public_routes).merge(protected_routes)
}
