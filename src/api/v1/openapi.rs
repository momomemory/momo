use axum::Json;
use utoipa::OpenApi;
use utoipa_redoc::{Redoc, Servable};

use super::dto;
use super::handlers;
use super::response;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Momo API",
        version = "1.0.0",
        description = "Self-hostable AI memory system. REST API for AI memory management.",
    ),
    paths(
        handlers::health::health_check,
        handlers::documents::create_document,
        handlers::documents::batch_create_documents,
        handlers::documents::upload_document,
        handlers::documents::get_document,
        handlers::documents::update_document,
        handlers::documents::delete_document,
        handlers::documents::list_documents,
        handlers::documents::get_ingestion_status,
        handlers::search::search,
        handlers::memories::create_memory,
        handlers::memories::get_memory,
        handlers::memories::update_memory,
        handlers::memories::delete_memory,
        handlers::memories::list_memories,
        handlers::memories::forget_memory,
        handlers::graph::get_memory_graph,
        handlers::graph::get_container_graph,
        handlers::graph::list_container_tags,
        handlers::admin::run_forgetting,
        handlers::profile::compute_profile,
        handlers::conversation::ingest_conversation,
    ),
    components(schemas(
        // Response envelope
        response::ErrorCode,
        response::ApiError,
        response::ResponseMeta,
        response::CursorPagination,
        // Common
        dto::common::IngestionStatus,
        dto::common::V1DocumentType,
        dto::common::V1MemoryType,
        // Documents
        dto::documents::CreateDocumentRequest,
        dto::documents::BatchCreateDocumentRequest,
        dto::documents::BatchDocumentItem,
        dto::documents::UpdateDocumentRequest,
        dto::documents::ListDocumentsQuery,
        dto::documents::CreateDocumentResponse,
        dto::documents::BatchCreateDocumentResponse,
        dto::documents::DocumentResponse,
        dto::documents::DocumentSummaryResponse,
        dto::documents::ListDocumentsResponse,
        dto::documents::IngestionStatusResponse,
        // Memories
        dto::memories::CreateMemoryRequest,
        dto::memories::UpdateMemoryRequest,
        dto::memories::ForgetMemoryRequest,
        dto::memories::ContentForgetRequest,
        dto::memories::ListMemoriesQuery,
        dto::memories::MemoryResponse,
        dto::memories::UpdateMemoryResponse,
        dto::memories::ForgetMemoryResponse,
        dto::memories::ListMemoriesResponse,
        // Search
        dto::search::SearchScope,
        dto::search::SearchIncludeFlags,
        dto::search::SearchRequest,
        dto::search::SearchResponse,
        dto::search::SearchResultItem,
        dto::search::DocumentSearchResult,
        dto::search::ChunkResult,
        dto::search::MemorySearchResult,
        dto::search::HybridSearchResultResponse,
        // Profile
        dto::profile::ComputeProfileRequest,
        dto::profile::ProfileResponse,
        dto::profile::ProfileFactResponse,
        // Conversation
        dto::conversation::ConversationIngestRequest,
        dto::conversation::ConversationMessageDto,
        dto::conversation::ConversationIngestResponse,
        // Graph
        dto::graph::GraphNodeType,
        dto::graph::GraphEdgeType,
        dto::graph::GraphNodeResponse,
        dto::graph::GraphEdgeResponse,
        dto::graph::GraphResponse,
        dto::graph::ContainerTagsResponse,
        // Admin
        dto::admin::ForgettingRunResponse,
        // Health (handler-local types)
        handlers::health::HealthData,
        handlers::health::DatabaseStatus,
        handlers::health::EmbeddingsStatus,
        handlers::health::LlmStatus,
        handlers::health::RerankerStatus,
    )),
    tags(
        (name = "health", description = "Health check"),
        (name = "documents", description = "Document ingestion, CRUD, and listing"),
        (name = "search", description = "Unified search across documents and memories"),
        (name = "memories", description = "Memory CRUD, listing, and forgetting"),
        (name = "graph", description = "Knowledge graph exploration"),
        (name = "profile", description = "User profile computation"),
        (name = "conversation", description = "Conversation ingestion and memory extraction"),
        (name = "admin", description = "Administrative operations (auth required)"),
    ),
    security(
        ("bearer_auth" = [])
    ),
    modifiers(&SecurityAddon),
)]
pub struct ApiDoc;

struct SecurityAddon;

impl utoipa::Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi.components.get_or_insert_with(Default::default);
        components.add_security_scheme(
            "bearer_auth",
            utoipa::openapi::security::SecurityScheme::Http(utoipa::openapi::security::Http::new(
                utoipa::openapi::security::HttpAuthScheme::Bearer,
            )),
        );
    }
}

pub async fn openapi_json() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}

pub fn redoc_router<S: Clone + Send + Sync + 'static>() -> axum::Router<S> {
    Redoc::with_url("/docs", ApiDoc::openapi()).into()
}
