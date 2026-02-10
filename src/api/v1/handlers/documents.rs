//! v1 Document handlers.
//!
//! Implements async document ingestion, CRUD, listing, batch ingestion,
//! file upload, and ingestion status polling. All responses are wrapped
//! in [`ApiResponse`] envelopes.

use axum::extract::{Multipart, Path, State};
use axum_extra::extract::Query;
use base64::Engine;
use chrono::Utc;
use nanoid::nanoid;

use crate::api::v1::dto::{
    BatchCreateDocumentRequest, BatchCreateDocumentResponse, CreateDocumentRequest,
    CreateDocumentResponse, DocumentResponse, DocumentSummaryResponse, IngestionStatusResponse,
    ListDocumentsQuery, ListDocumentsResponse, UpdateDocumentRequest,
};
use crate::api::v1::response::{ApiError, ApiResponse, ErrorCode, ResponseMeta};
use crate::api::AppState;
use crate::models::{Document, DocumentType, ProcessingStatus};
use crate::processing::ContentExtractor;

fn parse_form_bool(value: &str) -> Option<bool> {
    match value.trim().to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// `POST /api/v1/documents`
///
/// Creates a new document and queues it for async ingestion.
/// Returns 202 Accepted with `documentId` and `ingestionId`.
#[utoipa::path(
    post,
    path = "/api/v1/documents",
    tag = "documents",
    operation_id = "documents.create",
    request_body = CreateDocumentRequest,
    responses(
        (status = 202, description = "Document accepted for processing", body = CreateDocumentResponse),
        (status = 400, description = "Invalid request", body = ApiError),
    )
)]
pub async fn create_document(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<CreateDocumentRequest>,
) -> ApiResponse<CreateDocumentResponse> {
    // Validate content
    if req.content.trim().is_empty() {
        return ApiResponse::error(ErrorCode::InvalidRequest, "Content cannot be empty");
    }

    // Validate container_tag length
    if let Some(ref tag) = req.container_tag {
        if tag.len() > 255 {
            return ApiResponse::error(
                ErrorCode::InvalidRequest,
                "Container tag too long (max 255 characters)",
            );
        }
    }

    let id = nanoid!();
    let now = Utc::now();

    let mut container_tags = Vec::new();
    if let Some(ref tag) = req.container_tag {
        container_tags.push(tag.clone());
    }

    // Determine doc_type based on content_type hint
    let doc_type = if let Some(ref content_type) = req.content_type {
        let ct_lower = content_type.to_lowercase();
        match ct_lower.as_str() {
            "docx" => DocumentType::Docx,
            "xlsx" => DocumentType::Xlsx,
            "pptx" => DocumentType::Pptx,
            "csv" => DocumentType::Csv,
            "pdf" => DocumentType::Pdf,
            ct if ct.starts_with("image/")
                || ct == "image"
                || ct == "png"
                || ct == "jpg"
                || ct == "jpeg"
                || ct == "webp"
                || ct == "tiff"
                || ct == "bmp" =>
            {
                DocumentType::Image
            }
            ct if ct.starts_with("audio/")
                || ct == "audio"
                || ct == "mp3"
                || ct == "wav"
                || ct == "m4a"
                || ct == "ogg"
                || ct == "flac" =>
            {
                DocumentType::Audio
            }
            ct if ct.starts_with("video/")
                || ct == "video"
                || ct == "mp4"
                || ct == "webm"
                || ct == "avi"
                || ct == "mkv"
                || ct == "mov" =>
            {
                DocumentType::Video
            }
            _ => DocumentType::Text,
        }
    } else {
        DocumentType::Text
    };

    let mut metadata = req.metadata.unwrap_or_default();
    // v1 default: don't extract memories unless explicitly requested
    let extract_memories = req.extract_memories.unwrap_or(false);
    metadata.insert(
        "extract_memories".to_string(),
        serde_json::json!(extract_memories),
    );

    let doc = Document {
        id: id.clone(),
        custom_id: req.custom_id,
        connection_id: None,
        title: None,
        content: Some(req.content),
        summary: None,
        url: None,
        source: None,
        doc_type,
        status: ProcessingStatus::Queued,
        metadata,
        container_tags,
        chunk_count: 0,
        token_count: None,
        word_count: None,
        error_message: None,
        created_at: now,
        updated_at: now,
    };

    if let Err(e) = state.db.create_document(&doc).await {
        let resp: ApiResponse<CreateDocumentResponse> = e.into();
        return resp;
    }

    // Fire-and-forget background processing
    let pipeline = state.pipeline.clone();
    let doc_id = id.clone();
    tokio::spawn(async move {
        if let Err(e) = pipeline.process_document(&doc_id).await {
            tracing::error!(doc_id = %doc_id, error = %e, "Failed to process document");
        }
    });

    // ingestionId = documentId (no separate ingestion table)
    ApiResponse::accepted(CreateDocumentResponse {
        document_id: id.clone(),
        ingestion_id: id,
    })
}

const MAX_BATCH_SIZE: usize = 600;
const MAX_FILE_SIZE: usize = 25 * 1024 * 1024; // 25 MB

/// `POST /api/v1/documents:batch`
///
/// Creates multiple documents in a single request and queues them for
/// async ingestion. Returns 202 Accepted with an array of
/// `{ documentId, ingestionId }` pairs.
#[utoipa::path(
    post,
    path = "/api/v1/documents:batch",
    tag = "documents",
    operation_id = "documents.batch",
    request_body = BatchCreateDocumentRequest,
    responses(
        (status = 202, description = "Batch accepted for processing", body = BatchCreateDocumentResponse),
        (status = 400, description = "Invalid request", body = ApiError),
    )
)]
pub async fn batch_create_documents(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<BatchCreateDocumentRequest>,
) -> ApiResponse<BatchCreateDocumentResponse> {
    if req.documents.is_empty() {
        return ApiResponse::error(ErrorCode::InvalidRequest, "Documents array cannot be empty");
    }

    if req.documents.len() > MAX_BATCH_SIZE {
        return ApiResponse::error(
            ErrorCode::InvalidRequest,
            format!("Batch size exceeds maximum of {MAX_BATCH_SIZE} documents"),
        );
    }

    if let Some(ref tag) = req.container_tag {
        if tag.len() > 255 {
            return ApiResponse::error(
                ErrorCode::InvalidRequest,
                "Container tag too long (max 255 characters)",
            );
        }
    }

    let now = Utc::now();
    let mut results = Vec::with_capacity(req.documents.len());
    let mut doc_ids = Vec::with_capacity(req.documents.len());

    for item in &req.documents {
        if item.content.trim().is_empty() {
            return ApiResponse::error(ErrorCode::InvalidRequest, "Content cannot be empty");
        }

        let id = nanoid!();

        let mut container_tags = Vec::new();
        if let Some(ref tag) = req.container_tag {
            container_tags.push(tag.clone());
        }

        let mut metadata = req.metadata.clone().unwrap_or_default();
        if let Some(ref item_metadata) = item.metadata {
            metadata.extend(item_metadata.clone());
        }
        let extract_memories = item.extract_memories.unwrap_or(false);
        metadata.insert(
            "extract_memories".to_string(),
            serde_json::json!(extract_memories),
        );

        let doc = Document {
            id: id.clone(),
            custom_id: item.custom_id.clone(),
            connection_id: None,
            title: None,
            content: Some(item.content.clone()),
            summary: None,
            url: None,
            source: None,
            doc_type: DocumentType::Text,
            status: ProcessingStatus::Queued,
            metadata,
            container_tags,
            chunk_count: 0,
            token_count: None,
            word_count: None,
            error_message: None,
            created_at: now,
            updated_at: now,
        };

        if let Err(e) = state.db.create_document(&doc).await {
            let resp: ApiResponse<BatchCreateDocumentResponse> = e.into();
            return resp;
        }

        doc_ids.push(id.clone());
        results.push(CreateDocumentResponse {
            document_id: id.clone(),
            ingestion_id: id,
        });
    }

    // Fire-and-forget background processing for all documents
    let pipeline = state.pipeline.clone();
    tokio::spawn(async move {
        for doc_id in doc_ids {
            if let Err(e) = pipeline.process_document(&doc_id).await {
                tracing::error!(doc_id = %doc_id, error = %e, "Failed to process document");
            }
        }
    });

    ApiResponse::accepted(BatchCreateDocumentResponse { documents: results })
}

/// `POST /api/v1/documents:upload`
///
/// Accepts a multipart form with a `file` field and optional `containerTag`.
/// Creates a document from the uploaded file and queues it for async ingestion.
/// Returns 202 Accepted with `{ documentId, ingestionId }`.
#[utoipa::path(
    post,
    path = "/api/v1/documents:upload",
    tag = "documents",
    operation_id = "documents.upload",
    request_body(content_type = "multipart/form-data", content = String, description = "File upload with optional containerTag and metadata fields"),
    responses(
        (status = 202, description = "Upload accepted for processing", body = CreateDocumentResponse),
        (status = 400, description = "Invalid request", body = ApiError),
    )
)]
pub async fn upload_document(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> ApiResponse<CreateDocumentResponse> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut file_content_type: Option<String> = None;
    let mut container_tag: Option<String> = None;
    let mut metadata: Option<std::collections::HashMap<String, serde_json::Value>> = None;
    let mut extract_memories: Option<bool> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                if let Some(name) = field.file_name() {
                    file_name = Some(name.to_string());
                }
                if let Some(content_type) = field.content_type() {
                    file_content_type = Some(content_type.to_string());
                }

                let bytes = match field.bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        return ApiResponse::error(
                            ErrorCode::InvalidRequest,
                            format!("Failed to read file: {e}"),
                        );
                    }
                };

                if bytes.len() > MAX_FILE_SIZE {
                    return ApiResponse::error(
                        ErrorCode::InvalidRequest,
                        format!(
                            "File too large: {} bytes (max {} bytes)",
                            bytes.len(),
                            MAX_FILE_SIZE
                        ),
                    );
                }

                file_bytes = Some(bytes.to_vec());
            }
            "containerTag" | "container_tag" => {
                container_tag = match field.text().await {
                    Ok(t) => Some(t),
                    Err(e) => {
                        return ApiResponse::error(
                            ErrorCode::InvalidRequest,
                            format!("Invalid container tag: {e}"),
                        );
                    }
                };
            }
            "metadata" => {
                let json_str = match field.text().await {
                    Ok(t) => t,
                    Err(e) => {
                        return ApiResponse::error(
                            ErrorCode::InvalidRequest,
                            format!("Invalid metadata: {e}"),
                        );
                    }
                };
                metadata = serde_json::from_str(&json_str).ok();
            }
            "extractMemories" | "extract_memories" => {
                let raw = match field.text().await {
                    Ok(t) => t,
                    Err(e) => {
                        return ApiResponse::error(
                            ErrorCode::InvalidRequest,
                            format!("Invalid extractMemories value: {e}"),
                        );
                    }
                };
                match parse_form_bool(&raw) {
                    Some(value) => extract_memories = Some(value),
                    None => {
                        return ApiResponse::error(
                            ErrorCode::InvalidRequest,
                            "extractMemories must be one of true/false/1/0/yes/no",
                        );
                    }
                }
            }
            _ => {}
        }
    }

    let bytes = match file_bytes {
        Some(b) => b,
        None => {
            return ApiResponse::error(ErrorCode::InvalidRequest, "Missing required 'file' field");
        }
    };

    let doc_type = ContentExtractor::detect_type_from_upload(
        &bytes,
        file_name.as_deref(),
        file_content_type.as_deref(),
    );
    if matches!(doc_type, DocumentType::Unknown) {
        return ApiResponse::error(ErrorCode::InvalidRequest, "Unsupported file type");
    }

    let id = nanoid!();
    let now = Utc::now();

    let mut container_tags = Vec::new();
    if let Some(ref tag) = container_tag {
        if tag.len() > 255 {
            return ApiResponse::error(
                ErrorCode::InvalidRequest,
                "Container tag too long (max 255 characters)",
            );
        }
        container_tags.push(tag.clone());
    }

    let content_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

    let mut doc_metadata = metadata.unwrap_or_default();
    doc_metadata.insert(
        "extract_memories".to_string(),
        serde_json::json!(extract_memories.unwrap_or(false)),
    );

    let doc = Document {
        id: id.clone(),
        custom_id: None,
        connection_id: None,
        title: None,
        content: Some(content_b64),
        summary: None,
        url: None,
        source: None,
        doc_type,
        status: ProcessingStatus::Queued,
        metadata: doc_metadata,
        container_tags,
        chunk_count: 0,
        token_count: None,
        word_count: None,
        error_message: None,
        created_at: now,
        updated_at: now,
    };

    if let Err(e) = state.db.create_document(&doc).await {
        let resp: ApiResponse<CreateDocumentResponse> = e.into();
        return resp;
    }

    // Fire-and-forget background processing
    let pipeline = state.pipeline.clone();
    let doc_id = id.clone();
    tokio::spawn(async move {
        if let Err(e) = pipeline.process_document(&doc_id).await {
            tracing::error!(doc_id = %doc_id, error = %e, "Failed to process document");
        }
    });

    ApiResponse::accepted(CreateDocumentResponse {
        document_id: id.clone(),
        ingestion_id: id,
    })
}

/// `GET /api/v1/documents/{documentId}`
///
/// Retrieves a full document by ID. Also checks custom_id as fallback.
#[utoipa::path(
    get,
    path = "/api/v1/documents/{documentId}",
    tag = "documents",
    operation_id = "documents.get",
    params(("documentId" = String, Path, description = "Document ID or custom ID")),
    responses(
        (status = 200, description = "Document found", body = DocumentResponse),
        (status = 404, description = "Document not found", body = ApiError),
    )
)]
pub async fn get_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResponse<DocumentResponse> {
    match state.db.get_document_by_id(&id).await {
        Ok(Some(doc)) => return ApiResponse::success(doc.into()),
        Ok(None) => {}
        Err(e) => return e.into(),
    }

    // Fallback: try custom_id
    match state.db.get_document_by_custom_id(&id).await {
        Ok(Some(doc)) => ApiResponse::success(doc.into()),
        Ok(None) => ApiResponse::error(ErrorCode::NotFound, format!("Document {id} not found")),
        Err(e) => e.into(),
    }
}

/// `PATCH /api/v1/documents/{documentId}`
///
/// Updates document metadata, title, or container tags.
#[utoipa::path(
    patch,
    path = "/api/v1/documents/{documentId}",
    tag = "documents",
    operation_id = "documents.update",
    params(("documentId" = String, Path, description = "Document ID")),
    request_body = UpdateDocumentRequest,
    responses(
        (status = 200, description = "Document updated", body = DocumentResponse),
        (status = 404, description = "Document not found", body = ApiError),
    )
)]
pub async fn update_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
    axum::Json(req): axum::Json<UpdateDocumentRequest>,
) -> ApiResponse<DocumentResponse> {
    let mut doc = match state.db.get_document_by_id(&id).await {
        Ok(Some(doc)) => doc,
        Ok(None) => {
            return ApiResponse::error(ErrorCode::NotFound, format!("Document {id} not found"))
        }
        Err(e) => return e.into(),
    };

    if let Some(title) = req.title {
        doc.title = Some(title);
    }
    if let Some(metadata) = req.metadata {
        doc.metadata = metadata;
    }
    if let Some(tags) = req.container_tags {
        doc.container_tags = tags;
    }
    doc.updated_at = Utc::now();

    if let Err(e) = state.db.update_document(&doc).await {
        return e.into();
    }

    ApiResponse::success(doc.into())
}

/// `DELETE /api/v1/documents/{documentId}`
///
/// Deletes a document by ID. Also tries custom_id as fallback.
#[utoipa::path(
    delete,
    path = "/api/v1/documents/{documentId}",
    tag = "documents",
    operation_id = "documents.delete",
    params(("documentId" = String, Path, description = "Document ID or custom ID")),
    responses(
        (status = 200, description = "Document deleted", body = Object),
        (status = 404, description = "Document not found", body = ApiError),
    )
)]
pub async fn delete_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResponse<serde_json::Value> {
    match state.db.delete_document(&id).await {
        Ok(true) => {
            return ApiResponse::success(serde_json::json!({ "deleted": true }));
        }
        Ok(false) => {}
        Err(e) => return e.into(),
    }

    // Fallback: try custom_id
    match state.db.delete_document_by_custom_id(&id).await {
        Ok(true) => ApiResponse::success(serde_json::json!({ "deleted": true })),
        Ok(false) => ApiResponse::error(ErrorCode::NotFound, format!("Document {id} not found")),
        Err(e) => e.into(),
    }
}

/// `GET /api/v1/documents`
///
/// Lists documents with cursor-based pagination. Supports filtering by
/// `containerTags` query parameter.
#[utoipa::path(
    get,
    path = "/api/v1/documents",
    tag = "documents",
    operation_id = "documents.list",
    params(ListDocumentsQuery),
    responses(
        (status = 200, description = "Documents listed", body = ListDocumentsResponse),
    )
)]
pub async fn list_documents(
    State(state): State<AppState>,
    Query(query): Query<ListDocumentsQuery>,
) -> ApiResponse<ListDocumentsResponse> {
    let limit = query.limit.unwrap_or(20).clamp(1, 100);

    // Convert v1 query to internal ListDocumentsRequest (page-based internally)
    // We use cursor as a page number encoded as a string for simplicity.
    let page = query
        .cursor
        .as_ref()
        .and_then(|c| c.parse::<u32>().ok())
        .unwrap_or(1);

    let internal_req = crate::models::ListDocumentsRequest {
        container_tags: query.container_tags,
        filters: None,
        limit: Some(limit),
        page: Some(page),
        order: None,
        sort: None,
    };

    let (documents, pagination) = match state.db.list_documents(&internal_req).await {
        Ok(result) => result,
        Err(e) => return e.into(),
    };

    let docs: Vec<DocumentSummaryResponse> = documents.into_iter().map(Into::into).collect();

    // Build cursor: if there are more pages, encode next page number
    let next_cursor = if pagination.current_page < pagination.total_pages {
        Some((pagination.current_page + 1).to_string())
    } else {
        None
    };

    let meta = ResponseMeta {
        next_cursor,
        total: Some(pagination.total_items as u64),
    };

    ApiResponse::success_with_meta(ListDocumentsResponse { documents: docs }, meta)
}

/// `GET /api/v1/ingestions/{ingestionId}`
///
/// Polls the ingestion status for a document. Since `ingestionId` maps 1:1
/// to `documentId`, this is effectively a status-only document lookup.
#[utoipa::path(
    get,
    path = "/api/v1/ingestions/{ingestionId}",
    tag = "documents",
    operation_id = "documents.getIngestionStatus",
    params(("ingestionId" = String, Path, description = "Ingestion ID (same as document ID)")),
    responses(
        (status = 200, description = "Ingestion status", body = IngestionStatusResponse),
        (status = 404, description = "Ingestion not found", body = ApiError),
    )
)]
pub async fn get_ingestion_status(
    State(state): State<AppState>,
    Path(ingestion_id): Path<String>,
) -> ApiResponse<IngestionStatusResponse> {
    match state.db.get_document_by_id(&ingestion_id).await {
        Ok(Some(doc)) => {
            let resp = IngestionStatusResponse {
                document_id: doc.id,
                status: doc.status.into(),
                title: doc.title,
                created_at: doc.created_at,
            };
            ApiResponse::success(resp)
        }
        Ok(None) => ApiResponse::error(
            ErrorCode::NotFound,
            format!("Ingestion {ingestion_id} not found"),
        ),
        Err(e) => e.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::v1::dto::common::IngestionStatus;

    #[test]
    fn parse_form_bool_accepts_supported_values() {
        assert_eq!(parse_form_bool("true"), Some(true));
        assert_eq!(parse_form_bool("1"), Some(true));
        assert_eq!(parse_form_bool("yes"), Some(true));
        assert_eq!(parse_form_bool("on"), Some(true));
        assert_eq!(parse_form_bool("false"), Some(false));
        assert_eq!(parse_form_bool("0"), Some(false));
        assert_eq!(parse_form_bool("no"), Some(false));
        assert_eq!(parse_form_bool("off"), Some(false));
    }

    #[test]
    fn parse_form_bool_rejects_unknown_values() {
        assert_eq!(parse_form_bool("maybe"), None);
        assert_eq!(parse_form_bool(""), None);
    }

    #[test]
    fn create_document_response_serializes_correctly() {
        let resp = CreateDocumentResponse {
            document_id: "doc_123".to_string(),
            ingestion_id: "doc_123".to_string(),
        };
        let json = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(json["documentId"], "doc_123");
        assert_eq!(json["ingestionId"], "doc_123");
    }

    #[test]
    fn ingestion_status_response_serializes() {
        let resp = IngestionStatusResponse {
            document_id: "doc_1".to_string(),
            status: IngestionStatus::Processing,
            title: Some("My Doc".to_string()),
            created_at: Utc::now(),
        };
        let json = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(json["status"], "processing");
        assert_eq!(json["documentId"], "doc_1");
    }
}
