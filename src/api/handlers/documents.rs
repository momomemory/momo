use axum::{
    extract::{Multipart, Path, State},
    Json,
};
use base64::Engine;
use chrono::Utc;
use nanoid::nanoid;

use crate::api::{AppJson, AppState};
use crate::error::{MomoError, Result};
use crate::models::{
    BatchCreateDocumentRequest, BatchCreateDocumentResponse, CreateDocumentRequest,
    CreateDocumentResponse, Document, DocumentType, ListDocumentsRequest, ListDocumentsResponse,
    Metadata, ProcessingDocument, ProcessingStatus, UpdateDocumentRequest,
};
use crate::processing::ContentExtractor;

const MAX_FILE_SIZE: usize = 25 * 1024 * 1024; // 25 MB

pub async fn create_document(
    State(state): State<AppState>,
    AppJson(req): AppJson<CreateDocumentRequest>,
) -> Result<Json<CreateDocumentResponse>> {
    // Validate content
    if req.content.trim().is_empty() {
        return Err(MomoError::Validation("Content cannot be empty".to_string()));
    }

    // Validate container_tag length
    if let Some(ref tag) = req.container_tag {
        if tag.len() > 255 {
            return Err(MomoError::Validation(
                "Container tag too long (max 255 characters)".to_string(),
            ));
        }
    }

    if let Some(ref prompt) = req.filter_prompt {
        if prompt.len() > 1000 {
            return Err(MomoError::Validation(
                "Filter prompt too long (max 1000 characters)".to_string(),
            ));
        }
    }

    if req.should_llm_filter == Some(false) && req.filter_prompt.is_some() {
        return Err(MomoError::Validation(
            "Cannot set filter_prompt when should_llm_filter is false".to_string(),
        ));
    }

    let id = nanoid!();
    let now = Utc::now();

    let mut container_tags = Vec::new();
    if let Some(ref tag) = req.container_tag {
        container_tags.push(tag.clone());
    }

    // Determine doc_type based on content_type hint
    let doc_type = if let Some(ref content_type) = req.content_type {
        // User specified content type - treat content as base64
        let ct_lower = content_type.to_lowercase();
        match ct_lower.as_str() {
            "docx" => DocumentType::Docx,
            "xlsx" => DocumentType::Xlsx,
            "pptx" => DocumentType::Pptx,
            "csv" => DocumentType::Csv,
            "pdf" => DocumentType::Pdf,
            // Handle image/* content types
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
            _ => DocumentType::Text,
        }
    } else {
        // No content type - use auto-detection (existing behavior)
        DocumentType::Text
    };

    let mut metadata = req.metadata.unwrap_or_default();
    // Default to true for memory extraction
    // Users can opt-out with extract_memories: false
    let extract_memories = req.extract_memories.unwrap_or(true);
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

    state.db.create_document(&doc).await?;

    if let Some(ref tag) = req.container_tag {
        if req.should_llm_filter.is_some() || req.filter_prompt.is_some() {
            use crate::models::ContainerFilter;
            let filter = ContainerFilter {
                tag: tag.clone(),
                should_llm_filter: req.should_llm_filter.unwrap_or(false),
                filter_prompt: req.filter_prompt.clone(),
            };
            state.db.set_container_filter(tag, &filter).await?;
        }
    }

    let pipeline = state.pipeline.clone();
    let doc_id = id.clone();
    tokio::spawn(async move {
        if let Err(e) = pipeline.process_document(&doc_id).await {
            tracing::error!("Failed to process document {}: {}", doc_id, e);
        }
    });

    Ok(Json(CreateDocumentResponse {
        id,
        status: ProcessingStatus::Queued,
    }))
}

pub async fn batch_create_documents(
    State(state): State<AppState>,
    AppJson(req): AppJson<BatchCreateDocumentRequest>,
) -> Result<Json<BatchCreateDocumentResponse>> {
    // Validate shared container_tag length
    if let Some(ref tag) = req.container_tag {
        if tag.len() > 255 {
            return Err(MomoError::Validation(
                "Container tag too long (max 255 characters)".to_string(),
            ));
        }
    }

    let now = Utc::now();

    let mut responses = Vec::new();
    let mut doc_ids = Vec::new();

    for item in req.documents {
        // Validate each item's content
        if item.content.trim().is_empty() {
            return Err(MomoError::Validation("Content cannot be empty".to_string()));
        }
        let id = nanoid!();

        let mut container_tags = Vec::new();
        if let Some(ref tag) = req.container_tag {
            container_tags.push(tag.clone());
        }

        let mut metadata = req.metadata.clone().unwrap_or_default();
        if let Some(item_metadata) = item.metadata {
            metadata.extend(item_metadata);
        }
        let extract_memories = item.extract_memories.unwrap_or(true);
        metadata.insert(
            "extract_memories".to_string(),
            serde_json::json!(extract_memories),
        );

        let doc = Document {
            id: id.clone(),
            custom_id: item.custom_id,
            connection_id: None,
            title: None,
            content: Some(item.content),
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

        state.db.create_document(&doc).await?;
        doc_ids.push(id.clone());

        responses.push(CreateDocumentResponse {
            id,
            status: ProcessingStatus::Queued,
        });
    }

    let pipeline = state.pipeline.clone();
    tokio::spawn(async move {
        for doc_id in doc_ids {
            if let Err(e) = pipeline.process_document(&doc_id).await {
                tracing::error!("Failed to process document {}: {}", doc_id, e);
            }
        }
    });

    Ok(Json(BatchCreateDocumentResponse {
        documents: responses,
    }))
}

pub async fn get_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Document>> {
    let doc = state.db.get_document_by_id(&id)
        .await?
        .or_else(|| None);

    if doc.is_none() {
        if let Some(doc) = state.db.get_document_by_custom_id(&id).await? {
            return Ok(Json(doc));
        }
    }

    doc.map(Json)
        .ok_or_else(|| MomoError::NotFound(format!("Document {} not found", id)))
}

pub async fn update_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
    AppJson(req): AppJson<UpdateDocumentRequest>,
) -> Result<Json<Document>> {
    let mut doc = state.db.get_document_by_id(&id)
        .await?
        .ok_or_else(|| MomoError::NotFound(format!("Document {} not found", id)))?;

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

    state.db.update_document(&doc).await?;

    Ok(Json(doc))
}

pub async fn delete_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>> {
    let deleted = state.db.delete_document(&id).await?;

    if !deleted {
        let deleted = state.db.delete_document_by_custom_id(&id).await?;
        if !deleted {
            return Err(MomoError::NotFound(format!("Document {} not found", id)));
        }
    }

    Ok(Json(serde_json::json!({ "deleted": true })))
}

pub async fn list_documents(
    State(state): State<AppState>,
    AppJson(req): AppJson<ListDocumentsRequest>,
) -> Result<Json<ListDocumentsResponse>> {
    let (documents, pagination) = state.db.list_documents(&req).await?;

    Ok(Json(ListDocumentsResponse {
        documents,
        pagination,
    }))
}

pub async fn get_processing_documents(
    State(state): State<AppState>,
) -> Result<Json<Vec<ProcessingDocument>>> {
    let docs = state.db.get_processing_documents().await?;

    Ok(Json(docs))
}

pub async fn upload_document_file(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<CreateDocumentResponse>> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut container_tag: Option<String> = None;
    let mut metadata: Option<Metadata> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| MomoError::Validation(format!("Multipart error: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| MomoError::Validation(format!("Failed to read file: {}", e)))?;

                if bytes.len() > MAX_FILE_SIZE {
                    return Err(MomoError::Validation(format!(
                        "File too large: {} bytes (max {} bytes)",
                        bytes.len(),
                        MAX_FILE_SIZE
                    )));
                }

                file_bytes = Some(bytes.to_vec());
            }
            "container_tag" => {
                container_tag =
                    Some(field.text().await.map_err(|e| {
                        MomoError::Validation(format!("Invalid container_tag: {}", e))
                    })?);
            }
            "metadata" => {
                let json_str = field
                    .text()
                    .await
                    .map_err(|e| MomoError::Validation(format!("Invalid metadata: {}", e)))?;
                metadata = serde_json::from_str(&json_str).ok();
            }
            _ => {}
        }
    }

    let bytes =
        file_bytes.ok_or_else(|| MomoError::Validation("Missing required 'file' field".into()))?;

    let doc_type = ContentExtractor::detect_type_from_bytes(&bytes);
    if matches!(doc_type, DocumentType::Unknown) {
        return Err(MomoError::Validation("Unsupported file type".into()));
    }

    let id = nanoid!();
    let now = Utc::now();

    let mut container_tags = Vec::new();
    if let Some(ref tag) = container_tag {
        container_tags.push(tag.clone());
    }

    let content_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

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
        metadata: metadata.unwrap_or_default(),
        container_tags,
        chunk_count: 0,
        token_count: None,
        word_count: None,
        error_message: None,
        created_at: now,
        updated_at: now,
    };

    state.db.create_document(&doc).await?;

    let pipeline = state.pipeline.clone();
    let doc_id = id.clone();
    tokio::spawn(async move {
        if let Err(e) = pipeline.process_document(&doc_id).await {
            tracing::error!("Failed to process document {}: {}", doc_id, e);
        }
    });

    Ok(Json(CreateDocumentResponse {
        id,
        status: ProcessingStatus::Queued,
    }))
}
