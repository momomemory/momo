use std::sync::Arc;

use chrono::Utc;
use nanoid::nanoid;

use crate::config::Config;
use crate::db::DatabaseBackend;
use crate::embeddings::EmbeddingProvider;
use crate::error::Result;
use crate::intelligence::{LlmFilter, MemoryExtractor};
use crate::llm::LlmProvider;
use crate::models::{Chunk, DocumentType, Memory, MemoryType, ProcessingStatus};
use crate::ocr::OcrProvider;
use crate::transcription::TranscriptionProvider;

use super::extractors::{AudioExtractor, ExtractedContent, ImageExtractor, VideoExtractor};
use super::{ChunkContext, ChunkerRegistry, ContentExtractor};

pub struct ProcessingPipeline {
    db: Arc<dyn DatabaseBackend>,
    embeddings: EmbeddingProvider,
    ocr: OcrProvider,
    transcription: TranscriptionProvider,
    llm: LlmProvider,
    extractor: ContentExtractor,
    memory_extractor: MemoryExtractor,
    llm_filter: LlmFilter,
    registry: ChunkerRegistry,
    ocr_config: crate::config::OcrConfig,
    transcription_config: crate::config::TranscriptionConfig,
    enable_contradiction_detection: bool,
}

impl ProcessingPipeline {
    pub fn new(
        db: Arc<dyn DatabaseBackend>,
        embeddings: EmbeddingProvider,
        ocr: OcrProvider,
        transcription: TranscriptionProvider,
        llm: LlmProvider,
        config: &Config,
    ) -> Self {
        let memory_extractor = MemoryExtractor::new(llm.clone(), embeddings.clone());
        let llm_filter = LlmFilter::new(llm.clone(), config.clone());
        let enable_contradiction_detection = config
            .llm
            .as_ref()
            .is_some_and(|l| l.enable_contradiction_detection);
        Self {
            db,
            embeddings,
            ocr,
            transcription,
            llm,
            extractor: ContentExtractor::new(),
            memory_extractor,
            llm_filter,
            registry: ChunkerRegistry::new(&config.processing),
            ocr_config: config.ocr.clone(),
            transcription_config: config.transcription.clone(),
            enable_contradiction_detection,
        }
    }

    pub async fn process_document(&self, doc_id: &str) -> Result<()> {
        let doc = self.db.get_document_by_id(doc_id).await?.ok_or_else(|| {
            crate::error::MomoError::NotFound(format!("Document {doc_id} not found"))
        })?;

        self.db
            .update_document_status(doc_id, ProcessingStatus::Extracting, None)
            .await?;

        let content = doc.content.as_deref().unwrap_or("");

        let extracted = if doc.doc_type == DocumentType::Image {
            match self.extract_image(doc_id, content).await {
                Ok(e) => e,
                Err(e) => {
                    self.db
                        .update_document_status(
                            doc_id,
                            ProcessingStatus::Failed,
                            Some(&e.to_string()),
                        )
                        .await?;
                    return Err(e);
                }
            }
        } else if doc.doc_type == DocumentType::Audio {
            match self.extract_audio(doc_id, content).await {
                Ok(e) => e,
                Err(e) => {
                    self.db
                        .update_document_status(
                            doc_id,
                            ProcessingStatus::Failed,
                            Some(&e.to_string()),
                        )
                        .await?;
                    return Err(e);
                }
            }
        } else if doc.doc_type == DocumentType::Video {
            match self.extract_video(doc_id, content).await {
                Ok(e) => e,
                Err(e) => {
                    self.db
                        .update_document_status(
                            doc_id,
                            ProcessingStatus::Failed,
                            Some(&e.to_string()),
                        )
                        .await?;
                    return Err(e);
                }
            }
        } else {
            match self.extractor.extract(content).await {
                Ok(e) => e,
                Err(e) => {
                    self.db
                        .update_document_status(
                            doc_id,
                            ProcessingStatus::Failed,
                            Some(&e.to_string()),
                        )
                        .await?;
                    return Err(e);
                }
            }
        };

        // LLM Filter Step: Check if document should be filtered
        let container_tag = doc.container_tags.first().map(|s| s.as_str()).unwrap_or("");

        if !container_tag.is_empty() {
            use crate::intelligence::filter::FilterDecision;

            let container_filter = self.db.get_container_filter(container_tag).await?;
            let override_prompt = container_filter
                .as_ref()
                .filter(|cf| cf.should_llm_filter)
                .and_then(|cf| cf.filter_prompt.as_deref());

            let filter_result = self
                .llm_filter
                .filter_content(&extracted.text, container_tag, doc_id, override_prompt)
                .await?;

            match filter_result.decision {
                FilterDecision::Skip => {
                    let reason = filter_result
                        .reasoning
                        .as_deref()
                        .unwrap_or("Content filtered by LLM");
                    let error_message = format!("Filtered: {reason}");

                    tracing::info!(
                        container_tag = %container_tag,
                        doc_id = %doc_id,
                        decision = "skip",
                        filter_reasoning = %reason,
                        "Document filtered out by LLM"
                    );

                    self.db
                        .update_document_status(
                            doc_id,
                            ProcessingStatus::Done,
                            Some(&error_message),
                        )
                        .await?;

                    return Ok(());
                }
                FilterDecision::Include => {
                    tracing::info!(
                        container_tag = %container_tag,
                        doc_id = %doc_id,
                        decision = "include",
                        filter_reasoning = ?filter_result.reasoning,
                        "Document passed LLM filter"
                    );
                }
            }
        }

        self.db
            .update_document_status(doc_id, ProcessingStatus::Chunking, None)
            .await?;

        // Create chunk context with source_path and doc_type from ExtractedContent
        let chunk_context = ChunkContext {
            source_path: extracted.source_path.clone(),
        };

        // Use registry to route to appropriate chunker based on document type
        let chunker = self
            .registry
            .get_chunker(&extracted.doc_type, extracted.source_path.as_deref());

        tracing::debug!(
            "Using {:?} chunker for document type {:?}",
            std::any::type_name::<dyn crate::processing::chunker::ContentChunker>(),
            extracted.doc_type
        );

        let text_chunks = chunker.chunk(&extracted.text, Some(&chunk_context));

        let chunks: Vec<Chunk> = text_chunks
            .iter()
            .enumerate()
            .map(|(i, tc)| Chunk {
                id: nanoid!(),
                document_id: doc_id.to_string(),
                content: tc.content.clone(),
                embedded_content: Some(tc.content.clone()),
                position: i as i32,
                token_count: Some(tc.token_count),
                created_at: Utc::now(),
            })
            .collect();

        self.db.delete_chunks_by_document_id(doc_id).await?;
        self.db.create_chunks_batch(&chunks).await?;

        self.db
            .update_document_status(doc_id, ProcessingStatus::Embedding, None)
            .await?;

        let chunk_contents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

        let embeddings = match self.embeddings.embed_passages(chunk_contents).await {
            Ok(e) => e,
            Err(e) => {
                self.db
                    .update_document_status(doc_id, ProcessingStatus::Failed, Some(&e.to_string()))
                    .await?;
                return Err(e);
            }
        };

        let updates: Vec<(String, Vec<f32>)> = chunks
            .iter()
            .zip(embeddings.iter())
            .map(|(c, e)| (c.id.clone(), e.clone()))
            .collect();

        self.db.update_chunk_embeddings_batch(&updates).await?;

        self.db
            .update_document_status(doc_id, ProcessingStatus::Indexing, None)
            .await?;

        let mut updated_doc = doc.clone();
        updated_doc.title = extracted.title.or(doc.title);
        updated_doc.doc_type = match (&doc.doc_type, &extracted.doc_type) {
            // Don't downgrade specific types to generic Text/Unknown
            (DocumentType::Code, DocumentType::Text | DocumentType::Unknown) => {
                doc.doc_type.clone()
            }
            (DocumentType::Markdown, DocumentType::Text | DocumentType::Unknown) => {
                doc.doc_type.clone()
            }
            _ => extracted.doc_type,
        };
        updated_doc.url = extracted.url.or(doc.url);
        updated_doc.word_count = Some(extracted.word_count);
        updated_doc.chunk_count = chunks.len() as i32;
        updated_doc.token_count = Some(chunks.iter().filter_map(|c| c.token_count).sum());
        updated_doc.status = ProcessingStatus::Done;
        updated_doc.updated_at = Utc::now();

        self.db.update_document(&updated_doc).await?;

        // After document is done, check for extract_memories flag
        if updated_doc
            .metadata
            .get("extract_memories")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            if let Err(error) = self
                .extract_memories_from_document(&updated_doc, &extracted.text)
                .await
            {
                tracing::warn!(doc_id = %doc_id, error = %error, "Memory extraction failed (non-blocking)");
            }
        }

        tracing::info!(
            "Document {} processed: {} chunks, {} tokens",
            doc_id,
            updated_doc.chunk_count,
            updated_doc.token_count.unwrap_or(0)
        );

        Ok(())
    }

    async fn extract_memories_from_document(
        &self,
        doc: &crate::models::Document,
        content: &str,
    ) -> Result<()> {
        let container_tag = match doc.container_tags.first() {
            Some(tag) => tag.as_str(),
            None => {
                tracing::warn!(doc_id = %doc.id, "No container tag available for memory extraction");
                return Ok(());
            }
        };

        let extraction_result = self.memory_extractor.extract(content).await?;

        if extraction_result.memories.is_empty() {
            tracing::debug!(doc_id = %doc.id, "No memories extracted from document");
            return Ok(());
        }

        let total_extracted = extraction_result.memories.len();

        let memories = if self.enable_contradiction_detection {
            self.memory_extractor
                .check_contradictions(extraction_result.memories, container_tag, self.db.as_ref())
                .await?
        } else {
            extraction_result.memories
        };

        let unique_memories = self
            .memory_extractor
            .deduplicate(memories, container_tag, self.db.as_ref())
            .await?;

        tracing::info!(
            doc_id = %doc.id,
            total_extracted,
            unique_count = unique_memories.len(),
            "Memory extraction complete"
        );

        for extracted in unique_memories {
            let mut metadata = crate::models::Metadata::new();
            metadata.insert(
                "source_document_id".to_string(),
                serde_json::Value::String(doc.id.clone()),
            );
            metadata.insert(
                "memory_type".to_string(),
                serde_json::Value::String(extracted.memory_type.clone()),
            );
            metadata.insert(
                "confidence".to_string(),
                serde_json::json!(extracted.confidence),
            );
            if let Some(context) = extracted.context.clone() {
                metadata.insert("context".to_string(), serde_json::Value::String(context));
            }

            let embedding = self.embeddings.embed_passage(&extracted.content).await?;

            let parsed_memory_type = extracted
                .memory_type
                .parse::<MemoryType>()
                .unwrap_or(MemoryType::Fact);

            let memory = Memory {
                id: nanoid::nanoid!(),
                memory: extracted.content,
                space_id: "default".to_string(),
                container_tag: Some(container_tag.to_string()),
                version: 1,
                is_latest: true,
                parent_memory_id: None,
                root_memory_id: None,
                memory_relations: Default::default(),
                source_count: 1,
                is_inference: true,
                is_forgotten: false,
                is_static: false,
                forget_after: None,
                forget_reason: None,
                memory_type: parsed_memory_type,
                last_accessed: if parsed_memory_type == MemoryType::Episode {
                    Some(Utc::now())
                } else {
                    None
                },
                confidence: Some(extracted.confidence as f64),
                metadata,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };

            self.db.create_memory(&memory).await?;
            if let Err(error) = self
                .db
                .create_memory_source(&memory.id, &doc.id, None)
                .await
            {
                tracing::warn!(
                    doc_id = %doc.id,
                    memory_id = %memory.id,
                    error = %error,
                    "Failed to create memory source (non-blocking)"
                );
            }
            self.db
                .update_memory_embedding(&memory.id, &embedding)
                .await?;
        }

        Ok(())
    }

    async fn extract_image(&self, doc_id: &str, content: &str) -> Result<ExtractedContent> {
        if !self.ocr.is_available() {
            tracing::warn!(
                doc_id = %doc_id,
                "OCR unavailable - cannot process image document"
            );
            return Err(crate::error::MomoError::OcrUnavailable(
                "OCR provider not available - image processing skipped".to_string(),
            ));
        }

        tracing::info!(doc_id = %doc_id, "Processing image document with OCR");

        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, content)
            .map_err(|e| {
                crate::error::MomoError::Processing(format!("Failed to decode base64 image: {e}"))
            })?;

        let extracted = ImageExtractor::extract(&bytes, &self.ocr, &self.ocr_config).await?;

        tracing::info!(
            doc_id = %doc_id,
            word_count = extracted.word_count,
            "Image OCR extraction complete"
        );

        Ok(extracted)
    }

    async fn extract_audio(&self, doc_id: &str, content: &str) -> Result<ExtractedContent> {
        if !self.transcription.is_available() {
            tracing::warn!(
                doc_id = %doc_id,
                "Transcription unavailable - cannot process audio document"
            );
            return Err(crate::error::MomoError::TranscriptionUnavailable(
                "Transcription provider not available - audio processing skipped".to_string(),
            ));
        }

        tracing::info!(doc_id = %doc_id, "Processing audio document with transcription");

        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, content)
            .map_err(|e| {
                crate::error::MomoError::Processing(format!("Failed to decode base64 audio: {e}"))
            })?;

        let extracted =
            AudioExtractor::extract(&bytes, &self.transcription, &self.transcription_config)
                .await?;

        tracing::info!(
            doc_id = %doc_id,
            word_count = extracted.word_count,
            "Audio transcription complete"
        );

        Ok(extracted)
    }

    async fn extract_video(&self, doc_id: &str, content: &str) -> Result<ExtractedContent> {
        if !self.transcription.is_available() {
            tracing::warn!(
                doc_id = %doc_id,
                "Transcription unavailable - cannot process video document"
            );
            return Err(crate::error::MomoError::TranscriptionUnavailable(
                "Transcription provider not available - video processing skipped".to_string(),
            ));
        }

        tracing::info!(doc_id = %doc_id, "Processing video document with transcription");

        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, content)
            .map_err(|e| {
                crate::error::MomoError::Processing(format!("Failed to decode base64 video: {e}"))
            })?;

        let extracted =
            VideoExtractor::extract(&bytes, &self.transcription, &self.transcription_config)
                .await?;

        tracing::info!(
            doc_id = %doc_id,
            word_count = extracted.word_count,
            "Video transcription complete"
        );

        Ok(extracted)
    }

    pub async fn process_pending(&self) -> Result<()> {
        let pending = self.db.get_processing_documents().await?;

        for doc in pending {
            if let Err(e) = self.process_document(&doc.id).await {
                tracing::error!("Failed to process document {}: {}", doc.id, e);
            }
        }

        Ok(())
    }
}

impl Clone for ProcessingPipeline {
    fn clone(&self) -> Self {
        Self {
            db: self.db.clone(),
            embeddings: self.embeddings.clone(),
            ocr: self.ocr.clone(),
            transcription: self.transcription.clone(),
            llm: self.llm.clone(),
            extractor: ContentExtractor::new(),
            memory_extractor: self.memory_extractor.clone(),
            llm_filter: self.llm_filter.clone(),
            registry: ChunkerRegistry::default(),
            ocr_config: self.ocr_config.clone(),
            transcription_config: self.transcription_config.clone(),
            enable_contradiction_detection: self.enable_contradiction_detection,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DatabaseConfig, EmbeddingsConfig, LlmConfig};
    use crate::db::repository::{DocumentRepository, MemoryRepository, MemorySourcesRepository};
    use crate::db::{Database, LibSqlBackend};
    use crate::models::Document;
    use serde_json::json;
    use tempfile::tempdir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn llm_response(content: &str) -> serde_json::Value {
        json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        })
    }

    #[tokio::test]
    async fn pipeline_memory_sources() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 384];
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    {
                        "embedding": embedding
                    }
                ]
            })))
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"memories":[{"content":"User prefers dark mode","memory_type":"preference","confidence":0.9}]}"#,
            )))
            .mount(&mock_server)
            .await;

        let embeddings_config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };

        let embeddings = EmbeddingProvider::new(&embeddings_config)
            .expect("failed to create embeddings provider");
        let memory_embeddings = EmbeddingProvider::new(&embeddings_config)
            .expect("failed to create memory embeddings provider");

        let llm_config = LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(mock_server.uri()),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,
            filter_prompt: None,
        };

        let llm = LlmProvider::new(Some(&llm_config));

        let config = Config::default();
        let ocr = OcrProvider::new(&config.ocr).expect("failed to create ocr provider");
        let transcription = TranscriptionProvider::new(&config.transcription)
            .expect("failed to create transcription provider");

        let temp_dir = tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("pipeline_memory_sources.db");
        let db_config = DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };

        let db = Database::new(&db_config)
            .await
            .expect("failed to create database");
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db.clone()));

        let pipeline = ProcessingPipeline {
            db: backend,
            embeddings,
            ocr,
            transcription,
            llm: llm.clone(),
            extractor: ContentExtractor::new(),
            memory_extractor: MemoryExtractor::new(llm.clone(), memory_embeddings),
            llm_filter: LlmFilter::new(llm, config.clone()),
            registry: ChunkerRegistry::new(&config.processing),
            ocr_config: config.ocr.clone(),
            transcription_config: config.transcription.clone(),
            enable_contradiction_detection: false,
        };

        let conn = db.connect().expect("failed to connect to database");
        let mut doc = Document::new("doc-1".to_string());
        doc.content = Some("User prefers dark mode".to_string());
        doc.container_tags = vec!["user-123".to_string()];

        DocumentRepository::create(&conn, &doc)
            .await
            .expect("failed to create document");

        pipeline
            .extract_memories_from_document(&doc, doc.content.as_deref().unwrap())
            .await
            .expect("memory extraction failed");

        let sources = MemorySourcesRepository::get_by_document(&conn, &doc.id)
            .await
            .expect("failed to fetch memory sources");

        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].document_id, doc.id);
        assert!(sources[0].chunk_id.is_none());

        let memory = MemoryRepository::get_by_id(&conn, &sources[0].memory_id)
            .await
            .expect("failed to fetch memory");
        assert!(memory.is_some());
    }

    #[tokio::test]
    async fn test_pipeline_filter_integration_skip() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 384];
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": embedding}]
            })))
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"decision":"skip","reasoning":"Content does not match technical criteria"}"#,
            )))
            .mount(&mock_server)
            .await;

        let embeddings_config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };

        let embeddings = EmbeddingProvider::new(&embeddings_config)
            .expect("failed to create embeddings provider");

        let llm_config = LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(mock_server.uri()),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,
            filter_prompt: Some("technical documents only".to_string()),
        };

        let llm = LlmProvider::new(Some(&llm_config));

        let mut config = Config::default();
        config.llm = Some(llm_config.clone());
        let ocr = OcrProvider::new(&config.ocr).expect("failed to create ocr provider");
        let transcription = TranscriptionProvider::new(&config.transcription)
            .expect("failed to create transcription provider");

        let temp_dir = tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("test_filter_skip.db");
        let db_config = DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };

        let db = Database::new(&db_config)
            .await
            .expect("failed to create database");
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db.clone()));

        let pipeline = ProcessingPipeline::new(
            backend.clone(),
            embeddings,
            ocr,
            transcription,
            llm,
            &config,
        );

        let conn = db.connect().expect("failed to connect to database");
        let mut doc = Document::new("doc-filter-skip".to_string());
        doc.content = Some("This is marketing content about our products".to_string());
        doc.container_tags = vec!["user-123".to_string()];

        DocumentRepository::create(&conn, &doc)
            .await
            .expect("failed to create document");

        pipeline
            .process_document(&doc.id)
            .await
            .expect("pipeline processing should succeed");

        let updated_doc = backend
            .get_document_by_id(&doc.id)
            .await
            .expect("failed to get document")
            .expect("document should exist");

        assert_eq!(updated_doc.status, ProcessingStatus::Done);
        assert!(updated_doc.error_message.is_some());
        assert!(updated_doc.error_message.unwrap().contains("Filtered:"));
        assert_eq!(updated_doc.chunk_count, 0);
    }

    #[tokio::test]
    async fn test_pipeline_filter_integration_include() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 384];
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": embedding}]
            })))
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"decision":"include","reasoning":"Content matches technical criteria"}"#,
            )))
            .mount(&mock_server)
            .await;

        let embeddings_config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };

        let embeddings = EmbeddingProvider::new(&embeddings_config)
            .expect("failed to create embeddings provider");

        let llm_config = LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(mock_server.uri()),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,
            filter_prompt: Some("technical documents only".to_string()),
        };

        let llm = LlmProvider::new(Some(&llm_config));

        let mut config = Config::default();
        config.llm = Some(llm_config.clone());
        let ocr = OcrProvider::new(&config.ocr).expect("failed to create ocr provider");
        let transcription = TranscriptionProvider::new(&config.transcription)
            .expect("failed to create transcription provider");

        let temp_dir = tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("test_filter_include.db");
        let db_config = DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };

        let db = Database::new(&db_config)
            .await
            .expect("failed to create database");
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db.clone()));

        let pipeline = ProcessingPipeline::new(
            backend.clone(),
            embeddings,
            ocr,
            transcription,
            llm,
            &config,
        );

        let conn = db.connect().expect("failed to connect to database");
        let mut doc = Document::new("doc-filter-include".to_string());
        doc.content = Some("This is technical content about Rust programming".to_string());
        doc.container_tags = vec!["user-123".to_string()];

        DocumentRepository::create(&conn, &doc)
            .await
            .expect("failed to create document");

        pipeline
            .process_document(&doc.id)
            .await
            .expect("pipeline processing should succeed");

        let updated_doc = backend
            .get_document_by_id(&doc.id)
            .await
            .expect("failed to get document")
            .expect("document should exist");

        assert_eq!(updated_doc.status, ProcessingStatus::Done);
        assert!(updated_doc.error_message.is_none());
        assert!(updated_doc.chunk_count > 0);
    }

    #[tokio::test]
    async fn test_pipeline_filter_disabled_no_filter_prompt() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 384];
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": embedding}]
            })))
            .mount(&mock_server)
            .await;

        let embeddings_config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };

        let embeddings = EmbeddingProvider::new(&embeddings_config)
            .expect("failed to create embeddings provider");

        let llm_config = LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(mock_server.uri()),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,
            filter_prompt: None,
        };

        let llm = LlmProvider::new(Some(&llm_config));

        let mut config = Config::default();
        config.llm = Some(llm_config.clone());
        let ocr = OcrProvider::new(&config.ocr).expect("failed to create ocr provider");
        let transcription = TranscriptionProvider::new(&config.transcription)
            .expect("failed to create transcription provider");

        let temp_dir = tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("test_filter_disabled.db");
        let db_config = DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };

        let db = Database::new(&db_config)
            .await
            .expect("failed to create database");
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db.clone()));

        let pipeline = ProcessingPipeline::new(
            backend.clone(),
            embeddings,
            ocr,
            transcription,
            llm,
            &config,
        );

        let conn = db.connect().expect("failed to connect to database");
        let mut doc = Document::new("doc-no-filter".to_string());
        doc.content = Some("Any content should be included".to_string());
        doc.container_tags = vec!["user-123".to_string()];

        DocumentRepository::create(&conn, &doc)
            .await
            .expect("failed to create document");

        pipeline
            .process_document(&doc.id)
            .await
            .expect("pipeline processing should succeed");

        let updated_doc = backend
            .get_document_by_id(&doc.id)
            .await
            .expect("failed to get document")
            .expect("document should exist");

        assert_eq!(updated_doc.status, ProcessingStatus::Done);
        assert!(updated_doc.error_message.is_none());
        assert!(updated_doc.chunk_count > 0);
    }

    #[tokio::test]
    async fn test_pipeline_filter_graceful_degradation_no_llm() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 384];
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": embedding}]
            })))
            .mount(&mock_server)
            .await;

        let embeddings_config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };

        let embeddings = EmbeddingProvider::new(&embeddings_config)
            .expect("failed to create embeddings provider");

        let llm = LlmProvider::unavailable("test unavailable");

        let mut config = Config::default();
        config.llm = None;
        let ocr = OcrProvider::new(&config.ocr).expect("failed to create ocr provider");
        let transcription = TranscriptionProvider::new(&config.transcription)
            .expect("failed to create transcription provider");

        let temp_dir = tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("test_filter_no_llm.db");
        let db_config = DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };

        let db = Database::new(&db_config)
            .await
            .expect("failed to create database");
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db.clone()));

        let pipeline = ProcessingPipeline::new(
            backend.clone(),
            embeddings,
            ocr,
            transcription,
            llm,
            &config,
        );

        let conn = db.connect().expect("failed to connect to database");
        let mut doc = Document::new("doc-no-llm".to_string());
        doc.content = Some("Any content should be included when LLM unavailable".to_string());
        doc.container_tags = vec!["user-123".to_string()];

        DocumentRepository::create(&conn, &doc)
            .await
            .expect("failed to create document");

        pipeline
            .process_document(&doc.id)
            .await
            .expect("pipeline processing should succeed with graceful degradation");

        let updated_doc = backend
            .get_document_by_id(&doc.id)
            .await
            .expect("failed to get document")
            .expect("document should exist");

        assert_eq!(updated_doc.status, ProcessingStatus::Done);
        assert!(updated_doc.error_message.is_none());
        assert!(updated_doc.chunk_count > 0);
    }

    #[tokio::test]
    async fn test_pipeline_filter_multiple_documents() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 384];
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": embedding}]
            })))
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"decision":"skip","reasoning":"Marketing content filtered"}"#,
            )))
            .up_to_n_times(1)
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"decision":"include","reasoning":"Technical content accepted"}"#,
            )))
            .mount(&mock_server)
            .await;

        let embeddings_config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };

        let embeddings = EmbeddingProvider::new(&embeddings_config)
            .expect("failed to create embeddings provider");

        let llm_config = LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(mock_server.uri()),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,
            filter_prompt: Some("technical documents only".to_string()),
        };

        let llm = LlmProvider::new(Some(&llm_config));

        let mut config = Config::default();
        config.llm = Some(llm_config.clone());
        let ocr = OcrProvider::new(&config.ocr).expect("failed to create ocr provider");
        let transcription = TranscriptionProvider::new(&config.transcription)
            .expect("failed to create transcription provider");

        let temp_dir = tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("test_filter_multiple.db");
        let db_config = DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };

        let db = Database::new(&db_config)
            .await
            .expect("failed to create database");
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db.clone()));

        let pipeline = ProcessingPipeline::new(
            backend.clone(),
            embeddings,
            ocr,
            transcription,
            llm,
            &config,
        );

        let conn = db.connect().expect("failed to connect to database");

        let mut doc1 = Document::new("doc-multi-1".to_string());
        doc1.content = Some("Marketing content about our amazing products".to_string());
        doc1.container_tags = vec!["user-123".to_string()];

        let mut doc2 = Document::new("doc-multi-2".to_string());
        doc2.content = Some("Technical documentation about Rust programming".to_string());
        doc2.container_tags = vec!["user-123".to_string()];

        DocumentRepository::create(&conn, &doc1)
            .await
            .expect("failed to create doc1");
        DocumentRepository::create(&conn, &doc2)
            .await
            .expect("failed to create doc2");

        pipeline
            .process_document(&doc1.id)
            .await
            .expect("doc1 processing should succeed");
        pipeline
            .process_document(&doc2.id)
            .await
            .expect("doc2 processing should succeed");

        let updated_doc1 = backend
            .get_document_by_id(&doc1.id)
            .await
            .expect("failed to get doc1")
            .expect("doc1 should exist");

        assert_eq!(updated_doc1.status, ProcessingStatus::Done);
        assert!(updated_doc1.error_message.is_some());
        assert!(updated_doc1
            .error_message
            .as_ref()
            .unwrap()
            .contains("Filtered:"));
        assert_eq!(updated_doc1.chunk_count, 0);

        let updated_doc2 = backend
            .get_document_by_id(&doc2.id)
            .await
            .expect("failed to get doc2")
            .expect("doc2 should exist");

        assert_eq!(updated_doc2.status, ProcessingStatus::Done);
        assert!(updated_doc2.error_message.is_none());
        assert!(updated_doc2.chunk_count > 0);
    }

    #[tokio::test]
    async fn test_pipeline_extract_video_unavailable_provider() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 384];
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": embedding}]
            })))
            .mount(&mock_server)
            .await;

        let config = Config::default();

        let embeddings_config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };
        let embeddings = EmbeddingProvider::new(&embeddings_config)
            .expect("failed to create embedding provider");

        let ocr = OcrProvider::new(&config.ocr).expect("failed to create ocr provider");
        let transcription = TranscriptionProvider::unavailable("test unavailable");
        let llm = LlmProvider::unavailable("test unavailable");

        let temp_dir = tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("test_video_unavailable.db");
        let db_config = DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };

        let db = Database::new(&db_config)
            .await
            .expect("failed to create database");
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db.clone()));

        let pipeline = ProcessingPipeline::new(
            backend.clone(),
            embeddings,
            ocr,
            transcription,
            llm,
            &config,
        );

        let video_base64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            b"fake-video-data",
        );

        let result = pipeline.extract_video("test-doc-id", &video_base64).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("unavailable") || err.contains("not available"),
            "Error should mention unavailable: {err}"
        );
    }

    #[tokio::test]
    async fn test_pipeline_extract_video_invalid_base64() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 384];
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": embedding}]
            })))
            .mount(&mock_server)
            .await;

        let config = Config::default();

        let embeddings_config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };
        let embeddings = EmbeddingProvider::new(&embeddings_config)
            .expect("failed to create embedding provider");

        let ocr = OcrProvider::new(&config.ocr).expect("failed to create ocr provider");
        let transcription = TranscriptionProvider::unavailable("test");
        let llm = LlmProvider::unavailable("test unavailable");

        let temp_dir = tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("test_video_invalid_base64.db");
        let db_config = DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };

        let db = Database::new(&db_config)
            .await
            .expect("failed to create database");
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db.clone()));

        let pipeline = ProcessingPipeline::new(
            backend.clone(),
            embeddings,
            ocr,
            transcription,
            llm,
            &config,
        );

        let invalid_base64 = "not-valid-base64!!!";

        let result = pipeline.extract_video("test-doc-id", invalid_base64).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pipeline_end_to_end_video_processing() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 384];
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"embedding": embedding}]
            })))
            .mount(&mock_server)
            .await;

        let embeddings_config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };

        let embeddings = EmbeddingProvider::new(&embeddings_config)
            .expect("failed to create embeddings provider");

        let config = Config::default();
        let ocr = OcrProvider::new(&config.ocr).expect("failed to create ocr provider");
        let transcription = TranscriptionProvider::unavailable("test - integration test");
        let llm = LlmProvider::unavailable("test unavailable");

        let temp_dir = tempdir().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("test_video_e2e.db");
        let db_config = DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };

        let db = Database::new(&db_config)
            .await
            .expect("failed to create database");
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db.clone()));

        let pipeline = ProcessingPipeline::new(
            backend.clone(),
            embeddings,
            ocr,
            transcription,
            llm,
            &config,
        );

        let conn = db.connect().expect("failed to connect to database");
        let mut doc = Document::new("video-doc-e2e".to_string());
        doc.doc_type = DocumentType::Video;
        doc.content = Some(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            b"fake-video-bytes-for-testing",
        ));
        doc.container_tags = vec!["user-123".to_string()];

        DocumentRepository::create(&conn, &doc)
            .await
            .expect("failed to create document");

        let result = pipeline.process_document(&doc.id).await;

        assert!(
            result.is_err(),
            "Processing should fail with unavailable transcription provider"
        );

        let updated_doc = backend
            .get_document_by_id(&doc.id)
            .await
            .expect("failed to get document")
            .expect("document should exist");

        assert_eq!(updated_doc.status, ProcessingStatus::Failed);
        assert!(updated_doc.error_message.is_some());
        let error_msg = updated_doc.error_message.unwrap();
        assert!(
            error_msg.contains("unavailable") || error_msg.contains("not available"),
            "Error message should indicate transcription unavailability: {error_msg}"
        );
    }
}
