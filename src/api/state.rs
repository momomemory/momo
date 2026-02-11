use std::sync::Arc;

use crate::config::Config;
use crate::db::DatabaseBackend;
use crate::embeddings::{EmbeddingProvider, RerankerProvider};
use crate::intelligence::MemoryExtractor;
use crate::llm::LlmProvider;
use crate::ocr::OcrProvider;
use crate::processing::ProcessingPipeline;
use crate::services::{MemoryService, SearchService};
use crate::transcription::TranscriptionProvider;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    /// Primary read/write backend.
    pub db: Arc<dyn DatabaseBackend>,
    /// Read-preferred backend (can point to replica).
    pub read_db: Arc<dyn DatabaseBackend>,
    pub embeddings: EmbeddingProvider,
    pub reranker: Option<RerankerProvider>,
    pub llm: LlmProvider,
    pub search: SearchService,
    pub memory: MemoryService,
    pub pipeline: ProcessingPipeline,
    pub extractor: MemoryExtractor,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Config,
        db: Arc<dyn DatabaseBackend>,
        read_db: Arc<dyn DatabaseBackend>,
        embeddings: EmbeddingProvider,
        reranker: Option<RerankerProvider>,
        ocr: OcrProvider,
        transcription: TranscriptionProvider,
        llm: LlmProvider,
    ) -> Self {
        let config = Arc::new(config);
        let search = SearchService::new(
            read_db.clone(),
            db.clone(),
            embeddings.clone(),
            reranker.clone(),
            llm.clone(),
            &config,
        );
        let memory = MemoryService::new(db.clone(), embeddings.clone());
        let extractor = MemoryExtractor::new(llm.clone(), embeddings.clone());
        let pipeline = ProcessingPipeline::new(
            db.clone(),
            embeddings.clone(),
            ocr,
            transcription,
            llm.clone(),
            &config,
        );

        Self {
            config,
            db,
            read_db,
            embeddings,
            reranker,
            llm,
            search,
            memory,
            pipeline,
            extractor,
        }
    }
}
