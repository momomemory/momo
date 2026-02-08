use std::sync::Arc;

use crate::config::Config;
use crate::db::DatabaseBackend;
use crate::embeddings::{EmbeddingProvider, RerankerProvider};
use crate::intelligence::{MemoryExtractor, RelationshipDetector};
use crate::llm::LlmProvider;
use crate::ocr::OcrProvider;
use crate::processing::ProcessingPipeline;
use crate::services::{MemoryService, SearchService};
use crate::transcription::TranscriptionProvider;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub db: Arc<dyn DatabaseBackend>,
    pub embeddings: EmbeddingProvider,
    pub reranker: Option<RerankerProvider>,
    pub ocr: OcrProvider,
    pub transcription: TranscriptionProvider,
    pub llm: LlmProvider,
    pub search: SearchService,
    pub memory: MemoryService,
    pub pipeline: ProcessingPipeline,
    pub extractor: MemoryExtractor,
    pub relationship_detector: RelationshipDetector,
}

impl AppState {
    pub fn new(
        config: Config,
        db: Arc<dyn DatabaseBackend>,
        embeddings: EmbeddingProvider,
        reranker: Option<RerankerProvider>,
        ocr: OcrProvider,
        transcription: TranscriptionProvider,
        llm: LlmProvider,
    ) -> Self {
        let config = Arc::new(config);
        let search = SearchService::new(
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
            ocr.clone(),
            transcription.clone(),
            llm.clone(),
            &config,
        );
        let relationship_detector = RelationshipDetector::new(llm.clone(), embeddings.clone());

        Self {
            config,
            db,
            embeddings,
            reranker,
            ocr,
            transcription,
            llm,
            search,
            memory,
            pipeline,
            extractor,
            relationship_detector,
        }
    }
}
