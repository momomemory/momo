use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use crate::db::DatabaseBackend;
use crate::embeddings::{EmbeddingProvider, RerankerProvider};
use crate::error::{MomoError, Result};
use crate::intelligence::TemporalSearchRanker;
use crate::llm::LlmProvider;
use crate::models::{
    ChunkSearchResult, Document, DocumentSearchResult, HybridSearchRequest, HybridSearchResponse,
    HybridSearchResult, Memory, MemoryContext, MemoryRelationInfo, MemorySearchResult,
    SearchDocumentsRequest, SearchDocumentsResponse, SearchFilters, SearchMemoriesRequest,
    SearchMemoriesResponse, SearchMode,
};
use crate::search::QueryRewriteCache;

#[derive(Clone)]
pub struct SearchService {
    read_db: Arc<dyn DatabaseBackend>,
    write_db: Arc<dyn DatabaseBackend>,
    embeddings: EmbeddingProvider,
    reranker: Option<RerankerProvider>,
    llm: LlmProvider,
    rewrite_cache: Option<QueryRewriteCache>,
    episode_decay_days: f64,
    episode_decay_factor: f64,
}

fn apply_metadata_filters(
    results: Vec<DocumentSearchResult>,
    filters: &Option<SearchFilters>,
) -> Vec<DocumentSearchResult> {
    let Some(filters) = filters else {
        return results;
    };

    results
        .into_iter()
        .filter(|result| {
            let matches_and = if let Some(ref conditions) = filters.and {
                conditions.iter().all(|cond| {
                    let matches = result
                        .metadata
                        .get(&cond.key)
                        .and_then(|value| value.as_str())
                        .is_some_and(|value| value == cond.value);
                    if cond.negate.unwrap_or(false) {
                        !matches
                    } else {
                        matches
                    }
                })
            } else {
                true
            };

            let matches_or = if let Some(ref conditions) = filters.or {
                conditions.iter().any(|cond| {
                    let matches = result
                        .metadata
                        .get(&cond.key)
                        .and_then(|value| value.as_str())
                        .is_some_and(|value| value == cond.value);
                    if cond.negate.unwrap_or(false) {
                        !matches
                    } else {
                        matches
                    }
                })
            } else {
                true
            };

            matches_and && matches_or
        })
        .collect()
}

fn apply_memory_similarity(
    ranker: &TemporalSearchRanker,
    memory: &Memory,
    db_similarity: f32,
) -> f32 {
    ranker.apply_episode_decay(memory, db_similarity)
}

impl SearchService {
    pub fn new(
        read_db: Arc<dyn DatabaseBackend>,
        write_db: Arc<dyn DatabaseBackend>,
        embeddings: EmbeddingProvider,
        reranker: Option<RerankerProvider>,
        llm: LlmProvider,
        config: &crate::config::Config,
    ) -> Self {
        let rewrite_cache = if let Some(ref llm_config) = config.llm {
            if llm_config.enable_query_rewrite {
                Some(QueryRewriteCache::new(llm_config.query_rewrite_cache_size))
            } else {
                None
            }
        } else {
            None
        };

        Self {
            read_db,
            write_db,
            embeddings,
            reranker,
            llm,
            rewrite_cache,
            episode_decay_days: config.memory.episode_decay_days,
            episode_decay_factor: config.memory.episode_decay_factor,
        }
    }

    async fn maybe_rewrite_query(&self, req: &SearchDocumentsRequest) -> Option<String> {
        if !req.rewrite_query.unwrap_or(false) {
            return None;
        }

        let query_len = req.q.len();
        if !(3..=500).contains(&query_len) {
            return None;
        }

        if !self.llm.is_available() {
            let truncated = &req.q[..req.q.len().min(20)];
            tracing::debug!(
                "Query rewrite requested but LLM not available (query: '{}...')",
                truncated
            );
            return None;
        }

        if let Some(ref cache) = self.rewrite_cache {
            let cache_key = cache.generate_key(&req.q);
            if let Some(cached) = cache.get(&cache_key) {
                return Some(cached);
            }
        }

        let timeout = if let Some(config) = self.llm.config() {
            std::time::Duration::from_secs(config.query_rewrite_timeout_secs)
        } else {
            std::time::Duration::from_secs(5)
        };

        let prompt = crate::llm::prompts::query_rewrite_prompt(&req.q);
        let llm_call = self.llm.complete(&prompt, None);

        match tokio::time::timeout(timeout, llm_call).await {
            Ok(Ok(rewritten)) => {
                let rewritten = rewritten.trim().to_string();

                if rewritten.is_empty() || rewritten.len() < 3 || rewritten == req.q {
                    return None;
                }

                let truncated_orig = &req.q[..req.q.len().min(20)];
                let truncated_rewr = &rewritten[..rewritten.len().min(20)];
                tracing::info!(
                    "Query rewritten: '{}...' -> '{}...'",
                    truncated_orig,
                    truncated_rewr
                );

                if let Some(ref cache) = self.rewrite_cache {
                    let cache_key = cache.generate_key(&req.q);
                    cache.put(cache_key, rewritten.clone());
                }

                Some(rewritten)
            }
            Ok(Err(e)) => {
                tracing::warn!("Query rewrite failed: {}, using original", e);
                None
            }
            Err(_) => {
                tracing::warn!("Query rewrite timeout, using original");
                None
            }
        }
    }

    async fn maybe_rewrite_memory_query(&self, req: &SearchMemoriesRequest) -> Option<String> {
        if !req.rewrite_query.unwrap_or(false) {
            return None;
        }

        let query_len = req.q.len();
        if !(3..=500).contains(&query_len) {
            return None;
        }

        if !self.llm.is_available() {
            let truncated = &req.q[..req.q.len().min(20)];
            tracing::debug!(
                "Query rewrite requested but LLM not available (query: '{}...')",
                truncated
            );
            return None;
        }

        if let Some(ref cache) = self.rewrite_cache {
            let cache_key = cache.generate_key(&req.q);
            if let Some(cached) = cache.get(&cache_key) {
                return Some(cached);
            }
        }

        let timeout = if let Some(config) = self.llm.config() {
            std::time::Duration::from_secs(config.query_rewrite_timeout_secs)
        } else {
            std::time::Duration::from_secs(5)
        };

        let prompt = crate::llm::prompts::query_rewrite_prompt(&req.q);
        let llm_call = self.llm.complete(&prompt, None);

        match tokio::time::timeout(timeout, llm_call).await {
            Ok(Ok(rewritten)) => {
                let rewritten = rewritten.trim().to_string();

                if rewritten.is_empty() || rewritten.len() < 3 || rewritten == req.q {
                    return None;
                }

                let truncated_orig = &req.q[..req.q.len().min(20)];
                let truncated_rewr = &rewritten[..rewritten.len().min(20)];
                tracing::info!(
                    "Query rewritten: '{}...' -> '{}...'",
                    truncated_orig,
                    truncated_rewr
                );

                if let Some(ref cache) = self.rewrite_cache {
                    let cache_key = cache.generate_key(&req.q);
                    cache.put(cache_key, rewritten.clone());
                }

                Some(rewritten)
            }
            Ok(Err(e)) => {
                tracing::warn!("Query rewrite failed: {}, using original", e);
                None
            }
            Err(_) => {
                tracing::warn!("Query rewrite timeout, using original");
                None
            }
        }
    }

    pub async fn search_documents(
        &self,
        mut req: SearchDocumentsRequest,
    ) -> Result<SearchDocumentsResponse> {
        let start = Instant::now();

        // Try to rewrite query if requested
        let original_query = req.q.clone();
        if let Some(rewritten) = self.maybe_rewrite_query(&req).await {
            req.q = rewritten;
        }

        let query_embedding = self.embeddings.embed_query(&req.q).await?;

        let threshold = req.chunk_threshold.unwrap_or(0.5);
        let limit = req.limit.unwrap_or(10).min(100);

        let chunk_results = self
            .read_db
            .search_similar_chunks(
                &query_embedding,
                limit * 3,
                threshold,
                req.container_tags.as_deref(),
            )
            .await?;

        let mut doc_chunks: HashMap<String, Vec<_>> = HashMap::new();
        for chunk in chunk_results {
            doc_chunks
                .entry(chunk.document_id.clone())
                .or_default()
                .push(chunk);
        }

        let mut results: Vec<DocumentSearchResult> = Vec::new();

        let doc_ids: Vec<String> = doc_chunks.keys().cloned().collect();
        let docs = self.read_db.get_documents_by_ids(&doc_ids).await?;
        let doc_map: HashMap<String, Document> =
            docs.into_iter().map(|d| (d.id.clone(), d)).collect();

        for (doc_id, chunks) in doc_chunks {
            if let Some(doc) = doc_map.get(&doc_id) {
                let max_score = chunks.iter().map(|c| c.score).fold(0.0f32, f32::max);

                let chunk_results: Vec<ChunkSearchResult> = chunks
                    .iter()
                    .map(|c| ChunkSearchResult {
                        content: c.chunk_content.clone(),
                        score: c.score,
                        rerank_score: None,
                        is_relevant: c.score >= threshold,
                    })
                    .collect();

                results.push(DocumentSearchResult {
                    document_id: doc.id.clone(),
                    title: doc.title.clone(),
                    doc_type: Some(doc.doc_type.clone()),
                    score: max_score,
                    rerank_score: None,
                    chunks: if req.only_matching_chunks.unwrap_or(true) {
                        chunk_results
                            .into_iter()
                            .filter(|c| c.is_relevant)
                            .collect()
                    } else {
                        chunk_results
                    },
                    summary: if req.include_summary.unwrap_or(false) {
                        doc.summary.clone()
                    } else {
                        None
                    },
                    content: if req.include_full_docs.unwrap_or(false) {
                        doc.content.clone()
                    } else {
                        None
                    },
                    metadata: doc.metadata.clone(),
                    created_at: doc.created_at,
                    updated_at: doc.updated_at,
                });
            }
        }

        if req.rerank.unwrap_or(false) {
            if let Some(ref reranker) = self.reranker {
                if reranker.is_enabled() {
                    let rerank_level = req.rerank_level.as_deref().unwrap_or("auto");
                    let total_chunks: usize = results.iter().map(|r| r.chunks.len()).sum();

                    let use_chunk_level = match rerank_level {
                        "chunk" => true,
                        "document" => false,
                        "auto" => total_chunks < 20,
                        _ => {
                            tracing::warn!("Invalid rerank_level '{}', using 'auto'", rerank_level);
                            total_chunks < 20
                        }
                    };

                    let config_top_k = self.reranker.as_ref().map(|_| 100).unwrap_or(100);
                    let rerank_top_k = req.rerank_top_k.unwrap_or(config_top_k);

                    match self
                        .apply_reranking(&req.q, &mut results, use_chunk_level, rerank_top_k)
                        .await
                    {
                        Ok(_) => {
                            tracing::debug!("Reranking applied successfully");
                        }
                        Err(e) => {
                            tracing::warn!("Reranking failed, falling back to base scores: {}", e);
                        }
                    }
                }
            } else {
                tracing::debug!("Reranking requested but reranker not available");
            }
        }

        results.sort_by(|a, b| {
            let a_score = a.rerank_score.unwrap_or(a.score);
            let b_score = b.rerank_score.unwrap_or(b.score);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut results = apply_metadata_filters(results, &req.filters);
        results.truncate(limit as usize);

        let total = results.len() as u32;
        let timing = start.elapsed().as_millis() as u64;

        let rewritten_query = if req.q != original_query {
            Some(original_query)
        } else {
            None
        };

        Ok(SearchDocumentsResponse {
            results,
            total,
            timing,
            rewritten_query,
        })
    }

    async fn apply_reranking(
        &self,
        query: &str,
        results: &mut [DocumentSearchResult],
        use_chunk_level: bool,
        top_k: usize,
    ) -> Result<()> {
        let reranker = self.reranker.as_ref().ok_or_else(|| {
            crate::error::MomoError::Reranker("Reranker not available".to_string())
        })?;

        if use_chunk_level {
            let mut all_chunks_with_idx: Vec<(usize, usize, String)> = Vec::new();

            for (doc_idx, doc_result) in results.iter().enumerate() {
                for (chunk_idx, chunk) in doc_result.chunks.iter().enumerate() {
                    all_chunks_with_idx.push((doc_idx, chunk_idx, chunk.content.clone()));
                }
            }

            if !all_chunks_with_idx.is_empty() {
                let chunk_texts: Vec<String> = all_chunks_with_idx
                    .iter()
                    .map(|(_, _, text)| text.clone())
                    .collect();

                let rerank_results = reranker.rerank(query, chunk_texts, top_k).await?;

                for rerank_result in rerank_results {
                    if rerank_result.index < all_chunks_with_idx.len() {
                        let (doc_idx, chunk_idx, _) = all_chunks_with_idx[rerank_result.index];
                        if let Some(doc) = results.get_mut(doc_idx) {
                            if let Some(chunk) = doc.chunks.get_mut(chunk_idx) {
                                chunk.rerank_score = Some(rerank_result.score);
                            }
                        }
                    }
                }

                for doc in results.iter_mut() {
                    let max_rerank_score = doc
                        .chunks
                        .iter()
                        .filter_map(|c| c.rerank_score)
                        .fold(None, |max, score| {
                            Some(max.map_or(score, |m: f32| m.max(score)))
                        });

                    if let Some(score) = max_rerank_score {
                        doc.rerank_score = Some(score);
                    }
                }
            }
        } else {
            let mut doc_texts: Vec<String> = Vec::new();

            for doc_result in results.iter() {
                let doc_text = if let Some(ref content) = doc_result.content {
                    content.clone()
                } else {
                    doc_result
                        .chunks
                        .iter()
                        .take(5)
                        .map(|c| c.content.as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n")
                };
                doc_texts.push(doc_text);
            }

            if !doc_texts.is_empty() {
                let rerank_results = reranker.rerank(query, doc_texts, top_k).await?;

                for rerank_result in rerank_results {
                    if rerank_result.index < results.len() {
                        results[rerank_result.index].rerank_score = Some(rerank_result.score);
                    }
                }
            }
        }

        Ok(())
    }

    pub async fn search_memories(
        &self,
        mut req: SearchMemoriesRequest,
    ) -> Result<SearchMemoriesResponse> {
        let start = Instant::now();

        // Try to rewrite query if requested
        let original_query = req.q.clone();
        if let Some(rewritten) = self.maybe_rewrite_memory_query(&req).await {
            req.q = rewritten;
        }

        let query_embedding = self.embeddings.embed_query(&req.q).await?;

        let threshold = req.threshold.unwrap_or(0.6);
        let limit = req.limit.unwrap_or(10).min(100);

        let include_opts = req.include.as_ref();
        let include_forgotten = include_opts
            .and_then(|i| i.forgotten_memories)
            .unwrap_or(false);

        let memories = self
            .read_db
            .search_similar_memories(
                &query_embedding,
                limit,
                threshold,
                req.container_tag.as_deref(),
                include_forgotten,
            )
            .await?;

        let include_opts = req.include.unwrap_or_default();
        let ranker = TemporalSearchRanker::new(self.episode_decay_days, self.episode_decay_factor);

        let mut results: Vec<MemorySearchResult> = Vec::new();

        let all_related_ids: Vec<String> = if include_opts.related_memories.unwrap_or(false) {
            memories
                .iter()
                .flat_map(|hit| hit.memory.memory_relations.keys().cloned())
                .collect::<HashSet<String>>()
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };

        let related_map: HashMap<String, Memory> = if !all_related_ids.is_empty() {
            self.read_db
                .get_memories_by_ids(&all_related_ids)
                .await?
                .into_iter()
                .map(|m| (m.id.clone(), m))
                .collect()
        } else {
            HashMap::new()
        };

        for hit in memories {
            let db_similarity = hit.score;
            let memory = hit.memory;
            let similarity = apply_memory_similarity(&ranker, &memory, db_similarity);

            let context = if include_opts.related_memories.unwrap_or(false) {
                let parents = if let Some(ref root_id) = memory.root_memory_id {
                    self.read_db.get_memory_parents(root_id).await?
                } else {
                    Vec::new()
                };

                let children = self.read_db.get_memory_children(&memory.id).await?;

                let mut related = Vec::new();
                for (related_id, relation_type) in &memory.memory_relations {
                    if let Some(related_memory) = related_map.get(related_id) {
                        related.push(MemoryRelationInfo {
                            id: related_memory.id.clone(),
                            relation: relation_type.clone(),
                            version: Some(related_memory.version),
                            memory: related_memory.memory.clone(),
                            metadata: Some(related_memory.metadata.clone()),
                            updated_at: related_memory.updated_at,
                        });
                    }
                }

                Some(MemoryContext {
                    parents: parents
                        .into_iter()
                        .map(|m| MemoryRelationInfo {
                            id: m.id.clone(),
                            relation: crate::models::MemoryRelationType::Updates,
                            version: Some(m.version),
                            memory: m.memory,
                            metadata: Some(m.metadata),
                            updated_at: m.updated_at,
                        })
                        .collect(),
                    children: children
                        .into_iter()
                        .map(|m| MemoryRelationInfo {
                            id: m.id.clone(),
                            relation: crate::models::MemoryRelationType::Extends,
                            version: Some(m.version),
                            memory: m.memory,
                            metadata: Some(m.metadata),
                            updated_at: m.updated_at,
                        })
                        .collect(),
                    related,
                })
            } else {
                None
            };

            let documents = None;

            results.push(MemorySearchResult {
                id: memory.id,
                memory: Some(memory.memory.clone()),
                chunk: None,
                metadata: memory.metadata,
                similarity,
                rerank_score: None,
                version: Some(memory.version),
                updated_at: memory.updated_at,
                context,
                documents,
            });
        }

        // Apply reranking AFTER temporal decay if requested
        if req.rerank.unwrap_or(false) {
            if let Some(ref reranker) = self.reranker {
                if reranker.is_enabled() {
                    // Gather memory texts for reranking
                    let memory_texts: Vec<String> =
                        results.iter().filter_map(|r| r.memory.clone()).collect();

                    if !memory_texts.is_empty() {
                        let config_top_k = 100; // Default from RerankerConfig
                        let rerank_top_k = memory_texts.len().min(config_top_k);

                        match reranker.rerank(&req.q, memory_texts, rerank_top_k).await {
                            Ok(rerank_results) => {
                                // Store rerank scores in results
                                for rerank_result in rerank_results {
                                    if rerank_result.index < results.len() {
                                        results[rerank_result.index].rerank_score =
                                            Some(rerank_result.score);
                                    }
                                }
                                tracing::debug!("Memory reranking applied successfully");
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Memory reranking failed, using temporally-decayed scores: {}",
                                    e
                                );
                            }
                        }
                    }
                }
            } else {
                tracing::debug!("Memory reranking requested but reranker not available");
            }
        }

        // Sort by rerank_score when available, otherwise by similarity (with temporal decay)
        results.sort_by(|a, b| {
            let a_score = a.rerank_score.unwrap_or(a.similarity);
            let b_score = b.rerank_score.unwrap_or(b.similarity);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // After finalizing results (temporal decay and optional reranking applied),
        // update last_accessed for returned episode memories in batch.
        // Collect IDs from the finalized results only (do not include filtered-out items).
        let ids_vec: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();

        if !ids_vec.is_empty() {
            match self.write_db.update_memory_last_accessed_batch(&ids_vec).await {
                Ok(updated_rows) => {
                    tracing::debug!(count = updated_rows, "Updated last_accessed for memories")
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to update last_accessed for memories")
                }
            }
        }

        let total = results.len() as u32;
        let timing = start.elapsed().as_millis() as u64;

        let rewritten_query = if req.q != original_query {
            Some(original_query)
        } else {
            None
        };

        Ok(SearchMemoriesResponse {
            results,
            total,
            timing,
            rewritten_query,
        })
    }

    pub async fn search_hybrid(
        &self,
        mut req: HybridSearchRequest,
    ) -> Result<HybridSearchResponse> {
        let start = Instant::now();

        let original_query = req.q.clone();
        let rewrite_request = SearchMemoriesRequest {
            q: req.q.clone(),
            container_tag: req.container_tag.clone(),
            threshold: req.threshold,
            filters: req.filters.clone(),
            include: req.include.clone(),
            limit: req.limit,
            rerank: req.rerank,
            rewrite_query: req.rewrite_query,
        };

        if let Some(rewritten) = self.maybe_rewrite_memory_query(&rewrite_request).await {
            req.q = rewritten;
        }

        let query_embedding = self.embeddings.embed_query(&req.q).await?;

        let limit = req.limit.unwrap_or(10).min(100);
        let threshold = req.threshold.unwrap_or(0.6);
        let rerank_enabled = req.rerank.unwrap_or(false);
        let search_mode = req.search_mode;

        let include_opts = req.include.clone().unwrap_or_default();
        let include_forgotten = include_opts.forgotten_memories.unwrap_or(false);

        let container_tag = req.container_tag.clone();
        let container_tags = container_tag.as_ref().map(|tag| vec![tag.clone()]);
        let filters = req.filters.clone();
        let query = req.q.clone();

        let document_query = query.clone();
        let document_container_tags = container_tags.clone();

        let documents_future = async {
            if search_mode == SearchMode::Memories {
                return Ok(Vec::new());
            }

            let doc_limit = limit.saturating_mul(3);

            let chunk_results = self
                .read_db
                .search_similar_chunks(
                    &query_embedding,
                    doc_limit,
                    threshold,
                    document_container_tags.as_deref(),
                )
                .await?;

            let mut doc_chunks: HashMap<String, Vec<_>> = HashMap::new();
            for chunk in chunk_results {
                doc_chunks
                    .entry(chunk.document_id.clone())
                    .or_default()
                    .push(chunk);
            }

            let mut results: Vec<DocumentSearchResult> = Vec::new();
            let mut chunk_ids_by_doc: HashMap<String, Vec<String>> = HashMap::new();

            let doc_ids: Vec<String> = doc_chunks.keys().cloned().collect();
            let docs = self.read_db.get_documents_by_ids(&doc_ids).await?;
            let doc_map: HashMap<String, Document> =
                docs.into_iter().map(|d| (d.id.clone(), d)).collect();

            for (doc_id, chunks) in doc_chunks {
                if let Some(doc) = doc_map.get(&doc_id) {
                    let max_score = chunks.iter().map(|c| c.score).fold(0.0f32, f32::max);

                    let mut chunk_results: Vec<ChunkSearchResult> = Vec::new();
                    let mut chunk_ids: Vec<String> = Vec::new();

                    for chunk in &chunks {
                        chunk_results.push(ChunkSearchResult {
                            content: chunk.chunk_content.clone(),
                            score: chunk.score,
                            rerank_score: None,
                            is_relevant: chunk.score >= threshold,
                        });
                        chunk_ids.push(chunk.chunk_id.clone());
                    }

                    chunk_ids_by_doc.insert(doc_id.clone(), chunk_ids);

                    results.push(DocumentSearchResult {
                        document_id: doc.id.clone(),
                        title: doc.title.clone(),
                        doc_type: Some(doc.doc_type.clone()),
                        score: max_score,
                        rerank_score: None,
                        chunks: chunk_results,
                        summary: None,
                        content: None,
                        metadata: doc.metadata.clone(),
                        created_at: doc.created_at,
                        updated_at: doc.updated_at,
                    });
                }
            }

            if rerank_enabled {
                if let Some(ref reranker) = self.reranker {
                    if reranker.is_enabled() {
                        let total_chunks: usize = results.iter().map(|r| r.chunks.len()).sum();
                        if total_chunks > 0 {
                            let rerank_top_k = total_chunks.min(100);
                            if let Err(error) = self
                                .apply_reranking(&document_query, &mut results, true, rerank_top_k)
                                .await
                            {
                                tracing::warn!(
                                    "Hybrid document reranking failed, using base scores: {}",
                                    error
                                );
                            }
                        }
                    }
                } else {
                    tracing::debug!(
                        "Hybrid document reranking requested but reranker not available"
                    );
                }
            }

            let results = apply_metadata_filters(results, &filters);
            let mut hybrid_results = Vec::new();

            for doc in results {
                let Some(chunk_ids) = chunk_ids_by_doc.get(&doc.document_id) else {
                    continue;
                };

                for (idx, chunk) in doc.chunks.iter().enumerate() {
                    let Some(chunk_id) = chunk_ids.get(idx) else {
                        continue;
                    };

                    hybrid_results.push(HybridSearchResult {
                        id: chunk_id.clone(),
                        memory: None,
                        chunk: Some(chunk.content.clone()),
                        document_id: Some(doc.document_id.clone()),
                        similarity: chunk.score,
                        rerank_score: chunk.rerank_score,
                        metadata: doc.metadata.clone(),
                        updated_at: doc.updated_at,
                    });
                }
            }

            Ok(hybrid_results)
        };

        let memory_query = query.clone();
        let memory_container_tag = container_tag.clone();

        let memories_future = async {
            if search_mode == SearchMode::Documents {
                return Ok(Vec::new());
            }

            let memory_limit = limit.saturating_mul(3);

            let memories = self
                .read_db
                .search_similar_memories(
                    &query_embedding,
                    memory_limit,
                    threshold,
                    memory_container_tag.as_deref(),
                    include_forgotten,
                )
                .await?;

            let ranker =
                TemporalSearchRanker::new(self.episode_decay_days, self.episode_decay_factor);
            let mut results: Vec<MemorySearchResult> = Vec::new();

            for hit in memories {
                let db_similarity = hit.score;
                let memory = hit.memory;
                let similarity = apply_memory_similarity(&ranker, &memory, db_similarity);

                results.push(MemorySearchResult {
                    id: memory.id,
                    memory: Some(memory.memory.clone()),
                    chunk: None,
                    metadata: memory.metadata,
                    similarity,
                    rerank_score: None,
                    version: Some(memory.version),
                    updated_at: memory.updated_at,
                    context: None,
                    documents: None,
                });
            }

            if rerank_enabled {
                if let Some(ref reranker) = self.reranker {
                    if reranker.is_enabled() {
                        let memory_texts: Vec<String> =
                            results.iter().filter_map(|r| r.memory.clone()).collect();

                        if !memory_texts.is_empty() {
                            let config_top_k = 100;
                            let rerank_top_k = memory_texts.len().min(config_top_k);

                            match reranker
                                .rerank(&memory_query, memory_texts, rerank_top_k)
                                .await
                            {
                                Ok(rerank_results) => {
                                    for rerank_result in rerank_results {
                                        if rerank_result.index < results.len() {
                                            results[rerank_result.index].rerank_score =
                                                Some(rerank_result.score);
                                        }
                                    }
                                    tracing::debug!("Hybrid memory reranking applied successfully");
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Hybrid memory reranking failed, using temporally-decayed scores: {}",
                                        e
                                    );
                                }
                            }
                        }
                    }
                } else {
                    tracing::debug!("Hybrid memory reranking requested but reranker not available");
                }
            }

            Ok(results)
        };

        let (document_results, memory_results) = tokio::join!(documents_future, memories_future);

        let (mut chunk_results, document_error) = match document_results {
            Ok(results) => (results, None),
            Err(error) => {
                tracing::warn!("Hybrid document search failed: {}", error);
                (Vec::new(), Some(error))
            }
        };

        let (memory_results, memory_error) = match memory_results {
            Ok(results) => (results, None),
            Err(error) => {
                tracing::warn!("Hybrid memory search failed: {}", error);
                (Vec::new(), Some(error))
            }
        };

        if document_error.is_some() && memory_error.is_some() {
            return Err(document_error.unwrap_or_else(|| {
                memory_error
                    .unwrap_or_else(|| MomoError::Internal("Hybrid search failed".to_string()))
            }));
        }

        if !memory_results.is_empty() && !chunk_results.is_empty() {
            let mut memory_doc_ids: HashSet<String> = HashSet::new();

            for memory in &memory_results {
                match self.read_db.get_sources_by_memory(&memory.id).await {
                    Ok(sources) => {
                        for source in sources {
                            if !source.document_id.is_empty() {
                                memory_doc_ids.insert(source.document_id);
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            memory_id = %memory.id,
                            error = %error,
                            "Failed to load memory sources for hybrid dedup"
                        );
                    }
                }
            }

            if !memory_doc_ids.is_empty() {
                chunk_results.retain(|chunk| match chunk.document_id.as_deref() {
                    Some(doc_id) => !memory_doc_ids.contains(doc_id),
                    None => true,
                });
            }
        }

        let mut results: Vec<HybridSearchResult> = memory_results
            .into_iter()
            .map(|memory| HybridSearchResult {
                id: memory.id,
                memory: memory.memory,
                chunk: None,
                document_id: None,
                similarity: memory.similarity,
                rerank_score: memory.rerank_score,
                metadata: memory.metadata,
                updated_at: memory.updated_at,
            })
            .collect();

        results.extend(chunk_results);

        results.sort_by(|a, b| {
            let a_score = a.rerank_score.unwrap_or(a.similarity);
            let b_score = b.rerank_score.unwrap_or(b.similarity);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit as usize);

        // After finalizing hybrid results (deduplication and truncation), collect
        // memory IDs from returned results and update last_accessed in batch.
        let mut memory_ids: Vec<&str> = results
            .iter()
            .filter_map(|r| {
                if r.memory.is_some() {
                    Some(r.id.as_str())
                } else {
                    None
                }
            })
            .collect();

        // Deduplicate ids
        let mut seen_ids: HashSet<&str> = HashSet::new();
        memory_ids.retain(|id| seen_ids.insert(*id));

        if !memory_ids.is_empty() {
            match self
                .write_db
                .update_memory_last_accessed_batch(&memory_ids)
                .await
            {
                Ok(updated_rows) => tracing::debug!(
                    count = updated_rows,
                    "Updated last_accessed for hybrid memories"
                ),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to update last_accessed for hybrid memories")
                }
            }
        }

        let total = results.len() as u32;
        let timing = start.elapsed().as_millis() as u64;

        let rewritten_query = if req.q != original_query {
            Some(original_query)
        } else {
            None
        };

        Ok(HybridSearchResponse {
            results,
            total,
            timing,
            rewritten_query,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, EmbeddingsConfig};
    use crate::db::repository::{
        ChunkRepository, DocumentRepository, MemoryRepository, MemorySourcesRepository,
    };
    use crate::db::{Database, LibSqlBackend};
    use crate::embeddings::RerankResult;
    use crate::llm::LlmProvider;
    use crate::models::{Document, Memory, MemoryType, ProcessingStatus};
    use chrono::{DateTime, Utc};
    use serde_json::json;
    use std::collections::HashMap;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn test_rerank_score_sorting() {
        let mut results = vec![
            DocumentSearchResult {
                document_id: "doc1".to_string(),
                title: Some("Doc 1".to_string()),
                doc_type: None,
                score: 0.9,
                rerank_score: Some(0.7),
                chunks: vec![],
                summary: None,
                content: None,
                metadata: HashMap::new(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            },
            DocumentSearchResult {
                document_id: "doc2".to_string(),
                title: Some("Doc 2".to_string()),
                doc_type: None,
                score: 0.8,
                rerank_score: Some(0.95),
                chunks: vec![],
                summary: None,
                content: None,
                metadata: HashMap::new(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            },
            DocumentSearchResult {
                document_id: "doc3".to_string(),
                title: Some("Doc 3".to_string()),
                doc_type: None,
                score: 0.85,
                rerank_score: None,
                chunks: vec![],
                summary: None,
                content: None,
                metadata: HashMap::new(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            },
        ];

        results.sort_by(|a, b| {
            let a_score = a.rerank_score.unwrap_or(a.score);
            let b_score = b.rerank_score.unwrap_or(b.score);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        assert_eq!(results[0].document_id, "doc2");
        assert_eq!(results[1].document_id, "doc3");
        assert_eq!(results[2].document_id, "doc1");
    }

    #[test]
    fn test_rerank_level_determination() {
        assert!(determine_rerank_level("chunk", 10));
        assert!(!determine_rerank_level("document", 10));
        assert!(determine_rerank_level("auto", 10));
        assert!(!determine_rerank_level("auto", 25));
        assert!(determine_rerank_level("invalid", 10));
    }

    fn determine_rerank_level(level: &str, total_chunks: usize) -> bool {
        match level {
            "chunk" => true,
            "document" => false,
            "auto" => total_chunks < 20,
            _ => total_chunks < 20,
        }
    }

    #[test]
    fn test_search_memory_similarity_uses_db_score_for_fact() {
        let ranker = TemporalSearchRanker::default();
        let mut memory = Memory::new(
            "memory-id".to_string(),
            "Fact memory".to_string(),
            "space-id".to_string(),
        );
        memory.memory_type = crate::models::MemoryType::Fact;

        let db_similarity = 0.73;
        let similarity = apply_memory_similarity(&ranker, &memory, db_similarity);

        assert!((similarity - db_similarity).abs() < 0.0001);
    }

    #[test]
    fn test_search_memory_similarity_applies_episode_decay() {
        let ranker = TemporalSearchRanker::default();
        let mut memory = Memory::new(
            "memory-id".to_string(),
            "Episode memory".to_string(),
            "space-id".to_string(),
        );
        memory.memory_type = crate::models::MemoryType::Episode;
        memory.last_accessed = Some(chrono::Utc::now() - chrono::Duration::days(60));

        let db_similarity = 0.9;
        let similarity = apply_memory_similarity(&ranker, &memory, db_similarity);

        assert!(similarity < db_similarity);
        assert!(similarity > 0.0);
    }

    #[test]
    fn test_search_memory_similarity_uses_db_score_for_preference() {
        let ranker = TemporalSearchRanker::default();
        let mut memory = Memory::new(
            "memory-id".to_string(),
            "Preference memory".to_string(),
            "space-id".to_string(),
        );
        memory.memory_type = crate::models::MemoryType::Preference;

        let db_similarity = 0.64;
        let similarity = apply_memory_similarity(&ranker, &memory, db_similarity);

        assert!((similarity - db_similarity).abs() < 0.0001);
    }

    #[test]
    fn test_chunk_search_result_with_rerank_score() {
        let chunk = ChunkSearchResult {
            content: "test content".to_string(),
            score: 0.85,
            rerank_score: Some(0.92),
            is_relevant: true,
        };

        assert_eq!(chunk.score, 0.85);
        assert_eq!(chunk.rerank_score, Some(0.92));
    }

    #[test]
    fn test_chunk_search_result_without_rerank_score() {
        let chunk = ChunkSearchResult {
            content: "test content".to_string(),
            score: 0.75,
            rerank_score: None,
            is_relevant: true,
        };

        assert_eq!(chunk.score, 0.75);
        assert!(chunk.rerank_score.is_none());
    }

    #[test]
    fn test_memory_search_result_rerank_score_sorting() {
        let mut results = vec![
            MemorySearchResult {
                id: "mem1".to_string(),
                memory: Some("Memory 1".to_string()),
                chunk: None,
                metadata: HashMap::new(),
                similarity: 0.9,
                rerank_score: Some(0.7),
                version: Some(1),
                updated_at: chrono::Utc::now(),
                context: None,
                documents: None,
            },
            MemorySearchResult {
                id: "mem2".to_string(),
                memory: Some("Memory 2".to_string()),
                chunk: None,
                metadata: HashMap::new(),
                similarity: 0.8,
                rerank_score: Some(0.95),
                version: Some(1),
                updated_at: chrono::Utc::now(),
                context: None,
                documents: None,
            },
            MemorySearchResult {
                id: "mem3".to_string(),
                memory: Some("Memory 3".to_string()),
                chunk: None,
                metadata: HashMap::new(),
                similarity: 0.85,
                rerank_score: None,
                version: Some(1),
                updated_at: chrono::Utc::now(),
                context: None,
                documents: None,
            },
        ];

        results.sort_by(|a, b| {
            let a_score = a.rerank_score.unwrap_or(a.similarity);
            let b_score = b.rerank_score.unwrap_or(b.similarity);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        assert_eq!(results[0].id, "mem2");
        assert_eq!(results[1].id, "mem3");
        assert_eq!(results[2].id, "mem1");
    }

    #[test]
    fn test_memory_search_result_with_rerank_score() {
        let result = MemorySearchResult {
            id: "mem123".to_string(),
            memory: Some("Test memory".to_string()),
            chunk: None,
            metadata: HashMap::new(),
            similarity: 0.88,
            rerank_score: Some(0.95),
            version: Some(1),
            updated_at: chrono::Utc::now(),
            context: None,
            documents: None,
        };

        assert_eq!(result.similarity, 0.88);
        assert_eq!(result.rerank_score, Some(0.95));
    }

    #[test]
    fn test_memory_search_result_without_rerank_score() {
        let result = MemorySearchResult {
            id: "mem456".to_string(),
            memory: Some("Another memory".to_string()),
            chunk: None,
            metadata: HashMap::new(),
            similarity: 0.72,
            rerank_score: None,
            version: Some(2),
            updated_at: chrono::Utc::now(),
            context: None,
            documents: None,
        };

        assert_eq!(result.similarity, 0.72);
        assert!(result.rerank_score.is_none());
    }

    async fn setup_hybrid_db() -> (
        Arc<dyn DatabaseBackend>,
        libsql::Connection,
        tempfile::TempDir,
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("hybrid.db");
        let config = crate::config::DatabaseConfig {
            url: format!("file:{}", db_path.display()),
            auth_token: None,
            local_path: None,
        };
        let db = Database::new(&config).await.unwrap();
        let conn = db.connect().unwrap();
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db));
        (backend, conn, temp_dir)
    }

    async fn test_embeddings_provider() -> (EmbeddingProvider, MockServer) {
        let mock_server = MockServer::start().await;

        let mut embedding = vec![0.0f32; 384];
        embedding[0] = 1.0;

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

        let config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 2,
        };

        let provider =
            EmbeddingProvider::new(&config).expect("failed to create embeddings provider");

        (provider, mock_server)
    }

    async fn insert_document_with_chunks_real(
        conn: &libsql::Connection,
        doc_id: &str,
        chunk_contents: &[&str],
        embeddings: &EmbeddingProvider,
    ) {
        let now = Utc::now();
        let doc = Document {
            id: doc_id.to_string(),
            custom_id: None,
            connection_id: None,
            title: Some("Doc".to_string()),
            content: Some("Document content".to_string()),
            summary: None,
            url: None,
            source: None,
            doc_type: crate::models::DocumentType::Text,
            status: ProcessingStatus::Done,
            metadata: HashMap::new(),
            container_tags: vec!["space".to_string()],
            chunk_count: chunk_contents.len() as i32,
            token_count: None,
            word_count: None,
            error_message: None,
            created_at: now,
            updated_at: now,
        };

        DocumentRepository::create(conn, &doc).await.unwrap();

        for (index, content) in chunk_contents.iter().enumerate() {
            let chunk = crate::models::Chunk {
                id: format!("chunk_{doc_id}_{index}"),
                document_id: doc_id.to_string(),
                content: content.to_string(),
                embedded_content: None,
                position: index as i32,
                token_count: None,
                created_at: now,
            };

            ChunkRepository::create(conn, &chunk).await.unwrap();

            let embedding = embeddings.embed_passage(content).await.unwrap();
            ChunkRepository::update_embedding(conn, &chunk.id, &embedding)
                .await
                .unwrap();
        }
    }

    async fn insert_memory_real(
        conn: &libsql::Connection,
        memory_id: &str,
        container_tag: Option<&str>,
        updated_at: DateTime<Utc>,
        embeddings: &EmbeddingProvider,
    ) -> Memory {
        let mut memory = Memory::new(
            memory_id.to_string(),
            format!("Memory {memory_id}"),
            "default".to_string(),
        );
        memory.container_tag = container_tag.map(|tag| tag.to_string());
        memory.memory_type = MemoryType::Fact;
        memory.updated_at = updated_at;

        MemoryRepository::create(conn, &memory).await.unwrap();

        let embedding = embeddings.embed_passage(&memory.memory).await.unwrap();
        MemoryRepository::update_embedding(conn, &memory.id, &embedding)
            .await
            .unwrap();

        memory
    }

    #[tokio::test]
    async fn test_search_hybrid_updates_last_accessed_for_returned_episode_memories() {
        let (db, conn, _temp_dir) = setup_hybrid_db().await;
        let (embeddings, _mock_server) = test_embeddings_provider().await;

        insert_document_with_chunks_real(&conn, "doc1", &["chunk one"], &embeddings).await;

        // Create an episode memory that should be updated
        let mut episode = Memory::new(
            "ep_mem".to_string(),
            "Episode memory".to_string(),
            "default".to_string(),
        );
        episode.memory_type = MemoryType::Episode;
        episode.container_tag = Some("space".to_string());
        MemoryRepository::create(&conn, &episode).await.unwrap();
        let ep_embedding = embeddings.embed_passage(&episode.memory).await.unwrap();
        MemoryRepository::update_embedding(&conn, &episode.id, &ep_embedding)
            .await
            .unwrap();

        // Create a fact memory that should NOT be updated
        let mut fact = Memory::new(
            "fact_mem".to_string(),
            "Fact memory".to_string(),
            "default".to_string(),
        );
        fact.memory_type = MemoryType::Fact;
        fact.container_tag = Some("space".to_string());
        MemoryRepository::create(&conn, &fact).await.unwrap();
        let fact_embedding = embeddings.embed_passage(&fact.memory).await.unwrap();
        MemoryRepository::update_embedding(&conn, &fact.id, &fact_embedding)
            .await
            .unwrap();

        let service = SearchService::new(
            db.clone(),
            db,
            embeddings,
            None,
            LlmProvider::unavailable("tests"),
            &Config::from_env(),
        );

        let _response = service
            .search_hybrid(HybridSearchRequest {
                q: "query".to_string(),
                container_tag: Some("space".to_string()),
                threshold: Some(0.0),
                filters: None,
                include: None,
                limit: Some(10),
                rerank: Some(false),
                rewrite_query: Some(false),
                search_mode: SearchMode::Hybrid,
            })
            .await
            .unwrap();

        // Episode memory should have last_accessed set
        let fetched_episode = MemoryRepository::get_by_id(&conn, "ep_mem")
            .await
            .unwrap()
            .unwrap();
        assert!(fetched_episode.last_accessed.is_some());

        // Fact memory should not have last_accessed set
        let fetched_fact = MemoryRepository::get_by_id(&conn, "fact_mem")
            .await
            .unwrap()
            .unwrap();
        assert!(fetched_fact.last_accessed.is_none());
    }

    #[tokio::test]
    async fn test_search_hybrid_returns_both_types() {
        let (db, conn, _temp_dir) = setup_hybrid_db().await;

        let (embeddings, _mock_server) = test_embeddings_provider().await;

        insert_document_with_chunks_real(&conn, "doc1", &["chunk one"], &embeddings).await;
        insert_memory_real(&conn, "mem1", Some("space"), Utc::now(), &embeddings).await;

        let service = SearchService::new(
            db.clone(),
            db,
            embeddings,
            None,
            LlmProvider::unavailable("tests"),
            &Config::from_env(),
        );

        let response = service
            .search_hybrid(HybridSearchRequest {
                q: "query".to_string(),
                container_tag: Some("space".to_string()),
                threshold: Some(0.0),
                filters: None,
                include: None,
                limit: Some(10),
                rerank: Some(false),
                rewrite_query: Some(false),
                search_mode: SearchMode::Hybrid,
            })
            .await
            .unwrap();

        assert!(response.results.iter().any(|r| r.memory.is_some()));
        assert!(response.results.iter().any(|r| r.chunk.is_some()));
    }

    #[tokio::test]
    async fn test_search_documents_reads_from_read_backend() {
        let (read_db, read_conn, _read_temp_dir) = setup_hybrid_db().await;
        let (write_db, _write_conn, _write_temp_dir) = setup_hybrid_db().await;
        let (embeddings, _mock_server) = test_embeddings_provider().await;

        insert_document_with_chunks_real(
            &read_conn,
            "doc_read_backend",
            &["read backend chunk"],
            &embeddings,
        )
        .await;

        let service = SearchService::new(
            read_db,
            write_db,
            embeddings,
            None,
            LlmProvider::unavailable("tests"),
            &Config::from_env(),
        );

        let response = service
            .search_documents(SearchDocumentsRequest {
                q: "read backend chunk".to_string(),
                container_tags: None,
                chunk_threshold: Some(0.0),
                document_threshold: None,
                doc_id: None,
                filters: None,
                include_full_docs: Some(false),
                include_summary: Some(false),
                limit: Some(5),
                only_matching_chunks: Some(false),
                rerank: Some(false),
                rerank_level: None,
                rerank_top_k: None,
                rewrite_query: Some(false),
            })
            .await
            .unwrap();

        assert!(!response.results.is_empty());
        assert_eq!(response.results[0].document_id, "doc_read_backend");
    }

    #[tokio::test]
    async fn test_search_hybrid_deduplicates_document_chunks_when_memory_sources_exist() {
        let (db, conn, _temp_dir) = setup_hybrid_db().await;
        let (embeddings, _mock_server) = test_embeddings_provider().await;

        insert_document_with_chunks_real(&conn, "doc1", &["chunk one"], &embeddings).await;
        insert_memory_real(&conn, "mem1", Some("space"), Utc::now(), &embeddings).await;

        MemorySourcesRepository::create(&conn, "mem1", "doc1", None)
            .await
            .unwrap();
        let service = SearchService::new(
            db.clone(),
            db,
            embeddings,
            None,
            LlmProvider::unavailable("tests"),
            &Config::from_env(),
        );
        let response = service
            .search_hybrid(HybridSearchRequest {
                q: "query".to_string(),
                container_tag: Some("space".to_string()),
                threshold: Some(0.0),
                filters: None,
                include: None,
                limit: Some(10),
                rerank: Some(false),
                rewrite_query: Some(false),
                search_mode: SearchMode::Hybrid,
            })
            .await
            .unwrap();

        assert!(response.results.iter().any(|r| r.memory.is_some()));
        assert!(response.results.iter().all(|r| r.chunk.is_none()));
    }

    #[tokio::test]
    async fn test_search_hybrid_respects_limit() {
        let (db, conn, _temp_dir) = setup_hybrid_db().await;
        let (embeddings, _mock_server) = test_embeddings_provider().await;

        insert_document_with_chunks_real(&conn, "doc1", &["chunk one", "chunk two"], &embeddings)
            .await;
        insert_memory_real(&conn, "mem1", Some("space"), Utc::now(), &embeddings).await;
        insert_memory_real(&conn, "mem2", Some("space"), Utc::now(), &embeddings).await;
        let service = SearchService::new(
            db.clone(),
            db,
            embeddings,
            None,
            LlmProvider::unavailable("tests"),
            &Config::from_env(),
        );
        let response = service
            .search_hybrid(HybridSearchRequest {
                q: "query".to_string(),
                container_tag: Some("space".to_string()),
                threshold: Some(0.0),
                filters: None,
                include: None,
                limit: Some(2),
                rerank: Some(false),
                rewrite_query: Some(false),
                search_mode: SearchMode::Hybrid,
            })
            .await
            .unwrap();

        assert_eq!(response.results.len(), 2);
    }

    #[tokio::test]
    async fn test_search_hybrid_reranking_applies_to_memories() {
        let (db, conn, _temp_dir) = setup_hybrid_db().await;
        let (embeddings, _mock_server) = test_embeddings_provider().await;

        insert_memory_real(&conn, "mem1", Some("space"), Utc::now(), &embeddings).await;
        insert_memory_real(&conn, "mem2", Some("space"), Utc::now(), &embeddings).await;

        let reranker = RerankerProvider::new_mock(vec![
            RerankResult {
                document: "Memory mem2".to_string(),
                score: 0.95,
                index: 1,
            },
            RerankResult {
                document: "Memory mem1".to_string(),
                score: 0.4,
                index: 0,
            },
        ]);

        let service = SearchService::new(
            db.clone(),
            db,
            embeddings,
            Some(reranker),
            LlmProvider::unavailable("tests"),
            &Config::from_env(),
        );
        let response = service
            .search_hybrid(HybridSearchRequest {
                q: "query".to_string(),
                container_tag: Some("space".to_string()),
                threshold: Some(0.0),
                filters: None,
                include: None,
                limit: Some(10),
                rerank: Some(true),
                rewrite_query: Some(false),
                search_mode: SearchMode::Memories,
            })
            .await
            .unwrap();

        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].id, "mem2");
        assert_eq!(response.results[0].rerank_score, Some(0.95));
    }

    #[tokio::test]
    async fn test_search_hybrid_partial_failure_returns_other_domain() {
        let (db, conn, _temp_dir) = setup_hybrid_db().await;
        let (embeddings, _mock_server) = test_embeddings_provider().await;

        insert_memory_real(&conn, "mem1", Some("space"), Utc::now(), &embeddings).await;

        conn.execute("PRAGMA foreign_keys = OFF", ()).await.unwrap();
        conn.execute("DROP TABLE IF EXISTS chunks", ())
            .await
            .unwrap();

        let service = SearchService::new(
            db.clone(),
            db,
            embeddings,
            None,
            LlmProvider::unavailable("tests"),
            &Config::from_env(),
        );
        let response = service
            .search_hybrid(HybridSearchRequest {
                q: "query".to_string(),
                container_tag: Some("space".to_string()),
                threshold: Some(0.0),
                filters: None,
                include: None,
                limit: Some(10),
                rerank: Some(false),
                rewrite_query: Some(false),
                search_mode: SearchMode::Hybrid,
            })
            .await
            .unwrap();

        assert_eq!(response.results.len(), 1);
        assert!(response.results[0].memory.is_some());
    }
}
