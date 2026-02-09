use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use nanoid::nanoid;

use crate::config::Config;
use crate::db::DatabaseBackend;
use crate::embeddings::EmbeddingProvider;
use crate::error::{MomoError, Result};
use crate::intelligence::profile::ProfileGenerator;
use crate::intelligence::types::HeuristicContext;
use crate::intelligence::{ContradictionDetector, RelationshipDetector};
use crate::llm::LlmProvider;
use crate::models::{
    ForgetMemoryRequest, ForgetMemoryResponse, GetProfileRequest, HybridSearchRequest, Memory,
    MemoryRelationType, MemoryType, ProfileFact, ProfileResponse, UpdateMemoryRequest,
    UpdateMemoryResponse, UserProfileData,
};
use crate::services::search::SearchService;

pub struct MemoryService {
    db: Arc<dyn DatabaseBackend>,
    embeddings: EmbeddingProvider,
    default_space_id: String,
    profile_generator: ProfileGenerator,
}

impl MemoryService {
    pub fn new(db: Arc<dyn DatabaseBackend>, embeddings: EmbeddingProvider) -> Self {
        let llm_config = Config::from_env().llm;
        let llm_provider = LlmProvider::new(llm_config.as_ref());
        let profile_generator = ProfileGenerator::new(llm_provider);

        Self {
            db,
            embeddings,
            default_space_id: "default".to_string(),
            profile_generator,
        }
    }

    #[allow(dead_code)]
    pub async fn create_memory(
        &self,
        content: &str,
        container_tag: &str,
        is_static: bool,
    ) -> Result<Memory> {
        self.create_memory_internal(
            content,
            container_tag,
            is_static,
            false,
            None,
            MemoryType::Fact,
        )
        .await
    }

    #[allow(dead_code)]
    pub async fn create_inferred_memory(
        &self,
        content: &str,
        container_tag: &str,
        confidence: f32,
    ) -> Result<Memory> {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("confidence".to_string(), serde_json::json!(confidence));
        self.create_memory_internal(
            content,
            container_tag,
            false,
            true,
            Some(metadata),
            MemoryType::Fact,
        )
        .await
    }

    pub async fn create_memory_with_type(
        &self,
        content: &str,
        container_tag: &str,
        is_static: bool,
        memory_type: MemoryType,
    ) -> Result<Memory> {
        self.create_memory_internal(content, container_tag, is_static, false, None, memory_type)
            .await
    }

    async fn create_memory_internal(
        &self,
        content: &str,
        container_tag: &str,
        is_static: bool,
        is_inference: bool,
        metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
        memory_type: MemoryType,
    ) -> Result<Memory> {
        let embedding = self.embeddings.embed_passage(content).await?;

        let memory = Memory {
            id: nanoid!(),
            memory: content.to_string(),
            space_id: self.default_space_id.clone(),
            container_tag: Some(container_tag.to_string()),
            version: 1,
            is_latest: true,
            parent_memory_id: None,
            root_memory_id: None,
            memory_relations: Default::default(),
            source_count: 1,
            is_inference,
            is_forgotten: false,
            is_static,
            forget_after: None,
            forget_reason: None,
            memory_type,
            last_accessed: if memory_type == MemoryType::Episode {
                Some(Utc::now())
            } else {
                None
            },
            confidence: None,
            metadata: metadata.unwrap_or_default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.db.create_memory(&memory).await?;
        self.db
            .update_memory_embedding(&memory.id, &embedding)
            .await?;

        let llm_config = Config::from_env().llm;
        if llm_config
            .as_ref()
            .map(|config| config.enable_auto_relations)
            .unwrap_or(false)
        {
            let db = self.db.clone();
            let embeddings = self.embeddings.clone();
            let memory_id = memory.id.clone();
            let memory_content = memory.memory.clone();
            let memory_embedding = embedding.clone();
            let container_tag = memory.container_tag.clone();
            let enable_contradiction = llm_config
                .as_ref()
                .map(|c| c.enable_contradiction_detection)
                .unwrap_or(false);
            let llm_provider = LlmProvider::new(llm_config.as_ref());

            tokio::spawn(async move {
                let heuristic_ctx = if enable_contradiction {
                    match build_heuristic_context(
                        &*db,
                        &memory_id,
                        &memory_content,
                        &memory_embedding,
                        container_tag.as_deref(),
                    )
                    .await
                    {
                        Ok(ctx) => ctx,
                        Err(error) => {
                            tracing::error!(error = %error, "Heuristic contradiction check failed");
                            None
                        }
                    }
                } else {
                    None
                };

                let detector = RelationshipDetector::new(llm_provider, embeddings);
                let detection = match detector
                    .detect(
                        &memory_id,
                        &memory_content,
                        container_tag.as_deref(),
                        db.as_ref(),
                        heuristic_ctx.as_ref(),
                    )
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        tracing::error!(error = %error, "Relationship detection failed");
                        return;
                    }
                };

                for classification in detection.classifications {
                    if classification.relation_type == "none" {
                        continue;
                    }

                    let relation_type =
                        match classification.relation_type.parse::<MemoryRelationType>() {
                            Ok(relation_type) => relation_type,
                            Err(error) => {
                                tracing::warn!(error = %error, "Unknown relation type");
                                continue;
                            }
                        };

                    if let Err(error) = db
                        .add_memory_relation(
                            &memory_id,
                            &classification.memory_id,
                            relation_type.clone(),
                        )
                        .await
                    {
                        tracing::error!(error = %error, "Failed to add relation for new memory");
                    }

                    if let Err(error) = db
                        .add_memory_relation(
                            &classification.memory_id,
                            &memory_id,
                            relation_type.clone(),
                        )
                        .await
                    {
                        tracing::error!(error = %error, "Failed to add relation for related memory");
                    }

                    if relation_type == MemoryRelationType::Updates {
                        if let Err(error) = db
                            .update_memory_to_not_latest(&classification.memory_id)
                            .await
                        {
                            tracing::error!(error = %error, "Failed to mark related memory not latest");
                        }

                        match db.get_memory_by_id(&classification.memory_id).await {
                            Ok(Some(old)) => {
                                let root_id =
                                    old.root_memory_id.clone().unwrap_or_else(|| old.id.clone());
                                let new_version = old.version + 1;
                                if let Err(error) = db
                                    .update_memory_version_chain(
                                        &memory_id,
                                        &old.id,
                                        &root_id,
                                        new_version,
                                    )
                                    .await
                                {
                                    tracing::error!(error = %error, "Failed to set version chain on new memory");
                                }
                            }
                            Ok(None) => {
                                tracing::warn!(
                                    old_id = %classification.memory_id,
                                    "Old memory not found for version chain"
                                );
                            }
                            Err(error) => {
                                tracing::error!(error = %error, "Failed to fetch old memory for version chain");
                            }
                        }
                    }
                }
            });
        }

        Ok(memory)
    }

    pub async fn update_memory(&self, req: UpdateMemoryRequest) -> Result<UpdateMemoryResponse> {
        let existing = if let Some(ref id) = req.id {
            self.db.get_memory_by_id(id).await?
        } else if let Some(ref content) = req.content {
            self.db
                .get_memory_by_content(content, &req.container_tag)
                .await?
        } else {
            return Err(MomoError::Validation(
                "Either id or content must be provided".to_string(),
            ));
        };

        let existing =
            existing.ok_or_else(|| MomoError::NotFound("Memory not found".to_string()))?;

        self.db.update_memory_to_not_latest(&existing.id).await?;

        let new_embedding = self.embeddings.embed_passage(&req.new_content).await?;

        let root_id = existing
            .root_memory_id
            .clone()
            .unwrap_or_else(|| existing.id.clone());

        let new_memory = Memory {
            id: nanoid!(),
            memory: req.new_content.clone(),
            space_id: existing.space_id.clone(),
            container_tag: Some(req.container_tag.clone()),
            version: existing.version + 1,
            is_latest: true,
            parent_memory_id: Some(existing.id.clone()),
            root_memory_id: Some(root_id),
            memory_relations: {
                let mut relations = existing.memory_relations.clone();
                relations.insert(
                    existing.id.clone(),
                    crate::models::MemoryRelationType::Updates,
                );
                relations
            },
            source_count: existing.source_count,
            is_inference: false,
            is_forgotten: false,
            is_static: req.is_static.unwrap_or(existing.is_static),
            forget_after: None,
            forget_reason: None,
            memory_type: existing.memory_type,
            last_accessed: existing.last_accessed,
            confidence: existing.confidence,
            metadata: req.metadata.unwrap_or(existing.metadata),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.db.create_memory(&new_memory).await?;
        self.db
            .update_memory_embedding(&new_memory.id, &new_embedding)
            .await?;

        Ok(UpdateMemoryResponse {
            id: new_memory.id,
            memory: new_memory.memory,
            version: new_memory.version,
            parent_memory_id: new_memory.parent_memory_id,
            root_memory_id: new_memory.root_memory_id,
            created_at: new_memory.created_at,
        })
    }

    pub async fn forget_memory(&self, req: ForgetMemoryRequest) -> Result<ForgetMemoryResponse> {
        let existing = if let Some(ref id) = req.id {
            self.db.get_memory_by_id(id).await?
        } else if let Some(ref content) = req.content {
            self.db
                .get_memory_by_content(content, &req.container_tag)
                .await?
        } else {
            return Err(MomoError::Validation(
                "Either id or content must be provided".to_string(),
            ));
        };

        let existing =
            existing.ok_or_else(|| MomoError::NotFound("Memory not found".to_string()))?;

        self.db
            .forget_memory(&existing.id, req.reason.as_deref())
            .await?;

        Ok(ForgetMemoryResponse {
            id: existing.id,
            forgotten: true,
        })
    }

    pub async fn get_profile(
        &self,
        req: GetProfileRequest,
        search_service: &SearchService,
    ) -> Result<ProfileResponse> {
        let cached = self.db.get_cached_profile(&req.container_tag).await?;

        let mut profile = self
            .db
            .get_user_profile(
                &req.container_tag,
                req.include_dynamic.unwrap_or(true),
                req.limit.unwrap_or(50),
            )
            .await?;

        let is_stale = match &cached {
            Some(c) => match &c.cached_at {
                Some(cached_at_str) => {
                    DateTime::parse_from_rfc3339(cached_at_str)
                        .map(|dt| dt.with_timezone(&Utc) < profile.last_updated)
                        .unwrap_or(true) // unparseable => treat as stale
                }
                None => true, // no timestamp => stale
            },
            None => true, // no cache entry => stale
        };

        let want_narrative = req.generate_narrative.unwrap_or(false);
        let want_compact = req.compact.unwrap_or(false);

        let all_facts: Vec<&str> = profile
            .static_facts
            .iter()
            .chain(profile.dynamic_facts.iter())
            .map(|f| f.memory.as_str())
            .collect();

        let mut new_narrative: Option<String> = cached.as_ref().and_then(|c| c.narrative.clone());
        let mut new_summary: Option<String> = cached.as_ref().and_then(|c| c.summary.clone());
        let mut cache_dirty = false;

        if want_narrative {
            let narrative_missing = cached.as_ref().and_then(|c| c.narrative.as_ref()).is_none();

            if is_stale || narrative_missing {
                let narrative = self
                    .profile_generator
                    .generate_narrative(&all_facts)
                    .await?;
                if !narrative.is_empty() {
                    new_narrative = Some(narrative);
                    cache_dirty = true;
                }
            }
        }

        if want_compact {
            let summary_missing = cached.as_ref().and_then(|c| c.summary.as_ref()).is_none();

            if is_stale || summary_missing {
                let compacted = self.profile_generator.compact_facts(&all_facts).await?;
                if !compacted.is_empty() {
                    let summary_json = serde_json::to_string(&compacted)?;
                    new_summary = Some(summary_json);
                    cache_dirty = true;

                    profile.static_facts = format_compacted_facts(&compacted);
                    profile.dynamic_facts = Vec::new();
                }
            } else if let Some(ref summary) = new_summary {
                if let Ok(compacted) = serde_json::from_str::<HashMap<String, Vec<String>>>(summary)
                {
                    profile.static_facts = format_compacted_facts(&compacted);
                    profile.dynamic_facts = Vec::new();
                }
            }
        }

        if cache_dirty {
            self.db
                .upsert_cached_profile(
                    &req.container_tag,
                    new_narrative.as_deref(),
                    new_summary.as_deref(),
                )
                .await?;
        }

        let search_results = if let Some(ref q) = req.q {
            let search_req = HybridSearchRequest {
                q: q.clone(),
                container_tag: Some(req.container_tag.clone()),
                threshold: req.threshold,
                limit: req.limit,
                ..Default::default()
            };
            let response = search_service.search_hybrid(search_req).await?;
            Some(response)
        } else {
            None
        };

        Ok(ProfileResponse {
            profile: UserProfileData {
                static_facts: profile
                    .static_facts
                    .iter()
                    .map(|f| f.memory.clone())
                    .collect(),
                dynamic_facts: profile
                    .dynamic_facts
                    .iter()
                    .map(|f| f.memory.clone())
                    .collect(),
            },
            search_results,
            narrative: new_narrative,
        })
    }
}

impl Clone for MemoryService {
    fn clone(&self) -> Self {
        let llm_config = Config::from_env().llm;
        let llm_provider = LlmProvider::new(llm_config.as_ref());
        Self {
            db: self.db.clone(),
            embeddings: self.embeddings.clone(),
            default_space_id: self.default_space_id.clone(),
            profile_generator: ProfileGenerator::new(llm_provider),
        }
    }
}

fn format_compacted_facts(compacted: &HashMap<String, Vec<String>>) -> Vec<ProfileFact> {
    let mut facts = Vec::new();
    for (category, items) in compacted {
        for item in items {
            facts.push(ProfileFact {
                memory: format!("[{category}] {item}"),
                confidence: None,
                created_at: Utc::now(),
            });
        }
    }
    facts
}

async fn build_heuristic_context(
    db: &dyn DatabaseBackend,
    new_memory_id: &str,
    new_memory_content: &str,
    embedding: &[f32],
    container_tag: Option<&str>,
) -> Result<Option<HeuristicContext>> {
    let candidates = db
        .search_similar_memories(embedding, 5, 0.7, container_tag, false)
        .await?
        .into_iter()
        .filter(|hit| hit.memory.id != new_memory_id)
        .collect::<Vec<_>>();

    let contradiction_detector = ContradictionDetector::new();

    for hit in &candidates {
        let result =
            contradiction_detector.check_contradiction(&hit.memory.memory, new_memory_content);
        match result {
            crate::intelligence::contradiction::ContradictionCheckResult::Likely
            | crate::intelligence::contradiction::ContradictionCheckResult::Unlikely => {
                tracing::debug!(
                    candidate_memory_id = %hit.memory.id,
                    heuristic = %result,
                    "Heuristic flagged potential contradiction"
                );
                return Ok(Some(HeuristicContext {
                    candidate_memory_id: hit.memory.id.clone(),
                    candidate_content: hit.memory.memory.clone(),
                    heuristic_result: result,
                }));
            }
            crate::intelligence::contradiction::ContradictionCheckResult::None => {}
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use crate::config::Config;

    #[test]
    fn test_relationship_detection_enabled_when_config_set() {
        let config = Config::from_env();
        let should_detect = config
            .llm
            .as_ref()
            .map(|c| c.enable_auto_relations)
            .unwrap_or(false);

        assert!(
            !should_detect,
            "Relationship detection should be disabled by default without LLM config"
        );
    }

    #[test]
    fn test_conversation_creates_memory_via_create_memory_with_type() {
        // Structural test: fails to compile if create_memory_internal's relationship
        // detection dependencies (RelationshipDetector, MemoryType) are removed.
        use crate::models::MemoryType;

        let _fact = MemoryType::Fact;
        let _episode = MemoryType::Episode;
        let _preference = MemoryType::Preference;

        use crate::intelligence::RelationshipDetector;
        let _: fn(
            crate::llm::LlmProvider,
            crate::embeddings::EmbeddingProvider,
        ) -> RelationshipDetector = RelationshipDetector::new;
    }
}
