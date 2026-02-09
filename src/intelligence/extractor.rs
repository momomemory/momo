use crate::db::DatabaseBackend;
use crate::embeddings::EmbeddingProvider;
use crate::error::{MomoError, Result};
use crate::intelligence::contradiction::{ContradictionCheckResult, ContradictionDetector};
use crate::llm::{prompts, LlmProvider};
use crate::models::ConversationMessage;

use super::types::{ExtractedMemory, ExtractionResult};

/// Wrapper struct for parsing OpenAI JSON responses
#[derive(Debug, Clone, serde::Deserialize)]
struct MemoriesWrapper {
    memories: Vec<ExtractedMemory>,
}

pub struct MemoryExtractor {
    llm: LlmProvider,
    embeddings: EmbeddingProvider,
}

impl Clone for MemoryExtractor {
    fn clone(&self) -> Self {
        Self {
            llm: self.llm.clone(),
            embeddings: self.embeddings.clone(),
        }
    }
}

impl MemoryExtractor {
    pub fn new(llm: LlmProvider, embeddings: EmbeddingProvider) -> Self {
        Self { llm, embeddings }
    }

    pub async fn extract(&self, content: &str) -> Result<ExtractionResult> {
        tracing::trace!(
            embedding_dimensions = self.embeddings.dimensions(),
            "Extracting memories"
        );

        if !self.llm.is_available() {
            tracing::warn!("LLM unavailable, skipping memory extraction");
            return Ok(empty_result(content.to_string()));
        }

        let prompt = prompts::memory_extraction_prompt(content);
        match self
            .llm
            .complete_structured::<MemoriesWrapper>(&prompt)
            .await
        {
            Ok(wrapper) => Ok(ExtractionResult {
                memories: wrapper.memories,
                source_content: content.to_string(),
            }),
            Err(MomoError::LlmUnavailable(reason)) => {
                tracing::warn!(%reason, "LLM unavailable during extraction");
                Ok(empty_result(content.to_string()))
            }
            Err(error) => {
                // LLM may return valid JSON without memories field when nothing to extract
                // This is expected behavior, not an error condition
                tracing::debug!(error = %error, "LLM returned non-conforming JSON, returning empty result");
                Ok(empty_result(content.to_string()))
            }
        }
    }

    pub async fn extract_from_conversation(
        &self,
        messages: &[ConversationMessage],
    ) -> Result<ExtractionResult> {
        tracing::trace!(
            embedding_dimensions = self.embeddings.dimensions(),
            "Extracting memories from conversation"
        );

        let source_content = conversation_source(messages);

        if !self.llm.is_available() {
            tracing::warn!("LLM unavailable, skipping conversation memory extraction");
            return Ok(empty_result(source_content));
        }

        let prompt = prompts::conversation_extraction_prompt(messages);
        match self
            .llm
            .complete_structured::<MemoriesWrapper>(&prompt)
            .await
        {
            Ok(wrapper) => Ok(ExtractionResult {
                memories: wrapper.memories,
                source_content,
            }),
            Err(MomoError::LlmUnavailable(reason)) => {
                tracing::warn!(%reason, "LLM unavailable during conversation extraction");
                Ok(empty_result(source_content))
            }
            Err(error) => {
                tracing::debug!(error = %error, "LLM returned non-conforming JSON, returning empty result");
                Ok(empty_result(source_content))
            }
        }
    }

    /// Check extracted memories for contradictions against existing memories in the container.
    ///
    /// For each memory, embeds it, searches for similar existing memories in the same container,
    /// and runs the heuristic `ContradictionDetector` against each match. If a `Likely`
    /// contradiction is found, sets `potential_contradiction = true` on the memory.
    ///
    /// This method never blocks memory creation — it only flags.
    pub async fn check_contradictions(
        &self,
        mut memories: Vec<ExtractedMemory>,
        container_tag: &str,
        db: &dyn DatabaseBackend,
    ) -> Result<Vec<ExtractedMemory>> {
        let detector = ContradictionDetector::new();

        for memory in &mut memories {
            let embedding = match self.embeddings.embed_passage(&memory.content).await {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!(error = %e, "Failed to embed memory for contradiction check");
                    continue;
                }
            };

            let similar = db
                .search_similar_memories(&embedding, 5, 0.6, Some(container_tag), false)
                .await?;

            for hit in &similar {
                let result = detector.check_contradiction(&hit.memory.memory, &memory.content);
                if result == ContradictionCheckResult::Likely {
                    memory.potential_contradiction = true;
                    tracing::info!(
                        new_memory = %memory.content,
                        existing_memory_id = %hit.memory.id,
                        existing_memory = %hit.memory.memory,
                        "Potential contradiction detected (heuristic)"
                    );
                    break;
                }
            }
        }

        Ok(memories)
    }

    pub async fn deduplicate(
        &self,
        memories: Vec<ExtractedMemory>,
        container_tag: &str,
        db: &dyn DatabaseBackend,
    ) -> Result<Vec<ExtractedMemory>> {
        let mut result = Vec::new();

        for memory in memories {
            let embedding = match self.embeddings.embed_passage(&memory.content).await {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!(error = %e, "Failed to embed memory for deduplication");
                    continue;
                }
            };

            let similar = db
                .search_similar_memories(
                    &embedding,
                    1,
                    0.9,
                    Some(container_tag),
                    false, // exclude forgotten memories for deduplication
                )
                .await?;

            if let Some(existing) = similar.first().map(|hit| &hit.memory) {
                db.update_memory_source_count(&existing.id, existing.source_count + 1)
                    .await?;
                tracing::debug!(memory_id = %existing.id, "Found duplicate, incremented source_count");
            } else {
                result.push(memory);
            }
        }

        Ok(result)
    }
}

fn empty_result(source_content: String) -> ExtractionResult {
    ExtractionResult {
        memories: Vec::new(),
        source_content,
    }
}

fn conversation_source(messages: &[ConversationMessage]) -> String {
    messages
        .iter()
        .map(|message| format!("[{}]: {}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::config::{EmbeddingsConfig, LlmConfig};

    async fn test_embeddings_provider() -> EmbeddingProvider {
        let config = EmbeddingsConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            batch_size: 8,
        };

        EmbeddingProvider::new(&config).expect("failed to create test embeddings provider")
    }

    fn test_llm_unavailable() -> LlmProvider {
        LlmProvider::unavailable("test unavailable")
    }

    fn test_llm_provider(base_url: String) -> LlmProvider {
        let config = LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(base_url),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,
            filter_prompt: None,
        };

        LlmProvider::new(Some(&config))
    }

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
    async fn extract_returns_empty_result_when_llm_unavailable() {
        let embeddings = test_embeddings_provider().await;
        let extractor = MemoryExtractor::new(test_llm_unavailable(), embeddings);

        let source = "User prefers dark mode";
        let result = extractor
            .extract(source)
            .await
            .expect("extract should not fail");

        assert!(result.memories.is_empty());
        assert_eq!(result.source_content, source);
    }

    #[tokio::test]
    async fn extract_returns_memories_for_valid_llm_json() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"memories":[{"content":"User prefers dark mode","memory_type":"preference","confidence":0.9}]}"#,
            )))
            .mount(&llm_server)
            .await;

        let embeddings = test_embeddings_provider().await;
        let extractor = MemoryExtractor::new(test_llm_provider(llm_server.uri()), embeddings);

        let result = extractor
            .extract("User prefers dark mode")
            .await
            .expect("extract should not fail");

        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.memories[0].content, "User prefers dark mode");
        assert_eq!(result.memories[0].memory_type, "preference");
        assert_eq!(result.memories[0].confidence, 0.9);
        assert_eq!(result.source_content, "User prefers dark mode");
    }

    #[tokio::test]
    async fn extract_returns_empty_result_for_malformed_llm_json() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(llm_response(r#"{"unexpected":"shape"}"#)),
            )
            .mount(&llm_server)
            .await;

        let embeddings = test_embeddings_provider().await;
        let extractor = MemoryExtractor::new(test_llm_provider(llm_server.uri()), embeddings);

        let source = "User prefers dark mode";
        let result = extractor
            .extract(source)
            .await
            .expect("extract should not fail");

        assert!(result.memories.is_empty());
        assert_eq!(result.source_content, source);
    }

    #[tokio::test]
    async fn extract_from_conversation_uses_messages_as_source_content() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(llm_response(r#"{"memories":[]}"#)),
            )
            .mount(&llm_server)
            .await;

        let embeddings = test_embeddings_provider().await;
        let extractor = MemoryExtractor::new(test_llm_provider(llm_server.uri()), embeddings);

        let messages = vec![
            ConversationMessage {
                role: "user".to_string(),
                content: "I prefer dark mode".to_string(),
                timestamp: Some(Utc::now()),
            },
            ConversationMessage {
                role: "assistant".to_string(),
                content: "Got it".to_string(),
                timestamp: Some(Utc::now()),
            },
        ];

        let result = extractor
            .extract_from_conversation(&messages)
            .await
            .expect("extract_from_conversation should not fail");

        assert_eq!(
            result.source_content,
            "[user]: I prefer dark mode\n[assistant]: Got it"
        );
    }
}
