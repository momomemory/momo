use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;
use crate::llm::prompts::{fact_compaction_prompt, narrative_generation_prompt};
use crate::llm::LlmProvider;

/// Structure to hold the narrative extracted from LLM response
#[derive(Debug, Serialize, Deserialize)]
pub struct ExtractedNarrative {
    pub narrative: String,
}

/// Structure to hold the compacted facts from LLM response
///
/// We use flatten to capture all dynamic category keys into a HashMap
#[derive(Debug, Serialize, Deserialize)]
pub struct CompactedFacts {
    #[serde(flatten)]
    pub categories: HashMap<String, Vec<String>>,
}

/// Service for generating user profiles using LLM
pub struct ProfileGenerator {
    llm: LlmProvider,
}

impl ProfileGenerator {
    /// Create a new ProfileGenerator with the given LLM provider
    pub fn new(llm: LlmProvider) -> Self {
        Self { llm }
    }

    /// Generate a cohesive narrative from a list of memory contents
    ///
    /// # Arguments
    /// * `memories` - Slice of memory content strings
    ///
    /// # Returns
    /// * A 3rd person narrative string
    pub async fn generate_narrative(&self, memories: &[&str]) -> Result<String> {
        if !self.llm.is_available() {
            // Return empty string or specific message if LLM is not configured
            // Ideally the caller should check availability, but we handle it gracefully
            return Ok(String::new());
        }

        if memories.is_empty() {
            return Ok(String::new());
        }

        let prompt = narrative_generation_prompt(memories);
        let response: ExtractedNarrative = self.llm.complete_structured(&prompt).await?;

        Ok(response.narrative)
    }

    /// Compact and categorize a list of facts
    ///
    /// # Arguments
    /// * `facts` - Slice of fact strings
    ///
    /// # Returns
    /// * A map of categories to lists of consolidated facts
    pub async fn compact_facts(&self, facts: &[&str]) -> Result<HashMap<String, Vec<String>>> {
        if !self.llm.is_available() {
            return Ok(HashMap::new());
        }

        if facts.is_empty() {
            return Ok(HashMap::new());
        }

        // If very few facts, maybe skip compaction?
        // For now we process even small lists to get categorization

        let prompt = fact_compaction_prompt(facts);
        // We can deserialize directly into the wrapper struct
        let response: CompactedFacts = self.llm.complete_structured(&prompt).await?;

        Ok(response.categories)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::config::LlmConfig;

    // ── Test helpers ──────────────────────────────────────────────────

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
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        })
    }

    // ── generate_narrative tests ──────────────────────────────────────

    #[tokio::test]
    async fn generate_narrative_returns_empty_when_llm_unavailable() {
        let gen = ProfileGenerator::new(test_llm_unavailable());
        let result = gen
            .generate_narrative(&["User likes Rust"])
            .await
            .expect("should not fail");
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn generate_narrative_returns_empty_for_empty_memories() {
        let llm_server = MockServer::start().await;
        let gen = ProfileGenerator::new(test_llm_provider(llm_server.uri()));
        let result = gen.generate_narrative(&[]).await.expect("should not fail");
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn generate_narrative_returns_narrative_for_single_memory() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"narrative": "Alice is a dedicated Rust programmer."}"#,
            )))
            .mount(&llm_server)
            .await;

        let gen = ProfileGenerator::new(test_llm_provider(llm_server.uri()));
        let result = gen
            .generate_narrative(&["User programs in Rust"])
            .await
            .expect("should not fail");

        assert_eq!(result, "Alice is a dedicated Rust programmer.");
    }

    #[tokio::test]
    async fn generate_narrative_returns_narrative_for_multiple_memories() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"narrative": "The user is a software engineer who loves hiking and prefers dark mode in all editors."}"#,
            )))
            .mount(&llm_server)
            .await;

        let gen = ProfileGenerator::new(test_llm_provider(llm_server.uri()));
        let result = gen
            .generate_narrative(&[
                "User is a software engineer",
                "User loves hiking",
                "User prefers dark mode",
            ])
            .await
            .expect("should not fail");

        assert!(result.contains("software engineer"));
        assert!(result.contains("hiking"));
    }

    #[tokio::test]
    async fn generate_narrative_fails_on_malformed_json() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"not_narrative": "missing expected field"}"#,
            )))
            .mount(&llm_server)
            .await;

        let gen = ProfileGenerator::new(test_llm_provider(llm_server.uri()));
        let result = gen.generate_narrative(&["User likes Rust"]).await;

        assert!(result.is_err());
    }

    // ── compact_facts tests ──────────────────────────────────────────

    #[tokio::test]
    async fn compact_facts_returns_empty_when_llm_unavailable() {
        let gen = ProfileGenerator::new(test_llm_unavailable());
        let result = gen
            .compact_facts(&["User likes Rust"])
            .await
            .expect("should not fail");
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn compact_facts_returns_empty_for_empty_facts() {
        let llm_server = MockServer::start().await;
        let gen = ProfileGenerator::new(test_llm_provider(llm_server.uri()));
        let result = gen.compact_facts(&[]).await.expect("should not fail");
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn compact_facts_returns_categorized_facts_for_single_fact() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"Technical": ["Uses Rust programming language"]}"#,
            )))
            .mount(&llm_server)
            .await;

        let gen = ProfileGenerator::new(test_llm_provider(llm_server.uri()));
        let result = gen
            .compact_facts(&["User uses Rust"])
            .await
            .expect("should not fail");

        assert!(result.contains_key("Technical"));
        assert_eq!(result["Technical"], vec!["Uses Rust programming language"]);
    }

    #[tokio::test]
    async fn compact_facts_returns_multiple_categories() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"Professional": ["Software Engineer at Tech Corp", "Specializes in Rust"], "Preferences": ["Likes dark mode", "Drinks coffee"]}"#,
            )))
            .mount(&llm_server)
            .await;

        let gen = ProfileGenerator::new(test_llm_provider(llm_server.uri()));
        let result = gen
            .compact_facts(&[
                "User is a Software Engineer at Tech Corp",
                "User specializes in Rust",
                "User likes dark mode",
                "User drinks coffee",
            ])
            .await
            .expect("should not fail");

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("Professional"));
        assert!(result.contains_key("Preferences"));
        assert_eq!(result["Professional"].len(), 2);
        assert_eq!(result["Preferences"].len(), 2);
    }

    #[tokio::test]
    async fn compact_facts_deduplicates_similar_facts() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llm_response(
                r#"{"Preferences": ["Favorite color is red"]}"#,
            )))
            .mount(&llm_server)
            .await;

        let gen = ProfileGenerator::new(test_llm_provider(llm_server.uri()));
        let result = gen
            .compact_facts(&["User likes red", "Favorite color is red"])
            .await
            .expect("should not fail");

        assert_eq!(result["Preferences"].len(), 1);
        assert_eq!(result["Preferences"][0], "Favorite color is red");
    }

    #[tokio::test]
    async fn compact_facts_fails_on_malformed_json() {
        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(llm_response(r#"not valid json at all"#)),
            )
            .mount(&llm_server)
            .await;

        let gen = ProfileGenerator::new(test_llm_provider(llm_server.uri()));
        let result = gen.compact_facts(&["User likes Rust"]).await;

        assert!(result.is_err());
    }

    // ── Serde roundtrip tests ────────────────────────────────────────

    #[test]
    fn extracted_narrative_deserializes() {
        let json = r#"{"narrative": "Alice is a developer."}"#;
        let parsed: ExtractedNarrative = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.narrative, "Alice is a developer.");
    }

    #[test]
    fn compacted_facts_deserializes_flat_map() {
        let json = r#"{"Technical": ["Rust", "Python"], "Personal": ["Likes hiking"]}"#;
        let parsed: CompactedFacts = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.categories.len(), 2);
        assert_eq!(parsed.categories["Technical"], vec!["Rust", "Python"]);
        assert_eq!(parsed.categories["Personal"], vec!["Likes hiking"]);
    }

    #[test]
    fn compacted_facts_deserializes_empty_object() {
        let json = r#"{}"#;
        let parsed: CompactedFacts = serde_json::from_str(json).unwrap();
        assert!(parsed.categories.is_empty());
    }

    #[test]
    fn compacted_facts_deserializes_empty_arrays() {
        let json = r#"{"Empty": []}"#;
        let parsed: CompactedFacts = serde_json::from_str(json).unwrap();
        assert!(parsed.categories["Empty"].is_empty());
    }
}
