use crate::config::Config;
use crate::error::{MomoError, Result};
use crate::llm::{prompts, LlmProvider};
use serde::Deserialize;

/// Decision about whether to include or skip a piece of content
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterDecision {
    Include,
    Skip,
}

impl std::fmt::Display for FilterDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterDecision::Include => write!(f, "include"),
            FilterDecision::Skip => write!(f, "skip"),
        }
    }
}

/// Result of filtering operation
#[derive(Debug, Clone)]
pub struct FilterResult {
    pub decision: FilterDecision,
    pub reasoning: Option<String>,
}

/// Wrapper struct for parsing LLM JSON responses
#[derive(Debug, Deserialize)]
struct FilterResponse {
    decision: String,
    reasoning: Option<String>,
}

/// LLM-powered content filter for search results
pub struct LlmFilter {
    llm: LlmProvider,
    config: Config,
}

impl Clone for LlmFilter {
    fn clone(&self) -> Self {
        Self {
            llm: self.llm.clone(),
            config: self.config.clone(),
        }
    }
}

impl LlmFilter {
    pub fn new(llm: LlmProvider, config: Config) -> Self {
        Self { llm, config }
    }

    pub async fn filter_content(
        &self,
        content: &str,
        container_tag: &str,
        doc_id: &str,
        override_filter_prompt: Option<&str>,
    ) -> Result<FilterResult> {
        let global_prompt = self
            .config
            .llm
            .as_ref()
            .and_then(|llm| llm.filter_prompt.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("");

        let filter_prompt = override_filter_prompt.unwrap_or(global_prompt);

        if filter_prompt.is_empty() {
            return Ok(FilterResult {
                decision: FilterDecision::Include,
                reasoning: None,
            });
        }

        // String match short-circuit doesn't need LLM, check first
        if content.contains(filter_prompt) {
            return Ok(FilterResult {
                decision: FilterDecision::Include,
                reasoning: Some(format!("Content matches filter string: {filter_prompt}")),
            });
        }

        if !self.llm.is_available() {
            tracing::warn!("LLM unavailable, skipping filter (including all content)");
            return Ok(FilterResult {
                decision: FilterDecision::Include,
                reasoning: None,
            });
        }

        let prompt = prompts::llm_filter_prompt(content, filter_prompt);
        match self
            .llm
            .complete_structured::<FilterResponse>(&prompt)
            .await
        {
            Ok(response) => {
                let decision = match response.decision.to_lowercase().as_str() {
                    "include" => FilterDecision::Include,
                    "skip" => FilterDecision::Skip,
                    _ => {
                        tracing::warn!(
                            decision = %response.decision,
                            "LLM returned invalid decision, defaulting to Include"
                        );
                        FilterDecision::Include
                    }
                };

                let reasoning = if decision == FilterDecision::Skip {
                    response
                        .reasoning
                        .map(|r| r.chars().take(50).collect::<String>())
                } else {
                    response.reasoning
                };

                tracing::info!(
                    container_tag = %container_tag,
                    doc_id = %doc_id,
                    decision = %decision,
                    filter_reasoning = ?reasoning,
                    "LLM filter decision"
                );

                Ok(FilterResult {
                    decision,
                    reasoning,
                })
            }
            Err(MomoError::LlmUnavailable(reason)) => {
                tracing::warn!(%reason, "LLM unavailable during filtering");
                Ok(FilterResult {
                    decision: FilterDecision::Include,
                    reasoning: None,
                })
            }
            Err(e) => {
                tracing::error!(error = %e, "LLM filter failed, defaulting to Include");
                Ok(FilterResult {
                    decision: FilterDecision::Include,
                    reasoning: None,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DatabaseConfig, EmbeddingsConfig, InferenceConfig, McpConfig, MemoryConfig, OcrConfig,
        ProcessingConfig, ServerConfig, TranscriptionConfig,
    };

    fn test_config() -> Config {
        Config {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 3000,
                api_keys: vec![],
            },
            mcp: McpConfig::default(),
            database: DatabaseConfig {
                url: "file:test.db".to_string(),
                auth_token: None,
                local_path: None,
            },
            embeddings: EmbeddingsConfig {
                model: "BAAI/bge-small-en-v1.5".to_string(),
                dimensions: 384,
                batch_size: 8,
            },
            processing: ProcessingConfig {
                chunk_size: 512,
                chunk_overlap: 50,
            },
            memory: MemoryConfig {
                episode_decay_days: 30.0,
                episode_decay_factor: 0.9,
                episode_decay_threshold: 0.3,
                episode_forget_grace_days: 7,
                forgetting_check_interval_secs: 3600,
                profile_refresh_interval_secs: 86400,
                inference: InferenceConfig {
                    enabled: false,
                    interval_secs: 86400,
                    confidence_threshold: 0.7,
                    max_per_run: 10,
                    candidate_count: 5,
                    seed_limit: 100,
                    exclude_episodes: false,
                },
            },
            ocr: OcrConfig {
                model: "local/tesseract".to_string(),
                api_key: None,
                base_url: None,
                languages: "eng".to_string(),
                timeout_secs: 60,
                max_image_dimension: 4096,
                min_image_dimension: 50,
            },
            transcription: TranscriptionConfig {
                model: "local/whisper".to_string(),
                api_key: None,
                base_url: None,
                model_path: None,
                timeout_secs: 300,
                max_file_size: 52428800,
                max_duration_secs: 3600,
            },
            llm: None,
            reranker: None,
        }
    }

    fn test_llm_unavailable() -> LlmProvider {
        LlmProvider::unavailable("test unavailable")
    }

    #[tokio::test]
    async fn test_llm_filter_module_structure() {
        // Test FilterDecision enum
        let include_decision = FilterDecision::Include;
        let skip_decision = FilterDecision::Skip;
        assert_eq!(include_decision, FilterDecision::Include);
        assert_eq!(skip_decision, FilterDecision::Skip);
        assert_ne!(include_decision, skip_decision);

        // Test FilterResult struct
        let result = FilterResult {
            decision: FilterDecision::Include,
            reasoning: Some("test reasoning".to_string()),
        };
        assert_eq!(result.decision, FilterDecision::Include);
        assert_eq!(result.reasoning, Some("test reasoning".to_string()));

        // Test LlmFilter struct and constructor
        let config = test_config();
        let llm = test_llm_unavailable();
        let filter = LlmFilter::new(llm, config);

        // Test Clone implementation
        let cloned_filter = filter.clone();
        assert!(cloned_filter.llm.is_available() == filter.llm.is_available());
    }

    #[tokio::test]
    async fn test_filter_content_empty_filter_prompt() {
        let config = test_config();
        let llm = test_llm_unavailable();
        let filter = LlmFilter::new(llm, config);

        let result = filter
            .filter_content("This is content for user_123", "user_123", "doc_123", None)
            .await
            .expect("filter_content should not fail");

        assert_eq!(result.decision, FilterDecision::Include);
        assert!(result.reasoning.is_none());
    }

    #[tokio::test]
    async fn test_filter_content_string_matching() {
        use crate::config::LlmConfig;

        let mut config = test_config();
        config.llm = Some(LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test".to_string()),
            base_url: None,
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,

            filter_prompt: Some("technical".to_string()),
        });
        let llm = test_llm_unavailable();
        let filter = LlmFilter::new(llm, config);

        let result = filter
            .filter_content(
                "This is a technical document about Rust",
                "user_123",
                "doc_123",
                None,
            )
            .await
            .expect("filter_content should not fail");

        assert_eq!(result.decision, FilterDecision::Include);
        assert!(result
            .reasoning
            .as_ref()
            .map(|r| r.contains("technical"))
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn test_graceful_degradation_returns_include_when_llm_unavailable() {
        let config = test_config();
        let llm = test_llm_unavailable();
        let filter = LlmFilter::new(llm, config);

        let result = filter
            .filter_content("Any content", "any_tag", "doc_123", None)
            .await
            .expect("filter_content should not fail");

        assert_eq!(result.decision, FilterDecision::Include);
        assert!(result.reasoning.is_none());
    }

    #[tokio::test]
    async fn test_clone_implementation_works_correctly() {
        let config = test_config();
        let llm = test_llm_unavailable();
        let filter = LlmFilter::new(llm, config);

        let cloned = filter.clone();

        assert_eq!(
            filter.llm.is_available(),
            cloned.llm.is_available(),
            "Clone should preserve LLM availability"
        );

        let result1 = filter
            .filter_content("test content user_123", "user_123", "doc_123", None)
            .await
            .expect("filter should work");
        let result2 = cloned
            .filter_content("test content user_123", "user_123", "doc_123", None)
            .await
            .expect("clone should work");

        assert_eq!(result1.decision, result2.decision);
    }

    #[tokio::test]
    async fn test_filter_with_llm_returns_include() {
        use crate::config::LlmConfig;
        use serde_json::json;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": r#"{"decision":"include","reasoning":"Content matches technical criteria"}"#
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            })))
            .mount(&llm_server)
            .await;

        let mut config = test_config();
        config.llm = Some(LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(llm_server.uri()),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,

            filter_prompt: Some("technical documents only".to_string()),
        });

        let llm = LlmProvider::new(config.llm.as_ref());
        let filter = LlmFilter::new(llm, config);

        let result = filter
            .filter_content(
                "This is about marketing strategies",
                "user_123",
                "doc_123",
                None,
            )
            .await
            .expect("filter should work");

        assert_eq!(result.decision, FilterDecision::Include);
        assert!(result.reasoning.is_some());
    }

    #[tokio::test]
    async fn test_filter_with_llm_returns_skip_and_redacts_reasoning() {
        use crate::config::LlmConfig;
        use serde_json::json;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": r#"{"decision":"skip","reasoning":"This is a very long reasoning text that should be redacted because it exceeds fifty characters and we only want the first fifty"}"#
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            })))
            .mount(&llm_server)
            .await;

        let mut config = test_config();
        config.llm = Some(LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(llm_server.uri()),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,

            filter_prompt: Some("technical documents only".to_string()),
        });

        let llm = LlmProvider::new(config.llm.as_ref());
        let filter = LlmFilter::new(llm, config);

        let result = filter
            .filter_content(
                "This is about marketing strategies",
                "user_123",
                "doc_123",
                None,
            )
            .await
            .expect("filter should work");

        assert_eq!(result.decision, FilterDecision::Skip);
        assert!(result.reasoning.is_some());
        let reasoning = result.reasoning.unwrap();
        assert!(
            reasoning.len() <= 50,
            "Reasoning should be redacted to 50 chars"
        );
        assert_eq!(reasoning.len(), 50);
    }

    #[tokio::test]
    async fn test_filter_with_llm_error_returns_include() {
        use crate::config::LlmConfig;
        use serde_json::json;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_json(json!({
                "error": {"message": "Internal server error"}
            })))
            .mount(&llm_server)
            .await;

        let mut config = test_config();
        config.llm = Some(LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(llm_server.uri()),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,

            filter_prompt: Some("technical documents only".to_string()),
        });

        let llm = LlmProvider::new(config.llm.as_ref());
        let filter = LlmFilter::new(llm, config);

        let result = filter
            .filter_content("Any content", "user_123", "doc_123", None)
            .await
            .expect("filter should gracefully degrade");

        assert_eq!(result.decision, FilterDecision::Include);
        assert!(result.reasoning.is_none());
    }

    #[tokio::test]
    async fn test_filter_with_invalid_llm_decision_defaults_to_include() {
        use crate::config::LlmConfig;
        use serde_json::json;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let llm_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": r#"{"decision":"INVALID","reasoning":"Invalid decision value"}"#
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            })))
            .mount(&llm_server)
            .await;

        let mut config = test_config();
        config.llm = Some(LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(llm_server.uri()),
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,

            filter_prompt: Some("technical documents only".to_string()),
        });

        let llm = LlmProvider::new(config.llm.as_ref());
        let filter = LlmFilter::new(llm, config);

        let result = filter
            .filter_content("Any content", "user_123", "doc_123", None)
            .await
            .expect("filter should work");

        assert_eq!(result.decision, FilterDecision::Include);
    }

    #[tokio::test]
    async fn test_override_filter_prompt_uses_override_over_global() {
        use crate::config::LlmConfig;

        let mut config = test_config();
        config.llm = Some(LlmConfig {
            model: "openai/gpt-4o-mini".to_string(),
            api_key: Some("test".to_string()),
            base_url: None,
            timeout_secs: 5,
            max_retries: 0,
            enable_query_rewrite: false,
            query_rewrite_cache_size: 1000,
            query_rewrite_timeout_secs: 2,
            enable_auto_relations: false,
            enable_contradiction_detection: false,

            filter_prompt: Some("global prompt".to_string()),
        });
        let llm = test_llm_unavailable();
        let filter = LlmFilter::new(llm, config);

        let result = filter
            .filter_content(
                "This content contains the override keyword",
                "user_123",
                "doc_123",
                Some("override keyword"),
            )
            .await
            .expect("filter should work");

        assert_eq!(result.decision, FilterDecision::Include);
        assert!(result
            .reasoning
            .as_ref()
            .map(|r| r.contains("override keyword"))
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn test_override_filter_prompt_none_falls_back_to_global() {
        let config = test_config();
        let llm = test_llm_unavailable();
        let filter = LlmFilter::new(llm, config);

        let result = filter
            .filter_content("Any content", "user_123", "doc_123", None)
            .await
            .expect("filter should work");

        assert_eq!(result.decision, FilterDecision::Include);
        assert!(result.reasoning.is_none());
    }
}
