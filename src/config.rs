use serde::Deserialize;
use std::collections::HashMap;
use std::env;

fn parse_env_or<T: std::str::FromStr>(var: &str, default: T) -> T
where
    T::Err: std::fmt::Display,
{
    match env::var(var) {
        Ok(val) => match val.parse() {
            Ok(parsed) => parsed,
            Err(e) => {
                tracing::warn!("Invalid value '{}' for {}: {}. Using default.", val, var, e);
                default
            }
        },
        Err(_) => default,
    }
}

fn parse_env_opt<T: std::str::FromStr>(var: &str) -> Option<T>
where
    T::Err: std::fmt::Display,
{
    match env::var(var) {
        Ok(val) => match val.parse() {
            Ok(parsed) => Some(parsed),
            Err(e) => {
                tracing::warn!("Invalid value '{}' for {}: {}. Ignoring.", val, var, e);
                None
            }
        },
        Err(_) => None,
    }
}

/// Parse `RERANK_DOMAIN_MODELS` env var.
/// Format: comma-separated `domain:model` pairs, e.g. `code:jina-reranker-v1-turbo-en,docs:bge-reranker-v2-m3`
fn parse_domain_models() -> HashMap<String, String> {
    match env::var("RERANK_DOMAIN_MODELS") {
        Ok(val) if !val.is_empty() => val
            .split(',')
            .filter_map(|pair| {
                let mut parts = pair.splitn(2, ':');
                let domain = parts.next()?.trim();
                let model = parts.next()?.trim();
                if domain.is_empty() || model.is_empty() {
                    tracing::warn!(
                        "Invalid domain model pair '{}' in RERANK_DOMAIN_MODELS, skipping",
                        pair
                    );
                    None
                } else {
                    Some((domain.to_string(), model.to_string()))
                }
            })
            .collect(),
        _ => HashMap::new(),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub embeddings: EmbeddingsConfig,
    pub processing: ProcessingConfig,
    pub memory: MemoryConfig,
    pub ocr: OcrConfig,
    pub transcription: TranscriptionConfig,
    pub llm: Option<LlmConfig>,
    pub reranker: Option<RerankerConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_keys: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub auth_token: Option<String>,
    pub local_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsConfig {
    pub model: String,
    pub dimensions: usize,
    pub batch_size: usize,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub rate_limit: Option<f64>,
    pub timeout_secs: u64,
    pub max_retries: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OcrConfig {
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub languages: String,
    pub timeout_secs: u64,
    pub max_image_dimension: u32,
    pub min_image_dimension: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionConfig {
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model_path: Option<String>,
    pub timeout_secs: u64,
    pub max_file_size: u64,
    pub max_duration_secs: u64,
}

/// LLM configuration for chat/completion models
#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout_secs: u64,
    pub max_retries: u32,
    // Enable query rewrite stage (opt-in)
    pub enable_query_rewrite: bool,
    // Cache size for query rewrite results
    pub query_rewrite_cache_size: usize,
    // Timeout for query rewrite in seconds
    pub query_rewrite_timeout_secs: u64,
    pub enable_auto_relations: bool,
    /// Enable contradiction detection during LLM-driven processing.
    ///
    /// When true the system will run an optional contradiction-detection
    /// stage in LLM pipelines to surface conflicting statements or facts
    /// discovered during generation. This is opt-in and defaults to false
    /// to preserve backward compatibility and avoid extra LLM calls.
    pub enable_contradiction_detection: bool,
    /// Enable LLM filtering of search results.
    ///
    /// When true, search results will be filtered by an LLM to ensure
    /// relevance to the query. This is opt-in and defaults to false
    /// to preserve backward compatibility and avoid extra LLM calls.
    pub enable_llm_filter: bool,
    /// Custom prompt template for LLM filtering.
    ///
    /// If provided, this prompt will be used instead of the default
    /// filter prompt. The prompt should include placeholders for the
    /// query and results.
    pub filter_prompt: Option<String>,
}

/// Reranker configuration for improving search result ordering
#[derive(Debug, Clone, Deserialize)]
pub struct RerankerConfig {
    pub enabled: bool,
    pub model: String,
    pub top_k: usize,
    pub cache_dir: String,
    pub batch_size: usize,
    pub domain_models: HashMap<String, String>,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "bge-reranker-base".to_string(),
            top_k: 100,
            cache_dir: ".fastembed_cache".to_string(),
            batch_size: 64,
            domain_models: HashMap::new(),
        }
    }
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            model: "local/whisper-small".to_string(),
            api_key: None,
            base_url: None,
            model_path: None,
            timeout_secs: 300,
            max_file_size: 104857600,
            max_duration_secs: 7200,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessingConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub max_content_length: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MemoryConfig {
    pub episode_decay_days: f64,
    pub episode_decay_factor: f64,
    pub episode_decay_threshold: f64,
    pub episode_forget_grace_days: u32,
    pub forgetting_check_interval_secs: u64,
    pub profile_refresh_interval_secs: u64,
    pub inference: InferenceConfig,
}

/// Configuration for the background inference engine that derives new memories
#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    pub enabled: bool,
    pub interval_secs: u64,
    pub confidence_threshold: f32,
    pub max_per_run: usize,
    pub candidate_count: usize,
    pub seed_limit: usize,
    pub exclude_episodes: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: env::var("MOMO_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: parse_env_or("MOMO_PORT", 3000),
                api_keys: env::var("MOMO_API_KEYS")
                    .map(|keys| keys.split(',').map(|s| s.trim().to_string()).collect())
                    .unwrap_or_default(),
            },
            database: DatabaseConfig {
                url: env::var("DATABASE_URL").unwrap_or_else(|_| "file:momo.db".to_string()),
                auth_token: env::var("DATABASE_AUTH_TOKEN").ok(),
                local_path: env::var("DATABASE_LOCAL_PATH").ok(),
            },
            embeddings: EmbeddingsConfig {
                model: env::var("EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "BAAI/bge-small-en-v1.5".to_string()),
                dimensions: parse_env_or("EMBEDDING_DIMENSIONS", 384),
                batch_size: parse_env_or("EMBEDDING_BATCH_SIZE", 256),
                api_key: env::var("EMBEDDING_API_KEY").ok(),
                base_url: env::var("EMBEDDING_BASE_URL").ok(),
                rate_limit: parse_env_opt("EMBEDDING_RATE_LIMIT"),
                timeout_secs: parse_env_or("EMBEDDING_TIMEOUT", 30),
                max_retries: parse_env_or("EMBEDDING_MAX_RETRIES", 3),
            },
            processing: ProcessingConfig {
                chunk_size: parse_env_or("CHUNK_SIZE", 512),
                chunk_overlap: parse_env_or("CHUNK_OVERLAP", 50),
                max_content_length: parse_env_or("MAX_CONTENT_LENGTH", 10_000_000),
            },
            memory: MemoryConfig {
                episode_decay_days: parse_env_or("EPISODE_DECAY_DAYS", 30.0),
                episode_decay_factor: parse_env_or("EPISODE_DECAY_FACTOR", 0.9),
                episode_decay_threshold: parse_env_or("EPISODE_DECAY_THRESHOLD", 0.3),
                episode_forget_grace_days: parse_env_or("EPISODE_FORGET_GRACE_DAYS", 7),
                forgetting_check_interval_secs: parse_env_or("FORGETTING_CHECK_INTERVAL", 3600),
                profile_refresh_interval_secs: parse_env_or("PROFILE_REFRESH_INTERVAL_SECS", 86400),
                inference: InferenceConfig {
                    enabled: parse_env_or("ENABLE_INFERENCES", false),
                    interval_secs: parse_env_or("INFERENCE_INTERVAL_SECS", 86400),
                    confidence_threshold: parse_env_or("INFERENCE_CONFIDENCE_THRESHOLD", 0.7),
                    max_per_run: parse_env_or("INFERENCE_MAX_PER_RUN", 50),
                    candidate_count: parse_env_or("INFERENCE_CANDIDATE_COUNT", 5),
                    seed_limit: parse_env_or("INFERENCE_SEED_LIMIT", 50),
                    exclude_episodes: parse_env_or("INFERENCE_EXCLUDE_EPISODES", true),
                },
            },
            ocr: OcrConfig {
                model: env::var("OCR_MODEL").unwrap_or_else(|_| "local/tesseract".to_string()),
                api_key: env::var("OCR_API_KEY").ok(),
                base_url: env::var("OCR_BASE_URL").ok(),
                languages: env::var("OCR_LANGUAGES").unwrap_or_else(|_| "eng".to_string()),
                timeout_secs: parse_env_or("OCR_TIMEOUT", 60),
                max_image_dimension: parse_env_or("OCR_MAX_DIMENSION", 4096),
                min_image_dimension: parse_env_or("OCR_MIN_DIMENSION", 50),
            },
            transcription: TranscriptionConfig {
                model: env::var("TRANSCRIPTION_MODEL")
                    .unwrap_or_else(|_| "local/whisper-small".to_string()),
                api_key: env::var("TRANSCRIPTION_API_KEY").ok(),
                base_url: env::var("TRANSCRIPTION_BASE_URL").ok(),
                model_path: env::var("TRANSCRIPTION_MODEL_PATH").ok(),
                timeout_secs: parse_env_or("TRANSCRIPTION_TIMEOUT", 300),
                max_file_size: parse_env_or("TRANSCRIPTION_MAX_FILE_SIZE", 104857600),
                max_duration_secs: parse_env_or("TRANSCRIPTION_MAX_DURATION", 7200),
            },
            llm: env::var("LLM_MODEL").ok().map(|model| LlmConfig {
                model,
                api_key: env::var("LLM_API_KEY").ok(),
                base_url: env::var("LLM_BASE_URL").ok(),
                timeout_secs: parse_env_or("LLM_TIMEOUT", 30),
                max_retries: parse_env_or("LLM_MAX_RETRIES", 3),
                enable_query_rewrite: parse_env_or("ENABLE_QUERY_REWRITE", false),
                query_rewrite_cache_size: parse_env_or("QUERY_REWRITE_CACHE_SIZE", 1000),
                query_rewrite_timeout_secs: parse_env_or("QUERY_REWRITE_TIMEOUT_SECS", 2),
                enable_auto_relations: parse_env_or("ENABLE_AUTO_RELATIONS", true),
                enable_contradiction_detection: parse_env_or(
                    "ENABLE_CONTRADICTION_DETECTION",
                    false,
                ),
                enable_llm_filter: parse_env_or("DEFAULT_LLM_FILTER_ENABLED", false),
                filter_prompt: env::var("DEFAULT_FILTER_PROMPT").ok(),
            }),
            reranker: {
                let enabled = parse_env_or("RERANK_ENABLED", false);

                if enabled {
                    let top_k = parse_env_or("RERANK_TOP_K", 100);

                    if top_k == 0 {
                        panic!("RERANK_TOP_K must be greater than 0");
                    }

                    Some(RerankerConfig {
                        enabled,
                        model: env::var("RERANK_MODEL")
                            .unwrap_or_else(|_| "bge-reranker-base".to_string()),
                        top_k,
                        cache_dir: env::var("RERANK_CACHE_DIR")
                            .unwrap_or_else(|_| ".fastembed_cache".to_string()),
                        batch_size: parse_env_or("RERANK_BATCH_SIZE", 64),
                        domain_models: parse_domain_models(),
                    })
                } else {
                    None
                }
            },
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        Self::default()
    }
}

/// Known embedding providers that use OpenAI-compatible APIs
const KNOWN_PROVIDERS: &[&str] = &["openai", "openrouter", "ollama", "lmstudio", "local"];

/// Known LLM providers that use OpenAI-compatible APIs
pub const KNOWN_LLM_PROVIDERS: &[&str] = &["openai", "openrouter", "ollama", "lmstudio"];

/// Parse a model name into (provider, model) tuple.
///
/// Examples:
/// - "openai/text-embedding-3-small" -> ("openai", "text-embedding-3-small")
/// - "ollama/nomic-embed-text" -> ("ollama", "nomic-embed-text")
/// - "bge-small-en-v1.5" -> ("local", "bge-small-en-v1.5")
/// - "BAAI/bge-small-en-v1.5" -> ("local", "BAAI/bge-small-en-v1.5")
pub fn parse_provider_model(model: &str) -> (&str, &str) {
    if let Some((prefix, rest)) = model.split_once('/') {
        // Check if prefix is a known provider
        let prefix_lower = prefix.to_lowercase();
        if KNOWN_PROVIDERS.contains(&prefix_lower.as_str()) {
            return (prefix, rest);
        }
    }
    // Default to local provider
    ("local", model)
}

/// Parse an LLM model name into (provider, model) tuple.
///
/// Examples:
/// - "openai/gpt-4o" -> ("openai", "gpt-4o")
/// - "ollama/llama3.2" -> ("ollama", "llama3.2")
/// - "openrouter/anthropic/claude-3.5-sonnet" -> ("openrouter", "anthropic/claude-3.5-sonnet")
pub fn parse_llm_provider_model(model: &str) -> (&str, &str) {
    if let Some((prefix, rest)) = model.split_once('/') {
        let prefix_lower = prefix.to_lowercase();
        if KNOWN_LLM_PROVIDERS.contains(&prefix_lower.as_str()) {
            return (prefix, rest);
        }
    }
    // Default to treating the whole string as a local model
    ("local", model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Static mutex to ensure reranker config tests don't run in parallel
    // (they manipulate environment variables which are process-global)
    static RERANKER_TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_transcription_config_defaults() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::remove_var("RERANK_ENABLED");
        std::env::remove_var("RERANK_TOP_K");

        let config = Config::default();
        assert_eq!(config.transcription.model, "local/whisper-small");
        assert!(config.transcription.api_key.is_none());
        assert_eq!(config.transcription.timeout_secs, 300);
        assert_eq!(config.transcription.max_duration_secs, 7200);
    }

    #[test]
    fn test_reranker_config_defaults() {
        let defaults = RerankerConfig::default();
        assert_eq!(defaults.enabled, false);
        assert_eq!(defaults.model, "bge-reranker-base");
        assert_eq!(defaults.top_k, 100);
        assert_eq!(defaults.cache_dir, ".fastembed_cache");
        assert_eq!(defaults.batch_size, 64);
    }

    #[test]
    fn test_reranker_default_model_is_valid() {
        // Ensure the default model string is actually accepted by RerankerProvider::parse_model
        let defaults = RerankerConfig::default();
        let result = crate::embeddings::RerankerProvider::is_supported_model(&defaults.model);
        assert!(
            result,
            "Default reranker model '{}' must be a supported model",
            defaults.model
        );
    }

    #[test]
    fn test_reranker_config_disabled_by_default() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        // Clean up any leftover env vars from previous tests
        std::env::remove_var("RERANK_ENABLED");
        std::env::remove_var("RERANK_TOP_K");

        let config = Config::default();
        assert!(config.reranker.is_none());
    }

    #[test]
    fn test_reranker_config_from_env() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("RERANK_ENABLED", "true");
        std::env::set_var("RERANK_MODEL", "bge-reranker-base");
        std::env::set_var("RERANK_TOP_K", "50");
        std::env::set_var("RERANK_CACHE_DIR", "/custom/cache");
        std::env::set_var("RERANK_BATCH_SIZE", "32");

        let config = Config::default();

        assert!(config.reranker.is_some());
        let reranker = config.reranker.unwrap();
        assert_eq!(reranker.enabled, true);
        assert_eq!(reranker.model, "bge-reranker-base");
        assert_eq!(reranker.top_k, 50);
        assert_eq!(reranker.cache_dir, "/custom/cache");
        assert_eq!(reranker.batch_size, 32);

        // Cleanup
        std::env::remove_var("RERANK_ENABLED");
        std::env::remove_var("RERANK_MODEL");
        std::env::remove_var("RERANK_TOP_K");
        std::env::remove_var("RERANK_CACHE_DIR");
        std::env::remove_var("RERANK_BATCH_SIZE");
    }

    #[test]
    fn test_reranker_config_invalid_top_k() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        // Ensure a clean environment to avoid flakiness from other tests
        std::env::remove_var("RERANK_ENABLED");
        std::env::remove_var("RERANK_TOP_K");

        std::env::set_var("RERANK_ENABLED", "true");
        std::env::set_var("RERANK_TOP_K", "0");

        // Catch the panic so we can perform cleanup and assert it occurred
        let result = std::panic::catch_unwind(|| {
            let _ = Config::default();
        });

        assert!(
            result.is_err(),
            "expected Config::default() to panic for invalid RERANK_TOP_K"
        );

        // Cleanup after asserting panic
        std::env::remove_var("RERANK_ENABLED");
        std::env::remove_var("RERANK_TOP_K");
    }

    #[test]
    fn test_llm_config_defaults() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        // Ensure no LLM related env vars interfere
        std::env::remove_var("LLM_MODEL");
        std::env::remove_var("ENABLE_QUERY_REWRITE");
        std::env::remove_var("QUERY_REWRITE_CACHE_SIZE");
        std::env::remove_var("QUERY_REWRITE_TIMEOUT_SECS");

        // Without LLM_MODEL set, llm should be None
        let config = Config::default();
        assert!(config.llm.is_none());

        // Now set LLM_MODEL but leave new options unset to verify defaults
        std::env::set_var("LLM_MODEL", "openai/gpt-4o-mini");
        let config = Config::default();
        assert!(config.llm.is_some());
        let llm = config.llm.unwrap();
        assert_eq!(llm.model, "openai/gpt-4o-mini");
        assert_eq!(llm.enable_query_rewrite, false);
        assert_eq!(llm.query_rewrite_cache_size, 1000);
        assert_eq!(llm.query_rewrite_timeout_secs, 2);

        // Cleanup
        std::env::remove_var("LLM_MODEL");
        std::env::remove_var("ENABLE_QUERY_REWRITE");
        std::env::remove_var("QUERY_REWRITE_CACHE_SIZE");
        std::env::remove_var("QUERY_REWRITE_TIMEOUT_SECS");
    }

    #[test]
    fn test_llm_config_from_env() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("LLM_MODEL", "openai/gpt-4o-mini");
        std::env::set_var("ENABLE_QUERY_REWRITE", "true");
        std::env::set_var("QUERY_REWRITE_CACHE_SIZE", "2048");
        std::env::set_var("QUERY_REWRITE_TIMEOUT_SECS", "5");

        let config = Config::default();
        assert!(config.llm.is_some());
        let llm = config.llm.unwrap();
        assert_eq!(llm.model, "openai/gpt-4o-mini");
        assert_eq!(llm.enable_query_rewrite, true);
        assert_eq!(llm.query_rewrite_cache_size, 2048);
        assert_eq!(llm.query_rewrite_timeout_secs, 5);

        // Cleanup
        std::env::remove_var("LLM_MODEL");
        std::env::remove_var("ENABLE_QUERY_REWRITE");
        std::env::remove_var("QUERY_REWRITE_CACHE_SIZE");
        std::env::remove_var("QUERY_REWRITE_TIMEOUT_SECS");
    }

    #[test]
    fn test_forgetting_check_interval_defaults() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        // Ensure env var not set
        std::env::remove_var("FORGETTING_CHECK_INTERVAL");

        let config = Config::default();
        assert_eq!(config.memory.forgetting_check_interval_secs, 3600);
    }

    #[test]
    fn test_forgetting_check_interval_from_env() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("FORGETTING_CHECK_INTERVAL", "7200");
        let config = Config::default();
        assert_eq!(config.memory.forgetting_check_interval_secs, 7200);

        // Cleanup
        std::env::remove_var("FORGETTING_CHECK_INTERVAL");
    }

    #[test]
    fn test_episode_decay_threshold_defaults() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        // Ensure env vars not set
        std::env::remove_var("EPISODE_DECAY_THRESHOLD");

        let config = Config::default();
        assert_eq!(config.memory.episode_decay_threshold, 0.3);
    }

    #[test]
    fn test_episode_decay_threshold_from_env() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("EPISODE_DECAY_THRESHOLD", "0.5");
        let config = Config::default();
        assert_eq!(config.memory.episode_decay_threshold, 0.5);

        // Cleanup
        std::env::remove_var("EPISODE_DECAY_THRESHOLD");
    }

    #[test]
    fn test_episode_forget_grace_days_defaults() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::remove_var("EPISODE_FORGET_GRACE_DAYS");

        let config = Config::default();
        assert_eq!(config.memory.episode_forget_grace_days, 7);
    }

    #[test]
    fn test_episode_forget_grace_days_from_env() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("EPISODE_FORGET_GRACE_DAYS", "14");
        let config = Config::default();
        assert_eq!(config.memory.episode_forget_grace_days, 14);

        // Cleanup
        std::env::remove_var("EPISODE_FORGET_GRACE_DAYS");
    }

    #[test]
    fn test_inference_config_defaults() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        // Ensure inference env vars are not set
        std::env::remove_var("ENABLE_INFERENCES");
        std::env::remove_var("INFERENCE_INTERVAL_SECS");
        std::env::remove_var("INFERENCE_CONFIDENCE_THRESHOLD");
        std::env::remove_var("INFERENCE_MAX_PER_RUN");
        std::env::remove_var("INFERENCE_CANDIDATE_COUNT");
        std::env::remove_var("INFERENCE_SEED_LIMIT");
        std::env::remove_var("INFERENCE_EXCLUDE_EPISODES");

        let config = Config::default();
        let inf = &config.memory.inference;
        assert_eq!(inf.enabled, false);
        assert_eq!(inf.interval_secs, 86400);
        assert!((inf.confidence_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(inf.max_per_run, 50);
        assert_eq!(inf.candidate_count, 5);
        assert_eq!(inf.seed_limit, 50);
        assert_eq!(inf.exclude_episodes, true);
    }

    #[test]
    fn test_inference_config_from_env() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("ENABLE_INFERENCES", "true");
        std::env::set_var("INFERENCE_INTERVAL_SECS", "60");
        std::env::set_var("INFERENCE_CONFIDENCE_THRESHOLD", "0.85");
        std::env::set_var("INFERENCE_MAX_PER_RUN", "10");
        std::env::set_var("INFERENCE_CANDIDATE_COUNT", "8");
        std::env::set_var("INFERENCE_SEED_LIMIT", "20");
        std::env::set_var("INFERENCE_EXCLUDE_EPISODES", "false");

        let config = Config::default();
        let inf = &config.memory.inference;
        assert_eq!(inf.enabled, true);
        assert_eq!(inf.interval_secs, 60);
        assert!((inf.confidence_threshold - 0.85).abs() < f32::EPSILON);
        assert_eq!(inf.max_per_run, 10);
        assert_eq!(inf.candidate_count, 8);
        assert_eq!(inf.seed_limit, 20);
        assert_eq!(inf.exclude_episodes, false);

        // Cleanup
        std::env::remove_var("ENABLE_INFERENCES");
        std::env::remove_var("INFERENCE_INTERVAL_SECS");
        std::env::remove_var("INFERENCE_CONFIDENCE_THRESHOLD");
        std::env::remove_var("INFERENCE_MAX_PER_RUN");
        std::env::remove_var("INFERENCE_CANDIDATE_COUNT");
        std::env::remove_var("INFERENCE_SEED_LIMIT");
        std::env::remove_var("INFERENCE_EXCLUDE_EPISODES");
    }

    #[test]
    fn test_llm_config_default_values() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::remove_var("LLM_MODEL");
        std::env::remove_var("DEFAULT_LLM_FILTER_ENABLED");
        std::env::remove_var("DEFAULT_FILTER_PROMPT");

        let config = Config::default();
        assert!(config.llm.is_none());

        std::env::set_var("LLM_MODEL", "openai/gpt-4o-mini");
        let config = Config::default();
        assert!(config.llm.is_some());
        let llm = config.llm.unwrap();
        assert_eq!(llm.enable_llm_filter, false);
        assert!(llm.filter_prompt.is_none());

        std::env::remove_var("LLM_MODEL");
    }

    #[test]
    fn test_llm_config_defaults_loaded_correctly() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("LLM_MODEL", "openai/gpt-4o-mini");
        std::env::set_var("DEFAULT_LLM_FILTER_ENABLED", "true");
        std::env::set_var("DEFAULT_FILTER_PROMPT", "Custom filter prompt");

        let config = Config::from_env();
        assert!(config.llm.is_some());
        let llm = config.llm.unwrap();
        assert_eq!(llm.enable_llm_filter, true);
        assert_eq!(llm.filter_prompt.as_deref(), Some("Custom filter prompt"));
        assert_eq!(llm.enable_query_rewrite, false);
        assert_eq!(llm.enable_contradiction_detection, false);

        std::env::remove_var("LLM_MODEL");
        std::env::remove_var("DEFAULT_LLM_FILTER_ENABLED");
        std::env::remove_var("DEFAULT_FILTER_PROMPT");
    }

    #[test]
    fn test_parse_env_or_valid_value() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("__TEST_PARSE_PORT", "8080");
        let result: u16 = parse_env_or("__TEST_PARSE_PORT", 3000);
        assert_eq!(result, 8080);

        std::env::remove_var("__TEST_PARSE_PORT");
    }

    #[test]
    fn test_parse_env_or_invalid_value_uses_default() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("__TEST_PARSE_PORT", "abc");
        let result: u16 = parse_env_or("__TEST_PARSE_PORT", 3000);
        assert_eq!(result, 3000);

        std::env::remove_var("__TEST_PARSE_PORT");
    }

    #[test]
    fn test_parse_env_or_missing_var_uses_default() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::remove_var("__TEST_PARSE_MISSING");
        let result: u16 = parse_env_or("__TEST_PARSE_MISSING", 3000);
        assert_eq!(result, 3000);
    }

    #[test]
    fn test_parse_env_opt_valid() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("__TEST_PARSE_OPT", "42.5");
        let result: Option<f64> = parse_env_opt("__TEST_PARSE_OPT");
        assert_eq!(result, Some(42.5));

        std::env::remove_var("__TEST_PARSE_OPT");
    }

    #[test]
    fn test_parse_env_opt_invalid() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var("__TEST_PARSE_OPT", "not_a_number");
        let result: Option<f64> = parse_env_opt("__TEST_PARSE_OPT");
        assert_eq!(result, None);

        std::env::remove_var("__TEST_PARSE_OPT");
    }

    #[test]
    fn test_parse_domain_models_valid() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var(
            "RERANK_DOMAIN_MODELS",
            "code:jina-reranker-v1-turbo-en,docs:bge-reranker-v2-m3",
        );

        let result = parse_domain_models();
        assert_eq!(result.len(), 2);
        assert_eq!(result.get("code").unwrap(), "jina-reranker-v1-turbo-en");
        assert_eq!(result.get("docs").unwrap(), "bge-reranker-v2-m3");

        std::env::remove_var("RERANK_DOMAIN_MODELS");
    }

    #[test]
    fn test_parse_domain_models_empty() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::remove_var("RERANK_DOMAIN_MODELS");
        let result = parse_domain_models();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_domain_models_skips_invalid() {
        let _guard = RERANKER_TEST_MUTEX.lock().unwrap();

        std::env::set_var(
            "RERANK_DOMAIN_MODELS",
            "code:jina-reranker-v1-turbo-en,invalid_no_colon,:empty_domain,nomodel:",
        );

        let result = parse_domain_models();
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("code").unwrap(), "jina-reranker-v1-turbo-en");

        std::env::remove_var("RERANK_DOMAIN_MODELS");
    }
}
