use serde::{Deserialize, Serialize};

/// Container filter configuration for LLM-powered filtering
/// Controls whether and how memories should be filtered for specific container tags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerFilter {
    /// The container tag this filter applies to
    pub tag: String,
    /// Whether to apply LLM filtering for this container
    pub should_llm_filter: bool,
    /// Custom prompt to use for LLM filtering (if should_llm_filter is true)
    pub filter_prompt: Option<String>,
}

impl ContainerFilter {
    /// Create a new ContainerFilter with default settings
    pub fn new(tag: String) -> Self {
        Self {
            tag,
            should_llm_filter: false,
            filter_prompt: None,
        }
    }

    /// Create a ContainerFilter with LLM filtering enabled
    pub fn with_llm_filter(tag: String, filter_prompt: String) -> Self {
        Self {
            tag,
            should_llm_filter: true,
            filter_prompt: Some(filter_prompt),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_filter_model() {
        let filter = ContainerFilter::new("user_123".to_string());
        assert_eq!(filter.tag, "user_123");
        assert_eq!(filter.should_llm_filter, false);
        assert_eq!(filter.filter_prompt, None);
    }

    #[test]
    fn test_container_filter_with_llm() {
        let filter = ContainerFilter::with_llm_filter(
            "project_xyz".to_string(),
            "Only return work-related information".to_string(),
        );
        assert_eq!(filter.tag, "project_xyz");
        assert!(filter.should_llm_filter);
        assert_eq!(
            filter.filter_prompt,
            Some("Only return work-related information".to_string())
        );
    }

    #[test]
    fn test_container_filter_model_serialization() {
        let filter = ContainerFilter {
            tag: "test_container".to_string(),
            should_llm_filter: true,
            filter_prompt: Some("Test prompt".to_string()),
        };

        // Test serialization
        let json = serde_json::to_string(&filter).expect("Failed to serialize");
        assert!(json.contains("\"tag\":\"test_container\""));
        assert!(json.contains("\"should_llm_filter\":true"));
        assert!(json.contains("\"filter_prompt\":\"Test prompt\""));

        // Test deserialization
        let deserialized: ContainerFilter =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.tag, filter.tag);
        assert_eq!(deserialized.should_llm_filter, filter.should_llm_filter);
        assert_eq!(deserialized.filter_prompt, filter.filter_prompt);
    }

    #[test]
    fn test_container_filter_tag_required() {
        // Test that tag field is required (not optional)
        let json = r#"{"should_llm_filter":true,"filter_prompt":"test"}"#;
        let result: Result<ContainerFilter, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "Deserialization should fail without tag field"
        );
    }
}
