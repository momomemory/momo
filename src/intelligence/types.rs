use serde::{Deserialize, Serialize};

use super::contradiction::ContradictionCheckResult;

/// Context from heuristic contradiction detection, passed to the LLM
/// so it can confirm or override the heuristic decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeuristicContext {
    /// The ID of the candidate memory that was flagged
    pub candidate_memory_id: String,
    /// Content of the candidate memory
    pub candidate_content: String,
    /// The heuristic contradiction result (likely/unlikely)
    pub heuristic_result: ContradictionCheckResult,
}

/// Represents a single extracted memory from content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMemory {
    /// The actual memory content
    pub content: String,
    /// Type of memory (e.g., "Fact", "Preference", "Event")
    pub memory_type: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Optional context about where/how this memory was extracted
    pub context: Option<String>,
    /// Whether a heuristic contradiction was detected against existing memories.
    /// Set by `MemoryExtractor::check_contradictions()` when enabled.
    #[serde(default)]
    pub potential_contradiction: bool,
}

/// Result of extracting memories from content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// List of extracted memories
    pub memories: Vec<ExtractedMemory>,
    /// The original source content
    pub source_content: String,
}

/// Represents a detected relationship between a new memory and an existing one
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipClassification {
    /// ID of the existing memory this relates to
    pub memory_id: String,
    /// Type of relationship (e.g., "updates", "extends", "none")
    pub relation_type: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Optional explanation of why this relationship was detected
    pub reasoning: Option<String>,
}

/// Wrapper for LLM response that can handle both array and object formats
/// LLMs sometimes return `[...]` and sometimes `{"relationships": [...]}`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RelationshipClassificationsResponse {
    /// Direct array response: [...]
    Array(Vec<RelationshipClassification>),
    /// Wrapped object response: {"relationships": [...]}
    Wrapped {
        #[serde(alias = "classifications", alias = "results")]
        relationships: Vec<RelationshipClassification>,
    },
}

impl RelationshipClassificationsResponse {
    /// Extract the classifications regardless of wrapper format
    pub fn into_classifications(self) -> Vec<RelationshipClassification> {
        match self {
            Self::Array(v) => v,
            Self::Wrapped { relationships } => relationships,
        }
    }
}

/// Result of detecting relationships for a new memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub classifications: Vec<RelationshipClassification>,
    /// Whether the LLM overrode a heuristic contradiction flag.
    /// `Some(true)` = LLM confirmed contradiction, `Some(false)` = LLM overrode, `None` = no heuristic was involved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heuristic_overridden: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracted_memory_serializes() {
        let memory = ExtractedMemory {
            content: "User prefers dark mode".to_string(),
            memory_type: "Preference".to_string(),
            confidence: 0.8,
            context: Some("Settings conversation".to_string()),
            potential_contradiction: false,
        };

        let json = serde_json::to_string(&memory).unwrap();
        assert!(json.contains("User prefers dark mode"));
        assert!(json.contains("Preference"));
        assert!(json.contains("0.8"));
    }

    #[test]
    fn test_extracted_memory_deserializes() {
        let json = r#"{
            "content": "User lives in San Francisco",
            "memory_type": "Fact",
            "confidence": 0.95,
            "context": "Location discussion"
        }"#;

        let memory: ExtractedMemory = serde_json::from_str(json).unwrap();
        assert_eq!(memory.content, "User lives in San Francisco");
        assert_eq!(memory.memory_type, "Fact");
        assert_eq!(memory.confidence, 0.95);
        assert_eq!(memory.context, Some("Location discussion".to_string()));
    }

    #[test]
    fn test_extracted_memory_without_context() {
        let memory = ExtractedMemory {
            content: "Meeting scheduled for tomorrow".to_string(),
            memory_type: "Event".to_string(),
            confidence: 0.9,
            context: None,
            potential_contradiction: false,
        };

        let json = serde_json::to_string(&memory).unwrap();
        let deserialized: ExtractedMemory = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.context, None);
    }

    #[test]
    fn test_extraction_result_serializes() {
        let result = ExtractionResult {
            memories: vec![
                ExtractedMemory {
                    content: "First memory".to_string(),
                    memory_type: "Fact".to_string(),
                    confidence: 0.8,
                    context: None,
                    potential_contradiction: false,
                },
                ExtractedMemory {
                    content: "Second memory".to_string(),
                    memory_type: "Preference".to_string(),
                    confidence: 0.7,
                    context: Some("Context here".to_string()),
                    potential_contradiction: false,
                },
            ],
            source_content: "Original content here".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("First memory"));
        assert!(json.contains("Second memory"));
        assert!(json.contains("Original content here"));
    }

    #[test]
    fn test_extraction_result_deserializes() {
        let json = r#"{
            "memories": [
                {
                    "content": "Test memory",
                    "memory_type": "Fact",
                    "confidence": 0.85,
                    "context": null
                }
            ],
            "source_content": "Test source"
        }"#;

        let result: ExtractionResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.memories[0].content, "Test memory");
        assert_eq!(result.source_content, "Test source");
    }

    #[test]
    fn test_extraction_result_empty_memories() {
        let result = ExtractionResult {
            memories: vec![],
            source_content: "No memories found".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ExtractionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.memories.len(), 0);
        assert_eq!(deserialized.source_content, "No memories found");
    }

    #[test]
    fn test_confidence_bounds() {
        // Test valid confidence values
        let memory_low = ExtractedMemory {
            content: "Low confidence".to_string(),
            memory_type: "Fact".to_string(),
            confidence: 0.0,
            context: None,
            potential_contradiction: false,
        };
        assert_eq!(memory_low.confidence, 0.0);

        let memory_high = ExtractedMemory {
            content: "High confidence".to_string(),
            memory_type: "Fact".to_string(),
            confidence: 1.0,
            context: None,
            potential_contradiction: false,
        };
        assert_eq!(memory_high.confidence, 1.0);

        let memory_mid = ExtractedMemory {
            content: "Mid confidence".to_string(),
            memory_type: "Fact".to_string(),
            confidence: 0.5,
            context: None,
            potential_contradiction: false,
        };
        assert_eq!(memory_mid.confidence, 0.5);
    }

    #[test]
    fn test_relationship_classification_deserializes() {
        let json = r#"{
            "memory_id": "mem_abc123",
            "relation_type": "updates",
            "confidence": 0.85,
            "reasoning": "New content supersedes old preference"
        }"#;

        let classification: RelationshipClassification = serde_json::from_str(json).unwrap();
        assert_eq!(classification.memory_id, "mem_abc123");
        assert_eq!(classification.relation_type, "updates");
        assert_eq!(classification.confidence, 0.85);
        assert_eq!(
            classification.reasoning,
            Some("New content supersedes old preference".to_string())
        );
    }

    #[test]
    fn test_relationship_classification_without_reasoning() {
        let classification = RelationshipClassification {
            memory_id: "mem_xyz789".to_string(),
            relation_type: "extends".to_string(),
            confidence: 0.7,
            reasoning: None,
        };

        let json = serde_json::to_string(&classification).unwrap();
        let deserialized: RelationshipClassification = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.reasoning, None);
    }

    #[test]
    fn test_detection_result_serializes() {
        let result = DetectionResult {
            classifications: vec![
                RelationshipClassification {
                    memory_id: "mem_001".to_string(),
                    relation_type: "updates".to_string(),
                    confidence: 0.9,
                    reasoning: Some("Direct contradiction".to_string()),
                },
                RelationshipClassification {
                    memory_id: "mem_002".to_string(),
                    relation_type: "extends".to_string(),
                    confidence: 0.75,
                    reasoning: None,
                },
            ],
            heuristic_overridden: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("mem_001"));
        assert!(json.contains("updates"));
        assert!(json.contains("mem_002"));
        assert!(json.contains("extends"));
    }

    #[test]
    fn test_detection_result_empty_classifications() {
        let result = DetectionResult {
            classifications: vec![],
            heuristic_overridden: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: DetectionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.classifications.len(), 0);
    }

    #[test]
    fn test_relationship_classification_relation_types() {
        let updates = RelationshipClassification {
            memory_id: "mem_1".to_string(),
            relation_type: "updates".to_string(),
            confidence: 0.9,
            reasoning: None,
        };
        assert_eq!(updates.relation_type, "updates");

        let extends = RelationshipClassification {
            memory_id: "mem_2".to_string(),
            relation_type: "extends".to_string(),
            confidence: 0.8,
            reasoning: None,
        };
        assert_eq!(extends.relation_type, "extends");

        let none = RelationshipClassification {
            memory_id: "mem_3".to_string(),
            relation_type: "none".to_string(),
            confidence: 0.1,
            reasoning: None,
        };
        assert_eq!(none.relation_type, "none");
    }

    #[test]
    fn test_response_wrapper_handles_array() {
        let json = r#"[
            {"memory_id": "mem_1", "relation_type": "updates", "confidence": 0.9, "reasoning": null}
        ]"#;
        let response: RelationshipClassificationsResponse = serde_json::from_str(json).unwrap();
        let classifications = response.into_classifications();
        assert_eq!(classifications.len(), 1);
        assert_eq!(classifications[0].memory_id, "mem_1");
    }

    #[test]
    fn test_response_wrapper_handles_object() {
        let json = r#"{"relationships": [
            {"memory_id": "mem_2", "relation_type": "extends", "confidence": 0.8, "reasoning": "adds detail"}
        ]}"#;
        let response: RelationshipClassificationsResponse = serde_json::from_str(json).unwrap();
        let classifications = response.into_classifications();
        assert_eq!(classifications.len(), 1);
        assert_eq!(classifications[0].memory_id, "mem_2");
    }

    #[test]
    fn test_response_wrapper_handles_classifications_alias() {
        let json = r#"{"classifications": [
            {"memory_id": "mem_3", "relation_type": "none", "confidence": 0.5, "reasoning": null}
        ]}"#;
        let response: RelationshipClassificationsResponse = serde_json::from_str(json).unwrap();
        let classifications = response.into_classifications();
        assert_eq!(classifications.len(), 1);
        assert_eq!(classifications[0].memory_id, "mem_3");
    }
}
