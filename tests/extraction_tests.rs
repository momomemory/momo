use momo::intelligence::types::{ExtractedMemory, ExtractionResult};

#[test]
fn test_extracted_memory_creation() {
    let memory = ExtractedMemory {
        content: "User prefers dark mode".to_string(),
        memory_type: "Preference".to_string(),
        confidence: 0.85,
        context: Some("UI settings discussion".to_string()),
        potential_contradiction: false,
    };

    assert_eq!(memory.content, "User prefers dark mode");
    assert_eq!(memory.memory_type, "Preference");
    assert!((memory.confidence - 0.85).abs() < f32::EPSILON);
    assert_eq!(memory.context, Some("UI settings discussion".to_string()));
}

#[test]
fn test_extracted_memory_without_context() {
    let memory = ExtractedMemory {
        content: "User is a software engineer".to_string(),
        memory_type: "Fact".to_string(),
        confidence: 0.95,
        context: None,
        potential_contradiction: false,
    };

    assert_eq!(memory.memory_type, "Fact");
    assert!(memory.context.is_none());
}

#[test]
fn test_extraction_result_empty() {
    let result = ExtractionResult {
        memories: vec![],
        source_content: "Some content".to_string(),
    };

    assert!(result.memories.is_empty());
    assert_eq!(result.source_content, "Some content");
}

#[test]
fn test_extraction_result_with_memories() {
    let memories = vec![
        ExtractedMemory {
            content: "Fact 1".to_string(),
            memory_type: "Fact".to_string(),
            confidence: 0.9,
            context: None,
            potential_contradiction: false,
        },
        ExtractedMemory {
            content: "Preference 1".to_string(),
            memory_type: "Preference".to_string(),
            confidence: 0.8,
            context: Some("Context".to_string()),
            potential_contradiction: false,
        },
    ];

    let result = ExtractionResult {
        memories,
        source_content: "Original content".to_string(),
    };

    assert_eq!(result.memories.len(), 2);
    assert_eq!(result.memories[0].memory_type, "Fact");
    assert_eq!(result.memories[1].memory_type, "Preference");
}

#[test]
fn test_memory_type_values() {
    // Test that the expected memory type strings work
    let fact = ExtractedMemory {
        content: "test".to_string(),
        memory_type: "Fact".to_string(),
        confidence: 1.0,
        context: None,
        potential_contradiction: false,
    };
    let preference = ExtractedMemory {
        content: "test".to_string(),
        memory_type: "Preference".to_string(),
        confidence: 1.0,
        context: None,
        potential_contradiction: false,
    };
    let episode = ExtractedMemory {
        content: "test".to_string(),
        memory_type: "Episode".to_string(),
        confidence: 1.0,
        context: None,
        potential_contradiction: false,
    };

    assert_eq!(fact.memory_type, "Fact");
    assert_eq!(preference.memory_type, "Preference");
    assert_eq!(episode.memory_type, "Episode");
}
