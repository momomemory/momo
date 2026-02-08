//! Simple prompt templates for LLM-powered features
//!
//! These templates use basic `format!()` interpolation for type safety.
//! Missing variables will cause compile-time errors.

/// Generate a prompt for extracting memories from content
///
/// Returns a prompt that instructs the LLM to extract key facts, preferences,
/// and memories from the provided content as a JSON array.
///
/// # Arguments
/// * `content` - The text content to extract memories from
///
/// # Returns
/// A formatted prompt string ready for LLM completion
///
/// # Example
/// ```
/// use momo::llm::prompts::memory_extraction_prompt;
///
/// let prompt = memory_extraction_prompt("User prefers dark mode and uses vim");
/// assert!(prompt.contains("dark mode"));
/// ```
pub fn memory_extraction_prompt(content: &str) -> String {
    format!(
        r#"Extract key facts, preferences, and memories from the following content.
Return as a JSON array of memory objects with "content", "memory_type", and "confidence" fields.

Memory Types:
- Fact: Objective information about the user (e.g., occupation, location, skills)
- Preference: User choices, likes, dislikes, or stated preferences
- Episode: Events, experiences, or interactions the user has had

Confidence: A score from 0.0 to 1.0 indicating how certain you are about this memory.

Content:
{}

Respond with valid JSON only. Example format:
[
  {{"content": "User prefers dark mode", "memory_type": "preference", "confidence": 0.9}},
  {{"content": "User is a software engineer", "memory_type": "fact", "confidence": 0.85}},
  {{"content": "User attended a conference last week", "memory_type": "episode", "confidence": 0.8}}
]"#,
        content
    )
}

/// Generate a prompt for extracting memories from a conversation
///
/// Returns a prompt that instructs the LLM to extract key facts, preferences,
/// and memories from a conversation between user and assistant.
///
/// # Arguments
/// * `messages` - The conversation messages to extract memories from
///
/// # Returns
/// A formatted prompt string ready for LLM completion
///
/// # Example
/// ```
/// use momo::llm::prompts::conversation_extraction_prompt;
/// use momo::models::ConversationMessage;
/// use chrono::Utc;
///
/// let messages = vec![
///     ConversationMessage {
///         role: "user".to_string(),
///         content: "I prefer dark mode".to_string(),
///         timestamp: Some(Utc::now()),
///     },
/// ];
/// let prompt = conversation_extraction_prompt(&messages);
/// assert!(prompt.contains("dark mode"));
/// ```
pub fn conversation_extraction_prompt(messages: &[crate::models::ConversationMessage]) -> String {
    let conversation = messages
        .iter()
        .map(|msg| format!("[{}]: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"Extract key facts, preferences, and memories from the following conversation.
Return as a JSON array of memory objects with "content", "memory_type", and "confidence" fields.

Memory Types:
- Fact: Objective information about the user (e.g., occupation, location, skills)
- Preference: User choices, likes, dislikes, or stated preferences
- Episode: Events, experiences, or interactions the user has had

Confidence: A score from 0.0 to 1.0 indicating how certain you are about this memory.

Conversation:
{}

Respond with valid JSON only. Example format:
[
  {{"content": "User prefers dark mode", "memory_type": "preference", "confidence": 0.9}},
  {{"content": "User is a software engineer", "memory_type": "fact", "confidence": 0.85}},
  {{"content": "User attended a conference last week", "memory_type": "episode", "confidence": 0.8}}
]"#,
        conversation
    )
}

/// Generate a prompt for rewriting a search query
///
/// Returns a prompt that instructs the LLM to rewrite a search query
/// to improve semantic search results.
///
/// # Arguments
/// * `query` - The original search query to rewrite
///
/// # Returns
/// A formatted prompt string ready for LLM completion
///
/// # Example
/// ```
/// use momo::llm::prompts::query_rewrite_prompt;
///
/// let prompt = query_rewrite_prompt("how to use rust");
/// assert!(prompt.contains("rust"));
/// ```
pub fn query_rewrite_prompt(query: &str) -> String {
    format!(
        r#"Rewrite the following search query to improve semantic search results.
Expand abbreviations, add context, and make the query more specific.

Original query: {}

Respond with only the rewritten query, no explanation."#,
        query
    )
}

/// Generate a prompt for summarizing content
///
/// Returns a prompt that instructs the LLM to summarize content
/// to a specified maximum length.
///
/// # Arguments
/// * `content` - The text content to summarize
/// * `max_length` - Maximum length of the summary in words
///
/// # Returns
/// A formatted prompt string ready for LLM completion
///
/// # Example
/// ```
/// use momo::llm::prompts::summarize_prompt;
///
/// let prompt = summarize_prompt("Long article text...", 50);
/// assert!(prompt.contains("50 words"));
/// ```
pub fn summarize_prompt(content: &str, max_length: usize) -> String {
    format!(
        r#"Summarize the following content in {} words or less.
Focus on the key points and main ideas.

Content:
{}

Respond with only the summary, no preamble."#,
        max_length, content
    )
}

/// Generate a prompt for detecting relationships between a new memory and existing memories
///
/// Returns a prompt that instructs the LLM to classify relationships between
/// a new memory and a list of candidate memories. When `heuristic_context` is
/// provided, the prompt additionally asks the LLM to confirm or override the
/// heuristic contradiction assessment.
///
/// # Arguments
/// * `new_memory` - The new memory content to analyze
/// * `candidates` - List of (memory_id, content) tuples for candidate memories
/// * `heuristic_context` - Optional context from heuristic contradiction detection
///
/// # Returns
/// A formatted prompt string ready for LLM completion
///
/// # Example
/// ```
/// use momo::llm::prompts::relationship_detection_prompt;
///
/// let candidates = vec![
///     ("mem_123", "User prefers dark mode"),
///     ("mem_456", "User is a software engineer"),
/// ];
/// let prompt = relationship_detection_prompt("User now prefers light mode", &candidates, None);
/// assert!(prompt.contains("mem_123"));
/// ```
pub fn relationship_detection_prompt(
    new_memory: &str,
    candidates: &[(&str, &str)],
    heuristic_context: Option<&crate::intelligence::types::HeuristicContext>,
) -> String {
    let candidate_list = candidates
        .iter()
        .map(|(id, content)| format!("[ID: {}] {}", id, content))
        .collect::<Vec<_>>()
        .join("\n");

    let heuristic_section = match heuristic_context {
        Some(ctx) => format!(
            r#"

IMPORTANT — Heuristic Contradiction Flag:
A fast heuristic detector flagged a POTENTIAL CONTRADICTION between the new memory and candidate [ID: {}].
Heuristic assessment: "{}" contradiction.
Candidate content: "{}"

You MUST explicitly confirm or override this assessment for that candidate:
- If you agree it is a contradiction, classify it as "updates" with high confidence.
- If you disagree, classify it as "extends" or "none" as appropriate and explain why the heuristic was wrong in your reasoning.
"#,
            ctx.candidate_memory_id, ctx.heuristic_result, ctx.candidate_content
        ),
        None => String::new(),
    };

    format!(
        r#"Analyze the relationship between a new memory and existing candidate memories.
For each candidate, determine if the new memory has a meaningful relationship with it.

Relationship Types:
- updates: The new memory CONTRADICTS or REPLACES information in the candidate memory.
  Example: Candidate "User prefers dark mode" + New "User now prefers light mode" = updates
- extends: The new memory ADDS information to the candidate memory without contradicting it.
  Example: Candidate "User is a developer" + New "User specializes in Rust" = extends
- none: No meaningful relationship exists between the memories.
{heuristic_section}
Candidate Memories:
{candidate_list}

New Memory:
{new_memory}

Respond with valid JSON only. Return an array of relationship objects.
Each object must have: memory_id, relation_type, confidence (0.0-1.0), reasoning.

Example format:
[
  {{"memory_id": "mem_123", "relation_type": "updates", "confidence": 0.95, "reasoning": "New memory contradicts the preference stated in candidate"}},
  {{"memory_id": "mem_456", "relation_type": "extends", "confidence": 0.85, "reasoning": "New memory adds specific detail to the general fact in candidate"}},
  {{"memory_id": "mem_789", "relation_type": "none", "confidence": 0.9, "reasoning": "Memories are about unrelated topics"}}
]"#,
    )
}

/// Generate a prompt for synthesizing insights from a seed memory and related memories
///
/// Returns a prompt that instructs the LLM to synthesize information from a primary
/// seed memory and a list of related memories. The LLM should produce a JSON object
/// containing a concise synthesized "content", an explanation of the "reasoning",
/// a numeric "confidence" between 0.0 and 1.0, and a list of source memory IDs in
/// "source_ids".
///
/// # Arguments
/// * `seed_memory` - The primary memory content to synthesize from
/// * `related_memories` - List of (memory_id, content) tuples for related memories
///
/// # Returns
/// A formatted prompt string ready for LLM completion
///
/// # Example
/// ```
/// use momo::llm::prompts::inference_generation_prompt;
///
/// let related = vec![("mem_1", "User prefers dark mode"), ("mem_2", "User uses vim")];
/// let prompt = inference_generation_prompt("User likes low-light UIs", &related);
/// assert!(prompt.contains("source_ids"));
/// ```
pub fn inference_generation_prompt(seed_memory: &str, related_memories: &[(&str, &str)]) -> String {
    let related_list = related_memories
        .iter()
        .map(|(id, content)| format!("[ID: {}] {}", id, content))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"Synthesize a concise, actionable insight from the following seed memory and related memories.
Use only the information provided. If information contradicts, prefer the most recent or specific fact and explain the reasoning.

Output MUST be valid JSON with the following fields:
- content: a single concise synthesized insight (string)
- reasoning: brief explanation of how the insight was derived from the sources (string)
- confidence: a number between 0.0 and 1.0 indicating confidence in the synthesis (float)
- source_ids: an array of source memory IDs that support the synthesized insight (array of strings)

Seed Memory:
{}

Related Memories:
{}

Respond with valid JSON only. Example format:
{{
  "content": "User prefers low-light UI and often uses modal-less editors",
  "reasoning": "Seed memory states preference for low-light UI; related memories indicate editor usage, suggesting a preference for modal-less, distraction-free editors",
  "confidence": 0.87,
  "source_ids": ["mem_1", "mem_2"]
}}
"#,
        seed_memory, related_list
    )
}

/// Generate a prompt for synthesizing a 3rd person narrative from user memories
///
/// Returns a prompt that instructs the LLM to weave a set of memories into a
/// cohesive 3rd person narrative biography of the user.
///
/// # Arguments
/// * `memories` - List of memory contents to synthesize
///
/// # Returns
/// * A formatted prompt string
pub fn narrative_generation_prompt(memories: &[&str]) -> String {
    let memory_list = memories
        .iter()
        .map(|m| format!("- {}", m))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"Construct a cohesive, third-person narrative biography of the user based STRICTLY on the following memories.
The narrative must be factual, concise, and objective.
1. Do NOT add any details, flowery language, or information not explicitly present in the memories.
2. Do NOT hallucinate personal traits or history.
3. Use a neutral, journalistic tone.
4. If memories contradict, note the conflict objectively.

Memories:
{}

Respond with valid JSON only. The JSON must contain a single field "narrative" with the generated text.
Example:
{{
  "narrative": "The user is a software engineer born in 1990. They prefer dark mode interfaces and listen to jazz music."
}}"#,
        memory_list
    )
}

/// Generate a prompt for compacting and organizing user facts
///
/// Returns a prompt that instructs the LLM to deduplicate, consolidate, and
/// categorize a list of facts about the user.
///
/// # Arguments
/// * `facts` - List of fact strings to compact
///
/// # Returns
/// * A formatted prompt string
pub fn fact_compaction_prompt(facts: &[&str]) -> String {
    let fact_list = facts
        .iter()
        .map(|f| format!("- {}", f))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"Analyze the following list of facts about the user.
1. Remove duplicates and redundant information.
2. Consolidate similar facts (e.g., "likes red" and "favorite color is red" -> "favorite color is red").
3. Group them into logical categories (e.g., "Professional", "Personal", "Preferences", "Technical").

Facts:
{}

Respond with valid JSON only. The output must be a key-value map where keys are category names and values are arrays of concise fact strings.
Example:
{{
  "Professional": ["Software Engineer at Tech Corp", "Specializes in Rust"],
  "Preferences": ["Likes dark mode", "Drinks coffee"]
}}"#,
        fact_list
    )
}

/// Generate a prompt for LLM-based content filtering
///
/// Returns a prompt that instructs the LLM to determine whether content should be
/// included or skipped based on a custom filter prompt. The LLM returns a JSON
/// decision with reasoning.
///
/// # Arguments
/// * `content` - The text content to evaluate
/// * `filter_prompt` - The filtering criteria (e.g., "Include only technical documentation")
///
/// # Returns
/// A formatted prompt string ready for LLM completion
///
/// # Example
/// ```
/// use momo::llm::prompts::llm_filter_prompt;
///
/// let prompt = llm_filter_prompt("Technical guide to Rust", "Include only technical docs");
/// assert!(prompt.contains("technical"));
/// ```
pub fn llm_filter_prompt(content: &str, filter_prompt: &str) -> String {
    let criteria = if filter_prompt.trim().is_empty() {
        "Include all content (no filtering applied)"
    } else {
        filter_prompt
    };

    format!(
        r#"Evaluate whether the following content should be included or skipped based on the filtering criteria.

Filtering Criteria:
{}

Content:
{}

Respond with valid JSON only. Your response must contain exactly two fields:
- "decision": Must be either "include" or "skip" (lowercase, no other values allowed)
- "reasoning": Brief explanation for your decision (string)

Example format:
{{
  "decision": "include",
  "reasoning": "Content matches the technical documentation criteria"
}}

Return valid JSON only."#,
        criteria, content
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_extraction_prompt_contains_content() {
        let content = "User prefers dark mode and uses vim";
        let prompt = memory_extraction_prompt(content);

        assert!(prompt.contains(content));
        assert!(prompt.contains("JSON"));
        assert!(prompt.contains("memory_type"));
        assert!(prompt.contains("confidence"));
    }

    #[test]
    fn test_query_rewrite_prompt_contains_query() {
        let query = "how to use rust";
        let prompt = query_rewrite_prompt(query);

        assert!(prompt.contains(query));
        assert!(prompt.contains("Rewrite"));
    }

    #[test]
    fn test_summarize_prompt_contains_length() {
        let content = "Long article text...";
        let max_length = 50;
        let prompt = summarize_prompt(content, max_length);

        assert!(prompt.contains(content));
        assert!(prompt.contains("50 words"));
    }

    #[test]
    fn test_memory_extraction_prompt_format() {
        let prompt = memory_extraction_prompt("test content");

        // Should have clear instructions
        assert!(prompt.contains("Extract"));
        assert!(prompt.contains("memories"));

        // Should specify JSON format
        assert!(prompt.contains("JSON"));
        assert!(prompt.contains("array"));

        // Should provide example format
        assert!(prompt.contains("Example"));

        // Should include memory type categories
        assert!(prompt.contains("Fact"));
        assert!(prompt.contains("Preference"));
        assert!(prompt.contains("Episode"));

        // Should include confidence field
        assert!(prompt.contains("confidence"));
    }

    #[test]
    fn test_memory_extraction_prompt_has_type_definitions() {
        let prompt = memory_extraction_prompt("test content");

        // Should define what each type means
        assert!(prompt.contains("Fact"));
        assert!(prompt.contains("Preference"));
        assert!(prompt.contains("Episode"));
        assert!(prompt.contains("Objective information") || prompt.contains("objective"));
    }

    #[test]
    fn test_conversation_extraction_prompt_formats_messages() {
        use crate::models::ConversationMessage;
        use chrono::Utc;

        let messages = vec![
            ConversationMessage {
                role: "user".to_string(),
                content: "I prefer dark mode".to_string(),
                timestamp: Some(Utc::now()),
            },
            ConversationMessage {
                role: "assistant".to_string(),
                content: "I'll remember that".to_string(),
                timestamp: Some(Utc::now()),
            },
        ];

        let prompt = conversation_extraction_prompt(&messages);

        // Should contain conversation header
        assert!(prompt.contains("Conversation"));

        // Should format messages with roles
        assert!(prompt.contains("user:") || prompt.contains("[user]"));
        assert!(prompt.contains("assistant:") || prompt.contains("[assistant]"));

        // Should contain message content
        assert!(prompt.contains("I prefer dark mode"));
        assert!(prompt.contains("I'll remember that"));

        // Should include extraction instructions
        assert!(prompt.contains("Extract"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn test_query_rewrite_prompt_format() {
        let prompt = query_rewrite_prompt("test query");

        // Should have clear instructions
        assert!(prompt.contains("Rewrite"));
        assert!(prompt.contains("semantic search"));

        // Should ask for no explanation
        assert!(prompt.contains("no explanation"));
    }

    #[test]
    fn test_summarize_prompt_format() {
        let prompt = summarize_prompt("test content", 100);

        // Should have clear instructions
        assert!(prompt.contains("Summarize"));
        assert!(prompt.contains("100 words"));

        // Should ask for no preamble
        assert!(prompt.contains("no preamble"));
    }

    #[test]
    fn test_relationship_detection_prompt_contains_candidates() {
        let candidates = vec![
            ("mem_123", "User prefers dark mode"),
            ("mem_456", "User is a software engineer"),
        ];
        let new_memory = "User now prefers light mode";
        let prompt = relationship_detection_prompt(new_memory, &candidates, None);

        assert!(prompt.contains(new_memory));
        assert!(prompt.contains("mem_123"));
        assert!(prompt.contains("mem_456"));
        assert!(prompt.contains("dark mode"));
        assert!(prompt.contains("software engineer"));
        assert!(prompt.contains("JSON"));
        assert!(prompt.contains("memory_id"));
        assert!(prompt.contains("relation_type"));
        assert!(prompt.contains("confidence"));
        assert!(prompt.contains("reasoning"));
        assert!(prompt.contains("updates"));
        assert!(prompt.contains("extends"));
        assert!(prompt.contains("none"));
    }

    #[test]
    fn test_relationship_detection_prompt_no_derives() {
        let candidates = vec![("mem_123", "User prefers dark mode")];
        let new_memory = "User likes coffee";
        let prompt = relationship_detection_prompt(new_memory, &candidates, None);

        // Should NOT include "derives" as a relation type
        assert!(!prompt.contains("derives"));

        // Should only have updates, extends, none
        let relation_count = prompt.matches("updates").count()
            + prompt.matches("extends").count()
            + prompt.matches("none").count();
        assert!(
            relation_count >= 3,
            "Should define all three relation types"
        );
    }

    #[test]
    fn test_inference_prompt_generation_contains_fields() {
        let related = vec![
            ("mem_1", "User prefers low-light UI"),
            ("mem_2", "User uses vim"),
        ];

        let seed = "User likes dim themes";
        let prompt = inference_generation_prompt(seed, &related);

        assert!(prompt.contains(seed));
        assert!(prompt.contains("mem_1"));
        assert!(prompt.contains("mem_2"));
        assert!(prompt.contains("JSON"));
        assert!(prompt.contains("content"));
        assert!(prompt.contains("reasoning"));
        assert!(prompt.contains("confidence"));
        assert!(prompt.contains("source_ids"));
    }

    #[test]
    fn test_relationship_detection_prompt_with_heuristic_context() {
        use crate::intelligence::contradiction::ContradictionCheckResult;
        use crate::intelligence::types::HeuristicContext;

        let candidates = vec![
            ("mem_123", "User prefers dark mode"),
            ("mem_456", "User is a software engineer"),
        ];
        let ctx = HeuristicContext {
            candidate_memory_id: "mem_123".to_string(),
            candidate_content: "User prefers dark mode".to_string(),
            heuristic_result: ContradictionCheckResult::Likely,
        };
        let prompt =
            relationship_detection_prompt("User now prefers light mode", &candidates, Some(&ctx));

        assert!(prompt.contains("Heuristic Contradiction Flag"));
        assert!(prompt.contains("mem_123"));
        assert!(prompt.contains("likely"));
        assert!(prompt.contains("confirm or override"));
    }

    #[test]
    fn test_relationship_detection_prompt_without_heuristic_context() {
        let candidates = vec![("mem_123", "User prefers dark mode")];
        let prompt =
            relationship_detection_prompt("User now prefers light mode", &candidates, None);

        assert!(!prompt.contains("Heuristic Contradiction Flag"));
        assert!(!prompt.contains("confirm or override"));
    }

    #[test]
    fn test_narrative_generation_prompt_format() {
        let memories = vec!["User was born in 1990", "User likes jazz music"];
        let prompt = narrative_generation_prompt(&memories);

        assert!(prompt.contains("narrative biography"));
        assert!(prompt.contains("User was born in 1990"));
        assert!(prompt.contains("User likes jazz music"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn test_fact_compaction_prompt_format() {
        let facts = vec!["User likes red", "Favorite color is red"];
        let prompt = fact_compaction_prompt(&facts);

        assert!(prompt.contains("Analyze the following list"));
        assert!(prompt.contains("Remove duplicates"));
        assert!(prompt.contains("User likes red"));
        assert!(prompt.contains("Favorite color is red"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn test_llm_filter_prompt_template() {
        let content = "Technical guide to Rust programming";
        let filter_prompt = "Include only technical documentation";
        let prompt = llm_filter_prompt(content, filter_prompt);

        assert!(prompt.contains(content));
        assert!(prompt.contains(filter_prompt));
        assert!(prompt.contains("decision"));
        assert!(prompt.contains("reasoning"));
        assert!(prompt.contains("include"));
        assert!(prompt.contains("skip"));
        assert!(prompt.contains("valid JSON only"));
    }

    #[test]
    fn test_llm_filter_prompt_empty_filter() {
        let content = "Any content here";
        let filter_prompt = "";
        let prompt = llm_filter_prompt(content, filter_prompt);

        assert!(prompt.contains(content));
        assert!(prompt.contains("Include all content"));
        assert!(prompt.contains("no filtering applied"));
        assert!(prompt.contains("decision"));
    }

    #[test]
    fn test_llm_filter_prompt_technical_content() {
        let content = "Rust memory safety and ownership model explained";
        let filter_prompt = "technical documentation";
        let prompt = llm_filter_prompt(content, filter_prompt);

        assert!(prompt.contains("Rust memory safety"));
        assert!(prompt.contains("technical documentation"));
    }

    #[test]
    fn test_llm_filter_prompt_marketing_content() {
        let content = "Buy our amazing product now! Limited time offer!";
        let filter_prompt = "marketing content";
        let prompt = llm_filter_prompt(content, filter_prompt);

        assert!(prompt.contains("Buy our amazing product"));
        assert!(prompt.contains("marketing content"));
    }

    #[test]
    fn test_llm_filter_prompt_json_structure() {
        let prompt = llm_filter_prompt("test content", "test filter");

        assert!(prompt.contains(r#""decision""#));
        assert!(prompt.contains(r#""reasoning""#));
        assert!(prompt.contains("Example format"));
        assert!(prompt.contains("lowercase"));
    }
}
