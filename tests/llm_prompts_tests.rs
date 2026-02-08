use momo::llm::prompts::{memory_extraction_prompt, query_rewrite_prompt, summarize_prompt};

#[test]
fn test_memory_extraction_prompt_substitution() {
    let content = "User prefers dark mode and uses vim for coding";
    let prompt = memory_extraction_prompt(content);

    assert!(prompt.contains(content));
    assert!(prompt.contains("Extract"));
    assert!(prompt.contains("JSON"));
    assert!(prompt.contains("memory_type"));
}

#[test]
fn test_memory_extraction_prompt_has_example_format() {
    let prompt = memory_extraction_prompt("test");

    assert!(prompt.contains("Example format"));
    assert!(prompt.contains(r#""content""#));
    assert!(prompt.contains(r#""memory_type""#));
}

#[test]
fn test_query_rewrite_prompt_substitution() {
    let query = "how to use rust async";
    let prompt = query_rewrite_prompt(query);

    assert!(prompt.contains(query));
    assert!(prompt.contains("Rewrite"));
    assert!(prompt.contains("semantic search"));
}

#[test]
fn test_query_rewrite_prompt_no_explanation() {
    let prompt = query_rewrite_prompt("test query");

    assert!(prompt.contains("no explanation"));
}

#[test]
fn test_summarize_prompt_substitution() {
    let content = "This is a long article about Rust programming";
    let max_length = 50;
    let prompt = summarize_prompt(content, max_length);

    assert!(prompt.contains(content));
    assert!(prompt.contains("50 words"));
    assert!(prompt.contains("Summarize"));
}

#[test]
fn test_summarize_prompt_no_preamble() {
    let prompt = summarize_prompt("test", 100);

    assert!(prompt.contains("no preamble"));
}

#[test]
fn test_all_prompts_are_non_empty() {
    assert!(!memory_extraction_prompt("test").is_empty());
    assert!(!query_rewrite_prompt("test").is_empty());
    assert!(!summarize_prompt("test", 50).is_empty());
}

#[test]
fn test_prompts_handle_special_characters() {
    let content_with_quotes = r#"User said "hello world""#;
    let prompt = memory_extraction_prompt(content_with_quotes);
    assert!(prompt.contains(content_with_quotes));

    let query_with_symbols = "rust: async/await & futures?";
    let prompt = query_rewrite_prompt(query_with_symbols);
    assert!(prompt.contains(query_with_symbols));
}

#[test]
fn test_summarize_prompt_different_lengths() {
    let content = "test content";

    let prompt_10 = summarize_prompt(content, 10);
    assert!(prompt_10.contains("10 words"));

    let prompt_100 = summarize_prompt(content, 100);
    assert!(prompt_100.contains("100 words"));

    let prompt_1000 = summarize_prompt(content, 1000);
    assert!(prompt_1000.contains("1000 words"));
}
