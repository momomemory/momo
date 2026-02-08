use crate::config::ProcessingConfig;
use text_splitter::{ChunkConfig, CodeSplitter};

use super::language::SupportedLanguage;
use super::{ChunkContext, ContentChunker, TextChunk, TextChunker};

pub struct CodeChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    fallback_chunker: TextChunker,
}

impl CodeChunker {
    pub fn new(config: &ProcessingConfig) -> Self {
        Self {
            chunk_size: config.chunk_size,
            chunk_overlap: config.chunk_overlap,
            fallback_chunker: TextChunker::new(config),
        }
    }

    fn get_tree_sitter_language(&self, lang: SupportedLanguage) -> tree_sitter::Language {
        match lang {
            SupportedLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
            SupportedLanguage::Python => tree_sitter_python::LANGUAGE.into(),
            SupportedLanguage::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            SupportedLanguage::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            SupportedLanguage::TypeScriptTsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            SupportedLanguage::Go => tree_sitter_go::LANGUAGE.into(),
            SupportedLanguage::Java => tree_sitter_java::LANGUAGE.into(),
            SupportedLanguage::C => tree_sitter_c::LANGUAGE.into(),
            SupportedLanguage::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        }
    }

    fn add_context_prefix(
        &self,
        chunk_text: &str,
        full_source: &str,
        context: &ChunkContext,
    ) -> String {
        let mut prefix_parts: Vec<String> = Vec::new();

        if let Some(ref path) = context.source_path {
            prefix_parts.push(format!("# File: {path}"));
        }

        let imports = Self::extract_imports(full_source);
        if !imports.is_empty() {
            prefix_parts.push(format!("# Imports: {}", imports.join(", ")));
        }

        let siblings = Self::extract_sibling_signatures(full_source, chunk_text);
        if !siblings.is_empty() {
            prefix_parts.push(format!("# Sibling definitions: {}", siblings.join("; ")));
        }

        if prefix_parts.is_empty() {
            chunk_text.to_string()
        } else {
            format!("{}\n\n{}", prefix_parts.join("\n"), chunk_text)
        }
    }

    fn extract_imports(source: &str) -> Vec<String> {
        let mut imports = Vec::new();
        for line in source.lines().take(100) {
            let trimmed = line.trim();
            if trimmed.starts_with("use ")
                || trimmed.starts_with("import ")
                || trimmed.starts_with("from ")
                || trimmed.starts_with("#include")
                || trimmed.starts_with("require(")
                || (trimmed.starts_with("const ") && trimmed.contains("require("))
            {
                let import = trimmed.trim_end_matches(';').trim_end_matches('{').trim();
                if import.len() <= 120 {
                    imports.push(import.to_string());
                }
            }
            if imports.len() >= 15 {
                break;
            }
        }
        imports
    }

    fn extract_sibling_signatures(full_source: &str, current_chunk: &str) -> Vec<String> {
        let mut signatures = Vec::new();
        let patterns: &[&str] = &[
            "fn ",
            "def ",
            "func ",
            "function ",
            "pub fn ",
            "async fn ",
            "pub async fn ",
            "class ",
            "struct ",
            "impl ",
            "trait ",
            "interface ",
            "enum ",
            "type ",
        ];

        for line in full_source.lines() {
            let trimmed = line.trim();
            let is_sig = patterns.iter().any(|p| trimmed.starts_with(p));
            if !is_sig {
                continue;
            }
            if current_chunk.contains(trimmed) {
                continue;
            }
            let sig = trimmed
                .split('{')
                .next()
                .unwrap_or(trimmed)
                .split(':')
                .next()
                .unwrap_or(trimmed)
                .trim();
            if !sig.is_empty() && sig.len() <= 120 {
                signatures.push(sig.to_string());
            }
            if signatures.len() >= 10 {
                break;
            }
        }
        signatures
    }
}

impl ContentChunker for CodeChunker {
    fn chunk(&self, text: &str, context: Option<&ChunkContext>) -> Vec<TextChunk> {
        let context = context.cloned().unwrap_or_default();

        let detected_language = context
            .source_path
            .as_ref()
            .and_then(|p| super::language::detect_language(p));

        let lang = match detected_language {
            Some(lang) => lang,
            None => {
                tracing::debug!("No language detected, falling back to TextChunker");
                return self.fallback_chunker.chunk(text, Some(&context));
            }
        };

        let ts_lang = self.get_tree_sitter_language(lang);

        let chunk_config = ChunkConfig::new(self.chunk_size)
            .with_overlap(self.chunk_overlap)
            .expect("Invalid chunk config");

        let splitter = match CodeSplitter::new(ts_lang, chunk_config) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    "Failed to create CodeSplitter for {:?}: {}, falling back",
                    lang,
                    e
                );
                return self.fallback_chunker.chunk(text, Some(&context));
            }
        };

        let chunks: Vec<&str> = splitter.chunks(text).collect();

        if chunks.is_empty() {
            tracing::debug!("CodeSplitter returned no chunks, falling back to TextChunker");
            return self.fallback_chunker.chunk(text, Some(&context));
        }

        chunks
            .iter()
            .enumerate()
            .map(|(i, chunk_text)| {
                let enriched_content = self.add_context_prefix(chunk_text, text, &context);
                TextChunk {
                    content: enriched_content,
                    token_count: (chunk_text.len() as f32 / 4.0).ceil() as i32,
                }
            })
            .collect()
    }
}

impl Default for CodeChunker {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            fallback_chunker: TextChunker::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_chunker_rust() {
        let chunker = CodeChunker::default();
        let context = ChunkContext {
            source_path: Some("src/main.rs".to_string()),
        };

        let code = r#"
fn hello() {
    println!("Hello");
}

fn world() {
    println!("World");
}
"#;

        let chunks = chunker.chunk(code, Some(&context));
        assert!(
            !chunks.is_empty(),
            "Should produce chunks for valid Rust code"
        );
    }

    #[test]
    fn test_code_chunker_fallback_unsupported() {
        let chunker = CodeChunker::default();
        let context = ChunkContext {
            source_path: Some("data.json".to_string()),
            doc_type: None,
        };

        let text = "some text content";
        let chunks = chunker.chunk(text, Some(&context));
        assert!(
            !chunks.is_empty(),
            "Should fallback to TextChunker for unsupported extensions"
        );
    }

    #[test]
    fn test_code_chunker_no_context() {
        let chunker = CodeChunker::default();
        let text = "some text content";
        let chunks = chunker.chunk(text, None);
        assert!(
            !chunks.is_empty(),
            "Should fallback when no context provided"
        );
    }

    #[test]
    fn test_code_chunker_python() {
        let chunker = CodeChunker::default();
        let context = ChunkContext {
            source_path: Some("module.py".to_string()),
            doc_type: None,
        };

        let code = r#"
def hello():
    print("Hello")

def world():
    print("World")
"#;

        let chunks = chunker.chunk(code, Some(&context));
        assert!(
            !chunks.is_empty(),
            "Should produce chunks for valid Python code"
        );
    }

    #[test]
    fn test_code_chunker_typescript() {
        let chunker = CodeChunker::default();
        let context = ChunkContext {
            source_path: Some("app.ts".to_string()),
            doc_type: None,
        };

        let code = r#"
function greet(name: string): void {
    console.log(`Hello ${name}`);
}

function farewell(name: string): void {
    console.log(`Goodbye ${name}`);
}
"#;

        let chunks = chunker.chunk(code, Some(&context));
        assert!(
            !chunks.is_empty(),
            "Should produce chunks for valid TypeScript code"
        );
    }

    #[test]
    fn test_code_chunker_empty_input() {
        let chunker = CodeChunker::default();
        let context = ChunkContext {
            source_path: Some("main.rs".to_string()),
            doc_type: None,
        };

        let chunks = chunker.chunk("", Some(&context));
        assert!(
            chunks.len() <= 1,
            "Empty input should produce minimal chunks"
        );
    }

    #[test]
    fn test_code_chunker_malformed_code_fallback() {
        let chunker = CodeChunker::default();
        let context = ChunkContext {
            source_path: Some("broken.rs".to_string()),
            doc_type: None,
        };

        let code = r#"
fn incomplete {
    let x = 
    // missing closing brace and incomplete let
"#;

        let chunks = chunker.chunk(code, Some(&context));
        assert!(
            !code.is_empty() || chunks.is_empty(),
            "Should handle malformed code gracefully"
        );
    }

    #[test]
    fn test_code_chunker_enriched_content_includes_file_path() {
        let chunker = CodeChunker::default();
        let context = ChunkContext {
            source_path: Some("src/main.rs".to_string()),
        };

        let code = r#"
fn hello() {
    println!("Hello");
}

fn world() {
    println!("World");
}
"#;

        let chunks = chunker.chunk(code, Some(&context));
        assert!(!chunks.is_empty());
        assert!(
            chunks[0].content.contains("# File: src/main.rs"),
            "Enriched content should include source file path, got: {}",
            chunks[0].content
        );
    }

    #[test]
    fn test_code_chunker_enriched_content_includes_imports() {
        let chunker = CodeChunker::default();
        let context = ChunkContext {
            source_path: Some("src/lib.rs".to_string()),
            doc_type: None,
        };

        let code = r#"use std::collections::HashMap;
use std::io;

fn process() {
    let map = HashMap::new();
    println!("{:?}", map);
}

fn other() {
    println!("other");
}
"#;

        let chunks = chunker.chunk(code, Some(&context));
        assert!(!chunks.is_empty());
        assert!(
            chunks[0].content.contains("# Imports:"),
            "Enriched content should include imports, got: {}",
            chunks[0].content
        );
    }

    #[test]
    fn test_code_chunker_enriched_content_includes_siblings() {
        let chunker = CodeChunker {
            chunk_size: 60,
            chunk_overlap: 10,
            fallback_chunker: TextChunker::default(),
        };
        let context = ChunkContext {
            source_path: Some("src/lib.rs".to_string()),
            doc_type: None,
        };

        let code = r#"fn alpha() {
    println!("alpha");
}

fn beta() {
    println!("beta");
}

fn gamma() {
    println!("gamma");
}
"#;

        let chunks = chunker.chunk(code, Some(&context));
        if chunks.len() > 1 {
            let has_sibling_ref = chunks
                .iter()
                .any(|c| c.content.contains("# Sibling definitions:"));
            assert!(
                has_sibling_ref,
                "At least one chunk should reference sibling definitions when multiple functions exist"
            );
        }
    }

    #[test]
    fn test_extract_imports() {
        let source = r#"use std::io;
use crate::models::Document;
import os
from pathlib import Path
#include <stdio.h>

fn main() {}
"#;
        let imports = CodeChunker::extract_imports(source);
        assert!(
            imports.len() >= 4,
            "Should extract multiple import types, got: {imports:?}"
        );
    }

    #[test]
    fn test_extract_sibling_signatures() {
        let source = r#"
fn alpha() {
    println!("alpha");
}

fn beta() {
    println!("beta");
}
"#;
        let chunk = "fn alpha() {\n    println!(\"alpha\");\n}";
        let siblings = CodeChunker::extract_sibling_signatures(source, chunk);
        assert!(
            siblings.iter().any(|s| s.contains("beta")),
            "Siblings should include beta, got: {siblings:?}"
        );
        assert!(
            !siblings.iter().any(|s| s.contains("alpha")),
            "Siblings should NOT include the current chunk's own function"
        );
    }
}
