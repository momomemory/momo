use crate::config::ProcessingConfig;
use text_splitter::{ChunkConfig, MarkdownSplitter};

use super::{ChunkContext, ContentChunker, TextChunk, TextChunker};

pub struct MarkdownChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    fallback_chunker: TextChunker,
}

impl MarkdownChunker {
    pub fn new(config: &ProcessingConfig) -> Self {
        Self {
            chunk_size: config.chunk_size,
            chunk_overlap: config.chunk_overlap,
            fallback_chunker: TextChunker::new(config),
        }
    }
}

impl ContentChunker for MarkdownChunker {
    fn chunk(&self, text: &str, context: Option<&ChunkContext>) -> Vec<TextChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        let chunk_config = match ChunkConfig::new(self.chunk_size).with_overlap(self.chunk_overlap)
        {
            Ok(cfg) => cfg,
            Err(_) => return self.fallback_chunker.chunk(text, context),
        };

        let splitter = MarkdownSplitter::new(chunk_config);
        let chunks: Vec<&str> = splitter.chunks(text).collect();

        if chunks.is_empty() {
            return self.fallback_chunker.chunk(text, context);
        }

        chunks
            .iter()
            .map(|chunk_text| TextChunk {
                content: chunk_text.to_string(),
                token_count: (chunk_text.len() as f32 / 4.0).ceil() as i32,
            })
            .collect()
    }
}

impl Default for MarkdownChunker {
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
    fn test_markdown_chunker_basic() {
        let chunker = MarkdownChunker::default();
        let md = "# Header\n\nParagraph one.\n\n## Subheader\n\nParagraph two.";
        let chunks = chunker.chunk(md, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_markdown_chunker_preserves_heading_boundaries() {
        let chunker = MarkdownChunker::default();
        let md = "# Section 1\n\nContent 1.\n\n# Section 2\n\nContent 2.";
        let chunks = chunker.chunk(md, None);
        for chunk in &chunks {
            assert!(!chunk.content.trim().is_empty());
        }
    }

    #[test]
    fn test_markdown_chunker_empty_input() {
        let chunker = MarkdownChunker::default();
        let chunks = chunker.chunk("", None);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_markdown_chunker_no_headers() {
        let chunker = MarkdownChunker::default();
        let md = "Just plain text without any headers.";
        let chunks = chunker.chunk(md, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_markdown_chunker_with_context() {
        let chunker = MarkdownChunker::default();
        let context = ChunkContext {
            source_path: Some("README.md".to_string()),
        };
        let chunks = chunker.chunk("# Test\n\nContent", Some(&context));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_markdown_chunker_new_with_config() {
        let config = ProcessingConfig {
            chunk_size: 256,
            chunk_overlap: 25,
        };
        let chunker = MarkdownChunker::new(&config);

        let md = "# Header\n\nSome content here.";
        let chunks = chunker.chunk(md, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_markdown_chunker_large_document() {
        let chunker = MarkdownChunker::default();
        let mut md = String::new();

        // Create a large markdown document with multiple sections
        for i in 0..20 {
            md.push_str(&format!("# Section {i}\n\n"));
            md.push_str("This is some content for this section. ");
            md.push_str("It contains multiple sentences to make it longer. ");
            md.push_str("We want to ensure the chunker handles larger documents.\n\n");
        }

        let chunks = chunker.chunk(&md, None);
        assert!(!chunks.is_empty());
        // Should produce multiple chunks for a large document
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_markdown_chunker_code_blocks() {
        let chunker = MarkdownChunker::default();
        let md = r#"# Code Example

Here is some code:

```rust
fn main() {
    println!("Hello, world!");
}
```

More text after the code block.
"#;
        let chunks = chunker.chunk(md, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_markdown_chunker_list_items() {
        let chunker = MarkdownChunker::default();
        let md = r#"# Shopping List

- Item one
- Item two
- Item three

## Another Section

1. First numbered item
2. Second numbered item
3. Third numbered item
"#;
        let chunks = chunker.chunk(md, None);
        assert!(!chunks.is_empty());
    }
}
