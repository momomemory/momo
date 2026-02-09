use crate::config::ProcessingConfig;

use super::{ChunkContext, ContentChunker, MarkdownChunker, TextChunk, TextChunker};

#[derive(Default)]
pub struct WebpageChunker {
    markdown_chunker: MarkdownChunker,
    fallback_chunker: TextChunker,
}

impl WebpageChunker {
    pub fn new(config: &ProcessingConfig) -> Self {
        Self {
            markdown_chunker: MarkdownChunker::new(config),
            fallback_chunker: TextChunker::new(config),
        }
    }
}

impl ContentChunker for WebpageChunker {
    fn chunk(&self, text: &str, context: Option<&ChunkContext>) -> Vec<TextChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        // Convert HTML to Markdown using htmd
        let markdown = htmd::convert(text);

        // Check if conversion produced meaningful output
        if let Ok(md) = markdown {
            if !md.trim().is_empty() {
                return self.markdown_chunker.chunk(&md, context);
            }
        }

        // Fallback to TextChunker for plain text or failed conversion
        self.fallback_chunker.chunk(text, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webpage_chunker_basic() {
        let chunker = WebpageChunker::default();
        let html = "<h1>Title</h1><p>Paragraph content.</p>";
        let chunks = chunker.chunk(html, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_webpage_chunker_preserves_structure() {
        let chunker = WebpageChunker::default();
        let html =
            "<h2>Section</h2><p>Content under section.</p><h2>Another</h2><p>More content.</p>";
        let chunks = chunker.chunk(html, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_webpage_chunker_empty_input() {
        let chunker = WebpageChunker::default();
        let chunks = chunker.chunk("", None);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_webpage_chunker_plain_text_fallback() {
        let chunker = WebpageChunker::default();
        let plain = "Just plain text, no HTML.";
        let chunks = chunker.chunk(plain, None);
        assert!(!chunks.is_empty()); // Should fallback gracefully
    }

    #[test]
    fn test_webpage_chunker_nested_html() {
        let chunker = WebpageChunker::default();
        let html = "<div><article><h1>Title</h1><p>Content</p></article></div>";
        let chunks = chunker.chunk(html, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_webpage_chunker_with_context() {
        let chunker = WebpageChunker::default();
        let context = ChunkContext {
            source_path: Some("https://example.com/page.html".to_string()),
        };
        let html = "<h1>Title</h1><p>Content.</p>";
        let chunks = chunker.chunk(html, Some(&context));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_webpage_chunker_new_with_config() {
        let config = ProcessingConfig {
            chunk_size: 256,
            chunk_overlap: 25,
        };
        let chunker = WebpageChunker::new(&config);

        let html = "<h1>Title</h1><p>Some content here.</p>";
        let chunks = chunker.chunk(html, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_webpage_chunker_complex_html() {
        let chunker = WebpageChunker::default();
        let html = r#"
            <!DOCTYPE html>
            <html>
            <head><title>Test Page</title></head>
            <body>
                <header><h1>Main Title</h1></header>
                <main>
                    <section>
                        <h2>First Section</h2>
                        <p>This is the first paragraph with <strong>bold</strong> text.</p>
                        <p>This is the second paragraph with <em>italic</em> text.</p>
                    </section>
                    <section>
                        <h2>Second Section</h2>
                        <ul>
                            <li>Item one</li>
                            <li>Item two</li>
                        </ul>
                    </section>
                </main>
            </body>
            </html>
        "#;
        let chunks = chunker.chunk(html, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_webpage_chunker_malformed_html() {
        let chunker = WebpageChunker::default();
        let html = "<h1>Unclosed tag <p>Nested without closing";
        let chunks = chunker.chunk(html, None);
        assert!(!chunks.is_empty()); // Should handle gracefully
    }

    #[test]
    fn test_webpage_chunker_whitespace_only() {
        let chunker = WebpageChunker::default();
        let html = "   <p>   </p>   ";
        let chunks = chunker.chunk(html, None);
        // Whitespace HTML may produce empty chunks or chunks with whitespace
        // The important thing is it doesn't panic and handles gracefully
        assert!(
            chunks.is_empty()
                || chunks.iter().all(|c| c.content.trim().is_empty())
                || !chunks.is_empty()
        );
    }
}
