use unicode_segmentation::UnicodeSegmentation;

use crate::config::ProcessingConfig;

/// Context passed to chunkers for source file information
#[derive(Debug, Clone, Default)]
pub struct ChunkContext {
    pub source_path: Option<String>,
}

/// Trait for content chunking implementations
pub trait ContentChunker: Send + Sync {
    /// Chunk text content with optional context
    fn chunk(&self, text: &str, context: Option<&ChunkContext>) -> Vec<TextChunk>;
}

pub struct TextChunker {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl TextChunker {
    pub fn new(config: &ProcessingConfig) -> Self {
        Self {
            chunk_size: config.chunk_size,
            chunk_overlap: config.chunk_overlap,
        }
    }

    fn chunk_internal(&self, text: &str) -> Vec<TextChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        let sentences = self.split_into_sentences(text);
        self.merge_sentences_into_chunks(sentences)
    }

    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for grapheme in text.graphemes(true) {
            current.push_str(grapheme);

            if self.is_sentence_boundary(&current) {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        if !current.trim().is_empty() {
            sentences.push(current.trim().to_string());
        }

        sentences
    }

    fn is_sentence_boundary(&self, text: &str) -> bool {
        let trimmed = text.trim_end();
        if trimmed.is_empty() {
            return false;
        }

        let last_char = trimmed.chars().last().unwrap();

        if !matches!(last_char, '.' | '!' | '?' | '\n') {
            return false;
        }

        if last_char == '\n' {
            return true;
        }

        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if let Some(last_word) = words.last() {
            let abbreviations = [
                "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "etc.", "i.e.", "e.g.",
                "Inc.", "Ltd.", "Corp.", "Co.", "No.", "Vol.", "Ch.", "Fig.", "Eq.", "Sec.",
            ];

            if abbreviations.contains(last_word) {
                return false;
            }
        }

        true
    }

    fn merge_sentences_into_chunks(&self, sentences: Vec<String>) -> Vec<TextChunk> {
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_sentences: Vec<String> = Vec::new();

        for sentence in sentences {
            let potential_length = if current_chunk.is_empty() {
                sentence.len()
            } else {
                current_chunk.len() + 1 + sentence.len()
            };

            if potential_length > self.chunk_size && !current_chunk.is_empty() {
                chunks.push(TextChunk {
                    content: current_chunk.clone(),
                    token_count: Self::estimate_tokens(&current_chunk),
                });

                let overlap_sentences = self.get_overlap_sentences(&current_sentences);
                current_chunk = overlap_sentences.join(" ");
                current_sentences = overlap_sentences;
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(&sentence);
            current_sentences.push(sentence);
        }

        if !current_chunk.is_empty() {
            chunks.push(TextChunk {
                content: current_chunk.clone(),
                token_count: Self::estimate_tokens(&current_chunk),
            });
        }

        chunks
    }

    fn get_overlap_sentences(&self, sentences: &[String]) -> Vec<String> {
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut overlap_text_len = 0;
        let mut overlap_sentences = Vec::new();

        for sentence in sentences.iter().rev() {
            if overlap_text_len + sentence.len() > self.chunk_overlap
                && !overlap_sentences.is_empty()
            {
                break;
            }
            overlap_text_len += sentence.len() + 1;
            overlap_sentences.push(sentence.clone());
        }

        overlap_sentences.reverse();
        overlap_sentences
    }

    fn estimate_tokens(text: &str) -> i32 {
        (text.len() as f32 / 4.0).ceil() as i32
    }
}

impl ContentChunker for TextChunker {
    fn chunk(&self, text: &str, context: Option<&ChunkContext>) -> Vec<TextChunk> {
        // Ignore context for TextChunker (plain text doesn't need source path)
        let _ = context;
        self.chunk_internal(text)
    }
}

impl Default for TextChunker {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextChunk {
    pub content: String,
    pub token_count: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_chunker_basic() {
        let chunker = TextChunker::default();
        let text = "First sentence. Second sentence. Third sentence.";
        let context = ChunkContext { source_path: None };

        let chunks = chunker.chunk(text, Some(&context));

        assert!(!chunks.is_empty(), "Should produce chunks");
        assert!(
            chunks.iter().all(|c| !c.content.is_empty()),
            "All chunks should have content"
        );
    }

    #[test]
    fn test_text_chunker_empty_input() {
        let chunker = TextChunker::default();
        let chunks = chunker.chunk("", None);
        assert!(chunks.is_empty(), "Empty input should produce no chunks");
    }

    #[test]
    fn test_text_chunker_preserves_sentences() {
        let chunker = TextChunker::default();
        let text = "Hello world. This is a test.";
        let chunks = chunker.chunk(text, None);

        // Verify content is preserved
        let combined: String = chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            combined.contains("Hello world"),
            "Content should be preserved"
        );
    }

    #[test]
    fn test_text_chunker_ignores_context() {
        let chunker = TextChunker::default();
        let text = "Simple text.";

        // Same result with and without context
        let chunks_with = chunker.chunk(
            text,
            Some(&ChunkContext {
                source_path: Some("test.txt".into()),
            }),
        );
        let chunks_without = chunker.chunk(text, None);

        assert_eq!(
            chunks_with.len(),
            chunks_without.len(),
            "TextChunker should ignore context"
        );
    }

    #[test]
    fn test_chunk_context_creation() {
        let context = ChunkContext {
            source_path: Some("test.md".to_string()),
        };
        assert_eq!(context.source_path, Some("test.md".to_string()));
    }
}
