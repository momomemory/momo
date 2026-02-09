use super::{ChunkContext, ContentChunker, TextChunk};

/// Chunker for structured data formats like CSV and XLSX.
/// Preserves header rows in each chunk for context.
pub struct StructuredDataChunker {
    rows_per_chunk: usize,
}

impl StructuredDataChunker {
    /// Create a new chunker with specified rows per chunk
    #[allow(dead_code)]
    pub fn new(rows_per_chunk: usize) -> Self {
        Self { rows_per_chunk }
    }

    /// Estimate token count from text length
    fn estimate_tokens(text: &str) -> i32 {
        (text.len() as f32 / 4.0).ceil() as i32
    }
}

impl ContentChunker for StructuredDataChunker {
    fn chunk(&self, text: &str, _context: Option<&ChunkContext>) -> Vec<TextChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return Vec::new();
        }

        // First line is header
        let header = lines.first().copied().unwrap_or("");
        let data_rows = &lines[1..];

        if data_rows.is_empty() {
            // Only header, return single chunk
            return vec![TextChunk {
                content: text.to_string(),
                token_count: Self::estimate_tokens(text),
            }];
        }

        let mut chunks = Vec::new();

        for chunk_rows in data_rows.chunks(self.rows_per_chunk) {
            let mut chunk_content = String::new();
            chunk_content.push_str(header);
            chunk_content.push('\n');
            for row in chunk_rows {
                chunk_content.push_str(row);
                chunk_content.push('\n');
            }

            chunks.push(TextChunk {
                content: chunk_content.trim_end().to_string(),
                token_count: Self::estimate_tokens(&chunk_content),
            });
        }

        chunks
    }
}

impl Default for StructuredDataChunker {
    fn default() -> Self {
        Self { rows_per_chunk: 50 } // Default 50 rows per chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structured_chunker_csv_basic() {
        let chunker = StructuredDataChunker::default();
        let csv_text = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago";
        let chunks = chunker.chunk(csv_text, None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_structured_chunker_preserves_headers() {
        let chunker = StructuredDataChunker::default();
        let csv_text = "col1,col2\nval1,val2\nval3,val4";
        let chunks = chunker.chunk(csv_text, None);
        // Each chunk should include header context
        for chunk in &chunks {
            assert!(
                chunk.content.contains("col1") || chunk.content.contains("col2"),
                "Chunk should contain header: {}",
                chunk.content
            );
        }
    }

    #[test]
    fn test_structured_chunker_empty_input() {
        let chunker = StructuredDataChunker::default();
        let chunks = chunker.chunk("", None);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_structured_chunker_single_row() {
        let chunker = StructuredDataChunker::default();
        let csv_text = "header\nvalue";
        let chunks = chunker.chunk(csv_text, None);
        assert!(!chunks.is_empty());
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("header"));
        assert!(chunks[0].content.contains("value"));
    }

    #[test]
    fn test_structured_chunker_many_rows() {
        let chunker = StructuredDataChunker::new(10); // 10 rows per chunk
                                                      // 100 rows should produce multiple chunks
        let mut csv = String::from("id,name\n");
        for i in 0..100 {
            csv.push_str(&format!("{i},name{i}\n"));
        }
        let chunks = chunker.chunk(&csv, None);
        assert!(
            chunks.len() > 1,
            "Should produce multiple chunks for 100 rows"
        );
        // 100 data rows / 10 per chunk = 10 chunks
        assert_eq!(chunks.len(), 10);
    }

    #[test]
    fn test_structured_chunker_positions_increment() {
        let chunker = StructuredDataChunker::new(2);
        let csv_text = "h1,h2\na,b\nc,d\ne,f";
        let chunks = chunker.chunk(csv_text, None);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_structured_chunker_only_header() {
        let chunker = StructuredDataChunker::default();
        let csv_text = "col1,col2,col3";
        let chunks = chunker.chunk(csv_text, None);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, csv_text);
    }

    #[test]
    fn test_structured_chunker_token_count() {
        let chunker = StructuredDataChunker::default();
        let csv_text = "a,b\n1,2";
        let chunks = chunker.chunk(csv_text, None);
        assert_eq!(chunks.len(), 1);
        // token_count should be positive
        assert!(chunks[0].token_count > 0);
    }

    #[test]
    fn test_structured_chunker_with_context() {
        let chunker = StructuredDataChunker::default();
        let context = ChunkContext {
            source_path: Some("data.csv".to_string()),
        };
        let csv_text = "x,y\n1,2\n3,4";
        let chunks = chunker.chunk(csv_text, Some(&context));
        assert!(!chunks.is_empty());
    }
}
