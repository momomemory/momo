use crate::config::ProcessingConfig;
use crate::models::DocumentType;

use super::{
    detect_language, CodeChunker, ContentChunker, MarkdownChunker, StructuredDataChunker,
    TextChunker, WebpageChunker,
};

/// Registry that routes documents to appropriate chunkers based on DocumentType.
/// Stores owned chunker instances and returns trait object references for dispatch.
#[derive(Default)]
pub struct ChunkerRegistry {
    text_chunker: TextChunker,
    code_chunker: CodeChunker,
    markdown_chunker: MarkdownChunker,
    webpage_chunker: WebpageChunker,
    structured_data_chunker: StructuredDataChunker,
}

impl ChunkerRegistry {
    /// Create a new registry with chunkers configured from ProcessingConfig
    pub fn new(config: &ProcessingConfig) -> Self {
        Self {
            text_chunker: TextChunker::new(config),
            code_chunker: CodeChunker::new(config),
            markdown_chunker: MarkdownChunker::new(config),
            webpage_chunker: WebpageChunker::new(config),
            structured_data_chunker: StructuredDataChunker::default(),
        }
    }

    /// Get the appropriate chunker for a document type and optional source path.
    /// Returns a trait object reference for zero-cost dispatch.
    pub fn get_chunker(
        &self,
        doc_type: &DocumentType,
        source_path: Option<&str>,
    ) -> &dyn ContentChunker {
        match doc_type {
            DocumentType::Code => {
                if let Some(path) = source_path {
                    if detect_language(path).is_some() {
                        return &self.code_chunker;
                    }
                }
                &self.text_chunker
            }
            DocumentType::Markdown => &self.markdown_chunker,
            DocumentType::Webpage => &self.webpage_chunker,
            DocumentType::Csv | DocumentType::Xlsx => &self.structured_data_chunker,
            DocumentType::Pdf | DocumentType::Docx | DocumentType::Pptx => &self.text_chunker,
            DocumentType::Text
            | DocumentType::Unknown
            | DocumentType::Tweet
            | DocumentType::GoogleDoc
            | DocumentType::GoogleSlide
            | DocumentType::GoogleSheet
            | DocumentType::NotionDoc
            | DocumentType::Onedrive
            | DocumentType::Image
            | DocumentType::Video
            | DocumentType::Audio => {
                if let Some(path) = source_path {
                    if detect_language(path).is_some() {
                        return &self.code_chunker;
                    }
                }
                &self.text_chunker
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DocumentType;

    #[test]
    fn test_registry_routes_markdown() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Markdown, None);
        let chunks = chunker.chunk("# Test", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_webpage() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Webpage, None);
        let chunks = chunker.chunk("<p>Test</p>", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_csv() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Csv, None);
        let chunks = chunker.chunk("a,b\n1,2", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_xlsx() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Xlsx, None);
        let chunks = chunker.chunk("a,b\n1,2", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_code_by_extension() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Code, Some("test.rs"));
        let chunks = chunker.chunk("fn main() {}", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_fallback_unknown_code_extension() {
        let registry = ChunkerRegistry::default();
        // Unknown extension should fall back to text chunker
        let chunker = registry.get_chunker(&DocumentType::Code, Some("test.xyz"));
        let chunks = chunker.chunk("Plain text", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_fallback_unknown() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Unknown, None);
        let chunks = chunker.chunk("Plain text", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_pdf() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Pdf, None);
        let chunks = chunker.chunk("PDF content here", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_text() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Text, None);
        let chunks = chunker.chunk("Plain text content", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_docx() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Docx, None);
        let chunks = chunker.chunk("Word document content", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_pptx() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Pptx, None);
        let chunks = chunker.chunk("PowerPoint content", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_new_with_config() {
        let config = ProcessingConfig {
            chunk_size: 1024,
            chunk_overlap: 100,
        };
        let registry = ChunkerRegistry::new(&config);
        let chunker = registry.get_chunker(&DocumentType::Text, None);
        let chunks = chunker.chunk("Test content", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_code_python() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Code, Some("script.py"));
        let chunks = chunker.chunk("def main(): pass", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_code_javascript() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Code, Some("app.js"));
        let chunks = chunker.chunk("function main() {}", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_code_typescript() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Code, Some("index.ts"));
        let chunks = chunker.chunk("const x: number = 1;", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_code_go() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Code, Some("main.go"));
        let chunks = chunker.chunk("func main() {}", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_code_java() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Code, Some("Main.java"));
        let chunks = chunker.chunk("public class Main {}", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_code_c() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Code, Some("main.c"));
        let chunks = chunker.chunk("int main() {}", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_code_cpp() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Code, Some("main.cpp"));
        let chunks = chunker.chunk("int main() {}", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_code_no_path() {
        let registry = ChunkerRegistry::default();
        // Code type without path should fall back to text chunker
        let chunker = registry.get_chunker(&DocumentType::Code, None);
        let chunks = chunker.chunk("Some code", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_tweet() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Tweet, None);
        let chunks = chunker.chunk("Tweet content", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_google_doc() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::GoogleDoc, None);
        let chunks = chunker.chunk("Google doc content", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_notion_doc() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::NotionDoc, None);
        let chunks = chunker.chunk("Notion doc content", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_image() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Image, None);
        let chunks = chunker.chunk("Image description", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_routes_video() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Video, None);
        let chunks = chunker.chunk("Video transcript", None);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_text_with_code_source_path_routes_to_code_chunker() {
        let registry = ChunkerRegistry::default();
        let context = super::super::ChunkContext {
            source_path: Some("main.rs".to_string()),
        };
        let chunker = registry.get_chunker(&DocumentType::Text, Some("main.rs"));
        let chunks = chunker.chunk("fn main() {}", Some(&context));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_unknown_with_code_source_path_routes_to_code_chunker() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Unknown, Some("app.py"));
        let context = super::super::ChunkContext {
            source_path: Some("app.py".to_string()),
        };
        let chunks = chunker.chunk("def main(): pass", Some(&context));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_registry_text_without_code_path_stays_text() {
        let registry = ChunkerRegistry::default();
        let chunker = registry.get_chunker(&DocumentType::Text, Some("notes.txt"));
        let chunks = chunker.chunk("Just some text", None);
        assert!(!chunks.is_empty());
    }
}
