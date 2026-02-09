use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Metadata = HashMap<String, serde_json::Value>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingStatus {
    Unknown,
    #[default]
    Queued,
    Extracting,
    Chunking,
    Embedding,
    Indexing,
    Done,
    Failed,
}

impl std::fmt::Display for ProcessingStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => write!(f, "unknown"),
            Self::Queued => write!(f, "queued"),
            Self::Extracting => write!(f, "extracting"),
            Self::Chunking => write!(f, "chunking"),
            Self::Embedding => write!(f, "embedding"),
            Self::Indexing => write!(f, "indexing"),
            Self::Done => write!(f, "done"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

impl std::str::FromStr for ProcessingStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "unknown" => Ok(Self::Unknown),
            "queued" => Ok(Self::Queued),
            "extracting" => Ok(Self::Extracting),
            "chunking" => Ok(Self::Chunking),
            "embedding" => Ok(Self::Embedding),
            "indexing" => Ok(Self::Indexing),
            "done" => Ok(Self::Done),
            "failed" => Ok(Self::Failed),
            _ => Err(format!("Unknown processing status: {s}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DocumentType {
    #[default]
    Text,
    Pdf,
    Webpage,
    Tweet,
    GoogleDoc,
    GoogleSlide,
    GoogleSheet,
    NotionDoc,
    Onedrive,
    Image,
    Video,
    Audio,
    Markdown,
    Code,
    Csv,
    Docx,
    Pptx,
    Xlsx,
    Unknown,
}

impl std::fmt::Display for DocumentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Pdf => write!(f, "pdf"),
            Self::Webpage => write!(f, "webpage"),
            Self::Tweet => write!(f, "tweet"),
            Self::GoogleDoc => write!(f, "google_doc"),
            Self::GoogleSlide => write!(f, "google_slide"),
            Self::GoogleSheet => write!(f, "google_sheet"),
            Self::NotionDoc => write!(f, "notion_doc"),
            Self::Onedrive => write!(f, "onedrive"),
            Self::Image => write!(f, "image"),
            Self::Video => write!(f, "video"),
            Self::Audio => write!(f, "audio"),
            Self::Markdown => write!(f, "markdown"),
            Self::Code => write!(f, "code"),
            Self::Csv => write!(f, "csv"),
            Self::Docx => write!(f, "docx"),
            Self::Pptx => write!(f, "pptx"),
            Self::Xlsx => write!(f, "xlsx"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

impl std::str::FromStr for DocumentType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "pdf" => Ok(Self::Pdf),
            "webpage" | "web" => Ok(Self::Webpage),
            "tweet" => Ok(Self::Tweet),
            "google_doc" => Ok(Self::GoogleDoc),
            "google_slide" => Ok(Self::GoogleSlide),
            "google_sheet" => Ok(Self::GoogleSheet),
            "notion_doc" => Ok(Self::NotionDoc),
            "onedrive" => Ok(Self::Onedrive),
            "image" => Ok(Self::Image),
            "video" => Ok(Self::Video),
            "audio" => Ok(Self::Audio),
            "markdown" | "md" => Ok(Self::Markdown),
            "code" => Ok(Self::Code),
            "csv" => Ok(Self::Csv),
            "docx" => Ok(Self::Docx),
            "pptx" => Ok(Self::Pptx),
            "xlsx" => Ok(Self::Xlsx),
            _ => Ok(Self::Unknown),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRelationType {
    Updates,
    Extends,
    Derives,
}

impl std::fmt::Display for MemoryRelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Updates => write!(f, "updates"),
            Self::Extends => write!(f, "extends"),
            Self::Derives => write!(f, "derives"),
        }
    }
}

impl std::str::FromStr for MemoryRelationType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "updates" => Ok(Self::Updates),
            "extends" => Ok(Self::Extends),
            "derives" => Ok(Self::Derives),
            _ => Err(format!("Unknown relation type: {s}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    pub current_page: u32,
    pub limit: u32,
    pub total_items: u32,
    pub total_pages: u32,
}

impl Pagination {
    pub fn new(current_page: u32, limit: u32, total_items: u32) -> Self {
        let total_pages = total_items.div_ceil(limit);
        Self {
            current_page,
            limit,
            total_items,
            total_pages,
        }
    }
}

/// Type of memory extracted from content
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Factual information about the user or topic
    #[default]
    Fact,
    /// User preference or choice
    Preference,
    /// Event or experience
    Episode,
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fact => write!(f, "fact"),
            Self::Preference => write!(f, "preference"),
            Self::Episode => write!(f, "episode"),
        }
    }
}

impl std::str::FromStr for MemoryType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fact" => Ok(Self::Fact),
            "preference" => Ok(Self::Preference),
            "episode" => Ok(Self::Episode),
            _ => Err(format!("Unknown memory type: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_type_default() {
        assert_eq!(MemoryType::default(), MemoryType::Fact);
    }

    #[test]
    fn test_memory_type_display() {
        assert_eq!(MemoryType::Fact.to_string(), "fact");
        assert_eq!(MemoryType::Preference.to_string(), "preference");
        assert_eq!(MemoryType::Episode.to_string(), "episode");
    }

    #[test]
    fn test_memory_type_from_str() {
        assert_eq!("fact".parse::<MemoryType>().unwrap(), MemoryType::Fact);
        assert_eq!("Fact".parse::<MemoryType>().unwrap(), MemoryType::Fact);
        assert_eq!("FACT".parse::<MemoryType>().unwrap(), MemoryType::Fact);

        assert_eq!(
            "preference".parse::<MemoryType>().unwrap(),
            MemoryType::Preference
        );
        assert_eq!(
            "Preference".parse::<MemoryType>().unwrap(),
            MemoryType::Preference
        );

        assert_eq!(
            "episode".parse::<MemoryType>().unwrap(),
            MemoryType::Episode
        );
        assert_eq!(
            "Episode".parse::<MemoryType>().unwrap(),
            MemoryType::Episode
        );

        assert!("invalid".parse::<MemoryType>().is_err());
    }

    #[test]
    fn test_memory_type_serialization() {
        let fact = MemoryType::Fact;
        let json = serde_json::to_string(&fact).unwrap();
        assert_eq!(json, "\"fact\"");

        let preference = MemoryType::Preference;
        let json = serde_json::to_string(&preference).unwrap();
        assert_eq!(json, "\"preference\"");

        let episode = MemoryType::Episode;
        let json = serde_json::to_string(&episode).unwrap();
        assert_eq!(json, "\"episode\"");
    }

    #[test]
    fn test_memory_type_deserialization() {
        let fact: MemoryType = serde_json::from_str("\"fact\"").unwrap();
        assert_eq!(fact, MemoryType::Fact);

        let preference: MemoryType = serde_json::from_str("\"preference\"").unwrap();
        assert_eq!(preference, MemoryType::Preference);

        let episode: MemoryType = serde_json::from_str("\"episode\"").unwrap();
        assert_eq!(episode, MemoryType::Episode);
    }

    #[test]
    fn test_memory_type_clone_copy() {
        let fact = MemoryType::Fact;
        let fact_copy = fact;
        assert_eq!(fact, fact_copy);
    }
}
