use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{HybridSearchResponse, MemoryRelationType, MemoryType, Metadata};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub memory: String,
    pub space_id: String,
    pub container_tag: Option<String>,
    pub version: i32,
    pub is_latest: bool,
    pub parent_memory_id: Option<String>,
    pub root_memory_id: Option<String>,
    pub memory_relations: HashMap<String, MemoryRelationType>,
    pub source_count: i32,
    pub is_inference: bool,
    pub is_forgotten: bool,
    pub is_static: bool,
    pub forget_after: Option<DateTime<Utc>>,
    pub forget_reason: Option<String>,
    pub memory_type: MemoryType,
    pub last_accessed: Option<DateTime<Utc>>,
    pub confidence: Option<f64>,
    pub metadata: Metadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Memory {
    pub fn new(id: String, memory: String, space_id: String) -> Self {
        let now = Utc::now();
        Self {
            id,
            memory,
            space_id,
            container_tag: None,
            version: 1,
            is_latest: true,
            parent_memory_id: None,
            root_memory_id: None,
            memory_relations: HashMap::new(),
            source_count: 0,
            is_inference: false,
            is_forgotten: false,
            is_static: false,
            forget_after: None,
            forget_reason: None,
            memory_type: MemoryType::default(),
            last_accessed: None,
            confidence: None,
            metadata: Metadata::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Calculate episode relevance using sigmoid-based decay.
    ///
    /// Uses formula: `1.0 / (1.0 + exp((days_since_access - midpoint) * steepness))`
    /// where `midpoint = decay_days` (day at which relevance = 0.5) and
    /// `steepness = -ln(1/decay_factor - 1) / decay_days`.
    ///
    /// Non-Episode types always return 1.0 (no decay).
    pub fn calculate_episode_relevance(&self, decay_days: f64, decay_factor: f64) -> f64 {
        if self.memory_type != MemoryType::Episode {
            return 1.0;
        }

        let last_access = self.last_accessed.unwrap_or(self.created_at);
        let days_since_access = (Utc::now() - last_access).num_days() as f64;

        if days_since_access <= 0.0 {
            return 1.0;
        }

        // Derive steepness from decay_factor: steepness = -ln(1/decay_factor - 1) / decay_days
        // Clamp decay_factor to avoid division by zero or negative log
        let clamped_factor = decay_factor.clamp(0.01, 0.99);
        let steepness = -(1.0_f64 / clamped_factor - 1.0).ln() / decay_days;

        // Sigmoid decay: relevance = 0.5 at decay_days, approaches 0 for old episodes
        1.0 / (1.0 + ((days_since_access - decay_days) * steepness).exp())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMemoryRequest {
    pub id: Option<String>,
    pub content: Option<String>,
    pub container_tag: String,
    pub new_content: String,
    pub metadata: Option<Metadata>,
    pub is_static: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMemoryResponse {
    pub id: String,
    pub memory: String,
    pub version: i32,
    pub parent_memory_id: Option<String>,
    pub root_memory_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetMemoryRequest {
    pub id: Option<String>,
    pub content: Option<String>,
    pub container_tag: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetMemoryResponse {
    pub id: String,
    pub forgotten: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    pub parents: Vec<MemoryRelationInfo>,
    pub children: Vec<MemoryRelationInfo>,
    pub related: Vec<MemoryRelationInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelationInfo {
    pub id: String,
    pub relation: MemoryRelationType,
    pub version: Option<i32>,
    pub memory: String,
    pub metadata: Option<Metadata>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ConversationRequest {
    pub messages: Vec<ConversationMessage>,
    pub container_tag: String,
    pub session_id: Option<String>,
    pub metadata: Option<Metadata>,
    pub memory_type: Option<MemoryType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
    pub timestamp: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationResponse {
    pub memories_extracted: i32,
    pub memory_ids: Vec<String>,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub container_tag: String,
    pub narrative: Option<String>,
    #[serde(rename = "static", alias = "static_facts")]
    pub static_facts: Vec<ProfileFact>,
    #[serde(rename = "dynamic", alias = "dynamic_facts")]
    pub dynamic_facts: Vec<ProfileFact>,
    pub total_memories: i32,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileFact {
    pub memory: String,
    pub confidence: Option<f64>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetProfileRequest {
    #[serde(alias = "containerTag")]
    pub container_tag: String,
    pub q: Option<String>,
    pub threshold: Option<f32>,
    pub include_dynamic: Option<bool>,
    pub limit: Option<u32>,
    pub compact: Option<bool>,
    pub generate_narrative: Option<bool>,
}

/// User profile data with static and dynamic fact arrays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfileData {
    #[serde(rename = "static")]
    pub static_facts: Vec<String>,
    #[serde(rename = "dynamic")]
    pub dynamic_facts: Vec<String>,
}

/// User profile response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResponse {
    pub profile: UserProfileData,
    #[serde(rename = "searchResults", skip_serializing_if = "Option::is_none")]
    pub search_results: Option<HybridSearchResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub narrative: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_new_default_type() {
        let memory = Memory::new("id1".to_string(), "test".to_string(), "space1".to_string());
        assert_eq!(memory.memory_type, MemoryType::Fact);
        assert_eq!(memory.last_accessed, None);
    }

    #[test]
    fn test_episode_relevance_no_decay_same_day() {
        let mut memory = Memory::new("id1".to_string(), "test".to_string(), "space1".to_string());
        memory.memory_type = MemoryType::Episode;
        memory.last_accessed = Some(Utc::now());

        let relevance = memory.calculate_episode_relevance(30.0, 0.9);
        assert_eq!(relevance, 1.0);
    }

    #[test]
    fn test_episode_relevance_at_midpoint() {
        let mut memory = Memory::new("id1".to_string(), "test".to_string(), "space1".to_string());
        memory.memory_type = MemoryType::Episode;
        memory.last_accessed = Some(Utc::now() - chrono::Duration::days(30));

        let relevance = memory.calculate_episode_relevance(30.0, 0.9);
        assert!(
            (relevance - 0.5).abs() < 0.01,
            "Sigmoid should be 0.5 at midpoint (decay_days), got {relevance}"
        );
    }

    #[test]
    fn test_fact_relevance_no_decay() {
        let mut memory = Memory::new("id1".to_string(), "test".to_string(), "space1".to_string());
        memory.memory_type = MemoryType::Fact;
        memory.last_accessed = Some(Utc::now() - chrono::Duration::days(90));

        let relevance = memory.calculate_episode_relevance(30.0, 0.9);
        assert_eq!(relevance, 1.0);
    }

    #[test]
    fn test_preference_relevance_no_decay() {
        let mut memory = Memory::new("id1".to_string(), "test".to_string(), "space1".to_string());
        memory.memory_type = MemoryType::Preference;
        memory.last_accessed = Some(Utc::now() - chrono::Duration::days(90));

        let relevance = memory.calculate_episode_relevance(30.0, 0.9);
        assert_eq!(relevance, 1.0);
    }

    #[test]
    fn test_episode_relevance_old_approaches_zero() {
        let mut memory = Memory::new("id1".to_string(), "test".to_string(), "space1".to_string());
        memory.memory_type = MemoryType::Episode;
        memory.last_accessed = Some(Utc::now() - chrono::Duration::days(90));

        let relevance = memory.calculate_episode_relevance(30.0, 0.9);
        assert!(
            relevance < 0.1,
            "Very old episode should have near-zero relevance, got {relevance}"
        );
    }

    #[test]
    fn test_episode_relevance_recent_near_one() {
        let mut memory = Memory::new("id1".to_string(), "test".to_string(), "space1".to_string());
        memory.memory_type = MemoryType::Episode;
        memory.last_accessed = Some(Utc::now() - chrono::Duration::days(1));

        let relevance = memory.calculate_episode_relevance(30.0, 0.9);
        assert!(
            relevance > 0.8,
            "Recent episode should have high relevance, got {relevance}"
        );
    }

    #[test]
    fn test_episode_relevance_sigmoid_monotonic_decrease() {
        let mut m1 = Memory::new("id1".to_string(), "test".to_string(), "space1".to_string());
        m1.memory_type = MemoryType::Episode;
        m1.last_accessed = Some(Utc::now() - chrono::Duration::days(10));

        let mut m2 = Memory::new("id2".to_string(), "test".to_string(), "space1".to_string());
        m2.memory_type = MemoryType::Episode;
        m2.last_accessed = Some(Utc::now() - chrono::Duration::days(30));

        let mut m3 = Memory::new("id3".to_string(), "test".to_string(), "space1".to_string());
        m3.memory_type = MemoryType::Episode;
        m3.last_accessed = Some(Utc::now() - chrono::Duration::days(60));

        let r1 = m1.calculate_episode_relevance(30.0, 0.9);
        let r2 = m2.calculate_episode_relevance(30.0, 0.9);
        let r3 = m3.calculate_episode_relevance(30.0, 0.9);

        assert!(r1 > r2, "10-day should beat 30-day: {r1} vs {r2}");
        assert!(r2 > r3, "30-day should beat 60-day: {r2} vs {r3}");
    }
}
