use crate::models::{Memory, MemoryType, SearchIncludeOptions};

/// Temporal search ranker that applies episode decay scoring
#[derive(Debug, Clone)]
pub struct TemporalSearchRanker {
    decay_days: f64,
    decay_factor: f64,
}

impl TemporalSearchRanker {
    /// Create a new TemporalSearchRanker with configurable decay parameters
    pub fn new(decay_days: f64, decay_factor: f64) -> Self {
        Self {
            decay_days,
            decay_factor,
        }
    }

    /// Apply episode decay scoring to a memory
    ///
    /// For Episode type memories: multiplies base_score by episode relevance
    /// For Fact/Preference types: returns base_score unchanged
    pub fn apply_episode_decay(&self, memory: &Memory, base_score: f32) -> f32 {
        match memory.memory_type {
            MemoryType::Episode => {
                let relevance =
                    memory.calculate_episode_relevance(self.decay_days, self.decay_factor);
                base_score * (relevance as f32)
            }
            _ => base_score,
        }
    }

    /// Check if forgotten memories should be included based on options
    #[allow(dead_code)]
    pub fn should_include_forgotten(&self, include_options: &SearchIncludeOptions) -> bool {
        include_options.forgotten_memories.unwrap_or(false)
    }
}

impl Default for TemporalSearchRanker {
    fn default() -> Self {
        Self::new(30.0, 0.9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_memory(memory_type: MemoryType, days_ago: i64) -> Memory {
        let mut memory = Memory::new(
            "test-id".to_string(),
            "test content".to_string(),
            "test-space".to_string(),
        );
        memory.memory_type = memory_type;
        memory.last_accessed = Some(Utc::now() - chrono::Duration::days(days_ago));
        memory
    }

    #[test]
    fn test_apply_episode_decay_reduces_old_episode_score() {
        let ranker = TemporalSearchRanker::default();
        let old_episode = create_test_memory(MemoryType::Episode, 60);

        let base_score = 1.0;
        let decayed = ranker.apply_episode_decay(&old_episode, base_score);

        assert!(
            decayed < base_score,
            "Old episode should have decayed score"
        );
        assert!(decayed > 0.0, "Score should be positive");
    }

    #[test]
    fn test_apply_episode_decay_fresh_episode_full_score() {
        let ranker = TemporalSearchRanker::default();
        let fresh_episode = create_test_memory(MemoryType::Episode, 0);

        let base_score = 0.8;
        let result = ranker.apply_episode_decay(&fresh_episode, base_score);

        assert!(
            (result - base_score).abs() < 0.01,
            "Fresh episode should have full score"
        );
    }

    #[test]
    fn test_apply_episode_decay_no_decay_for_facts() {
        let ranker = TemporalSearchRanker::default();
        let old_fact = create_test_memory(MemoryType::Fact, 60);

        let base_score = 0.9;
        let result = ranker.apply_episode_decay(&old_fact, base_score);

        assert!((result - base_score).abs() < 0.01, "Facts should not decay");
    }

    #[test]
    fn test_apply_episode_decay_no_decay_for_preferences() {
        let ranker = TemporalSearchRanker::default();
        let old_pref = create_test_memory(MemoryType::Preference, 60);

        let base_score = 0.85;
        let result = ranker.apply_episode_decay(&old_pref, base_score);

        assert!(
            (result - base_score).abs() < 0.01,
            "Preferences should not decay"
        );
    }

    #[test]
    fn test_should_include_forgotten_returns_true_when_flag_set() {
        let ranker = TemporalSearchRanker::default();
        let opts = SearchIncludeOptions {
            documents: None,
            summaries: None,
            related_memories: None,
            forgotten_memories: Some(true),
        };

        assert!(ranker.should_include_forgotten(&opts));
    }

    #[test]
    fn test_should_include_forgotten_returns_false_when_flag_unset() {
        let ranker = TemporalSearchRanker::default();
        let opts = SearchIncludeOptions {
            documents: None,
            summaries: None,
            related_memories: None,
            forgotten_memories: None,
        };

        assert!(!ranker.should_include_forgotten(&opts));
    }

    #[test]
    fn test_should_include_forgotten_returns_false_when_flag_false() {
        let ranker = TemporalSearchRanker::default();
        let opts = SearchIncludeOptions {
            documents: None,
            summaries: None,
            related_memories: None,
            forgotten_memories: Some(false),
        };

        assert!(!ranker.should_include_forgotten(&opts));
    }
}
