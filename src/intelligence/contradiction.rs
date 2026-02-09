use crate::intelligence::utils::content_overlap_score;
use serde::{Deserialize, Serialize};

/// Result of a heuristic contradiction check between two memory contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContradictionCheckResult {
    /// No contradiction detected
    None,
    /// Weak signal — possible but uncertain contradiction
    Unlikely,
    /// Strong signal — pattern-matched contradiction
    Likely,
}

impl std::fmt::Display for ContradictionCheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Unlikely => write!(f, "unlikely"),
            Self::Likely => write!(f, "likely"),
        }
    }
}

impl ContradictionCheckResult {
    #[allow(dead_code)]
    pub fn is_contradiction(&self) -> bool {
        matches!(self, Self::Likely)
    }
}

/// A pair of antonyms used for contradiction detection.
#[derive(Debug, Clone)]
pub struct AntonymPair {
    pub word_a: &'static str,
    pub word_b: &'static str,
}

/// Common antonym pairs for heuristic contradiction detection.
const ANTONYM_PAIRS: &[AntonymPair] = &[
    AntonymPair {
        word_a: "love",
        word_b: "hate",
    },
    AntonymPair {
        word_a: "like",
        word_b: "dislike",
    },
    AntonymPair {
        word_a: "hot",
        word_b: "cold",
    },
    AntonymPair {
        word_a: "always",
        word_b: "never",
    },
    AntonymPair {
        word_a: "happy",
        word_b: "sad",
    },
    AntonymPair {
        word_a: "good",
        word_b: "bad",
    },
    AntonymPair {
        word_a: "fast",
        word_b: "slow",
    },
    AntonymPair {
        word_a: "big",
        word_b: "small",
    },
    AntonymPair {
        word_a: "tall",
        word_b: "short",
    },
    AntonymPair {
        word_a: "light",
        word_b: "dark",
    },
    AntonymPair {
        word_a: "open",
        word_b: "closed",
    },
    AntonymPair {
        word_a: "true",
        word_b: "false",
    },
    AntonymPair {
        word_a: "yes",
        word_b: "no",
    },
    AntonymPair {
        word_a: "enable",
        word_b: "disable",
    },
    AntonymPair {
        word_a: "enabled",
        word_b: "disabled",
    },
    AntonymPair {
        word_a: "active",
        word_b: "inactive",
    },
    AntonymPair {
        word_a: "prefer",
        word_b: "avoid",
    },
    AntonymPair {
        word_a: "start",
        word_b: "stop",
    },
    AntonymPair {
        word_a: "accept",
        word_b: "reject",
    },
    AntonymPair {
        word_a: "allow",
        word_b: "block",
    },
];

/// Negation patterns that flip the meaning of a statement.
#[derive(Debug, Clone)]
pub enum NegationPattern {
    /// "doesn't", "does not"
    DoesNot,
    /// "don't", "do not"
    DoNot,
    /// "isn't", "is not"
    IsNot,
    /// "wasn't", "was not"
    WasNot,
    /// "won't", "will not"
    WillNot,
    /// "can't", "cannot"
    Cannot,
    /// "never"
    Never,
    /// "no longer"
    NoLonger,
    /// "hates" (negates positive sentiment)
    Hates,
    /// "dislikes" (negates positive sentiment)
    Dislikes,
    /// "not"
    Not,
}

impl NegationPattern {
    /// All negation phrases this pattern matches (lowercase).
    fn phrases(&self) -> &[&str] {
        match self {
            Self::DoesNot => &["doesn't", "does not"],
            Self::DoNot => &["don't", "do not"],
            Self::IsNot => &["isn't", "is not"],
            Self::WasNot => &["wasn't", "was not"],
            Self::WillNot => &["won't", "will not"],
            Self::Cannot => &["can't", "cannot", "can not"],
            Self::Never => &["never"],
            Self::NoLonger => &["no longer"],
            Self::Hates => &["hates"],
            Self::Dislikes => &["dislikes"],
            Self::Not => &["not"],
        }
    }

    /// All known negation patterns.
    fn all() -> &'static [NegationPattern] {
        &[
            NegationPattern::DoesNot,
            NegationPattern::DoNot,
            NegationPattern::IsNot,
            NegationPattern::WasNot,
            NegationPattern::WillNot,
            NegationPattern::Cannot,
            NegationPattern::Never,
            NegationPattern::NoLonger,
            NegationPattern::Hates,
            NegationPattern::Dislikes,
            NegationPattern::Not,
        ]
    }
}

/// Heuristic contradiction detector using pattern matching.
///
/// This detector is intentionally simple and fast — no embeddings, no LLM calls,
/// no external API calls. It uses string pattern matching to catch obvious
/// contradictions before memory creation.
pub struct ContradictionDetector;

impl ContradictionDetector {
    pub fn new() -> Self {
        Self
    }

    /// Check whether two memory contents are contradictory using heuristics.
    ///
    /// Returns `ContradictionCheckResult::Likely` for strong pattern matches,
    /// `Unlikely` for weak signals, and `None` when no contradiction detected.
    pub fn check_contradiction(
        &self,
        existing_content: &str,
        new_content: &str,
    ) -> ContradictionCheckResult {
        let existing_lower = existing_content.to_lowercase();
        let new_lower = new_content.to_lowercase();

        // Exact same content is not a contradiction
        if existing_lower == new_lower {
            return ContradictionCheckResult::None;
        }

        // Check negation-based contradictions
        if let Some(result) = self.check_negation_contradiction(&existing_lower, &new_lower) {
            return result;
        }

        // Check antonym-based contradictions
        if let Some(result) = self.check_antonym_contradiction(&existing_lower, &new_lower) {
            return result;
        }

        // Check value-swap contradictions (e.g., "X is A" vs "X is B")
        if let Some(result) = self.check_value_contradiction(&existing_lower, &new_lower) {
            return result;
        }

        ContradictionCheckResult::None
    }

    /// Detect contradictions where one statement negates the other.
    ///
    /// Examples:
    /// - "User likes Python" vs "User doesn't like Python"
    /// - "User loves coffee" vs "User hates coffee"
    fn check_negation_contradiction(
        &self,
        existing: &str,
        new: &str,
    ) -> Option<ContradictionCheckResult> {
        for pattern in NegationPattern::all() {
            for phrase in pattern.phrases() {
                // Case 1: existing is positive, new contains negation
                if !contains_any_negation(existing) && new.contains(phrase) {
                    let stripped = strip_negation(new, phrase);
                    if crate::intelligence::utils::fuzzy_overlap_score(existing, &stripped) >= 0.5 {
                        return Some(ContradictionCheckResult::Likely);
                    }
                }

                // Case 2: existing contains negation, new is positive
                if existing.contains(phrase) && !contains_any_negation(new) {
                    let stripped = strip_negation(existing, phrase);
                    if crate::intelligence::utils::fuzzy_overlap_score(new, &stripped) >= 0.5 {
                        return Some(ContradictionCheckResult::Likely);
                    }
                }
            }
        }

        // Sentiment-flip patterns: "likes" → "hates", "loves" → "dislikes"
        if self.check_sentiment_flip(existing, new) {
            return Some(ContradictionCheckResult::Likely);
        }

        None
    }

    /// Check for sentiment-flip contradictions using verb substitution.
    ///
    /// E.g., "User likes X" vs "User hates X"
    fn check_sentiment_flip(&self, existing: &str, new: &str) -> bool {
        const POSITIVE_NEGATIVE_PAIRS: &[(&str, &str)] = &[
            ("likes", "hates"),
            ("likes", "dislikes"),
            ("loves", "hates"),
            ("loves", "dislikes"),
            ("enjoys", "hates"),
            ("enjoys", "dislikes"),
            ("prefers", "avoids"),
            ("wants", "doesn't want"),
        ];

        for &(positive, negative) in POSITIVE_NEGATIVE_PAIRS {
            // existing has positive, new has negative (or vice versa)
            if (existing.contains(positive) && new.contains(negative))
                || (existing.contains(negative) && new.contains(positive))
            {
                // Verify the rest of the content is similar
                let existing_stripped = existing.replace(positive, "").replace(negative, "");
                let new_stripped = new.replace(positive, "").replace(negative, "");
                if crate::intelligence::utils::content_overlap_score(
                    &existing_stripped,
                    &new_stripped,
                ) > 0.5
                {
                    return true;
                }
            }
        }
        false
    }

    /// Detect contradictions based on antonym pairs in similar contexts.
    ///
    /// Examples:
    /// - "The weather is hot" vs "The weather is cold"
    /// - "User prefers light mode" vs "User prefers dark mode"
    fn check_antonym_contradiction(
        &self,
        existing: &str,
        new: &str,
    ) -> Option<ContradictionCheckResult> {
        for pair in ANTONYM_PAIRS {
            let a_in_existing = existing.contains(pair.word_a);
            let b_in_existing = existing.contains(pair.word_b);
            let a_in_new = new.contains(pair.word_a);
            let b_in_new = new.contains(pair.word_b);

            // One has word_a, other has word_b (cross-match)
            if (a_in_existing && b_in_new && !b_in_existing && !a_in_new)
                || (b_in_existing && a_in_new && !a_in_existing && !b_in_new)
            {
                // Verify the surrounding context is similar
                let existing_stripped = existing.replace(pair.word_a, "").replace(pair.word_b, "");
                let new_stripped = new.replace(pair.word_a, "").replace(pair.word_b, "");

                let overlap = crate::intelligence::utils::content_overlap_score(
                    &existing_stripped,
                    &new_stripped,
                );
                if overlap > 0.5 {
                    return Some(ContradictionCheckResult::Likely);
                } else if overlap > 0.3 {
                    return Some(ContradictionCheckResult::Unlikely);
                }
            }
        }
        None
    }

    /// Detect contradictions where the same subject has different values.
    ///
    /// Pattern: "X is A" vs "X is B" where A != B
    /// Example: "User's favorite color is blue" vs "User's favorite color is red"
    fn check_value_contradiction(
        &self,
        existing: &str,
        new: &str,
    ) -> Option<ContradictionCheckResult> {
        // Look for "is" or "are" pivot patterns
        for pivot in &[" is ", " are ", " was ", " were "] {
            if let (Some(ex_pos), Some(new_pos)) = (existing.find(pivot), new.find(pivot)) {
                let ex_subject = &existing[..ex_pos];
                let new_subject = &new[..new_pos];
                let ex_value = &existing[ex_pos + pivot.len()..];
                let new_value = &new[new_pos + pivot.len()..];

                let ex_val_trimmed = ex_value.trim();
                let new_val_trimmed = new_value.trim();

                // Same subject but different value
                if content_overlap_score(ex_subject, new_subject) > 0.7
                    && !ex_val_trimmed.is_empty()
                    && !new_val_trimmed.is_empty()
                    && ex_val_trimmed != new_val_trimmed
                    // Skip if one value contains the other (extension, not contradiction)
                    && !ex_val_trimmed.contains(new_val_trimmed)
                    && !new_val_trimmed.contains(ex_val_trimmed)
                    // Skip if all words of one value appear in the other (superset = extension)
                    && !is_word_subset(ex_val_trimmed, new_val_trimmed)
                    && !is_word_subset(new_val_trimmed, ex_val_trimmed)
                {
                    // Different values for same subject → possible contradiction
                    return Some(ContradictionCheckResult::Unlikely);
                }
            }
        }
        None
    }
}

impl Default for ContradictionDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a string contains any negation phrase.
fn contains_any_negation(s: &str) -> bool {
    for pattern in NegationPattern::all() {
        for phrase in pattern.phrases() {
            if s.contains(phrase) {
                return true;
            }
        }
    }
    false
}

/// Remove a negation phrase from a string and clean up whitespace.
fn strip_negation(s: &str, negation: &str) -> String {
    s.replace(negation, "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns true if all significant words in `subset` appear in `superset`.
fn is_word_subset(subset: &str, superset: &str) -> bool {
    let sub_words: Vec<&str> = subset.split_whitespace().filter(|w| w.len() > 1).collect();
    let super_words: Vec<&str> = superset
        .split_whitespace()
        .filter(|w| w.len() > 1)
        .collect();

    if sub_words.is_empty() {
        return true;
    }

    sub_words.iter().all(|sw| super_words.contains(sw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> ContradictionDetector {
        ContradictionDetector::new()
    }

    // =======================================================================
    // Negation pattern tests
    // =======================================================================

    #[test]
    fn test_detect_negation_contradiction_doesnt() {
        let d = detector();
        let result = d.check_contradiction("User likes Python", "User doesn't like Python");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_negation_contradiction_does_not() {
        let d = detector();
        let result = d.check_contradiction("User likes Python", "User does not like Python");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_negation_contradiction_reverse() {
        let d = detector();
        // Negation in existing, positive in new
        let result = d.check_contradiction("User doesn't like Python", "User likes Python");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_negation_contradiction_never() {
        let d = detector();
        let result = d.check_contradiction(
            "User drinks coffee every morning",
            "User never drinks coffee",
        );
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_negation_contradiction_isnt() {
        let d = detector();
        let result = d.check_contradiction("User is a vegetarian", "User isn't a vegetarian");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_negation_contradiction_no_longer() {
        let d = detector();
        let result = d.check_contradiction("User uses Vim", "User no longer uses Vim");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    // =======================================================================
    // Sentiment-flip tests
    // =======================================================================

    #[test]
    fn test_detect_negation_contradiction_likes_hates() {
        let d = detector();
        let result = d.check_contradiction("User likes JavaScript", "User hates JavaScript");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_negation_contradiction_loves_dislikes() {
        let d = detector();
        let result = d.check_contradiction("User loves Go", "User dislikes Go");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_negation_contradiction_enjoys_hates() {
        let d = detector();
        let result = d.check_contradiction("User enjoys running", "User hates running");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    // =======================================================================
    // Antonym pair tests
    // =======================================================================

    #[test]
    fn test_detect_antonym_contradiction_hot_cold() {
        let d = detector();
        let result = d.check_contradiction("The weather is hot today", "The weather is cold today");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_antonym_contradiction_light_dark() {
        let d = detector();
        let result = d.check_contradiction("User prefers light mode", "User prefers dark mode");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_antonym_contradiction_love_hate() {
        let d = detector();
        let result = d.check_contradiction("I love mornings", "I hate mornings");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_antonym_contradiction_always_never() {
        let d = detector();
        let result = d.check_contradiction("User always uses tabs", "User never uses tabs");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_antonym_contradiction_enabled_disabled() {
        let d = detector();
        let result = d.check_contradiction("Dark mode is enabled", "Dark mode is disabled");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_antonym_contradiction_happy_sad() {
        let d = detector();
        let result = d.check_contradiction(
            "User is happy with the result",
            "User is sad with the result",
        );
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    #[test]
    fn test_detect_antonym_contradiction_good_bad() {
        let d = detector();
        let result = d.check_contradiction("The food is good", "The food is bad");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    // =======================================================================
    // Value contradiction tests
    // =======================================================================

    #[test]
    fn test_detect_value_contradiction() {
        let d = detector();
        let result = d.check_contradiction(
            "User's favorite color is blue",
            "User's favorite color is red",
        );
        // Value contradictions return Unlikely (weaker signal)
        assert_eq!(result, ContradictionCheckResult::Unlikely);
    }

    #[test]
    fn test_detect_value_contradiction_age() {
        let d = detector();
        let result = d.check_contradiction("User's age is 25", "User's age is 30");
        assert_eq!(result, ContradictionCheckResult::Unlikely);
    }

    // =======================================================================
    // No false positive tests
    // =======================================================================

    #[test]
    fn test_no_false_positive_different_subjects() {
        let d = detector();
        let result = d.check_contradiction("User likes Python", "User likes Rust");
        assert_eq!(result, ContradictionCheckResult::None);
    }

    #[test]
    fn test_no_false_positive_same_content() {
        let d = detector();
        let result = d.check_contradiction("User prefers dark mode", "User prefers dark mode");
        assert_eq!(result, ContradictionCheckResult::None);
    }

    #[test]
    fn test_no_false_positive_unrelated() {
        let d = detector();
        let result = d.check_contradiction("User lives in San Francisco", "User works at Google");
        assert_eq!(result, ContradictionCheckResult::None);
    }

    #[test]
    fn test_no_false_positive_complementary() {
        let d = detector();
        let result = d.check_contradiction("User likes coffee", "User also likes tea");
        assert_eq!(result, ContradictionCheckResult::None);
    }

    #[test]
    fn test_no_false_positive_extension() {
        let d = detector();
        let result = d.check_contradiction(
            "User is a software engineer",
            "User is a senior software engineer",
        );
        assert_eq!(result, ContradictionCheckResult::None);
    }

    #[test]
    fn test_no_false_positive_different_domains() {
        let d = detector();
        let result =
            d.check_contradiction("The weather is hot today", "User prefers cold brew coffee");
        // Different context — should not trigger despite hot/cold antonyms
        assert_ne!(result, ContradictionCheckResult::Likely);
    }

    // =======================================================================
    // Case insensitivity tests
    // =======================================================================

    #[test]
    fn test_case_insensitive_detection() {
        let d = detector();
        let result = d.check_contradiction("User LIKES Python", "User DOESN'T like Python");
        assert_eq!(result, ContradictionCheckResult::Likely);
    }

    // =======================================================================
    // Edge cases
    // =======================================================================

    #[test]
    fn test_empty_strings() {
        let d = detector();
        let result = d.check_contradiction("", "");
        assert_eq!(result, ContradictionCheckResult::None);
    }

    #[test]
    fn test_one_empty_string() {
        let d = detector();
        let result = d.check_contradiction("User likes Python", "");
        assert_eq!(result, ContradictionCheckResult::None);
    }

    // =======================================================================
    // ContradictionCheckResult Display
    // =======================================================================

    #[test]
    fn test_result_display() {
        assert_eq!(ContradictionCheckResult::None.to_string(), "none");
        assert_eq!(ContradictionCheckResult::Unlikely.to_string(), "unlikely");
        assert_eq!(ContradictionCheckResult::Likely.to_string(), "likely");
    }

    // =======================================================================
    // Serialization tests
    // =======================================================================

    #[test]
    fn test_result_serializes() {
        let result = ContradictionCheckResult::Likely;
        let json = serde_json::to_string(&result).unwrap();
        assert_eq!(json, r#""likely""#);
    }

    #[test]
    fn test_result_deserializes() {
        let result: ContradictionCheckResult = serde_json::from_str(r#""unlikely""#).unwrap();
        assert_eq!(result, ContradictionCheckResult::Unlikely);
    }

    // =======================================================================
    // Helper function tests
    // =======================================================================

    #[test]
    fn test_content_overlap_score_identical() {
        let score = content_overlap_score("hello world foo", "hello world foo");
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_content_overlap_score_no_overlap() {
        let score = content_overlap_score("hello world", "foo bar baz");
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_content_overlap_score_partial() {
        let score = content_overlap_score("user likes python", "user likes rust");
        // "user" and "likes" overlap, "python"/"rust" don't
        assert!(score > 0.3);
        assert!(score < 0.8);
    }

    #[test]
    fn test_strip_negation() {
        let result = strip_negation("user doesn't like python", "doesn't");
        assert_eq!(result, "user like python");
    }

    #[test]
    fn test_contains_any_negation() {
        assert!(contains_any_negation("user doesn't like python"));
        assert!(contains_any_negation("user never eats meat"));
        assert!(contains_any_negation("user is not available"));
        assert!(!contains_any_negation("user likes python"));
    }
}
