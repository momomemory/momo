use std::collections::HashSet;

/// Compute a simple word-overlap score between two strings (Jaccard-like).
/// Returns a value between 0.0 (no overlap) and 1.0 (identical word sets).
pub fn content_overlap_score(a: &str, b: &str) -> f64 {
    let words_a: HashSet<&str> = a.split_whitespace().filter(|w| w.len() > 1).collect();
    let words_b: HashSet<&str> = b.split_whitespace().filter(|w| w.len() > 1).collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    intersection as f64 / union as f64
}

/// Fuzzy word-overlap score that handles verb-form differences.
/// Like `content_overlap_score` but considers words matching if one is
/// a prefix of the other (min 3 chars). This handles "like"/"likes" etc.
pub fn fuzzy_overlap_score(a: &str, b: &str) -> f64 {
    let words_a: Vec<&str> = a.split_whitespace().filter(|w| w.len() > 1).collect();
    let words_b: Vec<&str> = b.split_whitespace().filter(|w| w.len() > 1).collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let mut matched_a = 0usize;
    for wa in &words_a {
        if words_b.iter().any(|wb| fuzzy_word_match(wa, wb)) {
            matched_a += 1;
        }
    }

    let mut matched_b = 0usize;
    for wb in &words_b {
        if words_a.iter().any(|wa| fuzzy_word_match(wa, wb)) {
            matched_b += 1;
        }
    }

    let total_unique = words_a.len() + words_b.len() - matched_a.min(matched_b);
    let total_matched = matched_a.max(matched_b);

    total_matched as f64 / total_unique as f64
}

fn fuzzy_word_match(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let min_len = a.len().min(b.len());
    if min_len < 3 {
        return false;
    }
    a.starts_with(b) || b.starts_with(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_overlap_identical() {
        let s1 = "hello world foo";
        let s2 = "hello world foo";
        assert!((content_overlap_score(s1, s2) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_content_overlap_none() {
        let s1 = "hello world";
        let s2 = "foo bar baz";
        assert!((content_overlap_score(s1, s2) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fuzzy_overlap_prefix() {
        let s1 = "user likes python";
        let s2 = "user like python";
        let score = fuzzy_overlap_score(s1, s2);
        assert!(score > 0.6);
    }
}
