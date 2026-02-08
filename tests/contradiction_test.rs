use momo::intelligence::contradiction::ContradictionDetector;

#[test]
fn test_contradiction_negation_pattern() {
    let detector = ContradictionDetector::new();

    let cases = vec![
        ("User likes pizza", "User doesn't like pizza", true),
        ("User is available", "User is not available", true),
        ("User prefers email", "User does not prefer email", true),
        (
            "User wants notifications",
            "User doesn't want notifications",
            true,
        ),
        ("System is enabled", "System is disabled", true),
        ("System is active", "System is not active", true),
    ];

    for (content1, content2, expected) in cases {
        let result = detector.check_contradiction(content1, content2);
        assert_eq!(
            result.is_contradiction(),
            expected,
            "Failed for: '{}' vs '{}'",
            content1,
            content2
        );
    }
}

#[test]
fn test_contradiction_antonym_patterns() {
    let detector = ContradictionDetector::new();

    let cases = vec![
        ("User loves coffee", "User hates coffee", true),
        ("Temperature is hot", "Temperature is cold", true),
        ("User is happy", "User is sad", true),
        ("User prefers light mode", "User prefers dark mode", true),
    ];

    for (content1, content2, expected) in cases {
        let result = detector.check_contradiction(content1, content2);
        assert_eq!(
            result.is_contradiction(),
            expected,
            "Failed for: '{}' vs '{}'",
            content1,
            content2
        );
    }
}

#[test]
fn test_no_contradiction_similar_content() {
    let detector = ContradictionDetector::new();

    let cases = vec![
        ("User likes pizza", "User likes pasta", false),
        ("User lives in Seattle", "User works in Seattle", false),
        ("User prefers dark mode", "User uses dark mode theme", false),
        ("System is running", "System is operating normally", false),
    ];

    for (content1, content2, expected) in cases {
        let result = detector.check_contradiction(content1, content2);
        assert_eq!(
            result.is_contradiction(),
            expected,
            "Failed for: '{}' vs '{}'",
            content1,
            content2
        );
    }
}

#[test]
fn test_no_contradiction_unrelated() {
    let detector = ContradictionDetector::new();

    let cases = vec![
        ("User likes pizza", "User is 25 years old", false),
        ("System is running", "Database is healthy", false),
        ("User prefers email", "User lives in Seattle", false),
    ];

    for (content1, content2, expected) in cases {
        let result = detector.check_contradiction(content1, content2);
        assert_eq!(
            result.is_contradiction(),
            expected,
            "Failed for: '{}' vs '{}'",
            content1,
            content2
        );
    }
}

#[test]
fn test_contradiction_detector_creation() {
    let detector = ContradictionDetector::new();
    let result = detector.check_contradiction("test content", "test content");
    assert!(!result.is_contradiction());
}
