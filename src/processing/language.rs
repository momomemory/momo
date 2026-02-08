/// Supported programming languages for AST-aware code chunking.
/// Maps to tree-sitter language grammars used by text-splitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    TypeScriptTsx,
    Go,
    Java,
    C,
    Cpp,
}

/// Detect programming language from file path extension.
/// Returns None for unsupported or unknown extensions.
pub fn detect_language(source_path: &str) -> Option<SupportedLanguage> {
    let path = source_path.to_lowercase();
    let extension = path.rsplit('.').next()?;

    match extension {
        "rs" => Some(SupportedLanguage::Rust),
        "py" => Some(SupportedLanguage::Python),
        "js" | "jsx" => Some(SupportedLanguage::JavaScript),
        "ts" => Some(SupportedLanguage::TypeScript),
        "tsx" => Some(SupportedLanguage::TypeScriptTsx),
        "go" => Some(SupportedLanguage::Go),
        "java" => Some(SupportedLanguage::Java),
        "c" | "h" => Some(SupportedLanguage::C),
        "cpp" | "hpp" | "cc" | "cxx" => Some(SupportedLanguage::Cpp),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_rust() {
        assert_eq!(
            detect_language("src/main.rs"),
            Some(SupportedLanguage::Rust)
        );
        assert_eq!(detect_language("lib.rs"), Some(SupportedLanguage::Rust));
    }

    #[test]
    fn test_detect_python() {
        assert_eq!(
            detect_language("script.py"),
            Some(SupportedLanguage::Python)
        );
        assert_eq!(
            detect_language("path/to/module.py"),
            Some(SupportedLanguage::Python)
        );
    }

    #[test]
    fn test_detect_javascript() {
        assert_eq!(
            detect_language("app.js"),
            Some(SupportedLanguage::JavaScript)
        );
        assert_eq!(
            detect_language("Component.jsx"),
            Some(SupportedLanguage::JavaScript)
        );
    }

    #[test]
    fn test_detect_typescript() {
        assert_eq!(
            detect_language("index.ts"),
            Some(SupportedLanguage::TypeScript)
        );
        assert_eq!(
            detect_language("Component.tsx"),
            Some(SupportedLanguage::TypeScriptTsx)
        );
    }

    #[test]
    fn test_detect_go() {
        assert_eq!(detect_language("main.go"), Some(SupportedLanguage::Go));
    }

    #[test]
    fn test_detect_java() {
        assert_eq!(detect_language("Main.java"), Some(SupportedLanguage::Java));
    }

    #[test]
    fn test_detect_c_cpp() {
        assert_eq!(detect_language("main.c"), Some(SupportedLanguage::C));
        assert_eq!(detect_language("header.h"), Some(SupportedLanguage::C));
        assert_eq!(detect_language("main.cpp"), Some(SupportedLanguage::Cpp));
        assert_eq!(detect_language("header.hpp"), Some(SupportedLanguage::Cpp));
    }

    #[test]
    fn test_detect_unsupported() {
        assert_eq!(detect_language("README.md"), None);
        assert_eq!(detect_language("data.json"), None);
        assert_eq!(detect_language("style.css"), None);
        assert_eq!(detect_language("noextension"), None);
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(detect_language("MAIN.RS"), Some(SupportedLanguage::Rust));
        assert_eq!(
            detect_language("Script.PY"),
            Some(SupportedLanguage::Python)
        );
    }
}
