// Common test utilities for integration tests
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize tracing subscriber once for tests
pub fn init_test_logger() {
    INIT.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
    });
}

// Re-export commonly used crates for convenience
pub use serial_test::serial;
pub use tempfile;
pub use wiremock;
