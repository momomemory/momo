#[allow(dead_code)]
mod api;
mod provider;
mod reranker;

#[cfg(test)]
mod tests;

pub use provider::EmbeddingProvider;
#[allow(unused_imports)] // RerankResult used in tests (services::search)
pub use reranker::{RerankResult, RerankerProvider};
