//! v1 API Data Transfer Objects.
//!
//! These types define the wire format for the v1 REST API. They are completely
//! separate from the internal domain models in `src/models/` and handle
//! serialization, deserialization, and domain-model conversion.

pub mod admin;
pub mod common;
pub mod conversation;
pub mod documents;
pub mod graph;
pub mod memories;
pub mod profile;
pub mod search;

// Re-export all public types for convenient access via `dto::*`.
pub use admin::*;
pub use documents::*;
pub use graph::*;
pub use memories::*;
pub use search::*;
