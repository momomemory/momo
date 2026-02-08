//! Graph DTOs for the v1 API.

use serde::{Deserialize, Serialize};

use super::common::Metadata;
use crate::models;

/// Node type in the knowledge graph.
///
/// Wire format: `"memory"` or `"document"`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub enum GraphNodeType {
    Memory,
    Document,
}

impl From<models::GraphNodeType> for GraphNodeType {
    fn from(nt: models::GraphNodeType) -> Self {
        match nt {
            models::GraphNodeType::Memory => GraphNodeType::Memory,
            models::GraphNodeType::Document => GraphNodeType::Document,
        }
    }
}

/// Edge type in the knowledge graph.
///
/// Wire format: lowercase string.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub enum GraphEdgeType {
    Updates,
    RelatesTo,
    ConflictsWith,
    DerivedFrom,
    Sources,
}

impl From<models::GraphEdgeType> for GraphEdgeType {
    fn from(et: models::GraphEdgeType) -> Self {
        match et {
            models::GraphEdgeType::Updates => GraphEdgeType::Updates,
            models::GraphEdgeType::RelatesTo => GraphEdgeType::RelatesTo,
            models::GraphEdgeType::ConflictsWith => GraphEdgeType::ConflictsWith,
            models::GraphEdgeType::DerivedFrom => GraphEdgeType::DerivedFrom,
            models::GraphEdgeType::Sources => GraphEdgeType::Sources,
        }
    }
}

/// A node in the knowledge graph response.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct GraphNodeResponse {
    /// Node identifier (memory ID or document ID).
    pub id: String,
    /// Node classification.
    #[serde(rename = "type")]
    pub node_type: GraphNodeType,
    /// Additional visualization metadata.
    #[schema(value_type = Object)]
    pub metadata: Metadata,
}

impl From<models::GraphNode> for GraphNodeResponse {
    fn from(node: models::GraphNode) -> Self {
        Self {
            id: node.id,
            node_type: node.node_type.into(),
            metadata: node.metadata,
        }
    }
}

/// An edge (link) in the knowledge graph response.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct GraphEdgeResponse {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Relationship type.
    #[serde(rename = "type")]
    pub edge_type: GraphEdgeType,
}

impl From<models::GraphEdge> for GraphEdgeResponse {
    fn from(edge: models::GraphEdge) -> Self {
        Self {
            source: edge.source,
            target: edge.target,
            edge_type: edge.edge_type.into(),
        }
    }
}

/// Graph response for `GET /v1/memories/{memoryId}/graph` and `GET /v1/containers/{tag}/graph`.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct GraphResponse {
    /// Graph nodes.
    pub nodes: Vec<GraphNodeResponse>,
    /// Graph edges (links between nodes).
    pub links: Vec<GraphEdgeResponse>,
}

impl From<models::GraphResponse> for GraphResponse {
    fn from(graph: models::GraphResponse) -> Self {
        Self {
            nodes: graph.nodes.into_iter().map(Into::into).collect(),
            links: graph.links.into_iter().map(Into::into).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_response_serializes_links_key() {
        let resp = GraphResponse {
            nodes: vec![GraphNodeResponse {
                id: "n1".to_string(),
                node_type: GraphNodeType::Memory,
                metadata: std::collections::HashMap::new(),
            }],
            links: vec![GraphEdgeResponse {
                source: "n1".to_string(),
                target: "n2".to_string(),
                edge_type: GraphEdgeType::Updates,
            }],
        };

        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("\"links\""));
        assert!(!json.contains("\"edges\""));
        assert!(json.contains("\"type\""));
    }
}
