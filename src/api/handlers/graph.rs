use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Deserialize;
use serde_json::json;

use crate::api::AppState;
use crate::error::{MomoError, Result};
use crate::models::{GraphData, GraphEdgeType, GraphNode, GraphNodeType, GraphResponse, Metadata};

#[derive(Debug, Deserialize)]
pub struct GraphNeighborhoodParams {
    /// Number of hops to traverse (default: 2)
    pub depth: Option<u32>,
    /// Maximum number of memory nodes to return (default: 50)
    pub max_nodes: Option<u32>,
    /// Comma-separated edge types to include (e.g. "updates,relatesto")
    pub relation_types: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ContainerGraphParams {
    /// Maximum number of memory nodes to return (default: 100)
    pub max_nodes: Option<u32>,
}

/// Convert repository `GraphData` into the D3.js-compatible `GraphResponse`.
fn graph_data_to_response(data: GraphData) -> GraphResponse {
    let mut nodes: Vec<GraphNode> = Vec::with_capacity(data.memories.len() + data.documents.len());

    for memory in &data.memories {
        let mut metadata = Metadata::new();
        metadata.insert("content".to_string(), json!(memory.memory));
        metadata.insert("version".to_string(), json!(memory.version));
        metadata.insert("memory_type".to_string(), json!(memory.memory_type));
        metadata.insert("is_latest".to_string(), json!(memory.is_latest));
        metadata.insert("created_at".to_string(), json!(memory.created_at));
        if let Some(ref tag) = memory.container_tag {
            metadata.insert("container_tag".to_string(), json!(tag));
        }

        nodes.push(GraphNode::with_metadata(
            memory.id.clone(),
            GraphNodeType::Memory,
            metadata,
        ));
    }

    for doc in &data.documents {
        let mut metadata = Metadata::new();
        metadata.insert("title".to_string(), json!(doc.title));
        metadata.insert("doc_type".to_string(), json!(doc.doc_type));
        metadata.insert("status".to_string(), json!(doc.status));
        metadata.insert("created_at".to_string(), json!(doc.created_at));
        if let Some(ref url) = doc.url {
            metadata.insert("url".to_string(), json!(url));
        }

        nodes.push(GraphNode::with_metadata(
            doc.id.clone(),
            GraphNodeType::Document,
            metadata,
        ));
    }

    GraphResponse::with_data(nodes, data.edges)
}

fn parse_relation_types(input: &str) -> Vec<GraphEdgeType> {
    input
        .split(',')
        .filter_map(|s| match s.trim().to_lowercase().as_str() {
            "updates" => Some(GraphEdgeType::Updates),
            "relatesto" => Some(GraphEdgeType::RelatesTo),
            "conflictswith" => Some(GraphEdgeType::ConflictsWith),
            "derivedfrom" => Some(GraphEdgeType::DerivedFrom),
            "sources" => Some(GraphEdgeType::Sources),
            _ => None,
        })
        .collect()
}

/// GET /v4/memories/{id}/graph?depth=2&max_nodes=50&relation_types=updates,relatesto
pub async fn get_memory_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(params): Query<GraphNeighborhoodParams>,
) -> Result<Json<GraphResponse>> {
    let _memory = state.db.get_memory_by_id(&id)
        .await?
        .ok_or_else(|| MomoError::NotFound(format!("Memory '{}' not found", id)))?;

    let depth = params.depth.unwrap_or(2);
    let max_nodes = params.max_nodes.unwrap_or(50);

    let types = params.relation_types.as_deref().map(parse_relation_types);
    let types_slice = types.as_deref();

    let graph_data =
        state.db.get_graph_neighborhood(&id, depth, max_nodes, types_slice).await?;

    Ok(Json(graph_data_to_response(graph_data)))
}

/// GET /v4/containers/{tag}/graph?max_nodes=100
pub async fn get_container_graph(
    State(state): State<AppState>,
    Path(tag): Path<String>,
    Query(params): Query<ContainerGraphParams>,
) -> Result<Json<GraphResponse>> {
    let max_nodes = params.max_nodes.unwrap_or(100);

    let graph_data = state.db.get_container_graph(&tag, max_nodes).await?;

    Ok(Json(graph_data_to_response(graph_data)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_relation_types_all_variants() {
        let result = parse_relation_types("updates,relatesto,conflictswith,derivedfrom,sources");
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], GraphEdgeType::Updates);
        assert_eq!(result[1], GraphEdgeType::RelatesTo);
        assert_eq!(result[2], GraphEdgeType::ConflictsWith);
        assert_eq!(result[3], GraphEdgeType::DerivedFrom);
        assert_eq!(result[4], GraphEdgeType::Sources);
    }

    #[test]
    fn test_parse_relation_types_case_insensitive() {
        let result = parse_relation_types("Updates,RELATESTO,DerivedFrom");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], GraphEdgeType::Updates);
        assert_eq!(result[1], GraphEdgeType::RelatesTo);
        assert_eq!(result[2], GraphEdgeType::DerivedFrom);
    }

    #[test]
    fn test_parse_relation_types_skips_invalid() {
        let result = parse_relation_types("updates,invalid,sources");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], GraphEdgeType::Updates);
        assert_eq!(result[1], GraphEdgeType::Sources);
    }

    #[test]
    fn test_parse_relation_types_trims_whitespace() {
        let result = parse_relation_types(" updates , relatesto ");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], GraphEdgeType::Updates);
        assert_eq!(result[1], GraphEdgeType::RelatesTo);
    }

    #[test]
    fn test_parse_relation_types_empty() {
        let result = parse_relation_types("");
        assert_eq!(result.len(), 0);
    }
}
