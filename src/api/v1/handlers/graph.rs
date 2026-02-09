//! v1 Graph handlers.
//!
//! Endpoints for exploring the knowledge graph around a specific memory
//! or across an entire container.

use axum::extract::{Path, Query, State};
use serde::Deserialize;
use serde_json::json;

use crate::api::v1::dto::GraphResponse;
use crate::api::v1::response::{ApiError, ApiResponse, ErrorCode};
use crate::api::AppState;
use crate::models::{
    GraphData, GraphEdgeType, GraphNode, GraphNodeType, GraphResponse as DomainGraphResponse,
    Metadata,
};

/// Query parameters for `GET /api/v1/memories/{memoryId}/graph`.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
#[serde(rename_all = "camelCase")]
pub struct MemoryGraphParams {
    /// Number of hops to traverse (default: 2).
    pub depth: Option<u32>,
    /// Maximum number of memory nodes to return (default: 50).
    pub max_nodes: Option<u32>,
    /// Comma-separated edge types to include (e.g. "updates,relatesto").
    pub relation_types: Option<String>,
}

/// Query parameters for `GET /api/v1/containers/{tag}/graph`.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
#[serde(rename_all = "camelCase")]
pub struct ContainerGraphParams {
    /// Maximum number of memory nodes to return (default: 100).
    pub max_nodes: Option<u32>,
}

/// Convert repository [`GraphData`] into the domain [`DomainGraphResponse`].
///
/// Constructs graph nodes from raw memory/document records with relevant
/// metadata fields for visualization.
fn graph_data_to_response(data: GraphData) -> DomainGraphResponse {
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

    DomainGraphResponse::with_data(nodes, data.edges)
}

/// Parse a comma-separated string of relation type names into typed variants.
///
/// Unknown names are silently ignored.
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

/// `GET /api/v1/memories/{memoryId}/graph`
///
/// Returns the knowledge graph neighborhood around a specific memory.
#[utoipa::path(
    get,
    path = "/api/v1/memories/{memoryId}/graph",
    tag = "graph",
    operation_id = "graph.getMemory",
    params(
        ("memoryId" = String, Path, description = "Memory ID"),
        MemoryGraphParams,
    ),
    responses(
        (status = 200, description = "Graph neighborhood", body = GraphResponse),
        (status = 404, description = "Memory not found", body = ApiError),
    )
)]
pub async fn get_memory_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(params): Query<MemoryGraphParams>,
) -> ApiResponse<GraphResponse> {
    // Verify memory exists
    let _memory = match state.db.get_memory_by_id(&id).await {
        Ok(Some(mem)) => mem,
        Ok(None) => {
            return ApiResponse::error(ErrorCode::NotFound, format!("Memory {id} not found"))
        }
        Err(e) => return e.into(),
    };

    let depth = params.depth.unwrap_or(2);
    let max_nodes = params.max_nodes.unwrap_or(50);

    let types = params.relation_types.as_deref().map(parse_relation_types);
    let types_slice = types.as_deref();

    let graph_data = match state
        .db
        .get_graph_neighborhood(&id, depth, max_nodes, types_slice)
        .await
    {
        Ok(data) => data,
        Err(e) => return e.into(),
    };

    let domain_response = graph_data_to_response(graph_data);
    ApiResponse::success(domain_response.into())
}

/// `GET /api/v1/containers/{tag}/graph`
///
/// Returns the knowledge graph for all memories within a container.
#[utoipa::path(
    get,
    path = "/api/v1/containers/{tag}/graph",
    tag = "graph",
    operation_id = "graph.getContainer",
    params(
        ("tag" = String, Path, description = "Container tag"),
        ContainerGraphParams,
    ),
    responses(
        (status = 200, description = "Container graph", body = GraphResponse),
    )
)]
pub async fn get_container_graph(
    State(state): State<AppState>,
    Path(tag): Path<String>,
    Query(params): Query<ContainerGraphParams>,
) -> ApiResponse<GraphResponse> {
    let max_nodes = params.max_nodes.unwrap_or(100);

    let graph_data = match state.db.get_container_graph(&tag, max_nodes).await {
        Ok(data) => data,
        Err(e) => return e.into(),
    };

    let domain_response = graph_data_to_response(graph_data);
    ApiResponse::success(domain_response.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_relation_types_all_variants() {
        let result = parse_relation_types("updates,relatesto,conflictswith,derivedfrom,sources");
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], GraphEdgeType::Updates);
        assert_eq!(result[1], GraphEdgeType::RelatesTo);
        assert_eq!(result[2], GraphEdgeType::ConflictsWith);
        assert_eq!(result[3], GraphEdgeType::DerivedFrom);
        assert_eq!(result[4], GraphEdgeType::Sources);
    }

    #[test]
    fn parse_relation_types_case_insensitive() {
        let result = parse_relation_types("Updates,RELATESTO,DerivedFrom");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], GraphEdgeType::Updates);
        assert_eq!(result[1], GraphEdgeType::RelatesTo);
        assert_eq!(result[2], GraphEdgeType::DerivedFrom);
    }

    #[test]
    fn parse_relation_types_skips_invalid() {
        let result = parse_relation_types("updates,invalid,sources");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], GraphEdgeType::Updates);
        assert_eq!(result[1], GraphEdgeType::Sources);
    }

    #[test]
    fn parse_relation_types_trims_whitespace() {
        let result = parse_relation_types(" updates , relatesto ");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], GraphEdgeType::Updates);
        assert_eq!(result[1], GraphEdgeType::RelatesTo);
    }

    #[test]
    fn parse_relation_types_empty() {
        let result = parse_relation_types("");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn memory_graph_params_deserializes_defaults() {
        let json = r#"{}"#;
        let params: MemoryGraphParams = serde_json::from_str(json).expect("deserialize");
        assert!(params.depth.is_none());
        assert!(params.max_nodes.is_none());
        assert!(params.relation_types.is_none());
    }

    #[test]
    fn memory_graph_params_deserializes_all_fields() {
        let json = r#"{"depth": 3, "maxNodes": 25, "relationTypes": "updates,sources"}"#;
        let params: MemoryGraphParams = serde_json::from_str(json).expect("deserialize");
        assert_eq!(params.depth, Some(3));
        assert_eq!(params.max_nodes, Some(25));
        assert_eq!(params.relation_types.as_deref(), Some("updates,sources"));
    }

    #[test]
    fn container_graph_params_deserializes_defaults() {
        let json = r#"{}"#;
        let params: ContainerGraphParams = serde_json::from_str(json).expect("deserialize");
        assert!(params.max_nodes.is_none());
    }

    #[test]
    fn container_graph_params_deserializes_max_nodes() {
        let json = r#"{"maxNodes": 200}"#;
        let params: ContainerGraphParams = serde_json::from_str(json).expect("deserialize");
        assert_eq!(params.max_nodes, Some(200));
    }

    #[test]
    fn graph_data_to_response_converts_memories() {
        let mut memory = crate::models::Memory::new(
            "mem_1".to_string(),
            "test memory".to_string(),
            "default".to_string(),
        );
        memory.container_tag = Some("test_tag".to_string());

        let data = GraphData {
            memories: vec![memory],
            edges: vec![],
            documents: vec![],
        };

        let response = graph_data_to_response(data);
        assert_eq!(response.nodes.len(), 1);
        assert_eq!(response.nodes[0].id, "mem_1");
        assert_eq!(response.links.len(), 0);

        // Convert to v1 DTO
        let v1_response: GraphResponse = response.into();
        assert_eq!(v1_response.nodes.len(), 1);
        assert_eq!(v1_response.nodes[0].id, "mem_1");
    }

    #[test]
    fn graph_data_to_response_converts_documents() {
        let mut doc = crate::models::Document::new("doc_1".to_string());
        doc.title = Some("Test Doc".to_string());
        doc.url = Some("https://example.com".to_string());

        let data = GraphData {
            memories: vec![],
            edges: vec![],
            documents: vec![doc],
        };

        let response = graph_data_to_response(data);
        assert_eq!(response.nodes.len(), 1);
        assert_eq!(response.nodes[0].id, "doc_1");

        // Verify URL is in metadata
        let url_val = response.nodes[0]
            .metadata
            .get("url")
            .expect("url in metadata");
        assert_eq!(url_val, "https://example.com");
    }
}
