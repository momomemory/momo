use serde::{Deserialize, Serialize};

use super::Metadata;

/// Graph node types for D3.js visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GraphNodeType {
    Memory,
    Document,
}

/// Graph edge types for D3.js visualization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GraphEdgeType {
    Updates,
    RelatesTo,
    ConflictsWith,
    DerivedFrom,
    Sources,
}

/// A node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier (memory_id or document_id)
    pub id: String,

    /// Node type (memory or document)
    #[serde(rename = "type")]
    pub node_type: GraphNodeType,

    /// Additional metadata for visualization (title, color, size, etc.)
    pub metadata: Metadata,
}

/// An edge in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub source: String,

    /// Target node ID
    pub target: String,

    /// Edge type (relationship kind)
    #[serde(rename = "type")]
    pub edge_type: GraphEdgeType,
}

/// D3.js compatible graph response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphResponse {
    /// List of nodes in the graph
    pub nodes: Vec<GraphNode>,

    /// List of links (edges) connecting nodes - named "links" for D3.js compatibility
    pub links: Vec<GraphEdge>,
}

impl GraphResponse {
    /// Create a new empty graph response
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            links: Vec::new(),
        }
    }

    /// Create a graph response with nodes and links
    pub fn with_data(nodes: Vec<GraphNode>, links: Vec<GraphEdge>) -> Self {
        Self { nodes, links }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.push(node);
    }

    /// Add a link (edge) to the graph
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.links.push(edge);
    }
}

impl Default for GraphResponse {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(id: String, node_type: GraphNodeType) -> Self {
        Self {
            id,
            node_type,
            metadata: Metadata::new(),
        }
    }

    /// Create a graph node with metadata
    pub fn with_metadata(id: String, node_type: GraphNodeType, metadata: Metadata) -> Self {
        Self {
            id,
            node_type,
            metadata,
        }
    }
}

impl GraphEdge {
    /// Create a new graph edge
    pub fn new(source: String, target: String, edge_type: GraphEdgeType) -> Self {
        Self {
            source,
            target,
            edge_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_graph_response_new() {
        let graph = GraphResponse::new();
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.links.len(), 0);
    }

    #[test]
    fn test_graph_response_add_node() {
        let mut graph = GraphResponse::new();
        let node = GraphNode::new("node1".to_string(), GraphNodeType::Memory);
        graph.add_node(node);
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_graph_response_add_edge() {
        let mut graph = GraphResponse::new();
        let edge = GraphEdge::new(
            "node1".to_string(),
            "node2".to_string(),
            GraphEdgeType::RelatesTo,
        );
        graph.add_edge(edge);
        assert_eq!(graph.links.len(), 1);
    }

    #[test]
    fn test_node_serialization() {
        let mut metadata = Metadata::new();
        metadata.insert("title".to_string(), json!("Test Memory"));

        let node = GraphNode::with_metadata("mem_123".to_string(), GraphNodeType::Memory, metadata);

        let json = serde_json::to_string(&node).unwrap();
        assert!(json.contains("\"type\":\"memory\""));
        assert!(json.contains("\"id\":\"mem_123\""));
    }

    #[test]
    fn test_edge_serialization() {
        let edge = GraphEdge::new(
            "mem_1".to_string(),
            "mem_2".to_string(),
            GraphEdgeType::Updates,
        );

        let json = serde_json::to_string(&edge).unwrap();
        assert!(json.contains("\"source\":\"mem_1\""));
        assert!(json.contains("\"target\":\"mem_2\""));
        assert!(json.contains("\"type\":\"updates\""));
    }

    #[test]
    fn test_graph_response_with_data() {
        let nodes = vec![
            GraphNode::new("n1".to_string(), GraphNodeType::Memory),
            GraphNode::new("n2".to_string(), GraphNodeType::Document),
        ];
        let edges = vec![GraphEdge::new(
            "n1".to_string(),
            "n2".to_string(),
            GraphEdgeType::DerivedFrom,
        )];

        let graph = GraphResponse::with_data(nodes, edges);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.links.len(), 1);
    }

    #[test]
    fn test_graph_edge_type_partial_eq() {
        assert_eq!(GraphEdgeType::Updates, GraphEdgeType::Updates);
        assert_eq!(GraphEdgeType::RelatesTo, GraphEdgeType::RelatesTo);
        assert_ne!(GraphEdgeType::Updates, GraphEdgeType::RelatesTo);
        assert_ne!(GraphEdgeType::Sources, GraphEdgeType::DerivedFrom);
    }

    #[test]
    fn test_graph_response_serializes_links_not_edges() {
        let graph = GraphResponse::with_data(
            vec![GraphNode::new("n1".to_string(), GraphNodeType::Memory)],
            vec![GraphEdge::new(
                "n1".to_string(),
                "n2".to_string(),
                GraphEdgeType::Updates,
            )],
        );
        let json = serde_json::to_string(&graph).expect("serialize");
        assert!(
            json.contains("\"links\""),
            "JSON should contain 'links' key"
        );
        assert!(
            !json.contains("\"edges\""),
            "JSON should NOT contain 'edges' key"
        );
    }
}
