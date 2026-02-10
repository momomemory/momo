import cytoscape, { type Core } from 'cytoscape';
import { useEffect, useMemo, useRef } from 'preact/hooks';

import type { GraphNodeResponse, GraphResponse } from '../types';

interface GraphCanvasProps {
  graph: GraphResponse | null;
  onNodeSelect: (node: GraphNodeResponse | null) => void;
}

function nodeLabel(node: GraphNodeResponse): string {
  const rawContent = typeof node.metadata.content === 'string' ? node.metadata.content : '';
  const rawTitle = typeof node.metadata.title === 'string' ? node.metadata.title : '';
  const raw = rawTitle || rawContent;
  const trimmed = raw.trim();

  if (!trimmed) {
    return `${node.type}:${node.id.slice(0, 8)}`;
  }

  const snippet = trimmed.length > 18 ? `${trimmed.slice(0, 18)}...` : trimmed;
  return `${node.type}:${snippet}`;
}

function graphLayout(graph: GraphResponse): cytoscape.LayoutOptions {
  const nodeCount = graph.nodes.length;
  const edgeCount = graph.links.length;
  const density = edgeCount / Math.max(nodeCount, 1);

  if (edgeCount === 0) {
    return {
      name: 'circle',
      animate: false,
      fit: true,
      padding: 32,
    };
  }

  if (density < 0.75) {
    return {
      name: 'concentric',
      animate: false,
      fit: true,
      padding: 32,
      spacingFactor: 1.1,
    };
  }

  return {
    name: 'cose',
    animate: false,
    fit: true,
    padding: 32,
    randomize: true,
    idealEdgeLength: 140,
    nodeRepulsion: 600000,
    componentSpacing: 90,
  };
}

export function GraphCanvas({ graph, onNodeSelect }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  const nodeIndex = useMemo(() => {
    const map = new Map<string, GraphNodeResponse>();
    if (!graph) {
      return map;
    }
    for (const node of graph.nodes) {
      map.set(node.id, node);
    }
    return map;
  }, [graph]);

  useEffect(() => {
    onNodeSelect(null);

    if (!containerRef.current || !graph) {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
      return;
    }

    if (cyRef.current) {
      cyRef.current.destroy();
      cyRef.current = null;
    }

    const degreeById = new Map<string, number>();
    for (const node of graph.nodes) {
      degreeById.set(node.id, 0);
    }
    for (const edge of graph.links) {
      degreeById.set(edge.source, (degreeById.get(edge.source) ?? 0) + 1);
      degreeById.set(edge.target, (degreeById.get(edge.target) ?? 0) + 1);
    }

    const maxDegree = Math.max(1, ...degreeById.values());

    const cy = cytoscape({
      container: containerRef.current,
      elements: [
        ...graph.nodes.map((node) => ({
          data: {
            id: node.id,
            label: nodeLabel(node),
            nodeType: node.type,
            degree: degreeById.get(node.id) ?? 0,
            maxDegree,
          },
        })),
        ...graph.links.map((edge, index) => ({
          data: {
            id: `${edge.source}-${edge.target}-${edge.type}-${index}`,
            source: edge.source,
            target: edge.target,
            label: edge.type,
            edgeType: edge.type,
          },
        })),
      ],
      layout: graphLayout(graph),
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#2f7f71',
            label: 'data(label)',
            color: '#e7f4ef',
            'font-size': '9px',
            'text-wrap': 'wrap',
            'text-max-width': '110px',
            'text-valign': 'bottom',
            'text-halign': 'center',
            'text-margin-y': 10,
            width: 'mapData(degree, 0, 12, 42, 70)',
            height: 'mapData(degree, 0, 12, 42, 70)',
            'border-width': '1px',
            'border-color': '#a7d4c7',
          },
        },
        {
          selector: 'node[nodeType = "document"]',
          style: {
            'background-color': '#5164b0',
            'border-color': '#cad4ff',
          },
        },
        {
          selector: 'edge',
          style: {
            width: '2px',
            'line-color': '#6f8e8b',
            'target-arrow-color': '#6f8e8b',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            opacity: 0.75,
          },
        },
      ],
    });

    cy.fit(undefined, 32);

    cy.on('tap', 'node', (event) => {
      const id = event.target.id();
      onNodeSelect(nodeIndex.get(id) ?? null);
    });

    cy.on('tap', (event) => {
      if (event.target === cy) {
        onNodeSelect(null);
      }
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graph, nodeIndex, onNodeSelect]);

  if (!graph) {
    return <div class="graph-empty">Run a graph query to visualize relationships.</div>;
  }

  if (graph.nodes.length === 0) {
    return <div class="graph-empty">Graph returned zero nodes.</div>;
  }

  return <div class="graph-canvas" ref={containerRef} />;
}
