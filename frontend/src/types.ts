export type JsonObject = Record<string, unknown>;

export interface ApiError {
  code: string;
  message: string;
}

export interface ApiMeta {
  nextCursor?: string;
  total?: number;
}

export interface ApiEnvelope<T> {
  data?: T;
  meta?: ApiMeta;
  error?: ApiError;
}

export interface ApiCallResult<T> {
  ok: boolean;
  status: number;
  envelope?: ApiEnvelope<T>;
  raw?: unknown;
  error?: string;
}

export interface HealthData {
  status: string;
  version: string;
  database: { status: string };
  embeddings: { status: string; model: string; dimensions: number };
  llm: { status: string; provider?: string; model?: string };
  reranker: { enabled: boolean; model?: string; status: string };
}

export interface GraphNodeResponse {
  id: string;
  type: 'memory' | 'document';
  metadata: JsonObject;
}

export interface GraphEdgeResponse {
  source: string;
  target: string;
  type: 'updates' | 'relatesTo' | 'conflictsWith' | 'derivedFrom' | 'sources';
}

export interface GraphResponse {
  nodes: GraphNodeResponse[];
  links: GraphEdgeResponse[];
}

export interface ContainerTagsResponse {
  tags: string[];
}
