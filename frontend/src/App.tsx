import { useCallback, useEffect, useMemo, useState } from 'preact/hooks';

import { apiEnvelope, apiRaw } from './api';
import { GraphCanvas } from './components/GraphCanvas';
import { JsonView } from './components/JsonView';
import type {
  ApiCallResult,
  ContainerTagsResponse,
  GraphNodeResponse,
  GraphResponse,
  HealthData,
  JsonObject,
} from './types';

const API_KEY_STORAGE_KEY = 'momo.ui.apiKey';
const CONTAINER_STORAGE_KEY = 'momo.ui.containerTag';
const SCOPE_EXPANDED_STORAGE_KEY = 'momo.ui.scopeExpanded';

type TabId =
  | 'settings'
  | 'system'
  | 'search'
  | 'documents'
  | 'memories'
  | 'graph'
  | 'profile'
  | 'conversation'
  | 'admin';

const TAB_ITEMS: Array<{ id: TabId; label: string }> = [
  { id: 'system', label: 'System' },
  { id: 'search', label: 'Search' },
  { id: 'documents', label: 'Documents' },
  { id: 'memories', label: 'Memories' },
  { id: 'graph', label: 'Graph' },
  { id: 'profile', label: 'Profile' },
  { id: 'conversation', label: 'Conversation' },
  { id: 'admin', label: 'Admin' },
  { id: 'settings', label: 'Settings' },
];

interface SharedTabProps {
  apiKey: string;
  containerTag: string;
  setContainerTag: (value: string) => void;
  onAuthFailure: (message: string) => void;
}

interface ApiActionState {
  loading: boolean;
  error: string | null;
  result: unknown;
}

type AuthState = 'missing' | 'checking' | 'valid' | 'invalid';

function isTabId(value: string): value is TabId {
  return TAB_ITEMS.some((tab) => tab.id === value);
}

function tabFromHash(hash: string): TabId | null {
  const candidate = hash.replace(/^#\/?/, '').trim();
  if (!candidate) {
    return null;
  }
  return isTabId(candidate) ? candidate : null;
}

function parseJsonObject(input: string): JsonObject | undefined {
  if (!input.trim()) {
    return undefined;
  }

  const parsed = JSON.parse(input) as unknown;
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('JSON must be an object');
  }

  return parsed as JsonObject;
}

function parseOptionalNumber(input: string): number | undefined {
  const trimmed = input.trim();
  if (!trimmed) {
    return undefined;
  }

  const value = Number(trimmed);
  if (!Number.isFinite(value)) {
    throw new Error(`Invalid numeric value: ${input}`);
  }

  return value;
}

function parseCommaList(input: string): string[] {
  return input
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

function parseHealthData(result: unknown): HealthData | null {
  if (!result || typeof result !== 'object') {
    return null;
  }

  const maybeEnvelope = result as { data?: unknown };
  const payload = maybeEnvelope.data ?? result;
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return null;
  }

  if (!('status' in payload)) {
    return null;
  }

  return payload as HealthData;
}

function healthStatusClass(status: string | undefined): string {
  const normalized = (status ?? '').toLowerCase();
  if (normalized === 'ok' || normalized === 'healthy' || normalized === 'up') {
    return 'health-ok';
  }
  if (!normalized || normalized === 'unknown') {
    return 'health-unknown';
  }
  return 'health-error';
}

function useApiAction(onAuthFailure?: (message: string) => void) {
  const [state, setState] = useState<ApiActionState>({
    loading: false,
    error: null,
    result: null,
  });

  const run = useCallback(async function runAction<T>(
    action: () => Promise<ApiCallResult<T>>,
  ): Promise<ApiCallResult<T>> {
    setState((current) => ({ ...current, loading: true, error: null }));

    const response = await action();

    if (!response.ok) {
      const message = response.error ?? `Request failed (${response.status})`;
      const isUnauthorized =
        response.status === 401 || response.envelope?.error?.code === 'unauthorized';

      if (isUnauthorized) {
        onAuthFailure?.(message);
      }

      setState({
        loading: false,
        error: message,
        result: response.envelope ?? response.raw ?? response,
      });
      return response;
    }

    setState({
      loading: false,
      error: null,
      result: response.envelope ?? response.raw ?? response,
    });

    return response;
  }, [onAuthFailure]);

  const clear = useCallback(() => {
    setState({ loading: false, error: null, result: null });
  }, []);

  return {
    ...state,
    run,
    clear,
  };
}

function Section({ title, children }: { title: string; children: preact.ComponentChildren }) {
  return (
    <section class="panel">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: preact.ComponentChildren;
}) {
  return (
    <label class="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function SystemTab({ apiKey }: { apiKey: string }) {
  const health = useApiAction();
  const openapi = useApiAction();
  const healthData = useMemo(() => parseHealthData(health.result), [health.result]);

  useEffect(() => {
    void health.run(() => apiEnvelope<HealthData>(apiKey, '/health', { auth: false }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div class="tab-grid">
      <Section title="Health">
        <div class="button-row">
          <button
            type="button"
            onClick={() => {
              void health.run(() => apiEnvelope<HealthData>(apiKey, '/health', { auth: false }));
            }}
          >
            Refresh health
          </button>
          <button type="button" class="ghost" onClick={health.clear}>
            Clear
          </button>
        </div>
        {health.loading && <p>Loading health...</p>}
        {health.error && <p class="error">{health.error}</p>}

        {healthData && (
          <div class="health-grid">
            <article class={`health-card ${healthStatusClass(healthData.status)}`}>
              <h4>Service</h4>
              <p class="health-value">{healthData.status}</p>
              <p class="muted">Version {healthData.version}</p>
            </article>

            <article class={`health-card ${healthStatusClass(healthData.database?.status)}`}>
              <h4>Database</h4>
              <p class="health-value">{healthData.database?.status ?? 'unknown'}</p>
              <p class="muted">LibSQL connectivity</p>
            </article>

            <article class={`health-card ${healthStatusClass(healthData.embeddings?.status)}`}>
              <h4>Embeddings</h4>
              <p class="health-value">{healthData.embeddings?.status ?? 'unknown'}</p>
              <p class="muted">
                {healthData.embeddings?.model ?? 'n/a'} ({healthData.embeddings?.dimensions ?? 'n/a'} dims)
              </p>
            </article>

            <article class={`health-card ${healthStatusClass(healthData.llm?.status)}`}>
              <h4>LLM</h4>
              <p class="health-value">{healthData.llm?.status ?? 'unknown'}</p>
              <p class="muted">{healthData.llm?.provider ?? 'local'} / {healthData.llm?.model ?? 'n/a'}</p>
            </article>

            <article class={`health-card ${healthStatusClass(healthData.reranker?.status)}`}>
              <h4>Reranker</h4>
              <p class="health-value">{healthData.reranker?.status ?? 'unknown'}</p>
              <p class="muted">
                {healthData.reranker?.enabled ? `Enabled (${healthData.reranker?.model ?? 'n/a'})` : 'Disabled'}
              </p>
            </article>
          </div>
        )}

        {!health.loading && !health.error && !healthData && (
          <p class="muted">Health endpoint responded without a structured payload.</p>
        )}
      </Section>

      <Section title="OpenAPI">
        <div class="button-row">
          <button
            type="button"
            onClick={() => {
              void openapi.run(() => apiRaw<Record<string, unknown>>(apiKey, '/openapi.json', { auth: false }));
            }}
          >
            Fetch OpenAPI JSON
          </button>
          <a href="/api/v1/docs" target="_blank" rel="noreferrer" class="button-link">
            Open docs UI
          </a>
        </div>
        {openapi.loading && <p>Loading OpenAPI spec...</p>}
        {openapi.error && <p class="error">{openapi.error}</p>}
      </Section>

      {health.result && <JsonView label="Raw health response" value={health.result} />}
      {openapi.result && <JsonView label="OpenAPI response" value={openapi.result} />}
    </div>
  );
}

function SearchTab({ apiKey, containerTag, onAuthFailure }: SharedTabProps) {
  const action = useApiAction(onAuthFailure);
  const [query, setQuery] = useState('');
  const [scope, setScope] = useState<'hybrid' | 'documents' | 'memories'>('hybrid');
  const [limit, setLimit] = useState('20');
  const [threshold, setThreshold] = useState('');
  const [rerank, setRerank] = useState(false);
  const [includeDocuments, setIncludeDocuments] = useState(true);
  const [includeChunks, setIncludeChunks] = useState(false);

  const onSubmit = async (event: Event) => {
    event.preventDefault();

    try {
      const parsedLimit = parseOptionalNumber(limit);
      const parsedThreshold = parseOptionalNumber(threshold);

      await action.run(() =>
        apiEnvelope(apiKey, '/search', {
          method: 'POST',
          body: {
            q: query,
            scope,
            containerTags: containerTag.trim() ? [containerTag.trim()] : undefined,
            threshold: parsedThreshold,
            limit: parsedLimit,
            include: {
              documents: includeDocuments,
              chunks: includeChunks,
            },
            rerank,
          },
        }),
      );
    } catch (error) {
      action.run(async () => ({
        ok: false,
        status: 0,
        error: error instanceof Error ? error.message : 'Invalid form input',
      }));
    }
  };

  return (
    <div class="tab-grid">
      <Section title="Unified search">
        <form onSubmit={onSubmit} class="form-grid">
          <Field label="Query">
            <input value={query} onInput={(event) => setQuery((event.target as HTMLInputElement).value)} required />
          </Field>

          <Field label="Scope">
            <select value={scope} onInput={(event) => setScope((event.target as HTMLSelectElement).value as typeof scope)}>
              <option value="hybrid">hybrid</option>
              <option value="documents">documents</option>
              <option value="memories">memories</option>
            </select>
          </Field>

          <Field label="Limit">
            <input value={limit} onInput={(event) => setLimit((event.target as HTMLInputElement).value)} />
          </Field>

          <Field label="Threshold (optional)">
            <input value={threshold} onInput={(event) => setThreshold((event.target as HTMLInputElement).value)} />
          </Field>

          <label class="checkbox-field">
            <input type="checkbox" checked={rerank} onInput={(event) => setRerank((event.target as HTMLInputElement).checked)} />
            Enable reranking
          </label>

          <label class="checkbox-field">
            <input
              type="checkbox"
              checked={includeDocuments}
              onInput={(event) => setIncludeDocuments((event.target as HTMLInputElement).checked)}
            />
            Include documents
          </label>

          <label class="checkbox-field">
            <input
              type="checkbox"
              checked={includeChunks}
              onInput={(event) => setIncludeChunks((event.target as HTMLInputElement).checked)}
            />
            Include chunks
          </label>

          <div class="button-row">
            <button type="submit" disabled={action.loading}>
              {action.loading ? 'Searching...' : 'Run search'}
            </button>
            <button type="button" class="ghost" onClick={action.clear}>
              Clear
            </button>
          </div>
        </form>

        {action.error && <p class="error">{action.error}</p>}
      </Section>

      {action.result && <JsonView label="Search response" value={action.result} />}
    </div>
  );
}

function DocumentsTab({ apiKey, containerTag, onAuthFailure }: SharedTabProps) {
  const action = useApiAction(onAuthFailure);
  const [documentsTab, setDocumentsTab] = useState<'list' | 'create' | 'batch' | 'ingest'>('list');

  const [listLimit, setListLimit] = useState('20');
  const [listCursor, setListCursor] = useState('');

  const [docContent, setDocContent] = useState('');
  const [docCustomId, setDocCustomId] = useState('');
  const [docContentType, setDocContentType] = useState('');
  const [docMetadata, setDocMetadata] = useState('');
  const [extractMemories, setExtractMemories] = useState(false);

  const [batchLines, setBatchLines] = useState('');
  const [batchMetadata, setBatchMetadata] = useState('');

  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadMetadata, setUploadMetadata] = useState('');

  const [documentId, setDocumentId] = useState('');
  const [ingestionId, setIngestionId] = useState('');
  const [updateTitle, setUpdateTitle] = useState('');
  const [updateMetadata, setUpdateMetadata] = useState('');
  const [updateContainerTags, setUpdateContainerTags] = useState('');

  const listDocuments = async () => {
    const query = new URLSearchParams();
    if (containerTag.trim()) {
      query.append('containerTags', containerTag.trim());
    }

    if (listLimit.trim()) {
      query.append('limit', listLimit.trim());
    }

    if (listCursor.trim()) {
      query.append('cursor', listCursor.trim());
    }

    const suffix = query.toString() ? `?${query.toString()}` : '';
    await action.run(() => apiEnvelope(apiKey, `/documents${suffix}`));
  };

  const createDocument = async (event: Event) => {
    event.preventDefault();

    try {
      const metadata = parseJsonObject(docMetadata);
      const response = await action.run(() =>
        apiEnvelope<{ documentId?: string; ingestionId?: string }>(apiKey, '/documents', {
          method: 'POST',
          body: {
            content: docContent,
            containerTag: containerTag.trim() || undefined,
            customId: docCustomId.trim() || undefined,
            metadata,
            contentType: docContentType.trim() || undefined,
            extractMemories,
          },
        }),
      );

      if (response.ok && response.envelope?.data?.documentId) {
        setDocumentId(response.envelope.data.documentId);
        setIngestionId(response.envelope.data.ingestionId ?? response.envelope.data.documentId ?? '');
      }
    } catch (error) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: error instanceof Error ? error.message : 'Invalid metadata JSON',
      }));
    }
  };

  const batchCreateDocuments = async () => {
    try {
      const lines = batchLines
        .split('\n')
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

      if (lines.length === 0) {
        throw new Error('Provide at least one non-empty line for batch create');
      }

      const metadata = parseJsonObject(batchMetadata);

      await action.run(() =>
        apiEnvelope(apiKey, '/documents:batch', {
          method: 'POST',
          body: {
            documents: lines.map((line) => ({
              content: line,
              extractMemories,
            })),
            containerTag: containerTag.trim() || undefined,
            metadata,
          },
        }),
      );
    } catch (error) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: error instanceof Error ? error.message : 'Invalid batch input',
      }));
    }
  };

  const uploadDocument = async () => {
    if (!uploadFile) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: 'Select a file before uploading',
      }));
      return;
    }

    const formData = new FormData();
    formData.append('file', uploadFile);

    if (containerTag.trim()) {
      formData.append('containerTag', containerTag.trim());
    }

    if (uploadMetadata.trim()) {
      formData.append('metadata', uploadMetadata);
    }

    await action.run(() =>
      apiEnvelope<{ documentId?: string; ingestionId?: string }>(apiKey, '/documents:upload', {
        method: 'POST',
        body: formData,
      }),
    );
  };

  const getDocument = async () => {
    await action.run(() => apiEnvelope(apiKey, `/documents/${encodeURIComponent(documentId.trim())}`));
  };

  const updateDocument = async () => {
    try {
      const metadata = parseJsonObject(updateMetadata);
      const tags = parseCommaList(updateContainerTags);

      await action.run(() =>
        apiEnvelope(apiKey, `/documents/${encodeURIComponent(documentId.trim())}`, {
          method: 'PATCH',
          body: {
            title: updateTitle.trim() || undefined,
            metadata,
            containerTags: tags.length > 0 ? tags : undefined,
          },
        }),
      );
    } catch (error) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: error instanceof Error ? error.message : 'Invalid update payload',
      }));
    }
  };

  const deleteDocument = async () => {
    await action.run(() =>
      apiEnvelope(apiKey, `/documents/${encodeURIComponent(documentId.trim())}`, {
        method: 'DELETE',
      }),
    );
  };

  const getIngestionStatus = async () => {
    await action.run(() =>
      apiEnvelope(apiKey, `/ingestions/${encodeURIComponent(ingestionId.trim())}`),
    );
  };

  return (
    <div class="tab-grid">
      <Section title="Documents">
        <div class="subtabs" role="tablist" aria-label="Documents actions">
          <button
            type="button"
            role="tab"
            class={documentsTab === 'list' ? 'subtab active' : 'subtab'}
            aria-selected={documentsTab === 'list'}
            onClick={() => setDocumentsTab('list')}
          >
            List
          </button>
          <button
            type="button"
            role="tab"
            class={documentsTab === 'create' ? 'subtab active' : 'subtab'}
            aria-selected={documentsTab === 'create'}
            onClick={() => setDocumentsTab('create')}
          >
            Create
          </button>
          <button
            type="button"
            role="tab"
            class={documentsTab === 'batch' ? 'subtab active' : 'subtab'}
            aria-selected={documentsTab === 'batch'}
            onClick={() => setDocumentsTab('batch')}
          >
            Batch
          </button>
          <button
            type="button"
            role="tab"
            class={documentsTab === 'ingest' ? 'subtab active' : 'subtab'}
            aria-selected={documentsTab === 'ingest'}
            onClick={() => setDocumentsTab('ingest')}
          >
            Ingest
          </button>
        </div>

        {documentsTab === 'list' && (
          <div class="form-grid compact">
            <Field label="Limit">
              <input value={listLimit} onInput={(event) => setListLimit((event.target as HTMLInputElement).value)} />
            </Field>
            <Field label="Cursor">
              <input value={listCursor} onInput={(event) => setListCursor((event.target as HTMLInputElement).value)} />
            </Field>
            <div class="button-row">
              <button type="button" onClick={() => void listDocuments()} disabled={action.loading}>
                List documents
              </button>
            </div>
          </div>
        )}

        {documentsTab === 'create' && (
          <form onSubmit={createDocument} class="form-grid">
            <Field label="Content">
              <textarea value={docContent} onInput={(event) => setDocContent((event.target as HTMLTextAreaElement).value)} rows={5} required />
            </Field>
            <Field label="Custom ID (optional)">
              <input value={docCustomId} onInput={(event) => setDocCustomId((event.target as HTMLInputElement).value)} />
            </Field>
            <Field label="Content type (optional)">
              <input value={docContentType} onInput={(event) => setDocContentType((event.target as HTMLInputElement).value)} />
            </Field>
            <Field label="Metadata JSON (optional)">
              <textarea value={docMetadata} onInput={(event) => setDocMetadata((event.target as HTMLTextAreaElement).value)} rows={3} />
            </Field>
            <label class="checkbox-field">
              <input
                type="checkbox"
                checked={extractMemories}
                onInput={(event) => setExtractMemories((event.target as HTMLInputElement).checked)}
              />
              Extract memories from document
            </label>
            <div class="button-row">
              <button type="submit" disabled={action.loading}>
                Create document
              </button>
            </div>
          </form>
        )}

        {documentsTab === 'batch' && (
          <div class="form-grid">
            <Field label="Batch lines (one document content per line)">
              <textarea value={batchLines} onInput={(event) => setBatchLines((event.target as HTMLTextAreaElement).value)} rows={7} />
            </Field>
            <Field label="Batch metadata JSON (optional)">
              <textarea value={batchMetadata} onInput={(event) => setBatchMetadata((event.target as HTMLTextAreaElement).value)} rows={3} />
            </Field>
            <div class="button-row">
              <button type="button" onClick={() => void batchCreateDocuments()} disabled={action.loading}>
                Batch create
              </button>
            </div>
          </div>
        )}

        {documentsTab === 'ingest' && (
          <div class="form-grid">
            <Field label="Upload file">
              <input type="file" onInput={(event) => setUploadFile((event.target as HTMLInputElement).files?.[0] ?? null)} />
            </Field>
            <Field label="Upload metadata JSON (optional)">
              <textarea value={uploadMetadata} onInput={(event) => setUploadMetadata((event.target as HTMLTextAreaElement).value)} rows={3} />
            </Field>
            <div class="button-row">
              <button type="button" onClick={() => void uploadDocument()} disabled={action.loading}>
                Upload document
              </button>
            </div>

            <div class="form-grid compact">
              <Field label="Document ID">
                <input value={documentId} onInput={(event) => setDocumentId((event.target as HTMLInputElement).value)} />
              </Field>
              <Field label="Ingestion ID">
                <input value={ingestionId} onInput={(event) => setIngestionId((event.target as HTMLInputElement).value)} />
              </Field>
              <Field label="Update title">
                <input value={updateTitle} onInput={(event) => setUpdateTitle((event.target as HTMLInputElement).value)} />
              </Field>
              <Field label="Update metadata JSON">
                <textarea value={updateMetadata} onInput={(event) => setUpdateMetadata((event.target as HTMLTextAreaElement).value)} rows={3} />
              </Field>
              <Field label="Update container tags (comma-separated)">
                <input
                  value={updateContainerTags}
                  onInput={(event) => setUpdateContainerTags((event.target as HTMLInputElement).value)}
                />
              </Field>
              <div class="button-row">
                <button type="button" onClick={() => void getDocument()} disabled={action.loading || !documentId.trim()}>
                  Get document
                </button>
                <button type="button" onClick={() => void updateDocument()} disabled={action.loading || !documentId.trim()}>
                  Update document
                </button>
                <button type="button" onClick={() => void deleteDocument()} disabled={action.loading || !documentId.trim()}>
                  Delete document
                </button>
                <button type="button" onClick={() => void getIngestionStatus()} disabled={action.loading || !ingestionId.trim()}>
                  Ingestion status
                </button>
              </div>
            </div>
          </div>
        )}
      </Section>

      {action.loading && <p>Working...</p>}
      {action.error && <p class="error">{action.error}</p>}
      {action.result && <JsonView label="Documents response" value={action.result} />}
    </div>
  );
}

function MemoriesTab({ apiKey, containerTag, onAuthFailure }: SharedTabProps) {
  const action = useApiAction(onAuthFailure);
  const [memoriesTab, setMemoriesTab] = useState<'list' | 'create' | 'manage' | 'forget'>('list');

  const [limit, setLimit] = useState('20');
  const [cursor, setCursor] = useState('');

  const [memoryContent, setMemoryContent] = useState('');
  const [memoryType, setMemoryType] = useState<'fact' | 'preference' | 'episode'>('fact');
  const [memoryMetadata, setMemoryMetadata] = useState('');

  const [memoryId, setMemoryId] = useState('');
  const [updatedContent, setUpdatedContent] = useState('');
  const [updatedMetadata, setUpdatedMetadata] = useState('');
  const [isStatic, setIsStatic] = useState(false);
  const [deleteReason, setDeleteReason] = useState('');

  const [forgetContent, setForgetContent] = useState('');
  const [forgetReason, setForgetReason] = useState('');

  const listMemories = async () => {
    const tag = containerTag.trim();
    if (!tag) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: 'Container tag is required for listing memories',
      }));
      return;
    }

    const params = new URLSearchParams();
    params.append('containerTag', tag);
    if (limit.trim()) {
      params.append('limit', limit.trim());
    }
    if (cursor.trim()) {
      params.append('cursor', cursor.trim());
    }

    await action.run(() => apiEnvelope(apiKey, `/memories?${params.toString()}`));
  };

  const createMemory = async (event: Event) => {
    event.preventDefault();

    const tag = containerTag.trim();
    if (!tag) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: 'Container tag is required for creating memory',
      }));
      return;
    }

    try {
      const metadata = parseJsonObject(memoryMetadata);
      await action.run(() =>
        apiEnvelope(apiKey, '/memories', {
          method: 'POST',
          body: {
            content: memoryContent,
            containerTag: tag,
            memoryType,
            metadata,
          },
        }),
      );
    } catch (error) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: error instanceof Error ? error.message : 'Invalid memory metadata JSON',
      }));
    }
  };

  const getMemory = async () => {
    await action.run(() => apiEnvelope(apiKey, `/memories/${encodeURIComponent(memoryId.trim())}`));
  };

  const updateMemory = async () => {
    try {
      const metadata = parseJsonObject(updatedMetadata);
      await action.run(() =>
        apiEnvelope(apiKey, `/memories/${encodeURIComponent(memoryId.trim())}`, {
          method: 'PATCH',
          body: {
            content: updatedContent,
            metadata,
            isStatic,
          },
        }),
      );
    } catch (error) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: error instanceof Error ? error.message : 'Invalid update metadata JSON',
      }));
    }
  };

  const deleteMemory = async () => {
    await action.run(() =>
      apiEnvelope(apiKey, `/memories/${encodeURIComponent(memoryId.trim())}`, {
        method: 'DELETE',
        body: {
          reason: deleteReason.trim() || undefined,
        },
      }),
    );
  };

  const forgetMemoryByContent = async () => {
    const tag = containerTag.trim();
    if (!tag) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: 'Container tag is required for content forget',
      }));
      return;
    }

    await action.run(() =>
      apiEnvelope(apiKey, '/memories:forget', {
        method: 'POST',
        body: {
          content: forgetContent,
          containerTag: tag,
          reason: forgetReason.trim() || undefined,
        },
      }),
    );
  };

  return (
    <div class="tab-grid">
      <Section title="Memories">
        <div class="subtabs" role="tablist" aria-label="Memory actions">
          <button
            type="button"
            role="tab"
            class={memoriesTab === 'list' ? 'subtab active' : 'subtab'}
            aria-selected={memoriesTab === 'list'}
            onClick={() => setMemoriesTab('list')}
          >
            List
          </button>
          <button
            type="button"
            role="tab"
            class={memoriesTab === 'create' ? 'subtab active' : 'subtab'}
            aria-selected={memoriesTab === 'create'}
            onClick={() => setMemoriesTab('create')}
          >
            Create
          </button>
          <button
            type="button"
            role="tab"
            class={memoriesTab === 'manage' ? 'subtab active' : 'subtab'}
            aria-selected={memoriesTab === 'manage'}
            onClick={() => setMemoriesTab('manage')}
          >
            Manage
          </button>
          <button
            type="button"
            role="tab"
            class={memoriesTab === 'forget' ? 'subtab active' : 'subtab'}
            aria-selected={memoriesTab === 'forget'}
            onClick={() => setMemoriesTab('forget')}
          >
            Forget
          </button>
        </div>

        {memoriesTab === 'list' && (
          <div class="form-grid compact">
            <Field label="Limit">
              <input value={limit} onInput={(event) => setLimit((event.target as HTMLInputElement).value)} />
            </Field>
            <Field label="Cursor">
              <input value={cursor} onInput={(event) => setCursor((event.target as HTMLInputElement).value)} />
            </Field>
            <div class="button-row">
              <button type="button" onClick={() => void listMemories()} disabled={action.loading}>
                List memories
              </button>
            </div>
          </div>
        )}

        {memoriesTab === 'create' && (
          <form onSubmit={createMemory} class="form-grid">
            <Field label="Content">
              <textarea
                value={memoryContent}
                onInput={(event) => setMemoryContent((event.target as HTMLTextAreaElement).value)}
                rows={4}
                required
              />
            </Field>
            <Field label="Memory type">
              <select
                value={memoryType}
                onInput={(event) => setMemoryType((event.target as HTMLSelectElement).value as typeof memoryType)}
              >
                <option value="fact">fact</option>
                <option value="preference">preference</option>
                <option value="episode">episode</option>
              </select>
            </Field>
            <Field label="Metadata JSON">
              <textarea
                value={memoryMetadata}
                onInput={(event) => setMemoryMetadata((event.target as HTMLTextAreaElement).value)}
                rows={3}
              />
            </Field>
            <div class="button-row">
              <button type="submit" disabled={action.loading}>
                Create memory
              </button>
            </div>
          </form>
        )}

        {memoriesTab === 'manage' && (
          <div class="form-grid">
            <Field label="Memory ID">
              <input value={memoryId} onInput={(event) => setMemoryId((event.target as HTMLInputElement).value)} />
            </Field>
            <Field label="Updated content">
              <textarea
                value={updatedContent}
                onInput={(event) => setUpdatedContent((event.target as HTMLTextAreaElement).value)}
                rows={3}
              />
            </Field>
            <Field label="Updated metadata JSON">
              <textarea
                value={updatedMetadata}
                onInput={(event) => setUpdatedMetadata((event.target as HTMLTextAreaElement).value)}
                rows={3}
              />
            </Field>
            <label class="checkbox-field">
              <input type="checkbox" checked={isStatic} onInput={(event) => setIsStatic((event.target as HTMLInputElement).checked)} />
              Set memory static
            </label>
            <Field label="Delete reason (optional)">
              <input value={deleteReason} onInput={(event) => setDeleteReason((event.target as HTMLInputElement).value)} />
            </Field>
            <div class="button-row">
              <button type="button" onClick={() => void getMemory()} disabled={action.loading || !memoryId.trim()}>
                Get memory
              </button>
              <button type="button" onClick={() => void updateMemory()} disabled={action.loading || !memoryId.trim()}>
                Update memory
              </button>
              <button type="button" onClick={() => void deleteMemory()} disabled={action.loading || !memoryId.trim()}>
                Delete memory
              </button>
            </div>
          </div>
        )}

        {memoriesTab === 'forget' && (
          <div class="form-grid compact">
            <Field label="Memory content">
              <textarea
                value={forgetContent}
                onInput={(event) => setForgetContent((event.target as HTMLTextAreaElement).value)}
                rows={3}
              />
            </Field>
            <Field label="Reason (optional)">
              <input value={forgetReason} onInput={(event) => setForgetReason((event.target as HTMLInputElement).value)} />
            </Field>
            <div class="button-row">
              <button type="button" onClick={() => void forgetMemoryByContent()} disabled={action.loading}>
                Forget by content
              </button>
            </div>
          </div>
        )}
      </Section>

      {action.loading && <p>Working...</p>}
      {action.error && <p class="error">{action.error}</p>}
      {action.result && <JsonView label="Memories response" value={action.result} />}
    </div>
  );
}

function GraphTab({ apiKey, containerTag, setContainerTag, onAuthFailure }: SharedTabProps) {
  const action = useApiAction(onAuthFailure);
  const tagsAction = useApiAction(onAuthFailure);

  const [mode, setMode] = useState<'container' | 'memory'>('container');
  const [memoryId, setMemoryId] = useState('');
  const [maxNodes, setMaxNodes] = useState('100');
  const [depth, setDepth] = useState('2');
  const [relationTypes, setRelationTypes] = useState('');
  const [containerTags, setContainerTags] = useState<string[]>([]);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNodeResponse | null>(null);
  const tagsRun = tagsAction.run;
  const graphStats = useMemo(() => {
    const nodeCount = graph?.nodes.length ?? 0;
    const edgeCount = graph?.links.length ?? 0;
    return {
      nodeCount,
      edgeCount,
      sparse: nodeCount > 0 && edgeCount < Math.max(2, Math.floor(nodeCount / 4)),
    };
  }, [graph]);

  const loadContainerTags = useCallback(async () => {
    const response = await tagsRun(() => apiEnvelope<ContainerTagsResponse>(apiKey, '/containers/tags'));
    if (!response.ok) {
      return;
    }

    const tags = (response.envelope?.data?.tags ?? [])
      .map((tag) => tag.trim())
      .filter((tag) => tag.length > 0)
      .sort((left, right) => left.localeCompare(right));

    setContainerTags(tags);
  }, [apiKey, tagsRun]);

  useEffect(() => {
    void loadContainerTags();
  }, [loadContainerTags]);

  const loadGraph = async (overrides?: { mode?: 'container' | 'memory'; containerTag?: string }) => {
    const requestMode = overrides?.mode ?? mode;
    const tag = (overrides?.containerTag ?? containerTag).trim();

    try {
      const maxNodesValue = parseOptionalNumber(maxNodes);
      const depthValue = parseOptionalNumber(depth);

      if (requestMode === 'container') {
        if (!tag) {
          throw new Error('Container tag is required for container graph');
        }

        const params = new URLSearchParams();
        if (maxNodesValue !== undefined) {
          params.append('maxNodes', String(Math.trunc(maxNodesValue)));
        }

        const suffix = params.toString() ? `?${params.toString()}` : '';
        const response = await action.run(() =>
          apiEnvelope<GraphResponse>(apiKey, `/containers/${encodeURIComponent(tag)}/graph${suffix}`),
        );

        if (response.ok) {
          setGraph(response.envelope?.data ?? null);
        }
        return;
      }

      if (!memoryId.trim()) {
        throw new Error('Memory ID is required for memory graph');
      }

      const params = new URLSearchParams();
      if (maxNodesValue !== undefined) {
        params.append('maxNodes', String(Math.trunc(maxNodesValue)));
      }
      if (depthValue !== undefined) {
        params.append('depth', String(Math.trunc(depthValue)));
      }

      const relationList = parseCommaList(relationTypes);
      if (relationList.length > 0) {
        params.append('relationTypes', relationList.join(','));
      }

      const suffix = params.toString() ? `?${params.toString()}` : '';
      const response = await action.run(() =>
        apiEnvelope<GraphResponse>(apiKey, `/memories/${encodeURIComponent(memoryId.trim())}/graph${suffix}`),
      );

      if (response.ok) {
        setGraph(response.envelope?.data ?? null);
      }
    } catch (error) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: error instanceof Error ? error.message : 'Invalid graph parameters',
      }));
    }
  };

  const loadGraphForContainerTag = async (tag: string) => {
    setContainerTag(tag);
    setMode('container');
    await loadGraph({ mode: 'container', containerTag: tag });
  };

  return (
    <div class="tab-grid">
      <Section title="Graph query">
        <div class="graph-query-layout">
          <div class="form-grid compact">
            <Field label="Mode">
              <select value={mode} onInput={(event) => setMode((event.target as HTMLSelectElement).value as typeof mode)}>
                <option value="container">container graph</option>
                <option value="memory">memory graph</option>
              </select>
            </Field>

            <p class="muted">Using current container tag: <strong>{containerTag.trim() || '(none configured)'}</strong></p>

            {mode === 'memory' && (
              <Field label="Memory ID">
                <input value={memoryId} onInput={(event) => setMemoryId((event.target as HTMLInputElement).value)} />
              </Field>
            )}

            <Field label="Max nodes">
              <input value={maxNodes} onInput={(event) => setMaxNodes((event.target as HTMLInputElement).value)} />
            </Field>

            {mode === 'memory' && (
              <Field label="Depth">
                <input value={depth} onInput={(event) => setDepth((event.target as HTMLInputElement).value)} />
              </Field>
            )}

            {mode === 'memory' && (
              <Field label="Relation types (comma-separated)">
                <input
                  value={relationTypes}
                  onInput={(event) => setRelationTypes((event.target as HTMLInputElement).value)}
                  placeholder="updates,relatesto,sources"
                />
              </Field>
            )}

            <div class="button-row">
              <button type="button" onClick={() => void loadGraph()} disabled={action.loading}>
                Load graph
              </button>
              <button
                type="button"
                class="ghost"
                onClick={() => {
                  setGraph(null);
                  setSelectedNode(null);
                  action.clear();
                }}
              >
                Clear
              </button>
            </div>
          </div>

          <aside class="container-tags-panel">
            <div class="container-tags-header">
              <h4>Container tags</h4>
              <button type="button" class="ghost" onClick={() => void loadContainerTags()} disabled={tagsAction.loading}>
                Refresh
              </button>
            </div>

            {tagsAction.loading && <p class="muted">Loading tags...</p>}
            {tagsAction.error && <p class="error">{tagsAction.error}</p>}
            {!tagsAction.loading && !tagsAction.error && containerTags.length === 0 && (
              <p class="muted">No active container tags found.</p>
            )}

            {containerTags.length > 0 && (
              <div class="container-tags-list">
                {containerTags.map((tag) => (
                  <button
                    key={tag}
                    type="button"
                    class={containerTag.trim() === tag ? 'tag-chip active' : 'tag-chip'}
                    onClick={() => {
                      void loadGraphForContainerTag(tag);
                    }}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            )}
          </aside>
        </div>
      </Section>

      <Section title="Graph visualization">
        {graph && (
          <p class="muted">
            Showing {graphStats.nodeCount} nodes and {graphStats.edgeCount} relationships.
          </p>
        )}
        {graphStats.sparse && (
          <p class="notice">
            This container graph is sparse. If this looks like isolated rows, the underlying memories may have few recorded relations.
          </p>
        )}
        <GraphCanvas graph={graph} onNodeSelect={setSelectedNode} />
        {selectedNode && (
          <div class="selected-node">
            <p>
              <strong>{selectedNode.id}</strong> ({selectedNode.type})
            </p>
            <pre>{JSON.stringify(selectedNode.metadata, null, 2)}</pre>
          </div>
        )}
      </Section>

      {action.loading && <p>Loading graph...</p>}
      {action.error && <p class="error">{action.error}</p>}
      {action.result && <JsonView label="Graph response" value={action.result} />}
    </div>
  );
}

function ProfileTab({ apiKey, containerTag, onAuthFailure }: SharedTabProps) {
  const action = useApiAction(onAuthFailure);
  const [query, setQuery] = useState('');
  const [threshold, setThreshold] = useState('');
  const [limit, setLimit] = useState('50');
  const [includeDynamic, setIncludeDynamic] = useState(true);
  const [generateNarrative, setGenerateNarrative] = useState(true);

  const computeProfile = async (event: Event) => {
    event.preventDefault();

    const tag = containerTag.trim();
    if (!tag) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: 'Container tag is required for profile computation',
      }));
      return;
    }

    try {
      const thresholdValue = parseOptionalNumber(threshold);
      const limitValue = parseOptionalNumber(limit);

      await action.run(() =>
        apiEnvelope(apiKey, '/profile:compute', {
          method: 'POST',
          body: {
            containerTag: tag,
            q: query.trim() || undefined,
            threshold: thresholdValue,
            includeDynamic,
            limit: limitValue !== undefined ? Math.trunc(limitValue) : undefined,
            generateNarrative,
          },
        }),
      );
    } catch (error) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: error instanceof Error ? error.message : 'Invalid profile parameters',
      }));
    }
  };

  return (
    <div class="tab-grid">
      <Section title="Compute profile">
        <form onSubmit={computeProfile} class="form-grid compact">
          <Field label="Query (optional)">
            <input value={query} onInput={(event) => setQuery((event.target as HTMLInputElement).value)} />
          </Field>
          <Field label="Threshold (optional)">
            <input value={threshold} onInput={(event) => setThreshold((event.target as HTMLInputElement).value)} />
          </Field>
          <Field label="Limit">
            <input value={limit} onInput={(event) => setLimit((event.target as HTMLInputElement).value)} />
          </Field>
          <label class="checkbox-field">
            <input
              type="checkbox"
              checked={includeDynamic}
              onInput={(event) => setIncludeDynamic((event.target as HTMLInputElement).checked)}
            />
            Include dynamic facts
          </label>
          <label class="checkbox-field">
            <input
              type="checkbox"
              checked={generateNarrative}
              onInput={(event) => setGenerateNarrative((event.target as HTMLInputElement).checked)}
            />
            Generate narrative
          </label>
          <div class="button-row">
            <button type="submit" disabled={action.loading}>
              Compute profile
            </button>
          </div>
        </form>
      </Section>

      {action.loading && <p>Computing profile...</p>}
      {action.error && <p class="error">{action.error}</p>}
      {action.result && <JsonView label="Profile response" value={action.result} />}
    </div>
  );
}

function ConversationTab({ apiKey, containerTag, onAuthFailure }: SharedTabProps) {
  const action = useApiAction(onAuthFailure);

  const [sessionId, setSessionId] = useState('');
  const [memoryType, setMemoryType] = useState<'fact' | 'preference' | 'episode' | ''>('');
  const [messagesJson, setMessagesJson] = useState(
    JSON.stringify(
      [
        { role: 'user', content: 'I prefer dark mode.' },
        { role: 'assistant', content: 'Noted.' },
      ],
      null,
      2,
    ),
  );

  const ingestConversation = async (event: Event) => {
    event.preventDefault();

    const tag = containerTag.trim();
    if (!tag) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: 'Container tag is required for conversation ingest',
      }));
      return;
    }

    try {
      const parsed = JSON.parse(messagesJson) as unknown;
      if (!Array.isArray(parsed)) {
        throw new Error('Messages JSON must be an array');
      }

      await action.run(() =>
        apiEnvelope(apiKey, '/conversations:ingest', {
          method: 'POST',
          body: {
            messages: parsed,
            containerTag: tag,
            sessionId: sessionId.trim() || undefined,
            memoryType: memoryType || undefined,
          },
        }),
      );
    } catch (error) {
      await action.run(async () => ({
        ok: false,
        status: 0,
        error: error instanceof Error ? error.message : 'Invalid conversation payload',
      }));
    }
  };

  return (
    <div class="tab-grid">
      <Section title="Ingest conversation">
        <form onSubmit={ingestConversation} class="form-grid">
          <Field label="Session ID (optional)">
            <input value={sessionId} onInput={(event) => setSessionId((event.target as HTMLInputElement).value)} />
          </Field>
          <Field label="Memory type (optional override)">
            <select
              value={memoryType}
              onInput={(event) => setMemoryType((event.target as HTMLSelectElement).value as typeof memoryType)}
            >
              <option value="">auto</option>
              <option value="fact">fact</option>
              <option value="preference">preference</option>
              <option value="episode">episode</option>
            </select>
          </Field>
          <Field label="Messages JSON array">
            <textarea
              value={messagesJson}
              onInput={(event) => setMessagesJson((event.target as HTMLTextAreaElement).value)}
              rows={8}
            />
          </Field>
          <div class="button-row">
            <button type="submit" disabled={action.loading}>
              Ingest conversation
            </button>
          </div>
        </form>
      </Section>

      {action.loading && <p>Ingesting conversation...</p>}
      {action.error && <p class="error">{action.error}</p>}
      {action.result && <JsonView label="Conversation response" value={action.result} />}
    </div>
  );
}

function AdminTab({ apiKey, onAuthFailure }: { apiKey: string; onAuthFailure: (message: string) => void }) {
  const action = useApiAction(onAuthFailure);

  return (
    <div class="tab-grid">
      <Section title="Forgetting manager">
        <p>Run one immediate forgetting cycle.</p>
        <div class="button-row">
          <button
            type="button"
            disabled={action.loading}
            onClick={() => {
              void action.run(() =>
                apiEnvelope(apiKey, '/admin/forgetting:run', {
                  method: 'POST',
                }),
              );
            }}
          >
            Run forgetting cycle
          </button>
        </div>
      </Section>

      {action.loading && <p>Running forgetting cycle...</p>}
      {action.error && <p class="error">{action.error}</p>}
      {action.result && <JsonView label="Admin response" value={action.result} />}
    </div>
  );
}

interface SettingsTabProps {
  apiKey: string;
  containerTag: string;
  onSave: (nextApiKey: string, nextContainerTag: string) => Promise<void>;
  onValidate: () => Promise<void>;
}

function SettingsTab({
  apiKey,
  containerTag,
  onSave,
  onValidate,
}: SettingsTabProps) {
  const [draftApiKey, setDraftApiKey] = useState(apiKey);
  const [draftContainerTag, setDraftContainerTag] = useState(containerTag);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setDraftApiKey(apiKey);
  }, [apiKey]);

  useEffect(() => {
    setDraftContainerTag(containerTag);
  }, [containerTag]);

  const save = async (event: Event) => {
    event.preventDefault();
    setSaving(true);
    try {
      await onSave(draftApiKey, draftContainerTag);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div class="tab-grid">
      <Section title="Credentials">
        <form class="form-grid" onSubmit={save}>
          <Field label="API key">
            <input
              type="password"
              value={draftApiKey}
              placeholder="Paste MOMO_API_KEYS value"
              onInput={(event) => setDraftApiKey((event.target as HTMLInputElement).value)}
            />
          </Field>

          <Field label="Current container tag">
            <input
              value={draftContainerTag}
              placeholder="user_123"
              onInput={(event) => setDraftContainerTag((event.target as HTMLInputElement).value)}
            />
          </Field>

          <div class="button-row">
            <button type="submit" disabled={saving}>
              {saving ? 'Saving...' : 'Save settings'}
            </button>
            <button
              type="button"
              class="ghost"
              onClick={() => {
                void onValidate();
              }}
            >
              Re-check auth
            </button>
          </div>
        </form>

        <p class="muted">
          API key and current container tag are stored in localStorage for this browser.
        </p>
      </Section>
    </div>
  );
}

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>(() => {
    if (typeof window === 'undefined') {
      return 'system';
    }
    return tabFromHash(window.location.hash) ?? 'system';
  });
  const [apiKey, setApiKey] = useState<string>(() => localStorage.getItem(API_KEY_STORAGE_KEY) ?? '');
  const [containerTag, setContainerTag] = useState<string>(() => localStorage.getItem(CONTAINER_STORAGE_KEY) ?? '');
  const [containerTagOptions, setContainerTagOptions] = useState<string[]>([]);
  const [containerTagOptionsLoading, setContainerTagOptionsLoading] = useState(false);
  const [scopeExpanded, setScopeExpanded] = useState<boolean>(() => {
    if (typeof window === 'undefined') {
      return true;
    }

    const stored = window.localStorage.getItem(SCOPE_EXPANDED_STORAGE_KEY);
    if (stored === '1') {
      return true;
    }
    if (stored === '0') {
      return false;
    }

    const initialTag = (window.localStorage.getItem(CONTAINER_STORAGE_KEY) ?? '').trim();
    return initialTag.length === 0;
  });
  const [authState, setAuthState] = useState<AuthState>(() => (apiKey.trim() ? 'checking' : 'missing'));
  const [authMessage, setAuthMessage] = useState<string | null>(null);
  const [modalApiKey, setModalApiKey] = useState(apiKey);
  const [modalSubmitting, setModalSubmitting] = useState(false);

  const refreshContainerTagOptions = useCallback(async () => {
    const trimmedApiKey = apiKey.trim();
    if (!trimmedApiKey) {
      setContainerTagOptions([]);
      return;
    }

    setContainerTagOptionsLoading(true);
    try {
      const response = await apiEnvelope<ContainerTagsResponse>(trimmedApiKey, '/containers/tags');
      if (!response.ok) {
        return;
      }

      const tags = (response.envelope?.data?.tags ?? [])
        .map((tag) => tag.trim())
        .filter((tag) => tag.length > 0)
        .sort((left, right) => left.localeCompare(right));

      setContainerTagOptions(tags);
    } finally {
      setContainerTagOptionsLoading(false);
    }
  }, [apiKey]);

  const validateApiKey = useCallback(async (candidateKey: string): Promise<boolean> => {
    const trimmed = candidateKey.trim();

    if (!trimmed) {
      setAuthState('missing');
      setAuthMessage('Enter an API key to continue.');
      return false;
    }

    setAuthState('checking');
    setAuthMessage(null);

    const response = await apiEnvelope<Record<string, unknown>>(trimmed, '/documents?limit=1');

    if (response.ok) {
      setAuthState('valid');
      setAuthMessage(null);
      return true;
    }

    setAuthState('invalid');
    setAuthMessage(response.error ?? 'Authentication failed');
    return false;
  }, []);

  const applyCredentials = useCallback(
    async (nextApiKey: string, nextContainerTag?: string) => {
      setApiKey(nextApiKey);
      if (nextContainerTag !== undefined) {
        setContainerTag(nextContainerTag);
      }
      return validateApiKey(nextApiKey);
    },
    [validateApiKey],
  );

  useEffect(() => {
    setModalApiKey(apiKey);
  }, [apiKey]);

  useEffect(() => {
    void validateApiKey(apiKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
  }, [apiKey]);

  useEffect(() => {
    localStorage.setItem(CONTAINER_STORAGE_KEY, containerTag);
  }, [containerTag]);

  useEffect(() => {
    localStorage.setItem(SCOPE_EXPANDED_STORAGE_KEY, scopeExpanded ? '1' : '0');
  }, [scopeExpanded]);

  useEffect(() => {
    if (authState === 'valid') {
      void refreshContainerTagOptions();
    }
  }, [authState, refreshContainerTagOptions]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const nextHash = `#${activeTab}`;
    if (window.location.hash !== nextHash) {
      window.location.hash = nextHash;
    }
  }, [activeTab]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const onHashChange = () => {
      const nextTab = tabFromHash(window.location.hash) ?? 'system';
      setActiveTab(nextTab);
    };

    window.addEventListener('hashchange', onHashChange);
    return () => {
      window.removeEventListener('hashchange', onHashChange);
    };
  }, []);

  const handleAuthFailure = useCallback((message: string) => {
    setAuthState('invalid');
    setAuthMessage(message || 'Authentication failed');
    setActiveTab('settings');
  }, []);

  const sharedProps = useMemo<SharedTabProps>(
    () => ({
      apiKey,
      containerTag,
      setContainerTag,
      onAuthFailure: handleAuthFailure,
    }),
    [apiKey, containerTag, handleAuthFailure],
  );

  const authModalVisible = authState !== 'valid';

  const submitModalAuth = async (event: Event) => {
    event.preventDefault();
    setModalSubmitting(true);
    try {
      const ok = await applyCredentials(modalApiKey);
      if (ok && activeTab === 'settings') {
        setActiveTab('system');
      }
    } finally {
      setModalSubmitting(false);
    }
  };

  return (
    <div class="layout">
      <aside class="sidebar">
        <div class="brand">
          <p>Momo</p>
          <h1>v1 Console</h1>
        </div>

        <nav>
          {TAB_ITEMS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              class={activeTab === tab.id ? 'tab active' : 'tab'}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </aside>

      <main class="content">
        <div class="content-inner">
          <header class="scope-bar panel">
            <div class="scope-summary">
              <div class="scope-summary-text">
                <p class="scope-kicker">Global scope</p>
                <p class="scope-summary-line">
                  Active tag:
                  {' '}
                  <span class={containerTag.trim() ? 'scope-pill' : 'scope-pill empty'}>
                    {containerTag.trim() ? containerTag.trim() : 'none configured'}
                  </span>
                </p>
              </div>
              <div class="scope-summary-actions">
                <button type="button" class="ghost" onClick={() => setActiveTab('settings')}>
                  Settings
                </button>
                <button
                  type="button"
                  class="ghost"
                  onClick={() => setScopeExpanded((current) => !current)}
                  aria-expanded={scopeExpanded}
                >
                  {scopeExpanded ? 'Collapse' : 'Edit'}
                </button>
              </div>
            </div>

            {scopeExpanded && (
              <div class="scope-row">
                <label class="scope-field">
                  <span>Container tag</span>
                  <input
                    list="global-container-tag-options"
                    value={containerTag}
                    onInput={(event) => setContainerTag((event.target as HTMLInputElement).value)}
                    placeholder="user_123"
                  />
                  <datalist id="global-container-tag-options">
                    {containerTagOptions.map((tag) => (
                      <option key={tag} value={tag} />
                    ))}
                  </datalist>
                </label>
                <div class="scope-row-actions">
                  <button
                    type="button"
                    class="ghost"
                    onClick={() => {
                      void refreshContainerTagOptions();
                    }}
                    disabled={containerTagOptionsLoading}
                  >
                    {containerTagOptionsLoading ? 'Refreshing...' : 'Refresh'}
                  </button>
                  <button
                    type="button"
                    class="ghost"
                    onClick={() => setContainerTag('')}
                    disabled={!containerTag.trim()}
                  >
                    Clear
                  </button>
                </div>
              </div>
            )}

            {scopeExpanded && (
              <p class="scope-note">
                This tag is used by Search, Documents, Memories, Graph, Profile, and Conversation tabs.
              </p>
            )}
            {!scopeExpanded && (
              <p class="scope-note collapsed">
                Expand to edit the current container tag.
              </p>
            )}
          </header>

          {activeTab === 'settings' && (
            <SettingsTab
              apiKey={apiKey}
              containerTag={containerTag}
              onSave={async (nextApiKey, nextContainerTag) => {
                await applyCredentials(nextApiKey, nextContainerTag);
              }}
              onValidate={async () => {
                await validateApiKey(apiKey);
              }}
            />
          )}
          {activeTab === 'system' && <SystemTab apiKey={apiKey} />}
          {activeTab === 'search' && <SearchTab {...sharedProps} />}
          {activeTab === 'documents' && <DocumentsTab {...sharedProps} />}
          {activeTab === 'memories' && <MemoriesTab {...sharedProps} />}
          {activeTab === 'graph' && <GraphTab {...sharedProps} />}
          {activeTab === 'profile' && <ProfileTab {...sharedProps} />}
          {activeTab === 'conversation' && <ConversationTab {...sharedProps} />}
          {activeTab === 'admin' && <AdminTab apiKey={apiKey} onAuthFailure={handleAuthFailure} />}
        </div>
      </main>

      {authModalVisible && (
        <div class="auth-modal-backdrop">
          <div class="auth-modal panel">
            <h2>Authentication required</h2>
            <p>
              Enter your API key to unlock protected v1 endpoints.
            </p>

            <form onSubmit={submitModalAuth} class="form-grid">
              <Field label="API key">
                <input
                  type="password"
                  value={modalApiKey}
                  placeholder="Paste MOMO_API_KEYS value"
                  onInput={(event) => setModalApiKey((event.target as HTMLInputElement).value)}
                  autoFocus
                />
              </Field>

              <div class="button-row">
                <button type="submit" disabled={modalSubmitting || authState === 'checking'}>
                  {modalSubmitting || authState === 'checking' ? 'Verifying...' : 'Save and verify'}
                </button>
              </div>
            </form>

            {authMessage && <p class={authState === 'invalid' ? 'error' : 'notice'}>{authMessage}</p>}
            <p class="muted">
              Credentials are stored in localStorage and can also be edited in the Settings tab.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
